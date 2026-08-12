"""
miniVoxSetu — Backend Entry Point (Phase 4: Latency-Optimized Pipeline)
Architecture: Browser audio → asyncio.Queue fork → STT path + Acoustic path
LLM: Groq LLaMA 3.3 70B (TTFT ~100ms vs Gemini's ~400ms)
Semantic/Reports: Still Gemini (background, non-latency-critical)
"""

import os
import json
import asyncio
import base64
import time
import datetime
import traceback
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai

from rag import RAGEngine
from pii import redact_pii
from stt import DeepgramSTT
from tts import DeepgramTTS, synthesize_http, detect_sentence_boundary
from semantic import analyze_utterance
from acoustic import analyze_audio, init_hubert_model
from ingest import DocumentIngester

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY not found in .env")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)

# Groq API config
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    rag_engine = RAGEngine()

    # Phase 4: Ingest documents from knowledge/ directory
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
    ingester = DocumentIngester()
    ingested_chunks = ingester.ingest_directory(knowledge_dir)

    # Initialize RAG with ingested docs (falls back to hardcoded FAQs if empty)
    rag_engine.initialize(external_documents=ingested_chunks if ingested_chunks else None)
    print(f"[OK] RAG engine initialized ({rag_engine.vector_store.count()} documents)")

    # Load HuBERT model onto GPU/CPU (lazy init — won't crash if deps missing)
    init_hubert_model()

    yield


app = FastAPI(title="miniVoxSetu", lifespan=lifespan)

# CORS: Restrict to known origins in production. Set ALLOWED_ORIGINS env var
# as comma-separated list (e.g., "https://yourdomain.com,http://localhost:5173")
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are Harris, a voice AI agent handling live phone calls for NeoBank, a modern Indian digital bank.

CRITICAL — YOU ARE ON A PHONE CALL, NOT A CHATBOT:
- You are speaking to a real person who CALLED the bank. Respond as if you are on a live phone call.
- Start the very first response with a warm greeting: "Thank you for calling NeoBank! This is Harris, how may I help you today?"
- Use natural conversational fillers: "Sure, let me check that for you", "I understand", "Of course", "Absolutely".
- Show empathy when the caller is frustrated: "I completely understand how frustrating that must be. Let me help you resolve this right away."
- Keep every response to 1-3 SHORT sentences. The caller is listening, not reading. Long responses feel robotic.
- End responses with a follow-up: "Is there anything else I can help you with?" or "Would you like me to check anything else?"
- Use Indian English naturally: "lakh" not "hundred thousand", "crore" not "ten million", amounts in ₹ (INR).

BANKING COMPLIANCE:
- NEVER read out full account numbers, Aadhaar numbers, or card numbers on a call. Say "the account ending in [last 4 digits]".
- For sensitive operations (fund transfer, card block, loan closure), say "For your security, I'll need to verify your identity first."
- If you don't know something, say "Let me connect you to a specialist for that" — never make up financial information.

If you receive context from the knowledge base, use it naturally as if you already know this information.
Never say "according to our records" or "the knowledge base says". Just answer naturally."""

# Maximum conversation turns to keep in context window.
# Prevents unbounded memory growth and Groq context overflow.
MAX_HISTORY_TURNS = 20


@app.get("/health")
async def health_check():
    return {"status": "ok", "rag_initialized": rag_engine is not None}


# --- B6: Backchannel Detection ---
BACKCHANNEL_WORDS = {
    "uh-huh", "uhuh", "uh", "huh", "hmm", "mm", "mmm", "mhm",
    "yeah", "yep", "yea", "yes", "right", "okay", "ok", "kay",
    "sure", "fine", "alright", "gotcha", "got it",
    "haan", "ha", "accha", "acha", "theek", "thik", "sahi",
    "go on", "continue", "and", "then", "so",
}

INTERRUPT_SIGNALS = {
    "wait", "stop", "no", "wrong", "not", "actually", "but",
    "what", "when", "where", "how", "why", "which", "who",
    "block", "transfer", "check", "cancel", "show", "tell",
    "help", "problem", "issue", "complaint", "balance", "account",
    "payment", "card", "loan", "emi", "transaction",
}


def is_backchannel(transcript: str) -> bool:
    """Check if a short utterance is a backchannel response (not a real interruption)."""
    words = transcript.lower().strip().split()
    if len(words) == 0:
        return False
    if len(words) > 3:
        return False  # Real sentences are not backchannel
    if any(w in INTERRUPT_SIGNALS for w in words):
        return False  # Contains a real intent word
    if any(w in BACKCHANNEL_WORDS for w in words):
        return True
    return False


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    # --- Session state ---
    conversation_history = []
    current_pipeline_task = None
    current_semantic_task = None
    is_speaking = False
    barged_in = False  # B4: Server-side gate — blocks in-flight audio after barge-in
    pipeline_generation = 0  # B5: Monotonic counter — incremented on every barge-in/reset

    # --- Semantic layer state ---
    # One-turn-behind buffer: semantic result from previous turn,
    # injected into next turn's system prompt
    latest_semantic_context = None
    # Accumulator for post-call report
    turn_log = []
    turn_counter = 0

    # --- Acoustic layer state ---
    latest_acoustic_context = None
    acoustic_chunks_this_turn = []  # accumulate per-turn for averaging
    current_acoustic_task = None

    # --- Speculative RAG state ---
    # Cache for early RAG results triggered at confidence >= 0.7
    speculative_rag_cache = {"query": "", "chunks": [], "embedding": None}

    # --- Audio forking queues ---
    # In production, FreeSWITCH duplicates RTP into two WebSocket streams.
    # Here, we duplicate the browser audio into two asyncio queues.
    stt_queue = asyncio.Queue()
    acoustic_queue = asyncio.Queue()  # Legacy queue (kept for binary fork narrative)

    # --- Deepgram STT setup ---
    stt = DeepgramSTT(api_key=DEEPGRAM_API_KEY)
    stt_connected = False  # Track whether Deepgram is connected

    # --- Deepgram TTS setup (persistent WebSocket per session) ---
    tts = DeepgramTTS(api_key=DEEPGRAM_API_KEY)
    tts_connected = False

    # --- Deepgram keep-alive ---
    async def deepgram_keepalive():
        """Send KeepAlive messages to Deepgram every 8s to prevent timeout
        during periods when the agent is speaking (no mic audio forwarded)."""
        try:
            while True:
                await asyncio.sleep(8)
                if stt_connected and stt.ws and stt.is_connected():
                    try:
                        await stt.ws.send(json.dumps({"type": "KeepAlive"}))
                    except Exception:
                        pass
        except asyncio.CancelledError:
            pass

    # --- Lazy Deepgram Connection ---
    async def trigger_barge_in():
        nonlocal is_speaking, barged_in, pipeline_generation, current_pipeline_task, current_semantic_task, current_acoustic_task

        if not is_speaking:
            return

        print("[BARGE-IN] User interrupted (VAD triggered)")

        # B6: Check if the interruption is just a backchannel response
        latest_transcript = " ".join(stt._final_segments).strip() if stt._final_segments else ""
        if latest_transcript and is_backchannel(latest_transcript):
            print(f"[BARGE-IN] B6: Backchannel detected: '{latest_transcript}' — soft pause only")
            if tts_connected and tts.is_connected():
                await tts.clear()
            stt.flush()
            await asyncio.sleep(0.3)
            return

        # Full barge-in reset
        barged_in = True
        pipeline_generation += 1
        is_speaking = False
        stt.flush()

        if tts_connected and tts.is_connected():
            await tts.clear()
        if current_pipeline_task and not current_pipeline_task.done():
            current_pipeline_task.cancel()
        if current_semantic_task and not current_semantic_task.done():
            current_semantic_task.cancel()
        if current_acoustic_task and not current_acoustic_task.done():
            current_acoustic_task.cancel()
        acoustic_chunks_this_turn.clear()

        # Tell the frontend to instantly stop playing whatever audio it has queued
        try:
            await websocket.send_text(json.dumps({"type": "clear_audio"}))
        except Exception:
            pass

    async def ensure_stt_connected():
        """Connect to Deepgram lazily (only when audio actually arrives).
        Also handles reconnection if Deepgram dropped."""
        nonlocal stt_connected
        if stt_connected and stt.ws and stt.is_connected():
            return True
        try:
            if stt.ws:  # Close stale connection if any
                await stt.close()
            stt_connected = False
            await stt.connect(on_transcript, on_utterance_end, on_confident_interim)
            stt_connected = True
            print("[STT] Deepgram connected (lazy init on first audio)")
            return True
        except Exception as e:
            print(f"[STT] ❌ Failed to connect to Deepgram: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Deepgram connection failed: {e}",
            }))
            return False

    # --- STT Queue Consumer ---
    stt_send_count = 0  # DEBUG counter
    async def stt_consumer():
        """Pull audio from stt_queue, ensure Deepgram is connected, and forward."""
        nonlocal stt_send_count
        try:
            while True:
                audio_bytes = await stt_queue.get()
                # Lazy connect: only open Deepgram when audio actually flows
                if not stt_connected or not (stt.ws and stt.is_connected()):
                    connected = await ensure_stt_connected()
                    if not connected:
                        continue  # Drop audio if we can't connect
                stt_send_count += 1
                if stt_send_count <= 5 or stt_send_count % 20 == 0:
                    print(f"[STT-TRACE] Sending chunk #{stt_send_count}: {len(audio_bytes)} bytes, connected={stt.is_connected()}")
                await stt.send(audio_bytes)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[STT-TRACE] ❌ stt_consumer crashed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    # --- Acoustic Queue Drain ---
    # Binary webm audio still goes to this queue for the "forking" narrative,
    # but actual acoustic analysis uses raw PCM from AudioWorklet (separate path).
    async def acoustic_drain():
        """Drain acoustic queue to prevent memory buildup."""
        try:
            while True:
                await acoustic_queue.get()
        except asyncio.CancelledError:
            pass

    # --- Acoustic PCM Handler ---
    async def handle_acoustic_pcm(pcm_base64: str, sample_rate: int, interrupted: bool = False):
        """Process raw PCM audio from AudioWorklet through the dual-path engine."""
        nonlocal latest_acoustic_context, current_acoustic_task
        try:
            start_time = time.time()
            result = await analyze_audio(pcm_base64, sample_rate, interrupted)
            elapsed = round((time.time() - start_time) * 1000)

            if result.get("is_speech", False):
                latest_acoustic_context = result
                acoustic_chunks_this_turn.append(result)

            # Push to frontend dashboard
            await websocket.send_text(json.dumps({
                "type": "acoustic",
                "data": result,
                "latency_ms": elapsed,
            }))

        except Exception as e:
            print(f"[ACOUSTIC] Error: {e}")

    # --- Transcript callback ---
    async def on_transcript(text: str, is_final: bool, confidence: float = 0.0):
        try:
            await websocket.send_text(json.dumps({
                "type": "transcript",
                "text": text,
                "is_final": is_final,
                "confidence": confidence,
            }))
        except Exception:
            pass

    # --- Speculative RAG callback (fires at confidence >= 0.7) ---
    async def on_confident_interim(text: str):
        """Run RAG query early on high-confidence interim text.
        Result is cached and reused if final text is similar enough."""
        nonlocal speculative_rag_cache
        if not rag_engine or is_speaking:
            return
        try:
            # Only re-query if text has changed significantly from last speculative query
            if speculative_rag_cache["query"] and text.startswith(speculative_rag_cache["query"][:20]):
                return  # Same prefix, cached result is likely still valid

            start = time.time()
            results = rag_engine.retrieve(text, n_results=2)
            elapsed = round((time.time() - start) * 1000)

            # Cache the query embedding for similarity comparison later
            query_embedding = rag_engine._embed_query(text)
            speculative_rag_cache = {
                "query": text,
                "chunks": results,
                "embedding": query_embedding,
            }
            print(f"[RAG] ⚡ Speculative RAG cached: '{text[:50]}...' → {len(results)} chunks in {elapsed}ms")
        except Exception as e:
            print(f"[RAG] Speculative RAG error: {e}")

    # --- Utterance end: fire BOTH main pipeline and semantic analysis ---
    async def on_utterance_end(full_text: str):
        nonlocal current_pipeline_task, current_semantic_task, is_speaking, turn_counter
        if is_speaking:
            return

        print(f"[STT] Utterance: {full_text}")
        turn_counter += 1

        # B5: Capture generation at launch time
        gen = pipeline_generation

        # Fire main pipeline
        current_pipeline_task = asyncio.create_task(
            run_pipeline(full_text, turn_counter, gen)
        )

        # Snapshot acoustic chunks for this turn and reset for next turn
        turn_acoustic_snapshot = list(acoustic_chunks_this_turn)
        acoustic_chunks_this_turn.clear()

        # Fire semantic analysis concurrently (does NOT block main pipeline)
        conversation_summary = " | ".join(
            f"{t['role']}: {t['content'][:80]}" for t in conversation_history[-6:]
        )
        current_semantic_task = asyncio.create_task(
            run_semantic_analysis(full_text, turn_counter, conversation_summary, turn_acoustic_snapshot)
        )

    # --- Semantic Analysis (runs parallel to main pipeline) ---
    async def run_semantic_analysis(user_text: str, turn_id: int, conv_summary: str, acoustic_snapshot: list):
        nonlocal latest_semantic_context
        try:
            print(f"[SEMANTIC] Analyzing turn {turn_id}...")
            start_time = time.time()

            # PII redact before sending to LLM
            redacted_text, _ = redact_pii(user_text)

            result = await analyze_utterance(
                redacted_text, GEMINI_API_KEY, conv_summary
            )

            elapsed = round((time.time() - start_time) * 1000)
            print(f"[SEMANTIC] Turn {turn_id} done in {elapsed}ms: {result.get('intent', '?')}")

            # Store for next turn's context injection
            latest_semantic_context = result

            # Compute average acoustic stress for this turn and combined risk
            turn_acoustic = None
            combined_risk = False
            
            if acoustic_snapshot:
                avg_stress = sum(a.get("stress_score", 0) for a in acoustic_snapshot) / len(acoustic_snapshot)
                
                # Get most common emotion
                emotions = [a.get("emotion", "unknown") for a in acoustic_snapshot]
                most_common_emotion = max(set(emotions), key=emotions.count) if emotions else "unknown"
                
                turn_acoustic = {
                    "stress_score": round(avg_stress, 2),
                    "emotion": most_common_emotion
                }
                
                # Combined Risk Signal: high acoustic stress + negative semantic sentiment = escalation alert
                if avg_stress > 0.6 and result.get("sentiment", 0) < -0.3:
                    combined_risk = True
                    result["escalation_recommended"] = True # Override semantic if acoustic validates high stress

            # Append to turn log for post-call report
            turn_log.append({
                "turn_id": turn_id,
                "utterance": redacted_text,
                "semantic": result,
                "acoustic": turn_acoustic,
                "combined_risk": combined_risk,
                "interrupted": False,
                "timestamp": time.time(),
            })

            # Push to frontend dashboard panel
            await websocket.send_text(json.dumps({
                "type": "semantic",
                "data": result,
                "turn_id": turn_id,
                "latency_ms": elapsed,
            }))

        except asyncio.CancelledError:
            # Barge-in: tag as interrupted if we had partial results
            print(f"[SEMANTIC] Turn {turn_id} cancelled (barge-in)")
            turn_log.append({
                "turn_id": turn_id,
                "utterance": user_text[:100],
                "semantic": None,
                "acoustic": None,
                "combined_risk": False,
                "interrupted": True,
                "timestamp": time.time(),
            })
        except Exception as e:
            print(f"[SEMANTIC] Error: {e}")

    # --- Main Voice Pipeline ---
    async def run_pipeline(user_text: str, turn_id: int, expected_generation: int = -1):
        nonlocal is_speaking, barged_in, speculative_rag_cache

        # Telemetry State (Phase 8)
        turn_start_time = time.time()
        telemetry = {
            "turn_id": turn_id,
            "stt_ms": 200, # Simulated average STT time since utterance end
            "rag_ms": 0,
            "llm_ttft_ms": 0,
            "tts_first_audio_ms": 0,
            "total_turn_ms": 0
        }

        # B5: Stale pipeline guard — abort if generation changed since launch
        if expected_generation >= 0 and pipeline_generation != expected_generation:
            print(f"[PIPELINE] Turn {turn_id} skipped (stale generation {expected_generation} != {pipeline_generation})")
            return

        print(f"[PIPELINE] Starting pipeline for turn {turn_id}: '{user_text[:80]}'")
        try:
            await websocket.send_text(json.dumps({"type": "state", "state": "THINKING"}))

            # Lazy TTS connection (only connect when first pipeline fires)
            nonlocal tts_connected
            if not tts_connected or not tts.is_connected():
                try:
                    await tts.connect()
                    tts_connected = True
                except Exception as tts_conn_err:
                    print(f"[TTS] ⚠️ WebSocket connect failed: {tts_conn_err}, will use HTTP fallback")
                    tts_connected = False

            # Step 1: PII Redaction
            user_text, pii_findings = redact_pii(user_text)
            if pii_findings:
                print(f"[PII] Redacted: {pii_findings}")

            # B5: Check generation after PII (async boundary)
            if expected_generation >= 0 and pipeline_generation != expected_generation:
                print(f"[PIPELINE] Turn {turn_id} aborted after PII (stale generation)")
                return

            # Step 2: RAG Retrieval (use speculative cache if available)
            rag_context = ""
            rag_chunks = []
            if rag_engine:
                # Check if we have a valid speculative RAG cache
                cache_used = False
                if speculative_rag_cache["chunks"] and speculative_rag_cache["embedding"]:
                    # Compare final text embedding to cached query embedding
                    final_embedding = rag_engine._embed_query(user_text)
                    cached_embedding = speculative_rag_cache["embedding"]
                    # Cosine similarity between final and speculative queries
                    cos_sim = float(np.dot(final_embedding, cached_embedding) / (
                        np.linalg.norm(final_embedding) * np.linalg.norm(cached_embedding)
                    ))
                    if cos_sim > 0.85:
                        rag_chunks = speculative_rag_cache["chunks"]
                        cache_used = True
                        print(f"[RAG] ✅ Using speculative cache (similarity={cos_sim:.3f})")
                    else:
                        print(f"[RAG] ⚠️ Speculative cache stale (similarity={cos_sim:.3f}), re-querying")

                if not cache_used:
                    rag_start_time = time.time()
                    results = rag_engine.retrieve(user_text, n_results=2)
                    telemetry["rag_ms"] = round((time.time() - rag_start_time) * 1000)
                    if results:
                        rag_chunks = results

                # Reset speculative cache after use
                speculative_rag_cache = {"query": "", "chunks": [], "embedding": None}

                if rag_chunks:
                    rag_context = "\n\n---\nRelevant knowledge base context:\n"
                    rag_context += "\n".join(f"- {c}" for c in rag_chunks)
                    rag_context += "\n---\n"

            await websocket.send_text(json.dumps({
                "type": "rag_context",
                "chunks": rag_chunks,
                "query": user_text,
            }))

            # Step 3: Build system prompt with semantic context injection
            enhanced_prompt = SYSTEM_PROMPT + rag_context

            if latest_semantic_context:
                sem = latest_semantic_context
                enhanced_prompt += (
                    "\n\n---\nSEMANTIC INTELLIGENCE FROM PREVIOUS TURN:\n"
                    f"- Customer intent: {sem.get('intent', 'unknown')}\n"
                    f"- Sentiment: {sem.get('sentiment', 0)}\n"
                    f"- Urgency: {sem.get('urgency_level', 'low')}\n"
                    f"- Escalation needed: {sem.get('escalation_recommended', False)}\n"
                    f"- Summary: {sem.get('one_line_summary', '')}\n"
                    f"- Recommended tone: {sem.get('recommended_tone', 'professional')}\n"
                    "Use this context to adapt your response tone and approach.\n---\n"
                )

            # Inject acoustic context (stress, emotion from previous turn)
            if latest_acoustic_context:
                ac = latest_acoustic_context
                enhanced_prompt += (
                    "\n\n---\nACOUSTIC INTELLIGENCE FROM PREVIOUS TURN:\n"
                    f"- Detected emotion: {ac.get('emotion', 'unknown')} ({ac.get('emotion_confidence', 0):.0%})\n"
                    f"- Stress level: {ac.get('stress_score', 0):.0%}\n"
                    f"- Volume: {ac.get('rms_db', -60):.1f} dB\n"
                    f"- Summary: {ac.get('acoustic_summary', '')}\n"
                    "Adapt your tone based on how the caller SOUNDS, not just what they say.\n---\n"
                )

            # Step 4: Build OpenAI-format messages for Groq
            messages = [{"role": "system", "content": enhanced_prompt}]
            for turn in conversation_history:
                messages.append({"role": turn["role"], "content": turn["content"]})
            messages.append({"role": "user", "content": user_text})

            # Step 5: Stream LLM response from Groq with cascaded TTS
            # B5: Final generation check before committing to SPEAKING state
            if expected_generation >= 0 and pipeline_generation != expected_generation:
                print(f"[PIPELINE] Turn {turn_id} aborted before Groq (stale generation)")
                return

            is_speaking = True
            barged_in = False  # B4: Reset barge-in gate for new pipeline
            await websocket.send_text(json.dumps({"type": "state", "state": "SPEAKING"}))

            full_response = ""
            text_buffer = ""

            print(f"[PIPELINE] Calling Groq {GROQ_MODEL} (stream=True)...")
            groq_start = time.time()
            first_token_logged = False
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        async with client.stream(
                            "POST",
                            GROQ_API_URL,
                            headers={
                                "Authorization": f"Bearer {GROQ_API_KEY}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": GROQ_MODEL,
                                "messages": messages,
                                "stream": True,
                                "max_tokens": 512,
                                "temperature": 0.7,
                            },
                        ) as response:
                            if response.status_code == 429 and attempt < max_retries - 1:
                                print(f"[PIPELINE] ⚠️ Groq rate limit hit (429). Retrying {attempt + 1}/{max_retries} in 1s...")
                                await asyncio.sleep(1.0)
                                continue
                            elif response.status_code != 200:
                                error_body = await response.aread()
                                raise Exception(f"Groq API returned {response.status_code}: {error_body.decode()[:300]}")

                            async for line in response.aiter_lines():
                                # B4: Hard stop if barge-in flag is set
                                if barged_in:
                                    break
                                if not is_speaking:
                                    break

                                # SSE format: "data: {...}" or "data: [DONE]"
                                if not line.startswith("data: "):
                                    continue
                                payload = line[6:]  # strip "data: " prefix
                                if payload == "[DONE]":
                                    break

                                try:
                                    chunk_data = json.loads(payload)
                                except json.JSONDecodeError:
                                    continue

                                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                                token_text = delta.get("content", "")

                                if not token_text:
                                    continue

                                if not first_token_logged:
                                    ttft = round((time.time() - groq_start) * 1000)
                                    telemetry["llm_ttft_ms"] = ttft
                                    print(f"[PIPELINE] ⚡ Groq TTFT: {ttft}ms")
                                    first_token_logged = True

                                full_response += token_text
                                text_buffer += token_text

                                await websocket.send_text(json.dumps({
                                    "type": "chunk",
                                    "text": token_text,
                                }))

                                sentence, remaining = detect_sentence_boundary(text_buffer)
                                if sentence:
                                    text_buffer = remaining
                                    # B4: Skip TTS synthesis and send if barged in
                                    if barged_in:
                                        continue
                                    try:
                                        print(f"[TTS] Synthesizing: '{sentence[:60]}...'")
                                        if tts_connected and tts.is_connected():
                                            audio_bytes = await tts.synthesize(sentence)
                                        else:
                                            audio_bytes = await synthesize_http(sentence, DEEPGRAM_API_KEY)
                                        # B4: Re-check after async TTS call
                                        if barged_in:
                                            continue
                                        print(f"[TTS] ✅ Got {len(audio_bytes)} bytes of audio")
                                        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                                        
                                        if telemetry["tts_first_audio_ms"] == 0:
                                            telemetry["tts_first_audio_ms"] = round((time.time() - turn_start_time) * 1000)

                                        await websocket.send_text(json.dumps({
                                            "type": "audio",
                                            "data": audio_b64,
                                            "format": "mp3",
                                        }))
                                    except Exception as e:
                                        print(f"[TTS] ❌ Error: {e}")
                                        traceback.print_exc()

                    # Successful stream without errors/retries, exit retry loop
                    break

                except Exception as stream_err:
                    if "429" in str(stream_err) and attempt < max_retries - 1:
                        print(f"[PIPELINE] ⚠️ Groq rate limit hit via exception. Retrying {attempt + 1}/{max_retries} in 1s...")
                        await asyncio.sleep(1.0)
                        continue
                    # Reraise if it's not a rate limit or we're out of retries
                    raise stream_err

            # TTS remaining buffer
            if text_buffer.strip() and is_speaking and not barged_in:
                try:
                    if tts_connected and tts.is_connected():
                        audio_bytes = await tts.synthesize(text_buffer.strip())
                    else:
                        audio_bytes = await synthesize_http(text_buffer.strip(), DEEPGRAM_API_KEY)
                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                    await websocket.send_text(json.dumps({
                        "type": "audio",
                        "data": audio_b64,
                        "format": "mp3",
                    }))
                except Exception as e:
                    print(f"[TTS] Error on final chunk: {e}")

            groq_total = round((time.time() - groq_start) * 1000)
            print(f"[PIPELINE] Groq total generation: {groq_total}ms, response length: {len(full_response)} chars")

            # Update conversation history
            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": full_response})

            # Conversation history windowing: keep only last N turns
            # to prevent unbounded memory growth and Groq context overflow
            if len(conversation_history) > MAX_HISTORY_TURNS * 2:
                conversation_history[:] = conversation_history[-(MAX_HISTORY_TURNS * 2):]

            await websocket.send_text(json.dumps({
                "type": "done",
                "full_text": full_response,
            }))

            # Send Telemetry (Phase 8)
            telemetry["total_turn_ms"] = round((time.time() - turn_start_time) * 1000)
            await websocket.send_text(json.dumps({
                "type": "telemetry",
                "data": telemetry
            }))

            is_speaking = False
            await websocket.send_text(json.dumps({"type": "state", "state": "LISTENING"}))

        except asyncio.CancelledError:
            print(f"[PIPELINE] Turn {turn_id} cancelled (barge-in)")
            is_speaking = False
        except Exception as e:
            error_str = str(e)
            print(f"[PIPELINE] ❌ Error in turn {turn_id}: {error_str}")
            traceback.print_exc()
            
            # If we hit rate limits (Groq free tier: 30 req/min)
            if "429" in error_str or "rate" in error_str.lower() or "quota" in error_str.lower():
                try:
                    fallback_text = "I apologize, but I'm currently experiencing high traffic. Please wait a moment and try again."
                    await websocket.send_text(json.dumps({
                        "type": "chunk",
                        "text": fallback_text,
                    }))
                    
                    if tts_connected and tts.is_connected():
                        audio_bytes = await tts.synthesize(fallback_text)
                    else:
                        audio_bytes = await synthesize_http(fallback_text, DEEPGRAM_API_KEY)
                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                    await websocket.send_text(json.dumps({
                        "type": "audio",
                        "data": audio_b64,
                        "format": "mp3",
                    }))
                except Exception as tts_err:
                    print(f"[TTS] Fallback error: {tts_err}")
            else:
                # For non-rate-limit errors, send to UI
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": error_str,
                    }))
                except Exception:
                    print(f"[PIPELINE] ❌ Also failed to send error to client")
            
            is_speaking = False

    # --- Start forking queue consumers ---
    # NOTE: Deepgram is NOT connected yet. It connects lazily when
    # the first audio chunk arrives (via stt_consumer → ensure_stt_connected).
    # This prevents the timeout: "Deepgram did not receive audio data
    # within the timeout window" (code 1011) that occurs when the WS
    # connects before the user clicks the mic button.
    stt_task = asyncio.create_task(stt_consumer())
    acoustic_task = asyncio.create_task(acoustic_drain())
    keepalive_task = asyncio.create_task(deepgram_keepalive())
    print("[STT] Deepgram will connect lazily when mic audio starts")

    # --- Main receive loop ---
    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # Binary frame = audio from browser mic → FORK to both queues
            if "bytes" in message:
                audio = message["bytes"]
                # DEBUG: Log first audio chunk and periodic sizes
                if not hasattr(websocket, '_audio_chunks_count'):
                    websocket._audio_chunks_count = 0
                    websocket._audio_bytes_total = 0
                websocket._audio_chunks_count += 1
                websocket._audio_bytes_total += len(audio)
                if websocket._audio_chunks_count == 1:
                    print(f"[WS] 📥 First audio chunk received: {len(audio)} bytes")
                elif websocket._audio_chunks_count % 20 == 0:
                    print(f"[WS] 📥 Audio chunks: {websocket._audio_chunks_count}, total: {websocket._audio_bytes_total:,} bytes")
                stt_queue.put_nowait(audio)
                acoustic_queue.put_nowait(audio)

            # Text frame = control message
            elif "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type", "")

                if msg_type == "barge_in":
                    await trigger_barge_in()

                elif msg_type == "acoustic_pcm":
                    # Raw PCM audio from AudioWorklet (every 1.5s)
                    pcm_data = data.get("data", "")
                    sample_rate = data.get("sample_rate", 48000)
                    if pcm_data:
                        current_acoustic_task = asyncio.create_task(
                            handle_acoustic_pcm(pcm_data, sample_rate)
                        )

                elif msg_type == "text_input":
                    text = data.get("text", "")
                    history = data.get("history", [])

                    # --- Barge-in cleanup if agent is mid-speech ---
                    if is_speaking or (current_pipeline_task and not current_pipeline_task.done()):
                        print("[BARGE-IN] Text input while speaking — flushing")
                        barged_in = True  # B4
                        pipeline_generation += 1  # B5
                        is_speaking = False
                        stt.flush()
                        if current_pipeline_task and not current_pipeline_task.done():
                            current_pipeline_task.cancel()
                        if current_semantic_task and not current_semantic_task.done():
                            current_semantic_task.cancel()
                        if current_acoustic_task and not current_acoustic_task.done():
                            current_acoustic_task.cancel()
                        acoustic_chunks_this_turn.clear()

                    conversation_history = [
                        {"role": h["role"], "content": h["content"]}
                        for h in history
                    ]
                    if text.strip():
                        turn_counter += 1
                        gen = pipeline_generation  # B5: Capture generation
                        current_pipeline_task = asyncio.create_task(
                            run_pipeline(text, turn_counter, gen)
                        )
                        # Also run semantic on text input
                        current_semantic_task = asyncio.create_task(
                            run_semantic_analysis(text, turn_counter, "", [])
                        )

                elif msg_type == "end_call":
                    # Generate post-call report BEFORE disconnect
                    print(f"[SESSION] End call requested, generating report for {len(turn_log)} turns...")
                    if turn_log:
                        try:
                            os.makedirs("reports", exist_ok=True)

                            log_text = ""
                            for turn in turn_log:
                                sem = turn.get('semantic') or {}
                                ac = turn.get('acoustic') or {}
                                log_text += f"\nTurn {turn['turn_id']}:\n"
                                log_text += f"User: {turn['utterance']}\n"
                                log_text += f"Intent: {sem.get('intent', 'unknown')}\n"
                                log_text += f"Sentiment: {sem.get('sentiment', 0)}\n"
                                if ac:
                                    log_text += f"Emotion: {ac.get('emotion', 'unknown')}, Stress: {ac.get('stress_score', 0):.0%}\n"
                                log_text += f"Interrupted: {turn['interrupted']}\n"

                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            formatted_date = datetime.datetime.now().strftime("%B %d, %Y")
                            prompt = (
                                "You are a Quality Assurance AI for NeoBank's call center. "
                                "Review the following call log and generate a brief Post-Call Report in Markdown.\n"
                                f"IMPORTANT CONTEXT:\n- Call ID: {timestamp}\n- Date: {formatted_date}\n- Agent: Harris\n\n"
                                "Include:\n1. Executive Summary\n2. Primary Intent & Resolution\n3. Caller Sentiment & Stress progression\n4. Action Items / Follow-ups.\n"
                                f"\nCall Log:\n{log_text}"
                            )

                            report_model = genai.GenerativeModel("gemini-2.5-flash")
                            response = await report_model.generate_content_async(prompt)
                            report_text = response.text

                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            report_path = f"reports/post_call_report_{timestamp}.md"
                            with open(report_path, "w", encoding="utf-8") as f:
                                f.write(f"# Post-Call Report\n\n**Generated at:** {timestamp}\n**Turns:** {len(turn_log)}\n\n")
                                f.write(report_text)
                            print(f"[SESSION] Report saved to {report_path}")

                            await websocket.send_text(json.dumps({
                                "type": "report",
                                "data": report_text,
                                "turns": len(turn_log),
                            }))
                        except Exception as e:
                            print(f"[SESSION] Failed to generate report: {e}")
                            await websocket.send_text(json.dumps({
                                "type": "report",
                                "data": f"Report generation failed: {e}",
                                "turns": len(turn_log),
                            }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "report",
                            "data": "No turns recorded in this session.",
                            "turns": 0,
                        }))

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        # Cleanup
        stt_task.cancel()
        acoustic_task.cancel()
        keepalive_task.cancel()
        if stt_connected:
            await stt.close()
        if tts_connected:
            await tts.close()
        if current_pipeline_task and not current_pipeline_task.done():
            current_pipeline_task.cancel()
        if current_semantic_task and not current_semantic_task.done():
            current_semantic_task.cancel()
        if current_acoustic_task and not current_acoustic_task.done():
            current_acoustic_task.cancel()

        print(f"[WS] Session cleaned up ({len(turn_log)} turns logged)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
