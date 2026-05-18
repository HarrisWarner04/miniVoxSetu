"""
miniVoxSetu — Backend Entry Point (Phase 3: Acoustic Intelligence Layer)
Architecture: Browser audio → asyncio.Queue fork → STT path + Acoustic path
On complete utterance: fires main pipeline AND semantic analysis concurrently.
Acoustic PCM chunks arrive separately via AudioWorklet → analyzed via ThreadPoolExecutor.
"""

import os
import json
import asyncio
import base64
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai

from rag import RAGEngine
from pii import redact_pii
from stt import DeepgramSTT
from tts import synthesize, detect_sentence_boundary
from semantic import analyze_utterance
from acoustic import analyze_audio, init_hubert_model

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY not found in .env")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)

rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    rag_engine = RAGEngine(api_key=GEMINI_API_KEY)
    rag_engine.initialize()
    print("[OK] RAG engine initialized")

    # Load HuBERT model onto GPU/CPU (lazy init — won't crash if deps missing)
    init_hubert_model()

    yield


app = FastAPI(title="miniVoxSetu", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.get("/health")
async def health_check():
    return {"status": "ok", "rag_initialized": rag_engine is not None}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    # --- Session state ---
    conversation_history = []
    current_pipeline_task = None
    current_semantic_task = None
    is_speaking = False

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

    # --- Audio forking queues ---
    # In production, FreeSWITCH duplicates RTP into two WebSocket streams.
    # Here, we duplicate the browser audio into two asyncio queues.
    stt_queue = asyncio.Queue()
    acoustic_queue = asyncio.Queue()  # Legacy queue (kept for binary fork narrative)

    # --- Deepgram STT setup ---
    stt = DeepgramSTT(api_key=DEEPGRAM_API_KEY)
    stt_connected = False  # Track whether Deepgram is connected

    # --- Lazy Deepgram Connection ---
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
            await stt.connect(on_transcript, on_utterance_end)
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
    async def on_transcript(text: str, is_final: bool):
        try:
            await websocket.send_text(json.dumps({
                "type": "transcript",
                "text": text,
                "is_final": is_final,
            }))
        except Exception:
            pass

    # --- Utterance end: fire BOTH main pipeline and semantic analysis ---
    async def on_utterance_end(full_text: str):
        nonlocal current_pipeline_task, current_semantic_task, is_speaking, turn_counter
        if is_speaking:
            return

        print(f"[STT] Utterance: {full_text}")
        turn_counter += 1

        # Fire main pipeline
        current_pipeline_task = asyncio.create_task(
            run_pipeline(full_text, turn_counter)
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
    async def run_pipeline(user_text: str, turn_id: int):
        nonlocal is_speaking

        try:
            await websocket.send_text(json.dumps({"type": "state", "state": "THINKING"}))

            # Step 1: PII Redaction
            user_text, pii_findings = redact_pii(user_text)
            if pii_findings:
                print(f"[PII] Redacted: {pii_findings}")

            # Step 2: RAG Retrieval
            rag_context = ""
            rag_chunks = []
            if rag_engine:
                results = rag_engine.retrieve(user_text, n_results=2)
                if results:
                    rag_chunks = results
                    rag_context = "\n\n---\nRelevant knowledge base context:\n"
                    rag_context += "\n".join(f"- {c}" for c in results)
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

            # Step 4: Gemini with conversation history
            gemini_history = []
            for turn in conversation_history:
                role = "user" if turn["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [turn["content"]]})

            model = genai.GenerativeModel(
                "gemini-2.5-flash",
                system_instruction=enhanced_prompt,
            )
            chat = model.start_chat(history=gemini_history)

            # Step 5: Stream LLM response with cascaded TTS
            is_speaking = True
            await websocket.send_text(json.dumps({"type": "state", "state": "SPEAKING"}))

            full_response = ""
            text_buffer = ""

            response = await chat.send_message_async(user_text, stream=True)

            async for chunk in response:
                if not is_speaking:
                    break

                if chunk.text:
                    full_response += chunk.text
                    text_buffer += chunk.text

                    await websocket.send_text(json.dumps({
                        "type": "chunk",
                        "text": chunk.text,
                    }))

                    sentence, remaining = detect_sentence_boundary(text_buffer)
                    if sentence:
                        text_buffer = remaining
                        try:
                            audio_bytes = await synthesize(sentence, ELEVENLABS_API_KEY)
                            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                            await websocket.send_text(json.dumps({
                                "type": "audio",
                                "data": audio_b64,
                                "format": "mp3",
                            }))
                        except Exception as e:
                            print(f"[TTS] Error: {e}")

            # TTS remaining buffer
            if text_buffer.strip() and is_speaking:
                try:
                    audio_bytes = await synthesize(text_buffer.strip(), ELEVENLABS_API_KEY)
                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                    await websocket.send_text(json.dumps({
                        "type": "audio",
                        "data": audio_b64,
                        "format": "mp3",
                    }))
                except Exception as e:
                    print(f"[TTS] Error on final chunk: {e}")

            # Update conversation history
            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": full_response})

            await websocket.send_text(json.dumps({
                "type": "done",
                "full_text": full_response,
            }))

            is_speaking = False
            await websocket.send_text(json.dumps({"type": "state", "state": "LISTENING"}))

        except asyncio.CancelledError:
            is_speaking = False
        except Exception as e:
            error_str = str(e)
            print(f"[PIPELINE] Error: {error_str}")
            
            # If we hit the Gemini Free Tier Rate Limit, speak a graceful fallback
            if "429" in error_str or "quota" in error_str.lower():
                try:
                    fallback_text = "I apologize, but my language processing systems are currently hitting a rate limit. Please wait about a minute before speaking again."
                    await websocket.send_text(json.dumps({
                        "type": "chunk",
                        "text": fallback_text,
                    }))
                    
                    audio_bytes = await synthesize(fallback_text, ELEVENLABS_API_KEY)
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
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": error_str,
                }))
            
            is_speaking = False

    # --- Start forking queue consumers ---
    # NOTE: Deepgram is NOT connected yet. It connects lazily when
    # the first audio chunk arrives (via stt_consumer → ensure_stt_connected).
    # This prevents the timeout: "Deepgram did not receive audio data
    # within the timeout window" (code 1011) that occurs when the WS
    # connects before the user clicks the mic button.
    stt_task = asyncio.create_task(stt_consumer())
    acoustic_task = asyncio.create_task(acoustic_drain())
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
                    print("[BARGE-IN] User interrupted")
                    is_speaking = False
                    stt.flush()
                    # Cancel main pipeline
                    if current_pipeline_task and not current_pipeline_task.done():
                        current_pipeline_task.cancel()
                    # Cancel semantic analysis (incomplete analysis is worthless)
                    if current_semantic_task and not current_semantic_task.done():
                        current_semantic_task.cancel()
                    # Cancel in-progress acoustic analysis
                    if current_acoustic_task and not current_acoustic_task.done():
                        current_acoustic_task.cancel()
                    # Clear acoustic buffer for this turn
                    acoustic_chunks_this_turn.clear()

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
                        current_pipeline_task = asyncio.create_task(
                            run_pipeline(text, turn_counter)
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
                            import datetime
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

                            prompt = (
                                "You are a Quality Assurance AI for NeoBank's call center. "
                                "Review the following call log and generate a brief Post-Call Report in Markdown. "
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
        if stt_connected:
            await stt.close()
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
