"""
miniVoxSetu — Backend Entry Point (Phase 2: Audio Forking + Semantic Layer)
Architecture: Browser audio → asyncio.Queue fork → STT path + Acoustic path (Phase 3)
On complete utterance: fires main pipeline AND semantic analysis concurrently.
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
    yield


app = FastAPI(title="miniVoxSetu", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are Neha, a voice AI agent handling live phone calls for NeoBank, a modern Indian digital bank.

CRITICAL — YOU ARE ON A PHONE CALL, NOT A CHATBOT:
- You are speaking to a real person who CALLED the bank. Respond as if you are on a live phone call.
- Start the very first response with a warm greeting: "Thank you for calling NeoBank! This is Neha, how may I help you today?"
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

    # --- Audio forking queues ---
    # In production, FreeSWITCH duplicates RTP into two WebSocket streams.
    # Here, we duplicate the browser audio into two asyncio queues.
    stt_queue = asyncio.Queue()
    acoustic_queue = asyncio.Queue()  # Consumer added in Phase 3

    # --- Deepgram STT setup ---
    stt = DeepgramSTT(api_key=DEEPGRAM_API_KEY)

    # --- STT Queue Consumer ---
    async def stt_consumer():
        """Pull audio from stt_queue and forward to Deepgram."""
        try:
            while True:
                audio_bytes = await stt_queue.get()
                await stt.send(audio_bytes)
        except asyncio.CancelledError:
            pass

    # --- Acoustic Queue Drain (placeholder for Phase 3) ---
    async def acoustic_drain():
        """Drain acoustic queue to prevent memory buildup. Phase 3 replaces this."""
        try:
            while True:
                await acoustic_queue.get()
                # Phase 3: resample → librosa features → buffer → WavLM inference
        except asyncio.CancelledError:
            pass

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

        # Fire semantic analysis concurrently (does NOT block main pipeline)
        conversation_summary = " | ".join(
            f"{t['role']}: {t['content'][:80]}" for t in conversation_history[-6:]
        )
        current_semantic_task = asyncio.create_task(
            run_semantic_analysis(full_text, turn_counter, conversation_summary)
        )

    # --- Semantic Analysis (runs parallel to main pipeline) ---
    async def run_semantic_analysis(user_text: str, turn_id: int, conv_summary: str):
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

            # Append to turn log for post-call report
            turn_log.append({
                "turn_id": turn_id,
                "utterance": redacted_text,
                "semantic": result,
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
            print(f"[PIPELINE] Error: {e}")
            is_speaking = False
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e),
            }))

    # --- Connect Deepgram + Start forking consumers ---
    try:
        await stt.connect(on_transcript, on_utterance_end)
        print("[STT] Deepgram connected")
    except Exception as e:
        print(f"[STT] Failed to connect: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Deepgram connection failed: {e}",
        }))
        await websocket.close()
        return

    # Start forking queue consumers
    stt_task = asyncio.create_task(stt_consumer())
    acoustic_task = asyncio.create_task(acoustic_drain())

    # --- Main receive loop ---
    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # Binary frame = audio from browser mic → FORK to both queues
            if "bytes" in message:
                audio = message["bytes"]
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

                elif msg_type == "text_input":
                    text = data.get("text", "")
                    history = data.get("history", [])
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
                            run_semantic_analysis(text, turn_counter, "")
                        )

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        # Cleanup
        stt_task.cancel()
        acoustic_task.cancel()
        await stt.close()
        if current_pipeline_task and not current_pipeline_task.done():
            current_pipeline_task.cancel()
        if current_semantic_task and not current_semantic_task.done():
            current_semantic_task.cancel()

        # Log turn summary for debugging
        if turn_log:
            print(f"[SESSION] {len(turn_log)} turns logged for post-call report")

        print("[WS] Session cleaned up")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
