"""
miniVoxSetu — Backend Entry Point

WHY THIS FILE EXISTS:
This is the FastAPI server that bridges the browser (which handles mic/speaker)
with the AI brain (Gemini LLM + RAG). In production voice AI systems like VoxSetu,
the backend orchestrates STT→LLM→TTS. Here, STT and TTS happen in the browser
(free!), so the backend only handles LLM + RAG — but the architecture mirrors
a real system so you learn the patterns.
"""

import os
import json
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai

from rag import RAGEngine

# WHY: .env keeps secrets out of source code. In production you'd use
# a secret manager (AWS Secrets Manager, GCP Secret Manager), but .env
# is the standard for local dev.
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file. "
        "Get a free key from https://aistudio.google.com/app/apikey"
    )

genai.configure(api_key=GEMINI_API_KEY)

# WHY: We initialize the RAG engine at startup so embeddings are computed once,
# not on every request. In production, you'd pre-compute embeddings offline
# and load the vector DB from disk.
rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    WHY: FastAPI's lifespan context manager lets us run setup code once
    at startup and cleanup at shutdown. We use it to initialize the RAG
    engine (embed FAQ documents into ChromaDB) before any requests arrive.
    """
    global rag_engine
    rag_engine = RAGEngine(api_key=GEMINI_API_KEY)
    rag_engine.initialize()
    print("[OK] RAG engine initialized with FAQ documents")
    yield
    # Cleanup would go here if needed


app = FastAPI(
    title="miniVoxSetu",
    description="Minimal Voice AI Agent for learning",
    lifespan=lifespan,
)

# WHY: CORS must be configured because the React frontend runs on a different
# port (5173) than the backend (8000). Without CORS, the browser blocks
# cross-origin requests — this is a security feature of browsers, not a bug.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, lock this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WHY: We define the system prompt here so it's consistent across all turns.
# The system prompt shapes the AI's personality and behavior. In production
# voice AI, this prompt is carefully tuned for the use case (banking, support, etc.)
SYSTEM_PROMPT = """You are a helpful voice AI assistant for NeoBank, a fictional digital bank.
Keep responses concise and conversational — remember, the user is LISTENING to your response,
not reading it. Aim for 2-3 sentences max unless the user asks for detail.
If you receive context from the knowledge base, use it to answer accurately.
Never say "according to the knowledge base" — just answer naturally as if you know it."""


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "rag_initialized": rag_engine is not None}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WHY WE USE WEBSOCKETS INSTEAD OF REST:
    Voice AI needs streaming — the LLM generates tokens one by one, and we want
    to send each chunk to the frontend immediately so TTS can start speaking
    before the full response is ready. This reduces perceived latency dramatically.
    REST would require waiting for the entire response before sending it back.
    In production systems like VoxSetu, WebSockets (or gRPC streams) are standard
    for this exact reason.
    """
    await websocket.accept()
    print("[WS] WebSocket client connected")

    try:
        while True:
            # WHY: We receive the full conversation history from the frontend
            # on every turn. This is how stateless LLM APIs work — the model
            # has NO memory between calls. The conversation history array IS
            # the model's memory. This is the "context window" concept.
            data = await websocket.receive_text()
            message = json.loads(data)

            user_text = message.get("text", "")
            conversation_history = message.get("history", [])

            if not user_text.strip():
                continue

            # --- RAG RETRIEVAL STEP ---
            # WHY: The LLM doesn't know anything about NeoBank (our fictional bank).
            # RAG (Retrieval-Augmented Generation) fixes this: before every LLM call,
            # we search our FAQ vector database for relevant information and inject
            # it into the prompt. This is how production AI agents know domain-specific
            # information without fine-tuning the model.
            rag_context = ""
            rag_chunks = []
            if rag_engine:
                results = rag_engine.retrieve(user_text, n_results=2)
                if results:
                    rag_chunks = results
                    rag_context = "\n\n---\nRelevant knowledge base context:\n"
                    rag_context += "\n".join(f"- {chunk}" for chunk in results)
                    rag_context += "\n---\n"

            # WHY: We send the RAG chunks back to the frontend so the user can SEE
            # what was retrieved. This transparency is crucial for debugging RAG
            # systems — in production, you'd log this for quality monitoring.
            await websocket.send_text(json.dumps({
                "type": "rag_context",
                "chunks": rag_chunks,
                "query": user_text,
            }))

            # --- BUILD THE PROMPT WITH FULL CONVERSATION HISTORY ---
            # WHY: We pass the FULL conversation history every single call because
            # Gemini (like all LLM APIs) has no memory between API calls — the
            # context window is our memory. Each element in this array is a turn.
            # In production, you'd also manage context window limits (truncating
            # old turns when approaching the token limit).
            gemini_history = []
            for turn in conversation_history:
                role = "user" if turn["role"] == "user" else "model"
                gemini_history.append({
                    "role": role,
                    "parts": [turn["content"]]
                })

            # WHY: We create a new model instance per request. This is stateless
            # by design — the model doesn't remember previous conversations.
            # The system instruction sets the AI's persona and behavior.
            model = genai.GenerativeModel(
                "gemini-2.5-flash",
                system_instruction=SYSTEM_PROMPT + rag_context,
            )

            # WHY: We start a chat with the history so Gemini sees the full
            # conversation context. Then we send the latest user message.
            chat = model.start_chat(history=gemini_history)

            # --- STREAMING RESPONSE ---
            # WHY: Streaming sends tokens as they're generated instead of waiting
            # for the complete response. This is CRITICAL for voice AI because:
            # 1. TTS can start speaking the first sentence while the rest generates
            # 2. Perceived latency drops from seconds to milliseconds
            # 3. The user feels like the AI is "thinking and speaking" naturally
            # In production, this is called "incremental TTS" or "streaming synthesis"
            full_response = ""
            try:
                # WHY: We use send_message_async (not send_message) because this
                # is an async handler. The sync version BLOCKS the entire event
                # loop during each network round-trip to Gemini, preventing the
                # server from handling ANY other WebSocket connections. Always
                # use async I/O inside async functions.
                response = await chat.send_message_async(user_text, stream=True)

                async for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        # WHY: Each chunk is sent immediately via WebSocket.
                        # The frontend accumulates these chunks and can start
                        # TTS on complete sentences while more text arrives.
                        await websocket.send_text(json.dumps({
                            "type": "chunk",
                            "text": chunk.text,
                        }))

                # WHY: We send a "done" message so the frontend knows the full
                # response is complete. This is important for updating the
                # conversation history with the complete assistant turn.
                await websocket.send_text(json.dumps({
                    "type": "done",
                    "full_text": full_response,
                }))

            except Exception as e:
                print(f"[ERROR] Gemini API error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e),
                }))

    except WebSocketDisconnect:
        print("[WS] WebSocket client disconnected")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")


if __name__ == "__main__":
    import uvicorn
    # WHY: We run on 0.0.0.0 so the server is accessible from any network
    # interface (not just localhost). Port 8000 is the FastAPI convention.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
