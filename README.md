# miniVoxSetu — Learn Voice AI by Building One

A minimal, fully-commented voice AI agent that teaches you how production systems like VoxSetu work. Every meaningful block of code has a **WHY** comment explaining the architectural reason it exists.

## What This Project Does

You speak into your microphone → the browser converts your speech to text → the text is sent to a Gemini LLM (with domain knowledge injected via RAG) → the LLM's response is streamed back → the browser speaks it aloud. You can **interrupt the AI mid-sentence** (barge-in), and the system resets instantly.

---

## Architecture

```
Browser (Free APIs)                 Backend (Python)
┌──────────────────┐               ┌──────────────────────┐
│  getUserMedia     │── audio ────▶│                      │
│  (WebRTC)         │              │     FastAPI           │
├──────────────────┤              │     WebSocket         │
│  Web Speech API   │              │                      │
│  (STT)            │── text ────▶│  ┌────────────────┐  │
├──────────────────┤              │  │  RAG Engine     │  │
│                   │              │  │  (NumPy Vector  │  │
│                   │◀─ streaming ─│  │   Store +       │  │
│                   │   chunks     │  │   Gemini        │  │
├──────────────────┤              │  │   Embeddings)   │  │
│  Web Speech       │              │  └────────────────┘  │
│  Synthesis (TTS)  │◀─ complete ─│                      │
│                   │   response  │  ┌────────────────┐  │
├──────────────────┤              │  │  Gemini 2.0     │  │
│  VAD (Audio       │              │  │  Flash LLM      │  │
│  Energy Detection)│── barge-in ▶│  │  (Async         │  │
└──────────────────┘              │  │   Streaming)    │  │
                                   │  └────────────────┘  │
                                   └──────────────────────┘
```

### What Lives Where and Why

| Component | Location | Why There |
|-----------|----------|-----------|
| **STT** | Browser | Web Speech API is free, runs locally, zero network latency |
| **TTS** | Browser | Speech Synthesis is free, instant cancel enables barge-in |
| **VAD** | Browser | Audio energy analysis must be real-time, can't afford network round-trip |
| **LLM** | Backend | Gemini API key must stay server-side (security) |
| **RAG** | Backend | Embeddings + vector search happen near the LLM for efficiency |
| **WebSocket** | Both | Enables streaming — tokens flow as they're generated, not all at once |

---

## 7 Core Concepts Taught

| # | Concept | Where in Code | Why It Matters |
|---|---------|---------------|----------------|
| 1 | **WebRTC Mic Capture** | `App.jsx` → `startVAD()` | How browsers access the microphone via `getUserMedia` |
| 2 | **Speech-to-Text** | `App.jsx` → `useSpeechRecognition` | Real-time voice-to-text with interim results |
| 3 | **LLM Streaming** | `main.py` → `websocket_chat()` | Async token-by-token response for low perceived latency |
| 4 | **Text-to-Speech** | `App.jsx` → `useSpeechSynthesis` | Making the AI speak aloud with preloaded voice selection |
| 5 | **Barge-in** | `App.jsx` → `handleBargeIn()` | Interrupting AI mid-sentence (< 200ms) |
| 6 | **Context Window** | `App.jsx` → `conversationHistory` | Full history sent every LLM call (stateless API) |
| 7 | **RAG** | `rag.py` → `RAGEngine` | Injecting domain knowledge into LLM prompts via vector search |

---

## Setup

### 1. Get a Free Gemini API Key

Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and create a free API key.

### 2. Backend

```bash
cd backend
pip install -r requirements.txt

# Create .env file with your API key
cp .env.example .env
# Edit .env and paste your Gemini API key

python main.py
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Open in Browser

Go to [http://localhost:5173](http://localhost:5173) in **Chrome** (best Web Speech API support).

Click the mic button and start talking!

---

## File Structure

```
miniVoxSetu/
├── backend/
│   ├── main.py              # FastAPI server — WebSocket, Gemini async streaming, RAG injection
│   ├── rag.py               # RAG engine — Gemini embeddings + NumPy cosine similarity vector store
│   ├── requirements.txt     # Python dependencies
│   ├── .env.example         # Template for API key
│   └── .env                 # Your actual API key (git-ignored)
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Entire voice pipeline in one file (883 lines, heavily commented)
│   │   ├── index.css        # Dark-theme design system with CSS custom properties
│   │   └── main.jsx         # React entry point
│   ├── index.html           # HTML entry with Google Fonts (Inter + JetBrains Mono)
│   ├── package.json         # NPM dependencies (React 19 + Vite 6)
│   └── vite.config.js       # Vite config with WebSocket proxy to backend
├── WALKTHROUGH.md           # Detailed architectural walkthrough (read this!)
└── README.md                # This file
```

---

## Tech Stack (All Free)

| Component | Technology | Cost |
|-----------|-----------|------|
| STT | Browser Web Speech API | Free |
| LLM | Google Gemini 2.0 Flash (async streaming) | Free tier |
| TTS | Browser Web Speech Synthesis (with voice preloading) | Free |
| Embeddings | Gemini `text-embedding-004` (768-dim) | Free tier |
| Vector DB | NumPy cosine similarity (mirrors ChromaDB interface) | Free |
| VAD | Audio energy detection with duplicate-prevention guards | Free |
| Transport | WebSocket (protocol-aware URL via Vite proxy) | Free |
| Backend | Python FastAPI (fully async) | Free |
| Frontend | React 19 + Vite 6 | Free |

---

## Edge Cases Handled

The codebase handles these edge cases that are commonly missed in voice AI projects:

| Edge Case | How It's Handled |
|-----------|-----------------|
| **Rapid mic clicks** | AudioContext creation is guarded — won't create duplicates |
| **Multiple barge-in intervals** | Previous interval is cleared before creating a new one |
| **WebSocket disconnect mid-conversation** | Safety `useEffect` resets state machine to IDLE |
| **TTS voices not loaded yet** | Voices are preloaded via `voiceschanged` event listener |
| **Server blocking during LLM streaming** | Uses `send_message_async` + `async for` (non-blocking) |
| **Component unmount** | VAD resources (MediaStream, AudioContext, intervals) are cleaned up |
| **Speech recognition errors** | `no-speech` and `aborted` are silently ignored (they're normal) |
| **Recognition already running** | `start()` is wrapped in try-catch to handle rapid state transitions |

---

## How It Works (Read the Code!)

Every meaningful block of code has a `WHY` comment explaining the architectural reason it exists. Start reading from:

1. **`App.jsx`** — Follow the voice pipeline: mic → STT → LLM → TTS
2. **`main.py`** — See how WebSocket streaming and RAG injection work
3. **`rag.py`** — Understand embedding, indexing, and cosine similarity retrieval

For a **comprehensive architectural walkthrough**, read [`WALKTHROUGH.md`](WALKTHROUGH.md).

---

## What Production Systems Do Differently

| This Project | Production (VoxSetu-class) |
|-------------|--------------------------|
| Web Speech API (browser STT) | Cloud STT (Deepgram/Google) — higher accuracy |
| Speech Synthesis (browser TTS) | Cloud TTS (ElevenLabs) — natural voices, SSML |
| Audio energy threshold for VAD | ML-based VAD (Silero) — better noise rejection |
| Full history every call | Token counting + truncation at context limit |
| In-memory NumPy vector store | Persistent vector DB (Pinecone/pgvector) with millions of docs |
| Single WebSocket | gRPC streams with connection pooling |
| No auth | JWT tokens, API keys, rate limiting |
| Hardcoded FAQ (8 documents) | Document ingestion pipeline, chunking strategies, metadata filters |

> **Despite these differences, the architectural patterns are identical.** The pipeline flow (STT → RAG → LLM → TTS), the state machine, the streaming approach, the conversation history management, and the barge-in mechanism are the same in this learning project and in production systems. That's the whole point.

---

## License

MIT — Learn, modify, build on top of it.
