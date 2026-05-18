# miniVoxSetu — Production Voice AI Pipeline MVP

A fully-featured, production-grade Voice AI agent that demonstrates enterprise voice architectures (like FreeSWITCH forking, multi-modal intelligence, and rapid barge-in) using a React + FastAPI stack.

This project upgraded from a simple browser-based bot to a full Server-Driven Intelligence pipeline.

---

## Architecture Overview

```
Browser (Audio Source)              Backend (FastAPI)
┌──────────────────┐               ┌─────────────────────────────────┐
│                  │── PCM ───────▶│  Acoustic Layer (HuBERT)        │
│  getUserMedia    │               │  extracts Pitch, Emotion, Stress│
│  (AudioWorklet)  │── WebM ──────▶│  STT Layer (Deepgram)           │
│                  │               │                                 │
├──────────────────┤               ├─────────────────────────────────┤
│  VAD Energy      │── barge-in ─▶ │  Main Pipeline:                 │
│  Detection       │               │   1. PII Redaction              │
│                  │               │   2. RAG Context Injection      │
├──────────────────┤               │   3. Gemini LLM Streaming       │
│                  │               │   4. Sentence Boundary Detection│
│  Live Dashboard  │◀─ JSON ───────│   5. ElevenLabs TTS             │
│  (React)         │               │                                 │
├──────────────────┤               ├─────────────────────────────────┤
│                  │               │  Semantic Layer (Parallel):     │
│  Audio Playback  │◀─ MP3 ────────│   Analyzes Intent, Sentiment,   │
│                  │               │   Urgency, Compliance Flags.    │
└──────────────────┘               └─────────────────────────────────┘
```

---

## The Intelligence Layers

This MVP implements three core layers of intelligence:

1. **Main Generative Pipeline (`main.py`)**: Handles the core conversational loop. Audio is transcribed via Deepgram, scrubbed of PII, enriched with RAG knowledge, and streamed to Gemini. Responses are converted to voice via ElevenLabs.
2. **Semantic Layer (`semantic.py`)**: Runs in parallel to the main pipeline. It analyzes the user's utterance for intent and sentiment, and feeds this data back into the LLM context for the *next* turn, allowing the agent to adapt its tone.
3. **Acoustic Layer (`acoustic.py`)**: A dual-path audio processor running on a separate thread. It extracts physical audio features (Pitch, Volume) via `librosa` and ML emotion features via `DistilHuBERT`.

These layers fuse their outputs into a **Combined Risk Signal** and generate an automated Post-Call QA Report when the session ends.

---

## Edge Cases & Enterprise Features Handled

| Feature | How It's Handled |
|---------|-----------------|
| **Barge-in / Interruption** | Browser VAD detects speech, stops TTS audio playback immediately, and sends a WebSocket signal that cancels all in-flight Python `asyncio` tasks. |
| **Audio Forking** | Emulating FreeSWITCH, the browser sends raw PCM via AudioWorklet and encoded WebM via MediaRecorder, effectively "forking" the stream to STT and Acoustic models simultaneously. |
| **Non-blocking ML** | The heavy HuBERT acoustic model runs in a `ThreadPoolExecutor` so it never blocks the FastAPI WebSocket loop. |
| **Safety Fallbacks** | If PyTorch/CUDA are unavailable during a demo, the acoustic layer gracefully falls back to a realistic "Simulation Mode" without crashing. |

---

## Setup & Running

### 1. Environment Variables
Create a `.env` file in the `backend/` directory with your API keys:
```
GEMINI_API_KEY=your_key_here
DEEPGRAM_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
```

### 2. Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```
*(Note: On first run, the HuBERT model weights will be downloaded. If `torch` or `librosa` fails to install on Windows, the app will auto-fallback to simulation mode.)*

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Open in Browser
Go to [http://localhost:5173](http://localhost:5173). Click the mic button to start the session.
