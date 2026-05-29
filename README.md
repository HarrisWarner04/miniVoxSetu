# miniVoxSetu

**Production-Grade Voice AI Pipeline MVP**

A fully-featured, server-driven voice AI agent that demonstrates enterprise voice architectures — streaming audio forking, multi-modal intelligence fusion, sub-200ms barge-in, speculative RAG, and automated post-call QA — built with React + FastAPI over a single WebSocket.

> **Who is this for?**
> Engineers, solution architects, and tech leads who want to understand how production voice AI systems (like those built on FreeSWITCH, Genesys, or Amazon Connect) actually work — without the vendor lock-in. Every architectural decision is documented in code comments and in this README.

---

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [System Architecture Diagram](#system-architecture-diagram)
- [The Three Intelligence Layers](#the-three-intelligence-layers)
  - [Layer 1 — Main Generative Pipeline](#layer-1--main-generative-pipeline-mainpy)
  - [Layer 2 — Semantic Intelligence](#layer-2--semantic-intelligence-semanticpy)
  - [Layer 3 — Acoustic Intelligence](#layer-3--acoustic-intelligence-acousticpy)
- [Audio Forking Architecture](#audio-forking-architecture)
- [Data Flow — End to End](#data-flow--end-to-end)
- [WebSocket Protocol Reference](#websocket-protocol-reference)
- [Barge-In System (B1–B7)](#barge-in-system-b1b7)
- [Speculative RAG Pipeline](#speculative-rag-pipeline)
- [RAG Engine Deep Dive](#rag-engine-deep-dive)
- [PII Redaction Layer](#pii-redaction-layer)
- [TTS Cascaded Streaming](#tts-cascaded-streaming)
- [Post-Call QA Report Generation](#post-call-qa-report-generation)
- [Combined Risk Signal & Escalation](#combined-risk-signal--escalation)
- [Frontend Architecture](#frontend-architecture)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Setup & Running](#setup--running)
- [Environment Variables](#environment-variables)
- [Latency Budget](#latency-budget)
- [Enterprise Edge Cases Handled](#enterprise-edge-cases-handled)
- [Production Upgrade Path](#production-upgrade-path)
- [FAQ](#faq)

---

## High-Level Architecture

miniVoxSetu implements a **Server-Driven Intelligence** pattern. The browser is a thin audio transport layer — all intelligence (STT, LLM, TTS, NLU, emotion detection) runs on the backend. This mirrors how production contact center platforms work, where the telephony edge (browser/SBC/FreeSWITCH) captures audio and the intelligence stack runs server-side.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          BROWSER (Thin Client)                        │
│                                                                       │
│   getUserMedia ──→ AudioWorklet (pcm-processor.js)                    │
│                         │                                              │
│                         ├──→ Int16 PCM (100ms chunks) ──→ [Binary WS] │
│                         ├──→ Float32 PCM (1.5s chunks) ──→ [JSON WS]  │
│                         └──→ Barge-in VAD (per 128-sample frame)       │
│                                                                       │
│   Audio Playback ◀── base64 MP3 ◀── [JSON WS]                        │
│   Live Dashboard ◀── JSON state  ◀── [JSON WS]                        │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │ Single WebSocket (ws://localhost:8000/ws/chat)
                                 │ Binary frames ↑ + JSON text frames ↕
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI + asyncio)                       │
│                                                                       │
│   ┌──────────────┐   ┌──────────────┐   ┌───────────────────────┐     │
│   │ STT Queue    │   │ Acoustic Q.  │   │ Pipeline Orchestrator │     │
│   │ (asyncio.Q)  │   │ (asyncio.Q)  │   │ (main.py)             │     │
│   └──────┬───────┘   └──────┬───────┘   └───────────┬───────────┘     │
│          │                  │                       │                  │
│          ▼                  ▼                       ▼                  │
│   ┌──────────────┐   ┌──────────────┐   ┌───────────────────────┐     │
│   │ Deepgram     │   │ HuBERT       │   │ PII → RAG → Groq LLM │     │
│   │ Nova-2 STT   │   │ + librosa    │   │ → Deepgram TTS        │     │
│   │ (WebSocket)  │   │ (ThreadPool) │   │ (Streaming)           │     │
│   └──────┬───────┘   └──────┬───────┘   └───────────┬───────────┘     │
│          │                  │                       │                  │
│          ▼                  ▼                       ▼                  │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │           Semantic Layer (Gemini — parallel, async)          │     │
│   │   Intent · Sentiment · Urgency · Compliance · Escalation    │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                              │                                        │
│                              ▼                                        │
│                    ┌─────────────────────┐                            │
│                    │ Combined Risk Signal │                            │
│                    │ + Post-Call Report   │                            │
│                    └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## System Architecture Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │              BROWSER (React + Vite)             │
                    │                                                 │
                    │  ┌────────────┐    ┌──────────────────────────┐ │
                    │  │ getUserMedia│───→│  AudioWorklet             │ │
                    │  │ (48kHz,    │    │  pcm-processor.js         │ │
                    │  │  mono)     │    │                            │ │
                    │  └────────────┘    │  ┌─────────────────────┐  │ │
                    │                    │  │ STT Path            │  │ │
                    │                    │  │ Float32→Int16       │  │ │
                    │                    │  │ 100ms chunks        │──┼─┼──→ Binary WS Frame
                    │                    │  └─────────────────────┘  │ │
                    │                    │  ┌─────────────────────┐  │ │
                    │                    │  │ Acoustic Path       │  │ │
                    │                    │  │ Float32 raw PCM     │  │ │
                    │                    │  │ 1.5s chunks         │──┼─┼──→ JSON WS (base64)
                    │                    │  └─────────────────────┘  │ │
                    │                    │  ┌─────────────────────┐  │ │
                    │                    │  │ B3: VAD Barge-in    │  │ │
                    │                    │  │ RMS per 128 samples │  │ │
                    │                    │  │ ~2.7ms resolution   │──┼─┼──→ JSON WS
                    │                    │  └─────────────────────┘  │ │
                    │                    └──────────────────────────┘ │
                    │                                                 │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │  Audio Playback Engine                   │   │
                    │  │  MP3 Queue → AudioContext.decodeAudioData│   │
                    │  │  Barge-in stops playback instantly       │   │
                    │  └──────────────────────────────────────────┘   │
                    │                                                 │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │  Dashboard Panels (React State)          │   │
                    │  │  • Live Transcript    • Semantic Intel.  │   │
                    │  │  • Acoustic Intel.    • RAG Context      │   │
                    │  │  • Conversation Hist. • Post-Call Report │   │
                    │  └──────────────────────────────────────────┘   │
                    └───────────────────┬─────────────────────────────┘
                                        │
                          Single WebSocket Connection
                        ws://localhost:8000/ws/chat
                                        │
                    ┌───────────────────┴─────────────────────────────┐
                    │            BACKEND (FastAPI + uvicorn)          │
                    │                                                 │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │  WebSocket Handler (main.py:132)         │   │
                    │  │  • Receives binary → stt_queue           │   │
                    │  │  • Receives JSON → route by msg_type     │   │
                    │  │  • Sends JSON → transcript/audio/state   │   │
                    │  └────────────────┬─────────────────────────┘   │
                    │                   │                              │
                    │    ┌──────────────┼───────────────┐              │
                    │    ▼              ▼               ▼              │
                    │  ┌─────────┐  ┌─────────┐  ┌──────────────┐     │
                    │  │stt_queue│  │acoustic │  │Pipeline Orch.│     │
                    │  │consumer │  │_queue   │  │run_pipeline()│     │
                    │  │(async)  │  │drain    │  │              │     │
                    │  └────┬────┘  └─────────┘  └──────┬───────┘     │
                    │       │                          │               │
                    │       ▼                          │               │
                    │  ┌──────────────┐               │               │
                    │  │ Deepgram STT │               │               │
                    │  │ (stt.py)     │               │               │
                    │  │ Nova-2 model │               │               │
                    │  │ Streaming WS │               │               │
                    │  │              │               │               │
                    │  │ Callbacks:   │               │               │
                    │  │ on_transcript│───→ UI        │               │
                    │  │ on_confident │───→ Spec. RAG │               │
                    │  │ on_utt_end  │───→ Pipeline ──┘               │
                    │  └──────────────┘                                │
                    │                                                  │
                    │  Pipeline Steps (sequential):                    │
                    │  ┌─────────────────────────────────────────────┐ │
                    │  │  1. PII Redaction (pii.py)                  │ │
                    │  │     Regex-based: Aadhaar, PAN, Card, Phone  │ │
                    │  │                                             │ │
                    │  │  2. RAG Retrieval (rag.py)                  │ │
                    │  │     Local embeddings: all-MiniLM-L6-v2      │ │
                    │  │     SimpleVectorStore (cosine similarity)    │ │
                    │  │     Speculative cache check (cos_sim > 0.85)│ │
                    │  │                                             │ │
                    │  │  3. Context Injection                       │ │
                    │  │     System prompt + RAG chunks              │ │
                    │  │     + Semantic context (prev turn)          │ │
                    │  │     + Acoustic context (prev turn)          │ │
                    │  │                                             │ │
                    │  │  4. LLM Streaming (Groq LLaMA 3.3 70B)     │ │
                    │  │     SSE streaming, ~100ms TTFT              │ │
                    │  │                                             │ │
                    │  │  5. Cascaded TTS (Deepgram Aura)            │ │
                    │  │     Sentence boundary detection → TTS       │ │
                    │  │     Persistent WebSocket, HTTP fallback     │ │
                    │  └─────────────────────────────────────────────┘ │
                    │                                                  │
                    │  Parallel Layers (fire-and-forget):               │
                    │  ┌──────────────────┐  ┌──────────────────────┐  │
                    │  │ Semantic Layer   │  │ Acoustic Layer       │  │
                    │  │ (semantic.py)    │  │ (acoustic.py)        │  │
                    │  │ Gemini 2.5 Flash │  │ HuBERT + librosa     │  │
                    │  │                  │  │ ThreadPoolExecutor    │  │
                    │  │ Outputs:         │  │                      │  │
                    │  │ • intent         │  │ Path A: Signal Proc. │  │
                    │  │ • sentiment      │  │  pitch, RMS, ZCR,    │  │
                    │  │ • urgency        │  │  spectral centroid   │  │
                    │  │ • compliance     │  │                      │  │
                    │  │ • escalation     │  │ Path B: ML Inference │  │
                    │  │ • tone           │  │  HuBERT emotion      │  │
                    │  │                  │  │  (IEMOCAP labels)    │  │
                    │  │                  │  │                      │  │
                    │  │                  │  │ Fusion Layer:        │  │
                    │  │                  │  │  50% physics +       │  │
                    │  │                  │  │  50% model           │  │
                    │  │                  │  │  + 4 physics overrides│  │
                    │  └────────┬─────────┘  └───────────┬──────────┘  │
                    │           │                        │              │
                    │           └────────┬───────────────┘              │
                    │                    ▼                              │
                    │           ┌─────────────────┐                    │
                    │           │ Combined Risk    │                    │
                    │           │ Signal Engine    │                    │
                    │           │                  │                    │
                    │           │ IF stress > 0.6  │                    │
                    │           │ AND sentiment    │                    │
                    │           │     < -0.3       │                    │
                    │           │ THEN escalate    │                    │
                    │           └─────────────────┘                    │
                    └──────────────────────────────────────────────────┘
```

---

## The Three Intelligence Layers

miniVoxSetu implements a **tri-layer intelligence architecture**. Each layer runs independently and their outputs are fused for decision-making.

### Layer 1 — Main Generative Pipeline (`main.py`)

The core conversational loop. This is the **latency-critical path** — every millisecond matters here because the user is waiting for a response.

| Step | Module | What Happens | Latency |
|------|--------|-------------|---------|
| 1 | `pii.py` | Regex-based PII redaction (Aadhaar, PAN, card, phone, email, IFSC, account) | <1ms |
| 2 | `rag.py` | Embed user query → cosine similarity search → top-2 FAQ chunks (or use speculative cache) | ~5ms (local CPU) |
| 3 | `main.py` | Build enhanced system prompt: base + RAG chunks + semantic context (N-1) + acoustic context (N-1) | <1ms |
| 4 | Groq API | Stream LLM response from LLaMA 3.3 70B via SSE | ~100ms TTFT |
| 5 | `tts.py` | Sentence boundary detection → Deepgram Aura TTS (persistent WebSocket) → base64 MP3 to browser | ~150ms per sentence |

**Why Groq for LLM?** Groq's LPU hardware delivers ~100ms TTFT vs Gemini's ~400ms. Since the user is waiting on a phone call, TTFT directly impacts perceived responsiveness. Gemini is still used for non-latency-critical tasks (semantic analysis, report generation).

**Cascaded TTS**: The LLM streams tokens. As soon as a sentence boundary is detected (`.`, `!`, `?` — with abbreviation and decimal guards), that sentence is sent to TTS *while the LLM is still generating the next sentence*. This overlaps LLM generation with TTS synthesis, saving ~150ms per sentence.

### Layer 2 — Semantic Intelligence (`semantic.py`)

Runs **in parallel** with the main pipeline (not blocking it). Analyzes the user's complete utterance using Gemini 2.5 Flash with structured JSON output.

**Output Schema:**
```json
{
  "intent": "balance_inquiry | loan_query | card_block | complaint | ...",
  "sentiment": -1.0 to 1.0,
  "urgency_level": "low | medium | high | critical",
  "compliance_flag": true/false,
  "escalation_recommended": true/false,
  "one_line_summary": "Customer is asking about...",
  "recommended_tone": "empathetic | professional | reassuring | apologetic | cheerful"
}
```

**One-Turn-Behind Injection**: Semantic results from turn N are injected into the system prompt for turn N+1. This allows the agent to adapt its tone based on the customer's emotional state without blocking the current turn.

### Layer 3 — Acoustic Intelligence (`acoustic.py`)

A **dual-path audio processor** that runs on a `ThreadPoolExecutor` to avoid blocking the asyncio event loop.

#### Path A — Signal Processing (librosa)
Extracts physics-based features from raw PCM audio:

| Feature | What It Measures | Why It Matters |
|---------|-----------------|----------------|
| **F0 Pitch** (pYIN algorithm) | Fundamental frequency (60–400 Hz) | Elevated pitch (>250 Hz) indicates frustration or agitation |
| **RMS Energy** | Volume in dB (log scale) | Shouting (>-22 dB) or whispering (<-35 dB) are stress signals |
| **ZCR** | Zero-crossing rate | Harsh/tense voice has high ZCR (>0.15) |
| **Spectral Centroid** | Brightness/timbre | High centroid (>3500 Hz) indicates sharp, tense speech |
| **Energy Variance** | Volume consistency | Erratic volume suggests emotional instability |

#### Path B — ML Inference (HuBERT)
Uses the `superb/hubert-base-superb-er` model fine-tuned on the IEMOCAP dataset for emotion recognition:

| Label | Description |
|-------|-------------|
| `neutral` | Calm, informational speech |
| `happy` | Positive, satisfied |
| `angry` | Frustrated, aggressive |
| `sad` | Disappointed, resigned |

#### Fusion Layer
Physics and ML outputs are combined with a **50/50 weighted fusion**:

```
stress_score = (physics_stress × 0.5) + (model_stress × 0.5)

where:
  physics_stress = (normalized_pitch × 0.4) + (normalized_volume × 0.4) + (zcr_stress × 0.2)
  model_stress   = (P(angry) × 1.0) + (P(sad) × 0.5)
```

**Why 50/50?** HuBERT on short, noisy browser audio clips tends to bias toward `neutral`. Physics features provide a reliable secondary signal that catches cases the ML model misses.

#### Physics Override Rules
When the ML model outputs `neutral` but physics disagree, the system overrides:

| Rule | Condition | Override To |
|------|-----------|-------------|
| **Shouting** | `neutral` + RMS > -22 dB | `agitated`, stress ≥ 55% |
| **High Pitch** | `neutral` + F0 > 250 Hz + RMS > -28 dB | `agitated`, stress ≥ 45% |
| **Harsh Voice** | `neutral` + spectral centroid > 3500 Hz + RMS > -30 dB | `agitated`, stress ≥ 40% |
| **Whisper-Angry** | `angry` + RMS < -35 dB | `agitated_whisper` |

#### Simulation Mode
If PyTorch, librosa, or transformers fail to import (common on Windows without CUDA), the acoustic layer falls back to **simulation mode** — generating realistic mock values so the entire pipeline continues to work without crashing.

---

## Audio Forking Architecture

In production telephony (e.g., FreeSWITCH), RTP audio is "forked" — duplicated into multiple WebSocket streams so that STT and other processors receive independent copies of the same audio. miniVoxSetu emulates this pattern using the browser's `AudioWorklet` API.

```
                getUserMedia (48kHz, mono)
                        │
                        ▼
              ┌────────────────────────┐
              │  AudioWorklet          │
              │  pcm-processor.js      │
              │  (dedicated audio      │
              │   rendering thread)    │
              │                        │
              │  ┌──────────────────┐  │
              │  │ STT Fork         │  │      Binary WS Frame
              │  │ Every 100ms      │──┼────→ (raw Int16 PCM)
              │  │ Float32 → Int16  │  │      → stt_queue
              │  │ (linear16)       │  │      → Deepgram Nova-2
              │  └──────────────────┘  │
              │                        │
              │  ┌──────────────────┐  │      JSON WS Frame
              │  │ Acoustic Fork    │──┼────→ {type: "acoustic_pcm",
              │  │ Every 1.5s       │  │       data: base64(Float32),
              │  │ Float32 raw PCM  │  │       sample_rate: 48000}
              │  └──────────────────┘  │      → HuBERT + librosa
              │                        │
              │  ┌──────────────────┐  │
              │  │ B3: VAD Engine   │──┼────→ {type: "barge_in"}
              │  │ Per 128 samples  │  │      (only during SPEAKING)
              │  │ ~2.7ms frames    │  │
              │  └──────────────────┘  │
              └────────────────────────┘
```

**Why AudioWorklet instead of MediaRecorder?**
- `MediaRecorder` outputs WebM/Opus containers, which Deepgram's streaming API can struggle with
- `AudioWorklet` runs on a dedicated thread with real-time priority, providing precise sample-level control
- We can output both Int16 (for STT) and Float32 (for acoustic analysis) from the same source without re-encoding

---

## Data Flow — End to End

Here is the complete lifecycle of a single conversational turn:

```
Time ──────────────────────────────────────────────────────────────────→

User speaks: "What is my account balance?"

 ┌─ Browser ─────────────────────────────────────────────────────────┐
 │                                                                   │
 │  1. AudioWorklet captures 48kHz mono PCM                          │
 │  2. Every 100ms: Int16 PCM → Binary WS frame → Backend           │
 │  3. Every 1.5s: Float32 PCM → JSON WS (base64) → Backend         │
 │                                                                   │
 └───────────────────────────────────────────────────────────────────┘
                              │
 ┌─ Backend ─────────────────┼───────────────────────────────────────┐
 │                           ▼                                       │
 │  4. stt_consumer pulls from stt_queue → Deepgram Nova-2           │
 │  5. Deepgram returns interim transcripts (pushed to UI)           │
 │                                                                   │
 │  --- At confidence ≥ 0.7: ---                                     │
 │  6. on_confident_interim fires → Speculative RAG caches results   │
 │                                                                   │
 │  --- At UtteranceEnd (1s silence) or speech_final: ---            │
 │  7. on_utterance_end fires with full text                         │
 │                                                                   │
 │  ┌─── PARALLEL ─────────────────────────────────────────────────┐ │
 │  │                                                               │ │
 │  │  MAIN PIPELINE (latency-critical):                            │ │
 │  │  8.  PII redaction → "What is my account balance?"            │ │
 │  │  9.  RAG retrieval (check speculative cache → cos_sim > 0.85) │ │
 │  │  10. Build system prompt + RAG + semantic(N-1) + acoustic(N-1)│ │
 │  │  11. Stream Groq LLaMA 3.3 70B (SSE)                         │ │
 │  │  12. Sentence boundary detected → TTS synthesis               │ │
 │  │  13. MP3 audio → base64 → JSON WS → Browser                  │ │
 │  │                                                               │ │
 │  │  SEMANTIC ANALYSIS (non-blocking):                            │ │
 │  │  14. PII-redacted text → Gemini 2.5 Flash                    │ │
 │  │  15. Returns structured JSON → stored for next turn           │ │
 │  │  16. Pushed to frontend dashboard                             │ │
 │  │                                                               │ │
 │  └───────────────────────────────────────────────────────────────┘ │
 │                                                                   │
 │  ACOUSTIC ANALYSIS (ongoing, every 1.5s):                         │
 │  17. Float32 PCM → librosa (pitch, RMS, ZCR, spectral centroid)   │
 │  18. Float32 PCM → HuBERT (emotion classification)               │
 │  19. Fusion → stress_score → pushed to frontend dashboard         │
 │                                                                   │
 │  COMBINED RISK SIGNAL (after semantic + acoustic complete):       │
 │  20. IF stress > 0.6 AND sentiment < -0.3 → escalation_recommended│
 │                                                                   │
 └───────────────────────────────────────────────────────────────────┘
                              │
 ┌─ Browser ─────────────────┼───────────────────────────────────────┐
 │                           ▼                                       │
 │  21. Audio queued → AudioContext.decodeAudioData → playback       │
 │  22. Dashboard updates: semantic, acoustic, RAG, transcript       │
 │  23. Conversation history appended                                │
 │                                                                   │
 └───────────────────────────────────────────────────────────────────┘
```

---

## WebSocket Protocol Reference

A single WebSocket connection (`ws://localhost:8000/ws/chat`) carries all communication. The protocol multiplexes binary audio and JSON control messages.

### Client → Server

| Frame Type | Message Type | Payload | Description |
|-----------|-------------|---------|-------------|
| **Binary** | — | Raw Int16 PCM bytes | Audio for STT (100ms chunks from AudioWorklet) |
| Text/JSON | `acoustic_pcm` | `{data: base64, sample_rate: int}` | Float32 PCM for acoustic analysis (1.5s chunks) |
| Text/JSON | `barge_in` | `{}` | User interrupted — cancel all in-flight tasks |
| Text/JSON | `text_input` | `{text: str, history: [...]}` | Text fallback input (bypasses STT) |
| Text/JSON | `end_call` | `{}` | Session end — triggers post-call report generation |

### Server → Client

| Message Type | Payload | Description |
|-------------|---------|-------------|
| `transcript` | `{text, is_final, confidence}` | Live STT from Deepgram (interim + final) |
| `state` | `{state: THINKING\|SPEAKING\|LISTENING}` | Agent state machine transitions |
| `rag_context` | `{chunks: [...], query: str}` | RAG retrieval results for dashboard display |
| `chunk` | `{text: str}` | Streaming LLM token (Groq SSE → client) |
| `audio` | `{data: base64, format: "mp3"}` | TTS audio chunk (Deepgram Aura) |
| `done` | `{full_text: str}` | Complete LLM response (for conversation history) |
| `semantic` | `{data: {...}, turn_id, latency_ms}` | Semantic analysis results |
| `acoustic` | `{data: {...}, latency_ms}` | Acoustic analysis results |
| `report` | `{data: str, turns: int}` | Markdown post-call QA report |
| `error` | `{message: str}` | Error notification |

---

## Barge-In System (B1–B7)

Barge-in (user interruption) is the hardest problem in voice AI. miniVoxSetu implements a **7-layer barge-in system**:

| Layer | Name | What It Does |
|-------|------|-------------|
| **B1** | Dynamic Threshold Calibration | On mic start, 500ms of ambient noise is sampled. VAD threshold = max(0.01, baseline × 2.5). Prevents false triggers in noisy environments. |
| **B2** | TTS Suppression Window | After each TTS audio chunk starts playing, VAD is suppressed for 200ms. This gives the browser's echo cancellation (AEC) time to settle and prevents the agent's own voice from triggering barge-in. |
| **B3** | AudioWorklet VAD | Barge-in detection runs in the `pcm-processor.js` AudioWorklet at ~2.7ms resolution (128 samples at 48kHz), not in a 100ms `setInterval`. The worklet knows the agent's speaking state and only fires during `SPEAKING`. |
| **B4** | Server-Side Gate | A `barged_in` boolean flag on the server. Once set, all in-flight TTS audio sends are blocked even if the async pipeline hasn't been cancelled yet. |
| **B5** | Generation Counter | A monotonic `pipeline_generation` counter. Incremented on every barge-in. Any pipeline task launched with an old generation number is silently discarded at multiple checkpoints (before PII, before RAG, before LLM, before TTS send). |
| **B6** | Backchannel Detection | Short utterances like "uh-huh", "okay", "hmm", "accha" are detected as backchannel (not real interruptions). The agent pauses 300ms for natural rhythm but does NOT kill the pipeline. Supports Hindi backchannels. |
| **B7** | Client-Side Audio Queue Drain | On barge-in, the browser immediately: (1) stops the current AudioBufferSourceNode, (2) clears the audio queue, (3) sets a `bargedInRef` flag that prevents any late-arriving audio from being enqueued or played. |

### Barge-In Sequence Diagram

```
User starts speaking over the agent:

Browser (AudioWorklet)                    Backend (FastAPI)
        │                                       │
        │ ← Agent is SPEAKING →                 │
        │                                       │
  B3: VAD detects energy > threshold            │
  B2: Check suppression window (OK)             │
  B3: Post "barge_in_detected" to main thread   │
        │                                       │
  B7: Stop AudioBufferSourceNode                │
  B7: Clear audio queue                         │
  B7: Set bargedInRef = true                    │
        │                                       │
        │──── {type: "barge_in"} ──────────────→│
        │                                       │
        │                              B6: Check backchannel
        │                                  "uh-huh" → soft pause only
        │                                  real words → full barge-in
        │                                       │
        │                              B4: barged_in = true
        │                              B5: pipeline_generation++
        │                              Cancel: pipeline task
        │                              Cancel: semantic task
        │                              Cancel: acoustic task
        │                              Clear: TTS buffer (WebSocket)
        │                              Clear: acoustic chunks
        │                                       │
        │◀─── {type: "state", state: "LISTENING"} │
        │                                       │
  New pipeline starts with                      │
  incremented generation number                 │
```

---

## Speculative RAG Pipeline

Standard RAG adds ~5ms to the critical path (embed query + search). Speculative RAG eliminates this by running the RAG query **before the user finishes speaking**.

```
Time ──────────────────────────────────────────────────────────→

User speaking: "What are your fixed deposit rates?"

  Deepgram interim (conf=0.4): "What are"        → No action (< 0.7)
  Deepgram interim (conf=0.7): "What are your"    → Skip (< 3 words)
  Deepgram interim (conf=0.8): "What are your fixed" → SPECULATIVE RAG FIRES
                                                         │
                                                         ├→ Embed "What are your fixed"
                                                         ├→ Search vector store → 2 chunks
                                                         └→ Cache: {query, chunks, embedding}
  Deepgram final (conf=0.95): "What are your fixed deposit rates?"
                                                         │
  Pipeline starts:                                       │
    PII redaction → "What are your fixed deposit rates?" │
    RAG step:                                            │
      1. Embed final text                                │
      2. Compare to cached embedding (cosine similarity) │
      3. cos_sim = 0.92 > 0.85 → USE CACHED RESULTS     │
      4. Skip full vector search                         │
                                                         │
  Result: RAG latency reduced from ~5ms to ~1ms (embedding comparison only)
```

**Cache invalidation**: If `cos_sim ≤ 0.85` (the user changed topics mid-sentence), the speculative cache is discarded and a fresh RAG query runs.

---

## RAG Engine Deep Dive

### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Runs locally** on CPU — no API calls needed
- **Embedding latency**: ~5ms per query (vs ~150ms for Gemini Embedding API)
- **Why local?** Eliminates network round-trip. In a voice pipeline, every 150ms matters.

### Vector Store
`SimpleVectorStore` — a minimal in-memory implementation that mirrors ChromaDB's interface:

```python
class SimpleVectorStore:
    def add(ids, embeddings, documents)     # Index phase (startup)
    def query(query_embedding, n_results)   # Retrieval phase (per query)
    def count()                             # Number of indexed documents
```

**Search algorithm**: Brute-force cosine similarity (O(n) where n = number of documents). In production with millions of docs, this would use HNSW approximate nearest neighbor (as ChromaDB uses internally via hnswlib).

### Knowledge Base
21 hardcoded FAQ documents covering NeoBank's product catalog:
- Account types (Basic, Premium, Current)
- Interest rates, FD rates
- UPI, NEFT, IMPS, RTGS limits
- Debit card details, lost card procedure
- Personal loans, EMI queries
- KYC, complaints, customer support
- International transactions, overdraft, cheque books
- Tax/TDS rules

**Production upgrade**: Replace hardcoded FAQ with a CMS-backed document store. Replace `SimpleVectorStore` with ChromaDB/Pinecone/pgvector. Add document chunking for longer documents.

---

## PII Redaction Layer

All user speech is scrubbed of personally identifiable information **before** being sent to any LLM API (Groq, Gemini). This is a compliance requirement for financial services.

### Patterns Detected (in priority order)

| Pattern | Regex | Replacement Token | Example |
|---------|-------|--------------------|---------|
| Credit/Debit Card | 16 digits (grouped by 4) | `[CARD_NO]` | `4111 1111 1111 1111` |
| Aadhaar | 12 digits (grouped as 4-4-4) | `[AADHAAR]` | `1234 5678 9012` |
| PAN Card | `ABCDE1234F` format | `[PAN]` | `ABCDE1234F` |
| Indian Mobile | `+91`/`0` prefix + 10 digits starting 6-9 | `[PHONE]` | `+91 98765 43210` |
| Email | Standard email regex | `[EMAIL]` | `user@neobank.in` |
| IFSC Code | 4 letters + 0 + 6 alphanumeric | `[IFSC]` | `HDFC0001234` |
| Bank Account | 8–18 digits (generic, last priority) | `[ACCOUNT_NO]` | `00123456789` |

**Pattern order matters**: Credit card (16 digits) is matched before Aadhaar (12 digits) to prevent partial matches.

**Production upgrade path**: Replace regex patterns with [Microsoft Presidio](https://github.com/microsoft/presidio) for ML-based PII detection that handles natural language variations.

---

## TTS Cascaded Streaming

miniVoxSetu uses **Deepgram Aura TTS** via a persistent WebSocket connection (one per session), with HTTP fallback.

### WebSocket Protocol (Deepgram Aura)

```
Client → Server:
  {"type": "Speak", "text": "Hello, how can I help?"}  → Start synthesis
  {"type": "Flush"}                                     → Force buffered audio out
  {"type": "Clear"}                                     → Discard buffer (barge-in)
  {"type": "Close"}                                     → Graceful disconnect

Server → Client:
  Binary frames           → MP3 audio chunks
  {"type": "Flushed"}     → All audio for current Speak sent (sentinel)
  {"type": "Warning"}     → Non-fatal warning
  {"type": "Error"}       → Error
```

### Sentence Boundary Detection (`tts.py`)

The `detect_sentence_boundary()` function splits streaming LLM text at sentence boundaries, with smart guards:

- **Abbreviation guard**: Won't split on `Mr.`, `Mrs.`, `Dr.`, `Rs.`, `No.`, `St.`, `etc.`
- **Decimal guard**: Won't split on `₹1.5` or `3.14`
- **Minimum length**: Requires at least 3 characters before a boundary (so `OK.` still triggers)

This enables **cascaded streaming**: Sentence 1 is sent to TTS while the LLM is still generating Sentence 2.

---

## Post-Call QA Report Generation

When the user clicks "End Call" or disconnects:

1. **Frontend** sends `{type: "end_call"}`
2. **Backend** compiles the `turn_log` — an array of all turns with:
   - User utterance (PII-redacted)
   - Semantic analysis results
   - Acoustic analysis results (averaged per turn)
   - Combined risk signal
   - Whether the turn was interrupted
3. **Gemini 2.5 Flash** generates a structured Markdown report with:
   - Executive Summary
   - Primary Intent & Resolution
   - Caller Sentiment & Stress progression
   - Action Items / Follow-ups
4. Report is saved to `backend/reports/post_call_report_YYYYMMDD_HHMMSS.md`
5. Report is sent to the frontend via WebSocket for modal display

---

## Combined Risk Signal & Escalation

The system fuses acoustic and semantic signals to make escalation decisions:

```
IF (avg_acoustic_stress_this_turn > 0.6)
   AND (semantic_sentiment < -0.3)
THEN
   escalation_recommended = true   // Override semantic layer
   combined_risk = true            // Flag in turn log
```

This **cross-modal validation** prevents false positives:
- A user speaking loudly in a noisy environment (high acoustic stress) but asking a neutral question (positive sentiment) → **no escalation**
- A user with elevated pitch and volume (high acoustic stress) AND expressing frustration (negative sentiment) → **escalate**

---

## Frontend Architecture

### Technology
- **React 19** (no router — single-page application)
- **Vite 6** (dev server with WebSocket proxy)
- **Vanilla CSS** (dark theme, Inter + JetBrains Mono fonts)

### State Machine

```
           startRecording()
  IDLE ─────────────────────→ LISTENING
   ▲                             │
   │ stopRecording()             │ on_utterance_end (from STT)
   │                             ▼
   │                          THINKING
   │                             │
   │ done + all audio played     │ LLM starts generating
   │                             ▼
   └──────────────────────── SPEAKING
                                 │
                     handleBargeIn() → back to LISTENING
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `useWebSocket` hook | Custom hook managing the WebSocket connection with auto-reconnect, binary+text frame support |
| AudioWorklet (`pcm-processor.js`) | Runs on dedicated audio thread: STT fork, Acoustic fork, VAD barge-in detection |
| Audio Playback Queue | `audioQueueRef` — FIFO queue of MP3 chunks decoded via `AudioContext.decodeAudioData` |
| Dashboard Panels | Live Transcript, Semantic Intelligence, Acoustic Intelligence, RAG Context, Conversation History, Post-Call Report |

### Dashboard Panels

| Panel | Data Source | Update Frequency |
|-------|------------|-------------------|
| **Live Transcript** | Deepgram STT (interim + final) | Real-time (per word) |
| **Semantic Intelligence** | Gemini 2.5 Flash | Per utterance (~500ms) |
| **Acoustic Intelligence** | HuBERT + librosa | Every 1.5 seconds |
| **RAG Context** | Local vector search | Per utterance |
| **Context Window** | Conversation history | Per completed turn |
| **Post-Call Report** | Gemini 2.5 Flash | On session end |

---

## Project Structure

```
miniVoxSetu/
├── README.md                          # This file
├── .gitignore
│
├── backend/
│   ├── main.py                        # FastAPI app, WebSocket handler, pipeline orchestrator
│   │                                  # (903 lines — the central nervous system)
│   ├── stt.py                         # Deepgram Nova-2 streaming STT client
│   │                                  # Hybrid early trigger: speech_final + conf ≥ 0.9
│   ├── tts.py                         # Deepgram Aura TTS (persistent WebSocket + HTTP fallback)
│   │                                  # Sentence boundary detection with abbreviation guards
│   ├── acoustic.py                    # HuBERT + librosa dual-path acoustic analysis
│   │                                  # ThreadPoolExecutor, simulation mode fallback
│   ├── semantic.py                    # Gemini-powered NLU (intent, sentiment, urgency)
│   │                                  # Structured JSON output via response_mime_type
│   ├── rag.py                         # Local RAG: sentence-transformers + SimpleVectorStore
│   │                                  # 21 NeoBank FAQ documents, cosine similarity search
│   ├── pii.py                         # Regex-based PII redaction (7 Indian data patterns)
│   ├── test_deepgram.py               # Deepgram integration test script
│   ├── requirements.txt               # Python dependencies
│   ├── .env                           # API keys (not committed)
│   └── reports/                       # Generated post-call QA reports (not committed)
│
├── frontend/
│   ├── index.html                     # Entry point (Inter + JetBrains Mono fonts)
│   ├── vite.config.js                 # Vite config with /ws proxy to backend
│   ├── package.json                   # React 19, Vite 6
│   ├── public/
│   │   └── pcm-processor.js           # AudioWorklet: triple-output PCM processor
│   │                                  # STT fork (Int16, 100ms) + Acoustic fork (Float32, 1.5s)
│   │                                  # + B3 VAD barge-in detection (2.7ms frames)
│   └── src/
│       ├── main.jsx                   # React entry point
│       ├── App.jsx                    # Main application component (1067 lines)
│       │                              # WebSocket hook, audio recording, playback,
│       │                              # barge-in handling, all dashboard panels
│       └── index.css                  # Full design system (dark theme)
│
└── Documentation/                     # Internal design docs (not committed to repo)
    ├── miniVoxSetu_Architecture_and_Code_Guide.md
    ├── how_the_mvp_actually_works.md
    ├── production_stack_decisions.md
    └── ... (additional deep-dive docs)
```

---

## Technology Stack

### Backend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI 0.115 + uvicorn 0.34 | Async WebSocket server |
| **STT** | Deepgram Nova-2 (streaming WebSocket) | Real-time speech-to-text |
| **LLM (latency-critical)** | Groq LLaMA 3.3 70B (SSE streaming) | Conversational AI (~100ms TTFT) |
| **LLM (background)** | Google Gemini 2.5 Flash | Semantic analysis, report generation |
| **TTS** | Deepgram Aura (persistent WebSocket) | Text-to-speech (MP3 output) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 (local) | RAG query embedding (~5ms) |
| **Vector Store** | SimpleVectorStore (in-memory, cosine similarity) | FAQ retrieval |
| **Acoustic ML** | HuBERT (superb/hubert-base-superb-er) | Emotion recognition |
| **Signal Processing** | librosa 0.10 | Pitch (pYIN), RMS, ZCR, spectral centroid |
| **WebSocket Client** | websockets 15.0 | Deepgram STT + TTS connections |
| **HTTP Client** | httpx 0.28 | Groq API streaming, TTS HTTP fallback |
| **PII** | Python `re` (regex) | 7 Indian data patterns |
| **Config** | python-dotenv | `.env` file loading |
| **Math** | numpy 2.2 | Cosine similarity, audio processing |

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI Framework** | React 19 | Single-page application |
| **Build Tool** | Vite 6 | Dev server with HMR + WS proxy |
| **Audio Capture** | Web Audio API + AudioWorklet | PCM extraction, dual-fork, VAD |
| **Audio Playback** | AudioContext + AudioBufferSourceNode | MP3 decoding and playback |
| **Styling** | Vanilla CSS | Dark theme, Inter font |
| **Typography** | Inter (UI) + JetBrains Mono (code) | Google Fonts |

---

## Setup & Running

### Prerequisites
- **Python 3.10+** with pip
- **Node.js 18+** with npm
- API keys for: Deepgram, Groq, Google Gemini

### 1. Clone the Repository

```bash
git clone https://github.com/HarrisWarner04/miniVoxSetu.git
cd miniVoxSetu
```

### 2. Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Required — Get from https://console.deepgram.com
DEEPGRAM_API_KEY=your_deepgram_api_key

# Required — Get from https://console.groq.com
GROQ_API_KEY=your_groq_api_key

# Required — Get from https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key
```

### 3. Backend Setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

The backend starts on `http://localhost:8000`.

> **Note**: On first run, the HuBERT model (~360MB) and sentence-transformers model (~80MB) will be downloaded. If `torch` or `librosa` fails to install (common on Windows ARM), the acoustic layer automatically falls back to **simulation mode** — the rest of the pipeline works normally.

### 4. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend starts on `http://localhost:5173`.

### 5. Open in Browser

Navigate to [http://localhost:5173](http://localhost:5173). Click the mic button to start a voice session.

> **Important**: Use Chrome or Edge. Safari has limited AudioWorklet support. Firefox works but may have echo cancellation issues.

---

## Environment Variables

| Variable | Required | Used By | Description |
|----------|----------|---------|-------------|
| `DEEPGRAM_API_KEY` | ✅ | `stt.py`, `tts.py` | Deepgram API key for STT (Nova-2) and TTS (Aura) |
| `GROQ_API_KEY` | ✅ | `main.py` | Groq API key for LLaMA 3.3 70B (latency-critical LLM) |
| `GEMINI_API_KEY` | ✅ | `semantic.py`, `main.py` | Google Gemini API key for semantic analysis and post-call reports |

---

## Latency Budget

Target: **< 1 second** from user stops speaking to first audio playback.

| Stage | Target | Actual | Notes |
|-------|--------|--------|-------|
| STT (utterance detection) | < 300ms | ~200ms | Hybrid trigger: `speech_final` + confidence ≥ 0.9 bypasses 1000ms `utterance_end_ms` wait |
| PII Redaction | < 5ms | < 1ms | Compiled regex patterns |
| RAG Retrieval | < 10ms | ~5ms | Local embeddings (no API call). Speculative cache reduces to ~1ms |
| LLM TTFT | < 200ms | ~100ms | Groq LPU hardware (vs ~400ms for Gemini) |
| Sentence Detection | < 1ms | < 1ms | Simple string scanning |
| TTS Synthesis | < 200ms | ~150ms | Persistent WebSocket eliminates connection overhead |
| Audio Delivery | < 50ms | ~30ms | Base64 JSON over existing WebSocket |
| **Total** | **< 800ms** | **~490ms** | Cascaded TTS overlaps with LLM generation |

---

## Enterprise Edge Cases Handled

| Feature | How It's Handled |
|---------|-----------------|
| **Barge-in / Interruption** | 7-layer system (B1–B7): AudioWorklet VAD at 2.7ms resolution, server-side generation counter, backchannel detection (Hindi + English), client-side audio queue drain |
| **Audio Forking** | Emulating FreeSWITCH, the `AudioWorklet` outputs both Int16 PCM (for STT) and Float32 PCM (for acoustic ML) from the same mic source |
| **Non-blocking ML** | HuBERT acoustic model runs in a `ThreadPoolExecutor` (1 worker) so it never blocks FastAPI's asyncio event loop |
| **Safety Fallbacks** | If PyTorch/CUDA/librosa are unavailable, acoustic layer falls back to simulation mode with realistic mock values |
| **Rate Limiting** | Groq free tier (30 req/min) rate limits are caught and produce a polite voice fallback message instead of crashing |
| **STT Reconnection** | Deepgram WebSocket is connected lazily (on first audio) and reconnects automatically if dropped. 8-second keepalive prevents timeout during agent speech |
| **TTS Fallback** | If Deepgram TTS WebSocket fails, falls back to HTTP-based synthesis (higher latency but functional) |
| **Speculative RAG** | RAG query fires at interim confidence ≥ 0.7 and caches results. If final text diverges (cos_sim < 0.85), cache is discarded and fresh query runs |
| **Stale Pipeline Guard** | Pipeline generation counter prevents cancelled pipelines from sending audio after a barge-in |
| **Backchannel Detection** | "uh-huh", "okay", "accha", "haan" etc. trigger a 300ms soft pause, not a full pipeline cancellation |
| **PII Compliance** | All user speech is PII-scrubbed before reaching any LLM API. Findings are logged as token types only (never actual values) |
| **Echo Cancellation** | Browser AEC + 200ms TTS suppression window prevent the agent's own voice from triggering barge-in |
| **Lazy Initialization** | Deepgram STT and TTS WebSocket connections are only opened when first needed (not on page load), preventing timeout errors |
| **Graceful Shutdown** | On disconnect: all asyncio tasks are cancelled, both Deepgram WebSockets are closed, media streams are stopped |

---

## Production Upgrade Path

This MVP is designed to be upgraded incrementally to a production system. Here's the mapping:

| MVP Component | Production Equivalent |
|--------------|----------------------|
| `SimpleVectorStore` (in-memory) | ChromaDB / Pinecone / pgvector |
| Regex PII (`pii.py`) | Microsoft Presidio (ML-based) |
| Hardcoded FAQ docs | CMS / Document Store + chunking pipeline |
| Single `main.py` monolith | Microservices (STT, LLM, TTS, NLU as separate services) |
| Browser audio capture | FreeSWITCH / Asterisk RTP forking |
| Single WebSocket | gRPC bidirectional streaming |
| In-process HuBERT | Dedicated GPU inference server (Triton / TorchServe) |
| Local sentence-transformers | Embedding API with caching layer |
| File-based reports | Database + analytics dashboard |
| Groq (cloud) | Self-hosted vLLM on GPU cluster |
| Hardcoded system prompt | Prompt management system (versioned, A/B tested) |

---

## FAQ

**Q: Why is the main LLM Groq (LLaMA 3.3 70B) instead of Gemini?**
A: Groq's custom LPU hardware delivers ~100ms time-to-first-token, which is critical for voice latency. Gemini's TTFT is ~400ms. For non-latency-critical tasks (semantic analysis, report generation), we still use Gemini.

**Q: Why does the acoustic layer use both librosa AND HuBERT?**
A: HuBERT excels at emotion classification on clean, long audio clips. But on short, noisy browser audio, it biases toward `neutral`. librosa provides deterministic physics-based features (pitch, volume) that catch cases the ML model misses. The 50/50 fusion with 4 physics override rules gives the best combined accuracy.

**Q: Why AudioWorklet instead of MediaRecorder?**
A: MediaRecorder outputs WebM/Opus containers, which require demuxing before Deepgram can process them as raw audio. AudioWorklet gives us direct access to raw PCM samples, allowing us to output both Int16 (for STT) and Float32 (for acoustic analysis) without re-encoding.

**Q: What happens if the user's machine doesn't have a GPU?**
A: HuBERT runs on CPU (slower but functional). If PyTorch itself fails to import, the acoustic layer falls back to **simulation mode** with realistic mock values. The pipeline never crashes.

**Q: Why is Deepgram used for both STT and TTS instead of using ElevenLabs for TTS?**
A: The original design used ElevenLabs for TTS. We migrated to Deepgram Aura because: (1) it supports persistent WebSocket TTS with `Speak`/`Flush`/`Clear` commands, which is ideal for cascaded streaming and barge-in, and (2) using the same provider for STT and TTS simplifies API key management.

**Q: How does the speculative RAG avoid stale results?**
A: When the final transcript arrives, its embedding is compared to the cached speculative embedding using cosine similarity. If `cos_sim > 0.85`, the cache is used. If the user changed topics mid-sentence (low similarity), the cache is discarded and a fresh search runs. The cache is always cleared after use.

**Q: What's the backchannel word list?**
A: English: "uh-huh", "hmm", "yeah", "okay", "sure", "alright", "got it", "go on", etc. Hindi: "haan", "ha", "accha", "theek", "sahi". Any utterance of ≤3 words containing only these words (and no "interrupt signal" words like "stop", "cancel", "balance") is classified as backchannel.

---

<p align="center">
  <sub>Built with ❤️ for learning enterprise voice AI architecture</sub>
</p>
