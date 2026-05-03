# miniVoxSetu — Future Upgrades & Feature Roadmap

The current version of the codebase (`v1.0`) is the established **stable benchmark**. It successfully implements a production-grade linear voice AI pipeline:
`Microphone (WebRTC) → STT (Web Speech API) → RAG Context Retrieval → LLM (Gemini Async Streaming) → TTS (Speech Synthesis) + Barge-in (VAD)`.

Any new features, architectural shifts, or UI enhancements built on top of this benchmark will be documented here before implementation.

---

## Proposed Upgrades (Planned for v2.0)

We have mapped out 5 major architectural upgrades to elevate the `miniVoxSetu` benchmark to a true production-grade architecture, directly mirroring how VoxSetu's real pipelines operate. These are being developed on the `feature/interview-upgrades` branch.

### 1. Audio Flows to Backend (Architectural Shift)
Right now STT happens in the browser via Web Speech API and only text reaches the backend. Real systems stream raw audio to the backend where production STT (like Deepgram) runs.
- **Action:** Add a "production mode" where `MediaRecorder` captures audio chunks and streams them via WebSocket to FastAPI.
- **Why:** This is 3-4 meaningful commits and demonstrates an understanding of real-world audio streaming topologies.

### 2. ML-Based VAD (Silero)
Replace the simple audio energy threshold with the `@ricky0123/vad-web` library.
- **Action:** Integrate Silero VAD (running locally in the browser) to fire deterministic `onSpeechStart` and `onSpeechEnd` events.
- **Why:** Fixes the barge-in manual mic button limitation, allows full hands-free operation, and uses exactly what production systems use for noise rejection.

### 3. Fallback Systems (Resilience)
Add retry logic and graceful degradation in the backend.
- **Action:** If Gemini API fails (e.g., HTTP 429), catch the error, retry once, and if it fails, yield a friendly `FALLBACK_RESPONSE` instead of crashing the WebSocket. If RAG retrieval fails, continue generation without context.
- **Why:** Shows production thinking regarding API resilience.

### 4. Persistent Conversation Storage (SQLite)
Frontend-only conversation state is lost on refresh.
- **Action:** Add Python's built-in `sqlite3` to intercept and save each conversation turn (`session_id`, `role`, `content`, `timestamp`) to a local database. Add a `GET /history/{session_id}` endpoint.
- **Why:** Demonstrates understanding of session management and durable state.

### 5. Basic Observability & Metrics
No system is complete without logs and metrics.
- **Action:** Add Python's built-in `logging` module with structured logs at every pipeline stage (WebSocket connect, RAG retrieval time, Gemini first token time). Add a `GET /metrics` endpoint returning average latencies for the last 10 turns.
- **Why:** Proves you understand how to monitor latency (the #1 challenge in Voice AI).

---

## Upgrade Implementation History

*Once proposed upgrades are built, tested, and merged into the main codebase, their details and technical tradeoffs will be moved here for historical tracking.*
