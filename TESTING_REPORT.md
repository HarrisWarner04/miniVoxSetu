# miniVoxSetu — Testing & Validation Report

This document serves as the historical record of the testing phases, edge cases validated, and real-world issues encountered while finalizing the `v1.0` benchmark of the miniVoxSetu Voice AI agent.

---

## 1. Happy Path Testing (Core Pipeline)

We validated the core linear pipeline to ensure the foundational architecture was sound:

- **Speech-to-Text (STT):** Verified that the Web Speech API correctly captures microphone audio and transcribes it in real-time. Both interim (tentative) and final transcripts successfully update the React UI.
- **WebSocket Streaming:** Confirmed the Python FastAPI backend successfully receives the user query and streams the LLM tokens back over a bidirectional WebSocket connection. This ensures TTS can begin before the full text is generated.
- **Text-to-Speech (TTS):** Validated that the browser successfully reads the final AI response aloud using `SpeechSynthesisUtterance`. The system was verified to preload voices using the `voiceschanged` event to avoid falling back to the robotic default voice.

## 2. Context & Knowledge Testing (RAG)

- **RAG Retrieval:** Tested by asking specific domain questions (e.g., *"What interest rate do you offer?"*). Verified that the NumPy-based vector store successfully returned the top 2 most relevant FAQ chunks.
- **System Prompt Constraints:** Asked an out-of-domain question (*"What's the weather?"*). Confirmed the AI correctly refused to answer, staying within its persona constraints as a NeoBank assistant.
- **Context Window (Memory):** Verified that the React UI correctly appends both `user` and `assistant` turns to the `conversationHistory` array, ensuring the LLM maintains conversational memory across multiple turns.

---

## 3. Edge Cases Validated

To ensure the bot acts like a production-grade system rather than a fragile toy, the following edge cases were explicitly tested and handled:

| Edge Case | Expected Behavior | Actual Result |
| :--- | :--- | :--- |
| **Barge-in / Interruption** | User speaking over the AI should instantly stop the AI. | **PASS with limitation:** VAD successfully triggers cancel() when energy > 30, but listening cycle requires manual mic button interaction. Full hands-free operation requires ML-based VAD like Silero. |
| **Spam Clicking Mic** | Rapidly clicking the mic shouldn't spawn duplicate resources. | **PASS:** Guard clauses prevent multiple `AudioContext` instances and clear existing VAD intervals before creating new ones. |
| **Empty Speech** | Clicking mic, staying silent, and clicking again shouldn't send empty queries to the LLM. | **PASS:** System detects empty string and transitions safely back to the `IDLE` state. |
| **WebSocket Disconnects** | Server crash mid-conversation shouldn't permanently freeze the UI. | **PASS:** A safety `useEffect` hook resets the state machine to `IDLE` if the connection drops while `THINKING`. |

---

## 4. Real-World Issues Encountered & Resolved

During live testing, we hit two significant backend issues that reflect real-world deployment challenges:

### Issue 1: Windows Console Encoding Crash
- **The Problem:** The FastAPI backend crashed immediately on startup on a Windows machine. The root cause was Python's `cp1252` encoding trying to print Unicode emojis (📚) to the Windows terminal.
- **The Fix:** We performed a global search-and-replace to strip emojis from all `print()` statements in `main.py` and `rag.py`, replacing them with ASCII-safe logging markers like `[INFO]` and `[OK]`.

### Issue 2: Gemini LLM Rate Limiting (HTTP 429)
- **The Problem:** When running the first live voice query, the backend threw an HTTP 429 Error (`Quota exceeded for metric: generate_content_free_tier_requests`). The free tier quota for `gemini-2.0-flash` was exhausted.
- **The Fix:** We programmatically listed available models via the Gemini SDK and swapped the text generation model to `gemini-2.5-flash`, which had a fresh, unexhausted free-tier quota. The system successfully recovered and processed queries.

### Issue 3: Embedding Model Deprecation
- **The Problem:** Our original embedding model `text-embedding-004` was found to be deprecated, causing a crash during the RAG indexing phase.
- **The Fix:** We updated the embedding model in `rag.py` to `gemini-embedding-001`, which resolved the crash and restored the vector search capability.
