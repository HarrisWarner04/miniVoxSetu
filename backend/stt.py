"""
stt.py — Deepgram Streaming STT Client
Connects to Deepgram's WebSocket API, forwards audio bytes,
and fires callbacks on transcript events.
"""

import asyncio
import json
import websockets


DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"

# Deepgram streaming params:
# - nova-2: best accuracy model
# - interim_results: get partial transcripts as user speaks
# - utterance_end_ms: fire UtteranceEnd after 1.5s silence
# - endpointing: consider speech done after 300ms pause
# - vad_events: get VAD start/stop events
# - encoding=linear16: raw Int16 PCM from AudioWorklet (NOT WebM container)
# - sample_rate=48000: browser AudioContext default sample rate
# - channels=1: mono audio
DEEPGRAM_PARAMS = (
    "?model=nova-2"
    "&language=en"
    "&smart_format=true"
    "&interim_results=true"
    "&utterance_end_ms=1000"
    "&endpointing=300"
    "&vad_events=true"
    "&encoding=linear16"
    "&sample_rate=48000"
    "&channels=1"
)


class DeepgramSTT:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws = None
        self._listener_task = None
        self._final_segments = []  # accumulate finalized text segments
        self._bytes_sent = 0       # DEBUG: track total bytes sent
        self._first_msg_logged = False  # DEBUG: log first Deepgram response
        self._early_fired = False  # Prevent double-fire when UtteranceEnd follows early trigger

    def is_connected(self):
        """Check if the WebSocket is open (compatible with websockets v15+)."""
        if not self.ws:
            return False
        try:
            # websockets v15+: use .state instead of .open
            from websockets.protocol import State
            return self.ws.state is State.OPEN
        except (AttributeError, ImportError):
            # Fallback for older versions
            return getattr(self.ws, 'open', False)

    async def connect(self, on_transcript, on_utterance_end, on_confident_interim=None, on_speech_started=None):
        """
        Open WebSocket to Deepgram. Callbacks:
        - on_transcript(text, is_final, confidence): called on every transcript event
        - on_utterance_end(full_text): called when user finishes speaking
        - on_confident_interim(text): called on interim/final with confidence >= 0.7 (for speculative RAG)
        """
        url = DEEPGRAM_WS_URL + DEEPGRAM_PARAMS
        headers = {"Authorization": f"Token {self.api_key}"}
        print(f"[STT] Connecting to Deepgram: {url}")
        print(f"[STT] API key (first 8 chars): {self.api_key[:8]}...")
        self.ws = await websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
        )
        print(f"[STT] ✅ Deepgram WebSocket connected successfully")
        self._on_transcript = on_transcript
        self._on_utterance_end = on_utterance_end
        self._on_confident_interim = on_confident_interim
        self._on_speech_started = on_speech_started
        self._final_segments = []
        self._bytes_sent = 0
        self._first_msg_logged = False
        self._early_fired = False
        self._listener_task = asyncio.create_task(self._listen())

    async def send(self, audio_bytes: bytes):
        """Forward raw audio bytes to Deepgram."""
        if not self.ws:
            print(f"[STT] ⚠️ send() called but self.ws is None!")
            return
        try:
            self._bytes_sent += len(audio_bytes)
            await self.ws.send(audio_bytes)
            # Log first 5 sends only to avoid terminal clutter
            if self._bytes_sent <= 5 * len(audio_bytes):
                print(f"[STT] 📤 Sent {len(audio_bytes)} bytes (total: {self._bytes_sent:,}), connected={self.is_connected()}")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[STT] ❌ Failed to send audio — connection closed: {e}")
        except Exception as e:
            print(f"[STT] ❌ Failed to send audio: {e}")

    async def close(self):
        """Gracefully close the Deepgram connection."""
        print(f"[STT] Closing Deepgram (total bytes sent: {self._bytes_sent:,})")
        if self._listener_task:
            self._listener_task.cancel()
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
            except Exception:
                pass

    async def _listen(self):
        """Background task: receive and parse Deepgram messages."""
        try:
            async for raw_msg in self.ws:
                data = json.loads(raw_msg)
                msg_type = data.get("type", "")

                # DEBUG: Log first message from Deepgram (usually "Metadata")
                if not self._first_msg_logged:
                    self._first_msg_logged = True
                    print(f"[STT] 📩 First Deepgram message: {msg_type}")
                    if msg_type == "Metadata":
                        req_id = data.get("request_id", "?")
                        print(f"[STT] 📩 Request ID: {req_id}")
                    elif msg_type == "Error":
                        print(f"[STT] ❌ FIRST MESSAGE IS ERROR: {json.dumps(data, indent=2)}")

                if msg_type == "Results":
                    alt = data["channel"]["alternatives"][0]
                    transcript = alt.get("transcript", "")
                    confidence = alt.get("confidence", 0.0)
                    is_final = data.get("is_final", False)
                    speech_final = data.get("speech_final", False)

                    if transcript:
                        print(f"[STT] 🗣️ {'FINAL' if is_final else 'interim'} (conf={confidence:.2f}): \"{transcript}\"")
                        await self._on_transcript(transcript, is_final, confidence)

                    # Speculative RAG callback: fire on any transcript with confidence >= 0.7
                    if transcript and confidence >= 0.7 and self._on_confident_interim:
                        # Build speculative text from accumulated segments + current
                        speculative_text = " ".join(self._final_segments + [transcript]).strip()
                        if len(speculative_text.split()) >= 3:  # at least 3 words
                            await self._on_confident_interim(speculative_text)

                    # Accumulate finalized segments for full utterance
                    if is_final and transcript:
                        self._final_segments.append(transcript)

                    # === HYBRID EARLY TRIGGER ===
                    # If Deepgram's endpointing detected a natural pause (speech_final=True)
                    # AND the segment has high confidence (>= 0.9), fire the pipeline immediately
                    # instead of waiting for the 800ms UtteranceEnd timeout.
                    if speech_final and is_final and confidence >= 0.9 and self._final_segments:
                        full_text = " ".join(self._final_segments).strip()
                        if full_text:
                            print(f"[STT] ⚡ EARLY TRIGGER (speech_final + conf={confidence:.2f}): \"{full_text}\"")
                            self._final_segments = []
                            self._early_fired = True
                            await self._on_utterance_end(full_text)

                elif msg_type == "UtteranceEnd":
                    # If we already early-fired for this utterance, skip the duplicate
                    if self._early_fired:
                        print(f"[STT] ⏭️ UtteranceEnd skipped (already early-fired)")
                        self._early_fired = False
                        self._final_segments = []  # Clean up any stragglers
                    else:
                        # Standard path: user finished speaking — combine all final segments
                        full_text = " ".join(self._final_segments).strip()
                        self._final_segments = []
                        if full_text:
                            print(f"[STT] ✅ Utterance complete: \"{full_text}\"")
                            await self._on_utterance_end(full_text)
                        else:
                            print(f"[STT] ⚠️ UtteranceEnd received but no final segments accumulated")
                        
                elif msg_type == "SpeechStarted":
                    print(f"[STT] 🎙️ Speech started detected by Deepgram VAD")
                    if self._on_speech_started:
                        await self._on_speech_started()

                elif msg_type == "Error":
                    print(f"[STT] ❌ Deepgram ERROR: {json.dumps(data, indent=2)}")
                    
                elif msg_type == "Metadata":
                    # Connection metadata — already logged on first message
                    pass
                    
                else:
                    print(f"[STT] ❓ Unknown Deepgram message type: {msg_type}: {raw_msg[:200]}")

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[STT] ❌ Deepgram connection closed: code={e.code}, reason={e.reason}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[STT] ❌ Deepgram listener error: {type(e).__name__}: {e}")

    def flush(self):
        """Clear accumulated segments (called on barge-in)."""
        self._final_segments = []
        self._early_fired = False
