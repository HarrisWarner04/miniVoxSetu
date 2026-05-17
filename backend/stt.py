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
DEEPGRAM_PARAMS = (
    "?model=nova-2"
    "&language=en"
    "&smart_format=true"
    "&interim_results=true"
    "&utterance_end_ms=1500"
    "&endpointing=300"
    "&vad_events=true"
)


class DeepgramSTT:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws = None
        self._listener_task = None
        self._final_segments = []  # accumulate finalized text segments

    async def connect(self, on_transcript, on_utterance_end):
        """
        Open WebSocket to Deepgram. Callbacks:
        - on_transcript(text, is_final): called on every transcript event
        - on_utterance_end(full_text): called when user finishes speaking
        """
        headers = {"Authorization": f"Token {self.api_key}"}
        self.ws = await websockets.connect(
            DEEPGRAM_WS_URL + DEEPGRAM_PARAMS,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
        )
        self._on_transcript = on_transcript
        self._on_utterance_end = on_utterance_end
        self._final_segments = []
        self._listener_task = asyncio.create_task(self._listen())

    async def send(self, audio_bytes: bytes):
        """Forward raw audio bytes to Deepgram."""
        if self.ws:
            try:
                await self.ws.send(audio_bytes)
            except Exception:
                pass

    async def close(self):
        """Gracefully close the Deepgram connection."""
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

                if msg_type == "Results":
                    alt = data["channel"]["alternatives"][0]
                    transcript = alt.get("transcript", "")
                    is_final = data.get("is_final", False)

                    if transcript:
                        await self._on_transcript(transcript, is_final)

                    # Accumulate finalized segments for full utterance
                    if is_final and transcript:
                        self._final_segments.append(transcript)

                elif msg_type == "UtteranceEnd":
                    # User finished speaking — combine all final segments
                    full_text = " ".join(self._final_segments).strip()
                    self._final_segments = []
                    if full_text:
                        await self._on_utterance_end(full_text)

        except websockets.exceptions.ConnectionClosed:
            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[STT] Deepgram listener error: {e}")

    def flush(self):
        """Clear accumulated segments (called on barge-in)."""
        self._final_segments = []
