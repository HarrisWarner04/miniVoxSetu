"""
tts.py — Deepgram Aura Streaming TTS (WebSocket)
Maintains a persistent WebSocket connection per session.
Eliminates HTTP round-trip overhead (~100ms per sentence saved).

Protocol:
  - Send: {"type": "Speak", "text": "..."} → receives binary audio frames
  - Send: {"type": "Flush"} → forces buffered audio to be sent
  - Send: {"type": "Clear"} → discards buffer (for barge-in)
  - Send: {"type": "Close"} → graceful disconnect
"""

import asyncio
import json
import websockets

DEEPGRAM_TTS_WS_URL = "wss://api.deepgram.com/v1/speak"
DEFAULT_MODEL = "aura-asteria-en"


class DeepgramTTS:
    """Persistent WebSocket TTS client — one connection per call session."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self.ws = None
        self._listener_task = None
        self._audio_queue = asyncio.Queue()  # audio chunks arrive here from the listener

    async def connect(self):
        """Open WebSocket to Deepgram Aura TTS."""
        url = f"{DEEPGRAM_TTS_WS_URL}?model={self.model}&encoding=mp3"
        headers = {"Authorization": f"Token {self.api_key}"}

        print(f"[TTS] Connecting to Deepgram Aura TTS WebSocket...")
        self.ws = await websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
        )
        print(f"[TTS] ✅ Deepgram TTS WebSocket connected")

        # Start background listener for incoming audio binary frames
        self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self):
        """Background task: receive binary audio frames from Deepgram TTS."""
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    # Binary frame = audio data
                    await self._audio_queue.put(message)
                elif isinstance(message, str):
                    # JSON text frame = metadata/control (Flushed, Warning, etc.)
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")
                        if msg_type == "Flushed":
                            # Signal that all buffered audio for the current Speak has been sent
                            await self._audio_queue.put(None)  # sentinel
                        elif msg_type == "Warning":
                            print(f"[TTS] ⚠️ Deepgram warning: {data.get('warn_msg', data)}")
                        elif msg_type == "Error":
                            print(f"[TTS] ❌ Deepgram error: {data}")
                    except json.JSONDecodeError:
                        pass
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[TTS] WebSocket closed: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[TTS] Listener error: {e}")

    async def synthesize(self, text: str) -> bytes:
        """
        Send text to Deepgram Aura TTS and collect all audio bytes.
        Returns the complete audio for this text segment as bytes.
        """
        if not self.ws:
            raise RuntimeError("TTS WebSocket not connected. Call connect() first.")

        # Send text to speak
        await self.ws.send(json.dumps({
            "type": "Speak",
            "text": text,
        }))

        # Send flush to force Deepgram to process and return audio immediately
        await self.ws.send(json.dumps({"type": "Flush"}))

        # Collect audio chunks until we get the Flushed sentinel
        audio_chunks = []
        try:
            while True:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=15.0)
                if chunk is None:
                    # Flushed sentinel — all audio for this text has been received
                    break
                audio_chunks.append(chunk)
        except asyncio.TimeoutError:
            print(f"[TTS] ⚠️ Timeout waiting for audio from Deepgram TTS")

        return b"".join(audio_chunks)

    async def clear(self):
        """Clear TTS buffer (for barge-in). Discards any pending audio."""
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "Clear"}))
                # Drain any pending audio from the queue
                while not self._audio_queue.empty():
                    self._audio_queue.get_nowait()
            except Exception:
                pass

    async def close(self):
        """Gracefully close the TTS WebSocket connection."""
        print(f"[TTS] Closing Deepgram TTS WebSocket")
        if self._listener_task:
            self._listener_task.cancel()
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "Close"}))
                await self.ws.close()
            except Exception:
                pass
        self.ws = None

    def is_connected(self):
        """Check if the WebSocket is open."""
        if not self.ws:
            return False
        try:
            from websockets.protocol import State
            return self.ws.state is State.OPEN
        except (AttributeError, ImportError):
            return getattr(self.ws, 'open', False)


# --- Legacy function kept for backward compatibility (reports, fallback) ---
async def synthesize_http(text: str, api_key: str) -> bytes:
    """HTTP-based TTS fallback. Used only for post-call report TTS or if WS fails."""
    import httpx
    url = f"https://api.deepgram.com/v1/speak?model={DEFAULT_MODEL}&encoding=mp3"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            print(f"[TTS] ❌ Deepgram Aura returned {response.status_code}: {response.text[:200]}")
        response.raise_for_status()
        return response.content


def detect_sentence_boundary(text: str) -> tuple:
    """
    Check if text ends with a sentence boundary.
    Returns (sentence, remaining) if boundary found, else (None, text).
    Used for cascaded streaming: send each sentence to TTS
    while LLM is still generating the rest.
    """
    # Common abbreviations that should NOT trigger a sentence split
    ABBREV = {"mr", "mrs", "ms", "dr", "sr", "jr", "vs", "etc", "inc", "ltd", "rs", "no", "st"}

    # Look for sentence-ending punctuation
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ".!?" and i > 3:  # min 3 chars to flush "OK." "Yes." etc.
            # Guard: check if this is an abbreviation (word before the period)
            if text[i] == ".":
                word_before = text[:i].split()[-1].lower().rstrip(".") if text[:i].split() else ""
                if word_before in ABBREV:
                    continue
                # Guard: don't split on decimal numbers like "₹1.5" or "3.14"
                if i > 0 and text[i-1].isdigit() and i < len(text) - 1 and text[i+1:i+2].isdigit():
                    continue

            sentence = text[:i + 1].strip()
            remaining = text[i + 1:].strip()
            return sentence, remaining

    return None, text
