"""
tts.py — ElevenLabs Streaming TTS Client
Sends text to ElevenLabs API and returns audio bytes (mp3).
"""

import httpx

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Default voice: "Rachel" — clear, professional female voice
# Default voice: "Roger" — allowed on free tier
# (Rachel 21m00Tcm4TlvDq8ikWAM is now restricted to paid plans)
DEFAULT_VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"

# eleven_multilingual_v2: standard model available on free tier
DEFAULT_MODEL = "eleven_multilingual_v2"


async def synthesize(
    text: str,
    api_key: str,
    voice_id: str = DEFAULT_VOICE_ID,
    model_id: str = DEFAULT_MODEL,
) -> bytes:
    """
    Convert text to speech using ElevenLabs API.
    Returns mp3 audio bytes.
    """
    url = f"{ELEVENLABS_API_URL}/{voice_id}/stream"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.content


def detect_sentence_boundary(text: str) -> tuple:
    """
    Check if text ends with a sentence boundary.
    Returns (sentence, remaining) if boundary found, else (None, text).
    Used for cascaded streaming: send each sentence to TTS
    while LLM is still generating the rest.
    """
    # Look for sentence-ending punctuation
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ".!?" and i > 10:  # min 10 chars to avoid false splits
            sentence = text[:i + 1].strip()
            remaining = text[i + 1:].strip()
            return sentence, remaining

    return None, text
