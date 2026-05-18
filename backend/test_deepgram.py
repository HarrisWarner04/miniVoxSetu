"""
Test Deepgram by sending a real WAV file to check if the API is working.
This proves if the issue is with audio format or with the API account itself.
"""
import asyncio
import websockets
import json
import wave
import struct
import io

DEEPGRAM_KEY = "f7af1ee1e6ceef1d868a276cff205f44dd56bbfc"

def generate_tone_wav(freq=440, duration=2.0, sample_rate=16000):
    """Generate a simple sine wave WAV file in memory."""
    import math
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        sample = int(32767 * 0.5 * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack('<h', sample))
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(samples))
    return buf.getvalue()

async def test_with_wav():
    # Use encoding=linear16 and sample_rate since we're sending raw PCM
    url = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-2"
        "&language=en"
        "&encoding=linear16"
        "&sample_rate=16000"
        "&channels=1"
    )
    headers = {"Authorization": f"Token {DEEPGRAM_KEY}"}
    
    print("Generating 2s tone at 440Hz...")
    wav_data = generate_tone_wav(440, 2.0)
    print(f"WAV size: {len(wav_data)} bytes")
    
    print(f"Connecting to Deepgram...")
    ws = await websockets.connect(url, additional_headers=headers, ping_interval=5, ping_timeout=5)
    print("Connected!")
    
    # Send WAV data in chunks
    chunk_size = 4000
    for i in range(0, len(wav_data), chunk_size):
        chunk = wav_data[i:i+chunk_size]
        await ws.send(chunk)
    print(f"Sent all audio data")
    
    # Send CloseStream to signal end of audio
    await ws.send(json.dumps({"type": "CloseStream"}))
    print("Sent CloseStream")
    
    # Collect all responses
    responses = []
    try:
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
            data = json.loads(msg)
            msg_type = data.get("type", "?")
            responses.append(msg_type)
            print(f"  Got: {msg_type}")
            if msg_type == "Error":
                print(f"  ERROR DETAIL: {json.dumps(data, indent=2)}")
            elif msg_type == "Metadata":
                print(f"  Request ID: {data.get('request_id', '?')}")
            elif msg_type == "Results":
                alt = data.get("channel", {}).get("alternatives", [{}])[0]
                transcript = alt.get("transcript", "")
                print(f"  Transcript: '{transcript}'")
    except asyncio.TimeoutError:
        print(f"  No more messages (timeout)")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"  Connection closed: {e.code} {e.reason}")
    
    print(f"\nTotal messages received: {len(responses)}")
    print(f"Message types: {responses}")
    
    if len(responses) == 0:
        print("\n⚠️  ZERO RESPONSES — Deepgram API is NOT processing audio!")
        print("This typically means:")
        print("  1. Free tier credits are EXHAUSTED")
        print("  2. Account is suspended/rate-limited")
        print("  3. The API key doesn't have streaming scope")
        print("\nPlease check your Deepgram console: https://console.deepgram.com/")

asyncio.run(test_with_wav())
