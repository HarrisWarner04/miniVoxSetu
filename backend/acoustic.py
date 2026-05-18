# acoustic.py
# Processes raw Float32 PCM audio to extract physics-based and ML-based features.
# Runs on a ThreadPoolExecutor to prevent blocking FastAPI's main event loop.

import numpy as np
import base64
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor

# Try to import heavy deep-learning dependencies dynamically
# so we can test librosa-only logic first (Step 4 & 5)
try:
    import torch
    import librosa
    from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
    DEPS_AVAILABLE = True
except (ImportError, OSError) as e:
    DEPS_AVAILABLE = False
    print(f"PyTorch/librosa/transformers not fully loaded: {e}. Running in simulation mode.")

# Global ThreadPoolExecutor for CPU/GPU-bound tasks
executor = ThreadPoolExecutor(max_workers=1)

# Device configuration — guarded so we don't crash if torch isn't installed
if DEPS_AVAILABLE:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ACOUSTIC] PyTorch device: {device}")
else:
    device = "cpu"
model = None
processor = None

def init_hubert_model():
    """Lazy initializer for HuBERT model to save startup memory if CUDA isn't verified yet."""
    global model, processor, DEPS_AVAILABLE
    if not DEPS_AVAILABLE:
        return
    try:
        print(f"Loading superb/hubert-base-superb-er onto {device}...")
        processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")
        model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-er").to(device)
        model.eval()
        print("HuBERT model loaded successfully.")
    except Exception as e:
        print(f"Failed to load HuBERT model: {e}")
        traceback.print_exc()

def sync_analyze_audio(pcm_array: np.ndarray, sr: int, interrupted: bool) -> dict:
    """
    Synchronous blocking function. Runs Path A (librosa) and Path B (HuBERT).
    This function MUST run in a ThreadPoolExecutor.
    """
    # 1. Fallback / Simulation mode if dependencies aren't loaded yet
    if not DEPS_AVAILABLE or librosa is None:
        # Mock some clean physical values for testing the websocket
        mock_pitch = 180.0 + np.random.uniform(-10, 10)
        mock_rms = -20.0 + np.random.uniform(-5, 5)
        mock_stress = 0.2 + (0.5 if interrupted else 0.0)
        return {
            "pitch_hz": round(mock_pitch, 1),
            "rms_db": round(mock_rms, 1),
            "zcr": 0.05,
            "spectral_centroid": 1800.0,
            "energy_variance": 0.05,
            "is_speech": True,
            "emotion": "neutral",
            "emotion_confidence": 0.9,
            "emotion_scores": {"neutral": 0.9, "happy": 0.05, "angry": 0.03, "sad": 0.02},
            "stress_score": round(mock_stress, 2),
            "acoustic_summary": f"Simulated: neutral ({mock_pitch:.0f}Hz, {mock_rms:.1f}dB)",
            "interrupted": interrupted
        }

    try:
        # 2. Resample if necessary (HuBERT requires 16000Hz)
        if sr != 16000:
            audio_16k = librosa.resample(pcm_array, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = pcm_array

        # === PATH A: librosa (Signal Processing) ===
        # Fundamental Frequency (F0) tracking via probabilistic YIN
        # Limit search range [60, 400] Hz for human vocal frequencies
        try:
            f0, _, _ = librosa.pyin(audio_16k, fmin=60, fmax=400, sr=16000, fill_value=np.nan)
            pitch_hz = float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else 0.0
        except Exception:
            pitch_hz = 0.0

        # RMS Energy (Volume)
        rms = librosa.feature.rms(y=audio_16k)
        rms_db = float(20 * np.log10(np.mean(rms) + 1e-10))
        energy_variance = float(np.std(rms))
        
        # Zero-Crossing Rate (ZCR)
        zcr_arr = librosa.feature.zero_crossing_rate(audio_16k)
        zcr = float(np.mean(zcr_arr))

        # Spectral Centroid (brightness/timbre of speech)
        sc_arr = librosa.feature.spectral_centroid(y=audio_16k, sr=16000)
        spectral_centroid = float(np.mean(sc_arr))

        is_speech = bool(rms_db > -42.0)

        # === PATH B: HuBERT Model Inference (GPU/CPU-bound) ===
        emotion = "neutral"
        emotion_confidence = 1.0
        emotion_scores = {"neutral": 1.0, "happy": 0.0, "angry": 0.0, "sad": 0.0}

        if model is not None and processor is not None:
            try:
                # Convert numpy to pytorch tensor on target device
                inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
                # Model labels are IEMOCAP: neutral, happy, angry, sad
                labels = ['neutral', 'happy', 'angry', 'sad']
                emotion_scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
                emotion = max(emotion_scores, key=emotion_scores.get)
                emotion_confidence = emotion_scores[emotion]
            except Exception as model_err:
                print(f"Error running HuBERT model: {model_err}")

        # === FUSION LAYER ===
        # Calculate pitch stress: human base is ~150Hz.
        # Volume stress: -42dB (silence) to 0dB (loud clipping)
        normalized_pitch = float(min(1.0, pitch_hz / 350.0) if pitch_hz > 0 else 0.4)
        normalized_volume = float(min(1.0, max(0.0, (rms_db + 42) / 42)))

        # ZCR stress indicator: rapid zero-crossings indicate harsh/tense voice
        zcr_stress = float(min(1.0, zcr / 0.15))  # 0.15 is high ZCR for speech

        # Base stress score from raw physics (pitch + volume + ZCR)
        physics_stress = float((normalized_pitch * 0.4) + (normalized_volume * 0.4) + (zcr_stress * 0.2))
        
        # Model contribution to stress (angry + sad raise stress; happy/neutral lower it)
        model_stress = float(emotion_scores.get("angry", 0.0) * 1.0 + emotion_scores.get("sad", 0.0) * 0.5)

        # Weighted combination (50% physics, 50% model)
        # HuBERT on noisy short clips tends toward neutral, so physics gets equal weight
        stress_score = float(round(min(1.0, max(0.0, (physics_stress * 0.5) + (model_stress * 0.5))), 2))

        # Physics Validation (Override Model Errors)
        # Override 1: Loud voice — if model is neutral but user is shouting (RMS > -22dB)
        #   Normal laptop speech is ~-30dB, shouting is ~-20dB, clipping is >-10dB
        if emotion == "neutral" and rms_db > -22.0:
            emotion = "agitated"
            emotion_confidence = float(min(1.0, physics_stress * 1.2))
            stress_score = max(stress_score, 0.55)  # Floor stress when shouting detected

        # Override 2: High pitch — raised pitch (>250Hz) + moderate volume (>-28dB) = frustration
        elif emotion == "neutral" and pitch_hz > 250.0 and rms_db > -28.0:
            emotion = "agitated"
            emotion_confidence = float(normalized_pitch)
            stress_score = max(stress_score, 0.45)

        # Override 3: High spectral centroid — harsh/bright voice = tension
        elif emotion == "neutral" and spectral_centroid > 3500 and rms_db > -30.0:
            emotion = "agitated"
            emotion_confidence = float(min(1.0, spectral_centroid / 5000))
            stress_score = max(stress_score, 0.40)

        # Override 4: Whisper / Sad — if model says angry but volume is whisper-quiet (RMS < -35dB)
        elif emotion == "angry" and rms_db < -35.0:
            emotion = "agitated_whisper"
            emotion_confidence = float(physics_stress)

        return {
            "pitch_hz": float(round(pitch_hz, 1)),
            "rms_db": float(round(rms_db, 1)),
            "zcr": float(round(zcr, 4)),
            "spectral_centroid": float(round(spectral_centroid, 1)),
            "energy_variance": float(round(energy_variance, 4)),
            "is_speech": bool(is_speech),
            "emotion": str(emotion),
            "emotion_confidence": float(round(emotion_confidence, 2)),
            "emotion_scores": {k: float(round(v, 3)) for k, v in emotion_scores.items()},
            "stress_score": float(stress_score),
            "acoustic_summary": f"Emotion: {emotion} ({emotion_confidence:.0%}), Stress: {stress_score:.0%}, Pitch: {pitch_hz:.0f}Hz",
            "interrupted": bool(interrupted)
        }

    except Exception as general_err:
        print(f"Error in acoustic feature extraction: {general_err}")
        traceback.print_exc()
        return {
            "pitch_hz": 0.0, "rms_db": -60.0, "zcr": 0.0, "spectral_centroid": 0.0, "energy_variance": 0.0,
            "is_speech": False, "emotion": "unknown", "emotion_confidence": 0.0,
            "emotion_scores": {"neutral": 0.0, "happy": 0.0, "angry": 0.0, "sad": 0.0},
            "stress_score": 0.0, "acoustic_summary": "Inference failed", "interrupted": interrupted
        }

async def analyze_audio(pcm_base64: str, sample_rate: int = 48000, interrupted: bool = False) -> dict:
    """
    Asynchronous event-loop wrapper. Decodes base64 raw PCM bytes and runs
    the heavy, blocking sync_analyze_audio function on the ThreadPoolExecutor.
    """
    try:
        # Decode base64 raw float32 PCM samples
        pcm_bytes = base64.b64decode(pcm_base64)
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.float32)
        
        if len(pcm_array) == 0:
            raise ValueError("Empty audio buffer received")
            
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            sync_analyze_audio,
            pcm_array,
            sample_rate,
            interrupted
        )
    except Exception as err:
        print(f"Async wrapper error: {err}")
        return {
            "pitch_hz": 0.0, "rms_db": -60.0, "zcr": 0.0, "spectral_centroid": 0.0, "energy_variance": 0.0,
            "is_speech": False, "emotion": "unknown", "emotion_confidence": 0.0,
            "emotion_scores": {"neutral": 0.0, "happy": 0.0, "angry": 0.0, "sad": 0.0},
            "stress_score": 0.0, "acoustic_summary": f"Failed to decode pcm: {err}", "interrupted": interrupted
        }
