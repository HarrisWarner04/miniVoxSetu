// pcm-processor.js
// Runs on a dedicated browser audio rendering thread.
// Dual-output processor:
//   1. "pcm_chunk" — every 1.5s, Float32 PCM for acoustic analysis (HuBERT)
//   2. "stt_chunk" — every 250ms, Int16 PCM for Deepgram STT (linear16)

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // --- Acoustic path (1.5s chunks, Float32) ---
    this.acousticChunks = [];
    this.acousticSampleCount = 0;
    this.ACOUSTIC_DURATION = 1.5;

    // --- STT path (250ms chunks, Int16 for Deepgram linear16) ---
    this.sttChunks = [];
    this.sttSampleCount = 0;
    this.STT_DURATION = 0.25; // 250ms — matches MediaRecorder timeslice
  }

  process(inputs, outputs, parameters) {
    const inputChannel = inputs[0] ? inputs[0][0] : null;

    if (inputChannel && inputChannel.length > 0) {
      // === STT PATH: Convert Float32 → Int16 and send every 250ms ===
      this.sttChunks.push(new Float32Array(inputChannel));
      this.sttSampleCount += inputChannel.length;

      const sttLimit = sampleRate * this.STT_DURATION;
      if (this.sttSampleCount >= sttLimit) {
        // Merge and convert Float32 → Int16 (linear16 for Deepgram)
        const merged = new Float32Array(this.sttSampleCount);
        let offset = 0;
        for (const chunk of this.sttChunks) {
          merged.set(chunk, offset);
          offset += chunk.length;
        }

        // Float32 [-1.0, 1.0] → Int16 [-32768, 32767]
        const int16 = new Int16Array(merged.length);
        for (let i = 0; i < merged.length; i++) {
          const s = Math.max(-1, Math.min(1, merged[i]));
          int16[i] = s < 0 ? s * 32768 : s * 32767;
        }

        this.port.postMessage({
          type: 'stt_chunk',
          samples: int16.buffer,   // raw Int16 ArrayBuffer
          sampleRate: sampleRate
        }, [int16.buffer]); // Transfer ownership for zero-copy

        this.sttChunks = [];
        this.sttSampleCount = 0;
      }

      // === ACOUSTIC PATH: Keep Float32 and send every 1.5s ===
      this.acousticChunks.push(new Float32Array(inputChannel));
      this.acousticSampleCount += inputChannel.length;

      const acousticLimit = sampleRate * this.ACOUSTIC_DURATION;
      if (this.acousticSampleCount >= acousticLimit) {
        const merged = new Float32Array(this.acousticSampleCount);
        let offset = 0;
        for (const chunk of this.acousticChunks) {
          merged.set(chunk, offset);
          offset += chunk.length;
        }

        this.port.postMessage({
          type: 'pcm_chunk',
          samples: merged,
          sampleRate: sampleRate
        });

        this.acousticChunks = [];
        this.acousticSampleCount = 0;
      }
    }

    return true; // Keep processor alive
  }
}

registerProcessor('pcm-processor', PCMProcessor);
