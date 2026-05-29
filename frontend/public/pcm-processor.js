// pcm-processor.js
// Runs on a dedicated browser audio rendering thread.
// Triple-output processor:
//   1. "pcm_chunk" — every 1.5s, Float32 PCM for acoustic analysis (HuBERT)
//   2. "stt_chunk" — every 100ms, Int16 PCM for Deepgram STT (linear16)
//   3. "barge_in_detected" — immediate, when RMS energy > threshold during agent speech (B3)

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // --- Acoustic path (1.5s chunks, Float32) ---
    this.acousticChunks = [];
    this.acousticSampleCount = 0;
    this.ACOUSTIC_DURATION = 1.5;

    // --- STT path (100ms chunks, Int16 for Deepgram linear16) ---
    this.sttChunks = [];
    this.sttSampleCount = 0;
    this.STT_DURATION = 0.10; // 100ms — optimized for low-latency STT streaming

    // --- B3: VAD / Barge-in detection ---
    this.isAgentSpeaking = false;    // Updated by main thread via port.postMessage
    this.energyThreshold = 0.01;       // B1: Updated after calibration via port.postMessage
    this.speakingThresholdMultiplier = 3.0; // B8: During agent speech, threshold is 3x higher
    this.ttsSuppressUntil = 0;       // B2: Timestamp until which VAD is suppressed
    this.lastBargeInTime = 0;        // Cooldown: prevent rapid-fire barge-in messages
    this.BARGE_IN_COOLDOWN_MS = 150; // Min interval between barge-in messages
    this.consecutiveAboveCount = 0;  // B9: Require sustained energy, not single-frame spikes
    this.CONSECUTIVE_REQUIRED = 3;   // ~8ms of sustained speech (3 frames × 2.7ms)

    // --- B3: Listen for messages from main thread ---
    this.port.onmessage = (event) => {
      const { type } = event.data;
      if (type === 'agent_state') {
        this.isAgentSpeaking = event.data.speaking;
      } else if (type === 'set_threshold') {
        this.energyThreshold = event.data.threshold;
      } else if (type === 'tts_suppress') {
        // B2: Suppress VAD for a duration (ms) to let AEC settle
        // NOTE: Using Date.now() because `currentTime` is only available inside process()
        this.ttsSuppressUntil = Date.now() + event.data.duration_ms;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const inputChannel = inputs[0] ? inputs[0][0] : null;

    if (inputChannel && inputChannel.length > 0) {

      // === B3: REAL-TIME BARGE-IN VAD (every ~2.7ms frame) ===
      if (this.isAgentSpeaking) {
        const now = Date.now(); // Must match Date.now() used in tts_suppress handler

        // B2: Skip during TTS suppression window
        if (now >= this.ttsSuppressUntil) {
          // Calculate RMS energy of this 128-sample frame
          let sumSquares = 0;
          for (let i = 0; i < inputChannel.length; i++) {
            sumSquares += inputChannel[i] * inputChannel[i];
          }
          const rms = Math.sqrt(sumSquares / inputChannel.length);

          // B8: Use elevated threshold during agent speech to reject echo leak
          // Echo-cancelled speaker leak is typically 2-3x ambient noise,
          // while actual human speech is 5-10x. The 3x multiplier sits between.
          const effectiveThreshold = this.energyThreshold * this.speakingThresholdMultiplier;

          if (rms > effectiveThreshold) {
            // B9: Require multiple consecutive frames above threshold
            // This filters out transient pops/clicks from triggering barge-in
            this.consecutiveAboveCount++;
            if (this.consecutiveAboveCount >= this.CONSECUTIVE_REQUIRED) {
              // Cooldown check: don't spam barge-in messages
              if (now - this.lastBargeInTime > this.BARGE_IN_COOLDOWN_MS) {
                this.lastBargeInTime = now;
                this.port.postMessage({ type: 'barge_in_detected', energy: rms });
              }
            }
          } else {
            // Reset consecutive counter on any below-threshold frame
            this.consecutiveAboveCount = 0;
          }
        } else {
          // During suppression window, also reset consecutive counter
          this.consecutiveAboveCount = 0;
        }
      } else {
        // Not speaking — reset consecutive counter
        this.consecutiveAboveCount = 0;
      }

      // === STT PATH: Convert Float32 → Int16 and send every 100ms ===
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
