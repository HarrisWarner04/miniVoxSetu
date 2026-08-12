/**
 * miniVoxSetu — Voice AI Pipeline
 * STT: Deepgram Nova-2 (server-side) via streaming audio
 * TTS: Deepgram Aura (server-side) via persistent WebSocket
 * LLM: Groq LLaMA 3.3 70B (server-side) via SSE streaming
 * Mic: Always-on after start (no click-per-message)
 * Barge-in: 7-layer detection system (B1–B7)
 */

import { useState, useRef, useCallback, useEffect } from 'react';

const STATES = {
  IDLE: 'idle',
  LISTENING: 'listening',
  THINKING: 'thinking',
  SPEAKING: 'speaking',
};

const STATE_LABELS = {
  [STATES.IDLE]: 'Ready',
  [STATES.LISTENING]: 'Listening',
  [STATES.THINKING]: 'Thinking',
  [STATES.SPEAKING]: 'Speaking',
};

// ============================================================
// CUSTOM HOOK: useWebSocket (supports binary + text frames)
// ============================================================

function useWebSocket(url) {
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const onTextMessageRef = useRef(null);
  const hasConnectedRef = useRef(false);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer'; // for receiving audio back

    ws.onopen = () => {
      hasConnectedRef.current = true;
      setIsConnected(true);
    };

    ws.onclose = () => {
      setIsConnected(false);
      setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      if (hasConnectedRef.current) {
        console.error('WebSocket connection lost');
      }
      ws.close();
    };

    ws.onmessage = (event) => {
      // All messages from backend are JSON text (audio is base64 inside JSON)
      if (typeof event.data === 'string' && onTextMessageRef.current) {
        try {
          onTextMessageRef.current(JSON.parse(event.data));
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      }
    };

    wsRef.current = ws;
  }, [url]);

  // Send JSON text message
  const sendJSON = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  // Send binary audio data
  const sendBinary = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  const setOnMessage = useCallback((handler) => {
    onTextMessageRef.current = handler;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { isConnected, sendJSON, sendBinary, setOnMessage, wsRef };
}

// ============================================================
// MAIN APP COMPONENT
// ============================================================

export default function App() {
  // Dynamic WebSocket URL:
  // - Production: Set VITE_WS_URL env var (e.g., wss://your-oracle-ip:8000/ws/chat)
  // - Dev: Falls back to same-origin via Vite proxy
  const wsUrl = import.meta.env.VITE_WS_URL ||
    `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/chat`;
  const { isConnected, sendJSON, sendBinary, setOnMessage, wsRef } =
    useWebSocket(wsUrl);

  // --- Agent state machine ---
  const [agentState, setAgentState] = useState(STATES.IDLE);
  const agentStateRef = useRef(STATES.IDLE);

  // --- Transcript ---
  const [liveTranscript, setLiveTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [streamingResponse, setStreamingResponse] = useState('');

  // --- Conversation ---
  const [conversationHistory, setConversationHistory] = useState([]);
  const [ragChunks, setRagChunks] = useState([]);
  const [ragQuery, setRagQuery] = useState('');

  // --- Semantic intelligence ---
  const [semanticData, setSemanticData] = useState(null);
  const [semanticLatency, setSemanticLatency] = useState(0);

  // --- Acoustic intelligence ---
  const [acousticData, setAcousticData] = useState(null);
  const [acousticLatency, setAcousticLatency] = useState(0);

  // --- Text input fallback ---
  const [textInput, setTextInput] = useState('');

  // --- Error ---
  const [sttError, setSttError] = useState('');

  // --- Post-call report ---
  const [reportData, setReportData] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);

  // --- Audio recording ---
  const mediaRecorderRef = useRef(null);
  const audioWorkletNodeRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const isRecordingRef = useRef(false);

  // --- Audio playback ---
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const currentAudioSourceRef = useRef(null);
  const audioContextRef = useRef(null);
  const bargedInRef = useRef(false); // Prevents queued audio from playing after barge-in

  // --- VAD / Barge-in ---
  const analyserRef = useRef(null);
  const vadIntervalRef = useRef(null);
  const energyThresholdRef = useRef(30);     // B1: Dynamic threshold (calibrated at mic start)
  const ttsSuppressUntilRef = useRef(0);     // B2: Timestamp until which VAD is suppressed

  // --- Refs ---
  const historyPanelRef = useRef(null);
  const pendingUtteranceRef = useRef('');
  const currentTurnUserTextRef = useRef('');

  // Sync state to ref for use in callbacks
  useEffect(() => {
    agentStateRef.current = agentState;

    // B3: Reset worklet speaking state when leaving SPEAKING state
    if (agentState !== STATES.SPEAKING && audioWorkletNodeRef.current) {
      audioWorkletNodeRef.current.port.postMessage({
        type: 'agent_state',
        speaking: false,
      });
    }
  }, [agentState]);

  // Auto-scroll conversation history
  useEffect(() => {
    if (historyPanelRef.current) {
      historyPanelRef.current.scrollTop = historyPanelRef.current.scrollHeight;
    }
  }, [conversationHistory]);

  // ============================================================
  // AUDIO RECORDING (always-on mic via MediaRecorder)
  // ============================================================

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      mediaStreamRef.current = stream;

      // Set up AnalyserNode for VAD energy detection (barge-in)
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
      audioContextRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Set up AudioWorklet for BOTH STT (linear16) and Acoustic (Float32) paths
      try {
        await audioCtx.audioWorklet.addModule('/pcm-processor.js');
        const pcmNode = new AudioWorkletNode(audioCtx, 'pcm-processor');
        
        pcmNode.port.onmessage = (event) => {
          if (!isRecordingRef.current) return;

          if (event.data.type === 'stt_chunk') {
            // STT path: Int16 PCM binary → sent as binary WebSocket frame → Deepgram
            const int16Buffer = event.data.samples; // ArrayBuffer of Int16
            sendBinary(int16Buffer);
          }

          else if (event.data.type === 'pcm_chunk') {
            // Acoustic path: Float32 PCM → base64 JSON → HuBERT analysis
            const { samples, sampleRate } = event.data;
            const pcmBytes = new Uint8Array(samples.buffer);
            // Fast base64 encoding using chunk-based approach
            // (avoids O(n) string concatenation of the old per-byte loop)
            const CHUNK_SIZE = 0x8000;
            let binary = '';
            for (let i = 0; i < pcmBytes.length; i += CHUNK_SIZE) {
              binary += String.fromCharCode.apply(null, pcmBytes.subarray(i, i + CHUNK_SIZE));
            }
            const base64 = btoa(binary);
            
            sendJSON({
              type: 'acoustic_pcm',
              data: base64,
              sample_rate: sampleRate
            });
          }

          // B3: Worklet-based barge-in detection (~2.7ms latency vs 100ms setInterval)
          else if (event.data.type === 'barge_in_detected') {
            handleBargeIn();
          }
        };

        source.connect(pcmNode);
        pcmNode.connect(audioCtx.destination);
        audioWorkletNodeRef.current = pcmNode;
      } catch (workletErr) {
        console.warn('AudioWorklet loading failed, falling back:', workletErr);
      }

      // NOTE: MediaRecorder removed — STT now uses raw PCM (linear16) from AudioWorklet
      // This avoids the WebM container parsing issue with Deepgram streaming API.

      isRecordingRef.current = true;

      // B1: Calibrate ambient noise for 500ms before starting VAD
      const calibrateSamples = [];
      const CALIBRATION_MS = 500;
      const calInterval = setInterval(() => {
        const calData = new Float32Array(analyser.fftSize);
        analyser.getFloatTimeDomainData(calData);
        let sumSquares = 0;
        for (let i = 0; i < calData.length; i++) {
          sumSquares += calData[i] * calData[i];
        }
        const rms = Math.sqrt(sumSquares / calData.length);
        calibrateSamples.push(rms);
      }, 50);

      await new Promise((resolve) => setTimeout(resolve, CALIBRATION_MS));
      clearInterval(calInterval);

      if (calibrateSamples.length > 0) {
        const baseline = calibrateSamples.reduce((a, b) => a + b, 0) / calibrateSamples.length;
        // Set dynamic threshold: 2.5x baseline, with a minimum floor of 0.01 (raw RMS)
        const dynamicThreshold = Math.max(0.01, baseline * 2.5);
        energyThresholdRef.current = dynamicThreshold;
        console.log(`[VAD] Baseline RMS: ${baseline.toFixed(4)}, Threshold: ${dynamicThreshold.toFixed(4)}`);

        // B3: Send calibrated threshold to AudioWorklet
        if (audioWorkletNodeRef.current) {
          audioWorkletNodeRef.current.port.postMessage({
            type: 'set_threshold',
            threshold: dynamicThreshold
          });
        }
      }

      // B3: Barge-in detection now runs in AudioWorklet (pcm-processor.js)
      // The setInterval VAD is no longer needed — worklet fires barge_in_detected
      // at ~2.7ms frame rate instead of 100ms polling.

      setAgentState(STATES.LISTENING);
      setSttError('');
    } catch (err) {
      console.error('Mic access failed:', err);
      setSttError('Microphone access denied. Please allow mic access.');
    }
  }, [sendBinary, sendJSON]);

  const stopRecording = useCallback(() => {
    const wasRecording = isRecordingRef.current;
    isRecordingRef.current = false;

    // Request post-call report from backend ONLY if we were actually in a call
    if (wasRecording) {
      sendJSON({ type: 'end_call' });
      setReportLoading(true);
    }

    if (audioWorkletNodeRef.current) {
      try {
        audioWorkletNodeRef.current.disconnect();
      } catch (e) {}
      audioWorkletNodeRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }

    // Close AudioContext to free browser audio resources
    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch (e) {}
      audioContextRef.current = null;
    }

    setAgentState(STATES.IDLE);
  }, [sendJSON]);

  // ============================================================
  // BARGE-IN DETECTION — B3: Now handled in AudioWorklet (pcm-processor.js)
  // The worklet calculates RMS energy per 128-sample frame (~2.7ms at 48kHz)
  // and posts 'barge_in_detected' to the main thread when threshold is exceeded.
  // The old setInterval(100ms) polling has been removed.
  // ============================================================

  // ============================================================
  // BARGE-IN HANDLER
  // ============================================================

  const handleBargeIn = useCallback(() => {
    // Set barge-in flag FIRST to block any incoming audio from being queued
    bargedInRef.current = true;

    // Stop TTS audio playback
    if (currentAudioSourceRef.current) {
      try {
        currentAudioSourceRef.current.stop();
      } catch (e) { /* already stopped */ }
      currentAudioSourceRef.current = null;
    }
    audioQueueRef.current = [];
    isPlayingRef.current = false;

    // Reset worklet VAD speaking flag
    if (audioWorkletNodeRef.current) {
      audioWorkletNodeRef.current.port.postMessage({
        type: 'agent_state',
        speaking: false,
      });
    }

    // Tell backend to cancel pipeline
    sendJSON({ type: 'barge_in' });

    // Reset UI
    setStreamingResponse('');
    setAgentState(STATES.LISTENING);
  }, [sendJSON]);

  // ============================================================
  // AUDIO PLAYBACK (play TTS audio from backend)
  // ============================================================

  const playNextAudio = useCallback(async () => {
    // B7 fix: If barge-in happened, drain queue and do not play
    if (bargedInRef.current) {
      audioQueueRef.current = [];
      isPlayingRef.current = false;
      return;
    }
    if (isPlayingRef.current || audioQueueRef.current.length === 0) return;

    isPlayingRef.current = true;
    const audioData = audioQueueRef.current.shift();

    try {
      const audioCtx = audioContextRef.current;
      if (!audioCtx) return;

      // Resume AudioContext if suspended (browser autoplay policy)
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      const audioBuffer = await audioCtx.decodeAudioData(audioData.buffer);

      // B7 fix: Re-check after async decode — barge-in may have fired during decodeAudioData
      if (bargedInRef.current) {
        isPlayingRef.current = false;
        audioQueueRef.current = [];
        return;
      }

      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);

      currentAudioSourceRef.current = source;

      source.onended = () => {
        currentAudioSourceRef.current = null;
        isPlayingRef.current = false;
        // B7 fix: Don't chain to next audio if barge-in happened
        if (bargedInRef.current) return;
        // Play next queued audio or return to listening
        if (audioQueueRef.current.length > 0) {
          playNextAudio();
        } else {
          // No more audio, signal worklet that speaking stopped
          if (audioWorkletNodeRef.current) {
            audioWorkletNodeRef.current.port.postMessage({
              type: 'agent_state',
              speaking: false,
            });
          }
        }
      };

      source.start(0);

      // Notify worklet that speaker is now actively outputting sound
      if (audioWorkletNodeRef.current) {
        audioWorkletNodeRef.current.port.postMessage({
          type: 'agent_state',
          speaking: true,
        });
      }

      // B2: Set TTS suppression window — let AEC settle before VAD resumes
      ttsSuppressUntilRef.current = Date.now() + 200;
      // B3: Also notify worklet of TTS suppression
      if (audioWorkletNodeRef.current) {
        audioWorkletNodeRef.current.port.postMessage({
          type: 'tts_suppress',
          duration_ms: 200
        });
      }
    } catch (err) {
      console.error('Audio playback error:', err);
      isPlayingRef.current = false;
      // Try next in queue
      if (audioQueueRef.current.length > 0) {
        playNextAudio();
      } else if (audioWorkletNodeRef.current) {
        audioWorkletNodeRef.current.port.postMessage({
          type: 'agent_state',
          speaking: false,
        });
      }
    }
  }, []);

  const queueAudio = useCallback((base64Data) => {
    // If barge-in happened, silently drop any audio that arrives late
    if (bargedInRef.current) return;

    const binaryStr = atob(base64Data);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }
    audioQueueRef.current.push(bytes);
    playNextAudio();
  }, [playNextAudio]);

  // ============================================================
  // WEBSOCKET MESSAGE HANDLER
  // ============================================================

  useEffect(() => {
    setOnMessage((message) => {
      switch (message.type) {
        case 'transcript':
          // Live STT from Deepgram
          if (message.is_final) {
            setLiveTranscript((prev) => (prev + ' ' + message.text).trim());
            setInterimTranscript('');
            pendingUtteranceRef.current =
              (pendingUtteranceRef.current + ' ' + message.text).trim();
          } else {
            setInterimTranscript(message.text);
          }
          break;

        case 'utterance':
          // Full utterance ready (for display)
          setLiveTranscript(message.text);
          setInterimTranscript('');
          break;

        case 'state':
          // Backend state change
          if (message.state === 'THINKING') {
            setAgentState(STATES.THINKING);
            setStreamingResponse('');
            const textToSave = pendingUtteranceRef.current || liveTranscript || '';
            if (textToSave) {
              currentTurnUserTextRef.current = textToSave;
              setLiveTranscript(textToSave);
            }
            pendingUtteranceRef.current = '';
          } else if (message.state === 'SPEAKING') {
            bargedInRef.current = false; // Reset barge-in flag for new response
            setAgentState(STATES.SPEAKING);
          } else if (message.state === 'LISTENING') {
            setAgentState(STATES.LISTENING);
          }
          break;

        case 'rag_context':
          setRagChunks(message.chunks || []);
          setRagQuery(message.query || '');
          break;

        case 'chunk':
          // LLM streaming text
          setStreamingResponse((prev) => prev + message.text);
          break;

        case 'audio':
          // TTS audio from Deepgram Aura (base64 mp3)
          queueAudio(message.data);
          break;

        case 'clear_audio':
          // Backend triggered a barge-in (VAD), clear local queue immediately
          handleBargeIn();
          break;

        case 'done': {
          const fullText = message.full_text || '';
          setConversationHistory((prev) => {
            const userText = currentTurnUserTextRef.current || pendingUtteranceRef.current || liveTranscript || '';
            const newHistory = [...prev];
            if (userText) {
              newHistory.push({ role: 'user', content: userText });
            }
            if (fullText) {
              newHistory.push({ role: 'assistant', content: fullText });
            }
            return newHistory;
          });
          setStreamingResponse('');
          currentTurnUserTextRef.current = '';
          break;
        }

        case 'semantic':
          // Semantic intelligence from parallel analysis
          setSemanticData(message.data);
          setSemanticLatency(message.latency_ms || 0);
          break;

        case 'acoustic':
          // Acoustic intelligence from dual-path engine
          setAcousticData(message.data);
          setAcousticLatency(message.latency_ms || 0);
          break;

        case 'report':
          // Post-call report from backend
          setReportData(message.data);
          setReportLoading(false);
          break;

        case 'error':
          console.error('Server error:', message.message);
          setSttError(message.message);
          setAgentState(STATES.LISTENING);
          break;

        default:
          break;
      }
    });
    // NOTE: liveTranscript intentionally excluded from deps to prevent
    // handler recreation on every transcript update (stale closure risk).
    // The 'done' handler uses pendingUtteranceRef instead.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setOnMessage, queueAudio]);

  // ============================================================
  // MIC BUTTON HANDLER
  // ============================================================

  const handleMicClick = useCallback(async () => {
    switch (agentState) {
      case STATES.IDLE:
        await startRecording();
        break;

      case STATES.LISTENING:
        stopRecording();
        break;

      case STATES.SPEAKING:
        handleBargeIn();
        break;

      case STATES.THINKING:
        break;

      default:
        break;
    }
  }, [agentState, startRecording, stopRecording, handleBargeIn]);

  // ============================================================
  // TEXT INPUT FALLBACK
  // ============================================================

  const handleTextSubmit = useCallback(
    (e) => {
      e.preventDefault();
      if (!textInput.trim() || !isConnected) return;

      // --- Barge-in: flush TTS if agent is speaking ---
      if (agentStateRef.current === STATES.SPEAKING) {
        // Stop current audio playback
        if (currentAudioSourceRef.current) {
          try { currentAudioSourceRef.current.stop(); } catch (_) {}
          currentAudioSourceRef.current = null;
        }
        audioQueueRef.current = [];
        isPlayingRef.current = false;
        // Tell backend to cancel old pipeline
        sendJSON({ type: 'barge_in' });
      }

      sendJSON({
        type: 'text_input',
        text: textInput,
        history: conversationHistory,
      });

      setLiveTranscript(textInput);
      setStreamingResponse('');
      setAgentState(STATES.THINKING);
      setTextInput('');
    },
    [textInput, isConnected, sendJSON, conversationHistory]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="app">
      {/* --- Header --- */}
      <header className="header">
        <h1 className="header__title">miniVoxSetu</h1>
        <p className="header__subtitle">
          Voice AI Pipeline — Deepgram STT → Groq LLM → Deepgram Aura TTS
        </p>
      </header>

      {/* --- Connection Status --- */}
      <div
        className={`connection-bar ${
          isConnected ? 'connection-bar--connected' : 'connection-bar--disconnected'
        }`}
      >
        <span className="connection-dot" />
        {isConnected ? 'Backend Connected' : 'Connecting to backend...'}
      </div>

      {/* --- Mic Button + State Label --- */}
      <div className="mic-area">
        <button
          id="mic-button"
          className={`mic-button ${
            agentState === STATES.LISTENING
              ? 'mic-button--active'
              : agentState === STATES.SPEAKING
              ? 'mic-button--speaking'
              : ''
          }`}
          onClick={handleMicClick}
          disabled={!isConnected || agentState === STATES.THINKING}
          aria-label={
            agentState === STATES.IDLE
              ? 'Start listening'
              : agentState === STATES.LISTENING
              ? 'Stop listening'
              : agentState === STATES.SPEAKING
              ? 'Interrupt (barge-in)'
              : 'Processing...'
          }
        >
          <MicIcon />
        </button>

        <div className={`state-label state-label--${agentState}`} id="state-label">
          {agentState === STATES.THINKING && (
            <span className="thinking-dots">
              <span />
              <span />
              <span />
            </span>
          )}
          {agentState === STATES.IDLE
            ? 'Click mic to start'
            : agentState === STATES.LISTENING
            ? 'Listening (always-on mic)'
            : STATE_LABELS[agentState]}
        </div>

        {/* --- Error Message --- */}
        {sttError && (
          <div className="stt-error" id="stt-error">
            <span className="stt-error__icon">⚠</span>
            <span className="stt-error__text">{sttError}</span>
            <button
              className="stt-error__dismiss"
              onClick={() => setSttError('')}
              aria-label="Dismiss error"
            >
              ✕
            </button>
          </div>
        )}

        {/* --- Text Input Fallback --- */}
        <form className="text-input-form" onSubmit={handleTextSubmit} id="text-input-form">
          <input
            type="text"
            className="text-input"
            id="text-input"
            placeholder="Or type a message..."
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            disabled={!isConnected || agentState === STATES.THINKING}
          />
          <button
            type="submit"
            className="text-input-send"
            id="text-input-send"
            disabled={!textInput.trim() || !isConnected || agentState === STATES.THINKING}
            aria-label="Send message"
          >
            <SendIcon />
          </button>
        </form>
      </div>

      {/* --- Live Transcript Panel --- */}
      <div className="panel" id="transcript-panel">
        <div className="panel__header">
          <span className="panel__title">Live Transcript</span>
          <span className="panel__badge">
            {agentState === STATES.LISTENING ? '● REC' : '○ OFF'}
          </span>
        </div>
        <div className="panel__body">
          {liveTranscript || interimTranscript || streamingResponse ? (
            <div className="transcript">
              {liveTranscript && <span>{liveTranscript} </span>}
              {interimTranscript && (
                <span className="transcript__interim">{interimTranscript}</span>
              )}
              {streamingResponse && (
                <>
                  {liveTranscript && <br />}
                  <span className="transcript__streaming">
                    {streamingResponse}
                    <span className="transcript__cursor" />
                  </span>
                </>
              )}
            </div>
          ) : (
            <p className="transcript transcript--empty">
              Click the mic to start speaking...
            </p>
          )}
        </div>
      </div>

      {/* --- Semantic Intelligence Panel --- */}
      <div className="panel" id="semantic-panel" style={{ borderLeft: semanticData?.escalation_recommended ? '3px solid #ef4444' : '3px solid #6366f1' }}>
        <div className="panel__header">
          <span className="panel__title">Semantic Intelligence</span>
          <span className="panel__badge">
            {semanticLatency > 0 ? `${semanticLatency}ms` : 'waiting'}
          </span>
        </div>
        <div className="panel__body">
          {semanticData ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.85rem' }}>
              {/* Intent */}
              <div style={{ padding: '6px 10px', background: 'rgba(99,102,241,0.15)', borderRadius: '6px' }}>
                <div style={{ color: '#a5b4fc', fontSize: '0.7rem', textTransform: 'uppercase', marginBottom: '2px' }}>Intent</div>
                <div style={{ color: '#e0e7ff', fontWeight: 600 }}>{semanticData.intent}</div>
              </div>

              {/* Sentiment */}
              <div style={{ padding: '6px 10px', background: 'rgba(99,102,241,0.15)', borderRadius: '6px' }}>
                <div style={{ color: '#a5b4fc', fontSize: '0.7rem', textTransform: 'uppercase', marginBottom: '2px' }}>Sentiment</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <div style={{
                    width: '60px', height: '6px', background: '#334155', borderRadius: '3px', overflow: 'hidden'
                  }}>
                    <div style={{
                      width: `${Math.abs(semanticData.sentiment) * 100}%`,
                      height: '100%',
                      background: semanticData.sentiment >= 0 ? '#22c55e' : '#ef4444',
                      borderRadius: '3px',
                      marginLeft: semanticData.sentiment >= 0 ? '50%' : `${50 - Math.abs(semanticData.sentiment) * 50}%`,
                    }} />
                  </div>
                  <span style={{ color: semanticData.sentiment >= 0 ? '#22c55e' : '#ef4444', fontWeight: 600 }}>
                    {semanticData.sentiment > 0 ? '+' : ''}{semanticData.sentiment}
                  </span>
                </div>
              </div>

              {/* Urgency */}
              <div style={{ padding: '6px 10px', background: 'rgba(99,102,241,0.15)', borderRadius: '6px' }}>
                <div style={{ color: '#a5b4fc', fontSize: '0.7rem', textTransform: 'uppercase', marginBottom: '2px' }}>Urgency</div>
                <div style={{
                  color: semanticData.urgency_level === 'critical' ? '#ef4444'
                    : semanticData.urgency_level === 'high' ? '#f59e0b'
                    : semanticData.urgency_level === 'medium' ? '#3b82f6'
                    : '#22c55e',
                  fontWeight: 600, textTransform: 'uppercase',
                }}>
                  {semanticData.urgency_level}
                </div>
              </div>

              {/* Tone */}
              <div style={{ padding: '6px 10px', background: 'rgba(99,102,241,0.15)', borderRadius: '6px' }}>
                <div style={{ color: '#a5b4fc', fontSize: '0.7rem', textTransform: 'uppercase', marginBottom: '2px' }}>Recommended Tone</div>
                <div style={{ color: '#e0e7ff', fontWeight: 600 }}>{semanticData.recommended_tone}</div>
              </div>

              {/* Summary — full width */}
              <div style={{ gridColumn: '1 / -1', padding: '6px 10px', background: 'rgba(99,102,241,0.1)', borderRadius: '6px' }}>
                <div style={{ color: '#a5b4fc', fontSize: '0.7rem', textTransform: 'uppercase', marginBottom: '2px' }}>Summary</div>
                <div style={{ color: '#cbd5e1' }}>{semanticData.one_line_summary}</div>
              </div>

              {/* Flags */}
              <div style={{ gridColumn: '1 / -1', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {semanticData.compliance_flag && (
                  <span style={{ padding: '2px 8px', background: '#7c3aed', borderRadius: '4px', fontSize: '0.75rem', color: '#e0e7ff' }}>
                    🔒 Compliance Flag
                  </span>
                )}
                {semanticData.escalation_recommended && (
                  <span style={{ padding: '2px 8px', background: '#ef4444', borderRadius: '4px', fontSize: '0.75rem', color: '#fff' }}>
                    ⚠ Escalation Recommended
                  </span>
                )}
              </div>
            </div>
          ) : (
            <p className="rag-empty">Semantic analysis will appear here after each utterance</p>
          )}
        </div>
      </div>

      {/* --- Acoustic Intelligence Panel --- */}
      <div className="panel" id="acoustic-panel" style={{ borderLeft: acousticData?.stress_score > 0.6 ? '3px solid #ef4444' : '3px solid #22c55e' }}>
        <div className="panel__header">
          <span className="panel__title">Acoustic Intelligence</span>
          <span className="panel__badge">
            {acousticLatency > 0 ? `${acousticLatency}ms` : 'waiting'}
          </span>
        </div>
        <div className="panel__body">
          {acousticData ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.85rem' }}>
              {/* Emotion */}
              <div style={{ padding: '6px 10px', background: 'rgba(34,197,94,0.12)', borderRadius: '6px' }}>
                <div style={{ color: '#86efac', fontSize: '0.7rem', textTransform: 'uppercase', marginBottom: '2px' }}>Emotion</div>
                <div style={{
                  color: acousticData.emotion === 'angry' ? '#ef4444'
                    : acousticData.emotion === 'sad' ? '#3b82f6'
                    : acousticData.emotion === 'happy' ? '#22c55e'
                    : '#94a3b8',
                  fontWeight: 700, fontSize: '1rem', textTransform: 'uppercase'
                }}>
                  {acousticData.emotion}
                  <span style={{ fontSize: '0.75rem', fontWeight: 400, marginLeft: '6px', opacity: 0.8 }}>
                    {(acousticData.emotion_confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Stress Score */}
              <div style={{ padding: '6px 10px', background: 'rgba(34,197,94,0.12)', borderRadius: '6px' }}>
                <div style={{ color: '#86efac', fontSize: '0.7rem', textTransform: 'uppercase', marginBottom: '4px' }}>Stress Score</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ flex: 1, height: '8px', background: '#1e293b', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${(acousticData.stress_score || 0) * 100}%`,
                      height: '100%',
                      borderRadius: '4px',
                      background: acousticData.stress_score > 0.6 ? '#ef4444'
                        : acousticData.stress_score > 0.3 ? '#f59e0b' : '#22c55e',
                      transition: 'width 0.3s ease',
                    }} />
                  </div>
                  <span style={{
                    color: acousticData.stress_score > 0.6 ? '#ef4444'
                      : acousticData.stress_score > 0.3 ? '#f59e0b' : '#22c55e',
                    fontWeight: 600, minWidth: '36px'
                  }}>
                    {(acousticData.stress_score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Pitch */}
              <div style={{ padding: '6px 10px', background: 'rgba(34,197,94,0.12)', borderRadius: '6px' }}>
                <div style={{ color: '#86efac', fontSize: '0.7rem', textTransform: 'uppercase', marginBottom: '2px' }}>Pitch (F0)</div>
                <div style={{ color: '#e0e7ff', fontWeight: 600 }}>
                  {acousticData.pitch_hz > 0 ? `${acousticData.pitch_hz.toFixed(0)} Hz` : '—'}
                  <span style={{ fontSize: '0.7rem', marginLeft: '6px', color: '#94a3b8' }}>
                    {acousticData.pitch_hz > 250 ? '↑ elevated' : acousticData.pitch_hz > 0 ? '→ normal' : ''}
                  </span>
                </div>
              </div>

              {/* Volume */}
              <div style={{ padding: '6px 10px', background: 'rgba(34,197,94,0.12)', borderRadius: '6px' }}>
                <div style={{ color: '#86efac', fontSize: '0.7rem', textTransform: 'uppercase', marginBottom: '4px' }}>Volume</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ flex: 1, height: '6px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${Math.min(100, Math.max(0, ((acousticData.rms_db || -60) + 60) / 60 * 100))}%`,
                      height: '100%',
                      borderRadius: '3px',
                      background: '#3b82f6',
                      transition: 'width 0.3s ease',
                    }} />
                  </div>
                  <span style={{ color: '#94a3b8', fontSize: '0.8rem', minWidth: '48px' }}>
                    {acousticData.rms_db?.toFixed(1)} dB
                  </span>
                </div>
              </div>

              {/* Speech + Interrupted flags */}
              <div style={{ gridColumn: '1 / -1', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <span style={{
                  padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem',
                  background: acousticData.is_speech ? '#166534' : '#1e293b',
                  color: acousticData.is_speech ? '#86efac' : '#64748b',
                }}>
                  {acousticData.is_speech ? '● Speech' : '○ Silence'}
                </span>
                {acousticData.interrupted && (
                  <span style={{ padding: '2px 8px', background: '#ef4444', borderRadius: '4px', fontSize: '0.75rem', color: '#fff' }}>
                    ⚡ Barge-in Captured
                  </span>
                )}
              </div>
            </div>
          ) : (
            <p className="rag-empty">Acoustic analysis updates every 1.5s while mic is active</p>
          )}
        </div>
      </div>

      {/* --- Two-Column Grid: History + RAG --- */}
      <div className="panels-grid">
        {/* --- Conversation History Panel --- */}
        <div className="panel" id="history-panel">
          <div className="panel__header">
            <span className="panel__title">Context Window</span>
            <span className="panel__badge">{conversationHistory.length} turns</span>
          </div>
          <div className="panel__body" ref={historyPanelRef}>
            {conversationHistory.length === 0 ? (
              <p className="history-empty">
                Conversation history will appear here
              </p>
            ) : (
              <div className="history-list">
                {conversationHistory.map((turn, idx) => (
                  <div className="history-turn" key={idx}>
                    <div
                      className={`history-turn__avatar history-turn__avatar--${
                        turn.role === 'user' ? 'user' : 'ai'
                      }`}
                    >
                      {turn.role === 'user' ? 'U' : 'AI'}
                    </div>
                    <div className="history-turn__content">
                      <div
                        className={`history-turn__role history-turn__role--${
                          turn.role === 'user' ? 'user' : 'ai'
                        }`}
                      >
                        {turn.role === 'user' ? 'User' : 'Assistant'}
                      </div>
                      <div className="history-turn__text">{turn.content}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* --- RAG Context Panel --- */}
        <div className="panel" id="rag-panel">
          <div className="panel__header">
            <span className="panel__title">RAG Retrieved</span>
            <span className="panel__badge">{ragChunks.length} chunks</span>
          </div>
          <div className="panel__body">
            {ragChunks.length === 0 ? (
              <p className="rag-empty">
                Retrieved FAQ chunks will appear here after each query
              </p>
            ) : (
              <>
                <div className="rag-query">
                  Query: <span>"{ragQuery}"</span>
                </div>
                {ragChunks.map((chunk, idx) => (
                  <div className="rag-chunk" key={idx}>
                    {chunk}
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      </div>

      {/* --- Post-Call Report Modal --- */}
      {(reportLoading || reportData) && (
        <div className="report-modal-overlay">
          <div className="report-modal">
            <div className="report-modal__header">
              <h2>Post-Call Report</h2>
              {reportData && (
                <button className="report-modal__close" onClick={() => setReportData(null)}>
                  ✕
                </button>
              )}
            </div>
            <div className="report-modal__body">
              {reportLoading ? (
                <div className="report-loading">
                  <span className="thinking-dots"><span /><span /><span /></span>
                  <p>Generating QA Report with Gemini...</p>
                </div>
              ) : (
                <div className="report-content markdown-body">
                  {reportData.split('\n').map((line, i) => {
                    // Simple markdown-like rendering without XSS risk
                    if (line.startsWith('# ')) return <h3 key={i}>{line.slice(2)}</h3>;
                    if (line.startsWith('## ')) return <h4 key={i}>{line.slice(3)}</h4>;
                    if (line.startsWith('### ')) return <h5 key={i}>{line.slice(4)}</h5>;
                    if (line.startsWith('- ')) return <li key={i}>{line.slice(2)}</li>;
                    if (line.trim() === '') return <br key={i} />;
                    return <p key={i}>{line}</p>;
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

// ============================================================
// ICONS
// ============================================================

function MicIcon() {
  return (
    <svg className="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}
