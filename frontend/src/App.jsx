/**
 * miniVoxSetu — Phase 1 Upgrade
 * STT: Deepgram (server-side) via streaming audio
 * TTS: ElevenLabs (server-side) via audio playback
 * Mic: Always-on after start (no click-per-message)
 * Barge-in: Energy-based detection cancels TTS playback
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
  // --- Connection ---
  const { isConnected, sendJSON, sendBinary, setOnMessage, wsRef } =
    useWebSocket('ws://localhost:8000/ws/chat');

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

  // --- Text input fallback ---
  const [textInput, setTextInput] = useState('');

  // --- Error ---
  const [sttError, setSttError] = useState('');

  // --- Audio recording ---
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const isRecordingRef = useRef(false);

  // --- Audio playback ---
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const currentAudioSourceRef = useRef(null);
  const audioContextRef = useRef(null);

  // --- VAD / Barge-in ---
  const analyserRef = useRef(null);
  const vadIntervalRef = useRef(null);

  // --- Refs ---
  const historyPanelRef = useRef(null);
  const pendingUtteranceRef = useRef('');

  // Sync state to ref for use in callbacks
  useEffect(() => {
    agentStateRef.current = agentState;
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
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      analyserRef.current = analyser;

      // MediaRecorder sends audio chunks to backend
      const recorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0 && isRecordingRef.current) {
          sendBinary(e.data);
        }
      };

      recorder.start(250); // 250ms chunks
      mediaRecorderRef.current = recorder;
      isRecordingRef.current = true;

      // Start barge-in energy monitoring
      startBargeInDetection();

      setAgentState(STATES.LISTENING);
      setSttError('');
    } catch (err) {
      console.error('Mic access failed:', err);
      setSttError('Microphone access denied. Please allow mic access.');
    }
  }, [sendBinary]);

  const stopRecording = useCallback(() => {
    isRecordingRef.current = false;

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }

    if (vadIntervalRef.current) {
      clearInterval(vadIntervalRef.current);
      vadIntervalRef.current = null;
    }

    setAgentState(STATES.IDLE);
  }, []);

  // ============================================================
  // BARGE-IN DETECTION (energy threshold on AnalyserNode)
  // ============================================================

  const startBargeInDetection = useCallback(() => {
    if (vadIntervalRef.current) clearInterval(vadIntervalRef.current);

    const ENERGY_THRESHOLD = 30;

    vadIntervalRef.current = setInterval(() => {
      if (!analyserRef.current) return;
      if (agentStateRef.current !== STATES.SPEAKING) return;

      const data = new Uint8Array(analyserRef.current.frequencyBinCount);
      analyserRef.current.getByteFrequencyData(data);
      const avg = data.reduce((a, b) => a + b, 0) / data.length;

      if (avg > ENERGY_THRESHOLD) {
        handleBargeIn();
      }
    }, 100);
  }, []);

  // ============================================================
  // BARGE-IN HANDLER
  // ============================================================

  const handleBargeIn = useCallback(() => {
    // Stop TTS audio playback
    if (currentAudioSourceRef.current) {
      try {
        currentAudioSourceRef.current.stop();
      } catch (e) { /* already stopped */ }
      currentAudioSourceRef.current = null;
    }
    audioQueueRef.current = [];
    isPlayingRef.current = false;

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
      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);

      currentAudioSourceRef.current = source;

      source.onended = () => {
        currentAudioSourceRef.current = null;
        isPlayingRef.current = false;
        // Play next queued audio or return to listening
        if (audioQueueRef.current.length > 0) {
          playNextAudio();
        }
      };

      source.start(0);
    } catch (err) {
      console.error('Audio playback error:', err);
      isPlayingRef.current = false;
      // Try next in queue
      if (audioQueueRef.current.length > 0) {
        playNextAudio();
      }
    }
  }, []);

  const queueAudio = useCallback((base64Data) => {
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
            setLiveTranscript(pendingUtteranceRef.current || '');
            pendingUtteranceRef.current = '';
          } else if (message.state === 'SPEAKING') {
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
          // TTS audio from ElevenLabs (base64 mp3)
          queueAudio(message.data);
          break;

        case 'done': {
          const fullText = message.full_text || '';
          setConversationHistory((prev) => {
            const userText = pendingUtteranceRef.current || liveTranscript || '';
            const newHistory = [...prev];
            if (userText) {
              newHistory.push({ role: 'user', content: userText });
            }
            newHistory.push({ role: 'assistant', content: fullText });
            return newHistory;
          });
          setStreamingResponse('');
          // State will be set to LISTENING by backend 'state' message
          // or when all audio finishes playing
          break;
        }

        case 'semantic':
          // Semantic intelligence from parallel analysis
          setSemanticData(message.data);
          setSemanticLatency(message.latency_ms || 0);
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
  }, [setOnMessage, queueAudio, liveTranscript]);

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
          Voice AI Pipeline — Deepgram STT → Gemini LLM → ElevenLabs TTS
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
