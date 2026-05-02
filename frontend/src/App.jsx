/**
 * miniVoxSetu — Main Application Component
 *
 * WHY THIS IS ONE BIG FILE INSTEAD OF MANY SMALL COMPONENTS:
 * This is a learning project. Splitting into 10 components would make you
 * jump between files to understand the flow. Voice AI has a LINEAR pipeline
 * (mic → STT → LLM → TTS → speaker) and keeping it in one file lets you
 * read that pipeline top to bottom. In production, you'd absolutely split this.
 */

import { useState, useRef, useCallback, useEffect } from 'react';

// ============================================================
// CONSTANTS
// ============================================================

/**
 * WHY THESE STATES EXIST:
 * Voice AI agents are state machines. At any moment, the system is in exactly
 * one of these states. The transitions between states define the conversation flow:
 *   IDLE → LISTENING → THINKING → SPEAKING → IDLE (or back to LISTENING on barge-in)
 * Understanding this state machine is key to understanding ALL voice AI systems.
 */
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
// CUSTOM HOOK: useWebSocket
// ============================================================

/**
 * WHY A CUSTOM HOOK FOR WEBSOCKET:
 * The WebSocket connection is a long-lived resource that needs lifecycle
 * management (connect, reconnect, cleanup). Encapsulating it in a hook
 * keeps the main component focused on UI logic.
 */
function useWebSocket(url) {
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const onMessageRef = useRef(null);

  const connect = useCallback(() => {
    // WHY: We check readyState to avoid creating duplicate connections.
    // WebSockets are expensive resources — one connection per client is the rule.
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);

    ws.onopen = () => {
      console.log('🔌 WebSocket connected');
      setIsConnected(true);
    };

    ws.onclose = () => {
      console.log('🔌 WebSocket disconnected');
      setIsConnected(false);
      // WHY: Auto-reconnect after 2 seconds. Network interruptions are common,
      // and a voice AI agent that stops working after a blip is frustrating.
      // In production, you'd use exponential backoff (2s, 4s, 8s...).
      setTimeout(connect, 2000);
    };

    ws.onerror = (err) => {
      console.error('❌ WebSocket error:', err);
      ws.close();
    };

    ws.onmessage = (event) => {
      if (onMessageRef.current) {
        onMessageRef.current(JSON.parse(event.data));
      }
    };

    wsRef.current = ws;
  }, [url]);

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const setOnMessage = useCallback((handler) => {
    onMessageRef.current = handler;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      // WHY: Cleanup on unmount prevents memory leaks and zombie connections.
      if (wsRef.current) {
        wsRef.current.onclose = null; // Prevent reconnect on intentional close
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { isConnected, send, setOnMessage };
}

// ============================================================
// CUSTOM HOOK: useSpeechRecognition (STT)
// ============================================================

/**
 * WHY WE USE THE BROWSER'S WEB SPEECH API FOR STT:
 * 1. It's completely FREE — no API key, no usage limits
 * 2. It runs locally in the browser — no audio sent to our server
 * 3. It provides real-time interim results (partial transcripts as you speak)
 *
 * TRADEOFF: Quality is lower than cloud STT (Google Cloud Speech, Deepgram,
 * AssemblyAI). In production voice AI, you'd use a paid cloud STT service
 * for accuracy. But for learning the architecture, this is perfect.
 */
function useSpeechRecognition() {
  const recognitionRef = useRef(null);
  const [isSupported, setIsSupported] = useState(true);

  useEffect(() => {
    // WHY: We check for browser support because Web Speech API isn't available
    // everywhere (notably Firefox has limited support). Chrome is the gold standard.
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setIsSupported(false);
      return;
    }

    const recognition = new SpeechRecognition();

    // WHY continuous=true: We want the mic to keep listening until WE stop it,
    // not stop after one sentence. This is essential for natural conversation.
    recognition.continuous = true;

    // WHY interimResults=true: This gives us partial transcripts as the user
    // speaks ("I want to..." → "I want to know..." → "I want to know my balance").
    // This is what makes voice AI feel responsive — you see words appearing in
    // real time. Without this, you'd wait for silence before seeing any text.
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognitionRef.current = recognition;
  }, []);

  const start = useCallback((onResult, onEnd) => {
    const recognition = recognitionRef.current;
    if (!recognition) return;

    recognition.onresult = (event) => {
      // WHY: We process ALL results, not just the latest one. The Speech API
      // continuously refines its transcription — what it thought was "I wanna"
      // might become "I want to" as more audio context arrives.
      let interim = '';
      let final = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += transcript;
        } else {
          interim += transcript;
        }
      }

      onResult({ interim, final });
    };

    recognition.onerror = (event) => {
      // WHY: 'no-speech' and 'aborted' are normal — they happen when the user
      // is silent or when we programmatically stop recognition. We don't want
      // to treat these as real errors.
      if (event.error !== 'no-speech' && event.error !== 'aborted') {
        console.error('🎤 Speech recognition error:', event.error);
      }
    };

    recognition.onend = () => {
      if (onEnd) onEnd();
    };

    try {
      recognition.start();
    } catch (e) {
      // WHY: start() throws if recognition is already running. This can happen
      // during rapid state transitions (barge-in). Catching it prevents crashes.
      console.warn('🎤 Recognition already started');
    }
  }, []);

  const stop = useCallback(() => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {
        // Safe to ignore — means it wasn't running
      }
    }
  }, []);

  return { start, stop, isSupported };
}

// ============================================================
// CUSTOM HOOK: useSpeechSynthesis (TTS)
// ============================================================

/**
 * WHY WE USE THE BROWSER'S WEB SPEECH SYNTHESIS API FOR TTS:
 * Same reasons as STT — it's free and runs locally. The quality isn't
 * as good as cloud TTS (ElevenLabs, Google Cloud TTS, Amazon Polly),
 * but it's instant (no network latency) and costs nothing.
 *
 * CRITICAL FOR BARGE-IN: We need the ability to CANCEL speech mid-sentence.
 * The Web Speech Synthesis API supports this via speechSynthesis.cancel().
 * This is what makes barge-in possible — when the user interrupts, we kill
 * TTS immediately and switch back to listening.
 */
function useSpeechSynthesis() {
  // WHY: getVoices() returns an empty array on first call in many browsers.
  // Voices are loaded asynchronously, so we cache them via the 'voiceschanged'
  // event. Without this, the first TTS call may use the robotic default voice.
  const voicesRef = useRef([]);

  useEffect(() => {
    const loadVoices = () => {
      voicesRef.current = window.speechSynthesis.getVoices();
    };
    loadVoices();
    window.speechSynthesis.addEventListener('voiceschanged', loadVoices);
    return () => {
      window.speechSynthesis.removeEventListener('voiceschanged', loadVoices);
    };
  }, []);

  const speak = useCallback((text, onEnd) => {
    // WHY: We cancel any ongoing speech before starting new speech.
    // This prevents overlapping utterances and ensures clean transitions.
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05; // WHY: Slightly faster than default feels more natural for AI
    utterance.pitch = 1.0;

    // WHY: We try to pick a good voice. The default voice on many systems
    // sounds robotic. Selecting a specific voice (like "Google UK English Female")
    // dramatically improves the experience.
    const voices = voicesRef.current;
    const preferred = voices.find(v =>
      v.name.includes('Google') && v.lang.startsWith('en')
    ) || voices.find(v => v.lang.startsWith('en'));
    if (preferred) utterance.voice = preferred;

    utterance.onend = () => {
      if (onEnd) onEnd();
    };

    window.speechSynthesis.speak(utterance);
  }, []);

  /**
   * WHY THIS CANCEL FUNCTION IS CRITICAL:
   * In real voice AI, barge-in means the user starts talking while the AI
   * is still speaking. The AI must IMMEDIATELY shut up and listen.
   * "Immediately" means < 200ms — any longer and the user feels ignored.
   * speechSynthesis.cancel() is synchronous, so it's instant.
   */
  const cancel = useCallback(() => {
    window.speechSynthesis.cancel();
  }, []);

  const isSpeaking = useCallback(() => {
    return window.speechSynthesis.speaking;
  }, []);

  return { speak, cancel, isSpeaking };
}

// ============================================================
// MAIN APP COMPONENT
// ============================================================

export default function App() {
  // --- Core state ---
  const [agentState, setAgentState] = useState(STATES.IDLE);
  /**
   * WHY WE KEEP CONVERSATION HISTORY IN STATE:
   * This array IS the AI's memory. Every time we call the LLM, we send this
   * entire array. The LLM has no memory between calls — this array is how it
   * knows what was said before. This is the "context window" concept.
   * We display it in the UI so you can literally SEE the memory being built.
   */
  const [conversationHistory, setConversationHistory] = useState([]);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [streamingResponse, setStreamingResponse] = useState('');
  const [ragChunks, setRagChunks] = useState([]);
  const [ragQuery, setRagQuery] = useState('');

  // --- Refs for values needed in callbacks ---
  /**
   * WHY REFS INSTEAD OF STATE FOR SOME VALUES:
   * Callbacks (WebSocket handlers, speech recognition handlers) capture
   * state values at creation time (closure). Using refs gives us access to
   * the CURRENT value, not the stale captured value. This is a common React
   * pattern for values that change frequently and are read in async callbacks.
   */
  const agentStateRef = useRef(STATES.IDLE);
  const accumulatedTextRef = useRef('');
  const fullResponseRef = useRef('');
  const historyRef = useRef([]);
  const historyPanelRef = useRef(null);

  // Keep refs in sync with state
  useEffect(() => { agentStateRef.current = agentState; }, [agentState]);
  useEffect(() => { historyRef.current = conversationHistory; }, [conversationHistory]);

  // --- Initialize hooks ---
  // WHY: We use the current page's host so the WebSocket goes through Vite's
  // proxy (configured in vite.config.js). Hardcoding port 8000 bypasses the
  // proxy and would break in production behind a reverse proxy.
  const { isConnected, send, setOnMessage } = useWebSocket(
    `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/chat`
  );
  const { start: startSTT, stop: stopSTT, isSupported: sttSupported } = useSpeechRecognition();
  const { speak, cancel: cancelTTS } = useSpeechSynthesis();

  // Auto-scroll history panel when new turns are added
  useEffect(() => {
    if (historyPanelRef.current) {
      historyPanelRef.current.scrollTop = historyPanelRef.current.scrollHeight;
    }
  }, [conversationHistory, streamingResponse]);

  // WHY: If the WebSocket disconnects mid-conversation (server restart, network
  // issue), the UI gets stuck in THINKING or SPEAKING forever because no more
  // messages will arrive. This safety net resets the state machine.
  useEffect(() => {
    if (!isConnected && (agentState === STATES.THINKING || agentState === STATES.SPEAKING)) {
      cancelTTS();
      setAgentState(STATES.IDLE);
      setStreamingResponse('');
      setLiveTranscript('');
    }
  }, [isConnected, agentState, cancelTTS]);

  // WHY: Cleanup VAD resources on unmount to prevent memory leaks.
  // MediaStreams and AudioContexts are expensive system resources.
  useEffect(() => {
    return () => {
      if (vadIntervalRef.current) clearInterval(vadIntervalRef.current);
      if (mediaStreamRef.current) mediaStreamRef.current.getTracks().forEach(t => t.stop());
      if (audioContextRef.current) audioContextRef.current.close();
    };
  }, []);

  // ============================================================
  // BARGE-IN DETECTION (VAD — Voice Activity Detection)
  // ============================================================

  /**
   * WHY BARGE-IN MATTERS:
   * In natural human conversation, people interrupt each other constantly.
   * If the AI can't handle interruption, it feels like talking to a machine.
   * Barge-in means: if the user starts speaking while the AI is talking,
   * IMMEDIATELY stop the AI's speech and start listening to the user.
   *
   * HOW WE DETECT IT:
   * We use audio energy detection — we capture the microphone stream,
   * analyze the audio levels, and if the energy exceeds a threshold while
   * the AI is speaking, we trigger barge-in. This is a simplified version
   * of what production systems use (they use ML-based VAD models like Silero).
   */
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const vadIntervalRef = useRef(null);

  const startVAD = useCallback(async () => {
    // WHY: Guard against creating duplicate AudioContexts from rapid mic clicks.
    // Each AudioContext consumes system resources, and browsers limit the count.
    if (analyserRef.current && audioContextRef.current?.state !== 'closed') return;

    try {
      // WHY: getUserMedia requests microphone access. This is the WebRTC API
      // that all voice AI systems use to capture audio from the user's mic.
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      // WHY fftSize=512: Smaller FFT = faster updates but less frequency resolution.
      // For VAD we only care about energy level, not frequency detail, so 512 is fine.
      analyser.fftSize = 512;
      source.connect(analyser);
      analyserRef.current = analyser;
    } catch (err) {
      console.error('❌ Microphone access denied:', err);
    }
  }, []);

  const stopVAD = useCallback(() => {
    if (vadIntervalRef.current) {
      clearInterval(vadIntervalRef.current);
      vadIntervalRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(t => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  }, []);

  /**
   * WHY WE MONITOR AUDIO ENERGY DURING SPEAKING STATE:
   * We only run the VAD check while the AI is speaking. If the audio energy
   * exceeds our threshold, it means the user is talking → trigger barge-in.
   * We poll every 100ms which gives us ~100ms detection latency — well under
   * the 200ms threshold that feels responsive to humans.
   */
  const startBargeInDetection = useCallback(() => {
    if (!analyserRef.current) return;

    // WHY: Clear any existing interval to prevent accumulation from repeated
    // mic clicks. Without this, each click adds a NEW interval — after 5 clicks
    // you'd have 5 intervals all checking audio energy simultaneously.
    if (vadIntervalRef.current) {
      clearInterval(vadIntervalRef.current);
      vadIntervalRef.current = null;
    }

    const analyser = analyserRef.current;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    vadIntervalRef.current = setInterval(() => {
      // WHY: We only check for barge-in when the AI is speaking.
      // Checking at other times would cause false triggers.
      if (agentStateRef.current !== STATES.SPEAKING) return;

      analyser.getByteFrequencyData(dataArray);
      // WHY: Average energy across frequency bins gives us a simple
      // "how loud is the input" measure. This is crude but effective.
      const average = dataArray.reduce((sum, val) => sum + val, 0) / dataArray.length;

      // WHY threshold=30: This is tuned to ignore background noise but
      // catch speech. You may need to adjust for your environment.
      // In production, you'd use a proper ML-based VAD (Silero VAD).
      if (average > 30) {
        console.log('🛑 BARGE-IN detected! Energy:', average.toFixed(1));
        handleBargeIn();
      }
    }, 100);
  }, []);

  // ============================================================
  // CORE VOICE AI PIPELINE
  // ============================================================

  /**
   * WHY THIS IS THE MOST IMPORTANT FUNCTION:
   * This is the barge-in handler — when the user interrupts the AI.
   * It must:
   * 1. Cancel TTS immediately (< 200ms latency requirement)
   * 2. Switch state to LISTENING
   * 3. Restart speech recognition
   * Speed is everything here — any delay and the user feels ignored.
   */
  const handleBargeIn = useCallback(() => {
    console.log('🛑 Executing barge-in');
    // WHY: cancel() is synchronous — TTS stops INSTANTLY.
    cancelTTS();
    setAgentState(STATES.LISTENING);
    setStreamingResponse('');
    fullResponseRef.current = '';
    // Restart listening immediately
    startListening();
  }, [cancelTTS]);

  /**
   * WHY: This starts the speech recognition and handles incoming transcripts.
   * We accumulate final transcripts (confirmed words) and show interim
   * transcripts (tentative words) separately for a real-time feel.
   */
  const startListening = useCallback(() => {
    setAgentState(STATES.LISTENING);
    accumulatedTextRef.current = '';
    setLiveTranscript('');
    setInterimTranscript('');

    startSTT(
      // onResult callback — called every time the Speech API has new text
      ({ interim, final: finalText }) => {
        if (finalText) {
          accumulatedTextRef.current += finalText;
          setLiveTranscript(accumulatedTextRef.current);
        }
        setInterimTranscript(interim);
      },
      // onEnd callback — called when speech recognition stops
      () => {
        // WHY: Speech recognition can stop on its own (timeout, browser decides
        // user stopped talking). If we were listening and got text, process it.
        // If no text, go back to idle.
        if (agentStateRef.current === STATES.LISTENING) {
          const text = accumulatedTextRef.current.trim();
          if (text) {
            processUserInput(text);
          }
        }
      }
    );
  }, [startSTT]);

  /**
   * WHY: When the user stops talking (clicks mic button again), we take
   * whatever they said, add it to conversation history, and send it to
   * the LLM. This is the STT → LLM handoff.
   */
  const stopListeningAndProcess = useCallback(() => {
    stopSTT();
    const text = accumulatedTextRef.current.trim();
    if (text) {
      processUserInput(text);
    } else {
      setAgentState(STATES.IDLE);
    }
  }, [stopSTT]);

  /**
   * WHY THIS FUNCTION IS THE CORE OF THE PIPELINE:
   * It takes the user's spoken text, adds it to conversation history,
   * and sends it (along with FULL history) to the backend via WebSocket.
   * The backend will:
   * 1. Search RAG for relevant FAQ chunks
   * 2. Call Gemini with the full history + RAG context
   * 3. Stream the response back token by token
   */
  const processUserInput = useCallback((text) => {
    setAgentState(STATES.THINKING);
    setInterimTranscript('');
    setStreamingResponse('');
    fullResponseRef.current = '';

    // WHY: We add the user's turn to history BEFORE sending to the LLM.
    // This ensures the history array always has the complete conversation
    // up to this point. The LLM needs this to maintain conversational context.
    const newHistory = [
      ...historyRef.current,
      { role: 'user', content: text }
    ];
    setConversationHistory(newHistory);

    // WHY: We send the FULL conversation history every time because the
    // LLM API is stateless — it has NO memory between calls. The history
    // array IS the model's memory. This is the "context window" concept
    // that's fundamental to all LLM-powered agents.
    send({
      text,
      history: historyRef.current, // Send previous history (without current user turn, it's in 'text')
    });
  }, [send]);

  // ============================================================
  // WEBSOCKET MESSAGE HANDLER
  // ============================================================

  /**
   * WHY: This handles messages streaming back from the backend.
   * The backend sends three types of messages:
   * 1. "rag_context" — what FAQ chunks were retrieved (for transparency)
   * 2. "chunk" — a piece of the LLM response (streaming)
   * 3. "done" — the complete response is finished
   * 4. "error" — something went wrong
   */
  useEffect(() => {
    setOnMessage((message) => {
      switch (message.type) {
        case 'rag_context':
          // WHY: We show RAG results in the UI so you can SEE what the
          // retrieval system found. This is crucial for debugging RAG quality
          // in production — if the wrong chunks are retrieved, the AI will
          // give wrong answers. Transparency > magic.
          setRagChunks(message.chunks || []);
          setRagQuery(message.query || '');
          break;

        case 'chunk':
          // WHY: Each chunk is appended to build the response incrementally.
          // We show this in the UI as "streaming text" — you see the AI's
          // response being generated word by word, just like ChatGPT.
          fullResponseRef.current += message.text;
          setStreamingResponse(fullResponseRef.current);
          // Transition to SPEAKING on first chunk
          if (agentStateRef.current === STATES.THINKING) {
            setAgentState(STATES.SPEAKING);
          }
          break;

        case 'done': {
          // WHY: When the full response is complete, we:
          // 1. Add the assistant's turn to conversation history
          // 2. Speak the response via TTS
          // 3. The history now has the complete exchange for the next turn
          const fullText = message.full_text;

          setConversationHistory(prev => [
            ...prev,
            { role: 'assistant', content: fullText }
          ]);

          setAgentState(STATES.SPEAKING);

          // WHY: We start TTS with the complete response. When TTS finishes,
          // we transition back to IDLE. If the user barge-ins during TTS,
          // the barge-in handler will cancel TTS and go to LISTENING instead.
          speak(fullText, () => {
            // WHY: Check state because barge-in might have already changed it.
            // If we blindly set IDLE here, we'd override the LISTENING state
            // that barge-in set, breaking the flow.
            if (agentStateRef.current === STATES.SPEAKING) {
              setAgentState(STATES.IDLE);
              setStreamingResponse('');
              setLiveTranscript('');
            }
          });
          break;
        }

        case 'error':
          console.error('❌ Server error:', message.message);
          setAgentState(STATES.IDLE);
          setStreamingResponse(`Error: ${message.message}`);
          break;

        default:
          break;
      }
    });
  }, [setOnMessage, speak]);

  // ============================================================
  // MIC BUTTON HANDLER
  // ============================================================

  /**
   * WHY: The mic button is a toggle:
   * - If IDLE → start listening (and start VAD for barge-in detection)
   * - If LISTENING → stop listening and process the text
   * - If SPEAKING → manual barge-in (user clicks mic while AI talks)
   * - If THINKING → do nothing (wait for the LLM)
   */
  const handleMicClick = useCallback(async () => {
    switch (agentState) {
      case STATES.IDLE:
        await startVAD();
        startBargeInDetection();
        startListening();
        break;

      case STATES.LISTENING:
        stopListeningAndProcess();
        break;

      case STATES.SPEAKING:
        // WHY: Clicking mic while AI speaks is a manual barge-in.
        // Same effect as voice barge-in but triggered by button.
        handleBargeIn();
        break;

      case STATES.THINKING:
        // WHY: We don't allow interruption during thinking because
        // the LLM is mid-generation. We could cancel the request,
        // but that adds complexity without teaching new concepts.
        break;

      default:
        break;
    }
  }, [agentState, startVAD, startBargeInDetection, startListening, stopListeningAndProcess, handleBargeIn]);

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="app">
      {/* --- Header --- */}
      <header className="header">
        <h1 className="header__title">miniVoxSetu</h1>
        <p className="header__subtitle">
          Learn voice AI by building one — STT → LLM → TTS with RAG &amp; barge-in
        </p>
      </header>

      {/* --- Connection Status --- */}
      <div className={`connection-bar ${isConnected ? 'connection-bar--connected' : 'connection-bar--disconnected'}`}>
        <span className="connection-dot" />
        {isConnected ? 'Backend Connected' : 'Connecting to backend...'}
      </div>

      {/* --- Mic Button + State Label --- */}
      <div className="mic-area">
        <button
          id="mic-button"
          className={`mic-button ${agentState === STATES.LISTENING ? 'mic-button--active' : ''}`}
          onClick={handleMicClick}
          disabled={!isConnected || !sttSupported || agentState === STATES.THINKING}
          aria-label={agentState === STATES.LISTENING ? 'Stop listening' : 'Start listening'}
        >
          <MicIcon />
        </button>

        <div className={`state-label state-label--${agentState}`} id="state-label">
          {agentState === STATES.THINKING && (
            <span className="thinking-dots">
              <span /><span /><span />
            </span>
          )}
          {STATE_LABELS[agentState]}
        </div>

        {!sttSupported && (
          <p style={{ color: 'var(--accent-red)', fontSize: '0.8rem' }}>
            ⚠ Speech recognition not supported. Use Chrome or Edge.
          </p>
        )}
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
          {(liveTranscript || interimTranscript || streamingResponse) ? (
            <div className="transcript">
              {/* WHY: We show THREE things in the transcript panel:
                  1. Final transcript (confirmed user speech) — white
                  2. Interim transcript (tentative user speech) — gray italic
                  3. Streaming AI response — blue with cursor
                  This mirrors what production STT dashboards show. */}
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
              Click the mic and start speaking...
            </p>
          )}
        </div>
      </div>

      {/* --- Two-Column Grid: History + RAG --- */}
      <div className="panels-grid">

        {/* --- Conversation History Panel --- */}
        {/*
          WHY THIS PANEL EXISTS:
          This is the MOST educational part of the UI. It shows you the
          exact array that gets sent to the LLM on every turn. You can
          literally watch the context window being built turn by turn.
          This is how ALL LLM chat applications work — the entire conversation
          is passed to the model every time because it has no memory.
        */}
        <div className="panel" id="history-panel">
          <div className="panel__header">
            <span className="panel__title">Context Window</span>
            <span className="panel__badge">{conversationHistory.length} turns</span>
          </div>
          <div className="panel__body" ref={historyPanelRef}>
            {conversationHistory.length === 0 ? (
              <p className="history-empty">
                Conversation history will appear here — this is the array sent to the LLM every turn
              </p>
            ) : (
              <div className="history-list">
                {conversationHistory.map((turn, idx) => (
                  <div className="history-turn" key={idx}>
                    <div className={`history-turn__avatar history-turn__avatar--${turn.role === 'user' ? 'user' : 'ai'}`}>
                      {turn.role === 'user' ? 'U' : 'AI'}
                    </div>
                    <div className="history-turn__content">
                      <div className={`history-turn__role history-turn__role--${turn.role === 'user' ? 'user' : 'ai'}`}>
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
        {/*
          WHY THIS PANEL EXISTS:
          RAG is invisible by default — the user never sees what was retrieved.
          Making it visible teaches you:
          1. What the embedding search actually found
          2. Whether the retrieved chunks are relevant (debugging RAG quality)
          3. How the LLM's answer changes based on what context it receives
          In production, this data goes to monitoring dashboards, not the user.
        */}
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

/**
 * WHY INLINE SVG INSTEAD OF AN ICON LIBRARY:
 * We don't need 500 icons — just one mic icon. Adding react-icons or
 * similar would bloat the bundle for no reason. Inline SVG is the lightest
 * approach and gives us full control over size and color via CSS.
 */
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
