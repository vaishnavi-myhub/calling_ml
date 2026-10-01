import { useCallback, useRef, useState } from "react";
import pcmWorkletSource from "../lib/pcm-worklet.js?raw";

export type TranscriptMessage = { role: "user" | "assistant"; text: string; language?: string; languageConfidence?: number };
export type LiveCallStatus = "idle" | "connecting" | "listening" | "responding" | "error";

type ServerMessage =
  | { type: "ready" }
  | { type: "transcript_interim"; text: string; is_final: false; language?: string; language_confidence?: number }
  | { type: "transcript"; role: "user"; text: string; is_final: true; language?: string; language_confidence?: number }
  | { type: "assistant_text"; text: string; final: boolean }
  | { type: "audio"; seq: number; text: string; contentType: string }
  | { type: "interrupted" }
  | { type: "empty_transcript" }
  | { type: "error"; message: string };

const apiBase = import.meta.env["VITE_API_URL"] ?? "http://127.0.0.1:4000";
const wsBase = apiBase.replace(/^http/, "ws");

// Web Audio's playback destination follows whatever the OS/browser treats as its
// general default output -- separate from the mic device actually selected above, and
// this diverges in exactly the way reported live: a Bluetooth headset was the active
// mic, but TTS replies kept coming out of the laptop speakers anyway. That's a known
// Bluetooth quirk (a headset can expose separate call-quality and media-playback
// profiles, and Windows/the browser don't always pick the same one for both mic input
// and arbitrary Web Audio output). Routes playback to the same physical device as the
// mic (matched by groupId, which pairs a device's input and output halves) wherever the
// browser exposes AudioContext.setSinkId (Chrome/Edge 110+); silently does nothing
// everywhere else, so this never blocks a call from starting.
async function routePlaybackToMicDevice(audioContext: AudioContext, micStream: MediaStream): Promise<void> {
  const setSinkId = (audioContext as AudioContext & { setSinkId?: (id: string) => Promise<void> }).setSinkId;
  if (typeof setSinkId !== "function") return;
  try {
    const micDeviceId = micStream.getAudioTracks()[0]?.getSettings().deviceId;
    if (!micDeviceId) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const micDevice = devices.find((device) => device.kind === "audioinput" && device.deviceId === micDeviceId);
    if (!micDevice?.groupId) return;
    const matchingOutput = devices.find((device) => device.kind === "audiooutput" && device.groupId === micDevice.groupId);
    if (matchingOutput) await setSinkId.call(audioContext, matchingOutput.deviceId);
  } catch {
    // Best-effort only -- playback just stays on whatever the default output is.
  }
}

export function useLiveCall() {
  const [status, setStatus] = useState<LiveCallStatus>("idle");
  const [messages, setMessages] = useState<TranscriptMessage[]>([]);
  const [interimTranscript, setInterimTranscript] = useState("");
  const [error, setError] = useState("");
  // Real, measured mic input level (0-1) so the UI can show whether audio is actually
  // being captured at all -- turns "the mic isn't working" from a black box into
  // something visible: if this stays at 0 while you talk, the browser isn't getting
  // audio from the mic; if it moves but nothing else happens, the problem is elsewhere.
  const [audioLevel, setAudioLevel] = useState(0);
  const lastLevelUpdateRef = useRef(0);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const playbackQueueRef = useRef<AudioBuffer[]>([]);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const isPlayingRef = useRef(false);
  // Timestamp of the last moment assistant audio was playing -- room echo/reverb can
  // briefly linger for a moment right after playback stops, so mic transmission stays
  // gated for a short cooldown past isPlayingRef going false, not just while it's true.
  const playbackEndedAtRef = useRef(0);
  const MIC_RESUME_COOLDOWN_MS = 400;

  const stopPlayback = useCallback(() => {
    playbackQueueRef.current = [];
    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.stop();
      } catch {
        // already stopped
      }
      currentSourceRef.current = null;
    }
    isPlayingRef.current = false;
    playbackEndedAtRef.current = performance.now();
  }, []);

  const playNext = useCallback(() => {
    const context = audioContextRef.current;
    const next = playbackQueueRef.current.shift();
    if (!context || !next) {
      isPlayingRef.current = false;
      playbackEndedAtRef.current = performance.now();
      return;
    }
    isPlayingRef.current = true;
    const source = context.createBufferSource();
    source.buffer = next;
    source.connect(context.destination);
    source.onended = () => playNext();
    currentSourceRef.current = source;
    source.start();
  }, []);

  const enqueueAudio = useCallback(
    async (arrayBuffer: ArrayBuffer) => {
      const context = audioContextRef.current;
      if (!context) return;
      try {
        const audioBuffer = await context.decodeAudioData(arrayBuffer);
        playbackQueueRef.current.push(audioBuffer);
        if (!isPlayingRef.current) playNext();
      } catch {
        // A malformed/partial audio chunk shouldn't kill the session.
      }
    },
    [playNext],
  );

  const stop = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
    wsRef.current = null;
    stopPlayback();
    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
    audioContextRef.current?.close().catch(() => {});
    audioContextRef.current = null;
    setStatus("idle");
    setInterimTranscript("");
    setAudioLevel(0);
  }, [stopPlayback]);

  const start = useCallback(
    async (options: { language: string; voice: string }) => {
      setError("");
      setMessages([]);
      setInterimTranscript("");
      setAudioLevel(0);
      setStatus("connecting");

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
        mediaStreamRef.current = stream;

        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;
        await routePlaybackToMicDevice(audioContext, stream);
        // Loaded from a blob: URL rather than Vite's emitted asset URL: small files like this
        // get inlined as a data: URI at build time, and audioWorklet.addModule() support for
        // data: URLs is inconsistent across browsers, while blob: URLs are well-supported.
        const workletUrl = URL.createObjectURL(new Blob([pcmWorkletSource], { type: "application/javascript" }));
        try {
          await audioContext.audioWorklet.addModule(workletUrl);
        } finally {
          URL.revokeObjectURL(workletUrl);
        }

        const source = audioContext.createMediaStreamSource(stream);
        const worklet = new AudioWorkletNode(audioContext, "pcm-worklet", {
          processorOptions: { targetSampleRate: 16000 },
        });
        workletNodeRef.current = worklet;
        source.connect(worklet);
        worklet.connect(audioContext.destination); // keeps the graph pulled; the node emits silent output

        const ws = new WebSocket(`${wsBase}/api/voice/session/live`);
        ws.binaryType = "arraybuffer";
        wsRef.current = ws;

        ws.onopen = () => {
          ws.send(JSON.stringify({ type: "start", language: options.language, voice: options.voice, sampleRate: 16000 }));
        };

        worklet.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
          // Compute level before sending -- WebSocket.send() reads the buffer but
          // doesn't detach/transfer it, so this stays safe to read afterward too,
          // but doing it first keeps the ordering obviously correct either way.
          const now = performance.now();
          if (now - lastLevelUpdateRef.current > 80) {
            lastLevelUpdateRef.current = now;
            const samples = new Int16Array(event.data);
            let sumSquares = 0;
            for (const sample of samples) sumSquares += sample * sample;
            const rms = samples.length > 0 ? Math.sqrt(sumSquares / samples.length) / 32768 : 0;
            setAudioLevel(rms);
          }
          // Don't transmit mic audio while the assistant's own reply is playing back.
          // Without headphones, speaker output leaks into the mic (echoCancellation
          // helps with human speech patterns but doesn't reliably suppress synthesized
          // TTS audio) -- observed this causing the server's VAD to fire on the
          // assistant's own voice, transcribing garbage/hallucinated text from the
          // echo and repeatedly self-interrupting mid-reply. This trades away true
          // talk-over-the-AI barge-in for a reliable turn-taking conversation instead.
          const muted = isPlayingRef.current || performance.now() - playbackEndedAtRef.current < MIC_RESUME_COOLDOWN_MS;
          if (ws.readyState === WebSocket.OPEN && !muted) ws.send(event.data);
        };

        ws.onmessage = (event: MessageEvent<string | ArrayBuffer>) => {
          if (typeof event.data !== "string") {
            void enqueueAudio(event.data);
            return;
          }
          const payload = JSON.parse(event.data) as ServerMessage;
          switch (payload.type) {
            case "ready":
              setStatus("listening");
              break;
            case "transcript_interim":
              setInterimTranscript(payload.text);
              break;
            case "transcript":
              setInterimTranscript("");
              // language/languageConfidence come straight from the backend's own STT
              // detection -- the UI shows exactly what the backend determined, never
              // its own guess.
              setMessages((current) => [
                ...current,
                {
                  role: "user", text: payload.text,
                  ...(payload.language !== undefined ? { language: payload.language } : {}),
                  ...(payload.language_confidence !== undefined ? { languageConfidence: payload.language_confidence } : {}),
                },
              ]);
              setStatus("responding");
              break;
            case "assistant_text":
              setMessages((current) => {
                const last = current[current.length - 1];
                if (last && last.role === "assistant") {
                  return [...current.slice(0, -1), { role: "assistant", text: payload.text }];
                }
                return [...current, { role: "assistant", text: payload.text }];
              });
              if (payload.final) setStatus("listening");
              break;
            case "audio":
              // The binary WebSocket frame carrying this sentence's audio follows
              // immediately; enqueueAudio() picks it up on the next onmessage call.
              break;
            case "interrupted":
              stopPlayback();
              break;
            case "empty_transcript":
              setInterimTranscript("");
              setStatus("listening");
              break;
            case "error":
              setError(payload.message);
              setStatus("listening");
              break;
          }
        };

        ws.onerror = () => {
          setError("The live call connection was lost.");
          setStatus("error");
        };
        ws.onclose = () => {
          if (wsRef.current === ws) setStatus("idle");
        };
      } catch (requestError) {
        let message = requestError instanceof Error ? requestError.message : "Microphone access failed.";
        if (requestError instanceof DOMException) {
          if (requestError.name === "NotAllowedError" || requestError.name === "SecurityError") {
            message = "Microphone access is blocked. Click the lock/camera icon in your browser's address bar, allow microphone access for this site, then reload the page.";
          } else if (requestError.name === "NotFoundError") {
            message = "No microphone was found. Check that one is connected and enabled in your system's sound settings.";
          } else if (requestError.name === "NotReadableError") {
            message = "The microphone couldn't be started -- another app may already be using it. Close other apps using the mic and try again.";
          }
        }
        setError(message);
        setStatus("error");
        stop();
      }
    },
    [enqueueAudio, stop, stopPlayback],
  );

  const sendText = useCallback(async (question: string, options: { language: string; voice: string }) => {
    const history = messages.map((message) => ({ role: message.role, content: message.text }));
    setMessages((current) => [...current, { role: "user", text: question }]);
    setError("");
    // Typed questions can arrive with no live mic session ever started, so there is
    // no AudioContext yet (that's normally created in start()) -- create one lazily
    // here, from this same user gesture (submitting the question), which browsers
    // allow without an autoplay-policy block.
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }
    const wasLive = wsRef.current !== null && wsRef.current.readyState === WebSocket.OPEN;
    try {
      const response = await fetch(`${apiBase}/api/voice/session/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, language: options.language, voice: options.voice, history }),
      });
      if (!response.ok || !response.body) throw new Error(`The voice session is unavailable (${response.status}).`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let assistantText = "";
      setMessages((current) => [...current, { role: "assistant", text: "" }]);
      // A typed question has no live mic session to put status into "responding" for
      // (that only happens inside the WS onmessage handler in start()) -- without this,
      // the reply's word-by-word reveal never activates and every typed answer just
      // snaps in whole, the exact thing the live-caption typewriter was built to avoid.
      setStatus("responding");
      while (true) {
        const { value: chunk, done } = await reader.read();
        buffer += decoder.decode(chunk ?? new Uint8Array(), { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const payload = JSON.parse(line);
          if (payload.error) throw new Error(payload.error);
          if (payload.sentence) {
            assistantText = assistantText ? `${assistantText} ${payload.sentence}` : payload.sentence;
            setMessages((current) => current.map((message, index) => (index === current.length - 1 && message.role === "assistant" ? { ...message, text: assistantText } : message)));
          }
          if (payload.audioBase64 && audioContextRef.current) {
            const binary = atob(payload.audioBase64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
            void enqueueAudio(bytes.buffer);
          }
        }
        if (done) break;
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The voice session is unavailable.");
    } finally {
      // Restore whichever state made sense before this reply started -- a live mic
      // session (if one was running underneath the typed question) goes back to
      // "listening", otherwise back to "idle".
      setStatus((current) => (current === "responding" ? (wasLive ? "listening" : "idle") : current));
    }
  }, [messages]);

  return { status, messages, interimTranscript, audioLevel, error, start, stop, sendText };
}
