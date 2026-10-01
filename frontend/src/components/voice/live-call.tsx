import { useEffect, useRef, useState } from "react";
import { Link } from "@tanstack/react-router";
import {
  Bot,
  FilePlus2,
  Languages,
  LoaderCircle,
  Mic,
  MicOff,
  Pause,
  PhoneCall,
  PhoneOff,
  UsersRound,
  Send,
  Upload,
  UserRound,
  Volume2,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { EmptyAction, IconTile, Panel, StatusBadge } from "./shared";
import { Waveform } from "./waveform";
import { useLiveCall, type LiveCallStatus } from "@/hooks/use-live-call";
import type { CallRecordDto } from "@/lib/types";

type TranscriptProps = { liveCall: ReturnType<typeof useLiveCall> };
const apiBase = import.meta.env["VITE_API_URL"] ?? "http://127.0.0.1:4000";

// The backend delivers text in clause/sentence-sized bursts (each one gated on a full
// LLM + translation round trip, so it can't arrive any faster than that) -- fine for
// audio, since TTS is synthesized on the same clause boundary, but reading as a chat
// bubble jumping in big chunks rather than a live caption growing word by word.
// Revealing each new burst at a steady per-character pace (independent of when the
// text itself becomes available) gets the word-by-word feel without touching how
// often the backend actually sends anything, or the audio pipeline's timing at all.
function useTypewriter(text: string, active: boolean, msPerChar = 22): string {
  // Seeded from `active` only at mount: a message that's already settled when it first
  // renders (anything but the one currently streaming) should show in full immediately,
  // not replay an animation for text the caller never saw arrive live.
  const [shown, setShown] = useState(() => (active ? "" : text));
  const previousRef = useRef(text);

  useEffect(() => {
    if (!text.startsWith(previousRef.current)) {
      // A genuinely new/replaced message, or the backend's own interim STT revising a
      // word it already sent (not a simple extension) -- show it immediately rather
      // than animating backwards or re-typing text that was already visible.
      previousRef.current = text;
      setShown(text);
      return;
    }
    previousRef.current = text;
    const target = text;
    const id = window.setInterval(() => {
      setShown((current) => {
        if (current.length >= target.length) {
          window.clearInterval(id);
          return current;
        }
        return target.slice(0, current.length + 1);
      });
    }, msPerChar);
    return () => window.clearInterval(id);
    // `active` deliberately left out: a reveal already in progress must run to
    // completion even if status flips (e.g. to "listening") in the same render that
    // delivers the final chunk of a short, one-burst reply -- otherwise that reply,
    // the common case for a quick answer, never visibly animates at all.
  }, [text, msPerChar]);

  return shown.length < text.length ? shown : text;
}

function TypewriterText({ text, active, className }: { text: string; active: boolean; className?: string }) {
  const shown = useTypewriter(text, active);
  return <p className={className}>{shown}</p>;
}

const statusLabel: Record<LiveCallStatus, string> = {
  idle: "Not connected",
  connecting: "Connecting",
  listening: "Listening",
  responding: "AI responding",
  error: "Connection error",
};

export function Transcript({ liveCall }: TranscriptProps) {
  const { status, messages, interimTranscript, audioLevel, error, start, stop, sendText } = liveCall;
  const [question, setQuestion] = useState("");
  const [sending, setSending] = useState(false);
  const [language, setLanguage] = useState("auto");
  const [voice, setVoice] = useState("default");
  const [languageOptions, setLanguageOptions] = useState<{ value: string; label: string }[]>([{ value: "auto", label: "Auto detect" }]);
  const [voiceOptions, setVoiceOptions] = useState<string[]>(["default"]);
  const [uploadedDocuments, setUploadedDocuments] = useState<string[]>([]);
  const [uploadingDocument, setUploadingDocument] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/api/voice/languages`)
      .then((response) => response.json())
      .then((data) => Array.isArray(data.languages) && data.languages.length && setLanguageOptions(data.languages))
      .catch(() => undefined);
    fetch(`${apiBase}/api/voice/voices`)
      .then((response) => response.json())
      .then((data) => Array.isArray(data.voices) && data.voices.length && setVoiceOptions(data.voices))
      .catch(() => undefined);
  }, []);

  async function handleDocumentUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;

    setUploadingDocument(true);
    try {
      const response = await fetch(`${apiBase}/api/voice/documents/upload`, {
        method: "POST",
        headers: {
          "Content-Type": nextFile.type || "application/octet-stream",
          "X-Filename": nextFile.name,
        },
        body: nextFile,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail ?? "Document upload failed.");
      }
      setUploadedDocuments((current) => (current.includes(nextFile.name) ? current : [...current, nextFile.name]));
      toast.success(`${nextFile.name} is now available to the live-call assistant.`);
    } catch (requestError) {
      toast.error(requestError instanceof Error ? requestError.message : "Document upload failed.");
    } finally {
      setUploadingDocument(false);
      event.target.value = "";
    }
  }

  async function askAgent(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = question.trim();
    if (!value || sending) return;
    setQuestion("");
    setSending(true);
    try {
      await sendText(value, { language, voice });
    } finally {
      setSending(false);
    }
  }

  function toggleSession() {
    if (status === "idle" || status === "error") {
      void start({ language, voice });
    } else {
      stop();
    }
  }

  const isLive = status !== "idle" && status !== "error";

  return (
    <div className="space-y-4">
      <div className="min-h-12 space-y-3">
        {messages.length === 0 && (
          <p className="rounded-lg border border-dashed border-border p-4 text-sm text-muted-foreground">
            Your conversation will appear here.
          </p>
        )}
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} className="transcript-enter flex gap-3">
            <IconTile icon={message.role === "user" ? UserRound : Bot} tone={message.role === "user" ? "info" : "success"} />
            <div>
              <div className="flex items-center gap-2 text-xs">
                <b>{message.role === "user" ? "You" : "AI Agent"}</b>
                <span className="text-muted-foreground">Now</span>
                {message.role === "assistant" && <Volume2 className="size-3 text-success" />}
                {message.role === "user" && message.language && (
                  <span
                    className="rounded-full bg-border px-1.5 py-0.5 font-mono text-[10px] uppercase text-muted-foreground"
                    title={message.languageConfidence !== undefined ? `Detected by the backend, ${Math.round(message.languageConfidence * 100)}% confidence` : "Detected by the backend"}
                  >
                    {message.language}
                  </span>
                )}
              </div>
              <TypewriterText
                text={message.text}
                active={message.role === "assistant" && index === messages.length - 1 && status === "responding"}
                className={message.role === "user" ? "mt-1 rounded-lg rounded-tl-none border border-border bg-card p-3 text-sm" : "mt-1 rounded-lg rounded-tl-none bg-success-soft p-3 text-sm"}
              />
            </div>
          </div>
        ))}
        {status === "listening" && interimTranscript && (
          <div className="transcript-enter flex gap-3">
            <IconTile icon={UserRound} tone="info" />
            <div>
              <div className="flex items-center gap-2 text-xs">
                <b>You</b>
                <span className="text-muted-foreground">Listening…</span>
              </div>
              <TypewriterText
                text={interimTranscript}
                active
                className="mt-1 rounded-lg rounded-tl-none border border-dashed border-border bg-card/60 p-3 text-sm italic text-muted-foreground"
              />
            </div>
          </div>
        )}
        {status === "responding" && (
          <p className="rounded-lg border border-info/30 bg-info-soft p-3 text-sm text-muted-foreground">
            Thinking<span className="thinking-dots"> ...</span>
          </p>
        )}
      </div>

      <>
          <div className="flex flex-wrap gap-2 border-t border-border pt-4">
            <label className="flex min-w-36 flex-1 flex-col gap-1 text-[11px] font-semibold text-muted-foreground">
              <span>Response language</span>
              <select
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
                disabled={isLive}
                className="h-9 rounded-lg border border-border bg-card px-2 text-xs text-foreground outline-hidden focus:border-primary disabled:opacity-60"
              >
                {languageOptions.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label className="flex min-w-36 flex-1 flex-col gap-1 text-[11px] font-semibold text-muted-foreground">
              <span>Voice profile</span>
              <select
                value={voice}
                onChange={(event) => setVoice(event.target.value)}
                disabled={isLive}
                className="h-9 rounded-lg border border-border bg-card px-2 text-xs text-foreground outline-hidden focus:border-primary disabled:opacity-60"
              >
                {voiceOptions.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </label>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-xs font-medium text-foreground hover:border-primary">
              <Upload className="size-3.5" />
              {uploadingDocument ? "Uploading..." : "Upload document"}
              <input type="file" accept=".txt,.md,.pdf,.docx,.rtf,.html" className="hidden" onChange={handleDocumentUpload} />
            </label>
            {uploadedDocuments.length > 0 && (
              <span className="text-xs text-muted-foreground">Context: {uploadedDocuments.join(", ")}</span>
            )}
          </div>

          <div className="mt-4 flex flex-col items-center gap-3 rounded-lg border border-border bg-card p-6">
            <button
              type="button"
              onClick={toggleSession}
              aria-label={isLive ? "End the call" : "Tap to talk"}
              className={cn(
                "flex size-16 items-center justify-center rounded-full shadow-md transition-all",
                isLive ? "bg-destructive text-destructive-foreground animate-pulse" : "bg-primary text-primary-foreground hover:scale-105",
              )}
            >
              {status === "connecting" ? <LoaderCircle className="size-6 animate-spin" /> : isLive ? <MicOff className="size-6" /> : <Mic className="size-6" />}
            </button>
            <StatusBadge tone={status === "error" ? "error" : status === "idle" ? "neutral" : "success"}>{statusLabel[status]}</StatusBadge>
            {isLive && (
              <div className="w-40" title="Mic input level -- should move while you talk">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-border">
                  <div
                    className="h-full rounded-full bg-success transition-[width] duration-75"
                    style={{ width: `${Math.min(100, audioLevel * 500)}%` }}
                  />
                </div>
                <p className="mt-1 text-center text-[10px] text-muted-foreground">Mic level -- should move as you speak</p>
              </div>
            )}
            {isLive && <span className="text-xs text-muted-foreground">Speak naturally — your words appear as you talk, and the AI answers after you pause. Tap again to end.</span>}
          </div>

          <form onSubmit={askAgent} className="mt-4 flex gap-2 sm:flex-row">
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Or type a question instead"
              className="h-11 flex-1 rounded-lg border border-border bg-card px-3 text-sm outline-hidden focus:border-primary"
            />
            <button
              type="submit"
              disabled={sending}
              className="h-11 rounded-lg bg-primary px-4 text-sm font-semibold text-primary-foreground disabled:opacity-60"
            >
              {sending ? <LoaderCircle className="size-4 animate-spin" /> : <Send className="size-4" />}
            </button>
          </form>

          {error && <p className="mt-2 text-xs font-medium text-destructive">{error}</p>}
      </>
    </div>
  );
}

export function CallControls() {
  const [paused, setPaused] = useState(false);
  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      <Button variant="outline" size="sm" onClick={() => toast("Microphone muted")}>
        <MicOff />
        Mute
      </Button>
      <Button
        variant={paused ? "secondary" : "outline"}
        size="sm"
        onClick={() => {
          setPaused(!paused);
          toast(paused ? "AI resumed" : "AI paused");
        }}
      >
        <Pause />
        {paused ? "Resume AI" : "Pause AI"}
      </Button>
      <Button variant="outline" size="sm" onClick={() => toast("Transfer options opened")}>
        <UsersRound />
        Transfer
      </Button>
      <Button
        variant="destructive"
        size="sm"
        onClick={() => toast.error("Call ended in demo mode")}
      >
        <PhoneOff />
        End Call
      </Button>
      <Button variant="outline" size="sm" onClick={() => toast.success("Note added")}>
        <FilePlus2 />
        Add Note
      </Button>
    </div>
  );
}

export function LiveCallCard({ latestCall }: { latestCall?: CallRecordDto | undefined }) {
  return (
    <Panel
      className="h-full border-success/30"
      title="Live Call"
      action={<StatusBadge tone={latestCall ? "success" : "neutral"}>{latestCall ? "Last session" : "No sessions yet"}</StatusBadge>}
    >
      {latestCall ? (
        <Link to="/live-calls" className="block rounded-lg outline-hidden focus:ring-3 focus:ring-ring/30">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-lg font-semibold">{latestCall.caller}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {latestCall.direction} · {latestCall.language} · {latestCall.agent}
              </p>
            </div>
            <span className="font-mono text-sm font-semibold">{latestCall.duration}</span>
          </div>
          <div className="my-4 rounded-lg bg-success-soft/60 px-2">
            <Waveform />
            <p className="pb-3 text-center text-xs font-medium text-success-strong">{latestCall.status}</p>
          </div>
          <div className="mb-2 flex flex-wrap gap-2">
            <StatusBadge tone="info">
              <Languages className="size-3" />
              {latestCall.language}
            </StatusBadge>
            <StatusBadge tone={latestCall.status === "Completed" ? "success" : latestCall.status === "Failed" ? "error" : "neutral"}>{latestCall.lead}</StatusBadge>
          </div>
        </Link>
      ) : (
        <EmptyAction title="No live calls yet" description="Start a test call on the Live Call page to see it summarized here." />
      )}
      <div className="mt-5 border-t border-border pt-4">
        <Button asChild className="w-full">
          <Link to="/live-calls">
            <PhoneCall />
            Open live call
          </Link>
        </Button>
      </div>
    </Panel>
  );
}
