import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import {
  Bot,
  BrainCircuit,
  Clock3,
  Database,
  Gauge,
  Languages,
  PhoneCall,
  Target,
} from "lucide-react";
import { CallControls, Transcript } from "@/components/voice/live-call";
import { Waveform } from "@/components/voice/waveform";
import { IconTile, PageHeader, Panel, StatusBadge } from "@/components/voice/shared";
import { useLiveCall } from "@/hooks/use-live-call";
export const Route = createFileRoute("/live-calls")({
  head: () => ({
    meta: [
      { title: "Live Call Control Center — AI Voice Calling" },
      {
        name: "description",
        content:
          "Monitor a connected AI call, transcript, intelligence, and controls in real time.",
      },
      { property: "og:title", content: "Live AI Call Control Center" },
      {
        property: "og:description",
        content: "A real-time multilingual AI call monitoring and control experience.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: LiveCalls,
});

function formatElapsed(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes.toString().padStart(2, "0")}:${remainder.toString().padStart(2, "0")}`;
}

function LiveCalls() {
  const liveCall = useLiveCall();
  const [providers, setProviders] = useState<{
    llm: { configured: boolean; model: string };
    stt: { configured: boolean };
    tts: { configured: boolean };
  } | null>(null);
  const [providerError, setProviderError] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    fetch(`${import.meta.env["VITE_API_URL"] ?? "http://127.0.0.1:4000"}/api/voice/providers`)
      .then(async (response) => {
        if (!response.ok) throw new Error("Provider status unavailable.");
        return response.json();
      })
      .then(setProviders)
      .catch((error: unknown) => setProviderError(error instanceof Error ? error.message : "Provider status unavailable."));
  }, []);

  const sessionActive = liveCall.status !== "idle" && liveCall.status !== "error";
  useEffect(() => {
    if (!sessionActive) {
      setElapsedSeconds(0);
      return;
    }
    const startedAt = Date.now();
    const timer = window.setInterval(() => setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [sessionActive]);

  const llmReady = providers?.llm.configured ?? false;
  const sttReady = providers?.stt.configured ?? false;
  const ttsReady = providers?.tts.configured ?? false;
  const sessionLabel = providers === null ? "Loading" : llmReady ? "Ready" : "LLM setup needed";

  return (
    <>
      <PageHeader title="Live Call" description="Speak naturally with the local AI agent — it answers with real speech once you pause." />
      <div className="surface mb-4 flex flex-wrap items-center gap-4 p-4">
        <IconTile icon={PhoneCall} />
        <div>
          <p className="text-xs text-muted-foreground">Voice session</p>
          <p className="font-semibold">Local AI agent</p>
        </div>
        <StatusBadge tone={llmReady ? "success" : "warning"}>{sessionLabel}</StatusBadge>
        <div className="ml-auto flex items-center gap-2 font-mono text-lg font-semibold">
          <Clock3 className="size-4 text-muted-foreground" />
          {sessionActive ? formatElapsed(elapsedSeconds) : "00:00"}
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-4">
          <Panel>
            <div className="rounded-lg bg-success-soft/50 px-4 py-8">
              <Waveform large />
              <div className="mt-4 flex items-center justify-center gap-2 text-sm font-semibold text-success-strong">
                <Bot className="size-4" />
                {sessionActive ? liveCall.status : llmReady ? "Ready for a call" : sessionLabel}
              </div>
            </div>
            <div className="mt-5">
              <CallControls />
            </div>
          </Panel>
          <Panel title="Real-time Transcript" action={<StatusBadge tone="info">Interactive</StatusBadge>}>
            <Transcript liveCall={liveCall} />
          </Panel>
        </div>
        <Panel
          title="Call Intelligence"
            action={<StatusBadge tone="ai">Local model</StatusBadge>}
        >
          <div className="divide-y divide-border">
            {[
              ["Session", sessionActive ? liveCall.status : "Not connected", Languages, sessionActive ? "success" : "neutral"],
              ["Turns", String(liveCall.messages.filter((m) => m.role === "user").length), Target, "ai"],
              ["LLM", providers?.llm.model ?? "Loading provider...", Gauge, llmReady ? "success" : "warning"],
              ["TTS", ttsReady ? "Configured" : "Disabled or not configured", BrainCircuit, ttsReady ? "success" : "neutral"],
              ["STT", sttReady ? "Configured (local Whisper)" : "Not configured", Database, sttReady ? "success" : "warning"],
            ].map(([l, v, I, t]) => (
              <div key={l as string} className="flex items-center gap-3 py-4">
                <IconTile icon={I as typeof Bot} tone={t as "success"} />
                <div>
                  <p className="text-xs text-muted-foreground">{l as string}</p>
                  <b className="text-sm">{v as string}</b>
                </div>
              </div>
            ))}
          </div>
          {providerError && <p className="mt-4 text-xs font-medium text-destructive">{providerError}</p>}
        </Panel>
      </div>
    </>
  );
}
