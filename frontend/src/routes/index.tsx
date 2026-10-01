import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import {
  Activity,
  Bot,
  BrainCircuit,
  Clock3,
  Database,
  Gauge,
  Megaphone,
  PhoneCall,
  Plus,
  Settings,
  UserPlus,
  UsersRound,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { ActivityChart } from "@/components/voice/charts";
import { AICallFlow } from "@/components/voice/flow";
import { LiveCallCard } from "@/components/voice/live-call";
import {
  CallsTable,
  IconTile,
  KPI,
  PageHeader,
  Panel,
  StatusBadge,
  type Tone,
} from "@/components/voice/shared";
import type { OverviewDto } from "@/lib/types";

const apiBase = import.meta.env["VITE_API_URL"] ?? "http://127.0.0.1:4000";
const serviceIcons: Record<string, typeof Bot> = { LLM: BrainCircuit, STT: Activity, TTS: Bot, Database: Gauge };

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Overview — AI Voice Calling" },
      {
        name: "description",
        content: "Monitor live AI calls, campaigns, performance, and local AI infrastructure.",
      },
      { property: "og:title", content: "AI Voice Calling Operations Overview" },
      {
        property: "og:description",
        content:
          "Real-time multilingual AI voice operations and local AI infrastructure monitoring.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Overview,
});

function Overview() {
  const [overview, setOverview] = useState<OverviewDto | null>(null);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    fetch(`${apiBase}/api/overview`)
      .then((response) => {
        if (!response.ok) throw new Error("Overview data is unavailable.");
        return response.json();
      })
      .then(setOverview)
      .catch((error: unknown) => setLoadError(error instanceof Error ? error.message : "Overview data is unavailable."));
  }, []);

  const metrics = overview?.metrics;
  const kpis = [
    ["Active Calls", metrics ? String(metrics.activeCalls) : "—", PhoneCall, "success"],
    ["Calls Today", metrics ? String(metrics.callsToday) : "—", Activity, "info"],
    ["Successful Calls", metrics ? String(metrics.successfulCalls) : "—", Gauge, "success"],
    ["Avg. Duration", metrics?.averageDuration ?? "—", Clock3, "ai"],
    ["Avg. Latency", metrics?.averageLatency ?? "—", Zap, "warning"],
    ["Leads Qualified", metrics ? String(metrics.leadsQualified) : "—", UsersRound, "info"],
  ] as const;

  return (
    <>
      <div className="relative">
        <PageHeader
          title="Good morning"
          description="Monitor your AI calling operations and system performance."
          actions={
            <>
              <Button variant="outline" asChild>
                <Link to="/live-calls">
                  <PhoneCall />
                  Start Test Call
                </Link>
              </Button>
              <Button onClick={() => toast.success("Campaign builder opened")}>
                <Plus />
                Create Campaign
              </Button>
            </>
          }
        />
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-6">
        {kpis.map(([l, v, I, t]) => (
          <KPI key={l} label={l} value={v} change="" tone={t as Tone} icon={I} />
        ))}
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1.7fr)_minmax(340px,0.8fr)]">
        <ActivityChart data={overview?.activity ?? []} />
        <LiveCallCard latestCall={overview?.recentCalls?.[0]} />
      </div>
      <div className="mt-4">
        <AICallFlow />
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-[1.5fr_0.85fr]">
        <Panel
          title="Recent Calls"
          action={
            <Button variant="ghost" size="sm" asChild>
              <Link to="/call-history">View all</Link>
            </Button>
          }
        >
          <CallsTable rows={overview?.recentCalls ?? []} />
        </Panel>
        <Panel title="Live System Status" action={<StatusBadge>{loadError ? "Unavailable" : "Live"}</StatusBadge>}>
          <div className="divide-y divide-border">
            {(overview?.services ?? []).map((service) => (
              <div key={service.name} className="flex items-center gap-3 py-2.5">
                <IconTile icon={serviceIcons[service.name] ?? Database} tone={service.status === "Online" ? "success" : "warning"} />
                <div className="min-w-0 flex-1">
                  <b className="block text-xs">{service.name}</b>
                  <span className="block truncate text-[11px] text-muted-foreground">{service.model}</span>
                </div>
                <StatusBadge tone={service.status === "Online" ? "success" : "warning"}>{service.status}</StatusBadge>
              </div>
            ))}
          </div>
        </Panel>
      </div>
      <Panel className="mt-4" title="Quick Actions">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
          {[
            ["Create Campaign", "Launch an outbound campaign", Megaphone, "success", "/campaigns"],
            ["Add Contact", "Add a new lead", UserPlus, "info", "/contacts"],
            ["Create AI Agent", "Design a voice agent", Bot, "ai", "/agents"],
            [
              "Add Knowledge",
              "Upload trusted content",
              BrainCircuit,
              "knowledge",
              "/knowledge-base",
            ],
            ["View Analytics", "Explore performance", Activity, "info", "/analytics"],
            ["System Settings", "Configure workspace", Settings, "neutral", "/settings"],
          ].map(([t, d, I, tone, to]) => (
            <Link
              key={t as string}
              to={to as "/campaigns"}
              className="rounded-lg border border-border p-4 transition hover:-translate-y-0.5 hover:bg-muted/50 hover:shadow-sm"
            >
              <IconTile icon={I as typeof Bot} tone={tone as Tone} />
              <b className="mt-4 block text-sm">{t as string}</b>
              <span className="mt-1 block text-xs text-muted-foreground">{d as string}</span>
            </Link>
          ))}
        </div>
      </Panel>
    </>
  );
}
