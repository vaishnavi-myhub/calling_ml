import { type ComponentType, type ReactNode } from "react";
import { Link } from "@tanstack/react-router";
import { ArrowDownRight, ArrowUpRight, Bot, CircleAlert, PhoneCall } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { CallRecordDto } from "@/lib/types";
import { VoiceHero } from "./voice-hero";

export type Tone = "success" | "info" | "ai" | "knowledge" | "warning" | "error" | "neutral";
export const toneClasses: Record<Tone, string> = {
  success: "bg-success-soft text-success", info: "bg-info-soft text-info", ai: "bg-ai-soft text-ai",
  knowledge: "bg-knowledge-soft text-knowledge", warning: "bg-warning-soft text-warning", error: "bg-error-soft text-destructive",
  neutral: "bg-muted text-muted-foreground",
};

export function StatusBadge({ children, tone = "success" }: { children: ReactNode; tone?: Tone }) {
  return <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold", toneClasses[tone])}>{tone === "success" && <span className="size-1.5 rounded-full bg-success" />}{children}</span>;
}

export function IconTile({ icon: Icon, tone = "success", className }: { icon: ComponentType<{ className?: string }>; tone?: Tone; className?: string }) {
  return <span className={cn("flex size-9 shrink-0 items-center justify-center rounded-lg", toneClasses[tone], className)}><Icon className="size-4" /></span>;
}

export function Panel({ title, subtitle, action, children, className }: { title?: string; subtitle?: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={cn("surface", className)}>{(title || action) && <div className="flex items-start justify-between gap-4 px-5 pt-5"><div><h2 className="text-base font-semibold text-foreground">{title}</h2>{subtitle && <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>}</div>{action}</div>}<div className={cn(title ? "p-5 pt-4" : "p-5")}>{children}</div></section>;
}

export function PageHeader({ title, description, actions }: { title: string; description: string; actions?: ReactNode }) {
  return <><div className="mb-4 flex flex-col justify-between gap-3 lg:flex-row lg:items-end"><div><h1 className="text-2xl font-semibold text-foreground">{title}</h1><p className="mt-1 text-sm text-muted-foreground">{description}</p></div>{actions && <div className="flex flex-wrap gap-2">{actions}</div>}</div>{title === "Good morning" && <VoiceHero />}</>;
}

export function KPI({ label, value, change, tone, icon: Icon, down = false }: { label: string; value: string; change?: string; tone: Tone; icon: ComponentType<{ className?: string }>; down?: boolean }) {
  return <div className="surface group min-w-0 p-4 transition-all hover:-translate-y-0.5 hover:shadow-card-hover"><div className="flex items-center justify-between"><IconTile icon={Icon} tone={tone} />{change && <span className="flex items-center gap-0.5 text-xs font-semibold text-success">{down ? <ArrowDownRight className="size-3" /> : <ArrowUpRight className="size-3" />}{change}</span>}</div><p className="mt-4 text-xs font-medium text-muted-foreground">{label}</p><strong className="mt-1 block text-2xl font-semibold text-foreground">{value}</strong></div>;
}

export function MetricCard({ label, value, hint, icon: Icon, tone = "success" }: { label: string; value: string; hint?: string; icon: ComponentType<{className?: string}>; tone?: Tone }) {
 return <div className="surface flex items-center gap-3 p-4"><IconTile icon={Icon} tone={tone}/><div><p className="text-xs text-muted-foreground">{label}</p><p className="mt-0.5 text-xl font-semibold">{value}</p>{hint && <p className="text-xs text-success">{hint}</p>}</div></div>
}

export function CallsTable({ rows, detailed = false }: { rows: CallRecordDto[]; detailed?: boolean }) {
 if (rows.length === 0) {
  return <EmptyAction title="No calls yet" description="Start a test call on the Live Call page — real sessions will show up here." />;
 }
 return <Table><TableHeader><TableRow><TableHead>Caller</TableHead><TableHead>AI Agent</TableHead><TableHead>Direction</TableHead><TableHead>Language</TableHead><TableHead>Duration</TableHead>{detailed && <TableHead>Sentiment</TableHead>}<TableHead>Lead Status</TableHead>{detailed && <TableHead>When</TableHead>}<TableHead>Status</TableHead></TableRow></TableHeader><TableBody>{rows.map((r)=><TableRow key={r.id} className="cursor-pointer"><TableCell className="font-medium">{r.caller}</TableCell><TableCell>{r.agent}</TableCell><TableCell>{r.direction}</TableCell><TableCell><StatusBadge tone="info">{r.language}</StatusBadge></TableCell><TableCell>{r.duration}</TableCell>{detailed && <TableCell><StatusBadge tone={r.sentiment==="Positive"?"success":r.sentiment==="Negative"?"error":"neutral"}>{r.sentiment ?? "Not analyzed"}</StatusBadge></TableCell>}<TableCell>{r.lead}</TableCell>{detailed && <TableCell className="whitespace-nowrap">{r.time}</TableCell>}<TableCell><StatusBadge tone={r.status==="Failed"?"error":r.status==="In Progress"?"warning":"success"}>{r.status}</StatusBadge></TableCell></TableRow>)}</TableBody></Table>
}

export function EmptyAction({ title, description }: { title:string; description:string }) { return <div className="flex min-h-48 flex-col items-center justify-center text-center"><IconTile icon={CircleAlert} tone="neutral"/><h3 className="mt-3 font-semibold">{title}</h3><p className="mt-1 max-w-sm text-sm text-muted-foreground">{description}</p></div> }

export function LiveLinkButton() { return <Button asChild><Link to="/live-calls"><PhoneCall/>Open live call</Link></Button> }
export const defaultIcon = Bot;
