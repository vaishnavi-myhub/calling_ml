import { useState } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import {
  Activity,
  BarChart3,
  Bell,
  BookOpen,
  Bot,
  ChevronDown,
  CircleHelp,
  Clock3,
  ContactRound,
  Gauge,
  Headphones,
  Menu,
  Megaphone,
  Search,
  Settings,
  SlidersHorizontal,
  UsersRound,
  Waves,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
const nav = [
  ["Overview", "/", Gauge],
  ["Live Calls", "/live-calls", Headphones],
  ["Campaigns", "/campaigns", Megaphone],
  ["Contacts / Leads", "/contacts", ContactRound],
  ["AI Agents", "/agents", Bot],
  ["Knowledge Base", "/knowledge-base", BookOpen],
  ["Call History", "/call-history", Clock3],
  ["Analytics", "/analytics", BarChart3],
  ["Models & Infrastructure", "/infrastructure", SlidersHorizontal],
  ["Settings", "/settings", Settings],
] as const;
const titles: Record<string, string> = {
  "/": "Overview",
  "/live-calls": "Live Calls",
  "/campaigns": "Campaigns",
  "/contacts": "Contacts / Leads",
  "/agents": "AI Agents",
  "/knowledge-base": "Knowledge Base",
  "/call-history": "Call History",
  "/analytics": "Analytics",
  "/infrastructure": "Models & Infrastructure",
  "/settings": "Settings",
};
export function AppShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const path = useRouterState({ select: (s) => s.location.pathname });
  return (
    <div className="min-h-screen bg-background">
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-60 flex-col border-r border-border bg-card transition-transform lg:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex h-[68px] items-center gap-3 border-b border-border px-5">
          <span className="flex size-10 items-center justify-center rounded-xl bg-primary text-primary-foreground">
            <Waves className="size-5" />
          </span>
          <div>
            <p className="font-semibold">AI Voice Calling</p>
            <p className="text-[11px] text-muted-foreground">Local AI • Real Conversations</p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="ml-auto lg:hidden"
            onClick={() => setOpen(false)}
            aria-label="Close navigation"
          >
            <X />
          </Button>
        </div>
        <nav className="flex-1 space-y-1 overflow-y-auto p-3">
          {nav.map(([label, to, Icon]) => (
            <Link
              key={to}
              to={to}
              onClick={() => setOpen(false)}
              className={cn(
                "flex h-10 items-center gap-3 rounded-lg px-3 text-[13px] font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
                path === to && "bg-success-soft text-success-strong",
              )}
            >
              <Icon className={cn("size-[17px]", path === to && "text-success")} />
              <span>{label}</span>
            </Link>
          ))}
        </nav>
        <div className="space-y-3 border-t border-border p-3">
          <button className="flex w-full items-center gap-2 rounded-lg border border-border p-3 text-left hover:bg-muted">
            <span className="flex size-8 items-center justify-center rounded-md bg-info-soft text-info">
              <UsersRound className="size-4" />
            </span>
            <span className="min-w-0 flex-1">
              <span className="block text-[11px] text-muted-foreground">Workspace</span>
              <span className="block truncate text-xs font-semibold">Healthcare Demo</span>
            </span>
            <ChevronDown className="size-4 text-muted-foreground" />
          </button>
          <div className="rounded-lg bg-success-soft p-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-success-strong">
              <span className="status-pulse size-2 rounded-full bg-success" />
              System Online
            </div>
            <p className="ml-4 mt-1 text-[11px] text-muted-foreground">All services running</p>
          </div>
          <div className="flex items-center gap-3 px-1">
            <span className="flex size-9 items-center justify-center rounded-full bg-ai-soft text-xs font-semibold text-ai">
              VR
            </span>
            <div>
              <p className="text-xs font-semibold">Workspace Admin</p>
              <p className="text-[11px] text-muted-foreground">Product Manager</p>
            </div>
          </div>
        </div>
      </aside>
      {open && (
        <button
          className="fixed inset-0 z-30 bg-overlay lg:hidden"
          aria-label="Close navigation overlay"
          onClick={() => setOpen(false)}
        />
      )}
      <div className="lg:pl-60">
        <header className="sticky top-0 z-20 flex h-[68px] items-center gap-4 border-b border-border bg-card/95 px-4 backdrop-blur md:px-6">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={() => setOpen(true)}
            aria-label="Open navigation"
          >
            <Menu />
          </Button>
          <h2 className="hidden min-w-36 text-sm font-semibold md:block">
            {titles[path] ?? "AI Voice Calling"}
          </h2>
          <label className="relative mx-auto block w-full max-w-xl">
            <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <input
              className="h-10 w-full rounded-lg border border-border bg-muted/60 pl-10 pr-4 text-sm outline-hidden transition focus:border-primary focus:ring-3 focus:ring-ring/20"
              placeholder="Search calls, contacts, campaigns..."
              aria-label="Global search"
            />
          </label>
          <Button variant="ghost" size="icon" aria-label="Notifications">
            <Bell />
          </Button>
          <Button variant="ghost" size="icon" className="hidden sm:inline-flex" aria-label="Help">
            <CircleHelp />
          </Button>
          <span className="hidden size-9 items-center justify-center rounded-full bg-success-soft text-xs font-bold text-success sm:flex">
            VR
          </span>
        </header>
        <main className="mx-auto max-w-[1600px] p-4 md:p-5">{children}</main>
      </div>
    </div>
  );
}
