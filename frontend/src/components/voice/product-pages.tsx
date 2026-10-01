import { useEffect, useState } from "react";
import { Activity, Bot, BrainCircuit, BriefcaseBusiness, CheckCircle2, Clock3, Cpu, Database, FileText, Gauge, Globe2, Languages, MemoryStick, Network, Phone, Plus, Search, ShieldCheck, Upload, UserPlus, UsersRound, Volume2, Zap } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CallsTable, EmptyAction, IconTile, MetricCard, PageHeader, Panel, StatusBadge, toneClasses, type Tone } from "./shared";
import { MiniBarChart, OutcomeChart } from "./charts";
import { cn } from "@/lib/utils";
import type { CallRecordDto, CampaignDto } from "@/lib/types";

const apiBase = import.meta.env["VITE_API_URL"] ?? "http://127.0.0.1:4000";

const agents=[
 {name:"Sales Assistant",desc:"Qualifies prospects and schedules product demos.",langs:"English • Hindi • Telugu",voice:"Kavya · Warm",calls:"1,248",rate:"86%",tone:"ai" as Tone},
 {name:"Appointment Assistant",desc:"Books, confirms, and reschedules appointments.",langs:"English • Hindi • Telugu",voice:"Meera · Clear",calls:"986",rate:"92%",tone:"success" as Tone},
 {name:"Support Agent",desc:"Resolves common customer and product questions.",langs:"English • Hindi",voice:"Arjun · Calm",calls:"742",rate:"89%",tone:"info" as Tone},
];
export function AgentsPage(){return <><PageHeader title="AI Agents" description="Configure multilingual voice agents and monitor their performance." actions={<Button onClick={()=>toast.success("Agent builder opened")}><Plus/>Create AI Agent</Button>}/><div className="grid gap-4 xl:grid-cols-3">{agents.map(a=><Panel key={a.name} className="transition hover:-translate-y-0.5 hover:shadow-card-hover"><div className="flex items-start gap-3"><IconTile icon={Bot} tone={a.tone} className="size-11"/><div className="flex-1"><div className="flex justify-between gap-2"><h3 className="font-semibold">{a.name}</h3><StatusBadge>Active</StatusBadge></div><p className="mt-1 text-xs text-muted-foreground">{a.desc}</p></div></div><div className="mt-5 space-y-3 border-y border-border py-4 text-xs">{[["Languages",a.langs],["Voice",a.voice],["STT","Faster-Whisper"],["LLM","LLaMA 3"],["TTS","Indic Parler-TTS"]].map(([k,v])=><div key={k} className="flex justify-between gap-3"><span className="text-muted-foreground">{k}</span><b className="text-right font-medium">{v}</b></div>)}</div><div className="mt-4 grid grid-cols-2 gap-3"><div><p className="text-xs text-muted-foreground">Calls handled</p><b className="text-lg">{a.calls}</b></div><div><p className="text-xs text-muted-foreground">Success rate</p><b className="text-lg text-success">{a.rate}</b></div></div></Panel>)}</div></>}

export function CampaignsPage(){
 const [campaignList,setCampaignList]=useState<CampaignDto[]>([]);
 const [name,setName]=useState("");
 const [creating,setCreating]=useState(false);

 const loadCampaigns=()=>{fetch(`${apiBase}/api/campaigns`).then(r=>r.json()).then(data=>setCampaignList(Array.isArray(data.campaigns)?data.campaigns:[])).catch(()=>toast.error("Campaigns are unavailable."))};
 useEffect(()=>{loadCampaigns()},[]);

 async function createCampaign(event:React.FormEvent<HTMLFormElement>){
  event.preventDefault();
  const value=name.trim();
  if(!value||creating)return;
  setCreating(true);
  try{
   const response=await fetch(`${apiBase}/api/campaigns`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:value})});
   if(!response.ok)throw new Error("Could not create the campaign.");
   setName("");
   toast.success(`${value} created`);
   loadCampaigns();
  }catch(error){
   toast.error(error instanceof Error?error.message:"Could not create the campaign.");
  }finally{
   setCreating(false);
  }
 }

 const active=campaignList.filter(c=>c.status==="Running").length;
 return <><PageHeader title="Campaigns" description="Plan and launch outbound AI calling campaigns."/>
 <form onSubmit={createCampaign} className="surface flex flex-wrap items-center gap-2 p-3"><Input value={name} onChange={e=>setName(e.target.value)} placeholder="Campaign name" className="max-w-xs"/><Button type="submit" disabled={creating}><Plus/>{creating?"Creating...":"Create Campaign"}</Button></form>
 <div className="mt-4 grid gap-4 lg:grid-cols-3"><MetricCard label="Total campaigns" value={String(campaignList.length)} icon={BriefcaseBusiness}/><MetricCard label="Active" value={String(active)} icon={CheckCircle2} tone="ai"/><MetricCard label="Draft" value={String(campaignList.length-active)} icon={UsersRound} tone="info"/></div>
 <Panel title="Campaigns" className="mt-4">{campaignList.length===0?<EmptyAction title="No campaigns yet" description="Create your first campaign above."/>:<div className="overflow-x-auto"><table className="w-full min-w-[600px] text-sm"><thead><tr className="border-b border-border text-left text-xs text-muted-foreground">{["Campaign Name","Status","Created"].map(h=><th key={h} className="px-3 py-3 font-medium">{h}</th>)}</tr></thead><tbody>{campaignList.map(c=><tr key={c.id} className="border-b border-border transition hover:bg-muted/50"><td className="px-3 py-4 font-medium">{c.name}</td><td className="px-3 py-4"><StatusBadge tone={c.status==="Running"?"success":c.status==="Completed"?"info":"neutral"}>{c.status}</StatusBadge></td><td className="px-3 py-4 text-muted-foreground">{new Date(c.createdAt).toLocaleString()}</td></tr>)}</tbody></table></div>}</Panel>
 <Panel className="mt-4" title="Campaign performance"><EmptyAction title="No performance data yet" description="Answer rates, contacts reached, and conversion charts will appear here once campaigns start placing real calls through a connected telephony provider."/></Panel>
 </>
}

const contacts=[
 ["Ananya Sharma","+91 98765 43210","Telugu","September Health Check","Interested","2 min ago","Positive","Tomorrow, 10 AM"],
 ["Rohan Verma","+91 87654 32109","Hindi","New Patient Outreach","Contacted","5 min ago","Neutral","Sep 27, 2 PM"],
 ["Meera Iyer","+91 76543 21098","English","Post-visit Feedback","Follow-up","12 min ago","Positive","Sep 26, 11 AM"],
 ["Vikram Singh","+91 65432 10987","Hindi","New Patient Outreach","Not interested","18 min ago","Negative","—"],
];
export function ContactsPage(){return <><PageHeader title="Contacts / Leads" description="Manage contact intelligence, campaign membership, and follow-ups." actions={<><Button variant="outline" onClick={()=>toast("CSV importer opened")}><Upload/>Import CSV</Button><Button onClick={()=>toast.success("Contact form opened")}><UserPlus/>Add Contact</Button></>}/><FilterBar labels={["Language","Campaign","Lead Status","Sentiment","Date"]}/><Panel className="mt-4"><div className="overflow-x-auto"><table className="w-full min-w-[1050px] text-sm"><thead><tr className="border-b border-border text-left text-xs text-muted-foreground">{["Name","Phone","Language","Campaign","Lead Status","Last Call","Sentiment","Next Follow-up"].map(h=><th key={h} className="px-3 py-3 font-medium">{h}</th>)}</tr></thead><tbody>{contacts.map(r=><tr key={r[0]} className="border-b border-border hover:bg-muted/50">{r.map((v,i)=><td key={i} className={cn("px-3 py-4",i===0&&"font-medium")}>{i===2?<StatusBadge tone="info">{v}</StatusBadge>:i===4?<StatusBadge tone={v==="Interested"?"success":v==="Not interested"?"error":"neutral"}>{v}</StatusBadge>:i===6?<StatusBadge tone={v==="Positive"?"success":v==="Negative"?"error":"neutral"}>{v}</StatusBadge>:v}</td>)}</tr>)}</tbody></table></div></Panel></>}

function FilterBar({labels}:{labels:string[]}){return <div className="surface flex flex-wrap items-center gap-2 p-3"><label className="relative min-w-56 flex-1"><Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"/><Input className="pl-9" placeholder="Search records..."/></label>{labels.map(l=><select key={l} className="h-9 rounded-md border border-input bg-card px-3 text-xs text-muted-foreground"><option>{l}: All</option></select>)}</div>}

const sources=[["Documents","128","8,492","2 min ago",FileText,"info"],["FAQs","86","312","Yesterday",Bot,"ai"],["Websites","12","2,180","4 hours ago",Globe2,"knowledge"],["Product Information","34","1,044","Today",Database,"success"],["Policies","18","426","Sep 22",ShieldCheck,"warning"]] as const;
export function KnowledgePage(){return <><PageHeader title="Knowledge Base" description="Ground AI responses in verified documents, FAQs, and policies." actions={<Button onClick={()=>toast.success("Knowledge uploader opened")}><Plus/>Add Knowledge</Button>}/><div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">{sources.map(([name,count,chunks,indexed,Icon,tone])=><div key={name} className="surface p-4"><div className="flex justify-between"><IconTile icon={Icon} tone={tone}/><StatusBadge>Indexed</StatusBadge></div><h3 className="mt-4 text-sm font-semibold">{name}</h3><p className="mt-2 text-2xl font-semibold">{count}</p><p className="mt-1 text-xs text-muted-foreground">{chunks} chunks · {indexed}</p></div>)}</div><Panel className="mt-4" title="Knowledge Search Preview" subtitle="Test retrieval quality before deploying changes"><div className="flex gap-2"><Input defaultValue="What are your appointment timings?"/><Button onClick={()=>toast.success("Found 3 relevant sources")}><Search/>Search</Button></div><div className="mt-5 grid gap-3 lg:grid-cols-3">{["Appointment Scheduling Guide","Clinic Operating Hours","Patient FAQ — Appointments"].map((d,i)=><div key={d} className="rounded-lg border border-border p-4"><div className="flex justify-between gap-3"><b className="text-sm">{d}</b><StatusBadge tone="knowledge">{["96%","91%","87%"][i]}</StatusBadge></div><p className="mt-2 text-xs leading-5 text-muted-foreground">Appointments are available Monday through Saturday from 9 AM to 6 PM...</p></div>)}</div></Panel></>}

export function HistoryPage(){
 const [calls,setCalls]=useState<CallRecordDto[]>([]);
 useEffect(()=>{fetch(`${apiBase}/api/calls`).then(r=>r.json()).then(data=>setCalls(Array.isArray(data.calls)?data.calls:[])).catch(()=>toast.error("Call history is unavailable."))},[]);
 return <><PageHeader title="Call History" description="Review real Live Call sessions and their outcomes."/><FilterBar labels={["Date","Agent","Language","Status"]}/><Panel className="mt-4" title="All calls" action={<Button variant="outline" size="sm" onClick={()=>toast("Export started")}>Export</Button>}><CallsTable rows={calls} detailed/></Panel></>
}

export function AnalyticsPage(){return <><PageHeader title="Analytics" description="Understand call quality, conversion, language performance, and latency."/><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8">{[["Total Calls","8,462",Phone,"success"],["Answer Rate","78.4%",Activity,"info"],["Completion Rate","86.2%",CheckCircle2,"success"],["Avg. Duration","4m 18s",Clock3,"ai"],["Avg. Latency","420ms",Zap,"warning"],["Lead Conversion","18.6%",UsersRound,"success"],["Appointment Rate","24.8%",Gauge,"info"],["Positive Sentiment","72%",Bot,"ai"]].map(([l,v,I,t])=><MetricCard key={l as string} label={l as string} value={v as string} icon={I as typeof Phone} tone={t as Tone}/>)}</div><div className="mt-4 grid gap-4 lg:grid-cols-2"><MiniBarChart/><OutcomeChart/><MiniBarChart title="Latency Performance"/><OutcomeChart title="Sentiment Distribution"/></div><Panel className="mt-4" title="Language Performance"><table className="w-full text-sm"><thead><tr className="border-b text-left text-xs text-muted-foreground"><th className="py-3">Language</th><th>Calls</th><th>Accuracy</th><th>Latency</th><th>Completion Rate</th></tr></thead><tbody>{[["English","3,248","96.8%","380ms","89%"],["Hindi","2,816","95.2%","410ms","87%"],["Telugu","1,984","94.6%","440ms","84%"],["Other Languages","414","91.8%","490ms","78%"]].map(r=><tr key={r[0]} className="border-b hover:bg-muted/50">{r.map((v,i)=><td key={v} className={cn("py-4",i===0&&"font-medium")}>{i===0?<span className="flex items-center gap-2"><Languages className="size-4 text-info"/>{v}</span>:v}</td>)}</tr>)}</tbody></table></Panel></>}

const modelCards=[["VAD","Silero VAD","WebRTC VAD",Activity,"success"],["STT","Faster-Whisper","Whisper Large v3",Languages,"info"],["LLM","LLaMA 3","Mistral 7B",BrainCircuit,"ai"],["TTS","Indic Parler-TTS","Piper TTS",Volume2,"success"],["RAG","Qdrant","—",Database,"knowledge"]] as const;
const computeMetrics = [["GPU","68%",Cpu,"ai"],["CPU","42%",Activity,"info"],["RAM","61%",MemoryStick,"success"]] as const;
export function InfrastructurePage(){return <><PageHeader title="Models & Infrastructure" description="Monitor local AI models, fallbacks, compute, and real-time services."/><div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">{modelCards.map(([type,name,fallback,Icon,tone])=><Panel key={type}><div className="flex justify-between"><IconTile icon={Icon} tone={tone}/><StatusBadge>Healthy</StatusBadge></div><p className="mt-4 text-xs font-semibold text-muted-foreground">{type}</p><h3 className="mt-1 font-semibold">{name}</h3><p className="mt-1 text-xs text-muted-foreground">Fallback: {fallback}</p><div className="mt-4 space-y-2 border-t pt-3 text-xs">{[["Latency",type==="LLM"?"182ms":"42ms"],["Requests","12.4k"],["Errors","0.08%"],["Throughput","48/s"]].map(([k,v])=><div key={k} className="flex justify-between"><span className="text-muted-foreground">{k}</span><b>{v}</b></div>)}</div></Panel>)}</div><div className="mt-4 grid gap-4 lg:grid-cols-2"><Panel title="Compute utilization"><div className="space-y-5">{computeMetrics.map(([n,v,Icon,t])=><div key={n}><div className="mb-2 flex items-center justify-between text-sm"><span className="flex items-center gap-2"><Icon className={cn("size-4",toneClasses[t])}/>{n}</span><b>{v}</b></div><div className="h-2 rounded-full bg-muted"><div className="h-full rounded-full bg-primary" style={{width:v}}/></div></div>)}</div></Panel><Panel title="Service health"><div className="grid grid-cols-2 gap-3">{["Redis","Vector DB","WebSocket","Telephony"].map((n,i)=><div key={n} className="rounded-lg border p-3"><div className="flex items-center justify-between"><IconTile icon={i===0?Database:Network} tone={i===3?"info":"success"}/><StatusBadge>Healthy</StatusBadge></div><b className="mt-3 block text-sm">{n}</b><span className="text-xs text-muted-foreground">99.99% uptime</span></div>)}</div></Panel></div></>}

export function SettingsPage(){const tabs=["General","Telephony","AI Models","Languages","Voices","Knowledge Base","Notifications","Security","Users & Roles"];return <><PageHeader title="Settings" description="Configure your workspace, calling infrastructure, models, and access." actions={<Button onClick={()=>toast.success("Settings saved")}>Save changes</Button>}/><Tabs defaultValue="AI Models" className="surface p-4"><TabsList className="h-auto w-full justify-start overflow-x-auto bg-muted p-1">{tabs.map(t=><TabsTrigger key={t} value={t} className="whitespace-nowrap">{t}</TabsTrigger>)}</TabsList>{tabs.map(t=><TabsContent key={t} value={t} className="mt-5">{t==="AI Models"?<div className="grid gap-4 lg:grid-cols-2">{modelCards.slice(0,4).map(([type,name,fallback,Icon,tone])=><div key={type} className="rounded-lg border p-5"><div className="flex items-center gap-3"><IconTile icon={Icon} tone={tone}/><div><p className="text-xs text-muted-foreground">{type}</p><h3 className="font-semibold">{type} Configuration</h3></div></div><label className="mt-5 block text-xs font-medium">Primary model<select className="mt-2 h-10 w-full rounded-md border bg-card px-3"><option>{name}</option></select></label><label className="mt-4 block text-xs font-medium">Fallback model<select className="mt-2 h-10 w-full rounded-md border bg-card px-3"><option>{fallback}</option></select></label></div>)}</div>:<div className="rounded-lg border border-dashed p-10 text-center"><IconTile icon={t==="Security"?ShieldCheck:SettingsIcon(t)} tone="neutral" className="mx-auto"/><h3 className="mt-3 font-semibold">{t} settings</h3><p className="mt-1 text-sm text-muted-foreground">Workspace controls for {t.toLowerCase()} are ready to configure.</p></div>}</TabsContent>)}</Tabs></>}
function SettingsIcon(t:string){return t==="Telephony"?Phone:t==="Languages"?Languages:t==="Users & Roles"?UsersRound:Gauge}
