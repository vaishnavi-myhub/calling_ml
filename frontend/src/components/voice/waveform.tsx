import { cn } from "@/lib/utils";
export function Waveform({ large=false, blue=false }: { large?:boolean; blue?:boolean }) {
 const heights=[18,30,46,28,58,38,72,44,82,55,68,38,76,48,88,56,70,40,60,32,48,28,40,22,34,18,26,14];
 return <div className={cn("waveform flex items-center justify-center gap-1 overflow-hidden",large?"h-40":"h-16")} role="img" aria-label={`${blue?"Customer":"AI agent"} audio waveform`}>{heights.map((h,i)=><span key={i} className={cn("wave-bar w-1 rounded-full", blue?"bg-info":"bg-success")} style={{height:`${large?h:h*.65}%`,animationDelay:`${i*45}ms`}} />)}</div>
}
