import { useEffect, useState } from "react";
import { AudioLines, Globe2, Radio, Sparkles } from "lucide-react";

const slides = [
  { eyebrow: "Live voice intelligence", title: "Conversations that", accent: "move work forward.", description: "Local models, real-time translation, and a calm command center for every call.", tags: [[Globe2, "Multi-language"], [Radio, "Real-time"], [Sparkles, "Local AI"]] },
  { eyebrow: "Local AI orchestration", title: "Every response,", accent: "beautifully in sync.", description: "Speech, retrieval, reasoning, and voice working together in one transparent pipeline.", tags: [[Sparkles, "LLaMA 3"], [Radio, "Low latency"], [Globe2, "Private by design"]] },
  { eyebrow: "Real-time operations", title: "Turn live calls into", accent: "clear next steps.", description: "See intent, sentiment, and outcomes while your team is still in the conversation.", tags: [[Radio, "420ms latency"], [Globe2, "38 leads"], [Sparkles, "Live insights"]] },
] as const;

export function VoiceHero() {
  const [activeSlide, setActiveSlide] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => setActiveSlide((slide) => (slide + 1) % slides.length), 5200);
    return () => window.clearInterval(timer);
  }, []);

  const slide = slides[activeSlide];

  return (
    <section className="voice-hero surface mb-4 overflow-hidden">
      <div className="voice-hero-copy voice-slide" key={activeSlide}>
        <span className="eyebrow"><span className="status-pulse size-2 rounded-full bg-success" /> {slide.eyebrow}</span>
        <h2>{slide.title}<br /><em>{slide.accent}</em></h2>
        <p>{slide.description}</p>
        <div className="voice-hero-tags">
          {slide.tags.map(([Icon, label]) => <span key={label}><Icon /> {label}</span>)}
        </div>
      </div>
      <div className="voice-hero-stage" aria-label="Animated AI voice signal visualization">
        <div className="voice-orbit voice-orbit-one" />
        <div className="voice-orbit voice-orbit-two" />
        <div className="voice-signal voice-signal-one" />
        <div className="voice-signal voice-signal-two" />
        <div className="voice-core-shadow" />
        <div className="voice-core"><div className="voice-core-face"><AudioLines /></div></div>
        <div className="voice-core-label"><span className="size-2 rounded-full bg-success" /> Listening</div>
        <div className="voice-slide-dots" aria-label="Hero slides">
          {slides.map((item, index) => <button key={item.eyebrow} className={index === activeSlide ? "is-active" : ""} onClick={() => setActiveSlide(index)} aria-label={`Show ${item.eyebrow}`} />)}
        </div>
      </div>
    </section>
  );
}