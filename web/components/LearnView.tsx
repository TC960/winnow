"use client";

import { useStore, totals } from "@/lib/store";
import { SourcesPanel } from "./learn/SourcesPanel";
import { ChatPanel } from "./learn/ChatPanel";
import { CompressedContextPanel } from "./learn/CompressedContextPanel";

// The Learn tab. Three-column workspace:
//   left   = sources (the live compressed transcript + user-added text/files)
//   center = streaming chat against those sources
//   right  = the actual compressed context being sent to the LLM

export function LearnView() {
  const name = useStore((s) => s.projectName);
  const desc = useStore((s) => s.projectDescription);
  const setName = useStore((s) => s.setProjectName);
  const setDesc = useStore((s) => s.setProjectDescription);
  const rows = useStore((s) => s.rows);
  const t = totals(rows);

  return (
    <div className="relative flex flex-col gap-4 flex-1 min-h-0">
      {/* Slow ambient drift so the empty space breathes instead of feeling dead */}
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        <div
          className="absolute top-[-20%] left-[20%] w-[60%] h-[60%] rounded-full blur-3xl opacity-20 animate-drift-slow"
          style={{ background: "radial-gradient(circle at center, #36f1a3 0%, transparent 65%)" }}
        />
        <div
          className="absolute bottom-[-15%] right-[10%] w-[55%] h-[55%] rounded-full blur-3xl opacity-15 animate-drift-slower"
          style={{ background: "radial-gradient(circle at center, #6ee7ff 0%, transparent 65%)" }}
        />
      </div>

      {/* Project header — editable like a Claude Project */}
      <header className="glass-strong rounded-2xl p-5">
        <div className="flex items-start gap-4">
          <div className="flex-1 min-w-0">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="bg-transparent text-xl font-bold tracking-tight text-ink focus:outline-none w-full"
            />
            <input
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              className="bg-transparent text-[13px] text-ink-dim mt-0.5 focus:outline-none w-full"
            />
          </div>
          <div className="flex items-center gap-5 text-right">
            <Stat label="utterances" value={t.utterances} />
            <Stat label="tokens in context" value={t.compressed} />
            <Stat label="saved" value={`${(t.pctSaved * 100).toFixed(0)}%`} accent />
          </div>
        </div>
      </header>

      {/* Three-column grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1 min-h-[520px] h-[65vh]">
        <div className="lg:col-span-3 min-h-0"><SourcesPanel /></div>
        <div className="lg:col-span-6 min-h-0"><ChatPanel /></div>
        <div className="lg:col-span-3 min-h-0"><CompressedContextPanel /></div>
      </div>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <div>
      <div className="text-[10px] font-mono uppercase tracking-wider text-ink-faint">{label}</div>
      <div className={`text-lg font-bold tabular-nums ${accent ? "neon-text-keep" : "text-ink"}`}>
        {typeof value === "number" ? value.toLocaleString() : value}
      </div>
    </div>
  );
}
