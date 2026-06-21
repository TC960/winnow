"use client";

import * as Slider from "@radix-ui/react-slider";
import { useStore } from "@/lib/store";

// Trace-pass controls + stats. The slider sets the pack token budget (same
// pattern as the live RateSlider): lower budget = more aggressive, pushing more
// turns down the keep > summarize > tombstone gradient. The stats line shows the
// last pass: tokens before and after, percent saved, and the action counts.

export function TraceBar() {
  const budget = useStore((s) => s.traceBudget);
  const setBudget = useStore((s) => s.setTraceBudget);
  const stats = useStore((s) => s.traceStats);

  return (
    <div className="flex flex-col gap-2 px-5 py-2.5 border-b border-white/5 text-[11px]">
      <div className="flex items-center gap-3">
        <span className="font-mono uppercase tracking-wider text-ink-faint">budget</span>
        <Slider.Root
          className="relative flex items-center select-none touch-none flex-1 h-5"
          value={[budget]}
          min={100}
          max={1200}
          step={50}
          onValueChange={(v) => setBudget(v[0])}
        >
          <Slider.Track className="bg-white/10 relative grow rounded-full h-[3px]">
            <Slider.Range className="absolute h-full rounded-full bg-gradient-to-r from-raw to-keep" />
          </Slider.Track>
          <Slider.Thumb
            className="block w-3.5 h-3.5 bg-ink rounded-full shadow-lg outline-none ring-2 ring-keep/30 hover:ring-keep/60 transition-all"
            aria-label="Pack token budget"
          />
        </Slider.Root>
        <span className="font-mono tabular-nums text-ink w-14 text-right">{budget} tok</span>
      </div>

      {stats ? (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono tabular-nums text-ink-dim">
          <span>
            <span className="text-ink-faint">tokens</span> {stats.before_tokens.toLocaleString()}{" "}
            <span className="text-ink-faint">&rarr;</span> {stats.after_tokens.toLocaleString()}
          </span>
          <span className="neon-text-keep">{stats.saved_pct}% saved</span>
          <span className="flex items-center gap-2">
            <Badge action="keep" n={stats.n_keep} />
            <Badge action="summarize" n={stats.n_summarize} />
            <Badge action="tombstone" n={stats.n_tombstone} />
            {stats.n_erase > 0 && <Badge action="erase" n={stats.n_erase} />}
          </span>
        </div>
      ) : (
        <div className="font-mono text-ink-faint">
          no pass yet (fires once history passes ~50% of the window)
        </div>
      )}
    </div>
  );
}

function Badge({ action, n }: { action: "keep" | "summarize" | "tombstone" | "erase"; n: number }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className={`inline-block w-1.5 h-1.5 rounded-full ${ACTION_DOT[action]}`} />
      {n} {action}
    </span>
  );
}

export const ACTION_DOT: Record<string, string> = {
  keep: "bg-keep",
  summarize: "bg-cyan-accent",
  tombstone: "bg-raw",
  erase: "bg-ink-faint",
};
