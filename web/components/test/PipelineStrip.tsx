"use client";

import { Mic, AudioLines, Scissors, Box, ChevronRight } from "lucide-react";
import { useStore, totals } from "@/lib/store";

// Visual pipeline strip for the Test tab. Purely architectural — tells the
// demo's story end-to-end so a judge can see every stage at a glance:
//   Voice → Deepgram → LLMLingua-2 → Blackbox LLM
// Sub-labels update with live numbers when compression runs.

export function PipelineStrip() {
  const rows = useStore((s) => s.rows);
  const t = totals(rows);
  const utt = t.utterances;
  const orig = t.origin;
  const comp = t.compressed;
  const savedPct = t.pctSaved;

  return (
    <div className="glass rounded-2xl px-4 py-3 flex items-stretch gap-1 overflow-x-auto">
      <Stage icon={<Mic className="w-3.5 h-3.5" />} label="Voice" sub={utt > 0 ? `${utt} utt` : "ready"} tint="#ff5fb1" />
      <Arrow />
      <Stage icon={<AudioLines className="w-3.5 h-3.5" />} label="Deepgram" sub="nova-3" tint="#6ee7ff" />
      <Arrow />
      <Stage
        icon={<Scissors className="w-3.5 h-3.5" />}
        label="LLMLingua-2"
        sub={orig > 0 ? `${orig}→${comp} tok` : "token compression"}
        tint="#36f1a3"
      />
      <Arrow />
      <Stage icon={<Box className="w-3.5 h-3.5" />} label="Blackbox LLM" sub="downstream.py" tint="#10a37f" />

      {orig > 0 && (
        <div className="ml-auto self-center pl-3 border-l border-white/8">
          <div className="text-[9px] font-mono uppercase tracking-wider text-ink-faint">overall</div>
          <div className="text-[13px] font-bold tabular-nums">
            <span className="text-ink">{orig.toLocaleString()}</span>
            <span className="text-ink-faint mx-1">→</span>
            <span className="neon-text-keep">{comp.toLocaleString()}</span>
            <span className="text-ink-faint text-[10px] ml-1.5">
              ({(savedPct * 100).toFixed(0)}% saved)
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function Stage({
  icon, label, sub, tint,
}: {
  icon: React.ReactNode;
  label: string;
  sub: string;
  tint: string;
}) {
  return (
    <div
      className="shrink-0 rounded-xl px-3 py-2 flex items-center gap-2.5 border bg-white/[0.04] border-white/15"
      style={{ boxShadow: `inset 0 0 0 1px ${tint}30` }}
    >
      <span style={{ color: tint }}>{icon}</span>
      <div className="text-left leading-tight">
        <div className="text-[11px] font-semibold text-ink">{label}</div>
        <div className="text-[9px] font-mono uppercase tracking-wider text-ink-faint">{sub}</div>
      </div>
    </div>
  );
}

function Arrow() {
  return (
    <div className="shrink-0 self-center text-ink-faint">
      <ChevronRight className="w-3.5 h-3.5" />
    </div>
  );
}
