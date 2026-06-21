"use client";

import { useEffect, useRef } from "react";
import confetti from "canvas-confetti";
import { useStore, totals } from "@/lib/store";
import { inputCost, fmtUsd, fmtPct, MODELS } from "@/lib/tokens";
import { AnimatedNumber } from "./AnimatedNumber";
import { Sparkline } from "./Sparkline";

// The headline number bar. Five live stats:
//   1. raw tokens (origin)
//   2. compressed tokens
//   3. % saved + sparkline of per-utterance ratios
//   4. $ saved for the currently selected model
//   5. avg compression latency
//
// Sixth, smaller projection: cost-per-hour at current rate — makes the savings
// feel real ("you'd save $X/month at 8 hrs/day").

export function StatsBar() {
  const rows = useStore((s) => s.rows);
  const model = useStore((s) => s.model);
  const t = totals(rows);
  const savedDollars = inputCost(t.saved, model);
  const savedTokensPerHour = estimatePerHour(t.saved, rows);
  const savedDollarsPerHour = inputCost(savedTokensPerHour, model);

  // Confetti pulse on each new ready row.
  const lastCount = useRef(0);
  useEffect(() => {
    const ready = rows.filter((r) => r.state === "ready").length;
    if (ready > lastCount.current && ready > 0) {
      confetti({
        particleCount: 14,
        spread: 30,
        origin: { y: 0.35, x: 0.5 },
        colors: ["#36f1a3", "#6ee7ff"],
        startVelocity: 18,
        gravity: 0.6,
        scalar: 0.55,
        ticks: 80,
      });
    }
    lastCount.current = ready;
  }, [rows]);

  return (
    <div className="glass rounded-2xl px-6 py-4 grid grid-cols-2 md:grid-cols-5 gap-6">
      <Stat label="Raw tokens" value={t.origin} color="#ff5fb1" />
      <Stat label="Compressed" value={t.compressed} color="#36f1a3" />
      <div>
        <Label>Saved</Label>
        <div className="flex items-baseline gap-2 mt-1">
          <AnimatedNumber
            value={t.saved}
            format={(n) => Math.round(n).toLocaleString()}
            className="text-2xl font-bold neon-text-keep tabular-nums"
          />
          <span className="text-keep/80 text-sm font-mono">
            ({fmtPct(t.pctSaved)})
          </span>
        </div>
        <div className="mt-1.5">
          <Sparkline data={t.ratios.map((r) => 1 - r)} />
        </div>
      </div>
      <div>
        <Label>$ saved — {MODELS[model].label.replace("Claude ", "")}</Label>
        <div className="flex items-baseline gap-2 mt-1">
          <AnimatedNumber
            value={savedDollars}
            format={(n) => fmtUsd(n)}
            className="text-2xl font-bold text-amber-accent tabular-nums"
          />
        </div>
        <div className="text-[10px] text-ink-faint font-mono mt-1">
          ≈ {fmtUsd(savedDollarsPerHour)}/hr at current pace
        </div>
      </div>
      <div>
        <Label>Pipeline</Label>
        <div className="flex items-baseline gap-2 mt-1">
          <AnimatedNumber
            value={t.avgLatency}
            format={(n) => `${Math.round(n)}ms`}
            className="text-2xl font-bold text-cyan-accent tabular-nums"
          />
        </div>
        <div className="text-[10px] text-ink-faint font-mono mt-1">
          avg per utterance · {t.utterances} total
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <Label>{label}</Label>
      <div className="flex items-baseline gap-2 mt-1">
        <AnimatedNumber
          value={value}
          format={(n) => Math.round(n).toLocaleString()}
          className="text-2xl font-bold tabular-nums"
        />
      </div>
      <div className="h-[2px] mt-2 rounded-full" style={{ background: color, opacity: 0.35 }} />
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] font-mono uppercase tracking-wider text-ink-faint">
      {children}
    </div>
  );
}

// Project current saved-tokens out to per-hour at the observed rate.
function estimatePerHour(savedTokens: number, rows: Array<{ raw: { startMs: number; endMs: number } }>) {
  if (rows.length < 2) return 0;
  const first = rows[0].raw.startMs;
  const last = rows[rows.length - 1].raw.endMs;
  const elapsedHours = Math.max(0.0001, (last - first) / 3_600_000);
  return savedTokens > 0 ? savedTokens / elapsedHours : 0;
}
