"use client";

import { FileDown } from "lucide-react";
import { useStore, totals } from "@/lib/store";
import { inputCost, MODELS } from "@/lib/tokens";

// Dump the whole session — raw, compressed, stats, and any Q&A — as a JSON
// artifact. Good for sharing with judges after the demo.

export function SessionExport() {
  const rows = useStore((s) => s.rows);
  const model = useStore((s) => s.model);
  const qaPair = useStore((s) => s.qaPair);

  function exportSession() {
    const t = totals(rows);
    const session = {
      generatedAt: new Date().toISOString(),
      model: MODELS[model].label,
      stats: {
        rawTokens: t.origin,
        compressedTokens: t.compressed,
        savedTokens: t.saved,
        pctSaved: t.pctSaved,
        dollarsSaved: inputCost(t.saved, model),
        avgLatencyMs: t.avgLatency,
        utterances: t.utterances,
      },
      utterances: rows.map((r) => ({
        id: r.id,
        raw: r.raw.text,
        compressed: r.compressed?.text,
        ratio: r.compressed?.ratio,
        originTokens: r.compressed?.originTokens,
        compressedTokens: r.compressed?.compressedTokens,
        latencyMs: r.compressed?.latencyMs,
        speaker: r.raw.speaker,
      })),
      qa: qaPair,
    };
    const blob = new Blob([JSON.stringify(session, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `winnow-session-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <button
      onClick={exportSession}
      disabled={rows.length === 0}
      className="px-3 py-1.5 rounded-full text-[11px] font-mono uppercase tracking-wider bg-white/5 border border-white/10 text-ink-dim hover:text-ink hover:bg-white/10 transition flex items-center gap-1.5 disabled:opacity-40"
    >
      <FileDown className="w-3 h-3" /> export
    </button>
  );
}
