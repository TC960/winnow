"use client";

import { useState } from "react";
import { ShieldCheck } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/cn";

// Fidelity check for Trace mode. One click asks Claude the current question TWICE
// in parallel: once against the full uncompressed history, once against the compact
// (packed) history. The two answers sit side by side. Matching answers are the
// proof that the keep / summarize / tombstone pack did not drop the buried detail.
//
// This reuses the Compare-tab parallel-Q&A endpoint (/api/qa) unchanged: raw =
// full history, compressed = compact packed context.

type Pair = { question: string; raw?: string; compact?: string };

function normalize(s: string) {
  return s.toLowerCase().replace(/[^a-z0-9\s]/g, "").replace(/\s+/g, " ").trim();
}

export function VerifyPanel() {
  const compactText = useStore((s) => s.traceCompactText);
  const rawText = useStore((s) => s.traceRawText);
  const goal = useStore((s) => s.traceGoal);
  const model = useStore((s) => s.model);

  const [pair, setPair] = useState<Pair | null>(null);
  const [loading, setLoading] = useState(false);

  // Only meaningful after a pack pass has actually fired (we have both contexts).
  if (!compactText || !rawText || !goal) return null;

  async function verify() {
    if (loading || !goal) return;
    setLoading(true);
    setPair({ question: goal });
    try {
      const res = await fetch("/api/qa", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question: goal, raw: rawText, compressed: compactText, model }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "verify failed");
      setPair({ question: goal, raw: data.raw.answer, compact: data.compressed.answer });
    } catch (e: any) {
      setPair({ question: goal, raw: `error: ${e.message}`, compact: `error: ${e.message}` });
    } finally {
      setLoading(false);
    }
  }

  const match =
    pair?.raw && pair?.compact ? normalize(pair.raw) === normalize(pair.compact) : null;

  return (
    <div className="px-5 py-2.5 border-b border-white/5 space-y-2.5">
      <div className="flex items-center gap-2">
        <button
          onClick={verify}
          disabled={loading}
          className="px-3 py-1.5 rounded-lg bg-keep/10 border border-keep/30 text-keep text-[11px] font-semibold flex items-center gap-1.5 hover:bg-keep/20 transition disabled:opacity-40"
        >
          <ShieldCheck className="w-3.5 h-3.5" />
          {loading ? "verifying…" : "verify fidelity"}
        </button>
        <span className="text-[10px] text-ink-faint font-mono">
          asks Claude the last question on full vs compact history
        </span>
      </div>

      <AnimatePresence>
        {pair && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-2"
          >
            <AnswerCard label="On full history" color="#ff5fb1" text={pair.raw} />
            <AnswerCard label="On compact history" color="#36f1a3" text={pair.compact} />
            {match !== null && (
              <div
                className={cn(
                  "md:col-span-2 rounded-lg px-3 py-2 text-[11px] font-mono uppercase tracking-wider text-center",
                  match
                    ? "bg-keep/10 border border-keep/30 neon-text-keep"
                    : "bg-amber-accent/10 border border-amber-accent/30 text-amber-accent"
                )}
              >
                {match ? "✓ answers match — pack preserved the answer" : "△ answers differ — inspect"}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function AnswerCard({ label, color, text }: { label: string; color: string; text?: string }) {
  return (
    <div className="rounded-lg border border-white/8 bg-white/3 p-3">
      <div className="text-[9px] font-mono uppercase tracking-wider mb-1.5" style={{ color }}>
        {label}
      </div>
      <div className="text-[13px] leading-relaxed min-h-[2.2em]">
        {text ?? <span className="text-ink-faint italic">thinking…</span>}
      </div>
    </div>
  );
}
