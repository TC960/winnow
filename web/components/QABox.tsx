"use client";

import { useState } from "react";
import { Send, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/cn";

// Proof-of-fidelity Q&A. Fires the user's question against BOTH the raw and
// compressed transcript via /api/qa (parallel calls). Side-by-side answers
// prove compression preserved the buried detail.
//
// Includes preset probes (#11) — one-click tests of specific facts in the
// hand-authored fixture. On stage you don't have to think of a question.

const PRESETS = [
  "Who has the authority to approve the budget exception?",
  "What is the exact deadline mentioned for the deliverables?",
  "How many incident reports were filed last quarter?",
];

export function QABox() {
  const rows = useStore((s) => s.rows);
  const model = useStore((s) => s.model);
  const qaPair = useStore((s) => s.qaPair);
  const setQa = useStore((s) => s.setQa);
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);

  const rawText = rows.map((r) => r.raw.text).join(" ");
  const compressedText = rows
    .filter((r) => r.compressed)
    .map((r) => r.compressed!.text)
    .join(" ");

  async function ask(question: string) {
    if (!question.trim() || loading) return;
    setLoading(true);
    setQa({ question, rawAnswer: undefined, compressedAnswer: undefined });
    try {
      const res = await fetch("/api/qa", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question, raw: rawText, compressed: compressedText, model }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "qa failed");
      setQa({
        question,
        rawAnswer: data.raw.answer,
        compressedAnswer: data.compressed.answer,
      });
    } catch (e: any) {
      setQa({ question, rawAnswer: `error: ${e.message}`, compressedAnswer: `error: ${e.message}` });
    } finally {
      setLoading(false);
    }
  }

  const canAsk = rows.length > 0;
  const match =
    qaPair?.rawAnswer && qaPair?.compressedAnswer
      ? normalize(qaPair.rawAnswer) === normalize(qaPair.compressedAnswer)
      : null;

  return (
    <section className="glass-strong rounded-2xl p-5 space-y-4">
      <div className="flex items-center gap-2">
        <Sparkles className="w-4 h-4 text-cyan-accent" />
        <h2 className="text-sm font-semibold tracking-wide">PROOF OF FIDELITY · A/B Q&A</h2>
        <span className="text-[10px] text-ink-faint font-mono ml-auto">
          ask the same question against raw vs compressed — answers should match
        </span>
      </div>

      <div className="flex items-center gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask(q)}
          placeholder={canAsk ? "Ask a question whose answer is buried in the transcript…" : "Run a session first."}
          disabled={!canAsk || loading}
          className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-[14px] text-ink placeholder:text-ink-faint focus:outline-none focus:border-cyan-accent/50 disabled:opacity-50"
        />
        <button
          onClick={() => ask(q)}
          disabled={!canAsk || loading || !q.trim()}
          className="px-4 py-2.5 rounded-xl bg-cyan-accent/15 border border-cyan-accent/40 text-cyan-accent text-sm font-semibold flex items-center gap-2 hover:bg-cyan-accent/25 transition disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Send className="w-3.5 h-3.5" /> Ask
        </button>
      </div>

      <div className="flex gap-2 flex-wrap">
        {PRESETS.map((p) => (
          <button
            key={p}
            onClick={() => { setQ(p); ask(p); }}
            disabled={!canAsk || loading}
            className="text-[11px] px-3 py-1 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-ink-dim hover:text-ink transition disabled:opacity-40"
          >
            {p}
          </button>
        ))}
      </div>

      {qaPair && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-3"
        >
          <Answer label="From raw" color="#ff5fb1" text={qaPair.rawAnswer} />
          <Answer label="From compressed" color="#36f1a3" text={qaPair.compressedAnswer} />
          <div className="md:col-span-2">
            <Verdict match={match} loading={loading} />
          </div>
        </motion.div>
      )}
    </section>
  );
}

function Answer({ label, color, text }: { label: string; color: string; text?: string }) {
  return (
    <div className="rounded-xl border border-white/8 bg-white/3 p-4">
      <div className="text-[10px] font-mono uppercase tracking-wider mb-2" style={{ color }}>
        {label}
      </div>
      <div className="text-[14px] leading-relaxed min-h-[2.5em]">
        {text ?? <span className="text-ink-faint italic">thinking…</span>}
      </div>
    </div>
  );
}

function Verdict({ match, loading }: { match: boolean | null; loading: boolean }) {
  if (loading || match === null) return null;
  return (
    <div
      className={cn(
        "rounded-xl px-4 py-3 text-[12px] font-mono uppercase tracking-wider text-center",
        match ? "bg-keep/10 border border-keep/30 neon-text-keep" : "bg-amber-accent/10 border border-amber-accent/30 text-amber-accent"
      )}
    >
      {match ? "✓ answers match — no information lost" : "△ answers differ — inspect"}
    </div>
  );
}

function normalize(s: string) {
  return s.toLowerCase().replace(/[^a-z0-9\s]/g, "").replace(/\s+/g, " ").trim();
}
