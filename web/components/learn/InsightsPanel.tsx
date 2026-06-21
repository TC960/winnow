"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, FileText, CheckSquare, ListChecks, Layers, BookA, Loader2, RefreshCw, RotateCw } from "lucide-react";
import { useStore } from "@/lib/store";
import type { Flashcard, GlossaryItem, Insights } from "@/lib/store";
import { cn } from "@/lib/cn";

type ActionId = "summary" | "decisions" | "actions" | "flashcards" | "glossary";

const ACTIONS: { id: ActionId; label: string; hint: string; icon: React.ReactNode; tint: string }[] = [
  { id: "summary",    label: "Summary",     hint: "3-5 sentence digest",       icon: <FileText className="w-4 h-4" />,    tint: "#6ee7ff" },
  { id: "decisions",  label: "Decisions",   hint: "What was resolved",         icon: <CheckSquare className="w-4 h-4" />, tint: "#36f1a3" },
  { id: "actions",    label: "Action items",hint: "Who does what next",        icon: <ListChecks className="w-4 h-4" />,  tint: "#ffd166" },
  { id: "flashcards", label: "Flashcards",  hint: "Study cards, flippable",    icon: <Layers className="w-4 h-4" />,      tint: "#ff5fb1" },
  { id: "glossary",   label: "Glossary",    hint: "Terms & names defined",     icon: <BookA className="w-4 h-4" />,       tint: "#c084fc" },
];

// One-click "study tools" panel. Each button fires a structured JSON action
// against the same sources bundle the chat uses. Results cached in store so
// flipping between them is instant after the first generate.

export function InsightsPanel() {
  const [open, setOpen] = useState<ActionId | null>(null);
  const insights = useStore((s) => s.insights);
  const loading = useStore((s) => s.insightLoading);
  const setInsight = useStore((s) => s.setInsight);
  const setLoading = useStore((s) => s.setInsightLoading);
  const rows = useStore((s) => s.rows);
  const extras = useStore((s) => s.extraSources);
  const model = useStore((s) => s.model);

  async function generate(id: ActionId) {
    setOpen(id);
    setLoading(id, true);
    try {
      const compressedText = rows.filter((r) => r.compressed).map((r) => r.compressed!.text).join(" ");
      const sources = [
        { title: "Compressed transcript", content: compressedText || "(empty)" },
        ...extras.map((s) => ({ title: s.title, content: s.content })),
      ];
      const res = await fetch("/api/project-action", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action: id, sources, model }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "action failed");
      const r = data.result;
      if (id === "summary")    setInsight("summary",    r.summary);
      if (id === "decisions")  setInsight("decisions",  r.decisions);
      if (id === "actions")    setInsight("actions",    r.actions);
      if (id === "flashcards") setInsight("flashcards", r.flashcards);
      if (id === "glossary")   setInsight("glossary",   r.glossary);
    } catch (e: any) {
      console.error(e);
    } finally {
      setLoading(id, false);
    }
  }

  function openCached(id: ActionId) {
    setOpen(id);
    if (insights[id] === undefined) generate(id);
  }

  return (
    <section className="glass rounded-2xl flex flex-col h-full overflow-hidden">
      <header className="flex items-center gap-2.5 px-5 py-3 border-b border-white/5">
        <Sparkles className="w-4 h-4 text-amber-accent" />
        <h2 className="text-sm font-semibold tracking-wide">INSIGHTS</h2>
      </header>

      <div className="p-3 grid grid-cols-2 gap-2">
        {ACTIONS.map((a) => {
          const has = insights[a.id] !== undefined;
          const isLoading = !!loading[a.id];
          const isOpen = open === a.id;
          return (
            <button
              key={a.id}
              onClick={() => openCached(a.id)}
              className={cn(
                "relative rounded-xl border p-3 text-left transition-all overflow-hidden group",
                isOpen
                  ? "border-white/25 bg-white/[0.06]"
                  : "border-white/10 bg-white/[0.025] hover:bg-white/[0.05] hover:border-white/20 hover:-translate-y-[1px]"
              )}
              style={
                isOpen
                  ? { boxShadow: `inset 0 0 0 1px ${a.tint}30, 0 0 20px ${a.tint}10` }
                  : undefined
              }
            >
              <span
                className="absolute -top-6 -right-6 w-16 h-16 rounded-full blur-2xl opacity-0 group-hover:opacity-60 transition-opacity"
                style={{ background: a.tint }}
              />
              <div className="relative flex items-start justify-between gap-2">
                <div className="flex flex-col gap-1.5">
                  <div style={{ color: a.tint }}>{a.icon}</div>
                  <div className="text-[12px] font-semibold text-ink leading-tight">{a.label}</div>
                  <div className="text-[10px] text-ink-faint leading-snug">{a.hint}</div>
                </div>
                <div className="shrink-0">
                  {isLoading ? (
                    <Loader2 className="w-3 h-3 text-ink-faint animate-spin" />
                  ) : has ? (
                    <span
                      className="w-1.5 h-1.5 rounded-full block"
                      style={{ background: a.tint, boxShadow: `0 0 8px ${a.tint}` }}
                    />
                  ) : null}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      <div className="flex-1 overflow-y-auto scroll-soft px-4 pb-4">
        <AnimatePresence mode="wait">
          {open && (
            <motion.div
              key={open}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2 }}
              className="rounded-xl border border-white/10 bg-white/3 p-4 space-y-3"
            >
              <div className="flex items-center justify-between">
                <div className="text-[10px] font-mono uppercase tracking-wider text-ink-faint">
                  {ACTIONS.find((a) => a.id === open)?.label}
                </div>
                <button
                  onClick={() => generate(open)}
                  disabled={!!loading[open]}
                  className="text-[10px] font-mono uppercase tracking-wider text-ink-faint hover:text-ink flex items-center gap-1 disabled:opacity-40"
                >
                  <RotateCw className={cn("w-2.5 h-2.5", loading[open] && "animate-spin")} /> regen
                </button>
              </div>
              <Render id={open} insights={insights} loading={!!loading[open]} />
            </motion.div>
          )}
          {!open && (
            <div className="text-[12px] text-ink-faint italic text-center py-8">
              Pick a tool above. Results stream in and stay cached.
            </div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}

function Render({ id, insights, loading }: { id: ActionId; insights: Insights; loading: boolean }) {
  if (loading && insights[id] === undefined) {
    return (
      <div className="flex items-center gap-2 text-ink-faint italic">
        <Loader2 className="w-3.5 h-3.5 animate-spin" /> generating…
      </div>
    );
  }
  if (id === "summary")    return <p className="text-[14px] leading-relaxed text-ink">{insights.summary}</p>;
  if (id === "decisions")  return <Bullets items={insights.decisions} tint="#36f1a3" />;
  if (id === "actions")    return <Bullets items={insights.actions} tint="#ffd166" />;
  if (id === "flashcards") return <Cards items={insights.flashcards} />;
  if (id === "glossary")   return <Glossary items={insights.glossary} />;
  return null;
}

function Bullets({ items, tint }: { items?: string[]; tint: string }) {
  if (!items?.length) return <div className="text-ink-faint text-sm italic">No items found.</div>;
  return (
    <ul className="space-y-2">
      {items.map((t, i) => (
        <li key={i} className="text-[14px] text-ink leading-relaxed flex gap-2">
          <span style={{ color: tint }} className="mt-0.5">▸</span>
          <span>{t}</span>
        </li>
      ))}
    </ul>
  );
}

function Cards({ items }: { items?: Flashcard[] }) {
  const [idx, setIdx] = useState(0);
  const [flipped, setFlipped] = useState(false);
  if (!items?.length) return <div className="text-ink-faint text-sm italic">No flashcards generated.</div>;
  const card = items[idx];
  return (
    <div className="space-y-3">
      <div
        onClick={() => setFlipped((f) => !f)}
        className="rounded-xl border border-white/15 bg-gradient-to-br from-white/5 to-transparent p-5 min-h-[110px] cursor-pointer select-none flex items-center justify-center text-center transition hover:border-white/25"
      >
        <div>
          <div className="text-[9px] font-mono uppercase tracking-wider text-ink-faint mb-2">
            {flipped ? "answer" : "question"} — {idx + 1} / {items.length}
          </div>
          <div className="text-[14px] text-ink leading-relaxed">
            {flipped ? card.a : card.q}
          </div>
        </div>
      </div>
      <div className="flex items-center justify-between">
        <button
          onClick={() => { setIdx((i) => (i - 1 + items.length) % items.length); setFlipped(false); }}
          className="text-[11px] font-mono uppercase tracking-wider text-ink-dim hover:text-ink"
        >
          ‹ prev
        </button>
        <button
          onClick={() => setFlipped((f) => !f)}
          className="text-[11px] font-mono uppercase tracking-wider px-3 py-1 rounded-full bg-white/5 hover:bg-white/10 text-ink-dim hover:text-ink"
        >
          flip
        </button>
        <button
          onClick={() => { setIdx((i) => (i + 1) % items.length); setFlipped(false); }}
          className="text-[11px] font-mono uppercase tracking-wider text-ink-dim hover:text-ink"
        >
          next ›
        </button>
      </div>
    </div>
  );
}

function Glossary({ items }: { items?: GlossaryItem[] }) {
  if (!items?.length) return <div className="text-ink-faint text-sm italic">No terms extracted.</div>;
  return (
    <dl className="space-y-2.5">
      {items.map((g, i) => (
        <div key={i}>
          <dt className="text-[13px] font-semibold neon-text-keep">{g.term}</dt>
          <dd className="text-[13px] text-ink-dim leading-relaxed">{g.definition}</dd>
        </div>
      ))}
    </dl>
  );
}
