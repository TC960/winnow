"use client";

import { useEffect, useRef } from "react";
import { Feather } from "lucide-react";
import { useStore, totals } from "@/lib/store";
import { cn } from "@/lib/cn";

// The right-side "what the LLM actually sees" panel.
// Shows per-utterance compressed text with word-level strike-through so the
// keep/drop decision is visible at a glance, plus a footer line noting that
// extra sources are compressed on-the-fly at chat time.

export function CompressedContextPanel() {
  const rows = useStore((s) => s.rows);
  const extras = useStore((s) => s.extraSources);
  const t = totals(rows);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [rows.length]);

  const readyRows = rows.filter((r) => r.compressed);

  return (
    <section className="glass rounded-2xl flex flex-col h-full overflow-hidden">
      <header className="flex items-center justify-between px-5 py-3 border-b border-white/5">
        <div className="flex items-center gap-2.5">
          <Feather className="w-4 h-4 text-keep" />
          <h2 className="text-sm font-semibold tracking-wide">CONTEXT SENT TO LLM</h2>
        </div>
        {readyRows.length > 0 && (
          <span className="text-[10px] font-mono uppercase tracking-wider text-ink-faint">
            {t.origin.toLocaleString()}→{t.compressed.toLocaleString()} tok
            <span className="neon-text-keep ml-1.5">
              {(t.pctSaved * 100).toFixed(0)}% saved
            </span>
          </span>
        )}
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto scroll-soft px-5 py-4 space-y-3">
        {readyRows.length === 0 && extras.length === 0 && (
          <div className="text-[12px] text-ink-faint italic">
            The compressed transcript appears here as you speak. Struck-through
            words were dropped by LLMLingua-2 — kept words are what the LLM sees.
          </div>
        )}

        {readyRows.map((r) => {
          const labels = r.compressed!.wordLabels;
          const hasLabels = Array.isArray(labels) && labels.length > 0 && Array.isArray(labels[0]);
          return (
            <div key={r.id} className="text-[14px] leading-relaxed">
              {hasLabels ? (
                <RenderLabeled labels={labels as [string, number][]} />
              ) : (
                <span className="text-ink">{r.compressed!.text}</span>
              )}
              <div className="mt-1 text-[10px] font-mono uppercase tracking-wider text-ink-faint">
                <span className="neon-text-keep">{r.compressed!.ratio}</span>
                <span className="mx-1.5">·</span>
                <span>{r.compressed!.originTokens}→{r.compressed!.compressedTokens} tok</span>
              </div>
            </div>
          );
        })}

        {extras.length > 0 && (
          <div className="pt-3 mt-2 border-t border-white/5 space-y-2">
            <div className="text-[10px] font-mono uppercase tracking-wider text-ink-faint">
              {extras.length} extra source{extras.length === 1 ? "" : "s"}
              <span className="ml-1.5 normal-case tracking-normal text-ink-faint">
                · compressed on-demand when you ask
              </span>
            </div>
            {extras.map((s) => (
              <div key={s.id} className="text-[12px] text-ink-dim">
                <span className="text-ink">{s.title}</span>
                <span className="text-ink-faint ml-1.5">
                  ({s.content.length.toLocaleString()} chars)
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function RenderLabeled({ labels }: { labels: [string, number][] }) {
  return (
    <span>
      {labels.map(([word, keep], i) => (
        <span
          key={i}
          className={cn(
            keep
              ? "text-ink"
              : "text-ink-faint line-through decoration-raw/60 decoration-[1.5px]"
          )}
        >
          {word}{i < labels.length - 1 ? " " : ""}
        </span>
      ))}
    </span>
  );
}
