"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef } from "react";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/cn";

// Raw transcript column. Each utterance lands as a finalized block; the running
// partial (interim Deepgram result) shows as a soft, dimmed tail.
// When word-level keep/drop labels arrive, dropped words are struck through —
// that's the visual proof of what compression cut.

const SPEAKER_COLORS = ["#ff5fb1", "#6ee7ff", "#ffd166", "#36f1a3", "#c084fc"];

export function TranscriptColumn() {
  const rows = useStore((s) => s.rows);
  const interim = useStore((s) => s.interim);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [rows, interim]);

  return (
    <section className="glass neon-border-raw rounded-2xl flex flex-col h-full overflow-hidden">
      <header className="flex items-center justify-between px-5 py-3 border-b border-white/5">
        <div className="flex items-center gap-2.5">
          <span className="w-2 h-2 rounded-full bg-raw animate-pulse-glow" />
          <h2 className="text-sm font-semibold tracking-wide text-ink">RAW TRANSCRIPT</h2>
        </div>
        <span className="text-[11px] text-ink-faint font-mono uppercase tracking-wider">
          {rows.length} utt
        </span>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto scroll-soft px-5 py-4 space-y-3.5">
        <AnimatePresence initial={false}>
          {rows.map((r) => {
            const labels = r.compressed?.wordLabels;
            const speakerColor = r.raw.speaker !== undefined
              ? SPEAKER_COLORS[r.raw.speaker % SPEAKER_COLORS.length]
              : undefined;

            return (
              <motion.div
                key={r.id}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="text-[15px] leading-relaxed"
              >
                {speakerColor && (
                  <span
                    className="inline-block text-[10px] font-mono uppercase tracking-wider mr-2 px-1.5 py-0.5 rounded"
                    style={{ color: speakerColor, background: `${speakerColor}14` }}
                  >
                    S{r.raw.speaker}
                  </span>
                )}
                {Array.isArray(labels) && labels.length > 0 && Array.isArray(labels[0]) ? (
                  <RenderWithLabels labels={labels as [string, number][]} />
                ) : (
                  <span className="text-ink">{r.raw.text}</span>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>

        {interim && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-[15px] leading-relaxed text-ink-dim italic"
          >
            {interim}
            <span className="inline-block w-1.5 h-4 ml-1 bg-raw align-middle animate-pulse-glow" />
          </motion.div>
        )}
      </div>
    </section>
  );
}

function RenderWithLabels({ labels }: { labels: [string, number][] }) {
  return (
    <span>
      {labels.map(([word, keep], i) => (
        <span
          key={i}
          className={cn(
            "transition-colors",
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
