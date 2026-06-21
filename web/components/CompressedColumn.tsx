"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef } from "react";
import { useStore } from "@/lib/store";
import { Loader2 } from "lucide-react";

// Compressed column. Each row lights up green once the LLMLingua-2 call
// returns. We show the per-utterance ratio and the round-trip latency
// (proves the pipeline is fast enough to actually use live).

export function CompressedColumn() {
  const rows = useStore((s) => s.rows);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [rows]);

  return (
    <section className="glass neon-border-keep rounded-2xl flex flex-col h-full overflow-hidden">
      <header className="flex items-center justify-between px-5 py-3 border-b border-white/5">
        <div className="flex items-center gap-2.5">
          <span className="w-2 h-2 rounded-full bg-keep animate-pulse-glow" />
          <h2 className="text-sm font-semibold tracking-wide text-ink">COMPRESSED</h2>
        </div>
        <span className="text-[11px] text-ink-faint font-mono uppercase tracking-wider">
          LLMLingua-2
        </span>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto scroll-soft px-5 py-4 space-y-3.5">
        <AnimatePresence initial={false}>
          {rows.map((r) => (
            <motion.div
              key={r.id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25 }}
              className="text-[15px] leading-relaxed"
            >
              {r.state === "compressing" && (
                <div className="flex items-center gap-2 text-ink-faint italic">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>compressing...</span>
                </div>
              )}
              {r.state === "error" && (
                <div className="text-raw text-sm">⚠ compression failed: {r.error}</div>
              )}
              {r.state === "ready" && r.compressed && (
                <div>
                  <span className="text-ink">{r.compressed.text}</span>
                  <div className="mt-1 flex items-center gap-2 text-[10px] font-mono uppercase tracking-wider text-ink-faint">
                    <span className="neon-text-keep">{r.compressed.ratio}</span>
                    <span>·</span>
                    <span>{r.compressed.originTokens}→{r.compressed.compressedTokens} tok</span>
                    {r.compressed.latencyMs !== undefined && (
                      <>
                        <span>·</span>
                        <span>{r.compressed.latencyMs}ms</span>
                      </>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
        {rows.length === 0 && (
          <div className="text-ink-faint text-sm italic">Waiting for the first utterance…</div>
        )}
      </div>
    </section>
  );
}
