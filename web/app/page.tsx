"use client";

import { Wind } from "lucide-react";
import { useStore } from "@/lib/store";
import { Tabs } from "@/components/Tabs";
import { SourceToggle } from "@/components/SourceToggle";
import { CompareView } from "@/components/CompareView";
import { TestView } from "@/components/TestView";
import { AnimatePresence, motion } from "framer-motion";

export default function Page() {
  const tab = useStore((s) => s.tab);

  return (
    <main className="min-h-screen flex flex-col p-5 gap-5 max-w-[1600px] mx-auto">
      {/* Top bar */}
      <header className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Wind className="w-7 h-7 text-keep" />
              <div className="absolute inset-0 blur-md bg-keep/40 -z-10" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                winnow
                <span className="text-ink-faint font-mono text-xs ml-2">v0.1</span>
              </h1>
              <p className="text-[11px] text-ink-faint font-mono uppercase tracking-wider">
                voice → deepgram → llmlingua-2 → llm
              </p>
            </div>
          </div>
          <div className="hidden lg:block w-px h-8 bg-white/8" />
          <Tabs />
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <SourceToggle />
        </div>
      </header>

      {/* Body — single AnimatePresence for tab transitions */}
      <div className="flex-1 flex flex-col gap-5 min-h-0">
        <AnimatePresence mode="wait">
          {tab === "compare" ? (
            <motion.div
              key="compare"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2 }}
              className="flex-1 flex flex-col gap-5 min-h-0"
            >
              <CompareView />
            </motion.div>
          ) : (
            <motion.div
              key="test"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2 }}
              className="flex-1 flex flex-col gap-5 min-h-0"
            >
              <TestView />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <footer className="text-center text-[10px] text-ink-faint font-mono uppercase tracking-wider pt-2 pb-1">
        press <span className="kbd">?</span> for keyboard shortcuts
      </footer>
    </main>
  );
}
