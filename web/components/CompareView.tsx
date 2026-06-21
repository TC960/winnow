"use client";

import { useState } from "react";
import { Play, Loader2 } from "lucide-react";
import { PlaygroundPanel } from "./PlaygroundPanel";
import { runPanel, type PanelConfig, type PlaygroundResult } from "@/lib/playground";

// The Compare playground: one text input up top, then two independently-
// configured pipelines (Layer 1 compression -> Layer 2 LLM) side by side. Pick
// each panel's parameters from the dropdowns and run the same input through both.

const DEFAULT_A: PanelConfig = { compression: "llmlingua", llm: "claude" };
const DEFAULT_B: PanelConfig = { compression: "none", llm: "chatgpt" };

export function CompareView() {
  const [input, setInput] = useState("");
  const [question, setQuestion] = useState("");
  const [cfgA, setCfgA] = useState<PanelConfig>(DEFAULT_A);
  const [cfgB, setCfgB] = useState<PanelConfig>(DEFAULT_B);
  const [resA, setResA] = useState<PlaygroundResult | null>(null);
  const [resB, setResB] = useState<PlaygroundResult | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    if (!input.trim() || busy) return;
    setBusy(true);
    setResA(null);
    setResB(null);
    // `question` is the instruction handed to the LLM (shared across panels).
    const [a, b] = await Promise.all([
      runPanel(input, question, cfgA).catch((e) => ({ error: String(e) })),
      runPanel(input, question, cfgB).catch((e) => ({ error: String(e) })),
    ]);
    setResA(a);
    setResB(b);
    setBusy(false);
  }

  return (
    <div className="flex flex-col gap-5">
      {/* input */}
      <div className="glass rounded-2xl p-4 flex flex-col gap-3">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Paste the text to run through both pipelines…"
          className="w-full h-32 resize-y bg-black/20 border border-white/10 rounded-xl px-4 py-3 text-[13px] text-ink leading-relaxed placeholder:text-ink-faint focus:outline-none focus:border-keep/40"
        />
        <div className="flex items-center gap-3 flex-wrap">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Question / instruction for the LLM (always applied)…"
            className="flex-1 min-w-[220px] bg-black/20 border border-white/10 rounded-full px-4 py-2 text-[12px] text-ink placeholder:text-ink-faint focus:outline-none focus:border-keep/40"
          />
          <button
            onClick={run}
            disabled={busy || !input.trim() || !question.trim()}
            className="flex items-center gap-2 rounded-full px-5 py-2 text-[12px] font-mono uppercase tracking-wider bg-keep/20 text-keep border border-keep/30 hover:bg-keep/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
            run comparison
          </button>
        </div>
      </div>

      {/* two configurable panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <PlaygroundPanel label="pipeline A" cfg={cfgA} onChange={setCfgA} result={resA} loading={busy} />
        <PlaygroundPanel label="pipeline B" cfg={cfgB} onChange={setCfgB} result={resB} loading={busy} />
      </div>
    </div>
  );
}
