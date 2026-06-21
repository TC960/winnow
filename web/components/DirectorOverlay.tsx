"use client";

import { useEffect, useState } from "react";
import { Keyboard } from "lucide-react";
import { useStore } from "@/lib/store";
import { startPipeline, stopPipeline } from "@/lib/pipeline";

// Demo Director mode: keyboard shortcuts so you can drive the entire demo
// without touching the trackpad. Hit ? to see the cheat sheet.
//
//   Space  — start/stop the active source
//   R      — swap live ⇄ recorded (and start it)
//   1 2 3  — fire preset Q&A probes
//   X      — clear session
//   ?      — toggle this overlay

const SHORTCUTS: { keys: string; action: string }[] = [
  { keys: "Space", action: "start / stop" },
  { keys: "R", action: "swap source (live ⇄ recorded)" },
  { keys: "1 / 2 / 3", action: "fire preset Q&A probe" },
  { keys: "X", action: "clear session" },
  { keys: "?", action: "toggle this overlay" },
];

const PRESETS = [
  "Who has the authority to approve the budget exception?",
  "What is the exact deadline mentioned for the deliverables?",
  "How many incident reports were filed last quarter?",
];

export function DirectorOverlay() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      // Don't capture keys when typing in an input/textarea.
      const target = e.target as HTMLElement;
      if (target?.tagName === "INPUT" || target?.tagName === "TEXTAREA") return;

      const s = useStore.getState();
      const running = s.status === "open" || s.status === "connecting";

      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.code === "Space") {
        e.preventDefault();
        if (running) void stopPipeline();
        else {
          s.reset();
          void startPipeline(s.source, { language: s.language, diarize: s.diarize });
        }
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        const next = s.source === "live" ? "recorded" : "live";
        s.reset();
        void (async () => {
          await stopPipeline();
          await startPipeline(next, { language: s.language, diarize: s.diarize });
        })();
      } else if (e.key === "x" || e.key === "X") {
        s.reset();
      } else if (e.key === "1" || e.key === "2" || e.key === "3") {
        const idx = parseInt(e.key) - 1;
        const q = PRESETS[idx];
        void firePresetQa(q);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <>
      <button
        onClick={() => setOpen((o) => !o)}
        title="Toggle keyboard shortcuts (?)"
        className="px-3 py-1.5 rounded-full text-[11px] font-mono uppercase tracking-wider bg-white/5 border border-white/10 text-ink-dim hover:text-ink hover:bg-white/10 transition flex items-center gap-1.5"
      >
        <Keyboard className="w-3 h-3" /> director
      </button>
      {open && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center"
          onClick={() => setOpen(false)}
        >
          <div
            className="glass-strong rounded-2xl p-7 max-w-md w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2 mb-5">
              <Keyboard className="w-4 h-4 text-cyan-accent" />
              <h3 className="text-sm font-semibold tracking-wide">DIRECTOR MODE</h3>
            </div>
            <div className="space-y-2.5">
              {SHORTCUTS.map((s) => (
                <div key={s.keys} className="flex items-center justify-between">
                  <span className="kbd">{s.keys}</span>
                  <span className="text-[13px] text-ink-dim">{s.action}</span>
                </div>
              ))}
            </div>
            <div className="mt-5 text-[10px] font-mono uppercase tracking-wider text-ink-faint text-center">
              press ? again or click outside to close
            </div>
          </div>
        </div>
      )}
    </>
  );
}

async function firePresetQa(question: string) {
  const s = useStore.getState();
  if (s.rows.length === 0) return;
  const rawText = s.rows.map((r) => r.raw.text).join(" ");
  const compressedText = s.rows.filter((r) => r.compressed).map((r) => r.compressed!.text).join(" ");
  s.setQa({ question });
  try {
    const res = await fetch("/api/qa", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, raw: rawText, compressed: compressedText, model: s.model }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "qa failed");
    s.setQa({ question, rawAnswer: data.raw.answer, compressedAnswer: data.compressed.answer });
  } catch (e: any) {
    s.setQa({ question, rawAnswer: `error: ${e.message}`, compressedAnswer: `error: ${e.message}` });
  }
}
