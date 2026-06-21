"use client";

import { useStore } from "@/lib/store";
import { DocumentCard } from "./test/DocumentCard";
import { SpeakCard } from "./test/SpeakCard";
import { PipelineStrip } from "./test/PipelineStrip";

// The Test tab. Two parallel demo flows: upload a document or speak. Each
// runs text through the compression pipeline and shows the output (with
// kept/dropped word coloring) + a one-click .md download.

export function TestView() {
  const name = useStore((s) => s.projectName);
  const desc = useStore((s) => s.projectDescription);
  const setName = useStore((s) => s.setProjectName);
  const setDesc = useStore((s) => s.setProjectDescription);

  return (
    <div className="relative flex flex-col gap-4 flex-1 min-h-0">
      {/* Slow ambient drift so the empty space breathes instead of feeling dead */}
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        <div
          className="absolute top-[-20%] left-[20%] w-[60%] h-[60%] rounded-full blur-3xl opacity-20 animate-drift-slow"
          style={{ background: "radial-gradient(circle at center, #36f1a3 0%, transparent 65%)" }}
        />
        <div
          className="absolute bottom-[-15%] right-[10%] w-[55%] h-[55%] rounded-full blur-3xl opacity-15 animate-drift-slower"
          style={{ background: "radial-gradient(circle at center, #6ee7ff 0%, transparent 65%)" }}
        />
      </div>

      <header className="glass-strong rounded-2xl p-5">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="bg-transparent text-xl font-bold tracking-tight text-ink focus:outline-none w-full"
        />
        <input
          value={desc}
          onChange={(e) => setDesc(e.target.value)}
          className="bg-transparent text-[13px] text-ink-dim mt-0.5 focus:outline-none w-full"
        />
      </header>

      {/* Pipeline strip — visual story + merge-mode / provider toggles */}
      <PipelineStrip />

      {/* Two parallel demo flows */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-[520px] h-[68vh]">
        <DocumentCard />
        <SpeakCard />
      </div>
    </div>
  );
}
