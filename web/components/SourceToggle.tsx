"use client";

import { Mic, FileAudio } from "lucide-react";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/cn";
import { startPipeline, stopPipeline } from "@/lib/pipeline";

// THE stage-safety button. One click hot-swaps the input source against the
// SAME downstream pipeline. If the live mic fails on stage, this gets you to
// a working demo in under a second.

export function SourceToggle() {
  const source = useStore((s) => s.source);
  const status = useStore((s) => s.status);
  const language = useStore((s) => s.language);
  const diarize = useStore((s) => s.diarize);
  const running = status === "open" || status === "connecting";

  async function pick(kind: "live" | "recorded") {
    if (kind === source && running) return;
    await stopPipeline();
    if (!running) {
      // Toggle only; user will press Start to actually run.
      useStore.getState().setSource(kind);
      return;
    }
    await startPipeline(kind, { language, diarize });
  }

  async function toggleRun() {
    if (running) {
      await stopPipeline();
    } else {
      useStore.getState().reset();
      await startPipeline(source, { language, diarize });
    }
  }

  return (
    <div className="flex items-center gap-2">
      <div className="glass rounded-full p-1 flex">
        <SegBtn active={source === "live"} onClick={() => pick("live")}>
          <Mic className="w-3.5 h-3.5" /> Live mic
        </SegBtn>
        <SegBtn active={source === "recorded"} onClick={() => pick("recorded")}>
          <FileAudio className="w-3.5 h-3.5" /> Recorded
        </SegBtn>
      </div>
      <button
        onClick={toggleRun}
        className={cn(
          "px-4 py-1.5 rounded-full text-sm font-semibold transition-all",
          running
            ? "bg-raw/15 text-raw border border-raw/40 hover:bg-raw/25"
            : "bg-keep/15 text-keep border border-keep/40 hover:bg-keep/25 neon-text-keep"
        )}
      >
        {running ? "Stop" : "Start"}
      </button>
      <StatusDot status={status} />
    </div>
  );
}

function SegBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "px-3 py-1.5 rounded-full text-[12px] font-medium tracking-wide flex items-center gap-1.5 transition-all",
        active ? "bg-white/10 text-ink" : "text-ink-dim hover:text-ink"
      )}
    >
      {children}
    </button>
  );
}

function StatusDot({ status }: { status: string }) {
  const color =
    status === "open" ? "#36f1a3" :
    status === "connecting" ? "#ffd166" :
    status === "error" ? "#ff5fb1" :
    "#5b6479";
  return (
    <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-ink-faint">
      <span
        className="w-1.5 h-1.5 rounded-full"
        style={{ background: color, boxShadow: status === "open" ? `0 0 8px ${color}` : "none" }}
      />
      {status}
    </div>
  );
}
