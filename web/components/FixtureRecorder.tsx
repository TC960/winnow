"use client";

import { Circle, StopCircle, Download } from "lucide-react";
import { useStore } from "@/lib/store";

// Records the current LIVE session's Utterance stream and lets the user save
// it as the recorded-source fixture. Hit it once before going on stage and
// you have a literal replay of yourself as the backup demo.

export function FixtureRecorder() {
  const recording = useStore((s) => s.recordingFixture);
  const buffer = useStore((s) => s.recordedBuffer);
  const source = useStore((s) => s.source);
  const start = useStore((s) => s.startRecording);
  const stop = useStore((s) => s.stopRecording);

  function download() {
    const blob = new Blob([JSON.stringify(buffer, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `winnow-fixture-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const canRecord = source === "live";

  return (
    <div className="flex items-center gap-2">
      {!recording ? (
        <button
          onClick={start}
          disabled={!canRecord}
          title={canRecord ? "Capture this live session as a JSON fixture" : "Switch to live mic to record"}
          className="px-3 py-1.5 rounded-full text-[11px] font-mono uppercase tracking-wider bg-white/5 border border-white/10 text-ink-dim hover:text-ink hover:bg-white/10 transition flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Circle className="w-3 h-3" /> rec fixture
        </button>
      ) : (
        <button
          onClick={() => stop()}
          className="px-3 py-1.5 rounded-full text-[11px] font-mono uppercase tracking-wider bg-raw/15 border border-raw/40 text-raw hover:bg-raw/25 transition flex items-center gap-1.5 animate-pulse-glow"
        >
          <StopCircle className="w-3 h-3" /> stop rec ({buffer.length})
        </button>
      )}
      {buffer.length > 0 && !recording && (
        <button
          onClick={download}
          className="px-3 py-1.5 rounded-full text-[11px] font-mono uppercase tracking-wider bg-keep/15 border border-keep/40 text-keep hover:bg-keep/25 transition flex items-center gap-1.5"
        >
          <Download className="w-3 h-3" /> save ({buffer.length})
        </button>
      )}
    </div>
  );
}
