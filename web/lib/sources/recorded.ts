"use client";

import type { SourceConfig, SourceListener, TranscriptSource, Utterance } from "./types";

// Recorded source: fetches a JSON fixture (array of Utterance objects), then
// replays them with their ORIGINAL relative timing. Indistinguishable from
// live mic to anything downstream — same Utterance shape, same event timing
// model (partials between, then a speech_final-equivalent emission).
//
// Two ways to drive timing:
//   - "preserve": real wall-clock between original startMs values
//   - "compressed": cap pauses at 1.5s so the demo doesn't crawl
//
// Default: "compressed" — best stage feel.

export type ReplayMode = "preserve" | "compressed";

export class RecordedSource implements TranscriptSource {
  readonly kind = "recorded" as const;
  private listeners = new Set<SourceListener>();
  private cancelled = false;
  private timers: number[] = [];

  constructor(
    private cfg: SourceConfig & { mode?: ReplayMode } = { fixtureUrl: "/fixtures/demo-transcript.json", mode: "compressed" }
  ) {}

  subscribe(listener: SourceListener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
  private emit(e: Parameters<SourceListener>[0]) {
    for (const l of this.listeners) l(e);
  }

  async start() {
    this.cancelled = false;
    this.emit({ type: "status", status: "connecting" });
    const url = this.cfg.fixtureUrl ?? "/fixtures/demo-transcript.json";
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      this.emit({ type: "error", error: new Error(`Fixture load failed: ${res.status}`) });
      return;
    }
    const fixture: Utterance[] = await res.json();
    this.emit({ type: "status", status: "open" });

    const mode: ReplayMode = (this.cfg as any).mode ?? "compressed";
    const compress = (gapMs: number) => (mode === "preserve" ? gapMs : Math.min(gapMs, 1500));

    let cursor = 0;
    let prevEnd = fixture[0]?.startMs ?? 0;

    for (let i = 0; i < fixture.length; i++) {
      const u = fixture[i];
      const gap = compress(Math.max(0, u.startMs - prevEnd));
      cursor += gap;

      // Stream partials inside the utterance: chunk words at ~120ms each.
      const words = u.text.split(/\s+/);
      const perWord = Math.max(60, Math.min(180, Math.floor((u.endMs - u.startMs) / Math.max(1, words.length))));
      const partialStart = cursor;
      let acc = "";
      for (let w = 0; w < words.length; w++) {
        acc = acc ? `${acc} ${words[w]}` : words[w];
        const partial = acc;
        const fireAt = partialStart + w * perWord;
        this.schedule(fireAt, () => this.emit({ type: "partial", text: partial }));
      }

      // Final utterance fires at the end.
      const finalAt = partialStart + words.length * perWord;
      this.schedule(finalAt, () => {
        this.emit({ type: "utterance", utterance: u });
      });

      cursor = finalAt;
      prevEnd = u.endMs;
    }

    this.schedule(cursor + 200, () => this.emit({ type: "ended" }));
  }

  private schedule(atMs: number, fn: () => void) {
    const id = window.setTimeout(() => {
      if (!this.cancelled) fn();
    }, atMs);
    this.timers.push(id);
  }

  async stop() {
    this.cancelled = true;
    this.timers.forEach((t) => window.clearTimeout(t));
    this.timers = [];
    this.emit({ type: "status", status: "closed" });
    this.emit({ type: "ended" });
  }
}
