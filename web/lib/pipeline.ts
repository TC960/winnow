"use client";

import { createSource, type SourceKind } from "./sources";
import type { SourceConfig, TranscriptSource } from "./sources/types";
import { useStore } from "./store";
import { stripFillers } from "./tokens";

// One pipeline at a time. Swapping kind tears down the old source and brings up
// the new one against the SAME store + SAME compression call — that's the whole
// point of the swappable interface.

let active: TranscriptSource | null = null;

export async function startPipeline(kind: SourceKind, cfg: SourceConfig = {}) {
  await stopPipeline();
  const src = createSource(kind, cfg);
  active = src;

  const s = useStore.getState();
  s.setSource(kind);
  s.setStatus("connecting");

  src.subscribe((evt) => {
    const store = useStore.getState();
    if (evt.type === "status") store.setStatus(evt.status);
    else if (evt.type === "partial") store.setInterim(evt.text);
    else if (evt.type === "utterance") {
      // 1. Optimistically show the row immediately on the raw side.
      store.appendRow(evt.utterance);
      // 2. Kick the compression call in parallel with the next utterance.
      void compressOne(evt.utterance.id, evt.utterance.text);
    } else if (evt.type === "error") {
      console.error("[source]", evt.error);
      store.setStatus("error");
    } else if (evt.type === "ended") {
      store.setStatus("closed");
    }
  });

  await src.start();
}

export async function stopPipeline() {
  if (active) {
    const a = active;
    active = null;
    try { await a.stop(); } catch {}
  }
}

async function compressOne(id: string, text: string) {
  const store = useStore.getState();
  const t0 = performance.now();
  const sendText = store.stripFillersFirst ? stripFillers(text) : text;
  try {
    const res = await fetch("/api/compress", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text: sendText, rate: store.rate, return_labels: true }),
    });
    if (!res.ok) throw new Error(`compress ${res.status}`);
    const data = await res.json();
    const latencyMs = Math.round(performance.now() - t0);
    store.attachCompressed(id, {
      text: data.compressed_prompt,
      originTokens: data.origin_tokens,
      compressedTokens: data.compressed_tokens,
      rate: data.rate,
      ratio: data.ratio,
      wordLabels: data.word_labels,
      latencyMs,
    });
  } catch (err: any) {
    store.failRow(id, err?.message ?? "compress failed");
  }
}
