"use client";

import { LiveMicSource } from "./live-mic";
import { RecordedSource } from "./recorded";
import type { SourceConfig, TranscriptSource } from "./types";

export type SourceKind = "live" | "recorded";

export function createSource(kind: SourceKind, cfg: SourceConfig = {}): TranscriptSource {
  return kind === "live" ? new LiveMicSource(cfg) : new RecordedSource(cfg);
}

export type { TranscriptSource, SourceConfig } from "./types";
export type { Utterance, Word, SourceEvent } from "./types";
