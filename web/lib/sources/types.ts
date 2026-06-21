// The contract every input source obeys. The pipeline ONLY talks to this.
// Swapping live mic for a recorded fixture is a one-line change at the call site.

export type Word = {
  text: string;
  confidence?: number;   // Deepgram per-word confidence in [0,1]
  speaker?: number;      // Diarization speaker index
  startMs?: number;
};

export type Utterance = {
  id: string;
  text: string;
  startMs: number;
  endMs: number;
  confidence?: number;
  speaker?: number;
  words?: Word[];
};

export type SourceEvent =
  | { type: "partial"; text: string }
  | { type: "utterance"; utterance: Utterance }
  | { type: "error"; error: Error }
  | { type: "status"; status: "connecting" | "open" | "closed" }
  | { type: "ended" };

export type SourceListener = (e: SourceEvent) => void;

export interface TranscriptSource {
  readonly kind: "live" | "recorded";
  start(): Promise<void>;
  stop(): Promise<void>;
  subscribe(listener: SourceListener): () => void;
}

export type SourceConfig = {
  language?: string;          // BCP-47, e.g. "en-US", "es", "multi"
  diarize?: boolean;
  fixtureUrl?: string;        // recorded source: where to fetch the JSON
};
