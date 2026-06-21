"use client";

import { create } from "zustand";
import type { ModelId } from "./tokens";
import type { Utterance } from "./sources/types";

// One entry per finalized utterance. Compression result attaches once /api/compress
// returns. Latency is measured client-side from utterance arrival → compression ack.
export type RowState = "compressing" | "ready" | "error";

export type Row = {
  id: string;
  raw: Utterance;
  compressed?: {
    text: string;
    originTokens: number;
    compressedTokens: number;
    rate: number;
    ratio: string;
    wordLabels?: [string, number][];   // LLMLingua-2 keep/drop per word (1=keep, 0=drop)
    latencyMs?: number;
  };
  state: RowState;
  error?: string;
};

export type SourceKind = "live" | "recorded";

// Trace mode: the per-turn action assigned by the backend packer. A gradient,
// never a hole: tombstone is the floor, erase only for must_purge content.
export type TraceAction = "keep" | "summarize" | "tombstone" | "erase";
export type TraceStats = {
  before_tokens: number;
  after_tokens: number;
  saved_pct: number;
  n_keep: number;
  n_summarize: number;
  n_tombstone: number;
  n_erase: number;
};

// Learn (Claude-Projects-style) workspace state. Lives alongside the Compare
// state so switching tabs is instant and nothing is recomputed.
export type ExtraSource = { id: string; title: string; content: string };
export type ChatMsg = { role: "user" | "assistant"; content: string; streaming?: boolean };
export type Flashcard = { q: string; a: string };
export type GlossaryItem = { term: string; definition: string };
export type Insights = {
  summary?: string;
  decisions?: string[];
  actions?: string[];
  flashcards?: Flashcard[];
  glossary?: GlossaryItem[];
};

type State = {
  rows: Row[];
  interim: string;                    // current partial transcript (raw column live tail)
  rate: number;                       // LLMLingua-2 keep rate
  model: ModelId;
  language: string;                   // BCP-47 or "multi"
  diarize: boolean;
  stripFillersFirst: boolean;         // optional pre-filter before LLMLingua
  source: SourceKind;
  status: "idle" | "connecting" | "open" | "closed" | "error";
  recordingFixture: boolean;          // when true, captured utterances are buffered for export
  recordedBuffer: Utterance[];
  qaPair?: { question: string; rawAnswer?: string; compressedAnswer?: string; streaming?: boolean };

  // Learn-tab state
  tab: "compare" | "learn";
  projectName: string;
  projectDescription: string;
  extraSources: ExtraSource[];
  chatMessages: ChatMsg[];
  insights: Insights;
  insightLoading: Partial<Record<keyof Insights, boolean>>;

  // Trace-tab state (turn-level pack over the chat history)
  traceSessionId: string;
  traceBudget: number;                              // pack token budget (slider)
  traceActions: Record<string, TraceAction>;        // turn index -> action, last pass
  tracePackedUpTo: number;                          // history length at last pass (0 = never)
  traceStats?: TraceStats;
  traceCompactText?: string;                        // compact context from last pass (verify)
  traceRawText?: string;                            // full raw history at last pass (verify)
  traceGoal?: string;                               // the question that drove the last pass

  // setters
  setRate: (r: number) => void;
  setModel: (m: ModelId) => void;
  setLanguage: (l: string) => void;
  setDiarize: (d: boolean) => void;
  setStripFillers: (s: boolean) => void;
  setSource: (k: SourceKind) => void;
  setStatus: (s: State["status"]) => void;
  setInterim: (s: string) => void;
  appendRow: (u: Utterance) => void;
  attachCompressed: (id: string, c: Row["compressed"]) => void;
  failRow: (id: string, err: string) => void;
  reset: () => void;
  startRecording: () => void;
  stopRecording: () => Utterance[];   // returns the captured buffer
  pushRecorded: (u: Utterance) => void;
  setQa: (qa: State["qaPair"]) => void;

  setTab: (t: "compare" | "learn") => void;
  setProjectName: (n: string) => void;
  setProjectDescription: (d: string) => void;
  addSource: (s: Omit<ExtraSource, "id">) => void;
  removeSource: (id: string) => void;
  appendChat: (m: ChatMsg) => void;
  updateLastChat: (delta: string) => void;
  finishLastChat: () => void;
  resetChat: () => void;
  setInsight: <K extends keyof Insights>(k: K, v: Insights[K]) => void;
  setInsightLoading: (k: keyof Insights, b: boolean) => void;

  setTraceBudget: (n: number) => void;
  setTracePack: (p: { actions: Record<string, TraceAction>; packedUpTo: number; stats: TraceStats; compactText?: string; rawText?: string; goal?: string }) => void;
  resetTrace: () => void;
};

// One trace session per page load. ingest / pack / recall share this id so they
// hit the same content-hash keyed Store on the backend.
const TRACE_SESSION_ID = `sess-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

export const useStore = create<State>()((set, get) => ({
  rows: [],
  interim: "",
  rate: 0.5,
  model: "sonnet-4-6",
  language: "en-US",
  diarize: false,
  stripFillersFirst: false,
  source: "live",
  status: "idle",
  recordingFixture: false,
  recordedBuffer: [],
  qaPair: undefined,

  tab: "compare",
  projectName: "Untitled Project",
  projectDescription: "A live workspace built around a compressed speech transcript.",
  extraSources: [],
  chatMessages: [],
  insights: {},
  insightLoading: {},

  traceSessionId: TRACE_SESSION_ID,
  traceBudget: 500,
  traceActions: {},
  tracePackedUpTo: 0,
  traceStats: undefined,

  setRate: (rate) => set({ rate }),
  setModel: (model) => set({ model }),
  setLanguage: (language) => set({ language }),
  setDiarize: (diarize) => set({ diarize }),
  setStripFillers: (stripFillersFirst) => set({ stripFillersFirst }),
  setSource: (source) => set({ source }),
  setStatus: (status) => set({ status }),
  setInterim: (interim) => set({ interim }),
  appendRow: (u) =>
    set((s) => ({
      rows: [...s.rows, { id: u.id, raw: u, state: "compressing" }],
      interim: "",
      recordedBuffer: s.recordingFixture ? [...s.recordedBuffer, u] : s.recordedBuffer,
    })),
  attachCompressed: (id, c) =>
    set((s) => ({
      rows: s.rows.map((r) => (r.id === id ? { ...r, compressed: c, state: "ready" } : r)),
    })),
  failRow: (id, err) =>
    set((s) => ({
      rows: s.rows.map((r) => (r.id === id ? { ...r, state: "error", error: err } : r)),
    })),
  reset: () => set({ rows: [], interim: "", qaPair: undefined, recordedBuffer: [] }),
  startRecording: () => set({ recordingFixture: true, recordedBuffer: [] }),
  stopRecording: () => {
    const buf = get().recordedBuffer;
    set({ recordingFixture: false });
    return buf;
  },
  pushRecorded: (u) => set((s) => ({ recordedBuffer: [...s.recordedBuffer, u] })),
  setQa: (qaPair) => set({ qaPair }),

  setTab: (tab) => set({ tab }),
  setProjectName: (projectName) => set({ projectName }),
  setProjectDescription: (projectDescription) => set({ projectDescription }),
  addSource: (s) =>
    set((st) => ({
      extraSources: [...st.extraSources, { id: `src-${Date.now()}-${Math.random().toString(36).slice(2,7)}`, ...s }],
    })),
  removeSource: (id) =>
    set((st) => ({ extraSources: st.extraSources.filter((x) => x.id !== id) })),
  appendChat: (m) => set((st) => ({ chatMessages: [...st.chatMessages, m] })),
  updateLastChat: (delta) =>
    set((st) => {
      if (st.chatMessages.length === 0) return st;
      const last = st.chatMessages[st.chatMessages.length - 1];
      const updated = { ...last, content: last.content + delta };
      return { chatMessages: [...st.chatMessages.slice(0, -1), updated] };
    }),
  finishLastChat: () =>
    set((st) => {
      if (st.chatMessages.length === 0) return st;
      const last = st.chatMessages[st.chatMessages.length - 1];
      return { chatMessages: [...st.chatMessages.slice(0, -1), { ...last, streaming: false }] };
    }),
  resetChat: () => set({ chatMessages: [], traceActions: {}, tracePackedUpTo: 0, traceStats: undefined, traceCompactText: undefined, traceRawText: undefined, traceGoal: undefined }),
  setInsight: (k, v) => set((st) => ({ insights: { ...st.insights, [k]: v } })),
  setInsightLoading: (k, b) =>
    set((st) => ({ insightLoading: { ...st.insightLoading, [k]: b } })),

  setTraceBudget: (traceBudget) => set({ traceBudget }),
  setTracePack: ({ actions, packedUpTo, stats, compactText, rawText, goal }) =>
    set({ traceActions: actions, tracePackedUpTo: packedUpTo, traceStats: stats,
          traceCompactText: compactText, traceRawText: rawText, traceGoal: goal }),
  resetTrace: () => set({ traceActions: {}, tracePackedUpTo: 0, traceStats: undefined, traceCompactText: undefined, traceRawText: undefined, traceGoal: undefined }),
}));

// Derived selectors live as plain functions so we never re-render unnecessarily.
export function totals(rows: Row[]) {
  let origin = 0;
  let compressed = 0;
  let utterances = 0;
  const ratios: number[] = [];
  const latencies: number[] = [];
  for (const r of rows) {
    if (!r.compressed) continue;
    origin += r.compressed.originTokens;
    compressed += r.compressed.compressedTokens;
    if (r.compressed.originTokens > 0) {
      ratios.push(r.compressed.compressedTokens / r.compressed.originTokens);
    }
    if (r.compressed.latencyMs) latencies.push(r.compressed.latencyMs);
    utterances += 1;
  }
  const saved = origin - compressed;
  const pctSaved = origin > 0 ? saved / origin : 0;
  const avgLatency = latencies.length ? latencies.reduce((a, b) => a + b, 0) / latencies.length : 0;
  return { origin, compressed, saved, pctSaved, utterances, ratios, avgLatency };
}
