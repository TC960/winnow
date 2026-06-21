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

// Two-stage backend stats. Returned by /compress_rag and surfaced in the UI
// so the user can SEE the savings from the BGE reranker + LLMLingua-2 pass.
export type RagStats = {
  contextOriginTokens: number;
  contextCompressedTokens: number;
  originTokens: number;
  compressedTokens: number;
  rate: number;
  keptDocuments: number;
  totalDocuments: number;
  rerankerScores: number[];
};

// Stats from the last /compress call made by the chat path. When the user has
// uploaded documents, this carries the parallel LLMLingua-2 + AttentionRAG
// merge results; otherwise it carries plain LLMLingua-2 numbers and
// `mergedRan` is false (the UI gates merge-only callouts on this flag).
export type MergeStats = {
  mergedRan: boolean;                   // false => plain LLMLingua-2 only
  mode: "intersection" | "union";
  usedLlmlinguaFallback: boolean;
  wordsTotal: number;
  wordsKept: number;
  mergedRatio: string;                  // e.g. "2.41x"
  originTokens: number;
  compressedTokens: number;
  attentionRagKeptChunks?: string;      // e.g. "3/5"
  attentionRagHintPrefix?: string;
  llmlinguaRate: number;
  llmlinguaRatio: string;
};

export type MergeMode = "intersection" | "union";
export type Provider = "claude" | "chatgpt";

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
  tab: "compare" | "test";
  projectName: string;
  projectDescription: string;
  extraSources: ExtraSource[];
  chatMessages: ChatMsg[];
  insights: Insights;
  insightLoading: Partial<Record<keyof Insights, boolean>>;
  lastChatStats?: RagStats;          // back-compat: filled when /compress_rag is used
  lastChatMerge?: MergeStats;        // filled when project-chat uses LLMLingua + AttentionRAG merge
  lastCompressedPrompt?: string;     // the actual compressed text the blackbox LLM saw
  lastChatQuestion?: string;         // the user question that produced the last compressed prompt
  lastInsightStats: Partial<Record<keyof Insights, RagStats>>;
  // Learn-tab pipeline knobs (Compare tab does not read these)
  mergeMode: MergeMode;              // intersection (default) or union
  provider: Provider;                // downstream blackbox: claude or chatgpt

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

  setTab: (t: "compare" | "test") => void;
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
  setChatStats: (s?: RagStats) => void;
  setChatMerge: (m?: MergeStats) => void;
  setLastCompressed: (prompt?: string, question?: string) => void;
  setInsightStats: (k: keyof Insights, s?: RagStats) => void;
  setMergeMode: (m: MergeMode) => void;
  setProvider: (p: Provider) => void;
};

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
  lastChatStats: undefined,
  lastChatMerge: undefined,
  lastCompressedPrompt: undefined,
  lastChatQuestion: undefined,
  lastInsightStats: {},
  mergeMode: "intersection",
  provider: "claude",

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
  resetChat: () => set({ chatMessages: [] }),
  setInsight: (k, v) => set((st) => ({ insights: { ...st.insights, [k]: v } })),
  setInsightLoading: (k, b) =>
    set((st) => ({ insightLoading: { ...st.insightLoading, [k]: b } })),
  setChatStats: (s) => set({ lastChatStats: s }),
  setChatMerge: (m) => set({ lastChatMerge: m }),
  setLastCompressed: (lastCompressedPrompt, lastChatQuestion) =>
    set({ lastCompressedPrompt, lastChatQuestion }),
  setInsightStats: (k, s) =>
    set((st) => ({ lastInsightStats: { ...st.lastInsightStats, [k]: s } })),
  setMergeMode: (mergeMode) => set({ mergeMode }),
  setProvider: (provider) => set({ provider }),
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
