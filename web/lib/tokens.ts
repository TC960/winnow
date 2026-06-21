// Per-model pricing for the live cost-saved gauge. Values are $/1M tokens.
// Output prices included for honesty in projection math, but the live "saved"
// number is input-only since that's what compression actually shrinks.

export type ModelId = "sonnet-4-6" | "opus-4-7" | "haiku-4-5" | "gpt-4o";

export type ModelInfo = {
  id: ModelId;
  label: string;
  inputPerMTok: number;
  outputPerMTok: number;
  anthropicId?: string;
};

export const MODELS: Record<ModelId, ModelInfo> = {
  "sonnet-4-6": { id: "sonnet-4-6", label: "Claude Sonnet 4.6",  inputPerMTok: 3,    outputPerMTok: 15,  anthropicId: "claude-sonnet-4-6" },
  "opus-4-7":   { id: "opus-4-7",   label: "Claude Opus 4.7",    inputPerMTok: 15,   outputPerMTok: 75,  anthropicId: "claude-opus-4-7" },
  "haiku-4-5":  { id: "haiku-4-5",  label: "Claude Haiku 4.5",   inputPerMTok: 1,    outputPerMTok: 5,   anthropicId: "claude-haiku-4-5-20251001" },
  "gpt-4o":     { id: "gpt-4o",     label: "GPT-4o",             inputPerMTok: 2.5,  outputPerMTok: 10 },
};

export function inputCost(tokens: number, model: ModelId) {
  return (tokens / 1_000_000) * MODELS[model].inputPerMTok;
}

export function fmtUsd(n: number) {
  if (n < 0.01) return `$${n.toFixed(4)}`;
  if (n < 1) return `$${n.toFixed(3)}`;
  return `$${n.toFixed(2)}`;
}

export function fmtPct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}

// Cheap client-side token estimate (~4 chars/token) for the trace-pass trigger
// and the per-message token badges. The authoritative counts come from the
// backend packer; this just decides when to fire a pass and what to label rows.
export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.max(1, Math.ceil(text.trim().length / 4));
}

// Trace mode (turn-level packer) demo scaling. Real Claude windows are ~200k
// tokens, which a hand-typed Learn-tab session never reaches, so a pass would
// never be observable. We scale the effective window down so a normal chat
// crosses the 50% trigger and the packer visibly fires. Bump for production.
export const TRACE_WINDOW_TOKENS = 1500;
export const TRACE_TRIGGER_TOKENS = Math.round(TRACE_WINDOW_TOKENS * 0.5); // ~50% of window

// Naive but consistent filler stripper used for the optional prefilter step.
const FILLERS = new Set([
  "uh","um","er","erm","ah","mm","mhm","like","you","know","i","mean",
  "basically","actually","literally","sort","kind","of","right","so","well",
]);
export function stripFillers(text: string): string {
  // Drop common discourse fillers as separate tokens. Keeps content words intact.
  return text
    .split(/(\s+|[.,!?;:])/)
    .filter((tok) => {
      const w = tok.trim().toLowerCase().replace(/[.,!?;:]/g, "");
      if (!w) return true;        // preserve whitespace + punctuation
      if (/^\d/.test(w)) return true;
      return !FILLERS.has(w);
    })
    .join("")
    .replace(/\s{2,}/g, " ")
    .replace(/\s+([.,!?;:])/g, "$1")
    .trim();
}
