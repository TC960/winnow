// Config + option definitions for the two-panel playground compare.
// Each dropdown choice maps to the fields the /playground backend expects.

export type CompressionKey =
  | "none"
  | "llmlingua"
  | "attentionrag"
  | "both-intersection"
  | "both-union";

export type LlmKey =
  | "claude"
  | "chatgpt"
  | "qwen"
  | "qwen-quant"
  | "lclm"
  | "lclm-quant";

export type PanelConfig = {
  compression: CompressionKey;
  llm: LlmKey;
};

export const COMPRESSION_OPTIONS: { key: CompressionKey; label: string }[] = [
  { key: "none", label: "No compression (raw)" },
  { key: "llmlingua", label: "LLMLingua-2 + reranker" },
  { key: "attentionrag", label: "AttentionRAG" },
  { key: "both-intersection", label: "Both · intersection" },
  { key: "both-union", label: "Both · union" },
];

export const LLM_OPTIONS: { key: LlmKey; label: string }[] = [
  { key: "claude", label: "Claude" },
  { key: "chatgpt", label: "ChatGPT" },
  { key: "qwen", label: "Qwen" },
  { key: "qwen-quant", label: "Qwen · quantized" },
  { key: "lclm", label: "LCLM" },
  { key: "lclm-quant", label: "LCLM · quantized" },
];

const COMPRESSION_MAP: Record<CompressionKey, { methods: string[]; combine: string }> = {
  none: { methods: [], combine: "intersection" },
  llmlingua: { methods: ["llmlingua"], combine: "intersection" },
  attentionrag: { methods: ["attentionrag"], combine: "intersection" },
  "both-intersection": { methods: ["llmlingua", "attentionrag"], combine: "intersection" },
  "both-union": { methods: ["llmlingua", "attentionrag"], combine: "union" },
};

const LLM_MAP: Record<LlmKey, { backend: string; quantized: boolean }> = {
  claude: { backend: "claude", quantized: false },
  chatgpt: { backend: "chatgpt", quantized: false },
  qwen: { backend: "qwen", quantized: false },
  "qwen-quant": { backend: "qwen", quantized: true },
  lclm: { backend: "lclm", quantized: false },
  "lclm-quant": { backend: "lclm", quantized: true },
};

export type PlaygroundResult = {
  layer1?: {
    compressed_text: string;
    methods?: string[];
    combine?: string;
    origin_words?: number; // HARD compression metric (token reduction)
    kept_words?: number;
    hard_ratio?: number;
    compress_time_s?: number; // omitted by the backend when there's no hard compression
    note?: string | null;
  };
  layer2?: {
    backend: string;
    model?: string;
    text: string;
    input_tokens?: number; // post-compression token count (as seen by the LLM)
    output_tokens?: number; // post-generation token count
    llm_time_s?: number;
    quantized?: boolean;
    // SOFT-compression metrics (Qwen / LCLM TurboQuant only):
    eff_bits?: number;
    kv_compression_x?: number;
  };
  error?: string;
};

export function buildRequest(text: string, question: string, cfg: PanelConfig) {
  const c = COMPRESSION_MAP[cfg.compression];
  const l = LLM_MAP[cfg.llm];
  return {
    text,
    // The top question is the instruction handed to the downstream LLM.
    question: question.trim() || null,
    methods: c.methods,
    combine: c.combine,
    rate: 0.7,
    backend: l.backend,
    quantized: l.quantized,
    max_new_tokens: 2048, // backend caps the output budget at 2048
  };
}

// FastAPI errors come back two ways: a plain string `detail`, or — for 422
// validation failures — an ARRAY of {type, loc, msg, ...} objects. Rendering
// that array directly crashes React ("Objects are not valid as a React child"),
// so always collapse it to a readable string here.
function formatError(data: any, status: number): string {
  const d = data?.detail ?? data?.error;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    const msg = d
      .map((e) => {
        const loc = Array.isArray(e?.loc) ? e.loc.filter((x: any) => x !== "body").join(".") : "";
        return [loc, e?.msg].filter(Boolean).join(": ");
      })
      .filter(Boolean)
      .join("; ");
    return msg || `error ${status}`;
  }
  if (d && typeof d === "object") return JSON.stringify(d);
  return `error ${status}`;
}

function wordCount(s?: string): number {
  return s ? s.trim().split(/\s+/).filter(Boolean).length : 0;
}

// The backend's layer1 shape differs per compression path: LLMLingua emits
// origin_tokens/compressed_tokens, while the passthrough path emits neither.
// The panel reads a single unified set (origin_words/kept_words/hard_ratio),
// so normalize here — preferring the backend's own counts, falling back to
// word counts of the in/out text.
function normalizeLayer1(l1: any, originalText: string) {
  if (!l1) return l1;
  const origin = l1.origin_tokens ?? l1.n_words ?? wordCount(originalText);
  const kept = l1.compressed_tokens ?? l1.n_kept ?? wordCount(l1.compressed_text);
  const ratio = kept > 0 ? Math.round((origin / kept) * 10) / 10 : undefined;
  return { ...l1, origin_words: origin, kept_words: kept, hard_ratio: ratio };
}

export async function runPanel(
  text: string,
  question: string,
  cfg: PanelConfig,
): Promise<PlaygroundResult> {
  const res = await fetch("/api/playground", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(buildRequest(text, question, cfg)),
  });
  const data = await res.json();
  if (!res.ok) return { error: formatError(data, res.status) };
  if (data?.layer1) data.layer1 = normalizeLayer1(data.layer1, text);
  return data as PlaygroundResult;
}
