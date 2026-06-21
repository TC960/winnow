// Config + option definitions for the two-panel playground compare.
// Each dropdown choice maps to the fields the /playground backend expects.

export type CompressionKey =
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
  query: string; // AttentionRAG query — only used/shown when AttentionRAG is selected
};

// AttentionRAG is involved whenever the panel runs it alone or merged with LLMLingua.
export function usesAttentionRag(cfg: PanelConfig): boolean {
  return (
    cfg.compression === "attentionrag" ||
    cfg.compression === "both-intersection" ||
    cfg.compression === "both-union"
  );
}

export const COMPRESSION_OPTIONS: { key: CompressionKey; label: string }[] = [
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
    n_words?: number;
    n_kept?: number;
    note?: string | null;
  };
  layer2?: {
    backend: string;
    model?: string;
    text: string;
    input_tokens?: number;
    output_tokens?: number;
    quantized?: boolean;
  };
  error?: string;
};

export function buildRequest(text: string, question: string, cfg: PanelConfig) {
  const c = COMPRESSION_MAP[cfg.compression];
  const l = LLM_MAP[cfg.llm];
  // When AttentionRAG runs, its dedicated per-panel query takes precedence.
  const q = usesAttentionRag(cfg) && cfg.query.trim() ? cfg.query : question;
  return {
    text,
    question: q.trim() || null,
    methods: c.methods,
    combine: c.combine,
    rate: 0.7,
    backend: l.backend,
    quantized: l.quantized,
    max_new_tokens: 256,
  };
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
  if (!res.ok) return { error: data?.detail ?? data?.error ?? `error ${res.status}` };
  return data as PlaygroundResult;
}
