"use client";

import { ChevronDown, Layers, Sparkles } from "lucide-react";
import {
  COMPRESSION_OPTIONS,
  LLM_OPTIONS,
  usesAttentionRag,
  type PanelConfig,
  type PlaygroundResult,
} from "@/lib/playground";

// One comparison window: two dropdowns up top (Layer 1 compression + Layer 2
// LLM), then the compressed context and the LLM output stacked below.

export function PlaygroundPanel({
  label,
  cfg,
  onChange,
  result,
  loading,
}: {
  label: string;
  cfg: PanelConfig;
  onChange: (c: PanelConfig) => void;
  result: PlaygroundResult | null;
  loading: boolean;
}) {
  const l1 = result?.layer1;
  const l2 = result?.layer2;
  const kept =
    l1?.n_kept != null && l1?.n_words != null ? `${l1.n_kept}/${l1.n_words} words kept` : null;

  return (
    <div className="glass rounded-2xl flex flex-col min-h-[420px] overflow-hidden">
      {/* dropdown header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-white/10 flex-wrap">
        <span className="text-[10px] font-mono uppercase tracking-widest text-ink-faint">
          {label}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <Select
            icon={<Layers className="w-3 h-3" />}
            value={cfg.compression}
            onChange={(v) => onChange({ ...cfg, compression: v as PanelConfig["compression"] })}
            options={COMPRESSION_OPTIONS}
          />
          <Select
            icon={<Sparkles className="w-3 h-3" />}
            value={cfg.llm}
            onChange={(v) => onChange({ ...cfg, llm: v as PanelConfig["llm"] })}
            options={LLM_OPTIONS}
          />
        </div>
      </div>

      {/* dedicated AttentionRAG query — only when AttentionRAG is selected */}
      {usesAttentionRag(cfg) && (
        <div className="px-4 pt-3">
          <input
            value={cfg.query}
            onChange={(e) => onChange({ ...cfg, query: e.target.value })}
            placeholder="AttentionRAG query (what to focus on)…"
            className="w-full bg-black/20 border border-cyan-accent/30 rounded-full px-4 py-2 text-[12px] text-ink placeholder:text-ink-faint focus:outline-none focus:border-cyan-accent/60"
          />
        </div>
      )}

      {/* body */}
      <div className="flex-1 flex flex-col gap-3 p-4 overflow-auto">
        {result?.error ? (
          <div className="text-[12px] font-mono text-amber-accent/90">{result.error}</div>
        ) : (
          <>
            <Section title="compressed context" meta={kept}>
              {loading ? <Skeleton /> : <Mono>{l1?.compressed_text ?? "—"}</Mono>}
              {l1?.note ? (
                <div className="mt-1 text-[10px] font-mono text-ink-faint">{l1.note}</div>
              ) : null}
            </Section>

            <Section
              title="llm output"
              meta={
                l2?.output_tokens != null ? `${l2.output_tokens} out · ${l2.model ?? l2.backend}` : null
              }
            >
              {loading ? <Skeleton /> : <div className="text-[13px] text-ink leading-relaxed whitespace-pre-wrap">{l2?.text ?? "—"}</div>}
            </Section>
          </>
        )}
      </div>
    </div>
  );
}

function Select({
  value,
  onChange,
  options,
  icon,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { key: string; label: string }[];
  icon: React.ReactNode;
}) {
  return (
    <div className="relative flex items-center">
      <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-dim pointer-events-none">
        {icon}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="appearance-none glass rounded-full pl-7 pr-7 py-1.5 text-[11px] text-ink focus:outline-none cursor-pointer hover:text-keep transition-colors"
      >
        {options.map((o) => (
          <option key={o.key} value={o.key}>
            {o.label}
          </option>
        ))}
      </select>
      <ChevronDown className="w-3 h-3 absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-ink-dim" />
    </div>
  );
}

function Section({
  title,
  meta,
  children,
}: {
  title: string;
  meta?: string | null;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-mono uppercase tracking-wider text-ink-faint">{title}</span>
        {meta ? <span className="text-[10px] font-mono text-ink-dim ml-auto">{meta}</span> : null}
      </div>
      <div className="rounded-xl bg-black/20 border border-white/5 px-3 py-2.5 min-h-[64px]">
        {children}
      </div>
    </div>
  );
}

function Mono({ children }: { children: React.ReactNode }) {
  return <div className="text-[12px] font-mono text-ink-dim leading-relaxed whitespace-pre-wrap">{children}</div>;
}

function Skeleton() {
  return (
    <div className="space-y-1.5 animate-pulse">
      <div className="h-2.5 bg-white/10 rounded w-full" />
      <div className="h-2.5 bg-white/10 rounded w-[85%]" />
      <div className="h-2.5 bg-white/10 rounded w-[60%]" />
    </div>
  );
}
