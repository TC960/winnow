"use client";

import { ChevronDown, Layers, Sparkles } from "lucide-react";
import {
  COMPRESSION_OPTIONS,
  LLM_OPTIONS,
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
  const hasHard = !!l1?.methods?.length;

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

      {/* body */}
      <div className="flex-1 flex flex-col gap-3 p-4 overflow-auto">
        {result?.error ? (
          <div className="text-[12px] font-mono text-amber-accent/90">{result.error}</div>
        ) : (
          <>
            {/* side-by-side HARD (Layer 1 token reduction) vs SOFT (Layer 2 LLM) metrics */}
            {(l1 || l2) && (
              <div className="grid grid-cols-2 gap-2">
                <Metric
                  title="hard · compression"
                  lines={
                    hasHard
                      ? [
                          `${l1?.origin_words ?? "?"}→${l1?.kept_words ?? "?"} tok · ${l1?.hard_ratio ?? "?"}×`,
                          l1?.compress_time_s != null ? `${l1.compress_time_s}s` : null,
                        ]
                      : ["none (raw)"] // no hard compression -> no compress timer
                  }
                />
                <Metric
                  title="soft · llm"
                  lines={[
                    l2 ? `${l2.input_tokens ?? "?"} in → ${l2.output_tokens ?? "?"} out` : "—",
                    l2?.kv_compression_x != null
                      ? `KV ${l2.eff_bits ?? "?"}b · ${l2.kv_compression_x}×`
                      : l2
                        ? "black-box (no KV)"
                        : null,
                    l2?.llm_time_s != null ? `${l2.llm_time_s}s` : null,
                  ]}
                />
              </div>
            )}

            <Section title="compressed context" meta={l1?.hard_ratio != null ? `${l1.hard_ratio}×` : null}>
              {loading ? <Skeleton /> : <Mono>{l1?.compressed_text ?? "—"}</Mono>}
              {l1?.note ? (
                <div className="mt-1 text-[10px] font-mono text-ink-faint">{l1.note}</div>
              ) : null}
            </Section>

            <Section title="llm output" meta={l2?.model ?? l2?.backend ?? null}>
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

function Metric({ title, lines }: { title: string; lines: (string | null)[] }) {
  const shown = lines.filter(Boolean) as string[];
  return (
    <div className="rounded-xl bg-black/20 border border-white/5 px-3 py-2">
      <div className="text-[9px] font-mono uppercase tracking-wider text-ink-faint mb-1">{title}</div>
      {shown.map((l, i) => (
        <div
          key={i}
          className={`font-mono ${i === 0 ? "text-[12px] text-keep" : "text-[11px] text-ink-dim"}`}
        >
          {l}
        </div>
      ))}
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
