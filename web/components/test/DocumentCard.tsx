"use client";

import { useRef, useState } from "react";
import { Upload, FileText, Loader2, X, AlertTriangle, RotateCw } from "lucide-react";
import { useStore } from "@/lib/store";
import { StrikeText } from "./StrikeText";

// Upload-a-document flow. Extracts the file's text and runs LLMLingua-2
//   PDF text extract → LLMLingua-2 (token compression)
// over it, showing kept/dropped words. "re-compress" reruns with the current
// keep-rate without re-extracting the PDF.

const ACCEPT = ".pdf,.txt,.md,.markdown,.json,.csv,.tsv,.log,.xml,.html,.htm,.yaml,.yml,.py,.js,.ts,.tsx,.jsx,.go,.rs,.java,.c,.cpp,.h,.css,.sh,application/pdf,text/*";

type Result = {
  filename: string;
  pages?: number;
  originalText: string;
  compressedText: string;
  originTokens: number;
  compressedTokens: number;
  ratio: string;
  wordLabels?: [string, number][];
  latencyMs: number;
};

export function DocumentCard() {
  const rate = useStore((s) => s.rate);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [stage, setStage] = useState<"idle" | "extracting" | "compressing" | "done" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  // Cache the extracted text so "re-compress" doesn't re-extract the PDF.
  const [extracted, setExtracted] = useState<{ text: string; filename: string; pages?: number } | null>(null);

  async function onFile(file: File) {
    setErrorMsg(null);
    setResult(null);
    setExtracted(null);

    setStage("extracting");
    let text = "";
    let pages: number | undefined;
    try {
      if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) {
        const form = new FormData();
        form.append("file", file);
        const r = await fetch("/api/extract-pdf", { method: "POST", body: form });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || j.detail || `extract failed (${r.status})`);
        text = j.text;
        pages = j.pages;
      } else {
        text = await file.text();
      }
      text = text.trim();
      if (!text) throw new Error("No extractable text in this file.");
      setExtracted({ text, filename: file.name, pages });
    } catch (e: any) {
      setStage("error");
      setErrorMsg(e.message);
      return;
    }
    await compress({ text, filename: file.name, pages });
  }

  async function compress(ex: { text: string; filename: string; pages?: number }) {
    setStage("compressing");
    setErrorMsg(null);
    const t0 = performance.now();
    try {
      const r = await fetch("/api/compress", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          text: ex.text,
          rate,
          return_labels: true,
        }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || j.detail || `compress failed (${r.status})`);
      setResult({
        filename: ex.filename,
        pages: ex.pages,
        originalText: ex.text,
        compressedText: j.compressed_prompt,
        originTokens: j.origin_tokens,
        compressedTokens: j.compressed_tokens,
        ratio: j.ratio,
        wordLabels: j.word_labels,
        latencyMs: Math.round(performance.now() - t0),
      });
      setStage("done");
    } catch (e: any) {
      setStage("error");
      setErrorMsg(e.message);
    }
  }

  function clear() {
    setResult(null);
    setExtracted(null);
    setStage("idle");
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  const pctSaved = result ? (1 - result.compressedTokens / Math.max(1, result.originTokens)) * 100 : 0;
  const busy = stage === "extracting" || stage === "compressing";

  return (
    <section className="glass rounded-2xl flex flex-col h-full overflow-hidden">
      <header className="flex items-center justify-between px-5 py-3 border-b border-white/5">
        <div className="flex items-center gap-2.5">
          <Upload className="w-4 h-4 text-cyan-accent" />
          <h2 className="text-sm font-semibold tracking-wide">UPLOAD A DOCUMENT</h2>
        </div>
        {result && (
          <button
            onClick={clear}
            className="text-[11px] font-mono uppercase tracking-wider text-ink-faint hover:text-ink flex items-center gap-1"
          >
            <X className="w-3 h-3" /> clear
          </button>
        )}
      </header>

      {/* Re-run LLMLingua-2 on the cached extract with the current keep-rate. */}
      {extracted && (
        <div className="px-5 py-3 border-b border-white/5 flex items-center justify-end">
          <button
            onClick={() => compress(extracted)}
            disabled={busy}
            className="text-[11px] font-mono uppercase tracking-wider px-3 py-2 rounded-lg bg-keep/15 border border-keep/40 text-keep hover:bg-keep/25 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
          >
            <RotateCw className={`w-3 h-3 ${busy ? "animate-spin" : ""}`} /> re-compress
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto scroll-soft p-5 space-y-4">
        {!result && !busy && (
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full rounded-xl border-2 border-dashed border-white/15 hover:border-cyan-accent/50 hover:bg-cyan-accent/5 transition py-12 flex flex-col items-center justify-center gap-3"
          >
            <FileText className="w-8 h-8 text-ink-faint" />
            <div className="text-center">
              <div className="text-[14px] font-semibold text-ink">Drop a PDF or click to browse</div>
              <div className="text-[11px] text-ink-faint mt-1">
                Document → LLMLingua-2 token compression
              </div>
            </div>
          </button>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPT}
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
          className="hidden"
        />

        {busy && (
          <div className="flex items-center justify-center py-16 gap-2 text-ink-dim italic">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>{stage === "extracting" ? "extracting text…" : "compressing (LLMLingua-2)…"}</span>
          </div>
        )}

        {stage === "error" && errorMsg && (
          <div className="rounded-xl border border-raw/30 bg-raw/5 p-3 flex items-start gap-2 text-[13px] text-raw">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold">Something went wrong</div>
              <div className="text-[12px] mt-0.5">{errorMsg}</div>
            </div>
          </div>
        )}

        {result && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-5 gap-y-2 text-[11px] font-mono uppercase tracking-wider">
              <Cell label="file" value={`${result.filename}${result.pages ? ` · ${result.pages}p` : ""}`} small />
              <Cell label="tokens" value={
                <>
                  {result.originTokens.toLocaleString()}
                  <span className="text-ink-faint mx-1">→</span>
                  <span className="neon-text-keep">{result.compressedTokens.toLocaleString()}</span>
                </>
              } />
              <Cell label="saved" value={`${pctSaved.toFixed(0)}%`} accent />
              <Cell label="compression time" value={`${result.latencyMs} ms`} />
            </div>

            {/* Pipeline diagnostics */}
            <div className="text-[10px] font-mono uppercase tracking-wider text-ink-faint">
              pipeline: llmlingua-2
            </div>

            <div className="rounded-xl border border-white/10 bg-white/[0.025] p-4">
              <div className="text-[10px] font-mono uppercase tracking-wider text-ink-faint mb-2">
                compressed (struck-through = dropped)
              </div>
              <div className="text-[14px] leading-relaxed max-h-[420px] overflow-y-auto scroll-soft">
                <StrikeText labels={result.wordLabels} fallback={result.compressedText} />
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function Cell({ label, value, accent, small }: { label: string; value: React.ReactNode; accent?: boolean; small?: boolean }) {
  return (
    <div>
      <div className="text-ink-faint">{label}</div>
      <div className={`${small ? "text-[12px] normal-case tracking-normal" : "text-[14px] tabular-nums font-bold"} ${accent ? "neon-text-keep" : "text-ink"} mt-0.5 truncate`}>
        {value}
      </div>
    </div>
  );
}
