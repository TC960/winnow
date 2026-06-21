"use client";

import { useRef, useState } from "react";
import { Upload, FileText, Loader2, Download, X, AlertTriangle } from "lucide-react";
import { useStore } from "@/lib/store";
import { StrikeText } from "./StrikeText";
import { downloadText, slugify } from "@/lib/downloadMd";

// Upload-a-document flow. Accepts a PDF or any text-readable file, extracts
// the text (PDF via /api/extract-pdf), pipes it through LLMLingua-2 token
// compression, and shows the result with kept-word / dropped-word coloring.
// One click downloads the compressed output as .md.

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

  async function onFile(file: File) {
    setErrorMsg(null);
    setResult(null);

    // 1) Extract text — PDF goes server-side, plain text reads in-browser.
    setStage("extracting");
    let text = "";
    let pages: number | undefined;
    try {
      if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) {
        const form = new FormData();
        form.append("file", file);
        const r = await fetch("/api/extract-pdf", { method: "POST", body: form });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || `extract failed (${r.status})`);
        text = j.text;
        pages = j.pages;
      } else {
        text = await file.text();
      }
      text = text.trim();
      if (!text) throw new Error("No extractable text in this file.");
    } catch (e: any) {
      setStage("error");
      setErrorMsg(e.message);
      return;
    }

    // 2) Compress — plain LLMLingua-2 (the merge path needs a question and
    //    multiple sources; this is a single-doc demo).
    setStage("compressing");
    const t0 = performance.now();
    try {
      const r = await fetch("/api/compress", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text, rate, return_labels: true }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || `compress failed (${r.status})`);
      setResult({
        filename: file.name,
        pages,
        originalText: text,
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
    setStage("idle");
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function download() {
    if (!result) return;
    const slug = slugify(result.filename.replace(/\.[^.]+$/, ""));
    const pct = ((1 - result.compressedTokens / Math.max(1, result.originTokens)) * 100).toFixed(1);
    const md = [
      `# ${result.filename} — compressed`,
      "",
      `_Exported ${new Date().toISOString()}_`,
      "",
      "## Stats",
      `- Tokens: ${result.originTokens} → ${result.compressedTokens} (${pct}% saved)`,
      `- LLMLingua-2 ratio: ${result.ratio}`,
      `- Compression latency: ${result.latencyMs} ms`,
      result.pages ? `- PDF pages: ${result.pages}` : "",
      "",
      "## Compressed text",
      "",
      result.compressedText,
      "",
    ].filter(Boolean).join("\n");
    downloadText(`${slug}-compressed.md`, md);
  }

  const pctSaved = result
    ? (1 - result.compressedTokens / Math.max(1, result.originTokens)) * 100
    : 0;

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
            title="Clear and upload another"
          >
            <X className="w-3 h-3" /> clear
          </button>
        )}
      </header>

      <div className="flex-1 overflow-y-auto scroll-soft p-5 space-y-4">
        {!result && stage !== "extracting" && stage !== "compressing" && (
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full rounded-xl border-2 border-dashed border-white/15 hover:border-cyan-accent/50 hover:bg-cyan-accent/5 transition py-12 flex flex-col items-center justify-center gap-3"
          >
            <FileText className="w-8 h-8 text-ink-faint" />
            <div className="text-center">
              <div className="text-[14px] font-semibold text-ink">Drop a PDF or click to browse</div>
              <div className="text-[11px] text-ink-faint mt-1">
                PDF, .txt, .md, .json, code files — compressed via LLMLingua-2
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

        {(stage === "extracting" || stage === "compressing") && (
          <div className="flex items-center justify-center py-16 gap-2 text-ink-dim italic">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>{stage === "extracting" ? "extracting text…" : "compressing…"}</span>
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
            {/* Stats row */}
            <div className="flex items-center gap-5 flex-wrap text-[11px] font-mono uppercase tracking-wider">
              <div>
                <div className="text-ink-faint">file</div>
                <div className="text-ink text-[12px] normal-case tracking-normal mt-0.5 truncate max-w-[280px]" title={result.filename}>
                  {result.filename}{result.pages ? ` · ${result.pages}p` : ""}
                </div>
              </div>
              <div>
                <div className="text-ink-faint">tokens</div>
                <div className="text-ink text-[14px] font-bold tabular-nums mt-0.5">
                  {result.originTokens.toLocaleString()}
                  <span className="text-ink-faint mx-1">→</span>
                  <span className="neon-text-keep">{result.compressedTokens.toLocaleString()}</span>
                </div>
              </div>
              <div>
                <div className="text-ink-faint">saved</div>
                <div className="neon-text-keep text-[14px] font-bold tabular-nums mt-0.5">
                  {pctSaved.toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-ink-faint">latency</div>
                <div className="text-ink text-[14px] font-bold tabular-nums mt-0.5">
                  {result.latencyMs}ms
                </div>
              </div>
              <button
                onClick={download}
                className="ml-auto text-[11px] font-mono uppercase tracking-wider px-3 py-1.5 rounded-full bg-keep/15 border border-keep/40 text-keep hover:bg-keep/25 transition flex items-center gap-1.5"
              >
                <Download className="w-3 h-3" /> download .md
              </button>
            </div>

            {/* Compressed output (with strike-through over the ORIGINAL words). */}
            <div className="rounded-xl border border-white/10 bg-white/[0.025] p-4">
              <div className="text-[10px] font-mono uppercase tracking-wider text-ink-faint mb-2">
                compressed (struck-through = dropped by LLMLingua-2)
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
