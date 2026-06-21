"use client";

import { useEffect, useRef, useState } from "react";
import { Mic, MicOff, Loader2, Download, Trash2, AlertTriangle } from "lucide-react";
import { LiveMicSource } from "@/lib/sources/live-mic";
import { useStore } from "@/lib/store";
import { StrikeText } from "./StrikeText";
import { downloadText, slugify } from "@/lib/downloadMd";
import { cn } from "@/lib/cn";

// Speak-and-compress flow. Tap the mic, talk; each pause produces an
// utterance that's piped through LLMLingua-2 token compression and rendered
// with kept/dropped coloring. Cumulative stats + .md download for the
// whole compressed transcript.

type Utt = {
  id: string;
  rawText: string;
  state: "compressing" | "ready" | "error";
  compressedText?: string;
  originTokens?: number;
  compressedTokens?: number;
  ratio?: string;
  wordLabels?: [string, number][];
  latencyMs?: number;
  error?: string;
};

export function SpeakCard() {
  const rate = useStore((s) => s.rate);
  const language = useStore((s) => s.language);
  const projectName = useStore((s) => s.projectName);

  const [listening, setListening] = useState(false);
  const [partial, setPartial] = useState("");
  const [audioLevel, setAudioLevel] = useState(0);
  const [voiceErr, setVoiceErr] = useState<string | null>(null);
  const [utts, setUtts] = useState<Utt[]>([]);

  const srcRef = useRef<LiveMicSource | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => () => { stop(); /* eslint-disable-line react-hooks/exhaustive-deps */ }, []);
  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [utts]);

  async function start() {
    if (listening) return;
    setVoiceErr(null);
    try {
      const s = new LiveMicSource({ language });
      srcRef.current = s;
      s.subscribe((evt) => {
        if (evt.type === "partial") setPartial(evt.text);
        else if (evt.type === "utterance") {
          setPartial("");
          const id = evt.utterance.id;
          const rawText = evt.utterance.text.trim();
          if (!rawText) return;
          setUtts((arr) => [...arr, { id, rawText, state: "compressing" }]);
          void compress(id, rawText);
        } else if (evt.type === "error") {
          setVoiceErr(evt.error.message);
          stop();
        }
      });
      await s.start();
      setListening(true);
      attachLevelMeter();
    } catch (e: any) {
      setVoiceErr(e?.message ?? "Failed to start voice input");
      stop();
    }
  }

  function stop() {
    if (srcRef.current) {
      void srcRef.current.stop();
      srcRef.current = null;
    }
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    if (audioCtxRef.current) {
      void audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    setListening(false);
    setPartial("");
    setAudioLevel(0);
  }

  function attachLevelMeter() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      const an = ctx.createAnalyser();
      an.fftSize = 256;
      ctx.createMediaStreamSource(stream).connect(an);
      const data = new Uint8Array(an.frequencyBinCount);
      const tick = () => {
        an.getByteFrequencyData(data);
        const avg = data.reduce((a, b) => a + b, 0) / data.length;
        setAudioLevel(Math.min(1, avg / 90));
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();
    }).catch(() => {});
  }

  async function compress(id: string, text: string) {
    const t0 = performance.now();
    try {
      const r = await fetch("/api/compress", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text, rate, return_labels: true }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || `compress ${r.status}`);
      setUtts((arr) =>
        arr.map((u) =>
          u.id === id
            ? {
                ...u,
                state: "ready",
                compressedText: j.compressed_prompt,
                originTokens: j.origin_tokens,
                compressedTokens: j.compressed_tokens,
                ratio: j.ratio,
                wordLabels: j.word_labels,
                latencyMs: Math.round(performance.now() - t0),
              }
            : u
        )
      );
    } catch (e: any) {
      setUtts((arr) =>
        arr.map((u) => (u.id === id ? { ...u, state: "error", error: e.message } : u))
      );
    }
  }

  function clearAll() {
    setUtts([]);
  }

  function download() {
    const totalOrigin = utts.reduce((a, u) => a + (u.originTokens ?? 0), 0);
    const totalCompressed = utts.reduce((a, u) => a + (u.compressedTokens ?? 0), 0);
    const pct = totalOrigin > 0
      ? ((1 - totalCompressed / totalOrigin) * 100).toFixed(1)
      : "0";
    const slug = slugify(`${projectName}-spoken`);
    const compressedBody = utts
      .filter((u) => u.state === "ready" && u.compressedText)
      .map((u) => u.compressedText)
      .join(" ");
    const md = [
      `# ${projectName || "Spoken session"} — compressed`,
      "",
      `_Exported ${new Date().toISOString()}_`,
      "",
      "## Stats",
      `- Utterances: ${utts.length}`,
      `- Tokens: ${totalOrigin} → ${totalCompressed} (${pct}% saved)`,
      "",
      "## Compressed transcript",
      "",
      compressedBody,
      "",
    ].join("\n");
    downloadText(`${slug}-compressed.md`, md);
  }

  const totalOrigin = utts.reduce((a, u) => a + (u.originTokens ?? 0), 0);
  const totalCompressed = utts.reduce((a, u) => a + (u.compressedTokens ?? 0), 0);
  const pct = totalOrigin > 0 ? (1 - totalCompressed / totalOrigin) * 100 : 0;
  const readyCount = utts.filter((u) => u.state === "ready").length;

  return (
    <section className="glass rounded-2xl flex flex-col h-full overflow-hidden">
      <header className="flex items-center justify-between px-5 py-3 border-b border-white/5">
        <div className="flex items-center gap-2.5">
          <Mic className="w-4 h-4 text-raw" />
          <h2 className="text-sm font-semibold tracking-wide">SPEAK</h2>
        </div>
        {utts.length > 0 && (
          <button
            onClick={clearAll}
            className="text-[11px] font-mono uppercase tracking-wider text-ink-faint hover:text-ink flex items-center gap-1"
          >
            <Trash2 className="w-3 h-3" /> clear
          </button>
        )}
      </header>

      <div className="p-5 border-b border-white/5 flex flex-col items-center gap-4">
        <button
          onClick={() => (listening ? stop() : start())}
          className="relative w-24 h-24"
        >
          {listening && (
            <>
              <span className="absolute inset-0 rounded-full bg-raw/20 animate-ping" style={{ animationDuration: "1.8s" }} />
              <span
                className="absolute inset-0 rounded-full bg-raw/15"
                style={{ transform: `scale(${1 + audioLevel * 0.5})`, transition: "transform 80ms ease-out" }}
              />
            </>
          )}
          <span
            className={cn(
              "absolute inset-0 rounded-full flex items-center justify-center transition",
              listening
                ? "bg-gradient-to-br from-raw/40 to-raw/20 border-2 border-raw shadow-[0_0_50px_rgba(255,95,177,0.45)]"
                : "bg-gradient-to-br from-keep/30 to-cyan-accent/20 border-2 border-keep/60 shadow-[0_0_40px_rgba(54,241,163,0.3)] hover:shadow-[0_0_55px_rgba(54,241,163,0.5)]"
            )}
          >
            {listening ? <MicOff className="w-9 h-9 text-raw" /> : <Mic className="w-9 h-9 text-keep" />}
          </span>
        </button>
        <div className="text-center">
          <div className="text-[13px] font-medium text-ink">
            {listening ? "Listening… speak naturally" : "Tap to start"}
          </div>
          <div className="text-[11px] text-ink-faint mt-0.5">
            {listening ? "Each pause produces a compressed utterance." : "Each pause is an utterance — compressed live."}
          </div>
        </div>
        {partial && (
          <div className="text-[12px] text-ink-dim italic max-w-md text-center">"{partial}"</div>
        )}
        {voiceErr && (
          <div className="text-[11px] text-raw flex items-center gap-1.5">
            <AlertTriangle className="w-3 h-3" /> {voiceErr}
          </div>
        )}
      </div>

      {/* Cumulative stats + download */}
      {readyCount > 0 && (
        <div className="px-5 py-3 border-b border-white/5 flex items-center gap-5 flex-wrap text-[11px] font-mono uppercase tracking-wider">
          <div>
            <div className="text-ink-faint">utterances</div>
            <div className="text-ink text-[14px] font-bold tabular-nums mt-0.5">{utts.length}</div>
          </div>
          <div>
            <div className="text-ink-faint">tokens</div>
            <div className="text-ink text-[14px] font-bold tabular-nums mt-0.5">
              {totalOrigin.toLocaleString()}
              <span className="text-ink-faint mx-1">→</span>
              <span className="neon-text-keep">{totalCompressed.toLocaleString()}</span>
            </div>
          </div>
          <div>
            <div className="text-ink-faint">saved</div>
            <div className="neon-text-keep text-[14px] font-bold tabular-nums mt-0.5">{pct.toFixed(0)}%</div>
          </div>
          <button
            onClick={download}
            className="ml-auto text-[11px] font-mono uppercase tracking-wider px-3 py-1.5 rounded-full bg-keep/15 border border-keep/40 text-keep hover:bg-keep/25 transition flex items-center gap-1.5"
          >
            <Download className="w-3 h-3" /> download .md
          </button>
        </div>
      )}

      {/* Per-utterance compressed output. */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto scroll-soft px-5 py-4 space-y-3.5">
        {utts.length === 0 && (
          <div className="text-[12px] text-ink-faint italic text-center pt-4">
            Compressed utterances will appear here.
          </div>
        )}
        {utts.map((u) => (
          <div key={u.id} className="text-[14px] leading-relaxed">
            {u.state === "compressing" && (
              <div className="flex items-center gap-2 text-ink-faint italic">
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> compressing…
              </div>
            )}
            {u.state === "error" && (
              <div className="text-raw text-[13px]">⚠ {u.error}</div>
            )}
            {u.state === "ready" && (
              <div>
                <StrikeText labels={u.wordLabels} fallback={u.compressedText} />
                <div className="mt-1 text-[10px] font-mono uppercase tracking-wider text-ink-faint">
                  <span className="neon-text-keep">{u.ratio}</span>
                  <span className="mx-1.5">·</span>
                  <span>{u.originTokens}→{u.compressedTokens} tok</span>
                  {u.latencyMs !== undefined && (
                    <>
                      <span className="mx-1.5">·</span>
                      <span>{u.latencyMs}ms</span>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
