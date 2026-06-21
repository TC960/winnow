"use client";

import { useEffect, useRef, useState } from "react";
import { Mic, MicOff, Loader2, Trash2, AlertTriangle, Sparkles } from "lucide-react";
import { LiveMicSource } from "@/lib/sources/live-mic";
import { useStore } from "@/lib/store";
import { StrikeText } from "./StrikeText";
import { cn } from "@/lib/cn";

// Speak-and-ask flow. Each utterance is compressed via plain LLMLingua-2
// (NO AttentionRAG — dictation is too short to chunk meaningfully) and the
// compressed prompt is sent to the downstream blackbox LLM. The LLM answer
// is the main view; per-turn compression details sit "on the side" so the
// audience can see what the LLM actually received vs. what was spoken.

type Turn = {
  id: string;
  rawText: string;
  // compression
  compressState: "compressing" | "ready" | "error";
  compressedText?: string;
  originTokens?: number;
  compressedTokens?: number;
  ratio?: string;
  wordLabels?: [string, number][];
  compressMs?: number;
  compressError?: string;
  // llm
  llmState?: "waiting" | "streaming" | "done" | "error";
  llmText?: string;
  llmMs?: number;
  llmError?: string;
};

export function SpeakCard() {
  const rate = useStore((s) => s.rate);
  const language = useStore((s) => s.language);

  const [listening, setListening] = useState(false);
  const [partial, setPartial] = useState("");
  const [audioLevel, setAudioLevel] = useState(0);
  const [voiceErr, setVoiceErr] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);

  const srcRef = useRef<LiveMicSource | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => () => { stop(); /* eslint-disable-line react-hooks/exhaustive-deps */ }, []);
  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [turns]);

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
          setTurns((arr) => [...arr, { id, rawText, compressState: "compressing", llmState: "waiting" }]);
          void compressAndAsk(id, rawText);
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

  function patchTurn(id: string, patch: Partial<Turn>) {
    setTurns((arr) => arr.map((t) => (t.id === id ? { ...t, ...patch } : t)));
  }

  async function compressAndAsk(id: string, rawText: string) {
    // --- 1. Compression: plain LLMLingua-2 (NO question -> NO AttentionRAG). ---
    const tc0 = performance.now();
    let compressedPrompt = "";
    try {
      const r = await fetch("/api/compress", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: rawText, rate, return_labels: true }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || j.detail || `compress ${r.status}`);
      compressedPrompt = j.compressed_prompt;
      patchTurn(id, {
        compressState: "ready",
        compressedText: j.compressed_prompt,
        originTokens: j.origin_tokens,
        compressedTokens: j.compressed_tokens,
        ratio: j.ratio,
        wordLabels: j.word_labels,
        compressMs: Math.round(performance.now() - tc0),
        llmState: "streaming",
      });
    } catch (e: any) {
      patchTurn(id, { compressState: "error", compressError: e.message, llmState: "error", llmError: "skipped (compression failed)" });
      return;
    }

    // --- 2. Send compressed prompt to the blackbox LLM via downstream.py. ---
    const tl0 = performance.now();
    try {
      const r = await fetch("/api/downstream", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          provider: "claude",
          prompt: compressedPrompt,
          max_tokens: 512,
        }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || j.detail || `downstream ${r.status}`);
      patchTurn(id, {
        llmState: "done",
        llmText: j.text ?? "",
        llmMs: Math.round(performance.now() - tl0),
      });
    } catch (e: any) {
      patchTurn(id, { llmState: "error", llmError: e.message });
    }
  }

  function clearAll() {
    setTurns([]);
  }

  return (
    <section className="glass rounded-2xl flex flex-col h-full overflow-hidden">
      <header className="flex items-center justify-between px-5 py-3 border-b border-white/5">
        <div className="flex items-center gap-2.5">
          <Mic className="w-4 h-4 text-raw" />
          <h2 className="text-sm font-semibold tracking-wide">SPEAK → ASK</h2>
        </div>
        {turns.length > 0 && (
          <button
            onClick={clearAll}
            className="text-[11px] font-mono uppercase tracking-wider text-ink-faint hover:text-ink flex items-center gap-1"
          >
            <Trash2 className="w-3 h-3" /> clear
          </button>
        )}
      </header>

      {/* Mic */}
      <div className="px-5 py-4 border-b border-white/5 flex items-center gap-4">
        <button onClick={() => (listening ? stop() : start())} className="relative w-16 h-16 shrink-0">
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
                ? "bg-gradient-to-br from-raw/40 to-raw/20 border-2 border-raw shadow-[0_0_30px_rgba(255,95,177,0.4)]"
                : "bg-gradient-to-br from-keep/30 to-cyan-accent/20 border-2 border-keep/60 shadow-[0_0_24px_rgba(54,241,163,0.3)] hover:shadow-[0_0_36px_rgba(54,241,163,0.5)]"
            )}
          >
            {listening ? <MicOff className="w-6 h-6 text-raw" /> : <Mic className="w-6 h-6 text-keep" />}
          </span>
        </button>
        <div className="flex-1 min-w-0">
          <div className="text-[13px] font-medium text-ink">
            {listening ? "Listening… speak naturally" : "Tap to dictate a question"}
          </div>
          <div className="text-[11px] text-ink-faint mt-0.5">
            LLMLingua-2 only · sent to Claude downstream
          </div>
          {partial && (
            <div className="text-[12px] text-ink-dim italic mt-1.5 truncate">"{partial}"</div>
          )}
          {voiceErr && (
            <div className="text-[11px] text-raw mt-1.5 flex items-center gap-1.5">
              <AlertTriangle className="w-3 h-3" /> {voiceErr}
            </div>
          )}
        </div>
      </div>

      {/* Per-turn: LLM answer on the left, compression on the side. */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto scroll-soft px-5 py-4 space-y-5">
        {turns.length === 0 && (
          <div className="text-[12px] text-ink-faint italic text-center pt-8">
            Speak a question — its compressed form is what reaches the LLM.
          </div>
        )}

        {turns.map((t) => (
          <Turn key={t.id} turn={t} />
        ))}
      </div>
    </section>
  );
}

function Turn({ turn: t }: { turn: Turn }) {
  const savedPct = t.originTokens && t.compressedTokens
    ? (1 - t.compressedTokens / Math.max(1, t.originTokens)) * 100
    : 0;
  return (
    <div className="space-y-2.5">
      {/* User's raw line */}
      <div className="text-[13px] text-ink-dim italic">
        <span className="text-ink-faint font-mono uppercase tracking-wider text-[10px] mr-2">you</span>
        "{t.rawText}"
      </div>

      <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
        {/* LLM answer — main column */}
        <div className="md:col-span-3 rounded-xl border border-white/10 bg-white/[0.025] p-3">
          <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-ink-faint mb-1.5">
            <Sparkles className="w-3 h-3 text-cyan-accent" /> claude
            {t.llmMs !== undefined && <span className="ml-1.5">· {t.llmMs}ms</span>}
          </div>
          {t.llmState === "waiting" && (
            <div className="flex items-center gap-2 text-ink-faint italic text-[13px]">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> waiting on compression…
            </div>
          )}
          {t.llmState === "streaming" && (
            <div className="flex items-center gap-2 text-ink-faint italic text-[13px]">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> generating…
            </div>
          )}
          {t.llmState === "error" && (
            <div className="text-[13px] text-raw">⚠ {t.llmError}</div>
          )}
          {t.llmState === "done" && (
            <div className="text-[14px] leading-relaxed text-ink whitespace-pre-wrap">{t.llmText}</div>
          )}
        </div>

        {/* Compression "on side" */}
        <aside className="md:col-span-2 rounded-xl border border-keep/25 bg-keep/[0.04] p-3">
          <div className="text-[10px] font-mono uppercase tracking-wider text-keep mb-1.5">
            compressed prompt
          </div>
          {t.compressState === "compressing" && (
            <div className="flex items-center gap-2 text-ink-faint italic text-[13px]">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> compressing…
            </div>
          )}
          {t.compressState === "error" && (
            <div className="text-[12px] text-raw">⚠ {t.compressError}</div>
          )}
          {t.compressState === "ready" && (
            <>
              <div className="text-[13px] leading-relaxed">
                <StrikeText labels={t.wordLabels} fallback={t.compressedText} />
              </div>
              <div className="mt-2 grid grid-cols-3 gap-1 text-[9px] font-mono uppercase tracking-wider">
                <div>
                  <div className="text-ink-faint">tokens</div>
                  <div className="text-ink text-[12px] tabular-nums font-bold mt-0.5">
                    {t.originTokens}→{t.compressedTokens}
                  </div>
                </div>
                <div>
                  <div className="text-ink-faint">saved</div>
                  <div className="neon-text-keep text-[12px] tabular-nums font-bold mt-0.5">
                    {savedPct.toFixed(0)}%
                  </div>
                </div>
                <div>
                  <div className="text-ink-faint">time</div>
                  <div className="text-ink text-[12px] tabular-nums font-bold mt-0.5">
                    {t.compressMs}ms
                  </div>
                </div>
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
