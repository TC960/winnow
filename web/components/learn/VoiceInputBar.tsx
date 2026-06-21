"use client";

import { useEffect, useRef, useState } from "react";
import { Mic, MicOff, Send, Keyboard } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { LiveMicSource } from "@/lib/sources/live-mic";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/cn";

// Voice-first input for the Learn-tab chat. Tap the mic, talk, each pause
// auto-sends an utterance into the conversation. Tap again to stop. A small
// keyboard toggle reveals a regular text input as a fallback.
//
// Uses the same LiveMicSource as the Compare tab, but its own Deepgram
// connection — they can run in parallel without stepping on each other.

type Props = {
  onSend: (text: string) => void;
  disabled?: boolean;
  empty: boolean;     // true when the chat has no messages yet (renders the BIG mic)
};

export function VoiceInputBar({ onSend, disabled, empty }: Props) {
  const language = useStore((s) => s.language);
  const [listening, setListening] = useState(false);
  const [partial, setPartial] = useState("");
  const [textMode, setTextMode] = useState(false);
  const [draft, setDraft] = useState("");
  const [audioLevel, setAudioLevel] = useState(0);
  const sourceRef = useRef<LiveMicSource | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => () => stop(), []);

  async function start() {
    if (listening || disabled) return;
    try {
      const src = new LiveMicSource({ language });
      sourceRef.current = src;
      src.subscribe((evt) => {
        if (evt.type === "partial") setPartial(evt.text);
        else if (evt.type === "utterance") {
          const text = evt.utterance.text.trim();
          setPartial("");
          if (text) onSend(text);
        } else if (evt.type === "error") {
          console.error("[voice]", evt.error);
          stop();
        }
      });
      await src.start();
      setListening(true);
      attachLevelMeter();
    } catch (e) {
      console.error("voice start failed", e);
      stop();
    }
  }

  function stop() {
    if (sourceRef.current) {
      void sourceRef.current.stop();
      sourceRef.current = null;
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
    // Tap the same mic stream the LiveMicSource opened so we can render a
    // pulsing visual. We open our OWN getUserMedia stream because the
    // LiveMicSource doesn't expose its internal MediaStream.
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      ctx.createMediaStreamSource(stream).connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        analyser.getByteFrequencyData(data);
        const avg = data.reduce((a, b) => a + b, 0) / data.length;
        setAudioLevel(Math.min(1, avg / 90));
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();
    }).catch(() => {});
  }

  function sendDraft() {
    if (!draft.trim()) return;
    onSend(draft.trim());
    setDraft("");
  }

  // BIG centered mic, used when the chat is empty.
  if (empty) {
    return (
      <div className="flex flex-col items-center justify-center gap-6 py-8">
        <BigMic
          listening={listening}
          level={audioLevel}
          onClick={() => (listening ? stop() : start())}
          disabled={disabled}
        />
        <div className="text-center">
          <div className="text-[15px] text-ink font-medium">
            {listening ? "Listening… speak naturally" : "Tap to start talking"}
          </div>
          <div className="text-[12px] text-ink-faint mt-1">
            {listening
              ? "Each pause sends a message. Tap again to stop."
              : "Or use the keyboard toggle below for typing."}
          </div>
        </div>
        <AnimatePresence>
          {partial && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="max-w-md text-center text-[14px] text-ink-dim italic px-4"
            >
              "{partial}"
            </motion.div>
          )}
        </AnimatePresence>
        <button
          onClick={() => setTextMode((m) => !m)}
          className="text-[10px] font-mono uppercase tracking-wider text-ink-faint hover:text-ink flex items-center gap-1.5"
        >
          <Keyboard className="w-3 h-3" /> {textMode ? "voice mode" : "type instead"}
        </button>
        {textMode && (
          <div className="w-full max-w-md flex items-center gap-2 px-4">
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendDraft()}
              placeholder="Ask anything…"
              className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-[14px] text-ink placeholder:text-ink-faint focus:outline-none focus:border-cyan-accent/50"
            />
            <button
              onClick={sendDraft}
              disabled={!draft.trim()}
              className="p-2.5 rounded-xl bg-cyan-accent/15 border border-cyan-accent/40 text-cyan-accent hover:bg-cyan-accent/25 transition disabled:opacity-40"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>
    );
  }

  // Compact bottom bar — but visually rich. The mic is large, audio-reactive,
  // and sits inside a pill-shaped container with a live waveform when active.
  return (
    <div className="border-t border-white/5 p-4">
      {listening && partial && (
        <div className="text-[13px] text-keep/90 italic px-3 pb-2 truncate">
          "{partial}"
          <span className="inline-block w-1 h-3 ml-1 bg-keep align-middle animate-pulse-glow" />
        </div>
      )}
      <div
        className={cn(
          "relative flex items-center gap-3 rounded-2xl p-2 transition-all duration-300",
          listening
            ? "bg-gradient-to-r from-raw/10 via-raw/5 to-transparent border border-raw/30 shadow-[0_0_30px_rgba(255,95,177,0.15)]"
            : "bg-white/3 border border-white/10"
        )}
      >
        {/* The mic — chunky, ringed, audio-reactive. */}
        <button
          onClick={() => (listening ? stop() : start())}
          disabled={disabled}
          className={cn(
            "relative shrink-0 w-12 h-12 rounded-xl flex items-center justify-center transition-all",
            listening
              ? "bg-raw text-white shadow-[0_0_24px_rgba(255,95,177,0.55)]"
              : "bg-gradient-to-br from-keep/80 to-cyan-accent/60 text-black shadow-[0_0_20px_rgba(54,241,163,0.4)] hover:shadow-[0_0_28px_rgba(54,241,163,0.6)]"
          )}
        >
          {listening && (
            <>
              <span
                className="absolute inset-0 rounded-xl bg-raw/40"
                style={{
                  transform: `scale(${1 + audioLevel * 0.7})`,
                  opacity: 0.3 + audioLevel * 0.5,
                  transition: "transform 80ms ease-out, opacity 80ms ease-out",
                }}
              />
              <span className="absolute inset-0 rounded-xl border-2 border-raw/50 animate-ping" style={{ animationDuration: "2s" }} />
            </>
          )}
          {listening ? <MicOff className="w-5 h-5 relative" /> : <Mic className="w-5 h-5 relative" />}
        </button>

        {/* Center: either waveform (listening + voice mode), text input (text mode), or hint */}
        {textMode ? (
          <>
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendDraft()}
              placeholder="Type instead…"
              autoFocus
              className="flex-1 bg-transparent text-[14px] text-ink placeholder:text-ink-faint focus:outline-none px-2"
            />
            <button
              onClick={sendDraft}
              disabled={!draft.trim()}
              className="shrink-0 p-2.5 rounded-xl bg-cyan-accent/15 border border-cyan-accent/40 text-cyan-accent hover:bg-cyan-accent/25 transition disabled:opacity-40"
            >
              <Send className="w-4 h-4" />
            </button>
          </>
        ) : listening ? (
          <div className="flex-1 flex items-center gap-3 px-1 min-w-0">
            <Waveform level={audioLevel} />
            <span className="text-[11px] font-mono uppercase tracking-wider text-raw shrink-0">
              listening · pause to send
            </span>
          </div>
        ) : (
          <div className="flex-1 text-[14px] text-ink-dim italic px-2 truncate">
            Tap the mic and start talking
          </div>
        )}

        <button
          onClick={() => setTextMode((m) => !m)}
          title={textMode ? "Switch to voice" : "Switch to typing"}
          className="shrink-0 p-2.5 rounded-xl bg-white/5 border border-white/10 text-ink-dim hover:text-ink hover:bg-white/10 transition"
        >
          <Keyboard className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}

// Compact live waveform driven by the audio level. Twelve bars, each with a
// pseudo-random envelope so the meter feels organic instead of metronomic.
function Waveform({ level }: { level: number }) {
  const bars = 14;
  return (
    <div className="flex-1 flex items-center gap-[3px] h-6 min-w-0 overflow-hidden">
      {Array.from({ length: bars }).map((_, i) => {
        const phase = (i / bars) * Math.PI * 2;
        const envelope = 0.4 + 0.6 * Math.abs(Math.sin(phase + Date.now() / 220));
        const h = Math.max(3, level * envelope * 24);
        return (
          <span
            key={i}
            className="w-[3px] rounded-full bg-gradient-to-t from-raw/60 to-keep/80"
            style={{
              height: `${h}px`,
              opacity: 0.5 + level * 0.5,
              transition: "height 80ms ease-out, opacity 80ms ease-out",
            }}
          />
        );
      })}
    </div>
  );
}

function BigMic({
  listening, level, onClick, disabled,
}: { listening: boolean; level: number; onClick: () => void; disabled?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="relative w-32 h-32 disabled:opacity-50"
    >
      {/* Ripples when listening — scale with audio level. */}
      {listening && (
        <>
          <span
            className="absolute inset-0 rounded-full bg-raw/20 animate-ping"
            style={{ animationDuration: "1.8s" }}
          />
          <span
            className="absolute inset-0 rounded-full bg-raw/15"
            style={{
              transform: `scale(${1 + level * 0.5})`,
              transition: "transform 80ms ease-out",
            }}
          />
        </>
      )}
      <span
        className={cn(
          "absolute inset-0 rounded-full flex items-center justify-center transition",
          listening
            ? "bg-gradient-to-br from-raw/40 to-raw/20 border-2 border-raw shadow-[0_0_60px_rgba(255,95,177,0.5)]"
            : "bg-gradient-to-br from-keep/30 to-cyan-accent/20 border-2 border-keep/60 shadow-[0_0_50px_rgba(54,241,163,0.35)] hover:shadow-[0_0_70px_rgba(54,241,163,0.6)]"
        )}
      >
        {listening ? <MicOff className="w-12 h-12 text-raw" /> : <Mic className="w-12 h-12 text-keep" />}
      </span>
    </button>
  );
}
