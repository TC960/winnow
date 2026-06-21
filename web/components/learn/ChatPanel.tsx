"use client";

import { useEffect, useRef } from "react";
import { Trash2, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import { useStore } from "@/lib/store";
import type { TraceAction } from "@/lib/store";
import { cn } from "@/lib/cn";
import { estimateTokens, TRACE_TRIGGER_TOKENS } from "@/lib/tokens";
import { VoiceInputBar } from "./VoiceInputBar";
import { TraceBar, ACTION_DOT } from "./TraceBar";

// Voice-first project chat. Speak naturally — each pause auto-sends as a
// chat message. Anthropic streams the answer back token-by-token. Sources
// (compressed transcript + extras) are sent with prompt-cache control so
// multi-turn conversations stay cheap.

const SUGGESTIONS = [
  "Summarize the main points",
  "What decisions were made?",
  "Quiz me on this",
  "What's unclear or contradictory?",
];

export function ChatPanel() {
  const messages = useStore((s) => s.chatMessages);
  const append = useStore((s) => s.appendChat);
  const update = useStore((s) => s.updateLastChat);
  const finish = useStore((s) => s.finishLastChat);
  const resetChat = useStore((s) => s.resetChat);
  const rows = useStore((s) => s.rows);
  const extras = useStore((s) => s.extraSources);
  const model = useStore((s) => s.model);
  const sessionId = useStore((s) => s.traceSessionId);
  const budget = useStore((s) => s.traceBudget);
  const setTracePack = useStore((s) => s.setTracePack);
  const traceActions = useStore((s) => s.traceActions);
  const tracePackedUpTo = useStore((s) => s.tracePackedUpTo);

  const sendingRef = useRef(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Ingest a new turn off the critical path: cache its embedding for scoring and
  // recall without blocking typing or the chat send. Fire-and-forget.
  function ingestTurn(index: number, type: string, content: string) {
    if (!content.trim()) return;
    fetch("/api/trace/ingest", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        turn: { index, type, content, tokens: estimateTokens(content) },
      }),
    }).catch(() => {});
  }

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  async function send(text: string) {
    if (!text.trim() || sendingRef.current) return;
    sendingRef.current = true;
    const q = text.trim();
    const compressedText = rows.filter((r) => r.compressed).map((r) => r.compressed!.text).join(" ");
    const sources = [
      { title: "Compressed transcript", content: compressedText || "(empty)" },
      ...extras.map((s) => ({ title: s.title, content: s.content })),
    ];
    // Full raw history including the new user turn. Turn index = position here.
    const rawHistory = [
      ...messages.map((m) => ({ role: m.role, content: m.content })),
      { role: "user" as const, content: q },
    ];
    const userIndex = rawHistory.length - 1;

    append({ role: "user", content: q });
    append({ role: "assistant", content: "", streaming: true });

    // Ingest the new user turn (do not await: must not block the send).
    ingestTurn(userIndex, "user", q);

    // Trigger: only run a pack pass once the live history exceeds ~50% of the
    // (demo-scaled) window. Below that, send the raw history unchanged. On any
    // pack failure we silently fall back to the raw history.
    let messagesToSend: { role: "user" | "assistant"; content: string }[] = rawHistory;
    const historyTokens = rawHistory.reduce((a, m) => a + estimateTokens(m.content), 0);
    if (historyTokens > TRACE_TRIGGER_TOKENS) {
      try {
        const turns = rawHistory.map((m, i) => ({
          index: i,
          type: m.role,
          content: m.content,
          tokens: estimateTokens(m.content),
        }));
        const pr = await fetch("/api/trace/pack", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, goal: q, turns, budget }),
        });
        if (pr.ok) {
          const pack = await pr.json();
          if (Array.isArray(pack.compact_messages) && pack.compact_messages.length) {
            messagesToSend = pack.compact_messages;
            setTracePack({
              actions: (pack.actions ?? {}) as Record<string, TraceAction>,
              packedUpTo: rawHistory.length,
              stats: pack.stats,
            });
          }
        }
      } catch {
        // fall back to raw history
      }
    }

    try {
      const res = await fetch("/api/project-chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: messagesToSend, sources, model }),
      });
      if (!res.body) throw new Error("no stream");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            const j = JSON.parse(payload);
            if (j.delta) update(j.delta);
            if (j.error) update(`\n⚠ ${friendlyError(j.error)}`);
          } catch {}
        }
      }
    } catch (e: any) {
      update(`\n⚠ ${friendlyError(e.message)}`);
    } finally {
      finish();
      sendingRef.current = false;
      // Ingest the completed assistant turn (index follows the user turn).
      const answer = useStore.getState().chatMessages.at(-1)?.content ?? "";
      ingestTurn(userIndex + 1, "assistant", answer);
    }
  }

  const empty = messages.length === 0;

  return (
    <section className="glass rounded-2xl flex flex-col h-full overflow-hidden">
      <header className="flex items-center justify-between px-5 py-3 border-b border-white/5">
        <div className="flex items-center gap-2.5">
          <Sparkles className="w-4 h-4 text-cyan-accent" />
          <h2 className="text-sm font-semibold tracking-wide">VOICE CHAT</h2>
        </div>
        {messages.length > 0 && (
          <button
            onClick={resetChat}
            className="text-[11px] font-mono uppercase tracking-wider text-ink-faint hover:text-raw transition flex items-center gap-1"
          >
            <Trash2 className="w-3 h-3" /> clear
          </button>
        )}
      </header>

      <TraceBar />

      <div ref={scrollRef} className="flex-1 overflow-y-auto scroll-soft px-5 py-4 space-y-4">
        {empty && (
          <div className="flex flex-col items-center justify-center h-full gap-6">
            <VoiceInputBar empty onSend={send} />
            <div className="flex flex-wrap gap-2 justify-center max-w-md">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="text-[12px] px-3 py-1.5 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-ink-dim hover:text-ink transition"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => {
          const isError = m.role === "assistant" && m.content.trim().startsWith("⚠");
          // A pass labels turns 0..tracePackedUpTo-1. Anything in that range
          // without an explicit action was in the verbatim keep-zone (KEEP).
          const action: TraceAction | undefined =
            i < tracePackedUpTo ? (traceActions[String(i)] ?? "keep") : undefined;
          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className={cn("flex flex-col gap-1 max-w-[88%]", m.role === "user" ? "ml-auto items-end" : "mr-auto items-start")}
            >
              <div
                className={cn(
                  "rounded-2xl px-4 py-2.5 text-[14px] leading-relaxed whitespace-pre-wrap",
                  m.role === "user"
                    ? "bg-cyan-accent/15 border border-cyan-accent/30 text-ink"
                    : isError
                      ? "bg-raw/10 border border-raw/30 text-raw"
                      : "bg-white/3 border border-white/10 text-ink"
                )}
              >
                {m.content}
                {m.streaming && (
                  <span className="inline-block w-1.5 h-3.5 ml-1 bg-keep align-middle animate-pulse-glow" />
                )}
              </div>
              <MessageBadge action={action} tokens={estimateTokens(m.content)} />
            </motion.div>
          );
        })}
      </div>

      {!empty && <VoiceInputBar empty={false} onSend={send} />}
    </section>
  );
}

// Surface API errors as one-line, human-readable messages instead of dumping
// raw JSON into the chat bubble. Pulls auth/quota/network signals out of the
// noisy upstream payloads.
function friendlyError(raw: string): string {
  if (!raw) return "Something went wrong.";
  const lower = raw.toLowerCase();
  if (lower.includes("invalid x-api-key") || lower.includes("invalid_api_key") || lower.includes("authentication_error"))
    return "Anthropic key is invalid. Check ANTHROPIC_API_KEY in web/.env and restart the dev server.";
  if (lower.includes("api_key not set") || lower.includes("anthropic_api_key not set"))
    return "ANTHROPIC_API_KEY isn't set. Add it to web/.env and restart npm run dev.";
  if (lower.includes("rate_limit") || lower.includes("rate limit") || lower.includes("429"))
    return "Anthropic rate limit hit. Wait a few seconds and try again.";
  if (lower.includes("overloaded") || lower.includes("503"))
    return "Anthropic is overloaded. Retrying often works.";
  if (lower.includes("fetch failed") || lower.includes("econnrefused") || lower.includes("network"))
    return "Couldn't reach the Anthropic API. Check your network.";
  // Try to surface a clean message field from a JSON error payload.
  try {
    const j = JSON.parse(raw.replace(/^[^{]*/, ""));
    const msg = j?.error?.message || j?.message;
    if (msg) return msg;
  } catch {}
  return raw.length > 160 ? raw.slice(0, 160) + "…" : raw;
}

// Per-message trace badge: the action assigned by the last pack pass plus the
// turn's token count. Tombstone is the floor (a pointer, never zero bytes); a
// turn outside the last pass shows just its token count.
function MessageBadge({ action, tokens }: { action?: TraceAction; tokens: number }) {
  return (
    <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-ink-faint px-1">
      {action && (
        <span className="inline-flex items-center gap-1">
          <span className={cn("inline-block w-1.5 h-1.5 rounded-full", ACTION_DOT[action])} />
          {action}
        </span>
      )}
      <span className="tabular-nums">{tokens} tok</span>
    </div>
  );
}
