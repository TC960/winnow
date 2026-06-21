"use client";

import type { SourceConfig, SourceListener, TranscriptSource, Utterance, Word } from "./types";

// Common WebSocket close codes Deepgram emits, mapped to actionable hints.
function dgCloseReason(code: number): string | null {
  switch (code) {
    case 1002: return "protocol error (audio format mismatch — likely sending containers when raw encoding was asserted)";
    case 1003: return "unsupported data (Deepgram couldn't decode the audio)";
    case 1008: return "policy violation (likely auth: API key invalid, expired, or missing scope)";
    case 1011: return "Deepgram internal error";
    case 4000: return "bad request (URL params rejected — check model/language/encoding)";
    case 4001: return "unauthorized (API key invalid or missing)";
    case 4008: return "payment required (Deepgram credits exhausted)";
    case 4029: return "rate limited";
    default: return null;
  }
}

// Live microphone → Deepgram streaming → speech_final Utterance events.
// The browser fetches a short-lived token from /api/deepgram-token and opens
// a WebSocket subprotocol-authenticated to Deepgram. Audio is sent as Opus-in-WebM
// chunks via MediaRecorder; transcripts come back as JSON over the same socket.

const DG_URL = (cfg: SourceConfig) => {
  // NB: do NOT set `encoding` here. MediaRecorder emits audio/webm;codecs=opus
  // *containers* (not raw Opus frames); Deepgram sniffs the codec from the
  // container MIME and closes the socket if `encoding=opus` is asserted but
  // raw Opus packets aren't actually sent.
  const params = new URLSearchParams({
    model: "nova-3",
    smart_format: "true",
    interim_results: "true",
    punctuate: "true",
    // 5-minute endpointing: we do NOT want real-time finalization. Deepgram
    // only emits speech_final after this much continuous silence, so a normal
    // Start→talk→Stop cycle never auto-finalizes mid-speech. The whole take is
    // flushed as a single utterance when stop() sends CloseStream.
    endpointing: "300000",
    // Deepgram caps utterance_end_ms at 5000; higher values (we used to send
    // 60000) reject the whole handshake with HTTP 400. The single-utterance
    // demo behavior is driven by endpointing above, and UtteranceEnd events
    // are ignored in handleMessage, so this value doesn't affect emission.
    utterance_end_ms: "5000",
    vad_events: "true",
  });
  if (cfg.language && cfg.language !== "multi") params.set("language", cfg.language);
  if (cfg.language === "multi") params.set("language", "multi");
  if (cfg.diarize) params.set("diarize", "true");
  return `wss://api.deepgram.com/v1/listen?${params.toString()}`;
};

export class LiveMicSource implements TranscriptSource {
  readonly kind = "live" as const;
  private listeners = new Set<SourceListener>();
  private ws: WebSocket | null = null;
  private recorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private uttCounter = 0;
  private startedAt = 0;
  // Single-take mode: we never emit mid-recording. Deepgram finalizes the audio
  // progressively (each is_final=true message is ONE segment, not the whole
  // transcript), so we ACCUMULATE every finalized segment here and emit the full
  // concatenation as one utterance when stop() flushes. Live partials are shown
  // as accumulated-so-far + the current interim hypothesis.
  private finalSegments: string[] = [];
  private finalWords: Word[] = [];
  private lastEndMs = 0;

  constructor(private cfg: SourceConfig = {}) {}

  subscribe(listener: SourceListener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
  private emit(e: Parameters<SourceListener>[0]) {
    for (const l of this.listeners) l(e);
  }

  async start() {
    this.emit({ type: "status", status: "connecting" });

    // 1. Get a short-lived token from our backend.
    const tokenRes = await fetch("/api/deepgram-token", { method: "POST" });
    if (!tokenRes.ok) throw new Error(`Token mint failed: ${tokenRes.status}`);
    const { access_token } = await tokenRes.json();

    // 2. Open mic.
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 48000 },
    });
    this.startedAt = performance.now();

    // 3. Open Deepgram WS with subprotocol auth.
    this.ws = new WebSocket(DG_URL(this.cfg), ["token", access_token]);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      this.emit({ type: "status", status: "open" });
      // 4. Start streaming Opus/WebM chunks from MediaRecorder.
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      this.recorder = new MediaRecorder(this.stream!, { mimeType: mime, audioBitsPerSecond: 32000 });
      this.recorder.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0 && this.ws?.readyState === WebSocket.OPEN) {
          ev.data.arrayBuffer().then((buf) => this.ws?.send(buf));
        }
      };
      this.recorder.start(250); // 250ms chunks — low enough latency, doesn't spam
    };

    this.ws.onmessage = (ev) => this.handleMessage(ev.data);
    // The browser WS API doesn't expose error details — the close event right
    // after onerror does (code + reason). Surface both so we can actually debug.
    let erroredAt = 0;
    this.ws.onerror = () => { erroredAt = Date.now(); };
    this.ws.onclose = (ev) => {
      const wasError = Date.now() - erroredAt < 250 || (ev.code !== 1000 && ev.code !== 1005);
      if (wasError) {
        const reason = ev.reason?.trim() || dgCloseReason(ev.code) || `code ${ev.code}`;
        this.emit({ type: "error", error: new Error(`Deepgram socket closed: ${reason}`) });
      }
      this.emit({ type: "status", status: "closed" });
    };
  }

  private handleMessage(data: any) {
    let msg: any;
    try {
      msg = typeof data === "string" ? JSON.parse(data) : null;
    } catch {
      return;
    }
    if (!msg) return;

    if (msg.type === "Results") {
      const alt = msg.channel?.alternatives?.[0];
      if (!alt) return;
      const text = (alt.transcript ?? "").trim();

      if (msg.is_final) {
        // A finalized segment — append it to the running take. We do NOT emit a
        // utterance here; the whole take is emitted once, on stop().
        if (text) {
          const words: Word[] = (alt.words ?? []).map((w: any) => ({
            text: w.punctuated_word ?? w.word,
            confidence: w.confidence,
            speaker: w.speaker,
            startMs: Math.round((w.start ?? 0) * 1000),
          }));
          this.finalSegments.push(text);
          this.finalWords.push(...words);
          if (alt.words?.length) {
            this.lastEndMs = Math.round((alt.words[alt.words.length - 1].end ?? 0) * 1000);
          }
        }
        // Show the full accumulated transcript so far.
        this.emit({ type: "partial", text: this.finalSegments.join(" ") });
      } else if (text) {
        // Interim hypothesis — show accumulated text + the live guess.
        const live = [this.finalSegments.join(" "), text].filter(Boolean).join(" ");
        this.emit({ type: "partial", text: live });
      }
    } else if (msg.type === "SpeechStarted") {
      // could surface a "you're talking" indicator; skipped for now
    } else if (msg.type === "UtteranceEnd") {
      // VAD-based end signal; speech_final already handled emission
    }
  }

  async stop() {
    try { this.recorder?.stop(); } catch {}
    try { this.stream?.getTracks().forEach((t) => t.stop()); } catch {}
    try {
      if (this.ws?.readyState === WebSocket.OPEN) {
        // Tell Deepgram we're done so it flushes any pending final segment,
        // then wait for that last is_final to land in finalSegments before we
        // build the utterance.
        this.ws.send(JSON.stringify({ type: "CloseStream" }));
        await new Promise((r) => setTimeout(r, 800));
      }
      this.ws?.close();
    } catch {}

    // Emit the entire take as a single utterance.
    const text = this.finalSegments.join(" ").trim();
    if (text) {
      const words = this.finalWords;
      const start = words[0]?.startMs ?? 0;
      const end = this.lastEndMs || start;
      const utterance: Utterance = {
        id: `u${++this.uttCounter}-${Date.now()}`,
        text,
        startMs: start,
        endMs: end,
        speaker: words[0]?.speaker,
        words,
      };
      this.emit({ type: "utterance", utterance });
    }

    this.recorder = null;
    this.stream = null;
    this.ws = null;
    this.finalSegments = [];
    this.finalWords = [];
    this.lastEndMs = 0;
    this.emit({ type: "ended" });
  }
}
