"use client";

import type { SourceConfig, SourceListener, TranscriptSource, Utterance, Word } from "./types";

// Live microphone → Deepgram streaming → speech_final Utterance events.
// The browser fetches a short-lived token from /api/deepgram-token and opens
// a WebSocket subprotocol-authenticated to Deepgram. Audio is sent as Opus-in-WebM
// chunks via MediaRecorder; transcripts come back as JSON over the same socket.

const DG_URL = (cfg: SourceConfig) => {
  const params = new URLSearchParams({
    model: "nova-3",
    smart_format: "true",
    interim_results: "true",
    punctuate: "true",
    endpointing: "300",
    utterance_end_ms: "1000",
    vad_events: "true",
    encoding: "opus",
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
    this.ws.onerror = () => this.emit({ type: "error", error: new Error("Deepgram socket error") });
    this.ws.onclose = () => this.emit({ type: "status", status: "closed" });
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
      if (!text) return;

      if (msg.speech_final || msg.is_final) {
        // Final or speech_final → emit utterance. We only treat speech_final
        // as "the user paused"; is_final is a partial commit and we coalesce
        // these into the next utterance event downstream. For simplicity we
        // emit on speech_final only.
        if (!msg.speech_final) {
          // partial final: surface it as a partial, do not fire the pipeline yet
          this.emit({ type: "partial", text });
          return;
        }
        const words: Word[] = (alt.words ?? []).map((w: any) => ({
          text: w.punctuated_word ?? w.word,
          confidence: w.confidence,
          speaker: w.speaker,
          startMs: Math.round((w.start ?? 0) * 1000),
        }));
        const start = words[0]?.startMs ?? 0;
        const end = words.length
          ? Math.round((alt.words[alt.words.length - 1].end ?? 0) * 1000)
          : start;
        const utterance: Utterance = {
          id: `u${++this.uttCounter}-${Date.now()}`,
          text,
          startMs: start,
          endMs: end,
          confidence: alt.confidence,
          speaker: words[0]?.speaker,
          words,
        };
        this.emit({ type: "utterance", utterance });
      } else {
        this.emit({ type: "partial", text });
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
        // Tell Deepgram we're done so it flushes any pending final.
        this.ws.send(JSON.stringify({ type: "CloseStream" }));
      }
      this.ws?.close();
    } catch {}
    this.recorder = null;
    this.stream = null;
    this.ws = null;
    this.emit({ type: "ended" });
  }
}
