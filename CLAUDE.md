# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend
```bash
# Full startup: deploy Modal GPU worker, warm it, start FastAPI on :8000
./run.sh

# Skip steps already done
SKIP_DEPLOY=1 ./run.sh    # worker already deployed
SKIP_WARMUP=1 ./run.sh    # worker already warm

# Backend only (worker must already be deployed)
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000

# Deploy GPU worker to Modal
.venv/bin/modal deploy llmlingua2_modal.py
```

### Frontend
```bash
cd web
npm run dev     # dev server on :3000
npm run build
npm run lint
```

### Python setup
```bash
pip install modal llmlingua fastapi "uvicorn[standard]" pydantic
modal setup     # authenticate Modal CLI
```

## Architecture

```
Microphone / fixture file
        │
        ▼
   Deepgram STT (browser WebSocket)
        │
        ▼
  Next.js frontend (React 18 + Zustand)   web/
        │
        ▼
  /api/compress (Next.js same-origin proxy)
        │
        ▼
  server.py (FastAPI on :8000)
        │
        ▼
  llmlingua2_modal.py (LLMLingua-2 on Modal T4 GPU)
```

The FastAPI server is a thin proxy: it holds a singleton `modal.Cls` handle and forwards compress calls to a warm GPU container. The Next.js proxy layer (`web/app/api/`) keeps the Modal endpoint and Anthropic/Deepgram API keys off the browser.

## Key files

- `server.py`: FastAPI proxy. Exposes `POST /compress` and `GET /health`. The Modal app/class names (`llmlingua2-xlm` / `Compressor`) must match `llmlingua2_modal.py`.
- `llmlingua2_modal.py`: Modal GPU worker. Loads `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` into a persistent volume with a memory snapshot for fast cold starts.
- `warmup.py`: One-shot script that builds the GPU snapshot. Called by `run.sh` step 2.
- `trace_packer.py`: Reference implementation of the turn-level context compression algorithm. See `docs/trace_spec.md` for the full integration spec. This is the algorithm of record for the in-progress Trace mode feature.
- `web/lib/store.ts`: Zustand store. Holds all app state: rows (utterance pipeline), Learn-tab chat, insights, UI controls.
- `web/lib/pipeline.ts`: Utterance pipeline. Subscribes to the active source, appends rows, fires `POST /api/compress` per utterance.
- `web/lib/sources/`: Swappable transcript sources (`live-mic.ts` for Deepgram, `recorded.ts` for fixture playback).
- `web/app/api/`: Next.js route handlers that proxy to FastAPI or call Anthropic directly.
- `web/components/CompareView.tsx`: Compare tab (raw vs compressed side-by-side, stats, parallel Q&A).
- `web/components/LearnView.tsx`: Learn tab shell. Three columns: Sources, Chat, Insights.
- `web/components/learn/ChatPanel.tsx`: Claude chat grounded in the compressed transcript.

## Environment variables (web/.env.local)

```
DEEPGRAM_API_KEY=       # mints short-lived browser tokens at /api/deepgram-token
ANTHROPIC_API_KEY=      # used by /api/qa and /api/project-action
COMPRESS_BACKEND_URL=http://localhost:8000   # optional, defaults to :8000
```

## Active development: Trace mode

`docs/trace_spec.md` is an integration spec for a turn-level context compression layer ("Trace mode") to be added on top of the existing utterance-level compression. `trace_packer.py` is the reference implementation of the core algorithm. The spec calls for:

- A `trace/` Python module porting `trace_packer.py` into the server, with `embed.py` (MiniLM, local CPU) and `summarize.py` (calls the Modal LLMLingua-2 worker).
- Three new FastAPI endpoints: `POST /trace/ingest`, `POST /trace/pack`, `POST /trace/recall`, each mirrored by a Next.js proxy in `web/app/api/trace/`.
- Frontend additions in the Learn tab: a Trace panel with per-message KEEP/SUMMARIZE/TOMBSTONE badges, a recall chip, and a fidelity verify view.

See `docs/trace_spec.md` for the full algorithm contract, API shape, build order, and acceptance criteria.

## Style conventions

- Do not use em dashes anywhere: in code comments, docstrings, UI copy, or documentation. Use commas, colons, or parentheses instead.
- New Next.js proxy routes follow the pattern in `web/app/api/compress/route.ts`: forward to `COMPRESS_BACKEND_URL`, surface errors as JSON.
- Compression results are cached by content hash; scores are never cached (they depend on the current goal).

## Tracer Mode rules 

GOAL: add a turn-level context compressor on top of winnow's existing LLMLingua-2 path, measured by compression ratio, answer fidelity, and a stub-vs-erase ablation on recall-dependent questions.

RULES:
1. All compression strategies implement ONE interface. The chat loop calls the interface and knows nothing else. Never inline compression into the loop, or the erase-vs-stub ablation becomes un-runnable.
2. Deterministic and cached: same input plus config gives same output. Cache per-turn embeddings and summaries by content hash. Identical Claude sampling params across variants; vary only a seed.
3. Do not invent library APIs. For sentence-transformers, the existing Modal LLMLingua-2 entrypoint, and the Anthropic SDK, read the actual signatures in this repo or the README. If unsure, stop and say so.
4. Reuse winnow's deployed LLMLingua-2 Modal worker for the SUMMARIZE tier. Do not stand up a new model.
5. In-memory Store keyed by content hash for now. LanceDB is a later swap, out of scope today.
6. Embeddings: sentence-transformers all-MiniLM-L6-v2, local, CPU, batched. Pin it in requirements.

INTERFACE:
  @dataclass
  class Turn: index:int; kind:str; content:str; tokens:int; meta:dict
  class Compressor(Protocol):
    def observe(self, turn: Turn) -> None: ...        # ingest, cache embedding off critical path; no-op for baselines
    def compress(self, turns, goal, budget) -> list[Turn]: ...
    def recall(self, query: str) -> list[Hit]: ...    # have I been here before; returns [] for baselines

STRATEGIES: NoOp, Truncate, NaiveSummarize, OurPolicy (port from trace_packer.py: keep/summarize/stub, stub carries content-hash identity plus pointer to cached blob, erase-vs-stub is a config flag).