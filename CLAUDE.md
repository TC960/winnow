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

### Trace mode (turn-level compression)
```bash
# Smoke test: gradient + recall + erase check, fully offline (no Modal/Claude)
.venv/bin/python -m trace.smoke

# Stage 4 ablation gate: stub vs erase on recall-dependent questions.
# Needs ANTHROPIC_API_KEY (env or web/.env.local) for the Claude answer + grade.
# Offline otherwise: the scripted turns carry precomputed summaries, so SUMMARIZE
# never calls Modal. Writes eval_results.csv, eval_summary.csv, eval_results.png.
.venv/bin/python -m trace.eval
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

- `server.py`: FastAPI proxy. Exposes `POST /compress` and `GET /health`, plus the Trace endpoints `POST /trace/ingest`, `POST /trace/pack`, `POST /trace/recall`. The Modal app/class names (`llmlingua2-xlm` / `Compressor`) must match `llmlingua2_modal.py`. Holds one `OurPolicy` per session id (the per-session content-hash Store).
- `llmlingua2_modal.py`: Modal GPU worker. Loads `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` into a persistent volume with a memory snapshot for fast cold starts.
- `warmup.py`: One-shot script that builds the GPU snapshot. Called by `run.sh` step 2.
- `trace_packer.py`: Reference implementation of the turn-level context compression algorithm (algorithm of record). See `docs/trace_spec.md` for the integration spec.
- `trace/core.py`: Ported packer. `Turn`, `Action`, `Record`, `Store`, `pack`, `render_history`, `recall`, `rehydrate`, folding. The keep/summarize/tombstone gradient and the tombstone-with-cached-embedding recall path.
- `trace/embed.py`: MiniLM (`all-MiniLM-L6-v2`) local CPU embedder, batched. Drops into `core.Store` as the Embedder.
- `trace/summarize.py`: SUMMARIZE tier. Calls the deployed Modal LLMLingua-2 worker at a low keep-rate, cached by (content hash, rate).
- `trace/strategies.py`: The one `Compressor` interface plus the four strategies the ablation compares: `NoOp`, `Truncate`, `NaiveSummarize`, `OurPolicy` (stub-vs-erase is a `stub_mode` flag).
- `trace/eval.py`: Stage 4 ablation gate. Scripted conversation + recall-dependent questions, stub vs erase x 3 seeds, Claude answers + grades, writes CSV and a bar chart.
- `trace/smoke.py`: Offline smoke test (gradient, recall hit, erase check). No Modal or Claude.
- `web/lib/store.ts`: Zustand store. Holds all app state: rows (utterance pipeline), Learn-tab chat, insights, UI controls, and the Trace pass state (actions, stats, compact/raw text for verify).
- `web/lib/pipeline.ts`: Utterance pipeline. Subscribes to the active source, appends rows, fires `POST /api/compress` per utterance.
- `web/lib/sources/`: Swappable transcript sources (`live-mic.ts` for Deepgram, `recorded.ts` for fixture playback).
- `web/app/api/`: Next.js route handlers that proxy to FastAPI (`compress`, `trace/*`) or call Anthropic directly (`qa`, `verify`, `project-chat`, `project-action`).
- `web/components/CompareView.tsx`: Compare tab (raw vs compressed side-by-side, stats, parallel Q&A).
- `web/components/LearnView.tsx`: Learn tab shell. Three columns: Sources, Chat, Insights.
- `web/components/learn/ChatPanel.tsx`: Claude chat grounded in the compressed transcript. Ingests each turn and runs a `/api/trace/pack` pass once history crosses the trigger.
- `web/components/learn/TraceBar.tsx`: Pack budget slider plus the per-pass stats line (tokens before/after, percent saved, keep/summarize/tombstone counts).
- `web/components/learn/VerifyPanel.tsx`: Fidelity check. Asks Claude the last question on full vs compact history via `/api/verify`, with a semantic grader verdict (same judge as `trace/eval.py`).

## Environment variables (web/.env.local)

```
DEEPGRAM_API_KEY=       # mints short-lived browser tokens at /api/deepgram-token
ANTHROPIC_API_KEY=      # used by /api/qa, /api/verify, /api/project-chat, /api/project-action, and trace/eval.py
COMPRESS_BACKEND_URL=http://localhost:8000   # optional, defaults to :8000
```

`trace/eval.py` reads `ANTHROPIC_API_KEY` from the environment, falling back to `web/.env.local`.

## Trace mode (shipped)

A turn-level context compression layer on top of the per-utterance LLMLingua-2 path. It decides, per turn, whether to KEEP it verbatim, SUMMARIZE it (LLMLingua-2 at a low keep-rate), or TOMBSTONE it to an in-context pointer, packing the growing Learn-tab chat to a token budget. ERASE (true zero bytes) is reserved for `must_purge` content and is never reached by budget logic. `docs/trace_spec.md` is the integration spec and `trace_packer.py` the algorithm of record.

What is implemented:

- `trace/` module: `core.py` (packer, Store, recall, folding), `embed.py` (local MiniLM), `summarize.py` (Modal LLMLingua-2 SUMMARIZE tier), `strategies.py` (the `Compressor` interface plus `NoOp` / `Truncate` / `NaiveSummarize` / `OurPolicy`).
- FastAPI endpoints `POST /trace/ingest`, `POST /trace/pack`, `POST /trace/recall`, mirrored by Next.js proxies in `web/app/api/trace/`.
- Learn tab: per-message KEEP/SUMMARIZE/TOMBSTONE badges (`ChatPanel.tsx`), the budget slider and pass stats (`TraceBar.tsx`), and the fidelity verify view (`VerifyPanel.tsx`).

The headline result is the stub-vs-erase ablation in `trace/eval.py`: on recall-dependent questions (about content the pack evicts), the tombstone path (recall + rehydrate allowed) answers correctly while a true-erase baseline cannot, because the erased turn leaves no cache entry to recall. Run `.venv/bin/python -m trace.eval` to regenerate the CSV and bar chart.

### Trace cacheability (get this right or the claims are false)
- The **embedding** is cached by content hash (content is immutable).
- The **relevance score** is never cached: it depends on the current goal, recomputed every pass.
- A generic LLMLingua-2 **summary** is cached by (content hash, keep-rate). Do not cache query-conditioned summaries.

### Trace trigger
A pack pass fires only once live history crosses a trigger (`TRACE_TRIGGER_TOKENS` in `web/lib/tokens.ts`, scaled down for the demo so a hand-typed session reaches it). Below the trigger the raw history is sent unchanged.

### Fidelity verify vs eval grading
- `web/app/api/verify/route.ts` (the Learn-tab verify button) asks the question on full vs compact history, then grades the two answers **semantically** with a Claude judge (ALIGNED / DIVERGED). It does not use string equality.
- `trace/eval.py` grades each answer against an expected answer with the same kind of strict Claude judge. Both use temperature 0.

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