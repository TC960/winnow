# winnow

> **voice → deepgram → llmlingua-2 → llm**

Real-time voice transcription with AI-powered token compression. Winnow listens to speech via Deepgram, compresses each utterance on a GPU with LLMLingua-2, and feeds the result to Claude — cutting context size while keeping the meaning intact.

---

## What it does

**Compare tab** — side-by-side live view of the raw transcript vs the compressed version. A stats bar tracks token counts and savings in real time. A Q&A box fires the same question at Claude using both versions in parallel, so you can see that the compressed text produces the same answer (proof of fidelity).

**Learn tab** — a three-column workspace built on top of the compressed transcript:
- **Sources** — the live compressed feed plus any extra text you paste in
- **Chat** — streaming Claude conversation grounded in those sources
- **Insights** — one-click study tools: summary, key decisions, action items, flashcards, glossary

**Trace mode** (in the Learn-tab chat): turn-level compression on top of the per-utterance path. Where the Compare tab compresses one utterance at a time, Trace mode compresses the whole growing conversation at the turn level. Each turn is assigned one of a gradient of actions:

- **KEEP**: verbatim content.
- **SUMMARIZE**: distilled via LLMLingua-2 at a low keep-rate.
- **TOMBSTONE**: a roughly 8-token in-context pointer (for example `[#7 tool_result elided ref=be86bf]`). This is the floor: budget pressure never deletes a turn to zero bytes.
- **ERASE**: true zero bytes, reserved for secrets/PII (`must_purge`), never reached by budget logic.

The pack runs a score-ordered, two-threshold greedy walk to a token budget, scoring each turn by cosine similarity to the current goal times a structural prior. The novel part is the system: turn granularity, the keep/summarize/tombstone gradient, and a tombstone whose recall signal lives in a cached embedding out of context (zero prompt tokens). A later question that is semantically similar to an evicted turn can recall and rehydrate it for free. A pass fires only once the chat crosses about 50% of the (demo-scaled) model window.

The chat shows a per-message KEEP/SUMMARIZE/TOMBSTONE badge, a budget slider, a pass stats line, and a one-click **verify fidelity** button that asks Claude the last question on the full vs compact history and grades the two answers semantically.

---

## Architecture

```
Microphone / fixture file
        │
        ▼
   Deepgram STT          (browser WebSocket)
        │
        ▼
  Next.js frontend       (React 18 + Zustand)
        │
        ▼
  /api/compress          (Next.js route — same-origin proxy)
        │
        ▼
  server.py (FastAPI)    (localhost:8000)
        │
        ▼
  llmlingua2_modal.py    (LLMLingua-2 on Modal T4 GPU)
```

The FastAPI server is a thin proxy — it looks up the deployed Modal class and forwards calls to a warm GPU container. The Next.js proxy keeps the Modal endpoint and Anthropic API key off the browser.

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | Next.js 15, React 18, TypeScript, Tailwind CSS, Framer Motion |
| State | Zustand |
| Speech-to-text | Deepgram (browser SDK) |
| Compression | LLMLingua-2 (`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`) |
| GPU inference | Modal (T4, with memory snapshots for fast cold starts) |
| Backend proxy | FastAPI + Uvicorn |
| LLM | Anthropic Claude (configurable model) |

---

## Setup

### Prerequisites

- Node.js 18+
- Python 3.11+
- [Modal account](https://modal.com) + CLI (`pip install modal && modal setup`)
- Deepgram API key
- Anthropic API key

### 1. Deploy the GPU worker

```bash
pip install modal llmlingua fastapi "uvicorn[standard]" pydantic
modal deploy llmlingua2_modal.py
```

The first run downloads the model into a persistent Modal volume and creates a GPU memory snapshot. Subsequent cold starts restore from the snapshot.

### 2. Start the backend

```bash
./run.sh
```

This deploys the worker (idempotent), warms it up, and starts the FastAPI server on port 8000. Flags:

```bash
SKIP_DEPLOY=1 ./run.sh   # worker already deployed
SKIP_WARMUP=1 ./run.sh   # worker already warm
PORT=9000 ./run.sh       # different port
```

### 3. Configure the frontend

```bash
cd web
cp .env.example .env.local
```

Fill in `.env.local`:

```
DEEPGRAM_API_KEY=...
ANTHROPIC_API_KEY=...
COMPRESS_BACKEND_URL=http://localhost:8000
```

### 4. Start the frontend

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `DEEPGRAM_API_KEY` | Yes | Used server-side to mint short-lived browser tokens |
| `ANTHROPIC_API_KEY` | Yes | Used by `/api/qa`, `/api/verify`, `/api/project-chat`, `/api/project-action`, and `trace/eval.py` |
| `COMPRESS_BACKEND_URL` | No | Defaults to `http://localhost:8000` |

---

## Trace mode internals

```
Learn-tab chat (the trace)
        │  every new message
        ▼
  POST /api/trace/ingest ──► FastAPI /trace/ingest   (embed locally with MiniLM, cache by content hash)
        │
        │  before each Claude call, once history crosses the trigger
        ▼
  POST /api/trace/pack ────► FastAPI /trace/pack      (score, pack to budget, SUMMARIZE via Modal LLMLingua-2)
        │                                              returns compact messages + per-turn actions + stats
        ▼
  /api/project-chat sends the COMPACT history to Claude
```

Cacheability rules that keep the claims honest:

- The **embedding** is cached by content hash (content is immutable).
- The **relevance score** is never cached: it depends on the current goal and is recomputed every pass.
- A generic LLMLingua-2 **summary** is cached by (content hash, keep-rate).

The `trace/` Python module holds the packer (`core.py`), the local MiniLM embedder (`embed.py`), the Modal SUMMARIZE tier (`summarize.py`), and one `Compressor` interface with four strategies (`strategies.py`): `NoOp`, `Truncate`, `NaiveSummarize`, and `OurPolicy`. The chat loop only ever calls the interface, so the stub-vs-erase ablation stays runnable by swapping a strategy or flipping one flag.

Smoke test (offline, no Modal or Claude):

```bash
.venv/bin/python -m trace.smoke
```

## The stub-vs-erase ablation

The headline measurement: when budget pressure forces an old turn out of context, does keeping a tombstone (a pointer plus a cached embedding, so recall can resurface it) beat truly erasing it? `trace/eval.py` replays one long scripted conversation, packs it under each mode for 3 seeds, then asks Claude a held-out set of recall-dependent questions (about content that got compressed away) using only the compacted history. In stub mode recall and rehydrate are allowed before answering; in erase mode they are not. Claude grades each answer against the expected answer.

```bash
pip install anthropic matplotlib
# ANTHROPIC_API_KEY in the environment or web/.env.local
.venv/bin/python -m trace.eval
```

Result on the scripted set (5 recall-dependent questions, 3 seeds):

| mode | recall-dependent accuracy | compact context tokens |
|---|---|---|
| **stub** (tombstone + cached embedding, recall allowed) | 100% (15/15) | 235 |
| **erase** (zero bytes, no cache, no recall) | 0% (0/15) | 165 |

Stub recovers every buried fact via recall; erase has nothing to recall and answers "Not in context" every time. The trade is real and shown straight: erase keeps a smaller standing context (165 vs 235 tokens), but loses the facts. Outputs land in `trace/`: `eval_results.csv` (per question), `eval_summary.csv`, and `eval_results.png` (the bar chart).

## Key controls

- **Rate slider** — fraction of tokens to keep (e.g. 0.5 = keep 50%)
- **Model picker** — choose the Claude model for Q&A and study tools
- **Language picker** — Deepgram transcription language
- **Source toggle** — switch between live microphone and a recorded fixture
- **Strip fillers** — remove filler words before compression
- **Session export** — download the full session as JSON
- `?` — keyboard shortcuts reference
