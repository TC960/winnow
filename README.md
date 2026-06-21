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
| `ANTHROPIC_API_KEY` | Yes | Used by `/api/qa` and `/api/project-action` |
| `COMPRESS_BACKEND_URL` | No | Defaults to `http://localhost:8000` |

---

## Key controls

- **Rate slider** — fraction of tokens to keep (e.g. 0.5 = keep 50%)
- **Model picker** — choose the Claude model for Q&A and study tools
- **Language picker** — Deepgram transcription language
- **Source toggle** — switch between live microphone and a recorded fixture
- **Strip fillers** — remove filler words before compression
- **Session export** — download the full session as JSON
- `?` — keyboard shortcuts reference
