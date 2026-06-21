# Coding agent task: add turn-level context compression ("Trace mode") to winnow

You are working inside the existing **winnow** repo (voice → Deepgram → LLMLingua-2 → Claude). Your job is to add a **turn-level context-compression layer** on top of the existing utterance-level compression, and wire it into the Learn-tab chat. Read the repo before writing anything.

A reference implementation of the core algorithm already exists at `trace_packer.py` in the repo root (the packer, the tombstone recall path, folding, the Store). Port and adapt it. Do not reinvent it. This prompt is the integration spec; `trace_packer.py` is the algorithm of record.

Style: do not use em dashes anywhere (code comments, docstrings, READMEs, UI copy). Use commas, colons, or parentheses.

---

## 0. The one-line framing

Winnow today compresses *one utterance at a time* with LLMLingua-2. Trace mode compresses *the whole growing conversation* at the turn level: it decides, per turn, whether to keep it verbatim, compress it (LLMLingua-2), or tombstone it to a pointer, packing the history to a token budget, with out-of-context cached embeddings that let the agent answer "have I been here before" for free. The Learn-tab chat is the trace.

Keep this honest in any copy you write: the novel part is the system (turn granularity, structural types, the tombstone+cached-embedding recall, off-critical-path precompute, and the fidelity measurement), not a new compression algorithm. LLMLingua-2 does the token-level work.

---

## 1. What already exists (confirm against the repo)

- `llmlingua2_modal.py`: LLMLingua-2 (`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`) on a Modal T4, with memory snapshots. Exposes a compress call that takes text and a keep-rate.
- `server.py`: thin FastAPI proxy on `:8000`. Looks up the deployed Modal class and forwards compress calls.
- `web/`: Next.js 15 + React 18 + TypeScript + Tailwind + Framer Motion, Zustand store.
- Next.js API routes (same-origin proxies): `/api/compress`, `/api/qa`, `/api/project-action`, plus a Deepgram token route. These keep the Modal endpoint and the Anthropic key off the browser.
- Two tabs: **Compare** (raw vs compressed, stats bar, parallel Q&A fidelity check) and **Learn** (Sources / Chat / Insights over the compressed transcript).

Locate the exact filenames for the Modal compress entrypoint, the FastAPI compress handler, the Zustand store, and the Learn-tab chat component before editing. Match existing patterns (naming, error handling, the proxy boundary).

---

## 2. Core algorithm contract (port from `trace_packer.py`)

Implement these exactly. The reference file already does; mirror its semantics.

**Action space (a gradient, never a hole):**
- `KEEP`: verbatim content.
- `SUMMARIZE`: compressed via LLMLingua-2 at a low keep-rate.
- `TOMBSTONE`: a roughly 8-token in-context stub, for example `[#7 tool_result elided ref=be86bf]`. This is the floor. `drop` is never zero bytes.
- `ERASE`: true zero bytes, reachable only via a `must_purge` flag (secrets / PII). Never reached by budget logic.

**Packer (stage 4), score-ordered two-threshold walk:**
- Two thresholds turn each turn's relevance score into a desired tier: `score >= keep_threshold` wants KEEP, `score >= summary_threshold` wants SUMMARIZE, else TOMBSTONE.
- Floor every candidate to a tombstone, then a greedy spends the budget: fund KEEPs (highest score first), then SUMMARIZEs.
- A want-KEEP that cannot afford verbatim degrades to SUMMARIZE. A want-SUMMARIZE that cannot afford a compressed line stays a TOMBSTONE. Deterministic given the scores.

**Tombstone recall (the bespoke value):**
- The in-context stub is a pointer. The recall signal lives in the **cached embedding**, out of context, zero prompt tokens.
- `recall(query)` cosines the query against cached tombstone embeddings and returns hits. `rehydrate(hit)` returns the cached summary, or the content if no summary.
- A turn can be TOMBSTONED in context while its summary still lives in the cache. The cached summary stays available for recall and rehydrate.

**Folding (bound O(N) tombstone growth):**
- When even the tombstone floor overflows the budget, collapse adjacent tombstone runs into range stubs, for example `[#10-15 6 tool_result turns elided]`.
- Folding only changes the rendered context. Every turn keeps its own Store record and embedding, so recall is unaffected by folding.

**Cacheability rules (get these right or the claims are false):**
- The **embedding** is cacheable, keyed by content hash, because content is immutable.
- The **relevance score** is NOT cacheable. It depends on the current goal, which moves every pass. Cache the embedding once, recompute the dot product against the new goal each pass.
- A generic LLMLingua-2 summary of a turn is cacheable by (content hash, keep-rate). A query-conditioned summary is not. Use generic summaries so they survive a goal change.

---

## 3. Architecture and integration map

```
Learn-tab chat (the trace)
        │  every new message
        ▼
  POST /api/trace/ingest ──► FastAPI /trace/ingest
        │                         │  embed (local MiniLM) + cache; optional speculative LLMLingua-2 summary
        │                         ▼  out-of-context Store (content-hash keyed)
        │
        │  before each Claude call, if history > trigger
        ▼
  POST /api/trace/pack ────► FastAPI /trace/pack
        │                         │  stage 1 segment+type, 2 partition, 3 score, 4 pack, 5 reconstruct, 6 reassemble
        │                         │  SUMMARIZE turns ► existing Modal LLMLingua-2 worker (cached)
        │                         ▼  compact messages + per-turn actions + stats
        ▼
  /api/qa (or chat route) sends the COMPACT history to Claude
```

Two new same-origin Next.js proxy routes (`/api/trace/ingest`, `/api/trace/pack`, `/api/trace/recall`) forward to FastAPI, same pattern as `/api/compress`. The Anthropic key and Modal endpoint stay server-side.

**Trigger:** run a pass only when the live history exceeds about 50% of the model window. Below that, send history unchanged. Compression fires occasionally, not every turn.

**Embeddings:** run a small local model in the FastAPI process for the score and recall path (`sentence-transformers/all-MiniLM-L6-v2`, CPU is fine, batched). This keeps the per-pass latency low and avoids a second Modal deploy. Leave a clearly marked seam to move embeddings to Modal later if you want parity with the compress path. Do not block on this decision; default to local.

**Summarize tier:** reuse the deployed LLMLingua-2 Modal worker. SUMMARIZE is just an LLMLingua-2 compress call at a low keep-rate on that one turn. Cache the result by (content hash, rate).

---

## 4. Backend tasks (FastAPI + Modal)

### 4.1 `trace/` module
Port `trace_packer.py` into a server-side module (for example `trace/core.py`): `Turn`, `Action`, `Record`, `Store`, `pack`, `render_history`, `recall`, `rehydrate`, folding. Replace the demo embedder with the real one. Replace the placeholder summary path so SUMMARIZE calls LLMLingua-2.

### 4.2 Embedding provider
Add `trace/embed.py`: load MiniLM once at startup, expose `embed(texts: list[str]) -> list[list[float]]`, batched. Cache vectors in the Store by content hash.

### 4.3 Summarizer wired to LLMLingua-2
Add `trace/summarize.py`: given a turn and a keep-rate, call the existing Modal compress entrypoint, return the compressed text and its token count. Cache by (content hash, rate). Used both for stage 5 reconstruct and for speculative precompute.

### 4.4 Store (in-memory for now)
A per-session, content-hash-keyed dict holding `{embedding, content, summary, type, index, action}`. In-memory is fine for the hackathon; mark persistence and eviction as out of scope but leave the interface clean so a vector DB can drop in.

### 4.5 Segmentation and typing (stage 1) and partition (stage 2)
- Stage 1: map each chat message to a typed turn. Minimum type set: `goal` (the latest user question), `user`, `assistant`, `source` (pasted or transcript-derived context), and if any tool events exist, `tool_call` / `tool_result` / `error`. Pure parsing, no model.
- Stage 2: keep-zone is never touched (system prompt, the current goal, the last k turns, default k = 4). Everything older is the candidate-zone.

### 4.6 Scoring (stage 3)
Embed the current goal once, cosine against each cached candidate embedding, multiply by a structural prior per type (a `source` dump starts low, an `assistant` decision or an `error` starts high). Output one score per candidate. Recompute every pass (scores are not cached).

### 4.7 Endpoints (FastAPI, then mirrored as Next.js proxies)
- `POST /trace/ingest`: cache embedding (and optionally a speculative summary) for a new turn. Return fast.
- `POST /trace/pack`: run stages 1 to 6, return the compact history and stats.
- `POST /trace/recall`: cosine a query against cached tombstone embeddings, return hits.

### 4.8 Background precompute (the off-critical-path piece)
On `ingest`, embed in a background task so it does not block the response. Behind a flag (`SPECULATIVE_SUMMARY=1`), also pre-compress turns above a token threshold so the summary is warm before any pass fires. By pack time the embeddings and summaries should already be in cache, so a pass is cheap bookkeeping.

---

## 5. Frontend tasks (Next.js, Learn tab)

### 5.1 Wire pack into the chat send path
Before the Learn-tab chat calls Claude, if the history exceeds the trigger, call `/api/trace/pack` and send the returned compact messages instead of the raw history. Below the trigger, send raw.

### 5.2 Ingest on every new message
After each message is appended, fire `/api/trace/ingest` (do not await in a way that blocks typing).

### 5.3 Trace panel (new, in the Learn tab)
A view of the history with a per-message badge: KEEP / SUMMARIZE / TOMBSTONE / ERASE, plus the turn's token count. Tombstones render as the stub. Folded ranges render as a single range chip. Use Framer Motion to animate a pass: turns visibly collapse when compression fires. Add a budget or aggressiveness slider (reuse the existing rate-slider pattern); higher aggressiveness lowers the budget and pushes more turns down the gradient. Reuse the existing model and language pickers.

### 5.4 Recall indicator
When the user (or the assistant about to act) hits a cached tombstone, surface a small chip: "already covered at turn N", with the rehydrated cached line on hover or click. Trigger recall on the latest user message against the session's tombstones.

### 5.5 Fidelity check (reuse the Compare-tab idea)
Add a one-click "verify" that asks Claude the current question twice in parallel: once against the full uncompressed history, once against the compact history, and shows the two answers side by side. This is the task-success measurement that makes the compression credible. Reuse the parallel-Q&A pattern already in the Compare tab.

### 5.6 Stats
Extend the stats bar for the trace pass: history tokens before and after, percent saved, count of keep / summarize / tombstone, and a running "redundant tool calls or repeats avoided by recall" counter.

---

## 6. API contracts

`POST /trace/ingest`
```json
// request
{ "session_id": "abc", "turn": { "index": 12, "type": "assistant", "content": "..." } }
// response
{ "ok": true, "cached": true, "has_summary": false }
```

`POST /trace/pack`
```json
// request
{
  "session_id": "abc",
  "goal": "fix the login bug",
  "turns": [ { "index": 7, "type": "tool_result", "content": "...", "tokens": 1800 } ],
  "budget": 1200,
  "summary_rate": 0.35,
  "keep_threshold": 0.85,
  "summary_threshold": 0.20,
  "keep_last_k": 4
}
// response
{
  "compact_messages": [ { "role": "user", "content": "..." } ],
  "actions": { "7": "tombstone", "12": "summarize", "18": "keep" },
  "folds": [],
  "stats": { "before_tokens": 3655, "after_tokens": 95, "saved_pct": 97.4,
             "n_keep": 1, "n_summarize": 3, "n_tombstone": 2, "n_erase": 1 }
}
```

`POST /trace/recall`
```json
// request
{ "session_id": "abc", "query": "ls src list files in repo" }
// response
{ "hits": [ { "index": 7, "type": "tool_result", "similarity": 0.71,
             "action": "tombstone", "rehydrated": "ls src/ -> 300 files; src has auth, db, api" } ] }
```

---

## 7. Build order (each step independently testable)

1. Port `trace_packer.py` into `trace/core.py`, keep the demo runnable as a smoke test (`python -m trace.core`). Confirm the gradient, the recall hit, the erase check, and the fold path all pass, same as the reference.
2. Add `trace/embed.py` (MiniLM) and swap it into the Store. Confirm recall still fires on a real embedder and retune the recall threshold (the fake embedder used about 0.40; a real one separates much more cleanly).
3. Add `trace/summarize.py` calling the Modal LLMLingua-2 worker; confirm a SUMMARIZE turn comes back compressed and cached.
4. Add the three FastAPI endpoints plus the three Next.js proxies. Smoke-test with curl.
5. Wire `pack` into the Learn-tab chat send path behind the trigger. Confirm Claude answers normally on the compact history.
6. Add `ingest` on message append plus background embedding. Confirm the cache is warm by pack time.
7. Build the Trace panel, badges, slider, recall chip.
8. Add the fidelity check.
9. Optional: speculative summary precompute behind the flag.

---

## 8. Acceptance criteria

- A Learn-tab session that grows past the trigger compresses its history: the panel shows a real KEEP / SUMMARIZE / TOMBSTONE gradient, not all-keep or all-drop, and `ERASE` only ever appears for `must_purge` content.
- No turn is ever dropped to zero bytes for budget reasons. Every non-purged candidate has at least a tombstone or a fold range in context.
- Recall: after a turn is tombstoned, asking a semantically similar question surfaces a recall chip and the rehydrated cached line, and the rehydrate works even when that turn's stub has been folded into a range.
- Cacheability: re-running a pass with the same goal does no new embedding work (cache hits); changing the goal recomputes scores without re-embedding.
- Fidelity: the verify view shows that Claude's answer on the compact history matches its answer on the full history for a held-out question.
- The secret-leak check: a `must_purge` turn appears in neither the compact context nor the Store.
- Latency: with the cache warm, a pack pass is dominated by the LLMLingua-2 calls for SUMMARIZE turns only, not by embedding the whole history.

---

## 9. Out of scope (note, do not build)

- Persistence and eviction for the Store (in-memory is fine; keep the interface clean).
- A learned ranker. The scorer is cosine similarity times a hand-set structural prior. Do not call it "learned" anywhere in code or copy.
- Replacing LLMLingua-2 with an abstractive summarizer. Token-level compression is the engine; keep it.

When you finish, write a short `trace/README.md` (no em dashes) covering the endpoints, the trigger, the cacheability rules, and how to run the smoke test.
