"""
FastAPI server in front of the LLMLingua-2 Modal worker.

POST text -> get compressed text back. The actual compression runs on a GPU in
Modal; this server just looks up the deployed Modal class and calls it.

Prereqs:
    pip install fastapi "uvicorn[standard]" pydantic modal
    modal deploy llmlingua2_modal.py     # deploy the GPU worker first

Run:
    uvicorn server:app --reload --port 8000

Try it:
    # plain token-level compression of one blob of text:
    curl -X POST http://localhost:8000/compress \
        -H "Content-Type: application/json" \
        -d '{"text": "your long text here ...", "rate": 0.5}'

    # two-stage, question-aware RAG compression of a list of documents:
    curl -X POST http://localhost:8000/compress_rag \
        -H "Content-Type: application/json" \
        -d '{"instruction": "Answer using only the context.",
             "question": "What is there to see in Paris?",
             "documents": ["doc one ...", "doc two ...", "doc three ..."],
             "rate": 0.5, "top_k": 3}'
"""

from typing import List, Optional

import modal
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from trace import Action, OurPolicy
from trace.core import Turn, count_tokens, render_history
from trace.strategies import rehydrate

# Must match the app/class names in llmlingua2_modal.py.
MODAL_APP_NAME = "llmlingua2-xlm"
MODAL_CLASS_NAME = "Compressor"

app = FastAPI(title="LLMLingua-2 Compression API")

# Look up the deployed Modal class once at import time and keep a single
# instance handle; Modal routes calls to a warm GPU container (or cold-starts one).
Compressor = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
compressor = Compressor()


class CompressRequest(BaseModel):
    text: str = Field(..., description="Text to compress")
    rate: float = Field(0.5, gt=0, le=1, description="Fraction of tokens to keep")
    return_labels: bool = Field(False, description="Include per-word keep/discard labels")


class CompressResponse(BaseModel):
    compressed_prompt: str
    origin_tokens: int
    compressed_tokens: int
    rate: float
    ratio: str
    # LLMLingua-2 per-word keep/drop labels: list of (word, 1|0). Optional —
    # populated only when return_labels=True. Powers the strike-through diff UI.
    word_labels: list | None = None


class RagRequest(BaseModel):
    instruction: str = Field("", description="System/task instruction (kept verbatim)")
    question: str = Field(..., description="User query (drives ranking, kept verbatim)")
    documents: List[str] = Field(..., description="Retrieved chunks, one per element")
    rate: float = Field(0.5, gt=0, le=1, description="Fine-stage fraction of tokens to keep")
    target_token: int = Field(-1, description="Hard token budget for the context (-1 = use rate)")
    top_k: Optional[int] = Field(None, description="Max documents to keep in the coarse stage")
    score_threshold: Optional[float] = Field(
        None, description="Min reranker score [0,1] to keep a document"
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/compress", response_model=CompressResponse)
async def compress(req: CompressRequest):
    try:
        # .aio() makes the remote call awaitable so the server stays non-blocking.
        out = await compressor.compress.remote.aio(
            req.text, rate=req.rate, return_labels=req.return_labels
        )
    except Exception as exc:  # surface Modal errors as a clean 502
        raise HTTPException(status_code=502, detail=f"Modal call failed: {exc}")

    return CompressResponse(
        compressed_prompt=out["compressed_prompt"],
        origin_tokens=out["origin_tokens"],
        compressed_tokens=out["compressed_tokens"],
        rate=out["rate"],
        ratio=out["ratio"],
        word_labels=out.get("fn_labeled_original_prompt") or out.get("word_labels"),
    )


@app.post("/compress_rag")
async def compress_rag(req: RagRequest):
    """Two-stage, question-aware compression: reranker coarse + LLMLingua-2 tokens.

    Returns the assembled prompt plus token counts and coarse-stage diagnostics
    (the dict from two_stage_compressor.two_stage_compress).
    """
    try:
        out = await compressor.compress_rag.remote.aio(
            req.instruction,
            req.question,
            req.documents,
            rate=req.rate,
            target_token=req.target_token,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
        )
    except Exception as exc:  # surface Modal errors as a clean 502
        raise HTTPException(status_code=502, detail=f"Modal call failed: {exc}")

    return out


# ---------------------------------------------------------------------------
# Trace mode: turn-level context compression on top of the per-utterance path.
#
# The trace engine lives in the trace/ module (packer, MiniLM embedder, Modal
# SUMMARIZE tier, the OurPolicy strategy). These endpoints are a thin session
# layer over it. One OurPolicy per session_id holds that session's content-hash
# keyed Store, so observe() (ingest) and recall() share the same cache that
# compress() (pack) populates.
# ---------------------------------------------------------------------------

# Per-session policies. In-memory for now (see trace_spec.md: persistence and
# eviction are out of scope). The Store inside each policy is the session cache.
_sessions: dict[str, OurPolicy] = {}


def _get_policy(session_id: str, stub_mode: str = "stub") -> OurPolicy:
    """Lazily create one OurPolicy per session. Construction is cheap: the MiniLM
    model loads lazily on the first embed call, not here."""
    pol = _sessions.get(session_id)
    if pol is None or pol.stub_mode != stub_mode:
        pol = OurPolicy(stub_mode=stub_mode)
        _sessions[session_id] = pol
    return pol


class IngestTurn(BaseModel):
    index: int
    type: str
    content: str
    tokens: int | None = Field(None, description="Verbatim cost; estimated if omitted")
    summary: str | None = None
    must_purge: bool = Field(False, description="Secrets/PII: force ERASE, never cached")


class IngestRequest(BaseModel):
    session_id: str
    turn: IngestTurn


@app.post("/trace/ingest")
async def trace_ingest(req: IngestRequest):
    """Observe a new turn: cache its embedding off the critical path. Returns fast."""
    pol = _get_policy(req.session_id)
    t = req.turn
    turn = Turn(
        index=t.index,
        type=t.type,
        content=t.content,
        tokens=t.tokens if t.tokens is not None else count_tokens(t.content),
        summary=t.summary,
        must_purge=t.must_purge,
    )
    # observe() embeds locally (blocking CPU work), so keep it off the event loop.
    await run_in_threadpool(pol.observe, turn)
    return {
        "ok": True,
        "cached": not t.must_purge,  # must_purge turns are never registered
        "has_summary": t.summary is not None,
    }


class PackTurn(BaseModel):
    index: int
    type: str
    content: str
    tokens: int | None = None
    summary: str | None = None
    summary_tokens: int = 0
    score: float = 0.0
    must_purge: bool = False


class PackRequest(BaseModel):
    session_id: str
    goal: str
    turns: list[PackTurn]
    budget: int = 1200
    summary_rate: float = 0.35
    keep_threshold: float = 0.85
    summary_threshold: float = 0.20
    keep_last_k: int = 4
    stub_mode: str = "stub"


@app.post("/trace/pack")
async def trace_pack(req: PackRequest):
    """Run the packer over the history and return the compact context plus stats."""
    pol = _get_policy(req.session_id, stub_mode=req.stub_mode)
    # Per-pass knobs (the aggressiveness slider varies these between passes).
    pol.keep_threshold = req.keep_threshold
    pol.summary_threshold = req.summary_threshold
    pol.summary_rate = req.summary_rate

    all_turns = [
        Turn(
            index=t.index,
            type=t.type,
            content=t.content,
            tokens=t.tokens if t.tokens is not None else count_tokens(t.content),
            summary=t.summary,
            summary_tokens=t.summary_tokens,
            score=t.score,
            must_purge=t.must_purge,
        )
        for t in req.turns
    ]

    # Stage 2 partition: the most recent keep_last_k turns are the protected
    # keep-zone (rendered verbatim, never packed); the rest are candidates.
    # must_purge turns are never eligible for the keep-zone: the keep-zone renders
    # verbatim and would leak the secret, so they always fall to the packer, which
    # ERASEs them (zero bytes, no cache entry).
    ordered = sorted(all_turns, key=lambda x: x.index)
    k = max(0, req.keep_last_k)
    keepable = [t for t in ordered if not t.must_purge]
    keep_zone = keepable[len(keepable) - k:] if k else []
    keep_idx = {t.index for t in keep_zone}
    candidates = [t for t in ordered if t.index not in keep_idx]

    try:
        # compress() scores, packs, runs the stage-5 SUMMARIZE reconstruct (which
        # may call the Modal worker), and applies the erase ablation.
        annotated = await run_in_threadpool(pol.compress, candidates, req.goal, req.budget)
    except Exception as exc:  # surface a Modal/summarize failure as a clean 502
        raise HTTPException(status_code=502, detail=f"Trace pack failed: {exc}")

    plan = pol.last_plan
    text, after = render_history(keep_zone, annotated, plan)

    before = sum(t.tokens for t in all_turns)
    counts = {a: 0 for a in (Action.KEEP, Action.SUMMARIZE, Action.TOMBSTONE, Action.ERASE)}
    for t in annotated:
        if t.action in counts:
            counts[t.action] += 1

    return {
        "compact_messages": [{"role": "user", "content": text}],
        "actions": {str(t.index): t.action.value for t in annotated if t.action},
        "folds": [[t.index for t in grp] for grp in plan.folds],
        "stats": {
            "before_tokens": before,
            "after_tokens": after,
            "saved_pct": round(100 * (before - after) / before, 1) if before else 0.0,
            "n_keep": counts[Action.KEEP],
            "n_summarize": counts[Action.SUMMARIZE],
            "n_tombstone": counts[Action.TOMBSTONE],
            "n_erase": counts[Action.ERASE],
        },
    }


class RecallRequest(BaseModel):
    session_id: str
    query: str


@app.post("/trace/recall")
async def trace_recall(req: RecallRequest):
    """Cosine the query against this session's cached tombstone embeddings."""
    pol = _sessions.get(req.session_id)
    if pol is None:
        return {"hits": []}
    hits = await run_in_threadpool(pol.recall, req.query)
    return {
        "hits": [
            {
                "index": h.record.index,
                "type": h.record.turn_type,
                "similarity": round(h.similarity, 3),
                "action": h.record.action.value,
                "rehydrated": rehydrate(h),
            }
            for h in hits
        ]
    }
