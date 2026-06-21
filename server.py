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

import asyncio
from typing import List, Optional

import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from token_merge import merge_compress, normalize_labels

# Must match the app/class names in llmlingua2_modal.py.
MODAL_APP_NAME = "llmlingua2-xlm"
MODAL_CLASS_NAME = "Compressor"
# AttentionRAG worker (deploy with: modal deploy attentionrag/modal_app.py).
ATTN_APP_NAME = "attentionrag"
ATTN_CLASS_NAME = "AttentionRAGService"

app = FastAPI(title="LLMLingua-2 + AttentionRAG Compression API")

# Look up the deployed Modal classes once at import time and keep single instance
# handles; Modal routes calls to a warm GPU container (or cold-starts one).
Compressor = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
compressor = Compressor()
AttnService = modal.Cls.from_name(ATTN_APP_NAME, ATTN_CLASS_NAME)
attn_service = AttnService()


class CompressRequest(BaseModel):
    text: str = Field(..., description="Text to compress")
    rate: float = Field(0.5, gt=0, le=1, description="LLMLingua fraction of tokens to keep")
    return_labels: bool = Field(False, description="Include per-word keep/discard labels")
    # --- AttentionRAG + merge controls -----------------------------------
    # When `question` is set, AttentionRAG runs in parallel with LLMLingua and
    # the two keep-decisions are merged token-by-token over the original text.
    # When `question` is empty, only LLMLingua runs (back-compat behavior).
    question: Optional[str] = Field(
        None, description="Query for AttentionRAG; enables the parallel merge"
    )
    mode: str = Field(
        "intersection", description="'intersection' (both keep) or 'union' (either keeps)"
    )
    chunk_size: int = Field(300, gt=0, description="AttentionRAG chunk size (tokens)")
    top_k: int = Field(12, gt=0, description="AttentionRAG top-k tokens per chunk")
    use_openai_hint: bool = Field(
        False, description="Author AttentionRAG hint prefix with GPT-4o-mini"
    )


class CompressResponse(BaseModel):
    compressed_prompt: str
    origin_tokens: int
    compressed_tokens: int
    rate: float
    ratio: str
    # Per-word keep/drop labels: list of (word, 1|0). When a merge ran these are
    # the MERGED labels; otherwise LLMLingua's. Powers the strike-through diff UI.
    word_labels: Optional[list] = None
    # --- merge diagnostics (populated only when a merge ran) --------------
    mode: Optional[str] = None
    merged: bool = False
    used_llmlingua_fallback: Optional[bool] = None
    words_total: Optional[int] = None
    words_kept: Optional[int] = None
    merged_ratio: Optional[str] = None
    attentionrag_hint_prefix: Optional[str] = None
    attentionrag_kept_chunks: Optional[str] = None


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


def _parse_rate(v) -> float:
    """LLMLingua returns `rate` as a display string like '45.0%'; coerce to a
    float fraction in [0,1] for the response model."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    pct = s.endswith("%")
    try:
        f = float(s.rstrip("%"))
    except ValueError:
        return 0.0
    return f / 100.0 if pct else f


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/compress", response_model=CompressResponse)
async def compress(req: CompressRequest):
    """Compress text with LLMLingua-2, and -- when `question` is provided -- run
    AttentionRAG in parallel and MERGE the two keep-decisions token-by-token over
    the original text (intersection or union). Without `question`, behaves as the
    original LLMLingua-only endpoint.
    """
    if req.mode not in ("intersection", "union"):
        raise HTTPException(status_code=422, detail=f"bad mode: {req.mode!r}")

    merging = bool(req.question and req.question.strip())
    # LLMLingua is the canonical spine for the merge, so we need its word labels.
    need_labels = req.return_labels or merging

    llm_coro = compressor.compress.remote.aio(
        req.text, rate=req.rate, return_labels=need_labels
    )

    if not merging:
        try:
            out = await llm_coro
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Modal call failed: {exc}")
        return CompressResponse(
            compressed_prompt=out["compressed_prompt"],
            origin_tokens=out["origin_tokens"],
            compressed_tokens=out["compressed_tokens"],
            rate=_parse_rate(out["rate"]),
            ratio=out["ratio"],
            word_labels=[
                [w, l] for w, l in normalize_labels(
                    out.get("fn_labeled_original_prompt") or out.get("word_labels")
                )
            ] if req.return_labels else None,
            merged=False,
        )

    # --- parallel: LLMLingua + AttentionRAG -------------------------------
    attn_coro = attn_service.compress_spans.remote.aio(
        req.text,
        req.question,
        chunk_size=req.chunk_size,
        top_k=req.top_k,
        use_openai_hint=req.use_openai_hint,
    )
    llm_out, attn_out = await asyncio.gather(
        llm_coro, attn_coro, return_exceptions=True
    )

    if isinstance(llm_out, Exception):
        raise HTTPException(status_code=502, detail=f"LLMLingua failed: {llm_out}")
    word_labels = llm_out.get("fn_labeled_original_prompt") or llm_out.get("word_labels")
    if not word_labels:
        raise HTTPException(status_code=502, detail="LLMLingua returned no labels")

    # AttentionRAG failure -> treat as empty -> fallback to LLMLingua-only.
    if isinstance(attn_out, Exception):
        kept_spans, attn_empty = [], True
        hint, kept_chunks = None, "0/0 (attnrag failed)"
    else:
        kept_spans = attn_out.get("kept_spans", [])
        attn_empty = attn_out.get("is_empty_prefix", False)
        hint = attn_out.get("hint_prefix")
        kept_chunks = f"{attn_out.get('n_kept_chunks', 0)}/{attn_out.get('n_chunks', 0)}"

    merged = merge_compress(
        req.text, word_labels, kept_spans, mode=req.mode, attnrag_empty=attn_empty
    )
    ratio = merged["n_words"] / max(merged["n_kept"], 1)
    return CompressResponse(
        compressed_prompt=merged["compressed_prompt"],
        origin_tokens=llm_out["origin_tokens"],
        compressed_tokens=llm_out["compressed_tokens"],
        rate=_parse_rate(llm_out["rate"]),
        ratio=llm_out["ratio"],
        word_labels=merged["word_labels"],
        mode=merged["mode"],
        merged=True,
        used_llmlingua_fallback=merged["used_llmlingua_fallback"],
        words_total=merged["n_words"],
        words_kept=merged["n_kept"],
        merged_ratio=f"{ratio:.2f}x",
        attentionrag_hint_prefix=hint,
        attentionrag_kept_chunks=kept_chunks,
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
