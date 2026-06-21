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
from pydantic import BaseModel, Field

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
