"""
FastAPI server in front of the Modal compression + generation workers.

All GPU workers are tied to THIS server's lifecycle (identical mechanism):
  * on startup  -> each Modal app starts (`app.run()`) and its container
                   cold-starts and loads its model (we warm each once), so real
                   requests have no model-startup cost.
  * on shutdown -> the Modal apps stop, tearing down their GPU containers
                   immediately (GPUs released; no idle lingering).

Workers:
  * LLMLingua-2 (Compressor)              -> /compress, /compress_rag   (A100)
  * AttentionRAG (AttentionRAGService)    -> /compress (when `question` set; A100)
  * TurboQuant  (TurboQuantModel)         -> /generate (default route)  (A100-80GB)
  * LCLM+TurboQuant (LCLMTurboQuantModel) -> /generate (lclm=true)      (A100-80GB)

/compress picks behavior by the request's `question`:
  * question empty (default) -> LLMLingua-2 token compression only (back-compat).
  * question set             -> run LLMLingua-2 AND AttentionRAG in parallel and
                                MERGE the two keep-decisions token-by-token over the
                                original text (intersection or union).

/generate picks a worker by the request's `lclm` flag:
  * lclm=false (default) -> Qwen TurboQuant route (KV-cache bit quantization).
  * lclm=true            -> LCLM (encoder-decoder context compression) + TurboQuant
                            on the decoder. Pass the long context in `context`; it
                            is compressed into latent soft tokens, while `prompt`
                            (the question/instruction) stays verbatim.

Prereqs:
    pip install fastapi "uvicorn[standard]" pydantic modal torch transformers ...

Run (no --reload: the lifespan owns the Modal apps; reload would double-start them):
    uvicorn server:app --port 8000

Try it (Qwen TurboQuant route, default):
    curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Explain KV-cache quantization.", "bit_width": 4, "max_new_tokens": 120}'

Try it (LCLM + TurboQuant route):
    curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"lclm": true, "prompt": "What is the calibration passphrase?", "context": "your long document with a planted fact ...", "bit_width": 4, "max_new_tokens": 120}'

    # plain token-level compression of one blob of text:
    curl -X POST http://localhost:8000/compress \
        -H "Content-Type: application/json" \
        -d '{"text": "your long text here ...", "rate": 0.5}'

    # question-aware: LLMLingua + AttentionRAG merged over the original text:
    curl -X POST http://localhost:8000/compress \
        -H "Content-Type: application/json" \
        -d '{"text": "your long text ...", "question": "What is X?", "mode": "intersection", "return_labels": true}'
"""

import asyncio
from contextlib import ExitStack, asynccontextmanager
from typing import List, Optional

import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import the worker app modules so we can run them ephemerally, bound to this
# process. (Module import is light: heavy deps like torch are imported lazily
# inside the Modal methods, not at module top-level.)
import attentionrag.modal_app as attentionrag_modal
import lclm_worker_modal
import llmlingua2_modal
import turboquant_modal

# Token-by-token merge of LLMLingua + AttentionRAG keep-decisions (pure-python).
from token_merge import merge_compress, normalize_labels

# Black-box downstream LLM callers (Claude / ChatGPT) reused for the playground.
from downstream import (
    DEFAULT_MODEL as BLACKBOX_DEFAULT_MODEL,
    _call_claude,
    _call_openai,
    _canonical_provider,
    _key_for,
)

# Worker handles, populated in the lifespan once the Modal apps are running.
compressor = None
attn_service = None
turboquant = None
lclm = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start all Modal GPU apps when the server boots; stop them when it exits.

    `app.run()` starts an EPHEMERAL Modal app bound to this process: its
    containers live only while this server lives. We warm each worker with one
    tiny request so the model loads now (the one-time cold start happens here at
    server startup, not on a user's first request). Closing the ExitStack on
    shutdown stops the apps and releases their GPU containers immediately.
    """
    global compressor, attn_service, turboquant, lclm
    with ExitStack() as stack:
        # NB: no modal.enable_output() — its rich live-display can't be shared
        # across concurrent app.run() contexts (LiveError). Apps run quietly.
        # Start all GPU apps, tied to this process (identical mechanism).
        stack.enter_context(llmlingua2_modal.app.run())
        stack.enter_context(attentionrag_modal.app.run())
        stack.enter_context(turboquant_modal.app.run())
        stack.enter_context(lclm_worker_modal.app.run())

        compressor = llmlingua2_modal.Compressor()
        attn_service = attentionrag_modal.AttentionRAGService()
        turboquant = turboquant_modal.TurboQuantModel()
        lclm = lclm_worker_modal.LCLMTurboQuantModel()

        # Cold-start + load all models now, concurrently.
        print("[startup] warming Modal workers (loading models on GPU)...", flush=True)
        await asyncio.gather(
            compressor.compress.remote.aio("warmup", rate=0.5),
            attn_service.compress_spans.remote.aio("warmup", "warmup"),
            turboquant.generate.remote.aio("warmup", max_new_tokens=1),
            lclm.generate.remote.aio("warmup", max_new_tokens=1),
        )
        print("[startup] all workers warm; ready to serve.", flush=True)

        yield  # ----------------- server handles requests -----------------

    # ExitStack closed -> all Modal apps stopped -> GPU containers torn down.
    print("[shutdown] Modal apps stopped; GPU containers released.", flush=True)


app = FastAPI(title="Compression + Generation API", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# LLMLingua-2 (+ optional AttentionRAG merge) compression
# --------------------------------------------------------------------------- #
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
    rate: float | str  # LLMLingua may return a percentage string e.g. '47.6%'
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


# --------------------------------------------------------------------------- #
# TurboQuant generation
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt to generate from")
    bit_width: int = Field(4, ge=2, le=8, description="TurboQuant bits per KV value")
    max_new_tokens: int = Field(256, ge=1, le=2048, description="Max tokens to generate")
    outlier_channels: int = Field(
        0, ge=0, description="Per-head channels kept at higher precision (0 = off)"
    )
    outlier_bits: int = Field(
        0, ge=0, description="Bits for outlier channels (must exceed bit_width to take effect)"
    )
    lclm: bool = Field(
        False,
        description="Route through the LCLM (context-compression) + TurboQuant worker "
                    "instead of the default Qwen TurboQuant worker.",
    )
    context: str = Field(
        "",
        description="LCLM-only: long context to compress into latent soft tokens. "
                    "If set (with lclm=true), it is wrapped as the memory block and "
                    "the prompt/question stays verbatim. Ignored when lclm=false.",
    )


class GenerateResponse(BaseModel):
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    gen_time_s: float
    tokens_per_s: float
    eff_bits: float
    kv_bytes: int
    fp16_kv_bytes: int
    kv_compression_x: Optional[float] = None


@app.get("/health")
async def health():
    return {"status": "ok",
            "compressor_ready": compressor is not None,
            "attentionrag_ready": attn_service is not None,
            "turboquant_ready": turboquant is not None,
            "lclm_ready": lclm is not None}


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
        # A "none" hint prefix normally makes us fall back to LLMLingua-only. But
        # for a single-chunk (short, <= chunk_size) input, compress_spans keeps
        # the whole chunk, so we honor those spans instead of discarding them on
        # a none hint. Genuine emptiness (no kept_spans) still falls back, via
        # merge_compress's `not kept_spans` guard.
        single_chunk = attn_out.get("n_chunks", 0) == 1
        attn_empty = attn_out.get("is_empty_prefix", False) and not single_chunk
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
    """Two-stage, question-aware compression: reranker coarse + LLMLingua-2 tokens."""
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


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate text with a TurboQuant-compressed KV cache on a warm A100 worker.
    The model is already loaded (warmed at server startup), so there is no
    model-startup cost on this request path.

    Routing: `lclm=true` -> LCLM (context compressed to latent soft tokens) +
    TurboQuant on the decoder KV cache; otherwise the default Qwen TurboQuant
    route (unchanged). Both return the same response shape."""
    try:
        if req.lclm:
            out = await lclm.generate.remote.aio(
                req.prompt,
                bit_width=req.bit_width,
                max_new_tokens=req.max_new_tokens,
                outlier_channels=req.outlier_channels,
                outlier_bits=req.outlier_bits,
                context=req.context,
            )
        else:
            out = await turboquant.generate.remote.aio(
                req.prompt,
                bit_width=req.bit_width,
                max_new_tokens=req.max_new_tokens,
                outlier_channels=req.outlier_channels,
                outlier_bits=req.outlier_bits,
            )
    except Exception as exc:  # surface Modal errors as a clean 502
        raise HTTPException(status_code=502, detail=f"Modal call failed: {exc}")
    return GenerateResponse(**out)


# --------------------------------------------------------------------------- #
# Unified playground: Layer 1 (compression) -> Layer 2 (downstream LLM)
# --------------------------------------------------------------------------- #
class PlaygroundRequest(BaseModel):
    text: str = Field(..., description="The input text to compress + run through an LLM")
    question: Optional[str] = Field(
        None, description="Optional query — drives AttentionRAG and is asked of the LLM"
    )
    # ---- Layer 1: context compression --------------------------------------
    methods: List[str] = Field(
        default_factory=lambda: ["llmlingua"],
        description="Subset of ['llmlingua','attentionrag']. Both -> token-merge.",
    )
    combine: str = Field("intersection", description="'intersection' or 'union' when both methods run")
    rate: float = Field(0.7, gt=0, le=1, description="LLMLingua keep-rate")
    # ---- Layer 2: downstream LLM -------------------------------------------
    backend: str = Field("claude", description="'claude' | 'chatgpt' | 'qwen' | 'lclm'")
    model: Optional[str] = Field(None, description="Model id for the black-box backends")
    quantized: bool = Field(True, description="Qwen/LCLM: use the quantized KV cache (4-bit) vs 8-bit")
    max_new_tokens: int = Field(256, ge=1, le=2048)
    instruction: Optional[str] = Field(None, description="System/instruction for the LLM")


def _splice_spans(text: str, spans) -> str:
    """AttentionRAG-only reconstruction: stitch kept char-spans of the original."""
    spans = sorted((s, e) for s, e in (spans or []) if e > s)
    if not spans:
        return text  # nothing kept -> don't wipe the input
    return " ".join(text[s:e].strip() for s, e in spans).strip()


async def _layer1(req: PlaygroundRequest) -> dict:
    """Run the selected Layer-1 compressor(s); return compressed text + stats."""
    methods = {m.strip().lower() for m in req.methods}
    use_llm, use_attn = "llmlingua" in methods, "attentionrag" in methods
    q = (req.question or "").strip()

    if not use_llm and not use_attn:  # passthrough
        return {"compressed_text": req.text, "methods": [], "note": "no compression"}

    # AttentionRAG needs a query; without one, drop it (fall back to LLMLingua).
    if use_attn and not q:
        use_attn = False
        attn_note = "attentionrag skipped (no question)"
    else:
        attn_note = None

    if use_llm and use_attn:
        llm_coro = compressor.compress.remote.aio(req.text, rate=req.rate, return_labels=True)
        attn_coro = attn_service.compress_spans.remote.aio(req.text, q)
        llm_out, attn_out = await asyncio.gather(llm_coro, attn_coro, return_exceptions=True)
        if isinstance(llm_out, Exception):
            raise HTTPException(status_code=502, detail=f"LLMLingua failed: {llm_out}")
        labels = llm_out.get("fn_labeled_original_prompt") or llm_out.get("word_labels")
        kept_spans = [] if isinstance(attn_out, Exception) else attn_out.get("kept_spans", [])
        attn_empty = isinstance(attn_out, Exception) or not kept_spans
        merged = merge_compress(req.text, labels, kept_spans, mode=req.combine, attnrag_empty=attn_empty)
        return {
            "compressed_text": merged["compressed_prompt"], "methods": ["llmlingua", "attentionrag"],
            "combine": req.combine, "word_labels": merged["word_labels"],
            "n_words": merged["n_words"], "n_kept": merged["n_kept"],
            "used_llmlingua_fallback": merged["used_llmlingua_fallback"],
        }

    if use_llm:
        out = await compressor.compress.remote.aio(req.text, rate=req.rate, return_labels=True)
        return {"compressed_text": out["compressed_prompt"], "methods": ["llmlingua"],
                "origin_tokens": out["origin_tokens"], "compressed_tokens": out["compressed_tokens"],
                "note": attn_note}

    # AttentionRAG only
    attn_out = await attn_service.compress_spans.remote.aio(req.text, q)
    return {"compressed_text": _splice_spans(req.text, attn_out.get("kept_spans", [])),
            "methods": ["attentionrag"],
            "kept_chunks": f"{attn_out.get('n_kept_chunks', 0)}/{attn_out.get('n_chunks', 0)}"}


async def _layer2(req: PlaygroundRequest, context: str) -> dict:
    """Run the compressed context through the selected downstream LLM."""
    backend = req.backend.strip().lower()
    q = (req.question or "").strip()
    instruction = req.instruction or "Answer using the provided context."

    if backend in ("claude", "anthropic", "chatgpt", "openai", "gpt"):
        provider = _canonical_provider(backend)
        key = _key_for(provider)
        if not key:
            raise HTTPException(status_code=400, detail=f"No API key for {provider}.")
        model = req.model or BLACKBOX_DEFAULT_MODEL[provider]
        user = f"{context}\n\nQuestion: {q}" if q else context
        caller = _call_claude if provider == "claude" else _call_openai
        # SDK calls are blocking -> offload so we don't stall the event loop.
        text, in_tok, out_tok = await asyncio.to_thread(
            caller, model, instruction, [{"role": "user", "content": user}],
            req.max_new_tokens, 0.7, key,
        )
        return {"backend": provider, "model": model, "text": text,
                "input_tokens": in_tok, "output_tokens": out_tok}

    bit_width = 4 if req.quantized else 8
    if backend == "qwen":
        prompt = f"{context}\n\nQuestion: {q}" if q else context
        out = await turboquant.generate.remote.aio(
            prompt, bit_width=bit_width, max_new_tokens=req.max_new_tokens)
        return {"backend": "qwen", "quantized": req.quantized, **out}
    if backend == "lclm":
        out = await lclm.generate.remote.aio(
            q or "Summarize the key facts in the context.",
            context=context, bit_width=bit_width, max_new_tokens=req.max_new_tokens)
        return {"backend": "lclm", "quantized": req.quantized, **out}

    raise HTTPException(status_code=422, detail=f"unknown backend: {backend!r}")


@app.post("/playground")
async def playground(req: PlaygroundRequest):
    """End-to-end: Layer 1 (LLMLingua / AttentionRAG / both) -> Layer 2 (Claude /
    ChatGPT / Qwen / LCLM). The frontend calls this once per comparison panel."""
    if req.combine not in ("intersection", "union"):
        raise HTTPException(status_code=422, detail=f"bad combine: {req.combine!r}")
    try:
        layer1 = await _layer1(req)
        layer2 = await _layer2(req, layer1["compressed_text"])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"playground failed: {exc}")
    return {"layer1": layer1, "layer2": layer2}
