"""
FastAPI server in front of the Modal compression + generation workers.

Both GPU workers are tied to THIS server's lifecycle (identical mechanism):
  * on startup  -> both Modal apps start (`app.run()`) and their containers
                   cold-start and load their models (we warm each once), so real
                   requests have no model-startup cost.
  * on shutdown -> both Modal apps stop, tearing down their GPU containers
                   immediately (GPUs released; no 30-min idle lingering).

Workers:
  * LLMLingua-2 (Compressor)            -> /compress, /compress_rag   (T4)
  * TurboQuant  (TurboQuantModel)       -> /generate (default route)  (A100-80GB)
  * LCLM+TurboQuant (LCLMTurboQuantModel) -> /generate (lclm=true)    (A100-80GB)

The /generate route picks a worker by the request's `lclm` flag:
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

    curl -X POST http://localhost:8000/compress \
        -H "Content-Type: application/json" \
        -d '{"text": "your long text here ...", "rate": 0.5}'
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
import llmlingua2_modal
import lclm_worker_modal
import turboquant_modal

# Worker handles, populated in the lifespan once the Modal apps are running.
compressor = None
turboquant = None
lclm = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start both Modal GPU apps when the server boots; stop them when it exits.

    `app.run()` starts an EPHEMERAL Modal app bound to this process: its
    containers live only while this server lives. We warm each worker with one
    tiny request so the model loads now (the one-time cold start happens here at
    server startup, not on a user's first request). Closing the ExitStack on
    shutdown stops both apps and releases their GPU containers immediately.
    """
    global compressor, turboquant, lclm
    with ExitStack() as stack:
        # NB: no modal.enable_output() — its rich live-display can't be shared
        # across concurrent app.run() contexts (LiveError). Apps run quietly.
        # Start all GPU apps, tied to this process (identical mechanism).
        stack.enter_context(llmlingua2_modal.app.run())
        stack.enter_context(turboquant_modal.app.run())
        stack.enter_context(lclm_worker_modal.app.run())

        compressor = llmlingua2_modal.Compressor()
        turboquant = turboquant_modal.TurboQuantModel()
        lclm = lclm_worker_modal.LCLMTurboQuantModel()

        # Cold-start + load all models now, concurrently.
        print("[startup] warming Modal workers (loading models on GPU)...", flush=True)
        await asyncio.gather(
            compressor.compress.remote.aio("warmup", rate=0.5),
            turboquant.generate.remote.aio("warmup", max_new_tokens=1),
            lclm.generate.remote.aio("warmup", max_new_tokens=1),
        )
        print("[startup] all workers warm; ready to serve.", flush=True)

        yield  # ----------------- server handles requests -----------------

    # ExitStack closed -> both Modal apps stopped -> GPU containers torn down.
    print("[shutdown] Modal apps stopped; GPU containers released.", flush=True)


app = FastAPI(title="Compression + Generation API", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# LLMLingua-2 compression
# --------------------------------------------------------------------------- #
class CompressRequest(BaseModel):
    text: str = Field(..., description="Text to compress")
    rate: float = Field(0.5, gt=0, le=1, description="Fraction of tokens to keep")
    return_labels: bool = Field(False, description="Include per-word keep/discard labels")


class CompressResponse(BaseModel):
    compressed_prompt: str
    origin_tokens: int
    compressed_tokens: int
    rate: float | str  # LLMLingua may return a percentage string e.g. '47.6%'
    ratio: str
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
    return {"status": "ok", "compressor_ready": compressor is not None,
            "turboquant_ready": turboquant is not None,
            "lclm_ready": lclm is not None}


@app.post("/compress", response_model=CompressResponse)
async def compress(req: CompressRequest):
    try:
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
