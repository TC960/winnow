"""
Downstream LLM endpoint — provider-agnostic completion behind one HTTP API.

This is the final hop of the pipeline: (Deepgram -> compressor ->) DOWNSTREAM LLM.
It lets the frontend pick the provider per request — Claude (Anthropic) or
ChatGPT (OpenAI) — by sending a `provider` field. The compressed prompt is just
text, so it works with either black-box API.

Endpoints
    GET  /health             liveness
    GET  /providers          which providers are configured (have a key) + defaults
                             -> use this to populate the frontend's model picker
    POST /generate           run a prompt through the selected provider

Keys (from environment or this folder's .env):
    CLAUDE_API_KEY  or ANTHROPIC_API_KEY   -> Claude
    OPENAI_API_KEY                          -> ChatGPT

Run standalone:
    pip install fastapi "uvicorn[standard]" anthropic openai
    python downstream.py                 # serves on :8100
    # or: uvicorn downstream:app --port 8100

The frontend should proxy to this (same pattern as web/app/api/compress/route.ts)
so keys stay server-side. POST body example:
    {"provider": "chatgpt", "prompt": "Summarize: ...", "model": "gpt-4o-mini"}
"""

import os
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- load keys from this folder's .env if present (env vars win) -------------
def _load_dotenv():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    import re
    for line in open(path):
        m = re.match(r"(?:export\s+)?([A-Za-z0-9_]+)\s*=\s*(.*)", line.strip())
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))


_load_dotenv()

# --- provider config --------------------------------------------------------
# Friendly aliases the frontend may send -> canonical provider name.
PROVIDER_ALIASES = {
    "claude": "claude", "anthropic": "claude",
    "chatgpt": "openai", "openai": "openai", "gpt": "openai",
}
DEFAULT_MODEL = {"claude": "claude-sonnet-4-6", "openai": "gpt-4o-mini"}
# Suggested models for the frontend dropdown (any model string is still accepted).
SUGGESTED_MODELS = {
    "claude": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
}


def _key_for(provider: str) -> Optional[str]:
    if provider == "claude":
        key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
    else:
        return None
    # Strip whitespace/newlines — a trailing "\n" in the key produces an illegal
    # Authorization header that surfaces (misleadingly) as a connection error.
    return key.strip() if key else None


def _canonical_provider(raw: str) -> str:
    p = PROVIDER_ALIASES.get((raw or "").strip().lower())
    if not p:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider '{raw}'. Use one of: claude, chatgpt.",
        )
    return p


app = FastAPI(title="Downstream LLM API")


# --- request / response schemas ---------------------------------------------
class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class GenerateRequest(BaseModel):
    provider: str = Field(..., description="'claude' or 'chatgpt' (aliases ok)")
    # Provide EITHER a single prompt OR a messages list.
    prompt: Optional[str] = Field(None, description="Single user message (e.g. compressed context + question)")
    messages: Optional[List[Message]] = Field(None, description="Chat turns (alternative to prompt)")
    system: Optional[str] = Field(None, description="System instruction")
    model: Optional[str] = Field(None, description="Model id; defaults per provider")
    max_tokens: int = Field(512, gt=0, le=8192)
    temperature: float = Field(0.7, ge=0, le=2)


class GenerateResponse(BaseModel):
    provider: str
    model: str
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: int


def _to_messages(req: GenerateRequest):
    """Normalize prompt/messages into a [{role, content}] list."""
    if req.messages:
        return [{"role": m.role, "content": m.content} for m in req.messages]
    if req.prompt is not None:
        return [{"role": "user", "content": req.prompt}]
    raise HTTPException(status_code=400, detail="Provide either 'prompt' or 'messages'.")


# --- provider callers (SDKs imported lazily so a missing one isn't fatal) ----
def _call_claude(model, system, messages, max_tokens, temperature, api_key):
    try:
        from anthropic import Anthropic
    except ImportError:
        raise HTTPException(status_code=500, detail="anthropic SDK not installed (pip install anthropic).")
    client = Anthropic(api_key=api_key)
    kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature, messages=messages)
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _call_openai(model, system, messages, max_tokens, temperature, api_key):
    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(status_code=500, detail="openai SDK not installed (pip install openai).")
    client = OpenAI(api_key=api_key)
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    resp = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature,
    )
    text = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    return text, (usage.prompt_tokens if usage else None), (usage.completion_tokens if usage else None)


# --- routes -----------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/providers")
def providers():
    """Tell the frontend which providers are usable (have a key) + model options."""
    out = {}
    for prov in ("claude", "openai"):
        out[prov] = {
            "available": _key_for(prov) is not None,
            "default_model": DEFAULT_MODEL[prov],
            "models": SUGGESTED_MODELS[prov],
            "label": "Claude (Anthropic)" if prov == "claude" else "ChatGPT (OpenAI)",
        }
    return out


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    provider = _canonical_provider(req.provider)
    api_key = _key_for(provider)
    if not api_key:
        env = "CLAUDE_API_KEY/ANTHROPIC_API_KEY" if provider == "claude" else "OPENAI_API_KEY"
        raise HTTPException(status_code=400, detail=f"No API key for {provider}. Set {env}.")

    model = req.model or DEFAULT_MODEL[provider]
    messages = _to_messages(req)
    caller = _call_claude if provider == "claude" else _call_openai

    t0 = time.time()
    try:
        text, in_tok, out_tok = caller(
            model, req.system, messages, req.max_tokens, req.temperature, api_key
        )
    except HTTPException:
        raise
    except Exception as e:  # surface provider/SDK errors cleanly
        raise HTTPException(status_code=502, detail=f"{provider} call failed: {e}")

    return GenerateResponse(
        provider=provider, model=model, text=text,
        input_tokens=in_tok, output_tokens=out_tok,
        latency_ms=int((time.time() - t0) * 1000),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
