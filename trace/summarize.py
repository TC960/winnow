"""
SUMMARIZE tier wired to the deployed winnow LLMLingua-2 Modal worker.

SUMMARIZE is just an LLMLingua-2 compress call at a low keep-rate on one turn.
This reuses the already deployed worker (no new model). The entrypoint signature
is the one server.py and warmup.py use:

    modal.Cls.from_name("llmlingua2-xlm", "Compressor")() \\
        .compress.remote(text, rate=rate)

which returns a dict with: compressed_prompt, origin_tokens, compressed_tokens,
rate, ratio.

Caching: a generic LLMLingua-2 summary is cacheable by (content hash, rate)
because it does not depend on the goal. Same content + same rate gives the same
output, so a re-pass with the same config does no new GPU work.
"""

from __future__ import annotations

import hashlib

# Must match the app/class names in llmlingua2_modal.py and server.py.
MODAL_APP_NAME = "llmlingua2-xlm"
MODAL_CLASS_NAME = "Compressor"

# (content_hash, rate) -> (compressed_text, compressed_tokens)
_cache: dict[tuple[str, float], tuple[str, int]] = {}

_compressor = None


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _get_compressor():
    """Lazy handle to the deployed Modal class. Importing modal is deferred so
    importing this module never requires modal to be installed or authenticated."""
    global _compressor
    if _compressor is None:
        import modal

        _compressor = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)()
    return _compressor


def summarize(content: str, rate: float = 0.35) -> tuple[str, int]:
    """Compress one turn via the Modal worker. Returns (text, token_count).

    Cached by (content hash, rate). Used for stage 5 reconstruct and for
    speculative precompute.
    """
    key = (_content_hash(content), rate)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    out = _get_compressor().compress.remote(content, rate=rate)
    result = (out["compressed_prompt"], out["compressed_tokens"])
    _cache[key] = result
    return result
