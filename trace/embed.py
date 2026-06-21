"""
Local embedding provider for the Trace recall/score path.

Loads sentence-transformers all-MiniLM-L6-v2 once (CPU is fine) and exposes a
batched embed() that returns plain Python lists, so it drops straight into
core.Store as the Embedder and works with core.cosine.

Seam: this is the local-CPU path. To get parity with the compress path you could
move embeddings onto Modal later; keep the embed(texts) -> list[list[float]]
signature and nothing downstream changes.
"""

from __future__ import annotations

import threading

from .core import Embedder

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_lock = threading.Lock()


def _get_model():
    """Lazy, thread-safe singleton. The model loads once per process."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Batched, normalized embeddings. Returns one vector (list[float]) per text.

    normalize_embeddings=True means cosine in core.py is a plain dot product, and
    similarity scores land in a clean range for the recall threshold.
    """
    if not texts:
        return []
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]


def get_embedder() -> Embedder:
    """Return the bound embed callable to hand to core.Store(embed)."""
    return embed
