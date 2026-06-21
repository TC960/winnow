"""AttentionRAG -- attention-guided context pruning for RAG (arXiv:2503.10720)."""

from .core import (
    AttentionRAG,
    ChunkResult,
    CompressionResult,
    FocusResultData,
    select_sentences,
    split_sentence_spans,
)

__all__ = [
    "AttentionRAG",
    "ChunkResult",
    "CompressionResult",
    "FocusResultData",
    "select_sentences",
    "split_sentence_spans",
]
