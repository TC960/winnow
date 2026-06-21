"""
AttentionRAG core (arXiv:2503.10720) -- backend-agnostic orchestration.

This module contains the parts of AttentionRAG that have NO dependency on torch
or any model: sentence segmentation, the top-k token -> sentence selection rule
(Eq. 3), the "none"-gating, and the end-to-end pipeline (Algorithm 1). All
model-specific work (tokenization, attention extraction = Eq. 1/2, generation)
is delegated to a `Backend` that satisfies the small protocol below. This split
lets us unit-test the selection logic deterministically without a GPU.

Mapping to the paper:

  Step 1 (sec 3.1)  generate answer hint prefix p          -> Backend.generate_hint_prefix
  Step 2 (sec 3.2)  chunk context into n = ceil(|C|/m)     -> Backend.chunk_context
  per chunk:
     - predict anchor token a_j; gate on 'none'            -> Backend.focus_attention
     - A_j = sum_l Attention_l(c_j, a_j)   (Eq. 2)         -> Backend.focus_attention
     - keep sentences holding the top-k tokens (Eq. 3)     -> select_sentences (HERE)
  Step 5            answer from compressed context C'       -> Backend.generate_answer
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

from .prompts import FIXED_HINT_PREFIX


# --------------------------------------------------------------------------- #
# Sentence segmentation with character spans                                  #
# --------------------------------------------------------------------------- #
def split_sentence_spans(text: str) -> List[Tuple[int, int, str]]:
    """Split `text` into sentences, returning (start_char, end_char, sentence).

    Boundaries are '.', '!', '?' (with trailing closing punctuation absorbed)
    and newlines. Whitespace between sentences is excluded from every span.
    """
    spans: List[Tuple[int, int, str]] = []
    n = len(text)
    start = 0
    i = 0

    def _flush(end: int) -> int:
        seg = text[start:end]
        if seg.strip():
            # tighten the span to the stripped sentence's real char range
            lead = len(seg) - len(seg.lstrip())
            trail = len(seg) - len(seg.rstrip())
            spans.append((start + lead, end - trail, seg.strip()))
        # skip following whitespace
        j = end
        while j < n and text[j].isspace():
            j += 1
        return j

    while i < n:
        ch = text[i]
        if ch in ".!?":
            j = i + 1
            while j < n and text[j] in ".!?\"')]}":
                j += 1
            start = _flush(j)
            i = start
            continue
        if ch == "\n":
            start = _flush(i)
            i = start
            continue
        i += 1

    if start < n and text[start:n].strip():
        seg = text[start:n]
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        spans.append((start + lead, n - trail, seg.strip()))
    return spans


def _sentence_index_for_char(spans: Sequence[Tuple[int, int, str]], pos: int) -> int:
    """Index of the sentence span containing char `pos` (or the last one starting
    at/before it, so tokens landing in inter-sentence whitespace attach to the
    preceding sentence). Returns 0 if `spans` is empty-safe-guarded by caller."""
    chosen = 0
    for idx, (s, e, _t) in enumerate(spans):
        if s <= pos < e:
            return idx
        if s <= pos:
            chosen = idx
    return chosen


# --------------------------------------------------------------------------- #
# Eq. 3: top-k token -> sentence selection                                    #
# --------------------------------------------------------------------------- #
def select_sentences(
    chunk_text: str,
    token_offsets: Sequence[Tuple[int, int]],
    attention: Sequence[float],
    top_k: int,
) -> Tuple[str, List[int]]:
    """Compress one chunk per Eq. 3.

    c'_j = Concat({ s | t_r in Top-k(A_j) and t_r in s })

    Args:
        chunk_text:    the decoded context chunk.
        token_offsets: (start_char, end_char) for each context token, aligned 1:1
                       with `attention`.
        attention:     A_j -- per-context-token attention feature (summed over
                       layers) for the focal/anchor token.
        top_k:         number of highest-attention tokens to keep.

    Returns (compressed_text, sorted_sentence_indices_kept).
    """
    if not attention or not token_offsets:
        return "", []
    assert len(attention) == len(token_offsets), (
        f"attention/offsets length mismatch: {len(attention)} != {len(token_offsets)}"
    )

    spans = split_sentence_spans(chunk_text)
    if not spans:
        return "", []

    k = min(top_k, len(attention))
    # indices of the k largest attention values (stable: ties broken by position)
    top_idx = sorted(range(len(attention)), key=lambda i: (-attention[i], i))[:k]

    kept: set[int] = set()
    for ti in top_idx:
        char_start = token_offsets[ti][0]
        kept.add(_sentence_index_for_char(spans, char_start))

    ordered = sorted(kept)
    compressed = " ".join(spans[j][2] for j in ordered)
    return compressed, ordered


# --------------------------------------------------------------------------- #
# Backend protocol                                                            #
# --------------------------------------------------------------------------- #
class FocusResult(Protocol):
    anchor: Optional[str]
    attention: Sequence[float]
    token_offsets: Sequence[Tuple[int, int]]


class Backend(Protocol):
    """Everything model-specific AttentionRAG needs."""

    def generate_hint_prefix(self, question: str) -> str:
        """Run the B.1 prompt -> incomplete-answer template (or 'None')."""

    def chunk_context(self, context: str, chunk_size: int) -> List[str]:
        """Split context into chunks of ~`chunk_size` tokens; return decoded text."""

    def focus_attention(
        self, chunk_text: str, question: str, prefix: str
    ) -> "FocusResultData":
        """Predict the anchor token a_j for this chunk, and if it is not 'none'
        return its per-context-token attention summed over all layers (Eq. 2),
        plus the char offsets of each context token."""

    def generate_answer(self, compressed_context: str, question: str) -> str:
        """Generation model produces the final answer from C'."""


@dataclass
class FocusResultData:
    anchor: Optional[str]
    attention: List[float] = field(default_factory=list)
    token_offsets: List[Tuple[int, int]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Result containers                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class ChunkResult:
    index: int
    anchor: Optional[str]
    skipped: bool
    kept_text: str
    kept_sentence_indices: List[int]
    n_context_tokens: int


@dataclass
class CompressionResult:
    hint_prefix: str
    effective_prefix: str
    is_empty_prefix: bool
    compressed_context: str
    chunks: List[ChunkResult]
    n_chunks: int

    @property
    def n_kept_chunks(self) -> int:
        return sum(1 for c in self.chunks if not c.skipped)


def _is_none_anchor(anchor: Optional[str]) -> bool:
    if anchor is None:
        return True
    a = anchor.strip().strip(".,;:!?\"'").lower()
    return a == "" or a.startswith("none")


# --------------------------------------------------------------------------- #
# Algorithm 1: AttentionRAG pipeline                                          #
# --------------------------------------------------------------------------- #
class AttentionRAG:
    def __init__(
        self,
        backend: Backend,
        chunk_size: int = 300,
        top_k: int = 12,
        use_fixed_prefix: bool = False,
    ):
        self.backend = backend
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.use_fixed_prefix = use_fixed_prefix

    def compress(
        self,
        context: str,
        question: str,
        hint_prefix: Optional[str] = None,
    ) -> CompressionResult:
        # --- Step 1: answer hint prefix -------------------------------------
        if self.use_fixed_prefix:
            raw_prefix = FIXED_HINT_PREFIX
        elif hint_prefix is not None:
            raw_prefix = hint_prefix
        else:
            raw_prefix = self.backend.generate_hint_prefix(question)

        is_empty = raw_prefix.strip().lower() in ("none", "")
        # Empty prefix (yes/no questions): the anchor is the first answer token
        # itself, so no prefix text is appended after "Answer:".
        effective_prefix = "" if is_empty else raw_prefix.strip()

        # --- Step 2: chunking ----------------------------------------------
        chunks = self.backend.chunk_context(context, self.chunk_size)

        # --- per-chunk compression -----------------------------------------
        chunk_results: List[ChunkResult] = []
        kept_blocks: List[str] = []
        for j, chunk_text in enumerate(chunks):
            focus = self.backend.focus_attention(chunk_text, question, effective_prefix)

            if _is_none_anchor(focus.anchor):
                chunk_results.append(
                    ChunkResult(
                        index=j,
                        anchor=focus.anchor,
                        skipped=True,
                        kept_text="",
                        kept_sentence_indices=[],
                        n_context_tokens=len(focus.token_offsets),
                    )
                )
                continue

            kept_text, kept_idx = select_sentences(
                chunk_text, focus.token_offsets, focus.attention, self.top_k
            )
            if kept_text:
                kept_blocks.append(kept_text)
            chunk_results.append(
                ChunkResult(
                    index=j,
                    anchor=focus.anchor,
                    skipped=False,
                    kept_text=kept_text,
                    kept_sentence_indices=kept_idx,
                    n_context_tokens=len(focus.token_offsets),
                )
            )

        compressed_context = "\n".join(kept_blocks)
        return CompressionResult(
            hint_prefix=raw_prefix.strip(),
            effective_prefix=effective_prefix,
            is_empty_prefix=is_empty,
            compressed_context=compressed_context,
            chunks=chunk_results,
            n_chunks=len(chunks),
        )

    def answer(self, compressed_context: str, question: str) -> str:
        return self.backend.generate_answer(compressed_context, question)

    def run(
        self, context: str, question: str, hint_prefix: Optional[str] = None
    ) -> Tuple[str, CompressionResult]:
        """Full pipeline: compress then answer. Returns (answer, compression)."""
        comp = self.compress(context, question, hint_prefix=hint_prefix)
        ans = self.answer(comp.compressed_context, question)
        return ans, comp
