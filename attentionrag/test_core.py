"""
Local, GPU-free tests for the AttentionRAG core (selection logic + pipeline).

A deterministic MockBackend stands in for the real HF model: it assigns high
"attention" to tokens matching a needle set and emits the anchor token 'none'
for chunks that contain no needle (mirroring the paper's relevance gate). This
lets us verify chunking orchestration, Eq. 3 top-k -> sentence selection, and
none-gating without loading any model.

Run:
    python -m attentionrag.test_core      # from repo root
    python attentionrag/test_core.py
    pytest attentionrag/test_core.py
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple

from .core import (
    AttentionRAG,
    FocusResultData,
    select_sentences,
    split_sentence_spans,
)
from .prompts import build_anchor_segments

_TOK = re.compile(r"\S+")


def _tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOK.finditer(text)]


class MockBackend:
    """Deterministic stand-in for the HF compression/generation model."""

    def __init__(self, needles: Set[str], hint: str = "The answer is"):
        self.needles = {n.lower() for n in needles}
        self.hint = hint

    def generate_hint_prefix(self, question: str) -> str:
        # yes/no questions -> 'None' (empty-prefix path), else canned prefix
        if question.strip().lower().startswith(("is ", "are ", "do ", "does ")):
            return "None"
        return self.hint

    def chunk_context(self, context: str, chunk_size: int) -> List[str]:
        # whitespace "token" chunking (stand-in for real tokenizer chunking)
        words = context.split()
        return [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ] or [""]

    def _needle_hit(self, tok: str) -> bool:
        return tok.lower().strip(".,!?;:\"'") in self.needles

    def focus_attention(self, chunk_text, question, prefix) -> FocusResultData:
        toks = _tokenize_with_offsets(chunk_text)
        offsets = [(s, e) for _t, s, e in toks]
        has_needle = any(self._needle_hit(t) for t, _s, _e in toks)
        if not has_needle:
            # irrelevant chunk -> anchor token is 'none' (relevance gate)
            return FocusResultData(anchor="none", attention=[], token_offsets=offsets)
        # needle tokens get high attention; everything else small & positional
        attention = [
            10.0 if self._needle_hit(t) else 1.0 / (i + 2)
            for i, (t, _s, _e) in enumerate(toks)
        ]
        return FocusResultData(
            anchor="park", attention=attention, token_offsets=offsets
        )

    def generate_answer(self, compressed_context: str, question: str) -> str:
        return f"ANSWER<<{compressed_context}>>"


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_split_sentence_spans_basic():
    text = "Daniel went home. Mary is in the park! Where now? Done."
    spans = split_sentence_spans(text)
    sents = [s for _a, _b, s in spans]
    assert sents == [
        "Daniel went home.",
        "Mary is in the park!",
        "Where now?",
        "Done.",
    ], sents
    # every span's char range must reproduce its sentence text
    for a, b, s in spans:
        assert text[a:b] == s, (text[a:b], s)


def test_split_sentence_spans_newlines():
    text = "First line\nSecond line here\nThird."
    sents = [s for _a, _b, s in split_sentence_spans(text)]
    assert sents == ["First line", "Second line here", "Third."], sents


def test_select_sentences_picks_needle_sentence():
    chunk = "Alpha beta gamma. The keyword apple is here. Delta epsilon zeta."
    toks = _tokenize_with_offsets(chunk)
    offsets = [(s, e) for _t, s, e in toks]
    attention = [10.0 if t.strip(".") == "apple" else 0.1 for t, _s, _e in toks]
    kept, idx = select_sentences(chunk, offsets, attention, top_k=1)
    assert kept == "The keyword apple is here.", kept
    assert idx == [1], idx


def test_select_sentences_topk_spans_multiple_sentences():
    chunk = "Cats sleep. Dogs run fast. Birds fly high."
    toks = _tokenize_with_offsets(chunk)
    offsets = [(s, e) for _t, s, e in toks]
    # boost one token in sentence 0 and one in sentence 2
    attention = []
    for t, _s, _e in toks:
        if t.strip(".") in ("Cats", "high"):
            attention.append(9.0)
        else:
            attention.append(0.1)
    kept, idx = select_sentences(chunk, offsets, attention, top_k=2)
    assert idx == [0, 2], idx
    assert kept == "Cats sleep. Birds fly high.", kept


def test_select_sentences_topk_capped_to_token_count():
    chunk = "Short one."
    toks = _tokenize_with_offsets(chunk)
    offsets = [(s, e) for _t, s, e in toks]
    attention = [1.0 for _ in toks]
    kept, idx = select_sentences(chunk, offsets, attention, top_k=50)
    assert kept == "Short one.", kept
    assert idx == [0]


def test_none_chunk_is_skipped():
    # chunk 0 has the needle, chunk 1 does not
    backend = MockBackend(needles={"park"})
    rag = AttentionRAG(backend, chunk_size=6, top_k=3)
    context = "Daniel is in the park today. " "Totally unrelated filler sentence here."
    comp = rag.compress(context, "Where is Daniel?")
    assert comp.n_chunks == 2, comp.n_chunks
    skipped = [c.skipped for c in comp.chunks]
    assert skipped == [False, True], skipped
    assert "park" in comp.compressed_context
    assert "filler" not in comp.compressed_context


def test_pipeline_reduces_and_keeps_answer_sentence():
    backend = MockBackend(needles={"park"})
    rag = AttentionRAG(backend, chunk_size=100, top_k=1)
    context = (
        "The weather was cold that morning. "
        "Daniel is in the park near the old fountain. "
        "Many tourists visit the city each summer."
    )
    answer, comp = rag.run(context, "Where is Daniel?")
    assert comp.compressed_context == "Daniel is in the park near the old fountain."
    assert len(comp.compressed_context) < len(context)
    assert answer.startswith("ANSWER<<")
    assert "park" in answer


def test_empty_prefix_path_for_yes_no():
    backend = MockBackend(needles={"park"})
    rag = AttentionRAG(backend, chunk_size=100, top_k=1)
    comp = rag.compress("Daniel is in the park.", "Is Daniel home?")
    assert comp.is_empty_prefix is True, comp.hint_prefix
    assert comp.effective_prefix == ""
    # still compresses normally
    assert "park" in comp.compressed_context


def test_fixed_prefix_mode_overrides():
    backend = MockBackend(needles={"park"})
    rag = AttentionRAG(backend, chunk_size=100, top_k=1, use_fixed_prefix=True)
    comp = rag.compress("Daniel is in the park.", "Where is Daniel?")
    assert comp.is_empty_prefix is False
    assert comp.effective_prefix.startswith("Please output the most relevant keyword")


def test_anchor_segments_reconstruct_prompt():
    pre, ctx, post = build_anchor_segments("Daniel is here.", "Where is Daniel?", "Daniel is in the")
    full = pre + ctx + post
    assert "Context: Daniel is here." in full
    assert "Question: Where is Daniel?" in full
    assert full.rstrip().endswith("Answer: Daniel is in the")
    # empty prefix -> nothing appended after "Answer:"
    pre2, ctx2, post2 = build_anchor_segments("X.", "Is it raining?", "")
    assert (pre2 + ctx2 + post2).rstrip().endswith("Answer:")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} core tests passed.")


if __name__ == "__main__":
    _run_all()
