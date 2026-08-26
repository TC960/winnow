"""Local tests for token_merge (no GPU). Run: python test_token_merge.py"""

from token_merge import (
    attnrag_mask,
    merge_compress,
    normalize_labels,
    reconstruct_word_spans,
    splice_kept,
)


def test_normalize_labels_list_and_string():
    assert normalize_labels([["a", 1], ["b", 0]]) == [("a", 1), ("b", 0)]
    assert normalize_labels("a,1 b,0") == [("a", 1), ("b", 0)]
    assert normalize_labels(None) == []


def test_reconstruct_handles_duplicates_in_order():
    text = "the cat and the dog"
    pairs = [("the", 0), ("cat", 1), ("and", 0), ("the", 1), ("dog", 1)]
    spans = reconstruct_word_spans(text, pairs)
    # the two "the"s map to successive occurrences (0 and 12), not the same one
    assert [(s, e) for _w, _l, s, e in spans] == [
        (0, 3), (4, 7), (8, 11), (12, 15), (16, 19)
    ], spans


def test_attnrag_mask_overlap():
    text = "the cat and the dog"
    pairs = [("the", 0), ("cat", 1), ("and", 0), ("the", 1), ("dog", 1)]
    spans = reconstruct_word_spans(text, pairs)
    mask = attnrag_mask(spans, [(4, 19)])  # covers "cat and the dog"
    assert mask == [False, True, True, True, True], mask


def test_intersection():
    text = "the cat and the dog"
    labels = [["the", 0], ["cat", 1], ["and", 0], ["the", 1], ["dog", 1]]
    out = merge_compress(text, labels, [(4, 19)], mode="intersection")
    assert out["mask_llmlingua"] == [0, 1, 0, 1, 1]
    assert out["mask_attentionrag"] == [0, 1, 1, 1, 1]
    assert [x[1] for x in out["word_labels"]] == [0, 1, 0, 1, 1]
    assert out["compressed_prompt"] == "cat the dog", out["compressed_prompt"]
    assert out["used_llmlingua_fallback"] is False


def test_union():
    text = "the cat and the dog"
    labels = [["the", 0], ["cat", 1], ["and", 0], ["the", 1], ["dog", 1]]
    out = merge_compress(text, labels, [(4, 19)], mode="union")
    assert [x[1] for x in out["word_labels"]] == [0, 1, 1, 1, 1]
    assert out["compressed_prompt"] == "cat and the dog", out["compressed_prompt"]


def test_empty_attnrag_falls_back_to_llmlingua():
    text = "the cat and the dog"
    labels = [["the", 0], ["cat", 1], ["and", 0], ["the", 1], ["dog", 1]]
    # intersection with empty AttentionRAG would normally wipe everything;
    # fallback keeps the LLMLingua mask instead.
    out = merge_compress(text, labels, [], mode="intersection")
    assert out["used_llmlingua_fallback"] is True
    assert [x[1] for x in out["word_labels"]] == [0, 1, 0, 1, 1]
    assert out["compressed_prompt"] == "cat the dog"


def test_punctuation_and_spacing_preserved_in_runs():
    text = "Daniel, who left, is in the park."
    # canonical words include punctuation chunks as LLMLingua emits them
    labels = [["Daniel,", 1], ["who", 0], ["left,", 0], ["is", 1], ["in", 1],
              ["the", 1], ["park.", 1]]
    out = merge_compress(text, labels, [(0, len(text))], mode="intersection")
    # "Daniel," then a gap (who left dropped) then the contiguous run
    assert out["compressed_prompt"] == "Daniel, is in the park.", out["compressed_prompt"]


def test_realistic_sentence_intersection():
    text = "The weather was cold. Daniel is in the park near the fountain."
    # LLMLingua keeps salient words across both sentences
    labels = [["The", 0], ["weather", 1], ["was", 0], ["cold.", 1],
              ["Daniel", 1], ["is", 0], ["in", 0], ["the", 0], ["park", 1],
              ["near", 0], ["the", 0], ["fountain.", 1]]
    # AttentionRAG keeps only the 2nd sentence (span of "Daniel ... fountain.")
    s = text.index("Daniel")
    out = merge_compress(text, labels, [(s, len(text))], mode="intersection")
    # 1st-sentence keeps (weather, cold.) are dropped (not in AttentionRAG span);
    # 2nd-sentence keeps (Daniel, park, fountain.) survive.
    assert out["compressed_prompt"] == "Daniel park fountain.", out["compressed_prompt"]
    # union keeps weather/cold. too
    out_u = merge_compress(text, labels, [(s, len(text))], mode="union")
    assert "weather" in out_u["compressed_prompt"] and "Daniel" in out_u["compressed_prompt"]


def test_unmatched_word_does_not_rewind_cursor():
    """Proper fix for the cursor-rewind bug: when a labeled word can't be
    found at-or-after the cursor (LLMLingua tokenization quirk), it gets a
    zero-width sentinel span at cursor and the cursor stays put. Previously
    a global find() fallback could return an index BEFORE the cursor,
    rewinding it and causing every subsequent word to re-match earlier text
    - a cascade of overlapping spans."""
    text = "alpha beta gamma delta epsilon"
    pairs = [("alpha", 1), ("beta", 1), ("gamma", 1),
             ("PHANTOM", 1),  # not in original
             ("delta", 1), ("epsilon", 1)]
    spans = reconstruct_word_spans(text, pairs)

    # Monotonic non-decreasing starts (the core invariant).
    starts = [s for _w, _l, s, _e in spans]
    assert starts == sorted(starts), starts

    # PHANTOM gets a zero-width span at cursor (right after "gamma").
    gamma_end = text.index("gamma") + len("gamma")
    assert spans[3] == ("PHANTOM", 1, gamma_end, gamma_end), spans[3]

    # delta/epsilon are still located AFTER the phantom, not from index 0.
    assert spans[4][2] == text.index("delta"), spans[4]
    assert spans[5][2] == text.index("epsilon"), spans[5]


def test_union_output_bounded_under_misalignment():
    """Regression for the union token explosion: a misaligned canonical word
    used to rewind the cursor, producing overlapping spans whose gap-fill
    blew the splice output up many-fold. Now spans stay monotonic and the
    splice joins only kept-word substrings - output is bounded by the kept
    content regardless of how many phantom tokens LLMLingua emits."""
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    pairs = [["alpha", 1], ["beta", 1], ["gamma", 1],
             ["PHANTOM_A", 1],  # not in original
             ["delta", 1], ["epsilon", 1],
             ["PHANTOM_B", 1],  # not in original
             ["zeta", 1], ["eta", 1], ["theta", 1],
             ["iota", 1], ["kappa", 1]]
    # AttentionRAG keeps the whole text -> union keeps everything.
    out = merge_compress(text, pairs, [(0, len(text))], mode="union")

    # Bounded: each kept word contributes at most its own length + one space.
    max_len = sum(len(p[0]) + 1 for p in pairs)
    assert len(out["compressed_prompt"]) <= max_len, (
        f"output {len(out['compressed_prompt'])} > bound {max_len}: "
        f"{out['compressed_prompt']!r}"
    )

    # All canonical tokens (real and phantom) survive in order.
    assert out["n_words"] == len(pairs)
    assert out["n_kept"] == len(pairs)
    assert "alpha" in out["compressed_prompt"]
    assert "kappa" in out["compressed_prompt"]
    assert "PHANTOM_A" in out["compressed_prompt"]
    assert "PHANTOM_B" in out["compressed_prompt"]
    # No duplicated content (the old bug would re-insert earlier text).
    assert out["compressed_prompt"].count("alpha") == 1
    assert out["compressed_prompt"].count("delta") == 1


def test_n_unlocated_words_counter():
    """The n_unlocated_words counter exposes how often the zero-width sentinel
    fallback in reconstruct_word_spans fires. Clean input -> 0. Phantom labels
    -> exact count. This is the falsifiability hook for the 'rare' claim."""
    text = "alpha beta gamma"
    # Clean: every word findable in order.
    clean = merge_compress(
        text, [["alpha", 1], ["beta", 1], ["gamma", 1]],
        [(0, len(text))], mode="union",
    )
    assert clean["n_unlocated_words"] == 0, clean["n_unlocated_words"]

    # Two phantoms that don't appear in `text`.
    dirty = merge_compress(
        text,
        [["alpha", 1], ["PHANTOM_A", 1], ["beta", 1],
         ["PHANTOM_B", 1], ["gamma", 1]],
        [(0, len(text))], mode="union",
    )
    assert dirty["n_unlocated_words"] == 2, dirty["n_unlocated_words"]
    # The bare empty-string entry is NOT counted (it's an explicit empty label,
    # not a tokenization-drift failure).
    with_empty = merge_compress(
        text, [["alpha", 1], ["", 1], ["beta", 1], ["gamma", 1]],
        [(0, len(text))], mode="union",
    )
    assert with_empty["n_unlocated_words"] == 0, with_empty["n_unlocated_words"]


def test_normalize_labels_real_llmlingua_pipe_format():
    """LLMLingua-2's `fn_labeled_original_prompt` real format is
    '<word> <label>' entries delimited by '\\t\\t|\\t\\t'. Cover it
    end-to-end including a multi-word entry and a punctuation-glued token."""
    s = (
        "Daniel 1\t\t|\t\ttravelled 1\t\t|\t\tto 0\t\t|\t\t"
        "the 1\t\t|\t\tpark. 1\t\t|\t\tNew York 1"
    )
    pairs = normalize_labels(s)
    assert pairs == [
        ("Daniel", 1), ("travelled", 1), ("to", 0),
        ("the", 1), ("park.", 1), ("New York", 1),
    ], pairs


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} token_merge tests passed.")


if __name__ == "__main__":
    _run_all()
