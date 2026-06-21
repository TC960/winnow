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


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} token_merge tests passed.")


if __name__ == "__main__":
    _run_all()
