# experiments/ — archived eval & research (not part of the product)

This folder holds the investigation into **semantic sentence-dedup (MMR)** on top
of the core `winnow` compressor. It is kept for reference and is **not** required
to deploy or run the service (see the repo root).

## What we tested
`(LLMLingua-2 + BGE reranker)` with vs. without an MMR sentence-dedup pre-pass,
across two regimes: a sparse spoken **ramble** and a medium-density **multi-doc**
set, scored deterministically by having Claude answer comprehension questions
(SQuAD exact-match + token-F1, temperature 0).

## Conclusion
- **MMR sentence-dedup added nothing** at sane thresholds on realistic text
  (identical tokens and F1 with/without). It only fired — and *over-pruned* — on
  an adversarially redundant synthetic doc. Not shipped in the core pipeline.
- At 50–75% retention, LLMLingua-2 + reranker preserves facts well; the ramble
  scored a perfect 7/7.
- In multi-doc, the real bottleneck was the **reranker** dropping a relevant
  passage while keeping a distractor — not compression and not dedup.

## Files
- `eval_modal.py` — Modal app exposing `compress_eval` (encoder-only) + the MMR pre-pass.
- `eval_sets.py` — the two task fixtures (ramble + multi-doc) with gold QA.
- `run_eval.py` — driver: runs both tasks × {no-dedup, +dedup} × {0.75, 0.5} retention.
- `score_eval.py` — Claude reader + deterministic scoring → `eval_report.md`.
- `run_arms.py` — older single-doc driver (superseded by `run_eval.py`).
- `test_data.py` — the original synthetic stress-test doc.
- `eval_report.md` — the scored comparison report.
- `scratchpad.ipynb` — exploratory notebook.

## Re-running (from this folder)
```bash
cd experiments
../../.venv/bin/modal deploy eval_modal.py     # uses two_stage_compressor.py from repo root
../../.venv/bin/python run_eval.py             # writes eval_results.json
../../.venv/bin/python score_eval.py           # reads ../.env, writes eval_report.md
```
