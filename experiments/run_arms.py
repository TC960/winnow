"""
Robust driver for the 4-arm eval: calls the DEPLOYED Modal class (via
modal.Cls.from_name) instead of `modal run`, which has flaky log-stream
heartbeats. Deploy first:

    modal deploy eval_modal.py
    python run_arms.py
"""

import json
import sys

import modal

from test_data import INSTRUCTION, INTENT, NARRATION
from eval_modal import (
    EMBEDDER_NAME, LONGLLMLINGUA_MODEL, MODEL_NAME, RERANKER_NAME,
)

RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
REDUNDANCY_THRESHOLD = 0.80

Compressor = modal.Cls.from_name("winnow-eval", "Compressor")
comp = Compressor()

plan = [
    ("L2",       "llmlingua2",    False),
    ("L2+MMR",   "llmlingua2",    True),
    ("Long",     "longllmlingua", False),
    ("Long+MMR", "longllmlingua", True),
]

arms = {}
for name, kind, use_mmr in plan:
    print(f"Running {name} (kind={kind}, mmr={use_mmr}) ...", flush=True)
    arms[name] = comp.compress_eval.remote(
        NARRATION, INTENT, INSTRUCTION,
        kind=kind, rate=RATE, use_mmr=use_mmr,
        redundancy_threshold=REDUNDANCY_THRESHOLD,
    )
    a = arms[name]
    print(f"  -> {a['context_origin_tokens']}->{a['context_compressed_tokens']} "
          f"tok (kept {a['retention']})", flush=True)

result = {
    "config": {
        "rate_kept": RATE,
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
        "chunk_size": 3,
        "model": MODEL_NAME,
        "reranker": RERANKER_NAME,
        "embedder": EMBEDDER_NAME,
        "longllmlingua_model": LONGLLMLINGUA_MODEL,
    },
    "intent": INTENT,
    "instruction": INSTRUCTION,
    "baseline_key": "L2",
    "arms": arms,
}
with open("eval_outputs.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nWrote eval_outputs.json", flush=True)
