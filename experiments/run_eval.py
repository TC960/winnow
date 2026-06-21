"""
Driver: runs (LLMLingua-2 + reranker) WITH and WITHOUT sentence dedup, on both
tasks (sparse ramble + medium-density multidoc), at two retention levels.
Calls the DEPLOYED Modal class (robust vs `modal run` log-stream flakiness).

    modal deploy eval_modal.py
    python run_eval.py
"""

import json

import modal

import eval_sets
from eval_modal import EMBEDDER_NAME, MODEL_NAME, RERANKER_NAME

RATES = [0.75, 0.5]          # 75% retention (their target) + 50% (where dedup can bite)
REDUNDANCY_THRESHOLD = 0.80

Compressor = modal.Cls.from_name("winnow-eval", "Compressor")
comp = Compressor()

out_tasks = {}
for tname, t in eval_sets.TASKS.items():
    arms = {}
    for rate in RATES:
        for armname, use_mmr in [("L2", False), ("L2+MMR", True)]:
            key = f"{rate}|{armname}"
            kwargs = dict(
                intent=t["intent"], instruction=t["instruction"], rate=rate,
                use_mmr=use_mmr, redundancy_threshold=REDUNDANCY_THRESHOLD,
                top_k=t["top_k"],
            )
            if t["kind_input"] == "documents":
                kwargs["documents"] = t["documents"]
            else:
                kwargs["text"] = t["text"]
            print(f"{tname:9s} {key:12s} ...", flush=True)
            a = comp.compress_eval.remote(**kwargs)
            arms[key] = a
            kept = a.get("kept_indices")
            print(f"    -> {a['context_origin_tokens']}->{a['context_compressed_tokens']} "
                  f"tok (kept {a['retention']}) | passages kept idx={kept}", flush=True)
    out_tasks[tname] = {
        "intent": t["intent"],
        "instruction": t["instruction"],
        "distractor_indices": t["distractor_indices"],
        "arms": arms,
    }

results = {
    "config": {
        "rates": RATES,
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
        "model": MODEL_NAME,
        "reranker": RERANKER_NAME,
        "embedder": EMBEDDER_NAME,
    },
    "tasks": out_tasks,
}
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nWrote eval_results.json", flush=True)
