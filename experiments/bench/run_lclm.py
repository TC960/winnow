"""Driver for the two LCLM model-space arms (vanilla_lclm, lclm_tq).

Reads data.json, calls generate_qa on both deployed Modal classes
(lclm-tq-timing / LCLMVanilla & LCLMTurboQuant), and writes lclm_answers.json
in the SCHEMA.md *_answers.json shape. Robust: per-call try/except, continues on
failure, logs failed (id, arm) pairs.

    cd experiments/bench && python3 run_lclm.py
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
APP = "lclm-tq-timing"
MAX_NEW_TOKENS = 48
BIT_WIDTH = 4
MAX_CONCURRENCY = 3           # per class; keep low (10-GPU shared limit)
TRUNCATE_WORDS = 2000         # fallback if a context errors on length


def load_examples():
    with open(os.path.join(HERE, "data.json")) as f:
        data = json.load(f)
    return data["examples"]


def call_arm(cls, arm, ex, truncate=False):
    """One generate_qa call. Returns an answer record or raises."""
    context = ex["context"]
    if truncate:
        context = " ".join(context.split()[:TRUNCATE_WORDS])
    kwargs = {"max_new_tokens": MAX_NEW_TOKENS}
    if arm == "lclm_tq":
        kwargs["bit_width"] = BIT_WIDTH
    out = cls().generate_qa.remote(context, ex["question"], **kwargs)
    return {
        "id": ex["id"],
        "arm": arm,
        "answer": out["text"],
        "extra": {
            "kv_compression_x": out["kv_compression_x"],
            "eff_bits": out["eff_bits"],
            "input_tokens": out["input_context_tokens"],
            "latent_tokens": out["latent_tokens"],
            "decoder_prompt_tokens": out["decoder_prompt_tokens"],
        },
    }


def run_arm(arm, cls_name, examples):
    cls = modal.Cls.from_name(APP, cls_name)
    answers, failed = [], []

    def task(ex):
        try:
            return call_arm(cls, arm, ex), None
        except Exception as e:
            msg = str(e)
            # one retry with truncated context if it looks length-related
            try:
                return call_arm(cls, arm, ex, truncate=True), f"recovered(trunc): {msg[:120]}"
            except Exception as e2:
                return None, f"{ex['id']}/{arm}: {str(e2)[:200]}"

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futs = {pool.submit(task, ex): ex for ex in examples}
        for fut in as_completed(futs):
            rec, note = fut.result()
            ex = futs[fut]
            if rec is not None:
                answers.append(rec)
                if note:
                    print(f"  [warn] {ex['id']}/{arm}: {note}", flush=True)
                else:
                    print(f"  [ok]   {ex['id']}/{arm}", flush=True)
            else:
                failed.append(note)
                print(f"  [FAIL] {note}", flush=True)
    return answers, failed


def main():
    examples = load_examples()
    print(f"Loaded {len(examples)} examples; running 2 arms each.", flush=True)
    t0 = time.time()

    all_answers, all_failed = [], []
    for arm, cls_name in [("vanilla_lclm", "LCLMVanilla"),
                          ("lclm_tq", "LCLMTurboQuant")]:
        print(f"\n=== arm {arm} ({cls_name}) ===", flush=True)
        ans, failed = run_arm(arm, cls_name, examples)
        all_answers.extend(ans)
        all_failed.extend(failed)

    out = {
        "arm_group": "lclm",
        "meta": {
            "app": APP,
            "checkpoint": "latent-context/0.6b-4b-LCLM-16x",
            "max_new_tokens": MAX_NEW_TOKENS,
            "bit_width": BIT_WIDTH,
            "n_examples": len(examples),
            "n_answers": len(all_answers),
            "failed": all_failed,
            "elapsed_s": round(time.time() - t0, 1),
        },
        "answers": all_answers,
    }
    path = os.path.join(HERE, "lclm_answers.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}", flush=True)
    print(f"answers: {len(all_answers)} / {2 * len(examples)} | "
          f"failed: {len(all_failed)} | {out['meta']['elapsed_s']}s", flush=True)
    if all_failed:
        print("FAILED:", file=sys.stderr)
        for f_ in all_failed:
            print("  ", f_, file=sys.stderr)


if __name__ == "__main__":
    main()
