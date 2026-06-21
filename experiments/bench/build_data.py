"""Build a small shared QA benchmark -> data.json (per SCHEMA.md)."""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data.json")
MAX_WORDS = 2500


def truncate(text):
    words = (text or "").split()
    return " ".join(words[:MAX_WORDS])


def _longbench_jsonl_dir():
    """LongBench ships one archive of per-task jsonl files; find the extracted dir."""
    base = os.path.expanduser("~/.cache/huggingface/datasets/downloads/extracted")
    for root, _dirs, files in os.walk(base):
        if "multifieldqa_en.jsonl" in files and "hotpotqa.jsonl" in files:
            return root
    return None


def try_longbench():
    """Primary: load LongBench. datasets 2.10 can't reopen its local-fs cache
    (NotImplementedError), so read the extracted per-task jsonl directly. If not
    yet downloaded, trigger the download via load_dataset, then read jsonl."""
    from datasets import load_dataset
    tasks = ["multifieldqa_en", "hotpotqa"]
    jdir = _longbench_jsonl_dir()
    if jdir is None:
        for task in tasks:
            try:
                load_dataset("THUDM/LongBench", task, split="test")
            except Exception:
                pass  # download succeeds; only the cache-reopen step raises
        jdir = _longbench_jsonl_dir()
    if jdir is None:
        raise RuntimeError("LongBench jsonl files not found after download")

    examples = []
    for task in tasks:
        path = os.path.join(jdir, f"{task}.jsonl")
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                ex = json.loads(line)
                examples.append({
                    "id": f"{task}-{i}",
                    "task": task,
                    "context": truncate(ex["context"]),
                    "question": ex["input"],
                    "answers": list(ex["answers"]),
                })
    cfg = {"source": "LongBench", "tasks": tasks, "n_per_task": 10,
           "max_context_words": MAX_WORDS}
    return cfg, examples


def try_hotpot():
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    examples = []
    for i, ex in enumerate(ds):
        if i >= 15:
            break
        sents = ex["context"]["sentences"]  # list of list of sentences
        ctx = " ".join(s for para in sents for s in para)
        examples.append({
            "id": f"hotpotqa-{i}",
            "task": "hotpotqa",
            "context": truncate(ctx),
            "question": ex["question"],
            "answers": [ex["answer"]],
        })
    cfg = {"source": "hotpot_qa/distractor", "tasks": ["hotpotqa"],
           "n_per_task": 15, "max_context_words": MAX_WORDS}
    return cfg, examples


def try_squad():
    from datasets import load_dataset
    ds = load_dataset("squad", split="validation")
    examples = []
    for i, ex in enumerate(ds):
        if i >= 20:
            break
        examples.append({
            "id": f"squad-{i}",
            "task": "squad",
            "context": truncate(ex["context"]),
            "question": ex["question"],
            "answers": list(ex["answers"]["text"]),
        })
    cfg = {"source": "squad", "tasks": ["squad"], "n_per_task": 20,
           "max_context_words": MAX_WORDS}
    return cfg, examples


def main():
    cfg = examples = None
    for fn in (try_longbench, try_hotpot, try_squad):
        try:
            print(f"trying {fn.__name__} ...", flush=True)
            cfg, examples = fn()
            if examples:
                break
        except Exception as e:
            print(f"  {fn.__name__} failed: {type(e).__name__}: {e}", flush=True)
            cfg = examples = None

    if not examples:
        raise SystemExit("All dataset sources failed.")

    with open(OUT, "w") as f:
        json.dump({"config": cfg, "examples": examples}, f, ensure_ascii=False, indent=2)

    # task breakdown
    breakdown = {}
    for ex in examples:
        breakdown[ex["task"]] = breakdown.get(ex["task"], 0) + 1
    print(f"\nWrote {OUT}")
    print(f"config: {cfg}")
    print(f"total examples: {len(examples)}")
    print(f"task breakdown: {breakdown}")
    print("\nSamples:")
    for ex in examples[:2] + examples[-1:]:
        print(f"  [{ex['id']}] Q: {ex['question'][:90]!r}")
        print(f"           A: {ex['answers']}  (ctx words={len(ex['context'].split())})")


if __name__ == "__main__":
    main()
