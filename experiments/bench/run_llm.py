"""
Drive the deployed winnow-bench-llm worker over all LLM/token arms and write
llm_answers.json in the SCHEMA.md shape.

Arms produced here:
  vanilla_llm  -- full context, fp16 KV
  llm_tq       -- full context, TurboQuant 4-bit KV
  lingua / union / intersection  -- token-compressed prompt, read by vanilla fp16 Qwen

    cd experiments/bench && python3 run_llm.py
"""

import concurrent.futures as cf
import json
import os
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))

INSTRUCTION = "Answer the question using ONLY the provided context. Answer in as few words as possible. If the answer is not present, respond with exactly: UNKNOWN."


def make_prompt(context, question):
    return f"{INSTRUCTION}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


def main():
    BenchLLM = modal.Cls.from_name("winnow-bench-llm", "BenchLLM")
    obj = BenchLLM()

    data = json.load(open(os.path.join(HERE, "data.json")))
    compressed = json.load(open(os.path.join(HERE, "compressed.json")))
    q_by_id = {ex["id"]: ex["question"] for ex in data["examples"]}

    # Build the full list of (id, arm, kwargs) jobs.
    jobs = []
    for ex in data["examples"]:
        prompt = make_prompt(ex["context"], ex["question"])
        jobs.append((ex["id"], "vanilla_llm", dict(prompt=prompt, mode="vanilla")))
        jobs.append((ex["id"], "llm_tq", dict(prompt=prompt, mode="tq", bit_width=4)))

    for ex in compressed["examples"]:
        ex_id = ex["id"]
        question = q_by_id.get(ex_id)
        if question is None:
            print(f"[warn] no question for {ex_id}, skipping token arms", flush=True)
            continue
        for arm in ("lingua", "union", "intersection"):
            arm_info = ex["arms"].get(arm)
            if not arm_info or not arm_info.get("prompt"):
                print(f"[warn] {ex_id}/{arm} missing prompt, skipping", flush=True)
                continue
            prompt = make_prompt(arm_info["prompt"], question)
            jobs.append((ex_id, arm, dict(prompt=prompt, mode="vanilla")))

    print(f"Total jobs: {len(jobs)}", flush=True)

    answers = []
    failures = []

    def run_one(job):
        ex_id, arm, kwargs = job
        out = obj.answer.remote(**kwargs)
        return ex_id, arm, out

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(run_one, j): j for j in jobs}
        done = 0
        for fut in cf.as_completed(futs):
            ex_id, arm, _ = futs[fut]
            try:
                ex_id, arm, out = fut.result()
                answers.append({
                    "id": ex_id,
                    "arm": arm,
                    "answer": out["text"],
                    "extra": {
                        "kv_compression_x": out["kv_compression_x"],
                        "eff_bits": out["eff_bits"],
                        "input_tokens": out["input_tokens"],
                        "output_tokens": out["output_tokens"],
                    },
                })
            except Exception as e:  # noqa: BLE001
                failures.append((ex_id, arm, repr(e)))
                print(f"[FAIL] {ex_id}/{arm}: {e!r}", flush=True)
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)

    doc = {
        "arm_group": "llm",
        "meta": {
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "arms": ["vanilla_llm", "llm_tq", "lingua", "union", "intersection"],
            "max_new_tokens": 48,
            "n_jobs": len(jobs),
            "n_success": len(answers),
            "n_failures": len(failures),
            "failures": failures,
        },
        "answers": answers,
    }
    out_path = os.path.join(HERE, "llm_answers.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nWrote {out_path}: {len(answers)} answers, {len(failures)} failures", flush=True)


if __name__ == "__main__":
    main()
