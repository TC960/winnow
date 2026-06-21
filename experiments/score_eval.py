"""
Scoring half (pure-local). Reads eval_results.json (from run_eval.py), feeds each
arm's compressed CONTEXT to Claude with that task's comprehension questions, and
scores deterministically.

  * Claude @ temperature=0; per-Q SQuAD normalized Exact-Match + token-F1 vs gold
    aliases. Correct if F1 >= 0.5. (The F1/EM math is fully deterministic.)
  * Fault localization per miss: READER_ERROR (fact survived in context but reader
    missed) vs INFO_LOST_IN_COMPRESSION, with MMR_OVERPRUNED when the fact sat in
    an MMR-dropped sentence.
  * Multidoc also reports the reranker's distractor handling (did it drop the
    distractor passages?).

    python score_eval.py
"""

import json
import os
import re
import string

import anthropic

import eval_sets

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
F1_CORRECT = 0.5
FACT_SURVIVED_RECALL = 0.6
HERE = os.path.dirname(os.path.abspath(__file__))


def load_key():
    # .env lives in the repo root; this script may sit in experiments/.
    for env_path in (os.path.join(HERE, ".env"), os.path.join(HERE, "..", ".env")):
        if os.path.exists(env_path):
            for line in open(env_path):
                m = re.match(r"(?:export\s+)?([A-Za-z0-9_]+)\s*=\s*(.*)", line.strip())
                if m:
                    os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
            break
    key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("No CLAUDE_API_KEY / ANTHROPIC_API_KEY found.")
    return key


def normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    same = sum(min(p.count(t), g.count(t)) for t in set(p) if t in g)
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def best_f1(pred, golds):
    return max(token_f1(pred, g) for g in golds)


def exact_match(pred, golds):
    return any(normalize(pred) == normalize(g) for g in golds)


def gold_recall_in_text(golds, text):
    toks = set(normalize(text).split())
    best = 0.0
    for g in golds:
        gt = normalize(g).split()
        if gt:
            best = max(best, sum(1 for t in gt if t in toks) / len(gt))
    return best


def ask_claude(client, instruction, context, question):
    prompt = f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=40, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def fault_reason(golds, arm, context):
    if gold_recall_in_text(golds, context) >= FACT_SURVIVED_RECALL:
        return "READER_ERROR"
    if arm.get("mmr"):
        dropped = " ".join(arm["mmr"].get("dropped_sentences", []))
        if gold_recall_in_text(golds, dropped) >= FACT_SURVIVED_RECALL:
            return "MMR_OVERPRUNED"
    return "INFO_LOST_IN_COMPRESSION"


def score_arm(client, arm, instruction, qa):
    context = arm["compressed_context"]
    rows = []
    for item in qa:
        ans = ask_claude(client, instruction, context, item["q"])
        f1 = best_f1(ans, item["answers"])
        correct = f1 >= F1_CORRECT
        rows.append({
            "q": item["q"], "gold": item["answers"][0], "single": item["single_mention"],
            "answer": ans, "f1": round(f1, 3), "em": exact_match(ans, item["answers"]),
            "correct": correct,
            "fault": "" if correct else fault_reason(item["answers"], arm, context),
        })
    return rows


def summarize(rows):
    n = len(rows)
    return {
        "mean_f1": round(sum(r["f1"] for r in rows) / n, 3),
        "em": sum(r["em"] for r in rows), "correct": sum(r["correct"] for r in rows),
        "total": n,
        "single_correct": sum(1 for r in rows if r["single"] and r["correct"]),
        "single_total": sum(1 for r in rows if r["single"]),
    }


def main():
    client = anthropic.Anthropic(api_key=load_key())
    data = json.load(open(os.path.join(HERE, "eval_results.json")))
    cfg = data["config"]
    L = ["# (LLMLingua-2 + reranker) with vs without sentence dedup\n"]
    L.append(f"- LLMLingua-2: `{cfg['model']}` | reranker: `{cfg['reranker']}` | MMR embedder: `{cfg['embedder']}`")
    L.append(f"- reader: `{CLAUDE_MODEL}` @ temperature=0 | redundancy_threshold={cfg['redundancy_threshold']}\n")

    for tname, task in data["tasks"].items():
        qa = eval_sets.TASKS[tname]["qa"]
        instruction = task["instruction"]
        distractors = set(task.get("distractor_indices", []))
        L.append(f"## Task: {tname}\n")
        L.append("| arm (rate \\| dedup) | retention | passages kept | distractors leaked | mean F1 | EM | correct | single-mention |")
        L.append("|---|---|---|---|---|---|---|---|")
        details = []
        for armkey, arm in task["arms"].items():
            print(f"scoring {tname} {armkey} ...", flush=True)
            rows = score_arm(client, arm, instruction, qa)
            s = summarize(rows)
            kept = arm.get("kept_indices") or []
            leaked = sorted(set(kept) & distractors)
            leaked_str = (",".join(map(str, leaked)) if distractors else "—") or "none"
            kept_str = f"{len(kept)}/{arm.get('total_units','?')}" if distractors else "(single doc)"
            L.append(
                f"| {armkey} | {arm['retention']} | {kept_str} | {leaked_str} | "
                f"{s['mean_f1']} | {s['em']}/{s['total']} | {s['correct']}/{s['total']} | "
                f"{s['single_correct']}/{s['single_total']} |"
            )
            details.append((armkey, arm, rows))

        # per-arm question detail + fault tally
        for armkey, arm, rows in details:
            faults = {}
            for r in rows:
                if r["fault"]:
                    faults[r["fault"]] = faults.get(r["fault"], 0) + 1
            L.append(f"\n<details><summary>{tname} — {armkey} per-question</summary>\n")
            L.append("| Q | gold | single? | answer | F1 | fault |")
            L.append("|---|---|---|---|---|---|")
            for r in rows:
                L.append(f"| {r['q']} | {r['gold']} | {'Y' if r['single'] else ''} | "
                         f"{r['answer'][:36]} | {r['f1']} | {r['fault']} |")
            mmr = arm.get("mmr")
            dropped = len(mmr["dropped_sentences"]) if mmr else 0
            L.append(f"\nfaults: {faults or 'none'} | MMR dropped sentences: {dropped}\n")
            L.append("</details>\n")

    report = "\n".join(L)
    with open(os.path.join(HERE, "eval_report.md"), "w") as f:
        f.write(report)
    print("\n" + report)
    print("\nWrote eval_report.md")


if __name__ == "__main__":
    main()
