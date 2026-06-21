"""Deterministic local scorer for the Winnow compression bench.

Reads data.json (gold answers, keyed by id), every *_answers.json present, and
optionally compressed.json (token-space retention), then writes REPORT.md.

No LLM / GPU / network: model answers come from the *_answers.json files; scoring
is pure SQuAD-normalized token-F1 / exact-match (math copied from
experiments/score_eval.py). Robust to missing answer files (arms are skipped).

    cd experiments/bench && python score.py
"""

import glob
import json
import os
import re
import string

HERE = os.path.dirname(os.path.abspath(__file__))
F1_CORRECT = 0.5

# 7 arms, in report order: token-space first, then model-space.
TOKEN_ARMS = ["lingua", "union", "intersection"]
MODEL_ARMS = ["vanilla_llm", "llm_tq", "vanilla_lclm", "lclm_tq"]
ALL_ARMS = TOKEN_ARMS + MODEL_ARMS


# ---- F1 / EM math (copied verbatim from experiments/score_eval.py) ----------
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


# ---------------------------------------------------------------------------
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def collect_answers():
    """Map arm -> {id: answer_record} from all *_answers.json files present."""
    by_arm = {}
    for path in sorted(glob.glob(os.path.join(HERE, "*_answers.json"))):
        doc = load_json(path)
        if not doc:
            continue
        for rec in doc.get("answers", []):
            arm = rec.get("arm")
            if arm:
                by_arm.setdefault(arm, {})[rec["id"]] = rec
    return by_arm


def load_retention():
    """arm -> mean retention from compressed.json (token-space arms)."""
    doc = load_json(os.path.join(HERE, "compressed.json"))
    if not doc:
        return {}
    retentions = {a: [] for a in TOKEN_ARMS}
    for ex in doc.get("examples", []):
        for arm, info in ex.get("arms", {}).items():
            if arm in retentions and info.get("retention") is not None:
                retentions[arm].append(info["retention"])
    return {a: mean(v) for a, v in retentions.items() if v}


def score_arm(arm, recs, golds):
    f1s, ems, correct, n = [], 0, 0, 0
    for ex_id, rec in recs.items():
        if ex_id not in golds:
            continue
        n += 1
        f1 = best_f1(rec.get("answer", ""), golds[ex_id])
        f1s.append(f1)
        if exact_match(rec.get("answer", ""), golds[ex_id]):
            ems += 1
        if f1 >= F1_CORRECT:
            correct += 1
    return {"mean_f1": mean(f1s), "em": ems, "correct": correct, "total": n}


def compression_cell(arm, recs, retentions):
    """Return (compression_str, notes_str) for an arm."""
    if arm in TOKEN_ARMS:
        r = retentions.get(arm)
        if r is None:
            return "—", "no compressed.json"
        return f"{r:.1%} kept ({1.0 / r:.2f}x)", ""
    if arm == "vanilla_llm":
        return "1.00x (baseline)", "full fp16 KV"
    # model-space quantized / latent arms: pull from extra fields
    kvx = mean([rec.get("extra", {}).get("kv_compression_x") for rec in recs.values()])
    bits = mean([rec.get("extra", {}).get("eff_bits") for rec in recs.values()])
    parts, notes = [], []
    if kvx is not None:
        parts.append(f"{kvx:.2f}x KV")
    if bits is not None:
        notes.append(f"{bits:.2f} eff_bits")
    if arm in ("vanilla_lclm", "lclm_tq"):
        seqx = mean([
            (rec.get("extra", {}).get("input_tokens") /
             rec["extra"]["latent_tokens"])
            for rec in recs.values()
            if rec.get("extra", {}).get("input_tokens")
            and rec.get("extra", {}).get("latent_tokens")
        ])
        if seqx is not None:
            parts.append(f"{seqx:.2f}x seq")
    return (" / ".join(parts) if parts else "—"), ", ".join(notes)


def main():
    data = load_json(os.path.join(HERE, "data.json"))
    if not data:
        raise SystemExit("data.json not found in this directory; run from bench/.")
    golds = {ex["id"]: ex["answers"] for ex in data["examples"]}
    total_examples = len(golds)

    by_arm = collect_answers()
    retentions = load_retention()

    lines = ["# Winnow compression bench — results\n"]
    cfg = data.get("config", {})
    lines.append(
        f"- dataset: `{cfg.get('source')}` | tasks: {cfg.get('tasks')} | "
        f"examples: {total_examples} | max_context_words: {cfg.get('max_context_words')}"
    )
    lines.append(f"- correct = token-F1 >= {F1_CORRECT}; scoring = SQuAD-normalized F1/EM (deterministic)\n")
    lines.append("| arm | mean F1 | EM | correct/total | compression | notes |")
    lines.append("|---|---|---|---|---|---|")

    present = []
    for arm in ALL_ARMS:
        recs = by_arm.get(arm)
        if not recs:
            continue
        present.append(arm)
        s = score_arm(arm, recs, golds)
        comp, notes = compression_cell(arm, recs, retentions)
        f1_str = f"{s['mean_f1']:.3f}" if s["mean_f1"] is not None else "—"
        lines.append(
            f"| {arm} | {f1_str} | {s['em']}/{s['total']} | "
            f"{s['correct']}/{s['total']} | {comp} | {notes} |"
        )

    if not present:
        lines.append("| _(no answer files yet)_ |  |  |  |  |  |")

    missing = [a for a in ALL_ARMS if a not in present]
    if missing:
        lines.append(f"\n_Pending arms (no answer file yet): {', '.join(missing)}_")

    report = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.write(report)
    print(report)
    if not present:
        print("No answer files yet — wrote REPORT.md with empty table.")
    else:
        print(f"Scored arms: {', '.join(present)}")
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
