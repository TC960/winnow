"""
Stage 4 GATE: the stub-vs-erase ablation on recall-dependent questions.

This is the headline measurement. It answers one question: when budget pressure
forces old turns out of context, does keeping a tombstone (an in-context pointer
plus a cached embedding, so recall can resurface the turn) beat truly erasing it?

The harness:
  1. Replays one long scripted conversation through OurPolicy.compress, packing it
     to a tight budget so the buried facts get tombstoned out of context.
  2. For stub_mode in (stub, erase) and 3 seeds, asks Claude a held-out set of
     recall-dependent questions (questions about content that got compressed away)
     using ONLY the compacted history.
       - stub: recall + rehydrate is allowed before answering. A cached tombstone
               embedding can resurface the buried turn into the answer prompt.
       - erase: recall is not allowed. The turn is zero bytes and not in the cache,
               so the buried fact is simply gone.
  3. Grades each answer correct/incorrect against the expected answer (Claude as a
     strict grader, temperature 0).
  4. Writes a per-question CSV, a summary CSV, and a grouped bar chart of
     recall-dependent accuracy and tokens-after, stub vs erase.

Determinism / honesty notes:
  - The scripted turns carry faithful precomputed summaries, so SUMMARIZE turns use
    the cached summary and the pack never needs the Modal GPU worker. The compression
    decisions (KEEP / SUMMARIZE / TOMBSTONE) are still made by the real packer over
    real MiniLM embedding scores, not hand-set.
  - The Anthropic Messages API has no `seed` parameter, so "3 seeds" means 3
    independent trials at temperature 0. Identical sampling params across both arms;
    the only thing that differs between arms is stub vs erase. The seeds establish
    that the result is stable, not a single lucky draw.

Run:
    pip install anthropic matplotlib          # one time
    export ANTHROPIC_API_KEY=...              # or put it in web/.env.local
    .venv/bin/python -m trace.eval
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .core import Turn, count_tokens, render_history
from .strategies import OurPolicy, rehydrate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"   # matches the id the web app pins (web/app/api/qa)
SEEDS = [0, 1, 2]
STUB_MODES = ["stub", "erase"]

# Tight budget so the candidate zone cannot keep everything: the buried facts get
# pushed down the gradient to TOMBSTONE. keep_last_k protects only the 4 most
# recent (deployment-focused) turns, none of which hold a buried fact.
BUDGET = 130
KEEP_LAST_K = 4
KEEP_THRESHOLD = 0.85
SUMMARY_THRESHOLD = 0.20

OUT_DIR = Path(__file__).resolve().parent
CSV_DETAIL = OUT_DIR / "eval_results.csv"
CSV_SUMMARY = OUT_DIR / "eval_summary.csv"
CHART_PNG = OUT_DIR / "eval_results.png"


# ---------------------------------------------------------------------------
# The scripted conversation
#
# A planning session whose CURRENT goal is "finalize the deployment plan for the
# payments service". The buried facts (db password rotation, offsite city, legacy
# auth error code, billing-script owner, webhook retry count) are off-topic for
# that goal, so a goal-conditioned pack tombstones them. The held-out questions
# then ask about exactly those buried facts: only a recall path can answer.
#
# Every turn carries a faithful summary, so SUMMARIZE turns never call Modal and
# a rehydrated tombstone still contains its fact.
# ---------------------------------------------------------------------------

GOAL = "finalize the deployment plan for the payments service rollout"


@dataclass
class Question:
    text: str
    expected: str
    # A keyword that must appear in the rehydrated turn for stub to have a chance;
    # used only for a sanity assertion, not for grading.
    fact_marker: str


def build_history() -> list[Turn]:
    """Fresh turns each call (compress mutates .action / .cost in place)."""
    raw = [
        # index, type, content, summary
        (1, "user",
         "Kicking off planning. Before deployment topics, two housekeeping items.",
         "Planning kickoff, housekeeping first."),
        (2, "assistant",
         "Noted. On security hygiene we agreed earlier to rotate the Postgres "
         "database password every 90 days, automated through Vault.",
         "Agreed: rotate the Postgres password every 90 days via Vault."),
        (3, "user",
         "Good. Also the team Q3 offsite logistics are locked in.",
         "Q3 offsite logistics are locked."),
        (4, "assistant",
         "Right, the Q3 offsite is scheduled in Lisbon, the week of September 14th.",
         "Q3 offsite is in Lisbon, week of September 14th."),
        (5, "user",
         "One unrelated IT ticket follow-up before we move on.",
         "An unrelated IT ticket follow-up."),
        (6, "assistant",
         "The internal HR wiki's legacy single sign-on returned HTTP 419 on every "
         "login yesterday; IT has a ticket open with the identity vendor.",
         "The HR wiki legacy SSO returned HTTP 419 on login."),
        (7, "user",
         "And the office onboarding docs ownership is settled?",
         "Office onboarding docs ownership."),
        (8, "assistant",
         "Yes. Priya owns the new-hire office onboarding checklist document going "
         "forward, including the desk and badge setup steps.",
         "Priya owns the new-hire office onboarding checklist document."),
        (9, "user",
         "Last housekeeping detail: the Slack alert bot tuning.",
         "Slack alert bot tuning."),
        (10, "assistant",
         "We set the Slack alert bot max retry count to 7 with exponential backoff "
         "so flaky channel posts stop double-firing.",
         "Slack alert bot max retry count set to 7 with exponential backoff."),
        (11, "user",
         "Great, housekeeping done. Now the real topic: the payments service "
         "deployment plan. What is the rollout strategy?",
         "Transition to the payments deployment plan and rollout strategy."),
        (12, "assistant",
         "For the payments service rollout I recommend a canary deployment: 5% of "
         "traffic first, then 25%, 50%, 100%, gated on error-rate SLOs at each step.",
         "Rollout: canary 5/25/50/100 percent, gated on error-rate SLOs."),
        (13, "user",
         "What is the rollback trigger for the payments deployment?",
         "Rollback trigger for the payments deployment."),
        (14, "assistant",
         "Roll back the payments service automatically if the checkout error rate "
         "exceeds 2% over any 5 minute window during the canary.",
         "Rollback if checkout error rate exceeds 2% over 5 minutes."),
        (15, "user",
         "Which environment do we deploy to first and how do we verify?",
         "First deploy environment and verification for payments."),
        (16, "assistant",
         "Deploy payments to staging first, run the synthetic checkout suite, then "
         "promote to production behind the canary once staging is green.",
         "Deploy payments to staging first, run synthetic checkout, then promote."),
    ]
    return [
        Turn(index=i, type=t, content=c, tokens=count_tokens(c), summary=s,
             summary_tokens=count_tokens(s))
        for (i, t, c, s) in raw
    ]


QUESTIONS = [
    Question(
        "What database password rotation interval did the team agree on?",
        "Every 90 days.", "90 days"),
    Question(
        "Which city is the Q3 offsite scheduled in?",
        "Lisbon.", "Lisbon"),
    Question(
        "What HTTP error code did the HR wiki legacy SSO return?",
        "HTTP 419.", "419"),
    Question(
        "Who owns the new-hire office onboarding checklist?",
        "Priya.", "Priya"),
    Question(
        "What is the Slack alert bot max retry count?",
        "7.", "7"),
]


# ---------------------------------------------------------------------------
# Offline deterministic summarizer (guard only)
#
# Every scripted turn already carries a summary, so the packer's stage-5
# reconstruct never fires. This guard keeps the eval fully offline even if a turn
# ever lacked one: it would not silently reach for the (unauthenticated here)
# Modal worker. Production SUMMARIZE still uses Modal LLMLingua-2; see
# trace/summarize.py.
# ---------------------------------------------------------------------------

def _offline_summarizer(content: str, rate: float) -> tuple[str, int]:
    words = content.split()
    keep = max(1, int(len(words) * rate))
    text = " ".join(words[:keep])
    return text, count_tokens(text)


# ---------------------------------------------------------------------------
# Anthropic client (answering + grading)
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = (
    "You are a careful assistant answering questions about a conversation history.\n"
    "Rules:\n"
    "- Answer ONLY using information present in the context provided.\n"
    "- If the answer is not in the context, reply exactly: Not in context.\n"
    "- Be concise: a few words or one short sentence.\n"
)

GRADER_SYSTEM = (
    "You are a strict grader. You are given a question, the expected answer, and a "
    "candidate answer. Reply with exactly one word: CORRECT if the candidate conveys "
    "the same key fact as the expected answer, otherwise INCORRECT. Treat 'Not in "
    "context' as INCORRECT."
)


def _load_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    # Fall back to the project's web env file, the same place the Next.js routes read.
    env_path = OUT_DIR.parent / "web" / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _make_client():
    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("anthropic SDK not installed. Run: pip install anthropic")
    key = _load_api_key()
    if not key:
        sys.exit(
            "ANTHROPIC_API_KEY not found. Set it in the environment or in "
            "web/.env.local, then rerun: .venv/bin/python -m trace.eval"
        )
    return Anthropic(api_key=key)


def _ask(client, system: str, user: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=128, temperature=0,
        system=system, messages=[{"role": "user", "content": user}],
    )
    block = resp.content[0]
    return (block.text if getattr(block, "type", None) == "text" else "").strip()


def answer(client, context: str, question: str) -> str:
    user = f'Context:\n"""\n{context}\n"""\n\nQuestion: {question}'
    return _ask(client, ANSWER_SYSTEM, user)


def grade(client, question: str, expected: str, candidate: str) -> bool:
    user = (
        f"Question: {question}\nExpected answer: {expected}\n"
        f"Candidate answer: {candidate}\n\nReply CORRECT or INCORRECT."
    )
    verdict = _ask(client, GRADER_SYSTEM, user).upper()
    return verdict.startswith("CORRECT")


# ---------------------------------------------------------------------------
# One pass: pack the history under a stub_mode, return compact context + tokens
# ---------------------------------------------------------------------------

@dataclass
class Pass:
    compact: str
    tokens_after: int
    tokens_before: int
    actions: dict[int, str]
    policy: OurPolicy
    keep_zone: list[Turn]


def run_pass(stub_mode: str) -> Pass:
    turns = build_history()
    policy = OurPolicy(
        stub_mode=stub_mode,
        summarizer=_offline_summarizer,
        keep_threshold=KEEP_THRESHOLD,
        summary_threshold=SUMMARY_THRESHOLD,
    )
    # Ingest every turn (caches embeddings off the critical path, as in production).
    for t in turns:
        policy.observe(t)

    ordered = sorted(turns, key=lambda x: x.index)
    keep_zone = ordered[len(ordered) - KEEP_LAST_K:] if KEEP_LAST_K else []
    keep_idx = {t.index for t in keep_zone}
    candidates = [t for t in ordered if t.index not in keep_idx]

    annotated = policy.compress(candidates, GOAL, BUDGET)
    # observe() registered every turn with the default TOMBSTONE action, including
    # the keep-zone. Mark the keep-zone as KEEP in the store so recall (which filters
    # tombstones_only) does not surface a recent, already-in-context turn as a hit.
    from .core import Action
    for t in keep_zone:
        policy.store.set_action(t.content_hash, Action.KEEP)
    plan = policy.last_plan
    compact, tokens_after = render_history(keep_zone, annotated, plan)
    tokens_before = sum(t.tokens for t in turns)
    actions = {t.index: (t.action.value if t.action else "keep") for t in annotated}
    return Pass(compact, tokens_after, tokens_before, actions, policy, keep_zone)


def build_answer_context(p: Pass, stub_mode: str, question: str) -> tuple[str, bool]:
    """Compact context, plus rehydrated recall hits when stub mode allows it.

    Returns (context, recall_fired). In erase mode recall is not allowed, so the
    context is exactly the compacted history.
    """
    context = p.compact
    recall_fired = False
    if stub_mode == "stub":
        hits = p.policy.recall(question)
        if hits:
            recall_fired = True
            notes = "\n".join(f"- {rehydrate(h)}" for h in hits)
            context = (
                f"{p.compact}\n\n"
                f"[recalled from cache via tombstone embedding match]\n{notes}"
            )
    return context, recall_fired


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class Row:
    stub_mode: str
    seed: int
    question: str
    expected: str
    model_answer: str
    correct: bool
    recall_fired: bool
    tokens_after: int


def main() -> None:
    client = _make_client()

    # Sanity: confirm the buried facts actually got tombstoned (compressed away).
    probe = run_pass("stub")
    tombstoned = [i for i, a in probe.actions.items() if a == "tombstone"]
    print(f"goal: {GOAL!r}")
    print(f"budget={BUDGET}, keep_last_k={KEEP_LAST_K}")
    print(f"candidate actions: {dict(sorted(probe.actions.items()))}")
    print(f"tombstoned turns: {sorted(tombstoned)}")
    print(f"compact context tokens: {probe.tokens_before} -> {probe.tokens_after}\n")

    rows: list[Row] = []
    for stub_mode in STUB_MODES:
        for seed in SEEDS:
            p = run_pass(stub_mode)  # fresh policy per (mode, seed): independent store
            for q in QUESTIONS:
                context, recall_fired = build_answer_context(p, stub_mode, q.text)
                ans = answer(client, context, q.text)
                ok = grade(client, q.text, q.expected, ans)
                rows.append(Row(stub_mode, seed, q.text, q.expected, ans, ok,
                                recall_fired, p.tokens_after))
            n_ok = sum(1 for r in rows if r.stub_mode == stub_mode and r.seed == seed and r.correct)
            print(f"  {stub_mode:5s} seed={seed}: {n_ok}/{len(QUESTIONS)} correct, "
                  f"tokens_after={p.tokens_after}")

    _write_detail_csv(rows)
    summary = _write_summary_csv(rows)
    _print_summary(summary)
    _make_chart(summary)


def _write_detail_csv(rows: list[Row]) -> None:
    with CSV_DETAIL.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stub_mode", "seed", "question", "expected", "model_answer",
                    "correct", "recall_fired", "tokens_after"])
        for r in rows:
            w.writerow([r.stub_mode, r.seed, r.question, r.expected, r.model_answer,
                        int(r.correct), int(r.recall_fired), r.tokens_after])


def _write_summary_csv(rows: list[Row]) -> list[dict]:
    summary = []
    for stub_mode in STUB_MODES:
        sub = [r for r in rows if r.stub_mode == stub_mode]
        n = len(sub)
        n_ok = sum(1 for r in sub if r.correct)
        acc = round(n_ok / n, 4) if n else 0.0
        toks = round(sum(r.tokens_after for r in sub) / n, 1) if n else 0.0
        summary.append({
            "stub_mode": stub_mode,
            "n_questions": n,
            "n_correct": n_ok,
            "recall_accuracy": acc,
            "tokens_after": toks,
        })
    with CSV_SUMMARY.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for s in summary:
            w.writerow(s)
    return summary


def _print_summary(summary: list[dict]) -> None:
    print("\n" + "=" * 64)
    print("SUMMARY (recall-dependent questions, averaged over 3 seeds)")
    print("=" * 64)
    print(f"{'mode':7s} {'accuracy':>10s} {'n_correct':>11s} {'tokens_after':>13s}")
    for s in summary:
        print(f"{s['stub_mode']:7s} {s['recall_accuracy']:>10.2%} "
              f"{str(s['n_correct']) + '/' + str(s['n_questions']):>11s} "
              f"{s['tokens_after']:>13.1f}")
    print(f"\nwrote {CSV_DETAIL.name}, {CSV_SUMMARY.name}, {CHART_PNG.name}")


def _make_chart(summary: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping chart. Run: pip install matplotlib")
        return

    modes = [s["stub_mode"] for s in summary]
    acc = [s["recall_accuracy"] * 100 for s in summary]
    toks = [s["tokens_after"] for s in summary]
    colors = {"stub": "#34d399", "erase": "#9ca3af"}
    bar_colors = [colors.get(m, "#60a5fa") for m in modes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.bar(modes, acc, color=bar_colors)
    ax1.set_title("Recall-dependent accuracy")
    ax1.set_ylabel("% correct")
    ax1.set_ylim(0, 100)
    for i, v in enumerate(acc):
        ax1.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)

    ax2.bar(modes, toks, color=bar_colors)
    ax2.set_title("Compact context tokens (after pack)")
    ax2.set_ylabel("tokens")
    for i, v in enumerate(toks):
        ax2.text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=10)

    fig.suptitle("Tombstone (stub) vs erase: recall-dependent ablation", fontsize=12)
    fig.tight_layout()
    fig.savefig(CHART_PNG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
