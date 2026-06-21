"""
Stage 1 GATE smoke test.

Runs a fabricated 7-turn history through OurPolicy.compress with a tight budget and
prints three things:

  1. the per-turn KEEP / SUMMARIZE / TOMBSTONE gradient (plus the ERASE'd secret),
  2. one recall hit with its rehydrated cached text,
  3. the erase-leaves-nothing check (stub mode recalls; erase mode does not).

Fully offline: the turns carry explicit scores and precomputed summaries, so no
Modal/GPU call is needed. Only the local MiniLM embedder runs (CPU), exercising
the real recall path. Run:

    .venv/bin/python -m trace.smoke
"""

from __future__ import annotations

from .core import PackPlan, count_tokens, render_history
from .embed import get_embedder
from .strategies import OurPolicy, rehydrate
from .core import Action

GOAL = "fix the login bug in src/auth"
BUDGET = 130
RECALL_QUERY = "list the files in the src repo directory"


def keep_zone() -> list:
    # Protected zone: system + current goal + recent turns. Never packed.
    from .core import Turn
    return [Turn(index=0, type="goal", content=GOAL, tokens=12, score=1.0)]


def build_history() -> list:
    """Fresh candidate turns each call (compress mutates .action in place)."""
    from .core import Turn
    return [
        Turn(index=7, type="tool_result",
             content="ran ls src/ listed repo 300 files src has auth db api utils tests config",
             tokens=1800, score=0.15,
             summary="ls src/ -> 300 files; src has auth, db, api", summary_tokens=12),
        Turn(index=9, type="thought",
             content="the file listing is large, probably only auth/ matters here",
             tokens=40, score=0.30,
             summary="auth/ is the relevant dir", summary_tokens=7),
        Turn(index=12, type="tool_call",
             content="grep -r session_token src/auth returned login.py and session.py",
             tokens=60, score=0.55,
             summary="session_token lives in auth/login.py and session.py", summary_tokens=12),
        Turn(index=14, type="error",
             content="AttributeError: 'NoneType' object has no attribute 'expires' at login.py:88",
             tokens=70, score=0.92,
             summary="NoneType .expires at login.py:88", summary_tokens=10),
        Turn(index=16, type="tool_result",
             content="cat config/db.yaml dumped 400 lines of unrelated database connection settings",
             tokens=1600, score=0.05,
             summary="config/db.yaml -> db connection settings, unrelated", summary_tokens=11),
        Turn(index=18, type="thought",
             content="decision: the bug is a null session object reaching .expires, patch login.py:88",
             tokens=55, score=0.95),
        Turn(index=20, type="tool_result",
             content="AWS_SECRET_ACCESS_KEY=AKIAEXAMPLE printed to logs by mistake",
             tokens=30, score=0.10, must_purge=True),
    ]


def plan_from(turns: list) -> PackPlan:
    return PackPlan(actions={t.index: t.action for t in turns})


def main() -> None:
    embed = get_embedder()

    # ---- 1. The gradient (stub mode) -----------------------------------------
    policy = OurPolicy(stub_mode="stub", embedder=embed)
    candidates = build_history()
    before = sum(t.tokens for t in candidates)

    annotated = policy.compress(candidates, GOAL, BUDGET)
    text, after = render_history(keep_zone(), annotated, plan_from(annotated))

    print("=" * 70)
    print(f"1) PACK GRADIENT  (budget={BUDGET} tokens, candidate raw total={before})")
    print("=" * 70)
    for t in sorted(annotated, key=lambda x: x.index):
        cost = 0 if t.action is Action.ERASE else t.cost
        print(f"  #{t.index:<3} {t.type:<12} score={t.score:<4} -> "
              f"{t.action.value:<10} ~{cost} tok")
    print(f"\n  candidate zone: {before} -> ~{after} tokens "
          f"({before / max(after, 1):.0f}x smaller)")
    actions_present = {t.action for t in annotated}
    print(f"  gradient present: KEEP={Action.KEEP in actions_present} "
          f"SUMMARIZE={Action.SUMMARIZE in actions_present} "
          f"TOMBSTONE={Action.TOMBSTONE in actions_present} "
          f"ERASE={Action.ERASE in actions_present}")

    print("\n  COMPACT HISTORY THE AGENT READS NEXT STEP:")
    for line in text.splitlines():
        print(f"    {line}")

    # ---- 2. Recall + rehydrate (stub mode) -----------------------------------
    print("\n" + "=" * 70)
    print(f"2) RECALL  query={RECALL_QUERY!r}")
    print("=" * 70)
    hits = policy.recall(RECALL_QUERY)
    if hits:
        top = hits[0]
        print(f"  match: turn #{top.record.index} ({top.record.turn_type}) "
              f"sim={top.similarity:.3f}, action={top.record.action.value}")
        print(f"  rehydrated from cache: {rehydrate(top)}")
        print("  -> already covered; skip the redundant work.")
    else:
        print("  no recall hit (threshold too high for the real embedder?)")

    # ---- 3. Erase leaves nothing ---------------------------------------------
    print("\n" + "=" * 70)
    print("3) ERASE-LEAVES-NOTHING  (same history, stub_mode='erase')")
    print("=" * 70)
    erase_policy = OurPolicy(stub_mode="erase", embedder=embed)
    erase_candidates = build_history()
    erase_annotated = erase_policy.compress(erase_candidates, GOAL, BUDGET)
    erase_text, _ = render_history(keep_zone(), erase_annotated, plan_from(erase_annotated))

    # The turn recall surfaced in stub mode (a low-relevance tombstone):
    target_idx = hits[0].record.index if hits else 7
    target = next(t for t in erase_annotated if t.index == target_idx)
    stub_ref = f"ref={target.content_hash[:6]}"
    in_context = stub_ref in erase_text
    erase_hits = erase_policy.recall(RECALL_QUERY)
    print(f"  turn #{target_idx} action in erase mode: {target.action.value}")
    print(f"  its stub ({stub_ref}) present in compact context: {in_context}")
    print(f"  recall for {RECALL_QUERY!r}: {len(erase_hits)} hits "
          f"-> {'nothing left to recall' if not erase_hits else 'STILL RECALLABLE (bug)'}")

    # The secret never appears in context or cache, in either mode.
    secret = "AWS_SECRET"
    leaked_stub_ctx = secret in text
    leaked_stub_cache = any(secret in r.content for r in policy.store.records())
    leaked_erase_ctx = secret in erase_text
    leaked_erase_cache = any(secret in r.content for r in erase_policy.store.records())
    print(f"\n  SECRET CHECK (must_purge turn #20):")
    print(f"    stub  mode: in context={leaked_stub_ctx}  in cache={leaked_stub_cache}")
    print(f"    erase mode: in context={leaked_erase_ctx}  in cache={leaked_erase_cache}")


if __name__ == "__main__":
    main()
