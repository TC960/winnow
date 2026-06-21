"""
Trace context compression: Stage 4 packer + tombstone-with-cached-embedding recall.

Design commitments baked into this module:

1. `drop` never means zero bytes. The packer's lowest budget tier is TOMBSTONE, a
   ~8-token in-context stub. True ERASE exists but is reserved for must-not-retain
   content (secrets / PII), never for budget pressure. So the budget action space
   is a gradient: KEEP (verbatim) > SUMMARIZE (distilled) > TOMBSTONE (pointer).

2. The in-context stub is a POINTER, not the recall signal. The "have I been here
   before" power lives in the cached embedding, which sits out of context in the
   Store and costs zero prompt tokens. Stub and embedding are kept separate so we
   never claim the stub does work it cannot.

3. A turn can be TOMBSTONED in context while its summary still lives in the Store.
   Background precompute may have written a summary, but a low-relevance pass can
   decide that turn is not worth even a summary line in context. The cached summary
   stays available for recall and rehydrate. This is the cleanest illustration of
   "the recall signal lives out of context."

4. The embedding is cacheable (content is immutable). The relevance SCORE is not
   (it depends on the current goal, which moves every pass), so the Store caches
   the embedding once and the score is recomputed upstream in stage 3.

5. Tombstones grow O(N). Folding collapses tombstone runs into range stubs to bound
   in-context growth. Folding only changes the rendered context, not the Store, so
   recall is unaffected: every turn keeps its own cached embedding even after its
   stub is folded into a range.

The packer itself is a score-ordered, two-threshold greedy walk (the pipeline doc's
"walk by score, assign keep/summarize/drop, pack to budget"). It is a greedy
approximation to a multiple-choice knapsack; an exact solve is a small DP if you
ever need it, but the greedy is deterministic and reproducible, which is what the
caching and the "is this reproducible" question care about.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Action(Enum):
    KEEP = "keep"            # verbatim content
    SUMMARIZE = "summarize"  # distilled line(s)
    TOMBSTONE = "tombstone"  # in-context pointer; recall via cached embedding
    ERASE = "erase"          # true zero bytes; reserved for must-not-retain content


Embedder = Callable[[list[str]], list[list[float]]]


@dataclass
class Turn:
    index: int
    type: str                 # goal | thought | tool_call | tool_result | error
    content: str
    tokens: int               # cost of verbatim content
    score: float              # relevance-to-goal x structural prior, from stage 3
    summary: Optional[str] = None     # precomputed in background (stage 5 / precompute)
    summary_tokens: int = 0           # cost of that summary
    must_purge: bool = False          # secrets/PII: force ERASE, bypass the packer

    # assigned by the packer:
    action: Optional[Action] = None
    cost: int = 0

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass
class Record:
    """Out-of-context cache entry. Embedding for recall, content/summary for rehydrate."""
    content_hash: str
    embedding: list[float]
    content: str
    summary: Optional[str]
    turn_type: str
    index: int
    action: Action = Action.TOMBSTONE


class Store:
    """Content-addressed cache. Out of context: never counts against the budget."""

    def __init__(self, embed: Embedder):
        self._embed = embed
        self._recs: dict[str, Record] = {}

    def register(self, turn: Turn) -> Record:
        h = turn.content_hash
        rec = self._recs.get(h)
        if rec is None:
            # Embed once. Content is immutable, so this write is what makes
            # "embedding cacheable by content hash" actually true.
            emb = self._embed([turn.content])[0]
            rec = Record(
                content_hash=h, embedding=emb, content=turn.content,
                summary=turn.summary, turn_type=turn.type, index=turn.index,
            )
            self._recs[h] = rec
        else:
            # Cheap per-pass refresh of mutable fields. The embedding is untouched.
            rec.summary = turn.summary or rec.summary
            rec.index = turn.index
        return rec

    def set_action(self, h: str, action: Action) -> None:
        if h in self._recs:
            self._recs[h].action = action

    def records(self) -> list[Record]:
        return list(self._recs.values())


# ---------------------------------------------------------------------------
# Embedding / similarity / tokens
# ---------------------------------------------------------------------------

def make_fake_embedder(dim: int = 256) -> Embedder:
    """Deterministic hashed bag-of-words. Swap for a real embedding API in prod.

    Real drop-in: any function taking list[str] and returning list[list[float]]
    (Voyage, OpenAI, sentence-transformers). Batch it; cache via Store.
    """
    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for tx in texts:
            v = [0.0] * dim
            for tok in re.findall(r"[a-z0-9]+", tx.lower()):
                idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
                v[idx] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out
    return embed


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def count_tokens(text: str) -> int:
    """Rough proxy. Replace with the model's real tokenizer."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Tombstone rendering
# ---------------------------------------------------------------------------

def render_tombstone(turn: Turn) -> str:
    # A pointer, not the signal. ref maps back to the cached embedding/content.
    return f"[#{turn.index} {turn.type} elided ref={turn.content_hash[:6]}]"


def render_range(turns: list[Turn]) -> str:
    a, b = turns[0].index, turns[-1].index
    types = {t.type for t in turns}
    label = next(iter(types)) if len(types) == 1 else "mixed"
    return f"[#{a}-{b} {len(turns)} {label} turns elided]"


# ---------------------------------------------------------------------------
# Stage 4: the packer
# ---------------------------------------------------------------------------

@dataclass
class PackPlan:
    actions: dict[int, Action]                  # turn index -> action
    folds: list[list[Turn]] = field(default_factory=list)
    floor_overflow: bool = False
    spent: int = 0


def _summary_cost(t: Turn, tombstone_tokens: int) -> int:
    # If no summary was precomputed, stage 5 will write one; estimate its size.
    raw = t.summary_tokens or min(t.tokens, 24)
    return max(tombstone_tokens + 1, raw)


def pack(candidates: list[Turn], budget: int, store: Store, *,
         tombstone_tokens: int = 8,
         keep_threshold: float = 0.85,
         summary_threshold: float = 0.20) -> PackPlan:
    """Assign KEEP / SUMMARIZE / TOMBSTONE to each candidate, packing to budget.

    Two thresholds turn the score into a desired tier:
        score >= keep_threshold      -> want KEEP
        summary_threshold <= score   -> want SUMMARIZE
        otherwise                    -> TOMBSTONE (the floor)

    Then a score-ordered greedy spends the budget: fund KEEPs first (high score
    first), then SUMMARIZEs. A want-KEEP that cannot afford verbatim degrades to a
    summary; a want-SUMMARIZE that cannot afford a line stays a tombstone. Every
    turn always has a tombstone floor, so nothing is ever truly dropped for budget.
    """
    # Secrets are handled out-of-band: true erase, no stub, no cache entry.
    purge = [t for t in candidates if t.must_purge]
    work = [t for t in candidates if not t.must_purge]
    actions: dict[int, Action] = {t.index: Action.ERASE for t in purge}

    # Floor everyone to a tombstone and register (caches the embedding for recall).
    for t in work:
        t.action = Action.TOMBSTONE
        t.cost = tombstone_tokens
        store.register(t)
        store.set_action(t.content_hash, Action.TOMBSTONE)

    floor_cost = tombstone_tokens * len(work)

    # Even the tombstone floor overflows -> fold runs into ranges.
    if floor_cost > budget:
        folds = _build_folds(work, budget, tombstone_tokens)
        for t in work:
            actions[t.index] = Action.TOMBSTONE
        return PackPlan(actions=actions, folds=folds, floor_overflow=True,
                        spent=tombstone_tokens * len(folds))

    remaining = budget - floor_cost
    by_score = sorted(work, key=lambda x: x.score, reverse=True)

    def want(t: Turn) -> Action:
        if t.score >= keep_threshold:
            return Action.KEEP
        if t.score >= summary_threshold:
            return Action.SUMMARIZE
        return Action.TOMBSTONE

    # Pass 1: fund KEEPs, highest score first.
    for t in by_score:
        if want(t) is Action.KEEP:
            dc = t.tokens - t.cost
            if dc <= remaining:
                t.action, t.cost = Action.KEEP, t.tokens
                remaining -= dc

    # Pass 2: fund SUMMARIZEs (including want-KEEPs that could not afford verbatim).
    for t in by_score:
        if t.action is Action.KEEP:
            continue
        if want(t) in (Action.KEEP, Action.SUMMARIZE):
            sc = _summary_cost(t, tombstone_tokens)
            if sc >= t.tokens:                      # tiny turn: verbatim is cheaper
                dc = t.tokens - t.cost
                if dc <= remaining:
                    t.action, t.cost = Action.KEEP, t.tokens
                    remaining -= dc
                continue
            dc = sc - t.cost
            if dc <= remaining:
                t.action, t.cost = Action.SUMMARIZE, sc
                remaining -= dc
        # want-TOMBSTONE and the unfunded both stay at the tombstone floor.

    for t in work:
        actions[t.index] = t.action
        store.set_action(t.content_hash, t.action)

    return PackPlan(actions=actions, spent=budget - remaining)


def _build_folds(turns: list[Turn], budget: int, tombstone_tokens: int) -> list[list[Turn]]:
    """Collapse adjacent tombstones into range stubs until the floor fits.

    Folding only changes the in-context render. Every turn keeps its own Store
    record (and embedding), so recall is unaffected by how aggressively we fold.
    """
    turns = sorted(turns, key=lambda t: t.index)
    max_groups = max(1, budget // tombstone_tokens)
    if len(turns) <= max_groups:
        return [[t] for t in turns]
    size = math.ceil(len(turns) / max_groups)
    return [turns[i:i + size] for i in range(0, len(turns), size)]


# ---------------------------------------------------------------------------
# Render the compact history the agent reads next step
# ---------------------------------------------------------------------------

def render_history(keep_zone: list[Turn], candidates: list[Turn],
                   plan: PackPlan) -> tuple[str, int]:
    by_index = {t.index: t for t in candidates}
    lines: list[str] = []

    for t in sorted(keep_zone, key=lambda x: x.index):
        lines.append(f"#{t.index} {t.type}: {t.content}")

    if plan.floor_overflow:
        for grp in plan.folds:
            lines.append(render_range(grp) if len(grp) > 1 else render_tombstone(grp[0]))
    else:
        for idx in sorted(by_index):
            t = by_index[idx]
            act = plan.actions[idx]
            if act is Action.ERASE:
                continue
            if act is Action.KEEP:
                lines.append(f"#{t.index} {t.type}: {t.content}")
            elif act is Action.SUMMARIZE:
                summ = t.summary or f"(summary of {t.type} #{t.index})"
                lines.append(f"#{t.index} {t.type} (summary): {summ}")
            else:
                lines.append(render_tombstone(t))

    text = "\n".join(lines)
    return text, count_tokens(text)


# ---------------------------------------------------------------------------
# Recall: "have I been here before"
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    similarity: float
    record: Record


def recall(query: str, store: Store, embed: Embedder, *,
           threshold: float = 0.40, k: int = 3,
           tombstones_only: bool = True) -> list[Hit]:
    """Cheap cosine of a proposed action against cached (tombstone) embeddings.

    The signal lives in the cache, not in the in-context stub, so this works even
    for turns whose stub was folded into a range. Recall is free in prompt tokens:
    it never touches the live context.
    """
    qv = embed([query])[0]
    hits: list[Hit] = []
    for rec in store.records():
        if tombstones_only and rec.action is not Action.TOMBSTONE:
            continue
        sim = cosine(qv, rec.embedding)
        if sim >= threshold:
            hits.append(Hit(sim, rec))
    hits.sort(key=lambda h: h.similarity, reverse=True)
    return hits[:k]


def rehydrate(hit: Hit) -> str:
    """Pull a matched turn back. Prefer the cheap cached summary; fall back to content."""
    return hit.record.summary or hit.record.content


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    embed = make_fake_embedder()
    store = Store(embed)

    # Keep-zone: protected, never touched (system + current goal + recent turns).
    keep_zone = [
        Turn(index=0, type="goal", content="fix the login bug in src/auth", tokens=12, score=1.0),
    ]

    # Candidate-zone: older turns the packer decides on. Current goal is the login bug.
    candidates = [
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

    BUDGET = 130
    before = sum(t.tokens for t in candidates)

    plan = pack(candidates, BUDGET, store)
    text, after = render_history(keep_zone, candidates, plan)

    print("=" * 70)
    print(f"PACK PLAN  (budget={BUDGET} tokens, candidate raw total={before})")
    print("=" * 70)
    for t in sorted(candidates, key=lambda x: x.index):
        act = plan.actions[t.index]
        cost = 0 if act is Action.ERASE else t.cost
        print(f"  #{t.index:<3} {t.type:<12} score={t.score:<4} -> {act.value:<10} ~{cost} tok")
    print(f"\n  candidate zone: {before} -> ~{after} tokens "
          f"({before / max(after,1):.0f}x smaller)")

    print("\n" + "-" * 70)
    print("COMPACT HISTORY THE AGENT READS NEXT STEP")
    print("-" * 70)
    print(text)

    # Twelve turns later, about to repeat work.
    print("\n" + "=" * 70)
    print("RECALL: agent is about to run `ls src/` again")
    print("=" * 70)
    proposed = "ls src list files in repo directory"
    hits = recall(proposed, store, embed)
    if hits:
        top = hits[0]
        print(f"  match: turn #{top.record.index} ({top.record.turn_type}) "
              f"sim={top.similarity:.3f}, action={top.record.action.value}")
        print(f"  -> already done. rehydrated from cache: {rehydrate(top)}")
        print("  -> skip the redundant tool call.")
    else:
        print("  no prior match, proceed with the call.")

    # The purged secret left no trace in context and no trace in the cache.
    print("\n" + "=" * 70)
    print("ERASE CHECK: secret turn #20 leaves no stub and no cache entry")
    print("=" * 70)
    leaked_ctx = "AWS_SECRET" in text
    leaked_cache = any("AWS_SECRET" in r.content for r in store.records())
    print(f"  in compact context: {leaked_ctx}   in store/cache: {leaked_cache}")
