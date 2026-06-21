"""
The one Compressor interface, plus the four strategies the ablation compares.

The chat loop calls this interface and knows nothing else. Compression is never
inlined into the loop, so the erase-vs-stub ablation stays runnable: swap the
strategy (or flip OurPolicy's stub_mode) and rerun.

  observe(turn)               ingest, cache the embedding off the critical path
  compress(turns, goal, budget) -> annotated turns (action assigned per turn)
  recall(query) -> list[Hit]  "have I been here before"; [] for baselines

compress returns every input turn with its `action` set (KEEP / SUMMARIZE /
TOMBSTONE / ERASE), so a caller can both render the compact context (via
core.render_history) and inspect the gradient. ERASE turns carry zero bytes into
context; everything else is on the KEEP > SUMMARIZE > TOMBSTONE gradient.
"""

from __future__ import annotations

from typing import Optional, Protocol

from .core import (
    Action,
    Embedder,
    Hit,
    Store,
    Turn,
    cosine,
    pack,
    recall as core_recall,
    rehydrate,  # re-exported for callers
)

__all__ = [
    "Compressor",
    "NoOp",
    "Truncate",
    "NaiveSummarize",
    "OurPolicy",
    "rehydrate",
]


class Compressor(Protocol):
    def observe(self, turn: Turn) -> None: ...
    def compress(self, turns: list[Turn], goal: str, budget: int) -> list[Turn]: ...
    def recall(self, query: str) -> list[Hit]: ...


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class NoOp:
    """Send everything verbatim. The do-nothing control."""

    def observe(self, turn: Turn) -> None:
        return None

    def compress(self, turns: list[Turn], goal: str, budget: int) -> list[Turn]:
        for t in turns:
            t.action, t.cost = Action.KEEP, t.tokens
        return turns

    def recall(self, query: str) -> list[Hit]:
        return []


class Truncate:
    """Hard recency window. Keep the most recent turns that fit the budget, drop
    the rest to zero bytes. No stub, no recall: the canonical "lossy truncation"
    baseline the gradient is meant to beat."""

    def observe(self, turn: Turn) -> None:
        return None

    def compress(self, turns: list[Turn], goal: str, budget: int) -> list[Turn]:
        spent = 0
        # Walk newest first, keep while we can afford verbatim.
        for t in sorted(turns, key=lambda x: x.index, reverse=True):
            if spent + t.tokens <= budget:
                t.action, t.cost = Action.KEEP, t.tokens
                spent += t.tokens
            else:
                t.action, t.cost = Action.ERASE, 0
        return turns

    def recall(self, query: str) -> list[Hit]:
        return []


class NaiveSummarize:
    """Summarize every candidate via LLMLingua-2 at a fixed rate. No gradient, no
    recall: the "compress everything the same" baseline."""

    def __init__(self, rate: float = 0.35, summarizer=None):
        self.rate = rate
        # Late-bound so importing this module never imports modal.
        if summarizer is None:
            from .summarize import summarize as summarizer
        self._summarize = summarizer

    def observe(self, turn: Turn) -> None:
        return None

    def compress(self, turns: list[Turn], goal: str, budget: int) -> list[Turn]:
        for t in turns:
            if t.summary is None:
                t.summary, t.summary_tokens = self._summarize(t.content, self.rate)
            t.action, t.cost = Action.SUMMARIZE, t.summary_tokens
        return turns

    def recall(self, query: str) -> list[Hit]:
        return []


# ---------------------------------------------------------------------------
# OurPolicy: the keep/summarize/tombstone gradient with stub-vs-erase ablation
# ---------------------------------------------------------------------------

# Structural priors per turn type: a source/result dump starts low, a decision or
# an error starts high. Multiplied by the goal-cosine to form the relevance score
# when a turn has no precomputed score. Hand-set, not learned.
STRUCTURAL_PRIOR: dict[str, float] = {
    "goal": 1.0,
    "error": 1.1,
    "assistant": 1.1,
    "thought": 0.9,
    "tool_call": 0.9,
    "user": 0.8,
    "tool_result": 0.7,
    "source": 0.6,
}


class OurPolicy:
    """Stage 4 packer behind the interface, with the tombstone recall path.

    stub_mode:
      "stub"  - budget-pressured turns become TOMBSTONEs: an in-context pointer
                plus a cached embedding, so recall can resurface them later.
      "erase" - budget-pressured turns become ERASE: zero bytes AND no cache
                entry, so recall finds nothing. This is the ablation arm.

    Secrets (must_purge) are always ERASE in both modes and are never registered
    in the Store, so they leak into neither context nor cache.
    """

    def __init__(self, stub_mode: str = "stub", embedder: Optional[Embedder] = None,
                 summarizer=None, summary_rate: float = 0.35,
                 keep_threshold: float = 0.85, summary_threshold: float = 0.20,
                 tombstone_tokens: int = 8, recall_threshold: float = 0.35):
        if stub_mode not in ("stub", "erase"):
            raise ValueError(f"stub_mode must be 'stub' or 'erase', got {stub_mode!r}")
        self.stub_mode = stub_mode
        if embedder is None:
            from .embed import get_embedder
            embedder = get_embedder()
        self._embed = embedder
        self.store = Store(embedder)
        self._summarizer = summarizer  # late-bound in compress to avoid importing modal
        self.summary_rate = summary_rate
        self.keep_threshold = keep_threshold
        self.summary_threshold = summary_threshold
        self.tombstone_tokens = tombstone_tokens
        self.recall_threshold = recall_threshold

    def observe(self, turn: Turn) -> None:
        # Ingest: cache the embedding off the critical path. No-op for secrets.
        if turn.must_purge:
            return None
        self.store.register(turn)

    def _score(self, turn: Turn, goal_vec: list[float]) -> float:
        # Score precedence: an explicit precomputed score (stage 3, upstream) wins.
        # Otherwise compute a goal-conditioned score here as a clean seam:
        # cosine(goal, turn) x structural prior, clamped to [0, 1].
        if turn.score:
            return turn.score
        rec = self.store.register(turn)
        prior = STRUCTURAL_PRIOR.get(turn.type, 0.8)
        sim = cosine(goal_vec, rec.embedding)
        return max(0.0, min(1.0, sim * prior))

    def compress(self, turns: list[Turn], goal: str, budget: int) -> list[Turn]:
        goal_vec = self._embed([goal])[0]
        for t in turns:
            t.score = self._score(t, goal_vec)

        plan = pack(
            turns, budget, self.store,
            tombstone_tokens=self.tombstone_tokens,
            keep_threshold=self.keep_threshold,
            summary_threshold=self.summary_threshold,
        )
        # Stash the plan so callers (the /trace/pack endpoint) can read back the
        # real folds / floor_overflow instead of reconstructing them from actions.
        self.last_plan = plan

        # pack assigns must_purge (ERASE) turns only in plan.actions, not on the
        # Turn (they are held out of the work set). Reconcile so t.action is
        # authoritative for every returned turn.
        for t in turns:
            t.action = plan.actions.get(t.index, t.action)

        # Stage 5 reconstruct: fill any SUMMARIZE turn that has no cached summary
        # by calling the Modal worker (cached by content hash + rate).
        for t in turns:
            if plan.actions.get(t.index) is Action.SUMMARIZE and t.summary is None:
                summarizer = self._summarizer
                if summarizer is None:
                    from .summarize import summarize as summarizer
                t.summary, t.summary_tokens = summarizer(t.content, self.summary_rate)
                self.store.register(t)  # refresh cached summary for rehydrate

        # Ablation arm: in erase mode, every budget-pressured tombstone becomes a
        # true erase with no cache entry, so it cannot be recalled.
        if self.stub_mode == "erase":
            for t in turns:
                if t.action is Action.TOMBSTONE:
                    t.action, t.cost = Action.ERASE, 0
                    plan.actions[t.index] = Action.ERASE
                    self.store.drop(t.content_hash)

        return turns

    def recall(self, query: str) -> list[Hit]:
        return core_recall(query, self.store, self._embed,
                           threshold=self.recall_threshold)
