"""Reproducible drift and retraining loop for a tiny local model.

The model is intentionally small: it predicts whether a transcript segment should
be retained from one numeric feature, ``signal_density``.  This is a monitoring
reference, not a claim about the production compressor.  The same interfaces can
wrap a real model once a labeled production fixture exists.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Example:
    signal_density: float
    label: int


@dataclass(frozen=True)
class ThresholdModel:
    threshold: float
    version: int = 1

    def predict_one(self, signal_density: float) -> int:
        return int(signal_density >= self.threshold)

    def predict(self, values: Iterable[float]) -> list[int]:
        return [self.predict_one(v) for v in values]


@dataclass(frozen=True)
class EvaluationReport:
    count: int
    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(model: ThresholdModel, examples: Sequence[Example]) -> EvaluationReport:
    if not examples:
        raise ValueError("evaluation set must not be empty")
    predictions = model.predict(e.signal_density for e in examples)
    labels = [e.label for e in examples]
    tp = sum(p == y == 1 for p, y in zip(predictions, labels))
    tn = sum(p == y == 0 for p, y in zip(predictions, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return EvaluationReport(
        count=len(examples),
        accuracy=(tp + tn) / len(examples),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def train_threshold(examples: Sequence[Example], version: int = 1) -> ThresholdModel:
    """Choose the threshold with the best F1, breaking ties toward simplicity."""
    if not examples:
        raise ValueError("training set must not be empty")
    candidates = sorted({e.signal_density for e in examples})
    best = None
    for threshold in candidates:
        candidate = ThresholdModel(threshold=threshold, version=version)
        score = evaluate(candidate, examples)
        key = (score.f1, score.accuracy, -threshold)
        if best is None or key > best[0]:
            best = (key, candidate)
    assert best is not None
    return best[1]


def _histogram(values: Sequence[float], edges: Sequence[float]) -> list[float]:
    counts = [0] * (len(edges) - 1)
    for value in values:
        if value < edges[0]:
            counts[0] += 1
            continue
        if value > edges[-1]:
            counts[-1] += 1
            continue
        for i in range(len(edges) - 1):
            if (edges[i] <= value < edges[i + 1]) or (
                i == len(edges) - 2 and value == edges[i + 1]
            ):
                counts[i] += 1
                break
    total = len(values)
    return [count / total for count in counts]


def psi(reference: Sequence[float], current: Sequence[float], bins: int = 10) -> float:
    """Population Stability Index using shared equal-width reference bins."""
    if not reference or not current:
        raise ValueError("PSI requires non-empty windows")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    lo, hi = min(reference), max(reference)
    if lo == hi:
        return 0.0 if all(v == lo for v in current) else float("inf")
    width = (hi - lo) / bins
    edges = [lo + width * i for i in range(bins)] + [hi]
    expected = _histogram(reference, edges)
    actual = _histogram(current, edges)
    value = 0.0
    for e, a in zip(expected, actual):
        # 1e-6 avoids log(0) while remaining negligible at fixture sizes.
        e = max(e, 1e-6)
        a = max(a, 1e-6)
        value += (a - e) * __import__("math").log(a / e)
    return value


@dataclass
class DriftMonitor:
    reference_values: list[float]
    psi_threshold: float = 0.20
    quality_threshold: float = 0.80
    consecutive_windows: int = 2
    bad_windows: int = 0

    def observe(self, values: Sequence[float], labels: Sequence[int], model: ThresholdModel) -> dict:
        if len(values) != len(labels) or not values:
            raise ValueError("values and labels must be non-empty and the same length")
        score = evaluate(model, [Example(v, y) for v, y in zip(values, labels)])
        drift = psi(self.reference_values, values)
        bad = drift >= self.psi_threshold or score.f1 < self.quality_threshold
        self.bad_windows = self.bad_windows + 1 if bad else 0
        return {
            "psi": drift,
            "quality": score.to_dict(),
            "drift": drift >= self.psi_threshold,
            "retraining_recommended": self.bad_windows >= self.consecutive_windows,
            "bad_windows": self.bad_windows,
        }

    def retrain(self, values: Sequence[float], labels: Sequence[int], version: int) -> ThresholdModel:
        model = train_threshold([Example(v, y) for v, y in zip(values, labels)], version=version)
        self.reference_values = list(values)
        self.bad_windows = 0
        return model
