"""Run the complete local drift, quality, retraining, and report loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import DriftMonitor, Example, evaluate, train_threshold


BASELINE = [Example(v, int(v >= 0.5)) for v in (0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90)]
DRIFT_WINDOWS = [
    [(0.15, 0), (0.25, 0), (0.75, 1), (0.85, 1)],
    [(0.10, 0), (0.20, 0), (0.70, 1), (0.90, 1)],
]


def run(output: Path) -> dict:
    model = train_threshold(BASELINE)
    initial_model = model
    monitor = DriftMonitor([e.signal_density for e in BASELINE], quality_threshold=0.8)
    windows = []
    for values in DRIFT_WINDOWS:
        window_values = [v for v, _ in values]
        window_labels = [y for _, y in values]
        windows.append(monitor.observe(window_values, window_labels, model))
    retrained = False
    if windows[-1]["retraining_recommended"]:
        latest = [Example(v, y) for v, y in DRIFT_WINDOWS[-1]]
        model = monitor.retrain([e.signal_density for e in latest], [e.label for e in latest], version=2)
        retrained = True
    report = {
        "baseline_model": {"threshold": initial_model.threshold, "version": initial_model.version},
        "windows": windows,
        "retrained": retrained,
        "final_model": {"threshold": model.threshold, "version": model.version},
        "baseline_evaluation": evaluate(train_threshold(BASELINE), BASELINE).to_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/winnow-ml-monitor/report.json"))
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
