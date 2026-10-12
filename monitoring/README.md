# Local ML monitoring milestone

This folder is a dependency-free reference for the evidence loop needed around
a deployed model: baseline evaluation, feature drift detection, a retraining
decision, and a local prediction endpoint. The model is deliberately a tiny
threshold classifier over synthetic `signal_density` values. It demonstrates
the control path without pretending to measure Winnow's GPU compressor or a
production workload.

## Reproduce the report

```bash
python -m monitoring.demo --output /tmp/winnow-ml-monitor/report.json
cat /tmp/winnow-ml-monitor/report.json
```

The fixture contains two consecutive shifted windows. Population Stability
Index (PSI) is compared with a `0.20` threshold; quality is compared with an F1
threshold of `0.80`. Two bad windows trigger retraining, after which the current
window becomes the new reference and the counter resets.

## Run the local serving path

```bash
python -m monitoring.service --port 8765
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/predict \
  -H 'content-type: application/json' \
  -d '{"signal_density": 0.8}'
curl http://127.0.0.1:8765/metrics
```

The service exposes `/health`, `/predict`, and `/metrics`. It is intentionally
local and in-memory. It does not prove cloud deployment, durable metric
storage, alert delivery, model registry integration, real user traffic, drift
on production distributions, or retraining quality on a labeled corpus.

## Verification

```bash
python -m pytest -q monitoring/test_pipeline.py
```

The tests cover deterministic training and evaluation, the PSI boundary, and
the two-window retraining state transition.
