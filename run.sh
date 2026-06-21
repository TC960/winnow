#!/usr/bin/env bash
#
# One command to bring the whole stack up for a demo:
#   1. Deploy the GPU worker to Modal (idempotent).
#   2. Warm it up — builds the GPU snapshot + loads the model so it's hot.
#   3. Start the FastAPI backend that proxies requests to the worker.
#
# Usage:
#   ./run.sh                 # deploy + warm + serve on port 8000
#   PORT=9000 ./run.sh       # serve on a different port
#   SKIP_DEPLOY=1 ./run.sh   # skip step 1 (worker already deployed)
#   SKIP_WARMUP=1 ./run.sh   # skip step 2 (worker already warm)
#
set -euo pipefail

# Run from this script's directory regardless of where it's invoked from.
cd "$(dirname "$0")"

VENV="${VENV:-.venv}"
# Fall back to the parent dir's venv if there isn't one alongside the script.
if [[ ! -x "$VENV/bin/python" && -x "../$VENV/bin/python" ]]; then
  VENV="../$VENV"
fi
PY="$VENV/bin/python"
MODAL="$VENV/bin/modal"
UVICORN="$VENV/bin/uvicorn"
PORT="${PORT:-8000}"

if [[ "${SKIP_DEPLOY:-0}" != "1" ]]; then
  echo "==> [1/3] Deploying GPU worker to Modal..."
  "$MODAL" deploy llmlingua2_modal.py
else
  echo "==> [1/3] Skipping deploy (SKIP_DEPLOY=1)."
fi

if [[ "${SKIP_WARMUP:-0}" != "1" ]]; then
  echo "==> [2/3] Warming up worker (one-time: snapshot + model load)..."
  "$PY" warmup.py
else
  echo "==> [2/3] Skipping warmup (SKIP_WARMUP=1)."
fi

echo "==> [3/3] Starting backend API on http://localhost:${PORT} (Ctrl-C to stop)..."
exec "$UVICORN" server:app --host 0.0.0.0 --port "$PORT"
