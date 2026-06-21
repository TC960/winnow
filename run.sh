#!/usr/bin/env bash
#
# One command to bring the whole stack up for a demo:
#   1. Deploy the GPU worker to Modal (idempotent).
#   2. Warm it up — builds the GPU snapshot + loads the model so it's hot.
#   3. Expose the backend on a stable public URL via ngrok (static domain).
#   4. Start the FastAPI backend that proxies requests to the worker.
#
# The ngrok static domain never changes, so COMPRESS_BACKEND_URL in the
# frontend can be set once and left alone across backend restarts.
#
# Usage:
#   ./run.sh                 # deploy + warm + tunnel + serve on port 8000
#   PORT=9000 ./run.sh       # serve on a different port
#   SKIP_DEPLOY=1 ./run.sh   # skip step 1 (worker already deployed)
#   SKIP_WARMUP=1 ./run.sh   # skip step 2 (worker already warm)
#   SKIP_NGROK=1 ./run.sh    # skip step 3 (local-only, no public URL)
#   NGROK_DOMAIN=... ./run.sh # override the static domain
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
NGROK_DOMAIN="${NGROK_DOMAIN:-intercartilaginous-collins-nimble.ngrok-free.dev}"

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

if [[ "${SKIP_NGROK:-0}" != "1" ]]; then
  echo "==> [3/4] Opening ngrok tunnel at https://${NGROK_DOMAIN} ..."
  # Reap any tunnel we leave behind when this script exits (Ctrl-C, etc.).
  ngrok http --url="$NGROK_DOMAIN" "$PORT" --log=stdout > ngrok.log 2>&1 &
  NGROK_PID=$!
  trap 'kill "$NGROK_PID" 2>/dev/null || true' EXIT
  echo "    Public URL: https://${NGROK_DOMAIN}  (logs: ./ngrok.log)"
  echo "    Set COMPRESS_BACKEND_URL=https://${NGROK_DOMAIN} in the frontend."
else
  echo "==> [3/4] Skipping ngrok (SKIP_NGROK=1) — backend will be local-only."
fi

echo "==> [4/4] Starting backend API on http://localhost:${PORT} (Ctrl-C to stop)..."
# Not exec'd, so the EXIT trap above can clean up the ngrok tunnel.
"$UVICORN" server:app --host 0.0.0.0 --port "$PORT"
