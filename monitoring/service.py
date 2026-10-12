"""Local monitored HTTP serving path.

Run ``python -m monitoring.service`` and send JSON to ``POST /predict``:
``{"signal_density": 0.8}``.  This service uses only the standard library and
is intended for local reproducibility, not production deployment.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .pipeline import ThresholdModel


class _Handler(BaseHTTPRequestHandler):
    model = ThresholdModel(0.5)
    requests = 0
    positives = 0

    def _write(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write(200, {"status": "ok", "model_version": self.model.version})
        elif self.path == "/metrics":
            self._write(200, {"predictions_total": self.requests, "positive_predictions": self.positives})
        else:
            self._write(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._write(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length))["signal_density"]
            value = float(value)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._write(400, {"error": "signal_density must be numeric"})
            return
        prediction = self.model.predict_one(value)
        type(self).requests += 1
        type(self).positives += prediction
        self._write(200, {"prediction": prediction, "model_version": self.model.version})

    def log_message(self, *_args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"serving monitored model on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
