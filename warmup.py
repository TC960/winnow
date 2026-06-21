"""
Warm the deployed Modal worker before a demo.

Sends one request to the deployed Compressor. On the first ever call this builds
the GPU memory snapshot and loads the model onto the GPU (slow, one-time). It
also leaves a warm container running, so the first real demo request is instant.

    python warmup.py
"""

import time

import modal

MODAL_APP_NAME = "llmlingua2-xlm"
MODAL_CLASS_NAME = "Compressor"

WARMUP_TEXT = (
    "This is a warmup request used to load the compression model onto the GPU "
    "before the live demo so that real requests respond instantly."
)


def main():
    Compressor = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    print("Warming up worker (building snapshot / loading model on first run)...")
    t0 = time.time()
    out = Compressor().compress.remote(WARMUP_TEXT, rate=0.5)
    elapsed = time.time() - t0
    print(f"Warm. First call took {elapsed:.1f}s.")
    print(f"Sample compressed output: {out['compressed_prompt']!r}")


if __name__ == "__main__":
    main()
