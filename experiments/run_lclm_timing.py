"""
Driver: LCLM vanilla (fp16 KV) vs LCLM + TurboQuant, swept over context length x
TQ configs x decode length. Calls the DEPLOYED Modal classes (robust vs the
`modal run` log-stream flakiness) and writes lclm_tq_timing_results.json.

    modal deploy lclm_tq_timing.py
    python run_lclm_timing.py

The two arms run in two SEPARATE warm containers (LCLMVanilla / LCLMTurboQuant),
so peak-memory and timing are isolated.
"""

import json

import modal

from lclm_tq_timing import DEFAULT_CHECKPOINT

# Modest sweep to control cost; 32k is optional (drop if too slow/expensive).
CONTEXT_TOKENS = [2000, 8000, 16000]
DECODE_LENGTHS = [128, 512]

# TQ configs: 4-bit, 3-bit, and 3.5-bit (bit_width=3 + 32 outlier channels @ 4-bit)
TQ_CONFIGS = [
    {"name": "TQ-4bit", "bit_width": 4, "outlier_channels": 0, "outlier_bits": 0},
    {"name": "TQ-3bit", "bit_width": 3, "outlier_channels": 0, "outlier_bits": 0},
    {"name": "TQ-3.5bit", "bit_width": 3, "outlier_channels": 32, "outlier_bits": 4},
]

Vanilla = modal.Cls.from_name("lclm-tq-timing", "LCLMVanilla")
TurboQuant = modal.Cls.from_name("lclm-tq-timing", "LCLMTurboQuant")
vanilla = Vanilla(checkpoint=DEFAULT_CHECKPOINT)
turbo = TurboQuant(checkpoint=DEFAULT_CHECKPOINT)

runs = []
for ctx in CONTEXT_TOKENS:
    for decode in DECODE_LENGTHS:
        print(f"\n=== ctx~{ctx} tok, decode={decode} ===", flush=True)

        print(f"  [vanilla] ...", flush=True)
        v = vanilla.generate.remote(target_tokens=ctx, max_new_tokens=decode)
        v["arm"] = "vanilla"
        v["config"] = "fp16"
        print(f"    ctx={v['input_context_tokens']} latents={v['latent_tokens']} "
              f"ttft={v['ttft_s']}s decode={v['decode_tokens_per_s']}tok/s "
              f"peak={v['peak_gpu_mb']}MB kv={v['kv_bytes']}B needle={v['needle_ok']}", flush=True)
        runs.append(v)

        for cfg in TQ_CONFIGS:
            print(f"  [{cfg['name']}] ...", flush=True)
            r = turbo.generate.remote(
                target_tokens=ctx, max_new_tokens=decode,
                bit_width=cfg["bit_width"],
                outlier_channels=cfg["outlier_channels"],
                outlier_bits=cfg["outlier_bits"],
            )
            r["arm"] = "turboquant"
            r["config"] = cfg["name"]
            print(f"    ttft={r['ttft_s']}s decode={r['decode_tokens_per_s']}tok/s "
                  f"peak={r['peak_gpu_mb']}MB kv={r['kv_bytes']}B "
                  f"({r['kv_compression_x']}x, eff={r['eff_bits']}b) "
                  f"needle={r['needle_ok']}", flush=True)
            runs.append(r)

out = {
    "config": {
        "checkpoint": DEFAULT_CHECKPOINT,
        "context_tokens": CONTEXT_TOKENS,
        "decode_lengths": DECODE_LENGTHS,
        "tq_configs": TQ_CONFIGS,
        "gpu": "A100-80GB",
    },
    "runs": runs,
}
with open("lclm_tq_timing_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote lclm_tq_timing_results.json", flush=True)
