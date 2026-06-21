# TurboQuant POC (Modal A100)

Functional proof-of-concept of **TurboQuant** KV-cache compression
(Google Research, ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874))
running on a Modal A100. Send a prompt → an LLM generates on GPU using a
TurboQuant-compressed KV cache → coherent output, with the realized
compression ratio reported next to an FP16 baseline.

Based on the reference implementation at https://github.com/OmarHory/turboquant
(custom `transformers` cache, not vLLM). The `TQCache` is a drop-in
`DynamicCache` subclass passed as `past_key_values`.

## Files
- `turboquant_poc.py` — standalone TurboQuant cache module (`TurboQuantMSE`, `TQLayer`, `TQCache`).
- `modal_app.py` — Modal app: loads a model on A100-80GB, runs baseline + TurboQuant, reports compression. `TQCache` is inlined so the image is self-contained.

## Run
```bash
# Default: Mistral-7B-Instruct-v0.3, 4-bit
modal run modal_app.py --prompt "Your prompt here"

# 3.5-bit outlier-aware (32 channels @ 4-bit, rest @ 3-bit; head_dim=128)
modal run modal_app.py --prompt "..." --bit-width 3 --outlier-channels 32 --outlier-bits 4

# Any standard MHA/GQA model — e.g. a 14B
modal run modal_app.py --model-id Qwen/Qwen2.5-14B-Instruct --prompt "..."
```
Weights are cached on a Modal volume, so reruns skip the download.

## Validated on NVIDIA A100 80GB
| Model | Config | Eff. bits | KV compression | Quality |
|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 4-bit | 4.0 | 3.21x | matches FP16 |
| Mistral-7B-Instruct-v0.3 | 3.5-bit outlier | 3.25 | 4.27x | correct code + explanation |
| **Qwen2.5-14B-Instruct** | 4-bit | 4.0 | **4.14x** | matches FP16 |
| **Qwen2.5-14B-Instruct** | 3.5-bit outlier | 3.25 | **4.27x** | correct linked-list code + explanation |

The 14B (48 layers, 8 KV heads, head_dim=128) ran on the **same code** as the 7B
with only a `--model-id` change — confirming the implementation generalizes
across model size and family (Mistral → Qwen). Sample logs in `run_14b_*.log`.
Note: the 3.5-bit outlier path is slower (pure-Python per-channel masking ×48
layers, ~10 tok/s) — a perf characteristic, not a correctness issue.

## Notes
- Model-agnostic for standard attention (Llama/Mistral/Qwen/Phi, MHA or GQA, head_dim ~64–256). It auto-adapts to layer/head counts from `model.config`. No calibration — the random rotation is data-free.
- Architectural exceptions: Multi-head Latent Attention (DeepSeek-V2/V3) stores a latent, not per-head K/V; models that force a non-`DynamicCache` (some hybrid/sliding caches) need the subclass to extend that cache type.
- Token rate is below FP16 because this is a pure-PyTorch dequantize path; the paper's speedup needs custom CUDA kernels (out of scope for a functional POC).
