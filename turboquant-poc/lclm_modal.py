"""
LCLM + TurboQuant POC on Modal (A100-80GB).

Goal: take a *Latent Context Language Model* (LCLM) from "End-to-End Context
Compression at Scale" (arXiv:2606.09659) -- an encoder-decoder soft-token
context compressor -- run it vanilla on a long-context prompt, then re-run it
with our TurboQuant KV-cache compression applied to the decoder, and compare.

Why these two compose
---------------------
The LCLM is encoder (Qwen3-Embedding-0.6B) -> adapter -> decoder
(Qwen3-4B-Instruct-2507). The encoder maps context chunks to a short sequence
of "latent"/"memory" soft tokens; the decoder consumes those latents *as input
embeddings* in place of the raw context and then generates normally.

  * LCLM compresses the *sequence length* of the KV cache (fewer entries).
  * TurboQuant compresses the *bits per entry* of the KV cache.

They are orthogonal -> the savings multiply. Crucially, `LCLM.generate` bottoms
out in a plain `decoder.generate(inputs_embeds=..., **generation_kwargs)`, and
the decoder is a stock `Qwen3ForCausalLM` (36 layers, GQA, 8 KV heads,
head_dim=128) with a standard `DynamicCache`. The latent tokens never change the
attention op or the cache layout, so our `TQCache` (already validated on
Qwen2.5-14B GQA, head_dim=128) drops straight in as `past_key_values`. The
*encoder* runs a single prefill forward (no autoregressive KV cache), so
TurboQuant simply doesn't apply there.

Usage:
    modal run --detach lclm_modal.py                          # default 16x LCLM, 4-bit TQ
    modal run --detach lclm_modal.py --checkpoint latent-context/0.6b-4b-LCLM-8x
    modal run --detach lclm_modal.py --bit-width 3 --outlier-channels 32 --outlier-bits 4  # 3.5-bit
"""

import modal

# Released LCLM checkpoints (Apache-2.0): latent-context/{0.6b-4b-LCLM-4x,-8x,-16x}.
# Each is ~18.5 GB (full Qwen3-Embedding-0.6B encoder + Qwen3-4B decoder + adapter).
DEFAULT_CHECKPOINT = "latent-context/0.6b-4b-LCLM-16x"
LCLM_REPO = "https://github.com/LeonLixyz/LCLM.git"

# Build the image: clone the LCLM repo so `latent_context`, `inference`, and
# `utils` are importable. Skip the repo's heavy training-only deps (deepspeed,
# liger-kernel, wandb); inference needs only the runtime stack below. No
# flash-attn -> the loader auto-falls back to sdpa, which is what our TQCache
# (returns full dequantized K/V) needs anyway.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=4.54,<5",
        "accelerate",
        "peft",
        "safetensors",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "scipy",
        "numpy",
    )
    .run_commands(f"git clone --depth 1 {LCLM_REPO} /LCLM")
    .env({"HF_HOME": "/cache/hf"})
)

app = modal.App("lclm-turboquant-poc")
hf_cache = modal.Volume.from_name("turboquant-hf-cache", create_if_missing=True)


# ===========================================================================
# TurboQuant core (inlined, identical to modal_app.py / turboquant_poc.py).
# A fixed random orthogonal rotation maps each KV vector to a near-Gaussian
# distribution where per-coordinate Lloyd-Max scalar quantization is provably
# near-optimal -- no training, no calibration. TQCache is a DynamicCache
# subclass usable directly as decoder `past_key_values`.
# ===========================================================================
def _build_tq_classes():
    import math
    import numpy as np
    import torch
    from scipy.stats import norm
    from transformers.cache_utils import DynamicCache, DynamicLayer

    def _lloyd_max_gaussian(num_levels, sigma=1.0, max_iter=200):
        k = num_levels
        centroids = np.array([sigma * norm.ppf((2 * i + 1) / (2 * k)) for i in range(k)])
        for _ in range(max_iter):
            boundaries = np.empty(k + 1)
            boundaries[0], boundaries[k] = -np.inf, np.inf
            for i in range(1, k):
                boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0
            new_c = np.empty(k)
            for i in range(k):
                lo, hi = boundaries[i], boundaries[i + 1]
                lo_c, hi_c = max(lo, -6 * sigma), min(hi, 6 * sigma)
                num = norm.expect(lambda x: x, loc=0, scale=sigma, lb=lo_c, ub=hi_c)
                den = norm.cdf(hi, scale=sigma) - norm.cdf(lo, scale=sigma)
                new_c[i] = num / den if den > 1e-15 else (lo_c + hi_c) / 2.0
            if np.allclose(centroids, new_c, atol=1e-12):
                break
            centroids = new_c
        boundaries = np.empty(k + 1)
        boundaries[0], boundaries[k] = -np.inf, np.inf
        for i in range(1, k):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0
        return centroids, boundaries

    class TurboQuantMSE:
        def __init__(self, bit_width, head_dim, device, rotation_seed=42):
            d = head_dim
            gen = torch.Generator(device="cpu").manual_seed(rotation_seed)
            G = torch.randn(d, d, generator=gen, dtype=torch.float32)
            Q, R = torch.linalg.qr(G)
            ds = torch.sign(torch.diag(R)); ds[ds == 0] = 1.0
            self.Pi = (Q * ds.unsqueeze(0)).to(device).contiguous()
            sigma = 1.0 / math.sqrt(d)
            c_np, b_np = _lloyd_max_gaussian(2 ** bit_width, sigma=sigma)
            self.centroids = torch.tensor(c_np, dtype=torch.float32, device=device).contiguous()
            self.boundaries = torch.tensor(b_np[1:-1], dtype=torch.float32, device=device).contiguous()
            self.head_dim = head_dim

        @torch.no_grad()
        def quantize(self, x):
            flat = x.float().reshape(-1, self.head_dim)
            norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            y = (flat / norms) @ self.Pi.T
            idx = torch.bucketize(y, self.boundaries).to(torch.uint8)
            return idx.view(x.shape), norms.squeeze(-1).view(x.shape[:-1])

        @torch.no_grad()
        def dequantize(self, idx, norms):
            flat_idx = idx.reshape(-1, self.head_dim)
            y_hat = self.centroids[flat_idx.long()]
            x_hat = y_hat @ self.Pi
            x_hat = x_hat * norms.reshape(-1, 1)
            return x_hat.view(idx.shape)

    # Memoize: every layer with the same (bit_width, dim, seed) builds an
    # IDENTICAL codebook; building it is CPU-bound scipy integration, so share.
    _QUANTIZER_CACHE = {}

    def _get_quantizer(bw, dim, dev, seed=42):
        key = (bw, dim, str(dev), seed)
        q = _QUANTIZER_CACHE.get(key)
        if q is None:
            q = TurboQuantMSE(bw, dim, dev, rotation_seed=seed)
            _QUANTIZER_CACHE[key] = q
        return q

    class TQLayer(DynamicLayer):
        def __init__(self, hd, bw, dev, num_outlier_ch=0, outlier_bw=0):
            super().__init__()
            self._bw = bw
            self._hd = hd
            self._outlier_ch = num_outlier_ch
            self._outlier_bw = outlier_bw
            use_out = num_outlier_ch > 0 and outlier_bw > bw
            self._regular_dim = hd - num_outlier_ch if use_out else hd
            self._outlier_dim = num_outlier_ch if use_out else 0
            self._tq = _get_quantizer(bw, self._regular_dim, dev)
            self._tq_out = _get_quantizer(outlier_bw, num_outlier_ch, dev, seed=43) if self._outlier_dim > 0 else None
            self._key_data, self._val_data = [], []
            self._ck = self._cv = None
            self._channel_mask = None

        def lazy_initialization(self, ks, vs):
            self.dtype, self.device, self.is_initialized = ks.dtype, ks.device, True
            if self._tq_out is not None and self._channel_mask is None:
                rms = ks.float().pow(2).mean(dim=(0, 1, 2)).sqrt()
                _, top = rms.topk(min(self._outlier_ch, rms.shape[0]))
                self._channel_mask = torch.zeros(rms.shape[0], dtype=torch.bool, device=ks.device)
                self._channel_mask[top] = True

        def _quant(self, x):
            shape = x.shape
            if self._tq_out is not None and self._channel_mask is not None:
                reg_mask = ~self._channel_mask
                xf = x.float()
                r = xf[..., reg_mask].reshape(-1, self._regular_dim)
                ri, rn = self._tq.quantize(r)
                o = xf[..., self._channel_mask].reshape(-1, self._outlier_dim)
                oi, on = self._tq_out.quantize(o)
                return {"ri": ri, "rn": rn, "oi": oi, "on": on, "s": shape}
            idx, norms = self._tq.quantize(x.float().reshape(-1, self._hd))
            return {"idx": idx, "norms": norms, "s": shape}

        def _dequant_one(self, d):
            shape = d["s"]
            if "ri" in d:
                r_hat = self._tq.dequantize(d["ri"], d["rn"]).reshape(shape[0], shape[1], shape[2], self._regular_dim)
                o_hat = self._tq_out.dequantize(d["oi"], d["on"]).reshape(shape[0], shape[1], shape[2], self._outlier_dim)
                out = torch.zeros(shape, dtype=torch.float32, device=self.device)
                out[..., ~self._channel_mask] = r_hat
                out[..., self._channel_mask] = o_hat
                return out.to(self.dtype)
            return self._tq.dequantize(d["idx"], d["norms"]).reshape(shape).to(self.dtype)

        def update(self, ks, vs, cache_kwargs=None):
            if not self.is_initialized:
                self.lazy_initialization(ks, vs)
            kd = self._quant(ks); self._key_data.append(kd)
            vd = self._quant(vs); self._val_data.append(vd)
            nk = self._dequant_one(kd)
            nv = self._dequant_one(vd)
            if self._ck is None:
                self._ck, self._cv = nk, nv
            else:
                self._ck = torch.cat([self._ck, nk], dim=-2)
                self._cv = torch.cat([self._cv, nv], dim=-2)
            return self._ck, self._cv

        def get_seq_length(self, *a, **k):
            return sum(d["s"][-2] for d in self._key_data) if self._key_data else 0

        def get_max_cache_shape(self, *a, **k):
            return -1

        def mem_bits(self):
            t = 0
            for d in self._key_data + self._val_data:
                if "ri" in d:
                    t += d["ri"].numel() * self._bw + d["oi"].numel() * self._outlier_bw
                    t += (d["rn"].numel() + d["on"].numel()) * 32
                else:
                    t += d["idx"].numel() * self._bw + d["norms"].numel() * 32
            return t

        def eff_bits(self):
            if self._outlier_dim > 0:
                return (self._regular_dim * self._bw + self._outlier_dim * self._outlier_bw) / self._hd
            return float(self._bw)

        @property
        def keys(self):
            return self._ck if self._ck is not None else torch.tensor([])

        @keys.setter
        def keys(self, v):
            pass

        @property
        def values(self):
            return self._cv if self._cv is not None else torch.tensor([])

        @values.setter
        def values(self, v):
            pass

    class TQCache(DynamicCache):
        def __init__(self, hd, bw, nl, dev, num_outlier_ch=0, outlier_bw=0):
            super().__init__()
            self.layers = [TQLayer(hd, bw, dev, num_outlier_ch, outlier_bw) for _ in range(nl)]

        def mem_bits(self):
            return sum(l.mem_bits() for l in self.layers)

        def eff_bits(self):
            return self.layers[0].eff_bits() if self.layers else 0.0

    return TQCache


def _baseline_kv_bytes(cache):
    """FP16 KV-cache size in bytes from a standard DynamicCache."""
    t = 0
    for l in cache.layers:
        for attr in ("keys", "values"):
            x = getattr(l, attr, None)
            if x is not None and hasattr(x, "numel") and x.numel() > 0:
                t += x.numel() * x.element_size()
    return t


# A long-ish context with a planted fact (needle) + a summary ask, wrapped in
# the LCLM memory tags so the encoder compresses it into latent tokens.
_DOC = """\
The Aurora Initiative is a multi-year research program coordinated across six
national laboratories. Its first phase, completed in 2021, focused on materials
discovery for solid-state batteries. The second phase pivoted toward
high-altitude atmospheric sensing using autonomous balloons. Internal reviews
noted that the program's defining constraint was always thermal management, not
compute. During the third phase, the team relocated its primary testbed to a
decommissioned observatory in the Atacama desert. The lead engineer for the
sensing payload was Dr. Mara Velez, who had previously designed cryogenic
cooling loops for space telescopes. IMPORTANT FACT: the calibration passphrase
for the Atacama testbed telemetry uplink is "violet-harbor-1987". The fourth
phase, planned for 2026, will integrate the atmospheric models with regional
climate forecasting. Funding for the program comes from a consortium of public
grants and three private foundations, with annual budgets reviewed every spring.
"""


def _build_prompt():
    needle = "Answer two things: (1) What is the exact calibration passphrase " \
             "for the Atacama testbed telemetry uplink? (2) In one sentence, " \
             "what was the program's defining constraint?"
    return f"<|memory_start|>{_DOC}<|memory_end|>\n\n{needle}"


@app.function(
    image=image,
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": hf_cache},
    timeout=3600,
)
def run(checkpoint: str = DEFAULT_CHECKPOINT, bit_width: int = 4,
        max_new_tokens: int = 200, outlier_channels: int = 0, outlier_bits: int = 0,
        prompt: str = ""):
    import sys, os, time, gc
    sys.path.insert(0, "/LCLM")
    import torch

    os.environ.setdefault("HF_HOME", "/cache/hf")
    device = "cuda"
    prompt = prompt or _build_prompt()

    print("=== LCLM + TurboQuant POC ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)
    import transformers
    print(f"transformers {transformers.__version__}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"TurboQuant bit_width={bit_width} outlier_channels={outlier_channels} outlier_bits={outlier_bits}", flush=True)

    from inference.hf import load_model  # noqa: E402

    print("\nLoading LCLM (downloads to /cache volume if not present)...", flush=True)
    model, decoder_tokenizer, processor = load_model(checkpoint, device=device, dtype="bf16")
    hf_cache.commit()  # persist downloaded weights to the volume

    dcfg = model.decoder.config
    hd = getattr(dcfg, "head_dim", None) or (dcfg.hidden_size // dcfg.num_attention_heads)
    nl = dcfg.num_hidden_layers
    nh = getattr(dcfg, "num_key_value_heads", dcfg.num_attention_heads)
    print(f"Decoder: {dcfg.architectures} | {nl} layers, {nh} KV heads, head_dim={hd}", flush=True)

    TQCache = _build_tq_classes()

    # ----- shared processing: turn the prompt into decoder inputs + latents -----
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    processed = processor.process_wrapped_batch(
        prompts=[formatted], targets=None, padding="longest",
        truncation=True, return_tensors="pt",
    )
    input_ids = processed["input_ids"].to(device)
    attention_mask = processed["attention_mask"].to(device)
    memory_positions = processed["memory_positions"]
    latent_counts = processed["latent_counts"]
    memory_token_ids = processed["memory_token_ids"]
    n_latents = sum(sum(c) if isinstance(c, list) else c for c in latent_counts)
    print(f"Prompt tokens (decoder, incl. latents): {input_ids.shape[1]} | latent tokens: {n_latents}", flush=True)

    def generate(cache):
        torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=decoder_tokenizer.pad_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id,
        )
        if cache is not None:
            gen_kwargs["past_key_values"] = cache
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                memory_token_ids=memory_token_ids,
                memory_positions=memory_positions,
                latent_counts=latent_counts,
                **gen_kwargs,
            )
        torch.cuda.synchronize()
        dt = time.time() - t0
        # With inputs_embeds, generate() returns only newly generated tokens.
        gen_ids = out.sequences[0]
        if cache is None:
            kv_bytes = _baseline_kv_bytes(out.past_key_values)
            eff = 16.0
        else:
            kv_bytes = out.past_key_values.mem_bits() // 8
            eff = out.past_key_values.eff_bits()
        return {
            "text": decoder_tokenizer.decode(gen_ids, skip_special_tokens=True),
            "tokens": int(gen_ids.shape[0]),
            "time_s": round(dt, 2),
            "tps": round(int(gen_ids.shape[0]) / dt, 1) if dt > 0 else 0,
            "kv_bytes": int(kv_bytes),
            "eff_bits": round(float(eff), 2),
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        }

    print("\n--- Vanilla LCLM (FP16 KV) ---", flush=True)
    baseline = generate(None)
    print(f"  {baseline['tokens']} tok, {baseline['tps']} tok/s, KV={baseline['kv_bytes']} B", flush=True)

    print(f"\n--- LCLM + TurboQuant {bit_width}-bit ---", flush=True)
    tq_cache = TQCache(hd, bit_width, nl, device, outlier_channels, outlier_bits)
    tq = generate(tq_cache)
    print(f"  {tq['tokens']} tok, {tq['tps']} tok/s, KV={tq['kv_bytes']} B, eff_bits={tq['eff_bits']}", flush=True)

    ratio = baseline["kv_bytes"] / tq["kv_bytes"] if tq["kv_bytes"] else float("inf")
    result = {
        "checkpoint": checkpoint, "prompt": prompt,
        "config": {"layers": nl, "kv_heads": nh, "head_dim": hd, "decoder_prompt_tokens": int(input_ids.shape[1]), "latent_tokens": int(n_latents)},
        "baseline": baseline, "turboquant": tq,
        "decoder_kv_compression_ratio": round(ratio, 2),
    }

    print("\n" + "=" * 78, flush=True)
    print(f"PROMPT:\n{prompt}", flush=True)
    print("-" * 78, flush=True)
    print(f"[VANILLA LCLM]\n{baseline['text']}", flush=True)
    print("-" * 78, flush=True)
    print(f"[LCLM + TURBOQUANT {tq['eff_bits']}-bit]\n{tq['text']}", flush=True)
    print("=" * 78, flush=True)
    print(f"Decoder KV-cache compression (TurboQuant on top of LCLM latents): "
          f"{ratio:.2f}x  ({baseline['kv_bytes']} B -> {tq['kv_bytes']} B)", flush=True)
    print("=" * 78, flush=True)
    return result


@app.local_entrypoint()
def main(checkpoint: str = DEFAULT_CHECKPOINT, bit_width: int = 4,
         max_new_tokens: int = 200, outlier_channels: int = 0, outlier_bits: int = 0,
         prompt: str = ""):
    r = run.remote(checkpoint, bit_width, max_new_tokens, outlier_channels, outlier_bits, prompt)
    print("\n========== RESULT (local) ==========")
    print(f"Checkpoint: {r['checkpoint']}")
    c = r["config"]
    print(f"Decoder: {c['layers']} layers, {c['kv_heads']} KV heads, head_dim={c['head_dim']}")
    print(f"Decoder prompt tokens: {c['decoder_prompt_tokens']} (of which {c['latent_tokens']} latent)")
    print(f"Decoder KV compression: {r['decoder_kv_compression_ratio']}x | TQ eff_bits: {r['turboquant']['eff_bits']}")
    print(f"Vanilla: {r['baseline']['tps']} tok/s | +TurboQuant: {r['turboquant']['tps']} tok/s")
    print(f"\n[Vanilla output]\n{r['baseline']['text']}")
    print(f"\n[TurboQuant output]\n{r['turboquant']['text']}")
