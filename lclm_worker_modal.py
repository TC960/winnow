"""
LCLM + TurboQuant text generation on Modal (warm A100-80GB worker).

Mirrors turboquant_modal.py: a `@app.cls` loads the model ONCE per container in
`@modal.enter`, and the container stays warm (`scaledown_window`). So when
server.py calls `.generate`, it hits an already-loaded model with no startup
cost.

LCLM = encoder (Qwen3-Embedding-0.6B) -> adapter -> decoder
(Qwen3-4B-Instruct-2507; `Qwen3ForCausalLM`, 36 layers, GQA, 8 KV heads,
head_dim=128). The encoder compresses a `<|memory_start|>...<|memory_end|>` text
block into a few latent soft tokens; the decoder consumes them as input
embeddings and generates normally.

  * LCLM compresses the *sequence length* of the KV cache (fewer entries).
  * TurboQuant compresses the *bits per entry* of the decoder KV cache.

The two are orthogonal -> the savings multiply. `LCLM.generate` bottoms out in a
plain `decoder.generate(inputs_embeds=..., **generation_kwargs)`, so our TQCache
(a DynamicCache subclass) drops straight in as the decoder's `past_key_values`.
TurboQuant only applies to the decoder; the encoder runs a single prefill.

The released checkpoints `latent-context/0.6b-4b-LCLM-{4x,8x,16x}` (~18.5 GB
each) are already cached on the `turboquant-hf-cache` volume (HF_HOME=/cache/hf).

Usage:
    pip install modal
    modal deploy lclm_worker_modal.py                          # deploy warm worker
    modal run lclm_worker_modal.py --prompt "..." --context "..."   # one-off test
"""

import modal

# Released LCLM checkpoints (Apache-2.0). The server pins 16x by default; each is
# ~18.5 GB and already cached on the volume. Swap here to serve a different one.
DEFAULT_CHECKPOINT = "latent-context/0.6b-4b-LCLM-16x"
LCLM_REPO = "https://github.com/LeonLixyz/LCLM.git"

CACHE_DIR = "/cache"
hf_cache_vol = modal.Volume.from_name("turboquant-hf-cache", create_if_missing=True)

# Image recipe is the proven one from turboquant-poc/lclm_modal.py: clone the
# LCLM repo so `inference`, `latent_context`, `utils` are importable. No
# flash-attn -> the loader auto-falls back to sdpa, which is what TQCache needs.
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
    .env({"HF_HOME": f"{CACHE_DIR}/hf"})
)

app = modal.App("lclm-turboquant-worker", image=image)

# Default instruction used when the caller passes a bare prompt with no explicit
# memory block and no separate context (we wrap the whole prompt as the memory).
_DEFAULT_INSTRUCTION = "Using the context above, answer concisely and accurately."


# --- TurboQuant KV-cache classes (built inside the container; deferred imports
#     keep this module importable without torch on the server side) --------------
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


@app.cls(
    gpu="A100-80GB",
    volumes={CACHE_DIR: hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    # Stay warm 30 min after the last request so a demo stays hot between
    # /generate calls without re-loading. Auto-scales to zero afterward.
    scaledown_window=1800,
    timeout=3600,
)
class LCLMTurboQuantModel:
    @modal.enter()
    def load(self):
        """Runs once per container start. Loads the LCLM checkpoint + pre-builds
        the TurboQuant codebooks so the first /generate has zero startup cost."""
        import os
        import sys
        import torch

        sys.path.insert(0, "/LCLM")
        os.environ.setdefault("HF_HOME", f"{CACHE_DIR}/hf")
        from inference.hf import load_model

        self.torch = torch
        self.checkpoint = DEFAULT_CHECKPOINT
        self.device = "cuda"
        # model, decoder_tokenizer, processor
        self.model, self.tokenizer, self.processor = load_model(
            self.checkpoint, device=self.device, dtype="bf16"
        )
        hf_cache_vol.commit()  # persist any newly downloaded weights

        dcfg = self.model.decoder.config
        self.hd = getattr(dcfg, "head_dim", None) or (
            dcfg.hidden_size // dcfg.num_attention_heads
        )
        self.nl = dcfg.num_hidden_layers
        self.nh = getattr(dcfg, "num_key_value_heads", dcfg.num_attention_heads)

        # Build the cache classes once; the shared-quantizer memo lives in this
        # closure for the container's lifetime. Pre-warm common configs so the
        # CPU-bound Lloyd-Max codebooks are never built on the request path.
        self.TQCache = _build_tq_classes()
        self.TQCache(self.hd, 4, self.nl, self.device)          # 4-bit
        self.TQCache(self.hd, 3, self.nl, self.device)          # 3-bit
        self.TQCache(self.hd, 3, self.nl, self.device, 32, 4)   # 3.5-bit outlier
        print(
            f"LCLM+TurboQuant worker ready: {self.checkpoint} | decoder "
            f"{self.nl} layers, {self.nh} KV heads, head_dim={self.hd}",
            flush=True,
        )

    def _build_lclm_prompt(self, prompt: str, context: str) -> str:
        """Compose the LCLM prompt so the *context* lands inside the memory tags
        (gets compressed into latents) and the *question* stays verbatim.

          * context non-empty -> wrap context as memory, append prompt verbatim.
          * context empty but prompt already has <|memory_start|> -> use as-is.
          * otherwise -> wrap the whole prompt as the memory block w/ default ask.
        """
        if context:
            return f"<|memory_start|>{context}<|memory_end|>\n\n{prompt}"
        if "<|memory_start|>" in prompt:
            return prompt
        return f"<|memory_start|>{prompt}<|memory_end|>\n\n{_DEFAULT_INSTRUCTION}"

    @modal.method()
    def generate(self, prompt: str, bit_width: int = 4, max_new_tokens: int = 256,
                 outlier_channels: int = 0, outlier_bits: int = 0, context: str = ""):
        """Generate with the LCLM (context compressed to latents) and a
        TurboQuant-compressed decoder KV cache. Model is already loaded (see
        load()), so there is no startup cost here. Returns the SAME dict shape as
        turboquant_modal.generate so GenerateResponse(**out) validates."""
        import time
        torch = self.torch

        lclm_prompt = self._build_lclm_prompt(prompt, context)
        formatted = f"<|im_start|>user\n{lclm_prompt}<|im_end|>\n<|im_start|>assistant\n"
        processed = self.processor.process_wrapped_batch(
            prompts=[formatted], targets=None, padding="longest",
            truncation=True, return_tensors="pt",
        )
        input_ids = processed["input_ids"].to(self.device)
        attention_mask = processed["attention_mask"].to(self.device)
        memory_positions = processed["memory_positions"]
        latent_counts = processed["latent_counts"]
        memory_token_ids = processed["memory_token_ids"]
        n_in = int(input_ids.shape[1])

        cache = self.TQCache(
            self.hd, bit_width, self.nl, self.device, outlier_channels, outlier_bits
        )
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            past_key_values=cache,
        )
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                memory_token_ids=memory_token_ids,
                memory_positions=memory_positions,
                latent_counts=latent_counts,
                **gen_kwargs,
            )
        dt = time.time() - t0

        # With inputs_embeds, generate() returns only the newly generated tokens.
        gen = out.sequences[0]
        n_out = int(gen.shape[0])
        kv_bytes = int(out.past_key_values.mem_bits() // 8)
        eff = float(out.past_key_values.eff_bits())
        # Realized decoder sequence length: the decoder's KV cache spans the
        # decoder prompt (incl. latent tokens) plus the generated tokens.
        seq = int(out.past_key_values.get_seq_length())
        fp16_kv = 2 * self.nl * self.nh * seq * self.hd * 2  # K+V, 2 bytes/elem
        return {
            "model": self.checkpoint,
            "text": self.tokenizer.decode(gen, skip_special_tokens=True),
            "input_tokens": n_in,
            "output_tokens": n_out,
            "gen_time_s": round(dt, 2),
            "tokens_per_s": round(n_out / dt, 1) if dt > 0 else 0,
            "eff_bits": round(eff, 2),
            "kv_bytes": kv_bytes,
            "fp16_kv_bytes": int(fp16_kv),
            "kv_compression_x": round(fp16_kv / kv_bytes, 2) if kv_bytes else None,
        }


@app.local_entrypoint()
def main(prompt: str = "What is the calibration passphrase for the Atacama testbed "
                       "telemetry uplink, and what was the program's defining constraint?",
         context: str = "",
         bit_width: int = 4, max_new_tokens: int = 200):
    if not context:
        context = (
            "The Aurora Initiative is a multi-year research program coordinated across "
            "six national laboratories. Internal reviews noted that the program's "
            "defining constraint was always thermal management, not compute. The lead "
            "engineer for the sensing payload was Dr. Mara Velez. IMPORTANT FACT: the "
            "calibration passphrase for the Atacama testbed telemetry uplink is "
            '"violet-harbor-1987". Funding comes from a consortium of public grants '
            "and three private foundations."
        )
    out = LCLMTurboQuantModel().generate.remote(
        prompt, bit_width=bit_width, max_new_tokens=max_new_tokens, context=context
    )
    print("\n=== generation ===")
    print(out["text"])
    print("\n=== stats ===")
    print(f"model:          {out['model']}")
    print(f"tokens:         {out['input_tokens']} in -> {out['output_tokens']} out")
    print(f"eff bits/value: {out['eff_bits']}")
    print(f"KV cache:       {out['kv_bytes']} B  (vs fp16 {out['fp16_kv_bytes']} B "
          f"= {out['kv_compression_x']}x)")
    print(f"throughput:     {out['tokens_per_s']} tok/s")
