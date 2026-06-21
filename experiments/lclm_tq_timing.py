"""
LCLM + TurboQuant: timing & memory benchmark on Modal (A100-80GB).

Compares an LCLM (Latent Context Language Model -- encoder->adapter->decoder soft
-token context compressor, arXiv:2606.09659) running the decoder with:

  * VANILLA fp16 KV cache (`transformers` DynamicCache), vs.
  * TurboQuant KV-cache compression (`TQCache` -- training-free random-rotation +
    Lloyd-Max scalar quant DynamicCache subclass).

The two arms live in TWO SEPARATE warm Modal workers (`LCLMVanilla` and
`LCLMTurboQuant`) so they are isolated -- different containers, no shared CUDA
state, no peak-memory cross-talk. Each loads the LCLM once in `@modal.enter` and
exposes a `generate(...)` method that returns timing + memory + KV stats.

TurboQuant here is PURE-PYTORCH dequant (no custom CUDA kernel). Expect:
  TQ = slower decode tok/s  +  lower peak GPU memory  +  much smaller KV bytes.
The win is MEMORY, not speed (matches the paper, which needs custom kernels for
speed). We report both, honestly.

To get meaningful differences we sweep over realistically long input contexts
(filler docs with a planted needle) -- the longer the context, the more latent
tokens the encoder emits, the larger the decoder KV cache, the bigger the gap.

Deploy:   modal deploy lclm_tq_timing.py
Drive:    python run_lclm_timing.py
"""

import modal

# 16x checkpoint: strongest sequence-length compression (32k input -> ~2k
# latents), so the decoder KV is still big enough to make the bit-width quant
# differences visible. ~18.5 GB, already on the volume.
DEFAULT_CHECKPOINT = "latent-context/0.6b-4b-LCLM-16x"
LCLM_REPO = "https://github.com/LeonLixyz/LCLM.git"

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

app = modal.App("lclm-tq-timing")
hf_cache = modal.Volume.from_name("turboquant-hf-cache", create_if_missing=True)


# ===========================================================================
# TurboQuant core (inlined, identical to turboquant-poc/lclm_modal.py).
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


# ===========================================================================
# Shared QA generation: instruction OUTSIDE memory, context INSIDE memory tags
# (mirrors lclm_worker_modal.generate + the SCHEMA.md contract).
# ===========================================================================
QA_INSTRUCTION = (
    "Answer the question using ONLY the provided context. Answer in as few words "
    "as possible. If the answer is not present, respond with exactly: UNKNOWN."
)


def _generate_qa(self, context, question, max_new_tokens, make_cache, is_vanilla):
    import torch

    lclm_prompt = (
        f"<|memory_start|>{context}<|memory_end|>\n\n{QA_INSTRUCTION}\n\n"
        f"Question: {question}\nAnswer:"
    )
    formatted = f"<|im_start|>user\n{lclm_prompt}<|im_end|>\n<|im_start|>assistant\n"
    processed = self.processor.process_wrapped_batch(
        prompts=[formatted], targets=None, padding="longest",
        truncation=True, return_tensors="pt",
    )
    input_ids = processed["input_ids"].to("cuda")
    attention_mask = processed["attention_mask"].to("cuda")
    memory_positions = processed["memory_positions"]
    latent_counts = processed["latent_counts"]
    memory_token_ids = processed["memory_token_ids"]

    decoder_prompt_tokens = int(input_ids.shape[1])
    n_latents = int(sum(
        sum(c) if isinstance(c, list) else c for c in latent_counts
    ))
    input_context_tokens = len(self.tok.encode(context))

    cache = make_cache()
    gkw = dict(
        max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        return_dict_in_generate=True,
        pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id,
    )
    if cache is not None:
        gkw["past_key_values"] = cache
    with torch.inference_mode():
        out = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            memory_token_ids=memory_token_ids,
            memory_positions=memory_positions,
            latent_counts=latent_counts, **gkw,
        )
    gen_ids = out.sequences[0]
    gen_tokens = int(gen_ids.shape[0])
    text = self.tok.decode(gen_ids, skip_special_tokens=True)

    if is_vanilla:
        kv_bytes = _baseline_kv_bytes(out.past_key_values)
        fp16_kv_bytes = kv_bytes
        kv_compression_x = 1.0
        eff_bits = 16.0
    else:
        kv_bytes = out.past_key_values.mem_bits() // 8
        eff_bits = out.past_key_values.eff_bits()
        seq = decoder_prompt_tokens + gen_tokens
        fp16_kv_bytes = self.num_layers * 2 * self.num_kv_heads * self.head_dim * seq * 2
        kv_compression_x = round(fp16_kv_bytes / kv_bytes, 2) if kv_bytes else None

    return {
        "text": text,
        "input_context_tokens": int(input_context_tokens),
        "latent_tokens": int(n_latents),
        "decoder_prompt_tokens": int(decoder_prompt_tokens),
        "kv_bytes": int(kv_bytes),
        "fp16_kv_bytes": int(fp16_kv_bytes),
        "kv_compression_x": kv_compression_x,
        "eff_bits": round(float(eff_bits), 2),
    }


# ===========================================================================
# Filler-context generator: coherent-ish prose of a target token length with a
# planted needle fact, so we can sweep input length AND verify correctness.
# ===========================================================================
_NEEDLE_FACT = (
    'IMPORTANT FACT: the calibration passphrase for the Atacama testbed '
    'telemetry uplink is "violet-harbor-1987".'
)
# Two-part ask: (1) retrieve the needle (correctness check), then (2) write a long
# free-form summary. The second part forces the decoder to actually generate the
# requested number of tokens, so decode tok/s is meaningful (otherwise the model
# emits the passphrase and immediately hits EOS, leaving nothing to time).
_NEEDLE_QUESTION = (
    "First, state the exact calibration passphrase for the Atacama testbed "
    "telemetry uplink. Then write a long, detailed multi-paragraph summary of the "
    "Aurora Initiative program: its phases, goals, constraints, personnel, "
    "logistics, funding, and engineering challenges. Be thorough and verbose."
)

_FILLER_SENTENCES = [
    "The Aurora Initiative is a multi-year research program coordinated across six national laboratories.",
    "Its first phase focused on materials discovery for solid-state batteries and lasted roughly two years.",
    "The second phase pivoted toward high-altitude atmospheric sensing using fleets of autonomous balloons.",
    "Internal reviews repeatedly noted that the program's defining constraint was thermal management, not compute.",
    "During the third phase the team relocated its primary testbed to a decommissioned observatory in the Atacama desert.",
    "The lead engineer for the sensing payload had previously designed cryogenic cooling loops for space telescopes.",
    "The fourth phase, planned for later years, will integrate the atmospheric models with regional climate forecasting.",
    "Funding for the program comes from a consortium of public grants and three private foundations.",
    "Annual budgets are reviewed every spring by an oversight board drawn from the participating institutions.",
    "Field crews rotate on six-week shifts to manage the harsh conditions at the high-altitude site.",
    "Data from the balloon fleet is downlinked nightly and reconciled against ground-station measurements.",
    "Calibration drift in the optical sensors remains one of the most stubborn engineering challenges.",
    "The team maintains a redundant array of backup batteries to survive extended cloud cover.",
    "Software for the sensing payload is written in a memory-safe systems language for reliability.",
    "Each balloon carries a small suite of instruments measuring temperature, pressure, and aerosols.",
    "The observatory's original dome was retrofitted to house the new telemetry and uplink equipment.",
    "Researchers publish quarterly progress notes that circulate among the consortium members.",
    "A dedicated logistics group coordinates shipments of helium and spare parts to the remote site.",
]


def build_context(target_tokens, tokenizer):
    """Build a filler document of approximately `target_tokens` tokens with the
    needle planted near the middle, plus the wrapped LCLM prompt."""
    parts, planted = [], False
    i = 0
    while True:
        text = " ".join(parts)
        ntok = len(tokenizer.encode(text, add_special_tokens=False)) if text else 0
        if ntok >= target_tokens:
            break
        # Plant the needle near the middle of the target length.
        if not planted and ntok >= target_tokens // 2:
            parts.append(_NEEDLE_FACT)
            planted = True
        parts.append(_FILLER_SENTENCES[i % len(_FILLER_SENTENCES)])
        i += 1
    if not planted:  # tiny target -- ensure the needle is present
        parts.append(_NEEDLE_FACT)
    doc = " ".join(parts)
    ctx_tokens = len(tokenizer.encode(doc, add_special_tokens=False))
    prompt = f"<|memory_start|>{doc}<|memory_end|>\n\n{_NEEDLE_QUESTION}"
    return prompt, ctx_tokens


def _needle_ok(text):
    return "violet-harbor-1987" in text.lower()


# ===========================================================================
# Shared base: load LCLM once, build decoder inputs, time generation.
# ===========================================================================
class _LCLMBase:
    def _load(self, checkpoint):
        import sys, os
        sys.path.insert(0, "/LCLM")
        import torch
        os.environ.setdefault("HF_HOME", "/cache/hf")
        from inference.hf import load_model

        self.checkpoint = checkpoint
        self.model, self.tok, self.processor = load_model(checkpoint, device="cuda", dtype="bf16")
        dcfg = self.model.decoder.config
        self.head_dim = getattr(dcfg, "head_dim", None) or (dcfg.hidden_size // dcfg.num_attention_heads)
        self.num_layers = dcfg.num_hidden_layers
        self.num_kv_heads = getattr(dcfg, "num_key_value_heads", dcfg.num_attention_heads)
        hf_cache.commit()
        print(f"Loaded {checkpoint}: {self.num_layers} layers, "
              f"{self.num_kv_heads} KV heads, head_dim={self.head_dim}", flush=True)

    def _prepare_inputs(self, target_tokens):
        import torch
        prompt, ctx_tokens = build_context(target_tokens, self.tok)
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        processed = self.processor.process_wrapped_batch(
            prompts=[formatted], targets=None, padding="longest",
            truncation=True, return_tensors="pt",
        )
        inp = {
            "input_ids": processed["input_ids"].to("cuda"),
            "attention_mask": processed["attention_mask"].to("cuda"),
            "memory_positions": processed["memory_positions"],
            "latent_counts": processed["latent_counts"],
            "memory_token_ids": processed["memory_token_ids"],
        }
        lc = processed["latent_counts"]
        n_latents = sum(sum(c) if isinstance(c, list) else c for c in lc)
        return inp, int(ctx_tokens), int(inp["input_ids"].shape[1]), int(n_latents)

    def _run(self, target_tokens, max_new_tokens, make_cache):
        """make_cache: () -> (cache_or_None, is_vanilla_bool)."""
        import time, gc
        import torch

        inp, ctx_tokens, prompt_tokens, n_latents = self._prepare_inputs(target_tokens)

        def gen(max_new, cache):
            gc.collect(); torch.cuda.empty_cache()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            gkw = dict(
                max_new_tokens=max_new, do_sample=False, use_cache=True,
                return_dict_in_generate=True,
                pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id,
            )
            if cache is not None:
                gkw["past_key_values"] = cache
            t0 = time.time()
            with torch.inference_mode():
                out = self.model.generate(
                    input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                    memory_token_ids=inp["memory_token_ids"],
                    memory_positions=inp["memory_positions"],
                    latent_counts=inp["latent_counts"], **gkw,
                )
            torch.cuda.synchronize()
            return out, time.time() - t0, torch.cuda.max_memory_allocated()

        # --- TTFT: prefill + 1 token ---
        cache0, _ = make_cache()
        _, ttft_s, _ = gen(1, cache0)

        # --- full decode ---
        cache, is_vanilla = make_cache()
        out, total_s, peak = gen(max_new_tokens, cache)
        gen_ids = out.sequences[0]
        gen_tokens = int(gen_ids.shape[0])
        text = self.tok.decode(gen_ids, skip_special_tokens=True)

        # decode tok/s excludes the prefill+first-token cost captured by ttft.
        # Only meaningful if we actually generated multiple tokens after the first
        # (model can hit EOS early); otherwise report None.
        decode_toks = gen_tokens - 1
        decode_s = total_s - ttft_s
        decode_tps = round(decode_toks / decode_s, 2) if decode_toks >= 2 and decode_s > 0 else None

        if is_vanilla:
            kv_bytes = _baseline_kv_bytes(out.past_key_values)
            fp16_kv_bytes = kv_bytes
            eff_bits = 16.0
        else:
            kv_bytes = out.past_key_values.mem_bits() // 8
            eff_bits = out.past_key_values.eff_bits()
            # fp16 reference for the SAME decoder seq length (prompt+latents+decoded)
            seq = prompt_tokens + gen_tokens
            fp16_kv_bytes = self.num_layers * 2 * self.num_kv_heads * self.head_dim * seq * 2

        return {
            "checkpoint": self.checkpoint,
            "target_context_tokens": int(target_tokens),
            "input_context_tokens": ctx_tokens,
            "decoder_prompt_tokens": prompt_tokens,
            "latent_tokens": n_latents,
            "max_new_tokens": int(max_new_tokens),
            "gen_tokens": gen_tokens,
            "output_text": text,
            "needle_ok": _needle_ok(text),
            "ttft_s": round(ttft_s, 4),
            "total_gen_time_s": round(total_s, 4),
            "decode_tokens_per_s": decode_tps,
            "peak_gpu_mb": round(peak / 1e6, 1),
            "kv_bytes": int(kv_bytes),
            "fp16_kv_bytes": int(fp16_kv_bytes),
            "kv_compression_x": round(fp16_kv_bytes / kv_bytes, 2) if kv_bytes else None,
            "eff_bits": round(float(eff_bits), 2),
        }


@app.cls(
    image=image,
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": hf_cache},
    timeout=3600,
    scaledown_window=600,
)
class LCLMVanilla(_LCLMBase):
    checkpoint: str = modal.parameter(default=DEFAULT_CHECKPOINT)

    @modal.enter()
    def enter(self):
        self._load(self.checkpoint)

    @modal.method()
    def generate(self, target_tokens: int = 8000, max_new_tokens: int = 128):
        return self._run(target_tokens, max_new_tokens, lambda: (None, True))

    @modal.method()
    def generate_qa(self, context: str, question: str,
                    max_new_tokens: int = 48, bit_width: int = 4):
        return _generate_qa(self, context, question, max_new_tokens,
                            make_cache=lambda: None, is_vanilla=True)


@app.cls(
    image=image,
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": hf_cache},
    timeout=3600,
    scaledown_window=600,
)
class LCLMTurboQuant(_LCLMBase):
    checkpoint: str = modal.parameter(default=DEFAULT_CHECKPOINT)

    @modal.enter()
    def enter(self):
        self._load(self.checkpoint)
        # Pre-build TQ classes + warm the codebooks for the configs we sweep so
        # the (CPU-bound scipy) Lloyd-Max integration isn't timed per-request.
        self.TQCache = _build_tq_classes()
        import torch  # noqa: F401
        for bw, oc, ob in [(4, 0, 0), (3, 0, 0), (3, 32, 4)]:
            self.TQCache(self.head_dim, bw, self.num_layers, "cuda", oc, ob)
        print("TurboQuant codebooks pre-built for 4-bit / 3-bit / 3.5-bit.", flush=True)

    @modal.method()
    def generate(self, target_tokens: int = 8000, max_new_tokens: int = 128,
                 bit_width: int = 4, outlier_channels: int = 0, outlier_bits: int = 0):
        def make_cache():
            c = self.TQCache(self.head_dim, bit_width, self.num_layers, "cuda",
                             outlier_channels, outlier_bits)
            return c, False
        return self._run(target_tokens, max_new_tokens, make_cache)

    @modal.method()
    def generate_qa(self, context: str, question: str,
                    max_new_tokens: int = 48, bit_width: int = 4):
        def make_cache():
            return self.TQCache(self.head_dim, bit_width, self.num_layers, "cuda", 0, 0)
        return _generate_qa(self, context, question, max_new_tokens,
                            make_cache=make_cache, is_vanilla=False)
