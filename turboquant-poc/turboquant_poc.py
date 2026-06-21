"""
turboquant_poc — minimal, self-contained TurboQuant KV-cache compression POC.

Implements just enough to run text generation with a TurboQuant-compressed
KV cache through HuggingFace `transformers`:

  * TurboQuantMSE  — Algorithm 1 from "TurboQuant: Online Vector Quantization
                     with Near-optimal Distortion Rate" (Zandieh et al.,
                     ICLR 2026, arXiv:2504.19874). A random-rotation +
                     Lloyd-Max scalar quantizer, fully torch / CUDA-capable.
  * TQLayer        — a `transformers.cache_utils.DynamicLayer` subclass that
                     stores per-layer KV state in TurboQuant-quantized form.
  * TQCache        — a `transformers.cache_utils.DynamicCache` subclass usable
                     directly as `past_key_values` in `model.generate(...)`.

This is a POC (prompt -> generate -> output), not the full benchmark suite.
It is based on the known-working inline implementation in
`benchmarks/gpu.py` (the `BENCHMARK_SCRIPT`), which targets the *current*
`transformers` cache API (`DynamicLayer.update`, `lazy_initialization`, the
`keys`/`values` property pattern). Differences from the `turboquant` package
version are intentional — see module notes at the bottom.

Pure torch: no triton, no custom CUDA, no bit-packing. The codebook tables are
computed once with numpy/scipy at construction time; everything on the hot path
is torch and runs on whatever device the KV tensors live on (CPU or CUDA).

Imports fine on CPU; runs on CUDA when tensors are on CUDA.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm
from transformers.cache_utils import DynamicCache, DynamicLayer


# ---------------------------------------------------------------------------
# Lloyd-Max codebook for a zero-mean Gaussian
# ---------------------------------------------------------------------------
def _lloyd_max_gaussian(num_levels: int, sigma: float = 1.0, max_iter: int = 200):
    """
    Optimal Lloyd-Max scalar quantizer (centroids + boundaries) for N(0, sigma^2).

    Each coordinate of a randomly rotated unit vector is ~ N(0, 1/d) in high
    dimension (paper Section 3.1), so we solve the 1-D k-means for that Gaussian.

    Returns:
        centroids:  (num_levels,) sorted centroid values
        boundaries: (num_levels + 1,) sorted decision boundaries incl. +/- inf
    """
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


# ---------------------------------------------------------------------------
# TurboQuant_mse (Algorithm 1)
# ---------------------------------------------------------------------------
class TurboQuantMSE:
    """
    MSE-optimal vector quantizer.

      1. Rotate input by a fixed random orthogonal matrix Pi:  y = x @ Pi^T
      2. Scalar-quantize each coordinate of y against the Lloyd-Max codebook
      3. Dequantize: look up centroids, rotate back:  x_hat = y_hat @ Pi

    `quantize` normalizes each row to unit length and returns the per-vector
    norms separately, so the codebook (calibrated for unit-norm Gaussian
    coordinates) is reused across vectors of any magnitude.

    Args:
        bit_width:     bits per coordinate; codebook has 2**bit_width levels.
        head_dim:      dimension d of each vector to quantize.
        device:        torch device the rotation / codebook tensors live on.
        rotation_seed: seed for the (CPU-generated) random rotation matrix.
    """

    def __init__(self, bit_width: int, head_dim: int,
                 device: torch.device | str = "cpu", rotation_seed: int = 42):
        d = head_dim
        device = torch.device(device)

        # Random orthogonal rotation via QR of a Gaussian matrix (seed on CPU
        # for cross-device reproducibility), with sign-fixed columns.
        gen = torch.Generator(device="cpu").manual_seed(rotation_seed)
        G = torch.randn(d, d, generator=gen, dtype=torch.float32)
        Q, R = torch.linalg.qr(G)
        ds = torch.sign(torch.diag(R))
        ds[ds == 0] = 1.0
        self.Pi = (Q * ds.unsqueeze(0)).to(device).contiguous()

        sigma = 1.0 / math.sqrt(d)
        c_np, b_np = _lloyd_max_gaussian(2 ** bit_width, sigma=sigma)
        self.centroids = torch.tensor(c_np, dtype=torch.float32, device=device).contiguous()
        # interior boundaries only (drop the -inf / +inf endpoints) for bucketize
        self.boundaries = torch.tensor(b_np[1:-1], dtype=torch.float32, device=device).contiguous()

        self.bit_width = bit_width
        self.head_dim = head_dim
        self.device = device

    @torch.no_grad()
    def quantize(self, x: torch.Tensor):
        """
        Args:
            x: (..., head_dim) tensor.

        Returns:
            idx:   tensor of uint8 codebook indices, same shape as `x`.
            norms: per-vector L2 norms, shape `x.shape[:-1]`.
        """
        flat = x.float().reshape(-1, self.head_dim)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        y = (flat / norms) @ self.Pi.T
        idx = torch.bucketize(y, self.boundaries).to(torch.uint8)
        return idx.view(x.shape), norms.squeeze(-1).view(x.shape[:-1])

    @torch.no_grad()
    def dequantize(self, idx: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx:   uint8 codebook indices (..., head_dim).
            norms: per-vector norms, shape idx.shape[:-1].

        Returns:
            x_hat: reconstructed tensor, same shape as `idx`.
        """
        flat_idx = idx.reshape(-1, self.head_dim)
        y_hat = self.centroids[flat_idx.long()]
        x_hat = y_hat @ self.Pi
        x_hat = x_hat * norms.reshape(-1, 1)
        return x_hat.view(idx.shape)

    @torch.no_grad()
    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Round-trip helper: quantize then dequantize back to `x`'s dtype/shape."""
        idx, norms = self.quantize(x)
        return self.dequantize(idx, norms).to(x.dtype)


# ---------------------------------------------------------------------------
# Per-layer TurboQuant KV storage
# ---------------------------------------------------------------------------
class TQLayer(DynamicLayer):
    """
    DynamicLayer that stores KV state in TurboQuant-quantized form.

    On `update` it quantizes the incoming key/value states, appends them to the
    per-layer history, dequantizes back to float, and returns the full
    (dequantized) key/value tensors so standard HuggingFace attention works
    unchanged.

    Optional per-channel outlier handling (paper Section 4.3): the
    `num_outlier_channels` channels with the largest RMS magnitude (measured on
    the first key tensor) are quantized at `outlier_bits` precision, the rest at
    `bit_width`. Outlier handling is only enabled when
    `num_outlier_channels > 0 and outlier_bits > bit_width`.

    KV tensor shape convention: (batch, num_heads, seq_len, head_dim).
    """

    def __init__(self, head_dim: int, bit_width: int,
                 device: torch.device | str = "cpu",
                 num_outlier_channels: int = 0, outlier_bits: int = 0):
        super().__init__()
        device = torch.device(device)
        self._bw = bit_width
        self._hd = head_dim
        self._outlier_ch = num_outlier_channels
        self._outlier_bw = outlier_bits

        use_outliers = num_outlier_channels > 0 and outlier_bits > bit_width
        self._regular_dim = head_dim - num_outlier_channels if use_outliers else head_dim
        self._outlier_dim = num_outlier_channels if use_outliers else 0

        self._tq = TurboQuantMSE(bit_width, self._regular_dim, device)
        self._tq_out = (
            TurboQuantMSE(outlier_bits, num_outlier_channels, device, rotation_seed=43)
            if self._outlier_dim > 0 else None
        )

        self._key_data: list[dict] = []
        self._val_data: list[dict] = []
        self._ck: torch.Tensor | None = None
        self._cv: torch.Tensor | None = None
        self._channel_mask: torch.Tensor | None = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True
        if self._tq_out is not None and self._channel_mask is None:
            rms = key_states.float().pow(2).mean(dim=(0, 1, 2)).sqrt()
            _, top = rms.topk(min(self._outlier_ch, rms.shape[0]))
            self._channel_mask = torch.zeros(rms.shape[0], dtype=torch.bool, device=key_states.device)
            self._channel_mask[top] = True

    def _quant(self, x: torch.Tensor) -> dict:
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

    def _dequant_one(self, d: dict) -> torch.Tensor:
        shape = d["s"]
        if "ri" in d:
            r_hat = self._tq.dequantize(d["ri"], d["rn"]).reshape(
                shape[0], shape[1], shape[2], self._regular_dim)
            o_hat = self._tq_out.dequantize(d["oi"], d["on"]).reshape(
                shape[0], shape[1], shape[2], self._outlier_dim)
            out = torch.zeros(shape, dtype=torch.float32, device=self.device)
            out[..., ~self._channel_mask] = r_hat
            out[..., self._channel_mask] = o_hat
            return out.to(self.dtype)
        return self._tq.dequantize(d["idx"], d["norms"]).reshape(shape).to(self.dtype)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               cache_kwargs: dict | None = None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        kd = self._quant(key_states)
        self._key_data.append(kd)
        vd = self._quant(value_states)
        self._val_data.append(vd)

        nk = self._dequant_one(kd)
        nv = self._dequant_one(vd)
        if self._ck is None:
            self._ck, self._cv = nk, nv
        else:
            self._ck = torch.cat([self._ck, nk], dim=-2)
            self._cv = torch.cat([self._cv, nv], dim=-2)
        return self._ck, self._cv

    def get_seq_length(self, *args, **kwargs) -> int:
        return sum(d["s"][-2] for d in self._key_data) if self._key_data else 0

    def get_max_cache_shape(self, *args, **kwargs) -> int:
        return -1

    def mem_bits(self) -> int:
        """Total bits stored for this layer (quantized indices + fp32 norms)."""
        t = 0
        for d in self._key_data + self._val_data:
            if "ri" in d:
                t += d["ri"].numel() * self._bw + d["oi"].numel() * self._outlier_bw
                t += (d["rn"].numel() + d["on"].numel()) * 32
            else:
                t += d["idx"].numel() * self._bw + d["norms"].numel() * 32
        return t

    def eff_bits(self) -> float:
        """Effective bits per stored value (index bits only, ignoring norms)."""
        if self._outlier_dim > 0:
            return (self._regular_dim * self._bw + self._outlier_dim * self._outlier_bw) / self._hd
        return float(self._bw)

    @property
    def keys(self):
        return self._ck if self._ck is not None else torch.tensor([])

    @keys.setter
    def keys(self, value):
        pass

    @property
    def values(self):
        return self._cv if self._cv is not None else torch.tensor([])

    @values.setter
    def values(self, value):
        pass


# ---------------------------------------------------------------------------
# Full cache
# ---------------------------------------------------------------------------
class TQCache(DynamicCache):
    """
    DynamicCache built from TurboQuant-compressed layers. Drop-in
    `past_key_values` for `model.generate(...)`.

    Example:
        hd = model.config.hidden_size // model.config.num_attention_heads
        nl = model.config.num_hidden_layers
        cache = TQCache(head_dim=hd, bit_width=3, num_layers=nl, device="cuda")
        out = model.generate(**inputs, past_key_values=cache, use_cache=True)
        print(cache.mem_bits() // 8, "bytes,", cache.eff_bits(), "eff bits/val")

    Args:
        head_dim:             per-head dimension d.
        bit_width:            base bits per coordinate.
        num_layers:           number of transformer layers.
        device:               device for codebook / rotation tensors.
        num_outlier_channels: channels per head treated as outliers (0 = off).
        outlier_bits:         bits for outlier channels (must exceed bit_width
                              to take effect).
    """

    def __init__(self, head_dim: int, bit_width: int, num_layers: int,
                 device: torch.device | str = "cpu",
                 num_outlier_channels: int = 0, outlier_bits: int = 0):
        super().__init__()
        self.head_dim = head_dim
        self.bit_width = bit_width
        self.layers = [
            TQLayer(head_dim, bit_width, device, num_outlier_channels, outlier_bits)
            for _ in range(num_layers)
        ]

    def mem_bits(self) -> int:
        """Total bits stored across all layers."""
        return sum(layer.mem_bits() for layer in self.layers)

    def eff_bits(self) -> float:
        """Effective bits per stored value (from the first layer's config)."""
        return self.layers[0].eff_bits() if self.layers else float(self.bit_width)


__all__ = ["TurboQuantMSE", "TQLayer", "TQCache"]


# ---------------------------------------------------------------------------
# Notes on the source of truth
# ---------------------------------------------------------------------------
# This module follows the inline implementation in benchmarks/gpu.py
# (BENCHMARK_SCRIPT), which is the version verified to work on GPU with the
# current transformers cache API. The notable differences vs. the `turboquant`
# package (turboquant/core.py + turboquant/cache.py):
#
#   1. Normalization location. Package `TurboQuantMSE.quantize` assumes unit
#      vectors and returns only indices; the cache layer normalizes externally.
#      Here (and in gpu.py) `TurboQuantMSE.quantize` normalizes internally and
#      returns (idx, norms), with `dequantize(idx, norms)` re-applying them.
#      This keeps the quantizer self-contained and matches the working script.
#
#   2. Cache accumulation. The package's TurboQuantLayer re-dequantizes the
#      entire history on every access via a dirty flag. The gpu.py version
#      (used here) caches dequantized K/V and concatenates only the newly added
#      step on each `update`, which is what runs on GPU during generation.
#
#   3. Reporting. The package exposes get_memory_bytes()/get_effective_bits();
#      this module exposes the gpu.py names `mem_bits()` (TQCache + TQLayer) and
#      `eff_bits()` as requested.
#
#   4. Bit-packing (turboquant/packing.py) is intentionally omitted — it is not
#      needed for a functional generate() POC, and mem_bits() already reports
#      the true logical bit count.
