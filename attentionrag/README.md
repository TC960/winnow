# AttentionRAG

A faithful, runnable implementation of **AttentionRAG: Attention-Guided Context
Pruning in Retrieval-Augmented Generation** (Fang, Sun, Shi, Gu — arXiv:2503.10720).

AttentionRAG compresses retrieved RAG context by reformulating the query into an
incomplete-answer template whose single blank ("focal token") does next-token
prediction over each context chunk. The focal token's attention over the chunk
(summed across all layers) ranks tokens; the **sentences** containing the top-k
tokens are kept. Irrelevant chunks are dropped when the model predicts `none`.

```
query ──(B.1)──▶ hint prefix ("Daniel is in the ___")
context ──chunk(m)──▶ c_1 … c_n
   per chunk c_j:
      predict anchor token a_j  ──▶  a_j == "none"?  ── yes ─▶ drop chunk
                                          │ no
      A_j = Σ_l Attention_l(c_j, a_j)     │   (Eq. 2, all layers, head-avg)
      keep sentences holding Top-k(A_j)   ▼   (Eq. 3)
compressed context C' ──▶ generation model ──▶ answer
```

## Files

| File | What |
|---|---|
| `prompts.py` | Verbatim paper prompts: B.1 (hint prefix), B.2 (fixed fallback prefix), B.3 (anchor prompt) + segment builder for exact context-token positions. |
| `core.py` | Backend-agnostic pipeline (Algorithm 1): sentence spans, **Eq. 3** top-k→sentence selection, `none`-gating, orchestration. No torch. |
| `hf_backend.py` | Causal-LM backend: chunking, anchor-token prediction, **Eq. 1/2** all-layer attention extraction (requires `attn_implementation="eager"`), answer generation. |
| `modal_app.py` | Modal GPU app (Qwen-2.5-7B-Instruct on A10G, shared `hf-cache` volume). |
| `test_core.py` | GPU-free unit tests using a deterministic mock backend. |

## Run

**Local logic tests (no GPU, no torch):**
```bash
python -m attentionrag.test_core      # from repo root  -> 10/10 passed
```

**End-to-end on Modal GPU (detached, per repo convention):**
```bash
python -m modal setup                                  # one-time auth
modal run --detach attentionrag/modal_app.py           # built-in bAbI + HotpotQA demos
modal run --detach attentionrag/modal_app.py \
    --question "Where is Daniel?" --context-file mycontext.txt \
    --chunk-size 60 --top-k 10
```

The first run downloads Qwen-2.5-7B into the `hf-cache` volume (~15 GB, once);
later runs read it from the volume. Each run prints the hint prefix, per-chunk
anchor tokens (with `*` = dropped), the compressed context, the compression
ratio, and the answer from both the compressed and the full context.

## Hyperparameters

`chunk_size` (`m`) and `top_k` (`k`) are the only compression knobs — exactly as
in the paper (no fixed token budget; the ratio is content-adaptive). Paper
guidance: larger `m` / higher `k` for long LongBench contexts (e.g. m=300,
k=12); smaller `m` / lower `k` for short, sparse BABILong-style contexts
(e.g. m=50, k=8). The demo defaults to m=60, k=10 for the short demo contexts.

## Fidelity to the paper

Implemented as described:
- **B.1/B.2/B.3 prompts** transcribed verbatim.
- **Single focal token** via next-token-prediction reformulation; one anchor
  token generated per chunk.
- **`none`-gating** relevance skip (Algorithm 1).
- **Eq. 2**: attention summed over **all** layers (head-averaged) for the anchor
  token over the chunk's context tokens.
- **Eq. 3**: top-k tokens promoted to their full sentences; sentences
  concatenated.
- **Two-pass attention read** keyed on the anchor token `a_j` (not the prefix's
  last token): pass 1 predicts `a_j`; pass 2 appends `a_j` and reads its
  attention over the context (via KV-cache, so only one extra token is encoded).

Deliberate, documented choices:
- **Hint-prefix author**: the paper uses GPT-4o Mini. By default we use the same
  local model (self-contained, no API key). Set `use_openai_hint=True` +
  `OPENAI_API_KEY` in `HFBackend` to match the paper exactly.
- **Backbone**: Qwen-2.5-7B-Instruct (one of the paper's models; ungated).
  Llama-3.1-8B-Instruct also works — change `MODEL_NAME` (gated; needs HF token).
- **Head aggregation**: the paper's Eq. 2 sums over layers but does not specify
  head reduction; we average over heads (standard), then sum over layers.
- **Final-answer prompt** is a standard LongBench-style RAG prompt (the paper's
  appendix specifies the anchor prompt, not the answer prompt).
