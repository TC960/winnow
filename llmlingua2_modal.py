"""
LLMLingua-2 (XLM-RoBERTa-large) on Modal.

First step toward a custom compression model: get the released checkpoint
running on a GPU, with weights baked into the image so cold starts are fast.

Usage:
    pip install modal
    python -m modal setup           # one-time auth
    modal run llmlingua2_modal.py                       # runs the demo
    modal run llmlingua2_modal.py --rate 0.33           # keep ~33% of tokens
    modal run llmlingua2_modal.py --text "your text..." --rate 0.5
"""

import modal

MODEL_NAME = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

# Persistent HF cache. Weights are downloaded into this volume once and read
# from it on every subsequent run — never re-downloaded, even across image
# rebuilds. create_if_missing=True means the first run provisions it automatically.
CACHE_DIR = "/cache"
hf_cache_vol = modal.Volume.from_name("llmlingua2-hf-cache", create_if_missing=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("llmlingua", "torch", "huggingface_hub", "hf_transfer")
    # Point Hugging Face at the mounted volume so snapshot_download and
    # PromptCompressor both read/write the standard HF cache layout there.
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("llmlingua2-xlm", image=image)


@app.cls(
    gpu="T4",
    volumes={CACHE_DIR: hf_cache_vol},
    # Keep a warmed container alive 30 min after the last request so it stays hot
    # through a demo (between questions) without re-warming. Auto-scales to zero
    # afterward — no lingering cost.
    scaledown_window=1800,
    # Memory snapshots: capture the fully-loaded model (incl. GPU memory) so that
    # future cold starts RESTORE that state instead of re-loading the model.
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},  # alpha: snapshot GPU memory too
)
class Compressor:
    @modal.enter(snap=True)
    def load(self):
        # Runs only when CREATING the snapshot — i.e. the very first cold start,
        # or after a code/image change invalidates the existing snapshot. The
        # loaded model and its GPU memory are captured here; every later cold
        # start restores this state directly and skips all of this work.
        #
        # Populate-once-then-read: downloads only if the volume is empty,
        # otherwise this resolves straight from the cached volume.
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL_NAME)
        hf_cache_vol.commit()  # persist any newly downloaded files to the volume

        from llmlingua import PromptCompressor

        self.compressor = PromptCompressor(
            model_name=MODEL_NAME,
            use_llmlingua2=True,
            device_map="cuda",
        )

    @modal.method()
    def compress(self, text: str, rate: float = 0.5, return_labels: bool = False):
        return self.compressor.compress_prompt(
            text,
            rate=rate,
            force_tokens=["\n", ".", "!", "?", ","],
            force_reserve_digit=True,      # don't drop digits (numbers stay intact)
            drop_consecutive=True,
            return_word_label=return_labels,  # per-word keep/discard labels
        )


SAMPLE = (
    "The committee convened on Tuesday to review the proposed budget allocations "
    "for the upcoming fiscal year. After a lengthy discussion that touched on "
    "several departmental requests, the members agreed to defer the final vote "
    "until additional documentation could be gathered from the finance office, "
    "which had not yet submitted its quarterly projections."
)


@app.local_entrypoint()
def main(text: str = SAMPLE, rate: float = 0.5, labels: bool = False):
    out = Compressor().compress.remote(text, rate=rate, return_labels=labels)
    print("\n=== compressed ===")
    print(out["compressed_prompt"])
    print("\n=== stats ===")
    print(f"origin tokens:     {out['origin_tokens']}")
    print(f"compressed tokens: {out['compressed_tokens']}")
    print(f"ratio:             {out['ratio']}")
    print(f"rate (kept):       {out['rate']}")