"""Trace mode: turn-level context compression on top of winnow's LLMLingua-2 path.

Public surface:
  core        the packer, Store, recall/rehydrate, render_history
  embed       local MiniLM embedder (get_embedder)
  summarize   SUMMARIZE tier wired to the Modal LLMLingua-2 worker
  strategies  the Compressor interface + NoOp / Truncate / NaiveSummarize / OurPolicy
"""

from .core import (
    Action,
    Hit,
    PackPlan,
    Record,
    Store,
    Turn,
    pack,
    recall,
    rehydrate,
    render_history,
)
from .strategies import Compressor, NaiveSummarize, NoOp, OurPolicy, Truncate

__all__ = [
    "Action",
    "Hit",
    "PackPlan",
    "Record",
    "Store",
    "Turn",
    "pack",
    "recall",
    "rehydrate",
    "render_history",
    "Compressor",
    "NoOp",
    "Truncate",
    "NaiveSummarize",
    "OurPolicy",
]
