from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _default_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


@dataclass
class Config:
    """Training/model configuration. Can construct other presets

    This file keeps backwards compatibility with older code that does
    `from config import *` by exporting module-level variables derived from
    a single `cfg` instance.
    """

    # model
    n_emb: int = 768
    T: int = 1024  # context size
    vocab_size: int = 50257  # or 50304 padded up to nearest multiple of 64 for efficiency
    n_layers: int = 12
    n_heads: int = 12
    bias: bool = True  # same value in GPT-2, but False is better
    dropout: float = 0.0  # GPT-2 value
    label_smoothing: float = 0.0
    weight_tying: bool = True

    device: torch.device = field(default_factory=_default_device)
    n_ffd_hidden: int = field(init=False)

    # data
    split: float = 0.9
    epoch_steps: int = 256000  # how many batched blocks to feed
    eval_interval: int = 2000
    batch_size: int = 16

    # optimizer
    lr: float = 1e-4
    lrReductionRatio: float = 2.0

    def __post_init__(self) -> None:
        self.n_ffd_hidden = 4 * self.n_emb

        assert self.epoch_steps % self.batch_size == 0
        assert self.epoch_steps % (self.eval_interval * self.batch_size) == 0


cfg = Config()

# Backwards-compatible module-level exports
n_emb = cfg.n_emb
T = cfg.T
vocab_size = cfg.vocab_size
n_layers = cfg.n_layers
n_heads = cfg.n_heads
n_ffd_hidden = cfg.n_ffd_hidden
bias = cfg.bias
dropout = cfg.dropout
label_smoothing = cfg.label_smoothing
weight_tying = cfg.weight_tying
device = cfg.device

split = cfg.split
epoch_steps = cfg.epoch_steps
eval_interval = cfg.eval_interval
batch_size = cfg.batch_size

lr = cfg.lr
lrReductionRatio = cfg.lrReductionRatio

__all__ = [
    "Config",
    "cfg",
    # model
    "n_emb",
    "T",
    "vocab_size",
    "n_layers",
    "n_heads",
    "n_ffd_hidden",
    "bias",
    "dropout",
    "label_smoothing",
    "weight_tying",
    "device",
    # data
    "split",
    "epoch_steps",
    "eval_interval",
    "batch_size",
    # optimizer
    "lr",
    "lrReductionRatio",
]


if __name__ == "__main__":
    print(f"\nComputation on {device}.")
