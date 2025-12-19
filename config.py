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
    """Training/model configuration. Can construct other presets.
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
    macro_batch_size: int = 512  # for gradient accumulation to simulate larger batch sizes

    # optimizer
    lr: float = 1e-4
    lrReductionRatio: float = 2.0
    weight_decay: float = 0.1  # GPT-2 value
    grad_clipping: float = 1.0  # gradient norm clipping

    # tokenizer
    use_tiktoken: bool = True

    def __post_init__(self) -> None:
        self.n_ffd_hidden = 4 * self.n_emb

        assert self.epoch_steps % self.macro_batch_size == 0
        assert self.epoch_steps % (self.eval_interval * self.macro_batch_size) == 0
        assert self.macro_batch_size % self.batch_size == 0
    
    @classmethod
    def small(cls) -> "Config":
        return cls(
            n_emb=240,
            n_layers=6,
            n_heads=8,
            T=64,
            vocab_size=512,
            dropout=0.3,
            batch_size=64,
            macro_batch_size=64,
            bias=False,
            use_tiktoken=False,
            epoch_steps=6400, # total number of tokens ~ 300k in tinyshakespeare.txt, context length 64
            eval_interval=100,
            weight_decay=0.01,
            weight_tying=False,
            grad_clipping=2.5
        )

cfg = Config().small()

# Backwards-compatible module-level exports, DO NOT DELETE
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
macro_batch_size = cfg.macro_batch_size

lr = cfg.lr
lrReductionRatio = cfg.lrReductionRatio
weight_decay = cfg.weight_decay
grad_clipping = cfg.grad_clipping

use_tiktoken = cfg.use_tiktoken

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
    "weight_decay",
    # tokenizer
    "use_tiktoken"
]