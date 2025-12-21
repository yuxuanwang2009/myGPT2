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
    max_steps: int = 262144  # how many batches to train for
    eval_interval: int = 600
    batch_size: int = 16
    macro_batch_size: int = 512  # for gradient accumulation to simulate larger batch sizes

    # optimizer
    lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1  # GPT-2 value
    grad_clipping: float = 1.0  # gradient norm clipping

    # reproducibility
    seed: int = 1337

    # tokenizer
    use_tiktoken: bool = True

    def __post_init__(self) -> None:
        self.n_ffd_hidden = 4 * self.n_emb
        assert self.macro_batch_size % self.batch_size == 0
    
    @classmethod
    def small(cls) -> "Config":
        return cls(
            n_emb=120,
            n_layers=6,
            n_heads=6,
            T=64,
            vocab_size=512,
            dropout=0.3,
            batch_size=32, # there ust be a bug! simply changing this from 32 to 16 causes a different training dynamic. Investigate.
            macro_batch_size=64,
            bias=False,
            use_tiktoken=False,
            max_steps=30000, # in terms of macrobatches
            eval_interval=600, # in terms of macrobatches
            warmup_ratio=0.0,
            weight_decay=0.02,
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
max_steps = cfg.max_steps
eval_interval = cfg.eval_interval
batch_size = cfg.batch_size
macro_batch_size = cfg.macro_batch_size

lr = cfg.lr
min_lr = cfg.min_lr
warmup_ratio = cfg.warmup_ratio
weight_decay = cfg.weight_decay
grad_clipping = cfg.grad_clipping

use_tiktoken = cfg.use_tiktoken
seed = cfg.seed
