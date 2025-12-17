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

    # optimizer
    lr: float = 1e-4
    lrReductionRatio: float = 2.0
    weight_decay: float = 1e-6  # GPT-2 value

    # tokenizer
    use_tiktoken: bool = True

    def __post_init__(self) -> None:
        self.n_ffd_hidden = 4 * self.n_emb

        assert self.epoch_steps % self.batch_size == 0
        assert self.epoch_steps % (self.eval_interval * self.batch_size) == 0


cfg = Config()