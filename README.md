# myGPT2

Minimal GPT‑2‑style language model with custom tokenizer and training pipeline for experimenting with transformer internals and training loops.

## Project layout

- `model.py` — GPT model, attention blocks, and generation helpers; includes a class method to load GPT‑2 weights directly.
- `train_utils.py` — optimizer construction, learning‑rate schedule, training loop, checkpointing.
- `data_utils.py` — tokenization helpers, dataset, and DataLoaders; supports tiktoken or a trainable custom tokenizer.
- `config.py` — model and training hyperparameters.
- `run_train.py` — training entrypoint.
- `run_pretrained.py` — load a checkpoint and sample text.
- `import_data.py` — loads the corpus into a token tensor.
- `Dataset/` — raw data (if present).

## Setup

Install dependencies:
```
pip install torch numpy matplotlib tiktoken
```

## Configuration

Edit `config.py`:

- **Model**: `n_layers`, `n_heads`, `n_emb`, `T` (context length).
- **Training**: `batch_size`, `macro_batch_size`, `max_steps`, `eval_interval`.
- **Optimizer**: `lr`, `min_lr`, `warmup_ratio`, `weight_decay`, `grad_clipping`.
- **Tokenizer**: `use_tiktoken` to switch between tiktoken and the regex tokenizer.

`macro_batch_size` is the global batch size used for gradient accumulation. The training loop uses:
```
accum_steps = macro_batch_size // batch_size
```

## Data pipeline

`data_utils.BlockPairDataset`:

- Builds a list of start positions (`starts`) with stride `T`.
- For each index `i`, selects `starts[i % len(starts)]` so training can repeat.
- Takes `T+1` tokens and forms `(x, y)` pairs where `y` is `x` shifted by one.
- Pads with newline tokens near the end of the stream.

Train/val split happens in `Construct_data_loaders` using `config.split`. Validation uses the actual number of starts; training uses a fixed length derived from `config.max_steps`.

## BPE tokenizer (custom)

This repo includes a trainable regex‑based BPE tokenizer in `regex_tokenizer.py`.

- Train a tokenizer:
  ```
  python train_tokenizer.py
  ```
  This writes `tokenizer.json`.
- Use it in training by setting `use_tiktoken = False` in `config.py`. `data_utils.py` will load `tokenizer.json`.

Implementation notes:
- The tokenizer splits text with a GPT‑4‑style regex and learns merges up to `config.vocab_size`.
- It supports `encode`, `decode`, and save/load for reproducibility.

## Training

Run training:
```
python run_train.py
```

Multi-GPU with DDP:
```
torchrun --nproc_per_node=4 run_train.py
```

Resume from a checkpoint:
```
python run_train.py --resume
```

The training loop:

- Uses mixed precision (`autocast`) on CUDA/MPS. Note: PyTorch's MPS support is questionable and suffers from nondeterministic behavior.
- Accumulates gradients to simulate `macro_batch_size`.
- Clips gradients if `grad_clipping > 0`.
- Evaluates every `eval_interval * accum_steps` micro‑batches.
- Saves `checkpoint.pt` at the end of training.

## Learning‑rate schedule

The LR schedule is cosine decay with optional warmup (see `_get_cos_lr` in `train_utils.py`).  
`max_steps` is treated as the number of **macro steps** for the schedule.

## Sampling

Generate text from a checkpoint:
```
python run_pretrained.py
```

Interactive prompt mode:
```
python run_pretrained.py --prompt
```

## Outputs

- `checkpoint.pt` — saved model + optimizer state.
- `loss_plot.png` — training/validation loss curves.
- `generated.txt` — latest sampled output.

## Notes

- Training repeats blocks by modulo indexing; if you want a single‑pass epoch, change `BlockPairDataset.__len__` to `len(starts)` for training.
- Validation batch size is currently set to the same as training; adjust in `data_utils.py` if you want a smaller eval batch.
