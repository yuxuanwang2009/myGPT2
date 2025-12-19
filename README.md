# DDP Compatibility Notes

This branch updates the training stack to run correctly under DistributedDataParallel (DDP).

## What Changed

- Added `world_size` to config and used it when computing gradient accumulation so `macro_batch_size` is global (`config.py`, `train_utils.py`).
- Added a post-DDP init assert to ensure `macro_batch_size % (batch_size * world_size) == 0` (`run_train.py`).
- Wrapped the model with `DistributedDataParallel` and set the per-rank device (`run_train.py`).
- Switched DataLoaders to use `DistributedSampler` (train shuffle on, val shuffle off, val drop_last on) and disabled DataLoader shuffling when distributed (`data_utils.py`).
- Added `no_sync()` for non-final micro-steps to avoid redundant all-reduce during gradient accumulation (`train_utils.py`).
- Averaged validation loss across ranks using all-reduce of sum and count, and limited logging/plotting/checkpointing to rank 0 (`train_utils.py`).
- Made checkpoint resume device-aware so each rank loads onto its own device, and moved optimizer state tensors to the same device (`run_pretrained.py`, `run_train.py`).

## Notes

- The dataset still repeats blocks via modulo indexing; this can reduce effective data diversity across ranks, but is kept as-is for now.
- If you want per-epoch reshuffling, call `train_sampler.set_epoch(epoch)` on each epoch boundary.
