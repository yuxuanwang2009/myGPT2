# count_train_loader.py
import os

import torch
import torch.distributed as dist

import config
from data_utils import Build_datasets, Construct_data_loaders


def _init_distributed() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = dist.is_available() and world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return rank, world_size


def main():
    rank, world_size = _init_distributed()

    config.batch_size = 512
    config.macro_batch_size = 512
    train_ds, val_ds = Build_datasets(rank, world_size)
    train_loader, _, _ = Construct_data_loaders((train_ds, val_ds))

    cap = 1_000_000
    count = 0
    for _ in train_loader:
        count += 1
        if count % 100 == 0:
            print(f"[rank {rank}] seen {count} batches...", flush=True)
        if count >= cap:
            print(f"[rank {rank}] Hit {cap} batches without exhausting the loader.", flush=True)
            break
    else:
        print(f"[rank {rank}] Loader exhausted at {count} batches.", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
