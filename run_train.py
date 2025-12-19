import argparse
import os
import torch
import torch.distributed as dist
import config
from model import GPTLanguageModel
from train_utils import Train, Construct_optimizer
from data_utils import Construct_data_loaders
from run_pretrained import load_pretrained
import logging


device_type = config.device 

def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def _get_rank_world_size():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 1

def _setup_distributed():
    rank, world_size = _get_rank_world_size()
    if world_size > 1 and not _is_distributed():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size

# Silence Inductor autotune logs
os.environ["TORCHINDUCTOR_VERBOSE"] = "0"
logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)

# Add global TF32 + matmul precision flags (CUDA only)
if torch.cuda.is_available() and device_type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
else:
    torch.set_float32_matmul_precision("highest")

def main():
    # CLI options
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-r",  action="store_true", help="Resume training from checkpoint.pt")
    args = parser.parse_args()

    rank, world_size = _setup_distributed()
    config.cfg.world_size = world_size
    config.world_size = world_size
    assert config.macro_batch_size % (config.batch_size * world_size) == 0, (
        "macro_batch_size must be divisible by batch_size * world_size for DDP"
    )

    if torch.cuda.is_available() and device_type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = config.device

    # 1. Construct the model
    if not args.resume:
        model = GPTLanguageModel(cfg=config.cfg).to(device)
        optimizer = Construct_optimizer(model, config.lr, config.weight_decay, device)
    else:
        model, optimizer = load_pretrained("checkpoint.pt", training=True, device=device)
    
    # Compile (CUDA only); drop max-autotune to avoid Triton benchmark spam
    if torch.cuda.is_available() and device_type == "cuda":
        model = torch.compile(model)

    if _is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank] if device.type == "cuda" else None)

    # 3. Build dataloaders
    from import_data import data
    train_loader, val_loader, train_sampler = Construct_data_loaders(data)
    if train_sampler is not None:
        train_sampler.set_epoch(0)

    # 4. Train and save weights
    Train(model, train_loader, val_loader, optimizer, config.eval_interval, device=device)

    if _is_distributed():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
