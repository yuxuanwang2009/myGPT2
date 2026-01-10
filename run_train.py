import os
import logging
import argparse
import random
import numpy as np
import torch
import config
from model import GPTLanguageModel
from train_utils import Train, Construct_optimizer
from data_utils import Build_datasets, Construct_data_loaders
from run_pretrained import Load_pretrained
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from pathlib import Path

def set_reproducible(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure CuBLAS chooses a deterministic algorithm on CUDA >= 10.2
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True) 

def main():
    # CLI options
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-r",  action="store_true", help="Resume training from checkpoint.pt")
    args = parser.parse_args()

    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK,LOCAL_RANK, and WORLD_SIZE
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = dist.is_available() and world_size > 1
    if distributed:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(),"for now i think we need CUDA for DDP"
        dist.init_process_group(backend='nccl')
        ddp_rank = int (os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{ddp_local_rank}') # does this work for DDP? the answer is yes, as torch.device can take a string like 'cuda:0'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        device = config.device
        master_process = True

    # Use TF32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        # Keep flash/mem-efficient SDP enabled for performance, even in deterministic mode
    else:
        torch.set_float32_matmul_precision("highest")

    # Silence Inductor autotune logs
    os.environ["TORCHINDUCTOR_VERBOSE"] = "0"
    logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)
    
    # Very important on mps, otherwise the nondeterminism affects training performance
    set_reproducible(config.seed, ddp_rank)

    # 1. Construct the model
    if not args.resume:
        model = GPTLanguageModel(cfg=config.cfg, verbose=master_process).to(device) 
        optimizer = Construct_optimizer(model, config.lr, config.weight_decay, device) # need to do this before the model becomes a DDP model
    else:
        ckpt = Path("checkpoint.pt")
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        model, optimizer = Load_pretrained(str(ckpt), training=True, device=device)
        for g in optimizer.param_groups:
            g["lr"] = config.lr
        if master_process:
            print("Model loaded from checkpoint. You will have to pick the proper lr schedule.\n")

    # 2. Wrap for DDP and compile (CUDA only); drop max-autotune to avoid Triton benchmark spam
    if distributed:
        model = DDP(model, device_ids=[ddp_local_rank])  # sync gradients across ranks for multi-GPU training
    if device.type == "cuda":
        model = torch.compile(model)

    # 3. Build dataloaders
    train_ds, val_ds = Build_datasets(ddp_rank, world_size)
    train_loader, val_loader = Construct_data_loaders((train_ds, val_ds))

    # 4. Train and save weights
    Train(model, train_loader, val_loader, optimizer, config.eval_interval, device)

    # Exit DDP
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
