import argparse
import torch
import config
from model import GPTLanguageModel
from train_utils import Train, Construct_optimizer
from data_utils import Construct_data_loaders
from run_pretrained import load_pretrained
import os
import logging


device_type = config.device 

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

    # 1. Construct the model
    if not args.resume:
        model = GPTLanguageModel(cfg=config.cfg).to(config.device)
        optimizer = Construct_optimizer(model, config.lr, config.weight_decay, config.device)
    else:
        model, optimizer = load_pretrained("checkpoint.pt", training=True)
    
    # Compile (CUDA only); drop max-autotune to avoid Triton benchmark spam
    if torch.cuda.is_available() and device_type == "cuda":
        model = torch.compile(model)

    # 3. Build dataloaders
    from import_data import data
    train_loader, val_loader = Construct_data_loaders(data, config.T, batch_size=config.batch_size)

    # 4. Train and save weights
    Train(model, train_loader, val_loader, optimizer, config.eval_interval, minimal_lr=1e-6, device=config.device)

if __name__ == "__main__":
    main()
