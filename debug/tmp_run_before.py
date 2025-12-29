import torch
import config
from model import GPTLanguageModel
from train_utils_before import Train, Construct_optimizer
from data_utils_before import Construct_data_loaders
from import_data import data


def run_case(bs: int) -> None:
    config.batch_size = bs
    config.cfg.batch_size = bs
    config.macro_batch_size = 64
    config.cfg.macro_batch_size = 64

    device = config.device
    if not isinstance(device, torch.device):
        device = torch.device(device)

    torch.manual_seed(42)
    if device.type == "mps":
        torch.mps.manual_seed(42)

    model = GPTLanguageModel(cfg=config.cfg).to(device)
    optimizer = Construct_optimizer(model, config.lr, config.weight_decay, device)

    train_loader, val_loader, train_sampler = Construct_data_loaders(data)
    if train_sampler is not None:
        train_sampler.set_epoch(0)

    print(f"\n=== BEFORE CASE batch_size={bs} macro_batch_size={config.macro_batch_size} ===", flush=True)
    Train(model, train_loader, val_loader, optimizer, config.eval_interval, device)


run_case(32)
run_case(16)
