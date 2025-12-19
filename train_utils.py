# Training utilities
import torch
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import config
from torch.utils.data import DataLoader
import inspect

def Construct_optimizer(model, lr, weight_decay, device_type):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: 
            continue
        if param.ndim == 1 or name.endswith(".bias") or "ln_" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay":weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        fused=use_fused,
    )
    return optimizer

def _get_cos_lr(step: int, max_steps: int, max_lr: float, min_lr: float, warmup_ratio: float) -> float:
    max_steps = max(1, max_steps)
    warmup_steps = max(1, int(warmup_ratio * max_steps))
    step = min(step, max_steps - 1)

    if step < warmup_steps:
        warmup_frac = (step + 1) / warmup_steps
        return min_lr + (max_lr - min_lr) * warmup_frac

    decay_steps = max(1, max_steps - warmup_steps)
    decay_frac = (step - warmup_steps) / decay_steps
    decay_frac = min(max(decay_frac, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_frac))
    return min_lr + (max_lr - min_lr) * cosine


def Train(m, train_loader: DataLoader, val_loader: DataLoader, optimizer, eval_interval, device):
    m.train()
    for p in m.parameters():
        p.requires_grad = True 

    device_type = device if isinstance(device, str) else device.type
    has_cuda = torch.cuda.is_available() and device_type == "cuda"
    has_mps = torch.backends.mps.is_available() and device_type == "mps"

    if has_cuda:
        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        maybe_sync = torch.cuda.synchronize
    elif has_mps:
        autocast_ctx = lambda: torch.autocast(device_type="mps", dtype=torch.float16)
        maybe_sync = torch.mps.synchronize
    else:
        autocast_ctx = contextlib.nullcontext
        maybe_sync = lambda: None

    loss_curve_tr = []
    loss_curve_val = []
    fig, ax = plt.subplots()
    lr = optimizer.param_groups[0]['lr']
    max_lr = lr

    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)  # clear old gradients efficiently
    lossi = []
    normi = []
    accum_steps = config.macro_batch_size // config.batch_size # gradient accumulation steps
    loss_accum = 0.0
    macro_batch_count = 0

    for microstep, (X, Y) in enumerate(train_loader): 
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        with autocast_ctx():
            _, loss = m(X, targets=Y)
            loss = loss / accum_steps           # scale loss for gradient accumulation
            loss_accum += loss.detach() # use .detach() to avoid keeping the full graph. not using loss.item() to keep it in the same device
        loss.backward() # why does this stay outside autocast_ctx? --- because we want to keep gradients in full precision
            
        if (microstep + 1) % accum_steps == 0:
            lr = _get_cos_lr(macro_batch_count, config.max_steps, max_lr, config.min_lr, config.warmup_ratio)
            for g in optimizer.param_groups:
                g["lr"] = lr
            if config.grad_clipping > 0.0: # gradient norm clipping
                normi = torch.nn.utils.clip_grad_norm_(m.parameters(), config.grad_clipping)  
            optimizer.step()                       # optimizer update
            optimizer.zero_grad(set_to_none=True)  # clear gradients for next step
            lossi.append(loss_accum.item())
            loss_accum = 0.0
            macro_batch_count += 1

        if (microstep + 1) % (eval_interval * accum_steps) == 0: # eval_interval in terms of macro batches
            loss_curve_tr.append(sum(lossi)/len(lossi))
            lossi = []
            if config.grad_clipping > 0.0:
                norm = normi.mean().item()
            m.eval()
            with torch.no_grad():
                lossi_val = []
                for idx_block, (X_val, Y_val) in enumerate(val_loader):
                    X_val, Y_val = X_val.to(device, non_blocking=True), Y_val.to(device, non_blocking=True)    
                    with autocast_ctx():
                        _, loss = m(X_val, targets=Y_val)  
                    lossi_val.append(loss.item())
            loss_val = sum(lossi_val)/len(lossi_val)
            loss_curve_val.append(loss_val)
            lossi_val = []
            m.train()

            # --- end timing ---
            maybe_sync()
            dt = time.time() - t0
            print(f"\nTrained on {(microstep + 1) // accum_steps} (macro)batches: {dt / eval_interval:.2f}s per (macro)batch", flush=True)
            t0= time.time()

            if config.grad_clipping:
                print(f"Norm of the gradient is {norm:.5g}. Learning rate is {lr:.5g}. Loss is {loss_val:.5g}.", flush=True)
            else:
                print(f"Learning rate is {lr:.5g}. Loss is {loss_val:.5g}.", flush=True)   

            if len(loss_curve_tr) > 1:
                # Build the plot
                ax.clear()
                x_axis = np.arange(1, len(loss_curve_tr)+1) * eval_interval
                ax.loglog(x_axis, loss_curve_tr, label=f"train, final = {loss_curve_tr[-1]:.4f}")
                ax.loglog(x_axis, loss_curve_val, label=f"validation, final = {loss_curve_val[-1]:.4f}")
                ax.set_xlabel(f"Training step")
                ax.set_ylabel("Loss")
                ax.legend()
                # Save first, then show
                fig.savefig("loss_plot.png", dpi=300, bbox_inches="tight")
    
    to_save = m._orig_mod if hasattr(m, "_orig_mod") else m
    torch.save({
    "model": to_save.state_dict(),
    "optimizer": optimizer.state_dict(),
    }, "checkpoint.pt")
    print("Model checkpoint saved to checkpoint.pt.\n", flush=True)
