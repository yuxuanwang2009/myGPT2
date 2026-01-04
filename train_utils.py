# Training utilities
import torch
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import config
from torch.utils.data import DataLoader
import torch.distributed as dist
import inspect

def Construct_optimizer(model, lr, weight_decay, device: torch.device):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: 
            continue
        if param.ndim == 1 or name.endswith(".bias") or "ln_" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device.type == "cuda"

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay":weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        lr=lr,
        fused=use_fused,
    )
    return optimizer

def _get_cos_lr(step: int, max_steps: int, max_lr: float, min_lr: float, warmup_ratio: float) -> float:
    max_steps = max(1, max_steps)
    warmup_steps = max(1, int(warmup_ratio * max_steps))
    step = min(step, max_steps)

    if step < warmup_steps:
        warmup_frac = (step + 1) / warmup_steps
        return min_lr + (max_lr - min_lr) * warmup_frac

    decay_steps = max(1, max_steps - warmup_steps)
    decay_frac = (step - warmup_steps) / decay_steps
    decay_frac = min(max(decay_frac, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_frac))
    return min_lr + (max_lr - min_lr) * cosine

def _get_plateau_lr(curr_lr: float,
                    backtrack_ratio: float,
                    lr_reduction: float,
                    trigger_ratio: float,
                    giveup_ratio: float,
                    min_lr: float,
                    loss_curve_val: list,
                    ) -> float:
    if len(loss_curve_val) <= 1:
        return curr_lr
    checkpoint = max(1, int(backtrack_ratio * len(loss_curve_val)))
    curr = min(5, len(loss_curve_val))
    loss_curve_val_curr = min(loss_curve_val[-curr:]) 
    ratio = loss_curve_val_curr / loss_curve_val[checkpoint - 1]
    if ratio < trigger_ratio:
        return curr_lr
    elif curr_lr * lr_reduction >= min_lr and ratio < giveup_ratio:
        print(f"plateau: n={len(loss_curve_val)}, ckpt={checkpoint}, "
              f"curr_min={loss_curve_val_curr:.6f}, ref={loss_curve_val[checkpoint-1]:.6f}, "
              f"ratio={ratio:.6g}, lr -> {curr_lr*lr_reduction:.6g}", flush=True)
        return curr_lr * lr_reduction
    else:
        print(f"plateau: n={len(loss_curve_val)}, ckpt={checkpoint}, "
              f"curr_min={loss_curve_val_curr:.6f}, ref={loss_curve_val[checkpoint-1]:.6f}, "
              f"ratio={ratio:.6g}, giving up (lr=0).", flush=True)
        return 0.0

def _ddp_mean(value: float) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


def Train(
        m, train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        eval_interval: int,
        device: torch.device
    ):
    m.train()
    for p in m.parameters():
        p.requires_grad = True 

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    master_process = rank == 0
    world_size = dist.get_world_size() if distributed else 1

    if device.type == "cuda":
        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        maybe_sync = torch.cuda.synchronize
    elif device.type == "mps":
        # Careful! autocast on mps has non-determinism problem and can give incorrect loss value at the zeroth step TODO: investigate
        autocast_ctx = lambda: torch.autocast(device_type="mps", dtype=torch.float16)
        maybe_sync = torch.mps.synchronize
    else:
        autocast_ctx = contextlib.nullcontext
        maybe_sync = lambda: None

    loss_curve_tr = []
    loss_curve_val = []
    lr = optimizer.param_groups[0]['lr']
    max_lr = lr
    if config.scheduler == "cosine":
        lr = _get_cos_lr(0, config.max_steps, max_lr, config.min_lr, config.warmup_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr
    if master_process: # only time and plot in the master process
        t0 = time.time()
        fig, ax = plt.subplots()

    optimizer.zero_grad(set_to_none=True)  # clear old gradients efficiently
    lossi = []
    normi = []
    accum_steps_all = config.macro_batch_size // config.batch_size # gradient accumulation steps
    assert accum_steps_all % world_size == 0
    accum_steps = accum_steps_all // world_size
    loss_accum = 0.0
    macro_batch_count = 0
    micro_batch_count = 0
    epoch_idx = 0

    while macro_batch_count < config.max_steps and lr != 0.0:
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch_idx)
        for (X, Y) in train_loader:
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            with autocast_ctx():
                _, loss = m(X, targets=Y)
                loss = loss / accum_steps           # scale loss for gradient accumulation
                loss_accum += loss.detach() # use .detach() to avoid keeping the full graph. not using loss.item() to keep it in the same device
                
            micro_batch_count += 1
            if micro_batch_count % accum_steps != 0: # not time for a macrostep yet
                if hasattr(m, "no_sync"):
                    with m.no_sync():
                        loss.backward() # no need to sync during the accumulation process
                else:
                    loss.backward() # recommendation by PyTorch, stay outside autocast.
            else: # time for a macrostep
                loss.backward()
                if config.grad_clipping > 0.0: # gradient norm clipping
                    normi = torch.nn.utils.clip_grad_norm_(m.parameters(), config.grad_clipping)  
                optimizer.step()                       # optimizer update
                optimizer.zero_grad(set_to_none=True)  # clear gradients for next step
                lossi.append(loss_accum.item())
                loss_accum = 0.0
                
                macro_batch_count += 1
                if macro_batch_count >= config.max_steps:
                    break
                if macro_batch_count % eval_interval == 0: # eval_interval in terms of macro batches
                    loss_tr = sum(lossi) / len(lossi)
                    loss_tr = _ddp_mean(loss_tr)
                    loss_curve_tr.append(loss_tr) if master_process else None
                    lossi = []
                    if config.grad_clipping > 0.0:
                        norm = _ddp_mean(normi.mean().item())
                    # Validation on all ranks; aggregate via all_reduce
                    m.eval()
                    with torch.no_grad():
                        lossi_val = []
                        for (X_val, Y_val) in val_loader:
                            X_val, Y_val = X_val.to(device, non_blocking=True), Y_val.to(device, non_blocking=True)
                            with autocast_ctx():
                                _, loss = m(X_val, targets=Y_val)
                            lossi_val.append(loss.item())
                        loss_val = sum(lossi_val) / len(lossi_val) if len(lossi_val) > 0 else float("nan")
                        loss_val = _ddp_mean(loss_val)
                        if master_process and len(lossi_val) > 0:
                            loss_curve_val.append(loss_val)
                    m.train()

                    # --- end timing ---
                    maybe_sync()
                    if master_process:
                        dt = time.time() - t0
                        print(f"\nEpoch {epoch_idx}: So far trained on {macro_batch_count} (macro)batches: {dt / eval_interval:.2f}s per (macro)batch", flush=True)
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

                lr_new = lr
                if config.scheduler == "cosine":
                    lr_new = _get_cos_lr(macro_batch_count, config.max_steps, max_lr, config.min_lr, config.warmup_ratio)
                elif config.scheduler == "plateau":
                    if macro_batch_count % eval_interval == 0 and len(loss_curve_val) > 0:
                        if master_process:
                            backtrack_ratio = 0.7
                            lr_reduction = 0.75
                            trigger_ratio = 0.992
                            giveup_ratio = 0.998
                            lr_new = _get_plateau_lr(curr_lr=lr,
                                                     backtrack_ratio=backtrack_ratio,
                                                     lr_reduction=lr_reduction,
                                                     trigger_ratio=trigger_ratio,
                                                     giveup_ratio=giveup_ratio,
                                                     loss_curve_val=loss_curve_val,
                                                     min_lr=config.min_lr
                                                     )
                        else:
                            lr_new = 0.0
                        lr_tensor = torch.tensor([lr_new], device=device)
                        if distributed:
                            dist.broadcast(lr_tensor, src=0)
                        lr_new = lr_tensor.item()
                else:
                    raise ValueError(f"Unknown scheduler {config.scheduler}.")
                if lr_new != lr:
                    lr = lr_new
                    for g in optimizer.param_groups:
                        g["lr"] = lr
                if lr == 0:
                    break
        epoch_idx += 1

    if master_process:
        to_save = m.module if hasattr(m, "module") else (m._orig_mod if hasattr(m, "_orig_mod") else m)
        torch.save({
        "model": to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        }, "checkpoint.pt")
        print("Model checkpoint saved to checkpoint.pt.\n", flush=True)
