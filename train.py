# Training loop
import torch
import contextlib
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import time
from config import lrReductionRatio

def unique_params(model):
    seen, uniq = set(), []
    for p in model.parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            uniq.append(p)
    return uniq

def Train(m, train_loader, val_loader, optimizer, eval_interval, minimal_lr, device):
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
    indices_back = 1
    fig, ax = plt.subplots()
    step = 0
    lr = optimizer.param_groups[0]['lr']
    epoch_idx = 0
    while len(loss_curve_val) < 2 or loss_curve_val[-1] / loss_curve_val[-indices_back] < 0.998 and lr > minimal_lr:
        # --- start timing this epoch ---
        maybe_sync() # only on CUDA
        t0 = time.time()

        lossi = []
        # Steps in an epoch
        for step, (X, Y) in enumerate(train_loader, start=step+1):
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            with autocast_ctx():
                logits, loss = m(X, targets=Y)
            optimizer.zero_grad(set_to_none=True)  # clear old gradients efficiently
            loss.backward()
            optimizer.step()                       # optimizer update
            
            lossi.append(loss.item())
            if step % eval_interval == 0:
                loss_curve_tr.append(torch.tensor(lossi).mean().item()) 

                m.eval()
                with torch.no_grad():
                    lossi = []
                    for idx_block, (X_val, Y_val) in enumerate(val_loader):
                        X_val, Y_val = X_val.to(device, non_blocking=True), Y_val.to(device, non_blocking=True)    
                        with autocast_ctx():
                            logits, loss = m(X_val, targets=Y_val)  
                        lossi.append(loss.item())
                loss_val = (sum(lossi) / (idx_block + 1))#.item()
                lossi = []
                m.train()
                loss_curve_val.append(loss_val) 

        # --- end timing this epoch ---
        maybe_sync()
        dt = time.time() - t0
        print(f"\nEpoch {epoch_idx}: {dt:.2f}s ({dt/60:.2f}min)", flush=True)
        epoch_idx += 1

        print(f"Trained on {step} batches in total. Loss is {loss_val:.5g}.", flush=True)

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
            
            # Adjust the learning rate
            indices_back = int(len(loss_curve_val) * 0.2) + 2
            print(f"Loss have reduced by {(1 - loss_curve_val[-1] / loss_curve_val[-indices_back]) * 100:.4g}% over the past 20% of the total training time.", flush=True)
            if loss_curve_val[-1] / loss_curve_val[-indices_back] > 0.99:
                lr /= lrReductionRatio
                for g in optimizer.param_groups:
                    g['lr'] = lr
                print(f"Reducing learning rate to {lr:.4g}.\n", flush=True)
            else: print(f"Learning rate kept at {lr:.4g}.\n", flush=True)
    
    to_save = m._orig_mod if hasattr(m, "_orig_mod") else m
    torch.save({
    "model": to_save.state_dict(),
    "optimizer": optimizer.state_dict(),
    }, "checkpoint.pt")
    print("Model checkpoint saved to checkpoint.pt.\n", flush=True)
