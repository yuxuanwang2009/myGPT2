# Hyperparameters
import torch

# model
n_emb = 768
T = 1024 # context size
vocab_size = 50257 # or 50304 padded up to nearest multiple of 64 for efficiency
n_layers = 12
n_heads = 12
n_ffd_hidden = 4 * n_emb
bias = True # same value in GPT-2, but False is better
dropout = 0.0 # GPT-2 value
label_smoothing = 0
weight_tying = True
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# data
split = 0.9
epoch_steps = 256000 # how many batched blocks to feed
eval_interval = 2000
batch_size = 12
assert epoch_steps % batch_size == 0
assert epoch_steps % (eval_interval * batch_size) == 0 

# optimizer
lr = 1e-4
lrReductionRatio = 2

if __name__ == "__main__":
    print(f"\nComputation on {device}.")