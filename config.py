# Hyperparameters
import torch

n_emb = 60
T = 32 # context size
vocab_size = 256
n_layers = 3
n_heads = 2
n_ffd_hidden = 4 * n_emb
dropout = 0.2
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
split = 0.9
lr = 1e-4
lrReductionRatio = 2
epoch_steps = 128000 # how many batched blocks to feed
eval_interval = 2000
batch_size = 32
assert epoch_steps % batch_size == 0
assert epoch_steps % (eval_interval * batch_size) == 0 

if __name__ == "__main__":
    print(f"\nComputation on {device}.")