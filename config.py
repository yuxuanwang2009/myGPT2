# Hyperparameters
import torch

n_emb = 128
T = 20 # context size
vocab_size = 7000
n_layers = 8
n_heads = 4
n_ffd_hidden = 4 * n_emb
dropout = 0.2
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
split = 0.99
lr = 1e-3
epoch_steps = 1280000 # how many batched blocks to feed
eval_interval = 5000
batch_size = 128
assert epoch_steps % batch_size == 0
assert epoch_steps % (eval_interval * batch_size) == 0 

if __name__ == "__main__":
    print(f"\nComputation on {device}.")