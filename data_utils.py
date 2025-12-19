import torch
from torch.utils.data import Dataset, DataLoader
import config
import regex_tokenizer as rt
import tiktoken

if config.use_tiktoken == True:
    tok = tiktoken.get_encoding("gpt2")
else:
    tok = rt.RegexTokenizer.load("tokenizer.json")


# tokenizer encoding: from a string to a tensor of tokens
def stot(s: str) -> torch.Tensor:
    ids = tok.encode(s)
    return torch.tensor(ids, dtype=torch.long)

# tokenizer encoding: from a tensor of tokens to a string
def ttos(t: torch.Tensor, for_output: bool = False) -> str:
    out = tok.decode(t.tolist())
    return out.replace("<|endoftext|>", "\n") if for_output else out

# a pair of tensors as chunks of text offset by one
class BlockPairDataset(Dataset):
    """
    Dataset that samples fixed-length windows of token IDs from a text stream.

    - Takes a long tensor as input
    - Each sample starts at the beginning of a text (just after a '<|endoftext|>').
    - Each sample is exactly block_size long (x,y), padded with '<|endoftext|>' if needed.
    - x: [block_size] token IDs
    - y: [block_size] token IDs (x shifted by one position)
    """
    def __init__(self, data: torch.Tensor, T: int, random=True):
        self.data = data         # token-id stream (int tensor)
        self.T = T
        self.random = random
        self.pad_id = stot("\n").item()

        # For boundary-free corpora, step through the stream with stride T when deterministic
        if self.random:
            self.starts = None
            self.N = config.epoch_steps
        else:
            max_start = max(len(self.data) - (self.T + 1), 0)
            self.starts = torch.arange(0, max_start + 1, self.T)
            self.N = len(self.starts)

    def __len__(self):
        # DataLoader will treat this as "samples per epoch"
        return self.N

    def __getitem__(self, i:int):

        if self.random:
            max_start = max(len(self.data) - (self.T + 1), 0)
            s = torch.randint(max_start + 1, (1,)).item() if max_start > 0 else 0
        else:
            s = self.starts[i].item()        

        # Take T+1 tokens so we can form (x,y) with a one-step shift
        end = min(s + self.T + 1, len(self.data))
        seq = self.data[s:end]

        # If near end-of-stream, pad with '\n' token to fixed length
        if seq.numel() < self.T + 1:
            pad_len = self.T + 1 - seq.numel()
            pad = torch.full((pad_len,), self.pad_id, dtype=self.data.dtype)
            seq = torch.cat([seq, pad], dim=0)

        # Already token IDs; ensure proper dtype for embeddings
        x = seq[:-1].long()   # input sequence
        y = seq[1:].long()    # target sequence
        return x, y

# constructing DataLoaders for train and val data
# Input: data - full text as a tensor of token IDs
# BlockPairDataset - Dataset whose items are (x,y) pairs of token ID tensors with context length T
# DataLoader - loads BlockPairDataset into batches for training
# Output: train_loader, val_loader which are DataLoader objects
# In Andrej's implementation, DataLoaderLite takes reads the .txt file and 
# return x, y pairs using a next_batch method -- much simpler.
def Construct_data_loaders(data:torch.Tensor, T, batch_size) -> DataLoader:
    # Train/val split into two tensors
    len_ = data.numel()
    len_tr = int(len_ * config.split)
    len_val = len_ - len_tr
    data_tr, data_val = data.split([len_tr, len_val])

    device = config.device
    device_type = device if isinstance(device, str) else device.type
    cuda = torch.cuda.is_available() and device_type == "cuda"

    # Convert torch.Tensor to Dataset of proper context blocks
    ds_tr = BlockPairDataset(data_tr, T, random = True) # Dataset objects 
    ds_va = BlockPairDataset(data_val, T, random = False)

    # Convert the Datasets to Dataloaders with backend-specific settings
    if cuda:
        train_loader = DataLoader(
            ds_tr,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            ds_va,
            batch_size=8,
            num_workers=2,
            pin_memory=True,
        )
    else:
        # CPU / MPS: simpler loader settings; adjust workers as desired
        train_loader = DataLoader(
            ds_tr,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            ds_va,
            batch_size=8,
            num_workers=0,
            pin_memory=False,
        )
    
    print(f"Training data (one epoch) consist of {len(ds_tr)} batched blocks of text.", flush=True)
    print(f"Validation data consist of {len(ds_va)} batched blocks of text.\n", flush=True)
    return train_loader, val_loader
