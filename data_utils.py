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
    def __init__(self, data: torch.Tensor, T: int, train=True):
        self.data = data         # token-id stream (int tensor)
        # print (f"Constructing BlockPairDataset, data length: {len(self.data)} tokens.", flush=True)
        self.T = T
        self.train = train
        self.pad_id = stot("\n").item()

        # For boundary-free corpora, step through the stream with stride T when deterministic
        self.starts = torch.arange(0, len(self.data), self.T) 
        if self.train:
            self.N = config.max_steps * config.macro_batch_size
        else:
            self.N = len(self.starts)

    def __len__(self):
        # DataLoader will treat this as "blocks in the dataset"
        return self.N

    def __getitem__(self, i:int):

        if self.train:
            s = self.starts[i % len(self.starts)].item()
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
def Construct_data_loaders(data:torch.Tensor) -> DataLoader:
    # Train/val split into two tensors
    len_ = data.numel()
    len_tr = int(len_ * config.split)
    len_val = len_ - len_tr
    data_tr, data_val = data.split([len_tr, len_val])

    device = config.device
    device_type = device if isinstance(device, str) else device.type
    cuda = torch.cuda.is_available() and device_type == "cuda"

    # Convert torch.Tensor to Dataset of proper context blocks
    ds_tr = BlockPairDataset(data_tr, config.T, train=True) # Dataset objects 
    ds_va = BlockPairDataset(data_val, config.T, train=False)

    # Convert the Datasets to Dataloaders with backend-specific settings
    if cuda:
        train_loader = DataLoader(
            ds_tr,
            batch_size=config.batch_size,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            ds_va,
            batch_size=config.batch_size,
            num_workers=2,
            pin_memory=True,
        )
    else:
        # CPU / MPS: simpler loader settings; adjust workers as desired
        train_loader = DataLoader(
            ds_tr,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            ds_va,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=False,
        )

    print(f"Training data (with repetition) consist of {len(ds_tr) // config.macro_batch_size} batches of text.", flush=True)
    print(f"Validation data consist of {len(ds_va) // config.macro_batch_size} microbatches of text.\n", flush=True)
    return train_loader, val_loader
