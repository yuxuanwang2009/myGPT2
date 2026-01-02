from typing import Iterable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import config
import regex_tokenizer as rt
import tiktoken
from datasets import load_dataset

if config.use_tiktoken == True:
    tok = tiktoken.get_encoding("gpt2")
else:
    tok = rt.RegexTokenizer.load("tokenizer.json")

# HF source for docs (FineWeb-Edu, 10B sample config)
DATASET_PATH = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"


class HFDocStream(IterableDataset):
    """Stream docs from HF, sharded per rank, with an optional doc cap."""

    def __init__(self, split: str, rank: int, world_size: int, limit: int | None = None):
        super().__init__()
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.limit = limit
        self.epoch = 0
        self.base_seed = int(config.seed)

    def __iter__(self) -> Iterable[str]:
        ds = load_dataset(DATASET_PATH, name=DATASET_CONFIG, split=self.split, streaming=True)
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        # shard across ranks * workers first to avoid overlaps, then shuffle deterministically
        total_shards = max(1, self.world_size * num_workers)
        shard_idx = self.rank * num_workers + worker_id
        ds = ds.shard(num_shards=total_shards, index=shard_idx)
        ds = ds.shuffle(buffer_size=10_000, seed=self.base_seed + self.epoch)
        count = 0
        for row in ds:
            text = row.get("text", "") if isinstance(row, dict) else ""
            if text:
                yield text
                count += 1
                if self.limit is not None and count >= self.limit:
                    break

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


class CachedHFDocStream(IterableDataset):
    """Stream a fixed cached slice from HF, broadcast to all ranks, then shard locally."""

    def __init__(self, split: str, rank: int, world_size: int, limit: int):
        super().__init__()
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.limit = limit
        self._cached_docs = None

    def _ensure_cache(self):
        if self._cached_docs is not None:
            return
        docs = []
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            if self.rank == 0:
                ds = load_dataset(DATASET_PATH, name=DATASET_CONFIG, split=self.split, streaming=True)
                for row in ds:
                    text = row.get("text", "") if isinstance(row, dict) else ""
                    if text:
                        docs.append(text)
                    if len(docs) >= self.limit:
                        break
            obj = [docs]
            dist.broadcast_object_list(obj, src=0)
            docs = obj[0]
        else:
            ds = load_dataset(DATASET_PATH, name=DATASET_CONFIG, split=self.split, streaming=True)
            for row in ds:
                text = row.get("text", "") if isinstance(row, dict) else ""
                if text:
                    docs.append(text)
                if len(docs) >= self.limit:
                    break
        if dist.is_available() and dist.is_initialized() and docs:
            total = len(docs) - (len(docs) % self.world_size)
            docs = docs[:total] if total > 0 else docs
        self._cached_docs = docs

    def __iter__(self) -> Iterable[str]:
        self._ensure_cache()
        total = len(self._cached_docs)
        start = (total * self.rank) // self.world_size
        end = (total * (self.rank + 1)) // self.world_size
        for doc in self._cached_docs[start:end]:
            yield doc

    def __len__(self):
        self._ensure_cache()
        total = len(self._cached_docs)
        start = (total * self.rank) // self.world_size
        end = (total * (self.rank + 1)) // self.world_size
        return end - start

# tokenizer encoding: from a string to a tensor of tokens
def stot(s: str) -> torch.Tensor:
    ids = tok.encode(s, allowed_special={"<|endoftext|>"}) # special characters will be dedicated to a single token if trained with them.
    return torch.tensor(ids, dtype=torch.long)

# tokenizer encoding: from a tensor of tokens to a string
def ttos(t: torch.Tensor, for_output: bool = False) -> str:
    out = tok.decode(t.tolist())
    return out.replace("<|endoftext|>", "\n") if for_output else out


class BlockStream(IterableDataset):
    """Stream fixed-length (x, y) blocks from a doc iterable (the DocStreams)."""

    def __init__(self, doc_iterable: IterableDataset, materialize: bool = False):
        super().__init__()
        self.doc_iterable = doc_iterable
        self.materialize = materialize
        self._blocks = None
        if self.materialize:
            self._blocks = list(self._iter_blocks())

    def _iter_blocks(self):
        T = config.T
        pad_id = stot("\n").item()
        eot_tokens = stot("<|endoftext|>")
        if eot_tokens.numel() != 1:
            raise ValueError(
                "Tokenizer does not have a single <|endoftext|> token. "
                "Please train your tokenizer with special_tokens=['<|endoftext|>'] or switch to tiktoken."
            )
        eot_id = eot_tokens.item()

        buf = []
        for doc in self.doc_iterable:
            ids = tok.encode(doc, allowed_special={"<|endoftext|>"})
            ids.append(eot_id)
            buf.extend(ids)
            while len(buf) >= T + 1:
                block = buf[: T + 1]
                buf = buf[T + 1 :]
                block_t = torch.tensor(block, dtype=torch.long)
                yield block_t[:-1], block_t[1:]
        if buf:
            if len(buf) < T + 1:
                buf = buf + [pad_id] * (T + 1 - len(buf))
            block_t = torch.tensor(buf[: T + 1], dtype=torch.long)
            yield block_t[:-1], block_t[1:]

    def __iter__(self):
        if self.materialize and self._blocks is not None:
            return iter(self._blocks)
        return self._iter_blocks()

    def __len__(self):
        if self.materialize and self._blocks is not None:
            return len(self._blocks)
        raise TypeError("Length not available for streaming BlockStream")

    def set_epoch(self, epoch: int):
        if hasattr(self.doc_iterable, "set_epoch"):
            self.doc_iterable.set_epoch(epoch)

def Build_datasets(rank: int = 0, world_size: int = 1):
    """Create streaming train/val IterableDatasets. FineWeb-Edu only exposes 'train'."""
    train_limit = int(0.01 * 10_000_000) if config.device.type != "cuda" else None
    train_ds = HFDocStream("train", rank, world_size, limit=train_limit)
    # Validation: cached HF slice, sharded per rank
    val_limit = max(1, int((1.0 - config.split) * 100_000)) if config.device.type != "cuda" else max(1, int((1.0 - config.split) * 10_000_000))  # rough doc count scale for ~10B slice
    val_ds = CachedHFDocStream("train", rank, world_size if dist.is_available() and dist.is_initialized() else 1, limit=val_limit)
    return train_ds, val_ds

def Construct_data_loaders(data) -> DataLoader:
    """Wrap train/val doc streams into block DataLoaders (batch_size = blocks)."""
    if isinstance(data, tuple) and len(data) == 2 and all(isinstance(d, IterableDataset) for d in data):
        ds_tr, ds_va = data
    else:
        raise TypeError(f"data must be a (train_ds, val_ds) tuple, got {type(data)}")

    # Wrap doc streams into block streams (train stays streaming, val may be materialized)
    bs_tr = BlockStream(ds_tr, materialize=False)
    materialize_val = isinstance(ds_va, CachedHFDocStream)
    bs_va = BlockStream(ds_va, materialize=materialize_val)

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0

    # Convert the Datasets to Dataloaders with backend-specific settings
    # IterableDatasets: no sampler/shuffle
    train_loader = DataLoader(
        bs_tr,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8 if dist.is_available() and dist.is_initialized() else 2 if config.device == 'mps' else 0,
        pin_memory=True if dist.is_available() and dist.is_initialized() else False,
        prefetch_factor=4 if dist.is_available() and dist.is_initialized() else None,
        persistent_workers=True if dist.is_available() and dist.is_initialized() else False,
    )
    val_loader = DataLoader(
        bs_va,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    if rank == 0:
        print("Training set: streaming blocks (length unknown)", flush=True)
        if hasattr(bs_va, "__len__"):
            total_blocks = len(bs_va)
            total_batches = (total_blocks + config.batch_size - 1) // config.batch_size
            print(f"Validation set: {total_blocks} blocks, loader batches: {total_batches}", flush=True)
        else:
            print("Validation set: streaming blocks (length unknown)", flush=True)
    return train_loader, val_loader, None
