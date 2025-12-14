# headlineGPT

A minimal, nanoGPT-style language model trained on **short headline-like text** (one snippet per line).  
This README focuses on explaining the **data pipeline** implemented in `data_utils.py`.

---

## How `data_utils.py` Works

### 1) Input format (one example per line)
- The dataset is a single text file of **short lines** (e.g., headlines).
- Each line is separated by a newline `\n`.  
- Newlines are treated as **record separators**; the model is trained on fixed-length chunks that **start right after** a newline.

> If you prefer a different separator (e.g., `"\t"`, `" ||| "`), change the **separator token** accordingly (see “Customization” below).

---

### 2) Tokenization
- `stot(s: str) -> torch.Tensor`: **s**tring **to** **t**okens. Converts raw text to a 1D tensor of token IDs.
- (Optionally) `ttos(t: torch.Tensor) -> str`: **t**okens **to** **s**tring. Converts token IDs back to text for debugging/sampling.
- The whole corpus is tokenized once into `self.data` (a 1D tensor of token IDs).

**Why compact tokenization?**  
Headlines are short and formulaic; a leaner vocab reduces fragmentation and improves sample efficiency.

---

### 3) Record boundaries via a special token
- Let `nl_id = stot("<|endoftext|>")` (or the token you use for line/record breaks).
- We locate all positions where `self.data == nl_id`.
  ```python
  nl_positions = torch.where(self.data == nl_id)[0]
  ```

`torch.where` returns a tuple; indexing with `[0]` extracts the **1D index tensor**.

* We define **start indices** immediately **after** each newline:

  ```python
  self.starts = nl_positions[:-1] + 1  # skip the last newline (no record after)
  ```

  Each `self.starts[i]` is the index of the **first token** of a record (the token after a newline).

**Why do this?**
Sampling training chunks only from these starts ensures we **don’t straddle two records**, which is important for very short sequences like headlines.

---

### 4) Fixed-length chunking (no cross-record bleed)

* Given `block_size` (context length), a sample at index `k` is:

  ```python
  i = self.starts[k]
  x = self.data[i : i + block_size]          # input tokens
  y = self.data[i + 1 : i + 1 + block_size]  # next-token targets
  ```
* If a record is **shorter** than `block_size`, the last part of the window simply spans into padding logic you define (common choices: drop incomplete windows, wrap to next start, or pad—this repo typically **drops**).

---

### 5) Batching & shuffling

* The dataset exposes `__len__` as the number of valid `self.starts`.
* The DataLoader:

  * **shuffles** indices each epoch (or uses a random permutation).
  * **batches** by gathering `B` start indices, slicing `(x, y)` windows as above.
* Because all windows are **fixed length**, no runtime padding is needed inside a batch—this keeps the training loop simple and fast.

---

### 6) Train/val split

* `data_utils.py` creates **disjoint** sets of start indices for train and val (e.g., 90/10 split).
* Splitting happens **after** tokenization and boundary detection, so validation samples are guaranteed not to overlap train samples at the **record** level.

---

## FAQ

**Q: Why not just stream the whole file without respecting boundaries?**
A: Headlines are short. Letting windows cross examples dilutes the learning signal and harms style and concision.

**Q: Do we pad?**
A: The simplest path is to **drop** windows that would run past a record end. For tiny models and short contexts, this is typically fine; it avoids padding logic entirely.

**Q: Can I sample with stride < `block_size` to increase diversity?**
A: Yes. Instead of one start per record, you can generate more starts within each record (e.g., every `stride` tokens), as long as you don’t cross the boundary.

---

## Files at a glance

* `data_utils.py` — tokenization, boundary finding, start-index cache, dataset & batching.
* `config.py` — `block_size`, batch size, split ratio, seed, and file paths.
* `model.py` — minimal GPT (Transformer + LM head).
* `train.py`, `run_train.py` — training loop & CLI.
* `run_pretrained.py` — sampling from checkpoints.
* `generated.txt`, `loss_plot.png` — artifacts for quick inspection.

---

## License

MIT. Inspired by the simplicity of nanoGPT.
