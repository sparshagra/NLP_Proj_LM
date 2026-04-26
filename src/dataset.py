"""
LM Dataset utilities for causal language modelling.

Three classes:
  - ArrowLMDataset  : wraps a HuggingFace Arrow dataset; tokenises on-the-fly
                      per story.  Memory efficient for large datasets like
                      TinyStories (2.1 M stories).
  - WikiBinDataset  : reads a raw int32 binary token file via numpy memmap.
                      Used for Phase 2 Wikipedia (large corpus, pre-tokenised).
                      O(1) random access, no RAM spike, handles 1B+ tokens.
  - LMDataset       : legacy class — takes a plain list[str], tokenises all at
                      once.  Fine for small datasets only.

Each __getitem__ returns (x, y) tensors of length `block_size` where y = x[1:].
"""

import os

import torch
from torch.utils.data import Dataset


class ArrowLMDataset(Dataset):
    """
    Wraps a HuggingFace datasets.Dataset (Arrow-backed) for causal LM.

    Instead of pre-loading all tokens into RAM, each __getitem__ maps
    directly to a story stored on disk via Arrow memory-mapping.

    The dataset treats each story as a self-contained block:
      - If a story is longer than block_size, it is truncated.
      - If shorter, it is left as-is (DataLoader will pad via collate).

    We pad short sequences to block_size in the collate function so all
    tensors in a batch have identical shape.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        block_size: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        text_column: str = "text",
        max_stories: int = None,
    ):
        self.ds         = hf_dataset
        self.tokenizer  = tokenizer
        self.block_size = block_size
        self.bos_id     = bos_id
        self.eos_id     = eos_id
        self.pad_id     = pad_id
        self.text_col   = text_column

        self.length = len(self.ds) if max_stories is None else min(max_stories, len(self.ds))
        print(f"  ArrowLMDataset: {self.length:,} stories  (block_size={block_size})")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        text = self.ds[idx][self.text_col]
        ids  = self.tokenizer.encode(text, add_special_tokens=False)

        # Wrap with BOS / EOS
        ids = [self.bos_id] + ids + [self.eos_id]

        # Truncate to block_size + 1 (we need +1 to produce x and y of block_size)
        ids = ids[: self.block_size + 1]

        # Pad if shorter than block_size + 1
        if len(ids) < self.block_size + 1:
            ids = ids + [self.pad_id] * (self.block_size + 1 - len(ids))

        t = torch.tensor(ids, dtype=torch.long)
        return t[:-1], t[1:]   # x, y  — both length block_size


class LMDataset(Dataset):
    """
    Legacy dataset: takes a list[str], tokenises everything into one flat
    token stream, then slices into blocks.  Use only for small datasets
    (< a few hundred MB of text).
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        block_size: int,
        bos_id: int,
        eos_id: int,
        stride: int = None,
        max_stories: int = None,
    ):
        if max_stories is not None:
            texts = texts[:max_stories]

        stride = stride or block_size

        token_ids: list[int] = []
        for txt in texts:
            ids = tokenizer.encode(txt, add_special_tokens=False)
            token_ids.extend([bos_id] + ids + [eos_id])

        print(f"  Total tokens in stream : {len(token_ids):,}")

        needed = block_size + 1
        self.examples: list[torch.Tensor] = []
        for start in range(0, len(token_ids) - needed + 1, stride):
            chunk = token_ids[start : start + needed]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))

        print(f"  Blocks created : {len(self.examples):,}  "
              f"(block_size={block_size}, stride={stride})")

        if len(self.examples) == 0:
            raise ValueError(
                f"No blocks. Got {len(token_ids)} tokens, need ≥ {needed}."
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        return chunk[:-1], chunk[1:]


class WikiBinDataset(Dataset):
    """
    Memory-mapped binary token dataset for large corpora (Phase 2 Wikipedia).

    Tokens are stored as raw int32 little-endian in a .bin file produced by
    `download_data.py --phase 2`.  numpy.memmap gives O(1) random access
    without ever loading the full file into RAM (safe for 4 GB+ files).

    Fixed-stride non-overlapping blocks (stride = block_size by default)
    give the most efficient coverage of the corpus per epoch.

    Usage:
        ds = WikiBinDataset("data/phase2_wikipedia/train_tokens.bin", block_size=512)
        x, y = ds[0]   # x.shape = y.shape = (512,)
    """

    def __init__(self, bin_path: str, block_size: int, stride: int = None):
        import numpy as np

        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"\n  Token binary not found: {bin_path}\n"
                "  → Run:  python download_data.py --phase 2\n"
            )

        self.data       = np.memmap(bin_path, dtype=np.int32, mode="r")
        self.block_size = block_size
        self.stride     = stride or block_size   # non-overlapping blocks by default

        # Total usable blocks: need block_size+1 tokens per block (x + y shift)
        self.n_blocks = max(0, (len(self.data) - self.block_size - 1) // self.stride)

        n_tok = len(self.data)
        print(
            f"  WikiBinDataset : {os.path.basename(bin_path)}"
            f"  {n_tok / 1e9:.3f}B tokens"
            f"  → {self.n_blocks:,} blocks"
            f"  (block={block_size}, stride={self.stride})"
        )

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        import numpy as np
        start = idx * self.stride
        # memmap slice → numpy array (copy to avoid memmap reference issues)
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y
