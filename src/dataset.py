"""
dataset.py — PyTorch Dataset for (abstract -> story) parallel pairs.

Expects processed/pairs.jsonl with lines like:
    {"abstract": "...", "story": "..."}
"""
import json
import torch
from torch.utils.data import Dataset


class AbstractStoryDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        # TODO: encode abstract and story, add BOS/EOS, truncate/pad to max_len
        src = torch.tensor(self.tokenizer.encode(pair["abstract"])[: self.max_len])
        tgt = torch.tensor(self.tokenizer.encode(pair["story"])[: self.max_len])
        return src, tgt


def collate_fn(batch, pad_id: int = 0):
    """TODO: pad src and tgt sequences to the longest in the batch."""
    raise NotImplementedError
