import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


class AbstractStoryDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer_path, max_src_len=512, max_tgt_len=512):
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        with open(jsonl_path, encoding="utf-8") as f:
            self.pairs = [json.loads(l) for l in f]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        src = self.tok.encode(p["abstract"]).ids[:self.max_src_len]
        tgt = self.tok.encode(p["story"]).ids[:self.max_tgt_len - 2]
        tgt = [BOS_ID] + tgt + [EOS_ID]
        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    sm = max(len(s) for s in srcs)
    tm = max(len(t) for t in tgts)

    def pad(x, n):
        out = torch.full((n,), PAD_ID, dtype=torch.long)
        out[:len(x)] = x
        return out

    return torch.stack([pad(s, sm) for s in srcs]), torch.stack([pad(t, tm) for t in tgts])


if __name__ == "__main__":
    ds = AbstractStoryDataset("data/processed/pairs.jsonl", "data/processed/tokenizer.json")
    print(len(ds))
    s, t = ds[0]
    print(s[:10].tolist())
    print(t[:10].tolist())
