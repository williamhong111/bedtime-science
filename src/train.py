import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tokenizers import Tokenizer

from src.model import Seq2SeqTransformer
from src.dataset import AbstractStoryDataset, collate_fn, PAD_ID


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(args):
    device = get_device()
    print("device:", device)

    ds = AbstractStoryDataset(args.data, args.tokenizer,
                              max_src_len=args.max_len, max_tgt_len=args.max_len)
    if args.limit:
        ds = Subset(ds, list(range(min(args.limit, len(ds)))))
    print("training on", len(ds), "pairs")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    vocab_size = Tokenizer.from_file(args.tokenizer).get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len,
        pad_id=PAD_ID,
    ).to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        t0 = time.time()
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = model(src, tgt_in)
            loss = crit(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        avg = total / max(1, len(loader))
        print(f"epoch {epoch}  loss {avg:.3f}  time {time.time() - t0:.1f}s")

        ckpt = Path(args.ckpt_dir) / f"model_epoch{epoch}.pt"
        torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/pairs.jsonl")
    p.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)
    main(p.parse_args())
