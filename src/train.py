"""
train.py — Training loop for the from-scratch Transformer.

Usage:
    python -m src.train --data data/processed/pairs.jsonl --epochs 10
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import Seq2SeqTransformer
# from .dataset import AbstractStoryDataset, collate_fn
# from .tokenizer import StoryTokenizer


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # TODO: tokenizer = StoryTokenizer(args.tokenizer)
    # TODO: dataset = AbstractStoryDataset(args.data, tokenizer)
    # TODO: loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
    #                            collate_fn=lambda b: collate_fn(b, tokenizer.pad_id))

    # model = Seq2SeqTransformer(vocab_size=VOCAB).to(device)
    # criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # for epoch in range(args.epochs):
    #     for src, tgt in loader:
    #         # teacher forcing: feed tgt[:, :-1], predict tgt[:, 1:]
    #         ...
    raise NotImplementedError("TODO: fill in the training loop")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/pairs.jsonl")
    p.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    train(p.parse_args())
