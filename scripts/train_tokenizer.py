"""
train_tokenizer.py — Train a BPE tokenizer on our (abstract, story) corpus.

The tokenizer is shared between encoder (reads abstracts) and decoder
(generates stories), so we train on the union of both sides.

Output: data/processed/tokenizer.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def iter_texts(jsonl_path: str):
    """Yield every abstract and every story as separate text samples."""
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("abstract"):
                yield d["abstract"]
            if d.get("story"):
                yield d["story"]


def main(args):
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Build a byte-level BPE tokenizer (the same family GPT-2 uses).
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"training BPE on {args.data} (vocab size = {args.vocab_size}) ...")
    tokenizer.train_from_iterator(iter_texts(args.data), trainer=trainer)

    tokenizer.save(args.out)
    print(f"\nsaved tokenizer to {args.out}")
    print(f"vocab size: {tokenizer.get_vocab_size()}")

    # Show special-token IDs (we'll need these in the model)
    print("\nspecial token IDs:")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok:<7} -> {tokenizer.token_to_id(tok)}")

    # Demo: tokenize a real sample from our dataset
    print("\n--- demo on real samples ---")
    with open(args.data, encoding="utf-8") as f:
        sample = json.loads(f.readline())

    for field in ["abstract", "story"]:
        text = sample[field]
        ids = tokenizer.encode(text).ids
        decoded = tokenizer.decode(ids)
        print(f"\n[{field}] first 80 chars of original:")
        print(f"  {text[:80]}...")
        print(f"[{field}] tokenized -> {len(ids)} tokens; first 20 ids: {ids[:20]}")
        print(f"[{field}] decoded back (first 80 chars):")
        print(f"  {decoded[:80]}...")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/pairs.jsonl")
    p.add_argument("--out", default="data/processed/tokenizer.json")
    p.add_argument("--vocab_size", type=int, default=8000)
    main(p.parse_args())
