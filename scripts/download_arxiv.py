"""
download_arxiv.py — Download arXiv abstracts (the "complex" side of the corpus).

Options:
  A) Kaggle arXiv dataset (~1.7M abstracts JSON). Download manually from
     https://www.kaggle.com/datasets/Cornell-University/arxiv and place in data/raw/.
  B) Hugging Face: load_dataset("arxiv_dataset") or the official arXiv API.

This script reads the raw dump and writes a clean data/raw/abstracts.jsonl,
optionally filtered to a single field (e.g., physics) for consistency.
"""
import argparse, json, os


def main(args):
    os.makedirs("data/raw", exist_ok=True)
    # TODO: read the raw arXiv dump, filter by category if requested,
    #       and write {"id":..., "abstract":...} lines to args.out
    raise NotImplementedError("TODO: implement arXiv download/cleaning")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--category", default=None, help="e.g. 'physics' to filter")
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--out", default="data/raw/abstracts.jsonl")
    main(p.parse_args())
