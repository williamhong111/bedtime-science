"""
build_parallel_data.py — Build the (abstract -> story) parallel corpus.

The "simple" side does not exist as a dataset, so we construct it: each abstract
is rewritten into a child-friendly bedtime story.

IMPORTANT: any large model used here is ONLY a data-construction tool. The model
we train/deliver is built in src/. This script just produces training data.

Prompt guidelines for the rewrite:
  - vocabulary a five-year-old knows
  - concrete imagery and gentle personification
  - a small narrative arc (beginning / middle / end)
  - stay faithful to what the science actually says

Output: data/processed/pairs.jsonl with {"abstract":..., "story":...} per line.
"""
import argparse, json, os


def rewrite_to_story(abstract: str) -> str:
    """TODO: call your chosen LLM with the bedtime-story prompt and return the story."""
    raise NotImplementedError


def main(args):
    os.makedirs("data/processed", exist_ok=True)
    # TODO: read abstracts, rewrite each, write pairs to args.out
    raise NotImplementedError("TODO: implement parallel-data construction")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", default="data/raw/abstracts.jsonl")
    p.add_argument("--out", default="data/processed/pairs.jsonl")
    main(p.parse_args())
