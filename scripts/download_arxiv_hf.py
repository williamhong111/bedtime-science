"""
download_arxiv_hf.py — Download astro-ph (astrophysics) abstracts from
Hugging Face. Streams UniverseTBD/arxiv-abstracts-large and keeps only
papers whose category list includes 'astro-ph'.
"""
import argparse
import json
import os

from datasets import load_dataset


def looks_like_astro(cats) -> bool:
    """Check whether a paper's category field contains astro-ph."""
    if not cats:
        return False
    if isinstance(cats, list):
        return any("astro-ph" in c for c in cats)
    return "astro-ph" in str(cats)


def main(args):
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Streaming UniverseTBD/arxiv-abstracts-large (filtering astro-ph) ...")
    ds = load_dataset(
        "UniverseTBD/arxiv-abstracts-large",
        split="train",
        streaming=True,
    )

    written = 0
    scanned = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for paper in ds:
            scanned += 1
            cats = paper.get("categories") or paper.get("category") or ""
            if not looks_like_astro(cats):
                continue
            abstract = (paper.get("abstract") or "").strip()
            title = (paper.get("title") or "").strip()
            if not abstract:
                continue
            rec = {
                "id": str(paper.get("id", scanned)),
                "title": title,
                "abstract": abstract,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written % 200 == 0:
                print(f"  written: {written}  (scanned {scanned})")
            if written >= args.limit:
                break

    print(f"\nDone. Wrote {written} abstracts (scanned {scanned}) to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--out", default="data/raw/abstracts.jsonl")
    main(p.parse_args())
