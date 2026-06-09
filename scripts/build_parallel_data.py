"""
build_parallel_data.py — Turn data/raw/abstracts.jsonl into data/processed/pairs.jsonl
by rewriting each abstract as a bedtime story via the Claude API.

Designed to survive interruptions:
  - Streams pairs to disk one at a time (no big buffer in memory)
  - Resumes from where it stopped (skips IDs already in pairs.jsonl)
  - Reports running cost
  - Retries with backoff on transient API errors
  - Safe to Ctrl+C and re-run

Usage:
    python scripts/build_parallel_data.py                          # full corpus
    python scripts/build_parallel_data.py --limit 200              # try 200 first
    python scripts/build_parallel_data.py --limit 1000 --start 200 # skip first 200
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os
import time

import anthropic
from anthropic import APIError, APIConnectionError, RateLimitError

from config import ANTHROPIC_API_KEY


MODEL = "claude-haiku-4-5"
PRICE_INPUT_PER_MTOK = 1.00
PRICE_OUTPUT_PER_MTOK = 5.00


PROMPT = """Rewrite the following astrophysics abstract as a bedtime story \
that a curious five-year-old would enjoy hearing at night.

Style guidelines:
  - Warm, gentle, slightly dreamy tone — this is meant to soothe at bedtime.
  - A small narrative arc: a question, a discovery, a quiet resolution.
  - Use concrete imagery and gentle personification (stars, dust, light, \
detectors can all be characters).
  - Convey the actual scientific finding through the story — don't just \
decorate the terms.
  - "Once upon a time" openers and a soft closing ("...and the universe \
hummed on" / "*the end*" / etc.) are welcome — this is a bedtime story.
  - Aim for roughly 200-350 words.

Return only the story.

Abstract:
{abstract}
"""


def load_done_ids(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def rewrite_one(client, abstract, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=800,
                messages=[{"role": "user", "content": PROMPT.format(abstract=abstract)}],
            )
            return resp.content[0].text.strip(), resp.usage
        except RateLimitError:
            wait = 20 * (attempt + 1)
            print(f"    rate-limited, sleeping {wait}s ...", flush=True)
            time.sleep(wait)
        except (APIConnectionError, APIError) as e:
            wait = 10 * (attempt + 1)
            print(f"    API error ({type(e).__name__}), sleeping {wait}s ...", flush=True)
            time.sleep(wait)
    raise RuntimeError("exceeded max retries")


def main(args):
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.src, encoding="utf-8") as f:
        sources = [json.loads(line) for line in f]
    print(f"loaded {len(sources)} source abstracts from {args.src}")

    sources = sources[args.start:]
    if args.limit is not None:
        sources = sources[: args.limit]
    print(f"target this run: {len(sources)} abstracts (start={args.start}, limit={args.limit})")

    done = load_done_ids(args.out)
    print(f"already done in {args.out}: {len(done)} abstracts")
    todo = [s for s in sources if s["id"] not in done]
    print(f"to process now: {len(todo)} abstracts\n")

    if not todo:
        print("nothing to do — all targets already in pairs.jsonl")
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    total_in = 0
    total_out = 0
    cost = 0.0
    t0 = time.time()

    with open(args.out, "a", encoding="utf-8") as f:
        for i, item in enumerate(todo, 1):
            try:
                story, usage = rewrite_one(client, item["abstract"])
            except Exception as e:
                print(f"  [{i}/{len(todo)}] id={item['id']}  FAILED: {e}")
                continue

            pair = {
                "id": item["id"],
                "title": item.get("title", ""),
                "abstract": item["abstract"],
                "story": story,
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            f.flush()

            total_in += usage.input_tokens
            total_out += usage.output_tokens
            cost = (
                total_in  / 1_000_000 * PRICE_INPUT_PER_MTOK
                + total_out / 1_000_000 * PRICE_OUTPUT_PER_MTOK
            )
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta_s = (len(todo) - i) / rate if rate > 0 else 0

            print(
                f"  [{i:>4}/{len(todo)}] id={item['id']:<8}  "
                f"in={usage.input_tokens:>4}  out={usage.output_tokens:>4}  "
                f"running cost ~ ${cost:.2f}  "
                f"ETA {eta_s/60:.1f} min",
                flush=True,
            )

    print(f"\nDone. Wrote {len(todo)} pairs to {args.out}")
    print(f"Total tokens - in: {total_in:,}   out: {total_out:,}")
    print(f"Total cost ~ ${cost:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/raw/abstracts.jsonl")
    p.add_argument("--out", default="data/processed/pairs.jsonl")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    main(p.parse_args())
