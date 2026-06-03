"""
preview_pair.py — Rewrite one random astrophysics abstract as a bedtime story.

Prints the abstract and the story side by side so we can sanity-check
quality before running the full batch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import textwrap

import anthropic
from config import ANTHROPIC_API_KEY


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


def main():
    with open("data/raw/abstracts.jsonl", encoding="utf-8") as f:
        abstracts = [json.loads(line) for line in f]
    print(f"loaded {len(abstracts)} abstracts; picking one at random...\n")

    item = random.choice(abstracts)
    abstract = item["abstract"]

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=800,
        messages=[{"role": "user", "content": PROMPT.format(abstract=abstract)}],
    )
    story = resp.content[0].text.strip()

    def wrap(s):
        return "\n".join(textwrap.fill(line, 80) for line in s.splitlines())

    print("=" * 80)
    if item.get("title"):
        print(f"TITLE: {item['title']}")
        print("=" * 80)
    print(f"ABSTRACT ({len(abstract.split())} words)")
    print("=" * 80)
    print(wrap(abstract))
    print()
    print("=" * 80)
    print(f"BEDTIME STORY ({len(story.split())} words)")
    print("=" * 80)
    print(wrap(story))
    print()


if __name__ == "__main__":
    main()
