"""Quick test of the bedtime-story rewrite prompt — runs once on a hard-coded sample."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from config import ANTHROPIC_API_KEY


PROMPT = """Rewrite the following astrophysics abstract as a short, warm story \
that a curious child could follow.

Hard rules:
  - Aim for 100-180 words. SHORTER than the abstract.
  - Convey the actual scientific finding through a concrete image or scene.
  - NO "Once upon a time". NO "The end". NO "close your eyes, little one".
  - NO emojis. NO Markdown headers. Just prose.
  - Keep the warmth and the imagery, lose the storytime cliches.
  - Return only the story.

Abstract:
{abstract}
"""

SAMPLE = (
    "Recent observations of distant galaxies suggest the universe is expanding "
    "faster than predicted by the standard cosmological model, hinting at "
    "previously unknown physics governing the early universe."
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
resp = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=500,
    messages=[{"role": "user", "content": PROMPT.format(abstract=SAMPLE)}],
)
print(resp.content[0].text)