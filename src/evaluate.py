"""
evaluate.py — Quality metrics.

  - Readability drop: Flesch-Kincaid grade level of input vs. output (textstat)
  - Fidelity: does the story preserve the science? (BLEU/semantic similarity)
"""
import textstat


def readability_grade(text: str) -> float:
    """Flesch-Kincaid grade level. Lower = easier to read."""
    return textstat.flesch_kincaid_grade(text)


def readability_drop(abstract: str, story: str) -> float:
    """How many grade levels simpler the story is than the abstract."""
    return readability_grade(abstract) - readability_grade(story)


def fidelity_score(abstract: str, story: str) -> float:
    """TODO: measure semantic overlap (e.g., embedding cosine similarity)."""
    raise NotImplementedError
