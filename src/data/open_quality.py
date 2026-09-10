from __future__ import annotations

import re
from collections.abc import Mapping

from src.analysis.recall import is_hit


_NOUNS = r"(statement|option|choice|answer|change|passage|graph|figure|table|list|item|scenario|reaction|compound|sequence)s?"
_NEEDS_OPTIONS = re.compile(
    rf"""(
        \bwhich\s+of\s+(the\s+|these\s+|those\s+)?(following|these|those|options|choices|statements|answers|pairs?)\b
      | \b(all|each|none|both|one|any)\s+of\s+the\s+(following|above|below)\b
      | \bof\s+the\s+following\b
      | \bthe\s+(following|above|below)\s+{_NOUNS}\b        # "the below changes"
      | \bthe\s+{_NOUNS}\s+(above|below|listed|shown)\b     # "the changes below"
      | \bEXCEPT\b
      | \b(I{{1,3}}|IV|V)\s*(,|and|&)\s*(I{{1,3}}|IV|V)\b    # roman-numeral option sets: I, II, III
      | \bselect\s+all\s+that\s+apply\b
      | \bnot\s+listed\s+(above|below)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def is_self_contained(question: str | None) -> bool:

    if not question or not question.strip():
        return False
    return _NEEDS_OPTIONS.search(question) is None


_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def gold_answer(item: Mapping) -> str | None:


    ref = item.get("reference")
    if ref:
        return str(ref)

    meta = item.get("meta") or {}
    choices = meta.get("choices") or item.get("choices")
    if not choices:
        return None
    key = item.get("answer_key")
    idx = _LETTER_TO_IDX.get(str(key).strip().upper()) if not isinstance(key, int) else key
    if idx is None or not (0 <= idx < len(choices)):
        return None
    return str(choices[idx])


def open_ended_correct(answer: str | None, gold: str | None, threshold: float = 0.6) -> bool:


    return is_hit(answer, gold, threshold=threshold)
