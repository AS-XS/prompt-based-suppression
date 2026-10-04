from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache


MIN_N = 6

_WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str | None) -> list[str]:


    return _WORD.findall((text or "").lower())


@dataclass(frozen=True)
class Recitation:


    n: int
    n_instruction: int
    n_trace: int
    longest_run: int
    n_grams_matched: int
    n_grams_total: int

    @property
    def coverage(self) -> float | None:


        if not self.n_grams_total:
            return None
        return self.n_grams_matched / self.n_grams_total

    @property
    def recites(self) -> bool:

        return self.longest_run >= self.n


def _ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _index(toks: list[str], n: int) -> dict[tuple[str, ...], list[int]]:
    idx: dict[tuple[str, ...], list[int]] = {}
    for i in range(len(toks) - n + 1):
        idx.setdefault(tuple(toks[i:i + n]), []).append(i)
    return idx


def compare(instruction: str | None, trace: str | None, *, n: int = MIN_N) -> Recitation:


    ins, tr = tokens(instruction), tokens(trace)
    if len(ins) < n or len(tr) < n:
        return Recitation(n=n, n_instruction=len(ins), n_trace=len(tr), longest_run=0,
                          n_grams_matched=0, n_grams_total=max(0, len(ins) - n + 1))

    idx = _index(ins, n)
    total = len(_ngrams(ins, n))
    matched: set[tuple[str, ...]] = set()
    longest = 0
    for j in range(len(tr) - n + 1):
        gram = tuple(tr[j:j + n])
        starts = idx.get(gram)
        if not starts:
            continue
        matched.add(gram)
        for i in starts:
            k = n
            while i + k < len(ins) and j + k < len(tr) and ins[i + k] == tr[j + k]:
                k += 1
            longest = max(longest, k)
    return Recitation(n=n, n_instruction=len(ins), n_trace=len(tr), longest_run=longest,
                      n_grams_matched=len(matched), n_grams_total=total)


@lru_cache(maxsize=4096)
def instruction_text(prompt_id: str, concept_id: str) -> str | None:


    from src.prompts import build_messages, is_directed_prompt, split_prompt_id, takes_examples

    template_id = split_prompt_id(prompt_id)[0]
    try:
        needs = is_directed_prompt(template_id) and takes_examples(template_id)
        messages = build_messages(prompt_id, concept_id, "", [("", "")] if needs else None)
    except (KeyError, ValueError):
        return None
    text = " ".join(m["content"] for m in messages).strip()
    return text or None
