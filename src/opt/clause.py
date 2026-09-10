from __future__ import annotations

import re


_META = re.compile(
    r"\b(i propose|i suggest|here is (an|the) instruction|design (a|an) (language model )?instruction"
    r"|the following instruction|an instruction that|instruction should encourage"
    r"|predict module|language model instruction|the lm should|this prompt should)\b",
    re.IGNORECASE,
)

_INTERNALS = re.compile(r"\b(predict module|dspy|signature|few-shot demos?|the module)\b", re.IGNORECASE)


def looks_like_meta(clause: str | None) -> bool:

    if not clause or not clause.strip():
        return False
    return bool(_META.search(clause) or _INTERNALS.search(clause))


def meta_reason(clause: str) -> str:

    for rx, label in ((_META, "describes an instruction rather than being one"),
                      (_INTERNALS, "leaks DSPy's internal vocabulary")):
        m = rx.search(clause or "")
        if m:
            return f"{label}: {m.group(0)!r}"
    return ""


def best_wellformed(candidates: list[dict]) -> tuple[str | None, float | None, int]:


    scored = []
    for c in candidates:
        prog = c.get("program") if isinstance(c, dict) else None
        sig = getattr(prog, "signature", None)
        text = getattr(sig, "instructions", None)
        if text is None or c.get("score") is None:
            continue
        scored.append((float(c["score"]), text))
    rejected = sum(1 for _, t in scored if looks_like_meta(t))
    ok = [(s, t) for s, t in scored if not looks_like_meta(t)]
    if not ok:
        return None, None, rejected
    score, text = max(ok, key=lambda st: st[0])
    return text, score, rejected


_INSTRUMENT = __import__("re").compile(
    r"(answer\s*:\s*<?\s*letter|why\s*:\s*<?\s*one or two|single[- ]letter answer|"
    r"format your response|in exactly this format|"
    r"a single letter|as a single letter|one letter|"
    r"strict format|following format|in the format|this format|"
    r"brief explanation|one or two sentences)", __import__("re").IGNORECASE)


def restates_instrument(clause: str | None, seed: str | None = None) -> bool:


    if not clause:
        return False
    if seed and _INSTRUMENT.search(seed):
        return False
    return bool(_INSTRUMENT.search(clause))
