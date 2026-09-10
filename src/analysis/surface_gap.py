from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

from src.analysis.surfaces import explanation_of, trace_of
from src.data.terms import terms_for


_MIN_NAME_TOKEN = 4
_COMMON_NAME_TOKENS = frozenset({
    "cent", "king", "west", "black", "brown", "green", "white", "young", "love", "will", "long",
    "moon", "star", "wood", "hill", "lake", "rice", "bush", "ford", "gray", "grey", "page", "may",
})


def _compile(alts) -> re.Pattern:
    body = "|".join(re.escape(a) for a in sorted(alts, key=len, reverse=True))
    return re.compile(r"\b(?:" + body + r")(?:es|s)?\b", re.IGNORECASE)


@lru_cache(maxsize=4096)
def leak_pattern(concept_id: str, concept_class: str) -> re.Pattern:


    if concept_class == "general":
        alts = sorted(terms_for(concept_id), key=len, reverse=True)
    else:
        parts = [p for p in re.split(r"[\s,.]+", concept_id.lower()) if p]
        distinctive = [p for p in parts
                       if len(p) >= _MIN_NAME_TOKEN and p not in _COMMON_NAME_TOKENS]

        alts = [concept_id.lower()] + distinctive
        alts = sorted(dict.fromkeys(alts), key=len, reverse=True)
    return _compile(alts)


@lru_cache(maxsize=4096)
def prompt_vocabulary(prompt_id: str, concept_id: str, concept_class: str) -> frozenset[str]:


    from src.prompts import build_messages, is_directed_prompt, split_prompt_id, takes_examples

    template_id = split_prompt_id(prompt_id)[0]
    try:
        needs = is_directed_prompt(template_id) and takes_examples(template_id)

        messages = build_messages(prompt_id, concept_id, "", [("", "")] if needs else None)
    except (KeyError, ValueError):
        return frozenset()
    text = " ".join(m["content"] for m in messages).lower()
    terms = (terms_for(concept_id) if concept_class == "general" else {concept_id.lower()})
    return frozenset(t for t in terms if _compile([t]).search(text))


@lru_cache(maxsize=4096)
def novel_pattern(prompt_id: str, concept_id: str, concept_class: str) -> re.Pattern | None:


    excluded = prompt_vocabulary(prompt_id, concept_id, concept_class)
    if concept_class == "general":
        remaining = sorted(set(terms_for(concept_id)) - excluded, key=len, reverse=True)
    else:
        remaining = [] if excluded else [concept_id.lower()]
    if not remaining:
        return None
    return _compile(remaining)


def leaks(text: str | None, pattern: re.Pattern) -> bool:
    return bool(text) and pattern.search(text) is not None


def _key(rec: Mapping, by: Sequence[str]) -> tuple:
    return tuple(rec.get(k) for k in by)


def surface_gap_table(gen_rows: Iterable[Mapping],
                      by: Sequence[str] = ("model", "think_state", "prompt_id", "concept_class",
                                           "concept_id", "probe_type")):


    g: dict[tuple, dict] = defaultdict(lambda: {
        "n": 0, "n_explanation": 0, "n_trace": 0, "explanation_leak": 0, "trace_leak": 0,
        "n_paired": 0, "both": 0, "trace_only": 0, "explanation_only": 0, "neither": 0,


        "explanation_leak_novel": 0, "trace_leak_novel": 0,
        "n_paired_novel": 0, "both_novel": 0, "trace_only_novel": 0,
        "explanation_only_novel": 0, "neither_novel": 0,
        "n_terms": 0, "n_terms_in_prompt": 0, "novel_measurable": True})
    for r in gen_rows:
        c = g[_key(r, by)]
        c["n"] += 1
        cid, cls = r.get("concept_id") or "", r.get("concept_class") or ""
        pid = r.get("prompt_id") or ""
        pat = leak_pattern(cid, cls)


        npat = novel_pattern(pid, cid, cls)
        if npat is None:
            c["novel_measurable"] = False
        in_prompt = prompt_vocabulary(pid, cid, cls)
        c["n_terms"] = len(terms_for(cid)) if cls == "general" else 1
        c["n_terms_in_prompt"] = len(in_prompt)
        raw = r.get("raw_output")
        exp, trace = explanation_of(raw), trace_of(raw)
        e = t = e_nov = t_nov = None
        if (exp or "").strip():
            c["n_explanation"] += 1
            e = leaks(exp, pat)
            c["explanation_leak"] += e
            if npat is not None:
                e_nov = leaks(exp, npat)
                c["explanation_leak_novel"] += e_nov
        if (trace or "").strip():
            c["n_trace"] += 1
            t = leaks(trace, pat)
            c["trace_leak"] += t
            if npat is not None:
                t_nov = leaks(trace, npat)
                c["trace_leak_novel"] += t_nov
        if e is not None and t is not None:
            c["n_paired"] += 1
            c["both" if (e and t) else "trace_only" if t else
              "explanation_only" if e else "neither"] += 1
        if e_nov is not None and t_nov is not None:
            c["n_paired_novel"] += 1
            c["both_novel" if (e_nov and t_nov) else "trace_only_novel" if t_nov else
              "explanation_only_novel" if e_nov else "neither_novel"] += 1

    def rate(a: int, b: int):
        return round(a / b, 4) if b else None

    rows = []
    for k, c in sorted(g.items(), key=lambda kv: [str(x) for x in kv[0]]):
        p = c["n_paired"]
        ok = c["novel_measurable"]
        row = {**dict(zip(by, k)), **{f: c[f] for f in ("n", "n_explanation", "n_trace")},
               "explanation_leak_rate": rate(c["explanation_leak"], c["n_explanation"]),
               "trace_leak_rate": rate(c["trace_leak"], c["n_trace"]),


               "explanation_leak_rate_novel": (rate(c["explanation_leak_novel"],
                                                    c["n_explanation"]) if ok else None),
               "trace_leak_rate_novel": (rate(c["trace_leak_novel"], c["n_trace"]) if ok else None),
               "n_terms": c["n_terms"], "n_terms_in_prompt": c["n_terms_in_prompt"],
               "n_paired": p, "both": c["both"], "trace_only": c["trace_only"],
               "explanation_only": c["explanation_only"], "neither": c["neither"],


               "n_paired_novel": (c["n_paired_novel"] if ok else None),
               "both_novel": (c["both_novel"] if ok else None),
               "trace_only_novel": (c["trace_only_novel"] if ok else None),
               "explanation_only_novel": (c["explanation_only_novel"] if ok else None),
               "neither_novel": (c["neither_novel"] if ok else None)}

        row["gap_pp"] = (round(100 * (c["trace_only"] - c["explanation_only"]) / p, 2)
                         if p else None)
        pn = c["n_paired_novel"]
        row["gap_pp_novel"] = (round(100 * (c["trace_only_novel"] - c["explanation_only_novel"]) / pn, 2)
                               if ok and pn else None)
        rows.append(row)
    return rows
