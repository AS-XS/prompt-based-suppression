from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

from src.analysis.surfaces import answer_of

_WS = re.compile(r"\s+")
_PUNC = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(?:a|an|the)\b")


def normalize(text: str | None) -> str:
    return _WS.sub(" ", (text or "").lower().translate(_PUNC)).strip()


def _tokens(text: str | None) -> list[str]:
    return normalize(text).split()


def squad_normalize(text: str | None) -> str:

    return _WS.sub(" ", _ARTICLES.sub(" ", normalize(text))).strip()


def exact_match(output: str | None, reference: str | None) -> bool:


    return squad_normalize(output) == squad_normalize(reference)


def token_f1(output: str | None, reference: str | None) -> float:


    pred, gold = squad_normalize(output).split(), squad_normalize(reference).split()
    if not pred or not gold:
        return round(float(pred == gold), 4)
    same = sum((Counter(pred) & Counter(gold)).values())
    if not same:
        return 0.0
    p, r = same / len(pred), same / len(gold)
    return round(2 * p * r / (p + r), 4)


def token_recall(output: str | None, reference: str | None) -> float:

    ref = _tokens(reference)
    if not ref:
        return 0.0
    out = set(_tokens(output))
    return round(sum(1 for t in ref if t in out) / len(ref), 4)


def is_hit(output: str | None, reference: str | None, threshold: float = 0.6) -> bool:

    nref = normalize(reference)
    if nref and nref in normalize(output):
        return True
    return token_recall(output, reference) >= threshold


def score_recall(gen_rows: Iterable[Mapping], item_index: Mapping[str, Mapping],
                 threshold: float = 0.6) -> list[dict]:

    scored = []
    for g in gen_rows:
        it = item_index.get(g["item_id"])
        if not it or not it.get("reference"):
            continue
        ref = it["reference"]
        ans = answer_of(g.get("raw_output"), expect_trace=g.get("think_state") == "on")
        scored.append({
            "model": g.get("model"), "prompt_id": g.get("prompt_id"), "condition": g.get("condition"),
            "think_state": g.get("think_state"), "concept_id": g.get("concept_id"),
            "probe_type": it.get("probe_type"), "item_id": g["item_id"],
            "recall": token_recall(ans, ref),
            "hit": is_hit(ans, ref, threshold),
            "f1": token_f1(ans, ref),
            "em": exact_match(ans, ref),
        })
    return scored


def _key(rec: Mapping, by: Sequence[str]) -> tuple:
    return tuple(rec.get(k) for k in by)


def recall_table(scored: Iterable[Mapping],
                 by: Sequence[str] = ("model", "think_state", "prompt_id", "probe_type")):

    g: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "recall": 0.0, "hit": 0, "f1": 0.0, "em": 0})
    for r in scored:
        c = g[_key(r, by)]
        c["n"] += 1
        c["recall"] += r["recall"]
        c["hit"] += bool(r["hit"])
        c["f1"] += r.get("f1", 0.0)
        c["em"] += bool(r.get("em"))
    rows = []
    for k, c in sorted(g.items(), key=lambda kv: [str(x) for x in kv[0]]):
        n = c["n"]
        rows.append({**dict(zip(by, k)), "n": n,
                     "mean_recall": round(c["recall"] / n, 4) if n else None,
                     "hit_rate": round(c["hit"] / n, 4) if n else None,
                     "mean_f1": round(c["f1"] / n, 4) if n else None,
                     "em_rate": round(c["em"] / n, 4) if n else None})
    return rows


def recall_delta(scored: Iterable[Mapping],
                 by: Sequence[str] = ("model", "think_state", "concept_id"),
                 baseline_prompt: str = "none-baseline", metric: str = "recall"):


    cell: dict[tuple, dict[str, float]] = defaultdict(dict)
    prompts: set[str] = set()
    for r in scored:
        prompts.add(r["prompt_id"])
        val = {"recall": float(r["recall"]), "hit": float(bool(r["hit"])),
               "f1": float(r.get("f1", 0.0)), "em": float(bool(r.get("em")))}[metric]
        cell[(_key(r, by), r["prompt_id"])][r["item_id"]] = val

    supp = sorted(prompts - {baseline_prompt})
    groups = sorted({k for (k, _) in cell}, key=lambda t: [str(x) for x in t])
    rows = []
    for grp in groups:
        base = cell.get((grp, baseline_prompt), {})
        row = {**dict(zip(by, grp)),
               "baseline": round(sum(base.values()) / len(base), 4) if base else None,
               "n_baseline": len(base)}
        for sp in supp:
            sup = cell.get((grp, sp), {})
            shared = [i for i in sup if i in base]
            if shared:
                b = sum(base[i] for i in shared) / len(shared)
                s = sum(sup[i] for i in shared) / len(shared)
                row[f"{sp}"] = round(s, 4)
                row[f"{sp}_drop"] = round(b - s, 4)
                row[f"{sp}_n"] = len(shared)
            else:
                row[f"{sp}"] = row[f"{sp}_drop"] = None
                row[f"{sp}_n"] = 0
        rows.append(row)
    return rows, supp
