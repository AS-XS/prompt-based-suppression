from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from src.analysis.surfaces import answer_of


_TAG = re.compile(r"<answer>\s*\(?\s*([A-D])\b", re.IGNORECASE)
_ANS = re.compile(r"answer\b[\s:*_`~\-–]*(?:is\b)?[\s:*_`~\-–]*\(?\s*([A-D])\b", re.IGNORECASE)
_PAREN = re.compile(r"\(\s*([A-D])\s*\)")
_LEAD = re.compile(r"^\s*\**\(?([A-D])\)?[\.\):\-]", re.MULTILINE)
_STANDALONE = re.compile(r"\b([A-D])\b")
_ARTICLE_AFTER = re.compile(r"\s+[a-z]")


def _is_article(text: str, m: "re.Match") -> bool:


    if m.group(1).upper() != "A":
        return False
    return bool(_ARTICLE_AFTER.match(text[m.end(1):]))


TIERS = ("tag", "answer_label", "paren", "lead", "scan")
STRICT_TIERS = ("tag", "answer_label", "lead")   


def extract_letter_tiered(text: str | None) -> tuple[str | None, str | None]:


    if not text:
        return None, None
    for tier, rx in zip(TIERS, (_TAG, _ANS, _PAREN, _LEAD, _STANDALONE)):
        for m in rx.finditer(text):
            if _is_article(text, m):
                continue
            return m.group(1).upper(), tier
    return None, None


def extract_letter(text: str | None) -> str | None:

    return extract_letter_tiered(text)[0]


def score_generations(gen_rows: Iterable[Mapping], item_index: Mapping[str, Mapping]) -> list[dict]:


    scored = []
    for g in gen_rows:
        it = item_index.get(g["item_id"])
        if not it or it.get("format") != "mc" or not it.get("answer_key"):
            continue


        pred, tier = extract_letter_tiered(
            answer_of(g.get("raw_output"), expect_trace=g.get("think_state") == "on"))
        correct = None if pred is None else (pred == it["answer_key"])
        scored.append({
            "model": g.get("model"), "prompt_id": g.get("prompt_id"), "condition": g.get("condition"),
            "think_state": g.get("think_state"), "concept_id": g.get("concept_id"),
            "mmlu_subject": it.get("mmlu_subject"), "probe_type": it.get("probe_type"),
            "item_id": g["item_id"], "pred": pred, "parsed": pred is not None, "correct": correct,
            "parse_tier": tier, "strict": tier in STRICT_TIERS,
        })
    return scored


def _key(rec: Mapping, by: Sequence[str]) -> tuple:
    return tuple(rec.get(k) for k in by)


def accuracy_table(scored: Iterable[Mapping],
                   by: Sequence[str] = ("model", "think_state", "mmlu_subject", "prompt_id",
                                        "probe_type")):


    g: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "parsed": 0, "correct": 0, "strict": 0})
    for r in scored:
        c = g[_key(r, by)]
        c["n"] += 1
        c["parsed"] += bool(r["parsed"])
        c["correct"] += bool(r["correct"])
        c["strict"] += bool(r.get("strict"))
    rows = []
    for k, c in sorted(g.items(), key=lambda kv: [str(x) for x in kv[0]]):
        n, p, cor, st = c["n"], c["parsed"], c["correct"], c["strict"]
        rows.append({**dict(zip(by, k)), "n": n, "n_parsed": p, "n_correct": cor,
                     "accuracy": round(cor / p, 4) if p else None,
                     "accuracy_strict": round(cor / n, 4) if n else None,
                     "parse_rate": round(p / n, 4) if n else None,
                     "format_rate": round(st / p, 4) if p else None})
    return rows


def delta_table(scored: Iterable[Mapping],
                by: Sequence[str] = ("model", "think_state", "mmlu_subject"),
                baseline_prompt: str = "none-baseline",
                contrast: str = "prompt_id"):


    if contrast in by:
        raise ValueError(
            f"contrast={contrast!r} is also a grouping key in by={tuple(by)}. Every group would then "
            f"hold exactly one level, nothing would pair, and the table would come out with no drop "
            f"columns — silently. Drop it from `by`.")

    cell: dict[tuple, dict[str, bool]] = defaultdict(dict)
    prompts: set[str] = set()
    for r in scored:
        if not r["parsed"]:
            continue
        prompts.add(r[contrast])
        cell[(_key(r, by), r[contrast])][r["item_id"]] = bool(r["correct"])

    supp_prompts = sorted(prompts - {baseline_prompt})
    groups = sorted({k for (k, _) in cell}, key=lambda t: [str(x) for x in t])
    rows = []
    for grp in groups:
        base = cell.get((grp, baseline_prompt), {})
        row = {**dict(zip(by, grp)),
               "baseline_acc": round(sum(base.values()) / len(base), 4) if base else None,
               "n_baseline": len(base)}
        for sp in supp_prompts:
            sup = cell.get((grp, sp), {})
            shared = [i for i in sup if i in base]
            if shared:
                b = sum(base[i] for i in shared) / len(shared)
                s = sum(sup[i] for i in shared) / len(shared)
                row[f"{sp}_acc"] = round(s, 4)
                row[f"{sp}_drop"] = round(b - s, 4)
                row[f"{sp}_n"] = len(shared)
            else:
                row[f"{sp}_acc"] = row[f"{sp}_drop"] = None
                row[f"{sp}_n"] = 0
        rows.append(row)
    return rows, supp_prompts
