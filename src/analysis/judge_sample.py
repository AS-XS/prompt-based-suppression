from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

from src.analysis.ratings import best_records, mode_of, rating_table  # noqa: F401  (re-export)

SURFACES = ("trace", "explanation")


def one_config(records: Iterable[Mapping]) -> tuple[str | None, str | None, str | None]:


    records = list(records)
    seen = Counter(r.get("judge_config_hash") or "unrecorded" for r in records)
    if len(seen) > 1:
        raise SystemExit(f"judgments from {len(seen)} judge configurations are mixed here "
                         f"({dict(seen)}); report each configuration separately.")
    later = []
    for stage in ("framing", "mention"):
        hashes = Counter(r.get(f"{stage}_config_hash") or "unrecorded" for r in records
                         if r.get(f"{stage}_score") is not None)
        if len(hashes) > 1:
            raise SystemExit(f"{stage} scores from {len(hashes)} {stage} rubrics are mixed here "
                             f"({dict(hashes)}); finish the re-run (judge.py --stage {stage}) "
                             f"before reading it.")
        later.append(next(iter(hashes), None))
    return next(iter(seen), None), later[0], later[1]


def lexical_flag(meta: Mapping, surface: str) -> bool | None:

    novel = meta.get(f"lex_{surface}_novel")
    return novel if novel is not None else meta.get(f"lex_{surface}")


def joined(best: Mapping[tuple, Mapping], manifest_rows: Mapping[str, Mapping],
           incoherent: frozenset | set = frozenset()) -> tuple[list, int]:


    out, foreign = [], 0
    for (gid, surface), rec in best.items():
        meta = manifest_rows.get(gid)
        if meta is None:
            foreign += 1
            continue
        rec = {**rec, **{k: v for k, v in meta.items() if k not in rec},
               "lexical": lexical_flag(meta, surface),
               "incoherent": surface == "trace" and gid in incoherent}
        rec["mode"] = mode_of(rec)
        out.append(rec)
    return out, foreign


def is_incoherent(row: Mapping, loop_score, loop_at: float) -> bool:


    return row.get("finish_reason") == "length" and loop_score(row.get("raw_output") or "") > loop_at


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:

    if not n:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 4), round(centre + half, 4)


def kappa(n11: int, n10: int, n01: int, n00: int) -> float | None:

    n = n11 + n10 + n01 + n00
    if not n:
        return None
    po = (n11 + n00) / n
    pe = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def _scored(r: Mapping) -> bool:
    return r.get("concept_score") not in ("", None)


def leak_table(rows: Sequence[Mapping], by: Sequence[str]) -> list[dict]:


    out = rating_table(rows, by=by)
    groups = defaultdict(list)
    for r in rows:
        if _scored(r):
            groups[tuple(r.get(k) for k in by)].append(r)
    for row in out:
        g = groups.get(tuple(row[k] for k in by), [])
        leaks = sum(int(r["concept_score"]) > 0 for r in g)
        row["leak_lo"], row["leak_hi"] = wilson(leaks, len(g))
        lex = [r["lexical"] for r in g if r.get("lexical") is not None]
        row["lexical_rate"] = round(sum(lex) / len(lex), 4) if lex else None
        row["n_lexical"] = len(lex)
    return out


def surface_table(rows: Sequence[Mapping], by: Sequence[str]) -> list[dict]:


    pairs = defaultdict(dict)
    for r in rows:
        if _scored(r) and r.get("surface") in SURFACES:
            pairs[r["generation_id"]][r["surface"]] = r
    groups = defaultdict(lambda: Counter())
    for surf in pairs.values():
        if len(surf) < 2:
            continue
        t, e = surf["trace"], surf["explanation"]
        c = groups[tuple(t.get(k) for k in by)]
        c["n_paired"] += 1
        jt, je = int(t["concept_score"]) > 0, int(e["concept_score"]) > 0
        c["both" if jt and je else "trace_only" if jt else "explanation_only" if je
          else "neither"] += 1
        if t.get("lexical") is not None and e.get("lexical") is not None:
            lt, le = bool(t["lexical"]), bool(e["lexical"])
            c["lex_n"] += 1
            c["lex_" + ("both" if lt and le else "trace_only" if lt else
                        "explanation_only" if le else "neither")] += 1
    cols = ("n_paired", "both", "trace_only", "explanation_only", "neither", "lex_n", "lex_both",
            "lex_trace_only", "lex_explanation_only", "lex_neither")
    return [{**dict(zip(by, k)), **{col: c.get(col, 0) for col in cols}}
            for k, c in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]])]


def agreement_table(rows: Sequence[Mapping], by: Sequence[str] = ("concept_id", "surface")):


    groups = defaultdict(lambda: Counter())
    for r in rows:
        if not _scored(r) or r.get("lexical") is None:
            continue
        c = groups[tuple(r.get(k) for k in by)]
        lex = bool(r["lexical"])
        for reading, judged in (("any", int(r["concept_score"]) > 0),
                                ("used", r.get("mode") == "leaked")):
            c[f"{reading}_{'1' if lex else '0'}{'1' if judged else '0'}"] += 1
    out = []
    for k, c in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = dict(zip(by, k))
        for reading in ("any", "used"):
            n11, n10 = c[f"{reading}_11"], c[f"{reading}_10"]
            n01, n00 = c[f"{reading}_01"], c[f"{reading}_00"]
            n = n11 + n10 + n01 + n00
            row.update({f"{reading}_n": n, f"{reading}_both": n11,
                        f"{reading}_lexical_only": n10, f"{reading}_judge_only": n01,
                        f"{reading}_neither": n00,
                        f"{reading}_agreement": round((n11 + n00) / n, 4) if n else None,
                        f"{reading}_kappa": kappa(n11, n10, n01, n00)})
        out.append(row)
    return out


def outcome_of(baseline_correct: bool | None, suppressed_correct: bool | None) -> str:

    if baseline_correct is None or suppressed_correct is None:
        return "unparsed"
    if not baseline_correct:
        return "no_headroom"
    return "held" if suppressed_correct else "removed"
