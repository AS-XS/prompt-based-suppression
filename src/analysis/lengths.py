from __future__ import annotations

import statistics as st
from collections.abc import Iterable, Mapping, Sequence

from src.analysis.surfaces import answer_of

BARE_MAX = 3


def length_table(gen_rows: Iterable[Mapping],
                 by: Sequence[str] = ("model", "think_state", "concept_class", "probe_type",
                                      "prompt_id"),
                 format_of: Mapping[str, str] | None = None):


    keyed = tuple(by) + (("format",) if format_of is not None else ())
    buckets: dict[tuple, list[int]] = {}
    trunc: dict[tuple, int] = {}
    for r in gen_rows:
        key = tuple(r.get(k) for k in by)
        if format_of is not None:
            key = key + (format_of.get(r.get("item_id"), "unknown"),)
        buckets.setdefault(key, []).append(len(answer_of(r.get("raw_output")) or ""))


        trunc[key] = trunc.get(key, 0) + (r.get("finish_reason") == "length")
    by = keyed
    rows = []
    for key, lens in buckets.items():
        n = len(lens)
        rows.append({
            **dict(zip(by, key)),
            "n": n,
            "median_chars": int(st.median(lens)),
            "mean_chars": round(st.mean(lens), 1),
            "bare_pct": round(sum(x <= BARE_MAX for x in lens) / n, 4),
            "empty_pct": round(sum(x == 0 for x in lens) / n, 4),


            "truncated_pct": round(trunc.get(key, 0) / n, 4),
        })
    return rows
