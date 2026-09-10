from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

SPLITS = ("val", "test")


def read_val_ids(path: str | Path) -> set[str]:


    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no split at {path} — build it first:\n"
                         f"  python scripts/build_split.py --run <run>")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {i for ids in payload.get("val", {}).values() for i in ids}


def in_split(row: Mapping, val_ids: set[str], which: str) -> bool:

    if which not in SPLITS:
        raise SystemExit(f"--split must be one of {SPLITS} (got {which!r})")
    if row.get("probe_type") != "forget":
        return True
    return (row["item_id"] in val_ids) if which == "val" else (row["item_id"] not in val_ids)


def select(rows: Iterable[Mapping], val_ids: set[str], which: str | None) -> list:


    rows = list(rows)
    return rows if which is None else [r for r in rows if in_split(r, val_ids, which)]


def split_counts(rows: Iterable[Mapping]) -> dict[str, int]:

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("probe_type") or "?"] = counts.get(r.get("probe_type") or "?", 0) + 1
    return counts
