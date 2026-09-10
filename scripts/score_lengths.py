#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.lengths import length_table  # noqa: E402
from src.gen.runner import iter_jsonl  # noqa: E402
from src.paths import analysis_dir, generations_glob, items_path  # noqa: E402


def _format_index(run: str) -> dict[str, str] | None:


    ip = items_path(run)
    if not ip.exists():
        return None
    idx = {}
    for it in iter_jsonl(ip):
        fmt = (it.get("meta") or {}).get("format") or it.get("format")
        if fmt:
            idx[it["item_id"]] = fmt
    return idx or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp1")
    args = ap.parse_args()

    paths = sorted(glob.glob(generations_glob(args.run)))
    if not paths:
        raise SystemExit(f"no generations for run {args.run!r} ({generations_glob(args.run)})")

    fmt = _format_index(args.run)


    forms = sorted({v for v in fmt.values()}) if fmt else []
    print("format: " + (f"{'/'.join(forms)} (from items.jsonl)" if forms else
                        "POOLED — items.jsonl not found, so forms can't be separated"))
    rows = []
    for p in paths:
        gens = iter_jsonl(p)
        rows += length_table(gens, format_of=fmt)

    out = analysis_dir(args.run) / "answer_lengths.csv"
    out.parent.mkdir(parents=True, exist_ok=True)


    lead = ["model", "think_state", "concept_class", "probe_type", "prompt_id"] \
        + (["format"] if fmt else [])
    seen = {k for r in rows for k in r}
    cols = [c for c in lead if c in seen] + [c for c in rows[0] if c not in lead] if rows else lead
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"answer lengths -> {out}  ({len(rows)} groups over {len(paths)} model file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
