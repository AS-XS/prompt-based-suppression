#!/usr/bin/env python


from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import judgments_dir  # noqa: E402


def family(rec: dict) -> str:
    if rec.get("condition") == "baseline":
        return "baseline"
    return "decoy" if "decoy" in (rec.get("prompt_id") or "") else "other"


def drift(records, stage: str, old: str, new: str) -> dict[tuple, Counter]:

    scores: dict[str, dict] = {old: {}, new: {}}
    fam = {}
    for d in records:
        k = (d.get("generation_id"), d.get("surface"))
        fam[k] = family(d)
        if d.get(f"{stage}_score") is not None and d.get(f"{stage}_config_hash") in scores:
            scores[d[f"{stage}_config_hash"]][k] = d[f"{stage}_score"]
    out: dict[tuple, Counter] = {}
    for k, before in scores[old].items():
        if k in scores[new]:
            out.setdefault((k[1], fam[k]), Counter())[(before, scores[new][k])] += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--stage", choices=("framing", "mention"), default="framing")
    ap.add_argument("--old", required=True, help='Old.')
    ap.add_argument("--new", required=True, help='New.')
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(judgments_dir(args.run) / f"{args.sample}*.jsonl")))
    if not files:
        raise SystemExit(f"no judgments for {args.sample} under {judgments_dir(args.run)}")
    records = (json.loads(line) for f in files for line in open(f, encoding="utf-8"))
    table = drift(records, args.stage, args.old, args.new)
    if not table:
        raise SystemExit(f"no row carries {args.stage} scores under both {args.old} and {args.new}")
    print(f"{args.stage}: {args.old[:8]} -> {args.new[:8]}, (old, new): rows")
    for (surface, fam), moves in sorted(table.items()):
        print(f"  {surface:<12} {fam:<9} n={sum(moves.values()):<5} "
              f"{dict(sorted(moves.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
