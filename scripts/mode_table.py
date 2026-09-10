#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.modes import MODES, conditional_stats, mode_distribution  # noqa: E402
from src.paths import analysis_dir, primitive_judgments_glob  # noqa: E402

_PRIMS = ("semantic_leak", "meta_reference", "fluent")


def _read_dicts(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("judgments", nargs="*", help='Judgments.')
    ap.add_argument("--run", default=None,
                    help='Run.')
    ap.add_argument("--by", nargs="+", default=["concept_class", "prompt_id"],
                    help='By.')
    ap.add_argument("--surface", default=None, choices=["answer", "trace"],
                    help='Surface.')
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if not args.run and not args.judgments:
        raise SystemExit("pass --run exp1|round1 (or judgment paths explicitly)")
    pats = args.judgments or [primitive_judgments_glob(args.run)]
    paths = [f for pat in pats for f in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no files matched: {pats}")


    records, skipped, dupes = [], 0, 0
    seen: set[tuple] = set()
    for p in paths:
        for rec in _read_dicts(Path(p)):
            if any(rec.get(k) is None for k in _PRIMS):
                skipped += 1
                continue
            if args.surface and rec.get("surface") not in (None, args.surface):
                continue
            key = (rec.get("generation_id"), rec.get("surface"))
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            records.append(rec)
    if not records:
        raise SystemExit(f"no primitive judgments found (skipped {skipped} rows lacking "
                         f"{_PRIMS} — did you point at the 0-2 prototype judge output?)")

    by = args.by
    dist = mode_distribution(records, by=by)
    cond = conditional_stats(records, by=by)

    dist_cols = list(by) + ["n"] + [c for m in MODES for c in (m, f"{m}_frac")]
    cond_cols = list(by) + ["n", "p_leak", "p_meta", "p_incoherent", "p_guess",
                            "p_leak_given_meta", "p_leak_given_no_meta"]

    out = Path(args.out_dir) if args.out_dir else analysis_dir(args.run)
    _write_csv(out / "mode_distribution.csv", dist, dist_cols)
    _write_csv(out / "conditional_stats.csv", cond, cond_cols)
    print(f"{len(records)} judgments ({skipped} skipped) over {len(paths)} file(s), grouped by {by}"
          + (f"  ({dupes} duplicate row(s) across files ignored)" if dupes else ""))
    print(f"  mode distribution -> {out / 'mode_distribution.csv'}  ({len(dist)} groups)")
    print(f"  conditional stats -> {out / 'conditional_stats.csv'}  (RQ3 P(leak|meta))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
