#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.agreement import compare, judge_labels, parse_labels  # noqa: E402
from src.io import write_csv  # noqa: E402
from src.paths import analysis_dir  # noqa: E402

COLS = ["blind", "a", "b", "axis", "n", "agreement", "kappa", "both", "a_only", "b_only",
        "neither", "exact_0_1_2"]


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _pct(v) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--dir", default=None,
                    help='Dir.')
    args = ap.parse_args(argv)

    d = Path(args.dir) if args.dir else analysis_dir(args.run) / "judge_sample" / args.sample
    sheets = sorted(p for p in d.glob("*.labels-*.csv")
                    if not p.name.endswith(".labels-blank.csv"))
    if not sheets:
        raise SystemExit(f"no filled label sheets in {d}: copy a `.labels-blank.csv` to "
                         f"`.labels-<initials>.csv` and fill it (the questions are at the top of "
                         f"the matching blind .md)")
    rows, raters = [], defaultdict(dict)
    for sheet in sheets:
        stem, who = sheet.name[:-len(".csv")].split(".labels-", 1)
        try:
            labels = parse_labels(_read(sheet))
        except ValueError as e:
            raise SystemExit(f"{sheet.name}: {e}")
        raters[stem][who] = labels
        keys = sorted(d.glob(f"{stem}.key*.csv"))
        if not keys:
            print(f"⚠️  {sheet.name}: no key beside it — scored against other annotators only")
        for key in keys:
            config = key.name[len(stem) + len(".key"):-len(".csv")].lstrip("-") or "this audit"
            for r in compare(labels, judge_labels(_read(key))):
                rows.append({"blind": stem, "a": who, "b": f"judge {config}", **r})
    for stem, by_who in raters.items():
        for a, b in combinations(sorted(by_who), 2):
            rows.extend({"blind": stem, "a": a, "b": b, **r}
                        for r in compare(by_who[a], by_who[b]))
    write_csv(d / "human_agreement.csv", rows, COLS)

    print(f"{'blind':<34} {'a':<6} {'b':<32} {'axis':<8} {'n':>4} {'agree':>6} {'kappa':>6} "
          f"{'2x2 (both/a/b/neither)':>23} {'exact':>6}")
    for r in rows:
        k = "–" if r["kappa"] is None else f"{r['kappa']:.2f}"
        cells = f"{r['both']}/{r['a_only']}/{r['b_only']}/{r['neither']}"
        print(f"{r['blind']:<34} {r['a']:<6} {r['b']:<32} {r['axis']:<8} {r['n']:>4} "
              f"{_pct(r['agreement']):>6} {k:>6} {cells:>23} {_pct(r['exact_0_1_2']):>6}")
    print(f"\n-> {d / 'human_agreement.csv'}  (κ falls when one class dominates: read it with the "
          f"2x2, not alone)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
