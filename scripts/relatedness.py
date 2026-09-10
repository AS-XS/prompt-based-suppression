#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.dependency import pairwise_matrix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", nargs="+", required=True, help='Terms.')
    ap.add_argument("--out", default="results/analysis/relatedness.csv")
    args = ap.parse_args()

    rows = pairwise_matrix(args.terms)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["a", "b", "subtree_overlap", "tree_distance"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
