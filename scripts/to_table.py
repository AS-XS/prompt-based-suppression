#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

COLS = ["model", "concept_class", "concept_id", "probe_type", "condition", "think_state",
        "prompt_id", "finish_reason", "prompt_tokens", "completion_tokens", "preview", "raw_output"]


def _rows(paths: list[str]):
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help='Inputs.')
    ap.add_argument("-o", "--out", default=None, help='Out.')
    ap.add_argument("--preview-len", type=int, default=140)
    args = ap.parse_args()

    paths = [f for pat in args.inputs for f in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no files matched: {args.inputs}")
    out = Path(args.out) if args.out else Path(paths[0]).with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in _rows(paths):
            r["preview"] = " ".join(r.get("raw_output", "").split())[:args.preview_len]
            w.writerow(r)
            n += 1
    print(f"wrote {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
