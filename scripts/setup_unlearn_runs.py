#!/usr/bin/env python3


import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import generations_dir, items_path, split_path  # noqa: E402

BASES = {"r2mu-e2b": "gemma-4-E2B-it", "r2mu-g31": "gemma-4-31B-it",
         "rmu2": "gemma-4-E2B-it", "rmu3": "Meta-Llama-3-8B-Instruct",
         "rmu4": "gemma-4-31B-it", "elm1": "Meta-Llama-3-8B-Instruct",
         "elm2": "gemma-4-E2B-it", "elm3": "gemma-4-31B-it"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", choices=list(BASES))
    args = ap.parse_args()
    pairs = []
    for run in args.runs:
        pairs += [(items_path("wmdp8"), items_path(run)),
                  (split_path("wmdp8"), split_path(run))]
        sources = sorted(generations_dir("wmdp8").glob(f"{BASES[run]}__none-baseline*.jsonl"))
        if not sources:
            ap.error(f"no wmdp8 baseline files for {BASES[run]}")
        pairs += [(src, generations_dir(run) / src.name) for src in sources]

    for src, dst in pairs:
        if not src.is_file():
            ap.error(f"missing {src}")
        if dst.exists() and src.read_bytes() != dst.read_bytes():
            ap.error(f"different existing file {dst}; inspect instead of overwriting")
    for src, dst in pairs:
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    print(f"Verified/copied {len(pairs)} frozen input and baseline files")


if __name__ == "__main__":
    main()
