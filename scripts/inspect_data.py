#!/usr/bin/env python


from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def show(name: str, repo: str, config: str, split: str = "test") -> None:
    from datasets import load_dataset
    try:
        ds = load_dataset(repo, config, split=split)
        print(f"\n[{name}] {repo}:{config} split={split}  rows={len(ds)}")
        print(f"  columns: {ds.column_names}")
        print(f"  row[0] : { {k: (str(v)[:80]) for k, v in ds[0].items()} }")
    except Exception as e:  # noqa: BLE001
        print(f"\n[{name}] {repo}:{config} — FAILED: {e}")


def main() -> int:

    show("RWKU forget QA", "jinzhuoran/RWKU", "forget_level2")
    show("RWKU forget cloze", "jinzhuoran/RWKU", "forget_level1")
    show("RWKU neighbor", "jinzhuoran/RWKU", "neighbor_level1")

    show("MMLU biology", "cais/mmlu", "high_school_biology")
    show("MMLU physics", "cais/mmlu", "high_school_physics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
