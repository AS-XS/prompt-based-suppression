from __future__ import annotations

import csv
import json
from pathlib import Path


def read_dicts(path: Path):


    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
