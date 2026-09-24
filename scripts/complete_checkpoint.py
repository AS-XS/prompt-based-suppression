#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import UNLEARNED_ROOT, unlearn_report_path  # noqa: E402
from src.unlearn.rmu import complete_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help='Names.')
    args = ap.parse_args()

    for name in args.names:
        d = UNLEARNED_ROOT / name
        report = unlearn_report_path(d)
        if not report.is_file():
            raise SystemExit(f"no {report} — cannot tell which base model {name} came from")
        base = Path(json.loads(report.read_text(encoding="utf-8")).get("model") or "")
        if not base.is_dir():
            raise SystemExit(f"{name}: its recorded base {base} is not a directory")
        copied = complete_checkpoint(base, d)
        print(f"{name}  <-  {base}:  " + (", ".join(copied) if copied else "already complete"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
