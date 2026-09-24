#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import UNLEARNED_ROOT, unlearn_report_path  # noqa: E402
from src.unlearn.rmu import complete_checkpoint  # noqa: E402
from src.unlearn.tensors import carry_over_missing, missing_tensors  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help='Names.')
    ap.add_argument("--check", action="store_true",
                    help='Check.')
    args = ap.parse_args()

    incomplete = 0
    for name in args.names:
        d = UNLEARNED_ROOT / name
        report = unlearn_report_path(d)
        if not report.is_file():
            raise SystemExit(f"no {report} — cannot tell which base model {name} came from")
        base = Path(json.loads(report.read_text(encoding="utf-8")).get("model") or "")
        if not base.is_dir():
            raise SystemExit(f"{name}: its recorded base {base} is not a directory")

        if args.check:
            files = complete_checkpoint(base, d, dry_run=True)
            tensors = missing_tensors(base, d)
        else:
            files = complete_checkpoint(base, d)
            tensors = carry_over_missing(base, d)
        verb = "missing" if args.check else "added"
        print(f"{name}  <-  {base}")
        print(f"    tensors {verb}: {len(tensors)}" + (f"  (e.g. {tensors[0]})" if tensors else ""))
        print(f"    files {verb}:   " + (", ".join(files) if files else "none"))
        incomplete += bool(tensors) and args.check
    if args.check and incomplete:
        print(f"\n{incomplete} checkpoint(s) lack base tensors — vLLM will refuse them. "
              f"Re-run without --check to carry them over.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
