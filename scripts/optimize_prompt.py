#!/usr/bin/env python


from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.opt.metrics import OBJECTIVES  # noqa: E402
from src.opt.search import default_trials as _default_trials  # noqa: E402,F401


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="wmdp1",
                    help='Run.')
    ap.add_argument("--round", type=int, default=1, metavar="N",
                    help='Round.')
    ap.add_argument("--out-run", default=None,
                    help='Out run.')
    ap.add_argument("--force", action="store_true",
                    help='Force.')
    ap.add_argument("--model", required=True, help='Model.')
    ap.add_argument("--objective", default="A", choices=OBJECTIVES)
    ap.add_argument("--seed-prompt", default="scoped-world@generic",
                    help='Seed prompt.')
    ap.add_argument("--concept", nargs="+", default=["wmdp-bio"],
                    help='Concept.')
    ap.add_argument("--think-state", default="off")
    ap.add_argument("--limit", type=int, default=None, help='Limit.')
    ap.add_argument("--candidates", type=int, default=12,
                    help='Candidates.')
    ap.add_argument("--trials", type=int, default=None,
                    help='Trials.')
    ap.add_argument("--minibatch", action="store_true",
                    help='Minibatch.')
    ap.add_argument("--demos", type=int, default=0, metavar="N",
                    help='Demos.')
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--prompt-model", default=None,
                    help='Prompt model.')
    ap.add_argument("--prompt-model-api-base", default="http://localhost:8001/v1")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help='Max tokens.')
    ap.add_argument("--dry-run", action="store_true",
                    help='Dry run.')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from src.opt.runner import run
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
