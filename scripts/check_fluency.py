#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import UNLEARNED_ROOT, unlearn_report_path  # noqa: E402
from src.unlearn.corpora import load_retain  # noqa: E402
from src.unlearn.rmu import RMUReport, _fluency_check  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="+", help='Name.')
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n", type=int, default=3, help='N.')
    ap.add_argument("--write", action="store_true",
                    help='Write.')
    args = ap.parse_args()

    dirs = [Path(n) if "/" in n else UNLEARNED_ROOT / n for n in args.name]
    for d in dirs:
        if not d.is_dir():
            raise SystemExit(f"no such checkpoint: {d}")

    import torch


    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "no GPU on this node — this is a forward pass over an 8B model, so it needs one.\n"
            "Run this command on an allocated GPU, or explicitly select --device cpu.")

    worst = 0
    for d in dirs:
        print(f"\n=== {d.name} ===")
        worst = max(worst, _one(d, args))
    return worst


def _one(d: Path, args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(d)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(d, dtype=getattr(torch, args.dtype)).to(args.device)
    report = RMUReport(config={}, model=str(d), concept="", n_forget=0, n_retain=0)
    _fluency_check(model, tok, load_retain(limit=args.n)[:args.n], args.device, report)
    del model
    torch.cuda.empty_cache()

    if args.write:
        p = unlearn_report_path(d)
        if not p.is_file():
            raise SystemExit(f"no {p} to write into")
        meta = json.loads(p.read_text(encoding="utf-8"))
        meta["fluency_unique_ratio"] = report.fluency_unique_ratio
        meta["fluency_samples"] = report.fluency_samples
        p.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(f"\nrecorded in {p}")
    r = report.fluency_unique_ratio
    return 1 if r is not None and r < 0.35 else 0


if __name__ == "__main__":
    raise SystemExit(main())
