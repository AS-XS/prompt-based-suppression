#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unlearn.r2mu import run
from src.unlearn.r2mu_data import R2MU_MODELS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, choices=R2MU_MODELS)
    ap.add_argument('--models-dir', default='models')
    ap.add_argument('--name', required=True)
    ap.add_argument('--concept', choices=['wmdp-bio','wmdp-cyber'], default='wmdp-bio')
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--limo-file', type=Path, default=Path('data/raw/r2mu-limo.jsonl'))
    ap.add_argument('--layer', required=True)
    ap.add_argument('--coeff', type=float, default=81)
    ap.add_argument('--steering-ratio', type=float)
    ap.add_argument('--gamma', type=float, default=100)
    ap.add_argument('--trace-weight', type=float, default=1)
    ap.add_argument('--cot-weight', type=float, default=1)
    ap.add_argument('--lr', type=float, default=0.00025)
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--frozen-dtype', choices=['float32','bfloat16'], default='bfloat16')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--frozen-device', default='cuda:0')
    ap.add_argument('--max-seconds', type=int, default=37800)
    ap.add_argument('--mixed-precision', action='store_true',
                    help='Mixed precision.')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if Path(args.name).name != args.name or args.name in ('.','..'):
        ap.error('name must be a plain directory name')
    if min(args.steps, args.batch_size, args.lr, args.coeff) <= 0 or min(
            args.trace_weight, args.cot_weight, args.gamma, args.max_seconds) < 0:
        ap.error('invalid training bounds')
    if args.steering_ratio is not None and args.steering_ratio <= 0:
        ap.error('steering-ratio must be positive')
    run(args)


if __name__ == '__main__':
    main()
