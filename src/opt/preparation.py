from __future__ import annotations

import sys
from collections import Counter
from typing import NamedTuple

from src.opt.dataset import build_examples, stratified_split
from src.opt.metrics import PROBE_TYPES
from src.paths import generations_dir, items_path, split_path


class PreparedExamples(NamedTuple):
    examples: list
    trainset: list
    valset: list
    probe_types: tuple
    mix: Counter
    missing: list
    dev_mix: Counter


def prepare_examples(args) -> PreparedExamples:
    items = items_path(args.run)
    baseline = generations_dir(args.run) / f"{args.model}__none-baseline.jsonl"
    for p in (items, baseline, split_path(args.run)):
        if not p.exists():
            raise SystemExit(f"missing {p} — the DSPy arm needs the item set, the frozen split, and "
                             f"this model's own baseline wave (it is what every metric pairs against)")

    probe_types = PROBE_TYPES[args.objective]
    examples = build_examples(items, baseline, split_path(args.run), think_state=args.think_state,
                              concepts=tuple(args.concept), probe_types=probe_types,
                              limit=args.limit)
    if not examples:
        raise SystemExit(
            "0 examples. Every one of these silently produces an empty set: the baseline wave is for "
            "a different think-state, the concept has no val items, or the model got nothing right "
            "(examples the baseline failed are dropped — there is nothing there to suppress).")

    mix = Counter(e.probe_type for e in examples)
    print(f"example mix: " + ", ".join(f"{k} {v}" for k, v in sorted(mix.items())))
    if args.objective == "A" and set(mix) == {"forget"}:
        raise SystemExit(
            "objective A has ONLY forget examples, so its `intact` term is unreachable and it is "
            "identical to objective B — the two would be the same experiment run twice (this "
            "happened). Check that the run's retain arms reached build_examples.")


    missing = [p for p in probe_types if p not in mix]
    if missing:
        print(f"WARNING: probe arm(s) {missing} were requested for objective {args.objective} but "
              f"contributed ZERO examples. Objective A's `intact` term is measured only over the "
              f"arms that are present, so its retain claim is narrower than it looks. Most likely "
              f"cause: those probes are open-ended, and build_examples keeps MC items only.",
              file=sys.stderr)


    trainset, valset = stratified_split(examples)
    dev_mix = Counter(e.probe_type for e in valset)
    print(f"dev mix: " + ", ".join(f"{k} {v}" for k, v in sorted(dev_mix.items())))
    if args.objective == "A" and "forget" not in dev_mix:
        raise SystemExit("objective A's dev set has no forget examples — its score would measure "
                         "only the retain arm. Check "
                         "stratified_split.")

    return PreparedExamples(examples, trainset, valset, probe_types, mix, missing, dev_mix)
