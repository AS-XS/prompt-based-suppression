#!/usr/bin/env python


from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import read_dicts as _read_dicts, write_csv as _write_csv  # noqa: E402
from src.analysis.accuracy import accuracy_table, delta_table, score_generations  # noqa: E402
from src.analysis.grouping import eval_subject  # noqa: E402
from src.data.mmlu import SUBJECT_TO_CONCEPT  # noqa: E402
from src.data.split import SPLITS, read_val_ids, select, split_counts  # noqa: E402
from src.paths import analysis_dir, generations_glob, items_path, split_path, stale_analysis  # noqa: E402
from src.provenance import select_one_config  # noqa: E402


def _item_index(path: Path) -> dict[str, dict]:
    idx = {}
    for it in _read_dicts(path):
        meta = it.get("meta") or {}
        idx[it["item_id"]] = {"answer_key": it.get("answer_key"), "format": meta.get("format"),
                              "mmlu_subject": eval_subject(it),
                              "probe_type": it.get("probe_type"), "concept_id": it.get("concept_id")}
    return idx


def _delta_cols(base: list[str], supp_prompts: list[str]) -> list[str]:
    return base + ["baseline_acc", "n_baseline"] + [c for p in supp_prompts
                                                     for c in (f"{p}_acc", f"{p}_drop", f"{p}_n")]


def _one_row_per_cell(gens: list[dict]) -> list[dict]:


    seen: dict[tuple, dict] = {}
    for g in gens:
        seen.setdefault((g.get("model"), g.get("prompt_id"), g.get("think_state"),
                         g.get("condition"), g.get("item_id")), g)
    if len(seen) != len(gens):
        dropped = len(gens) - len(seen)
        by_model: dict[str, int] = {}
        for g in gens:
            by_model[g.get("model")] = by_model.get(g.get("model"), 0) + 1
        for m, n in sorted(by_model.items()):
            kept = sum(1 for k in seen if k[0] == m)
            if n != kept:
                print(f"  ⚠️ {m}: {n:,} rows -> {kept:,} ({n - kept:,} duplicate cells dropped, "
                      f"first kept). Concurrent re-submissions of the same sweep.", flush=True)
        print(f"  de-duplicated {dropped:,} of {len(gens):,} generation rows", flush=True)
    return list(seen.values())


def _stamp_reference(rows: list[dict], reference: str) -> list[dict]:


    return [{**r, "model": reference} for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help='Run.')
    ap.add_argument("--items", default=None, help='Items.')
    ap.add_argument("--generations", nargs="+", default=None, help='Generations.')
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--config-hash", default=None,
                    help='Config hash.')
    ap.add_argument("--split", choices=SPLITS, default=None,
                    help='Split.')
    ap.add_argument("--split-file", default=None, help='Split file.')
    ap.add_argument("--baseline-model", default=None,
                    help='Baseline model.')
    args = ap.parse_args()

    if not args.run and not (args.items and args.generations):
        raise SystemExit("pass --run exp1|round1 (or --items and --generations explicitly)")
    items_file = Path(args.items) if args.items else items_path(args.run)
    gen_globs = args.generations or [generations_glob(args.run)]


    out_dir = (Path(args.out_dir) if args.out_dir
               else analysis_dir(args.run) / (args.split or ""))

    idx = _item_index(items_file)
    paths = [f for pat in gen_globs for f in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no generation files matched: {gen_globs}")
    gens = [g for p in paths for g in _read_dicts(Path(p))]
    gens = select_one_config(gens, args.config_hash, what="generations")
    gens = _one_row_per_cell(gens)

    scored = score_generations(gens, idx)
    if not scored:
        raise SystemExit("no MC generations scored — are there MC items (meta.format=='mc') "
                         "with answer_key, and generations over them?")

    if args.split:
        split_file = Path(args.split_file) if args.split_file else split_path(args.run)
        scored = select(scored, read_val_ids(split_file), args.split)
        if not scored:
            raise SystemExit(f"--split {args.split} kept no rows — does {split_file} match "
                             "these items? Check with build_split.py --check")
        print(f"split={args.split} ({split_file}): " +
              ", ".join(f"{k} {v:,}" for k, v in sorted(split_counts(scored).items())) +
              "   [retain/neighbor/utility are NOT split — they pass through both halves]")

    out = out_dir
    acc = accuracy_table(scored)


    _write_csv(out / "accuracy.csv", acc,
               ["model", "think_state", "mmlu_subject", "prompt_id", "probe_type", "n", "n_parsed",
                "n_correct", "accuracy", "accuracy_strict", "parse_rate", "format_rate"])


    contrast = "model" if args.baseline_model else "prompt_id"
    base_level = args.baseline_model or "none-baseline"
    if args.baseline_model and not any(s["model"] == args.baseline_model for s in scored):
        raise SystemExit(
            f"--baseline-model {args.baseline_model!r} has no rows in this run. It is the reference "
            f"half of a paired comparison, so it must be swept into THIS namespace with the SAME "
            f"config — a row from another run was generated on other items and cannot be paired.\n"
            f"  present: {sorted({s['model'] for s in scored})}")
    if args.baseline_model:
        from src.analysis.checkpoints import refuse_cross_family
        refuse_cross_family((s["model"] for s in scored), args.baseline_model)

    def _delta(rows, keys, path):
        keys = [k for k in keys if k != contrast]
        d, levels = delta_table(rows, by=tuple(keys), baseline_prompt=base_level, contrast=contrast)
        if contrast == "model":
            d = _stamp_reference(d, base_level)
            keys = ["model"] + keys
        _write_csv(out / path, d, _delta_cols(keys, levels))
        return d, levels

    forget = [s for s in scored if s["probe_type"] == "forget"]
    d_on, sp_on = _delta(forget, ["model", "think_state", "mmlu_subject"], "accuracy_delta.csv")


    collateral = [s for s in scored if s["probe_type"] in ("retain", "neighbor", "utility")]
    d_se, sp_se = _delta(collateral, ["model", "think_state", "concept_id", "mmlu_subject"],
                         "side_effect.csv")


    mx = [{**s, "tested_concept": SUBJECT_TO_CONCEPT[s["mmlu_subject"]]}
          for s in scored
          if s["probe_type"] in ("forget", "retain") and s["mmlu_subject"] in SUBJECT_TO_CONCEPT]
    d_mx, sp_mx = _delta(mx, ["model", "think_state", "concept_id", "tested_concept"],
                         "subject_matrix.csv")

    n_parsed = sum(s["parsed"] for s in scored)
    print(f"scored {len(scored)} MC generations ({n_parsed} parsed, "
          f"{len(scored) - n_parsed} unparseable) over {len(paths)} file(s)")
    print(f"  accuracy       -> {out / 'accuracy.csv'}  ({len(acc)} cells)")
    print(f"  on-target drop -> {out / 'accuracy_delta.csv'}  ({len(d_on)} groups; prompts {sp_on})")
    print(f"  side effect    -> {out / 'side_effect.csv'}  ({len(d_se)} groups)")
    print(f"  subject matrix -> {out / 'subject_matrix.csv'}  ({len(d_mx)} cells; suppress×test)")


    if args.split and args.run:
        old = stale_analysis(args.run)
        if old:
            print(f"\n⚠️  {len(old)} file(s) in analysis/ are OLDER than your generations and were "
                  f"not refreshed by this --split run:")
            for f in old[:6]:
                print(f"      {f}")
            print("    report.py reads those. Refresh with:  "
                  f"python scripts/score_accuracy.py --run {args.run}")
    if not d_mx:


        print("     (empty: no MMLU suppress×test design in this run — see side_effect.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
