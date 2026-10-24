#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.identity import generation_id_of as _gid  # noqa: E402,F401
from src.analysis.repetition import LOOP_AT, loop_score  # noqa: E402
from src.analysis.items import mc_item_index as _item_index  # noqa: E402  (the scorer's own answer-key index)
from src.analysis.join import LONG, joined_record  # noqa: E402
from src.analysis.judge_sample import is_incoherent, one_config  # noqa: E402
from src.analysis.outcomes import item_details, outcomes  # noqa: E402
from src.analysis.ratings import best_records  # noqa: E402
from src.io import read_dicts  # noqa: E402
from src.paths import (analysis_dir, items_path, judge_samples_dir, judgments_dir,  # noqa: E402
                       sample_row_files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help='Run.')
    ap.add_argument("--sample", required=True)
    ap.add_argument("--items", default=None, help='Items.')
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    sdir = judge_samples_dir(args.run)
    manifest = json.loads((sdir / f"{args.sample}.manifest.json").read_text(encoding="utf-8"))
    prompt_run = manifest.get("prompt_run") or args.run
    items_file = Path(args.items) if args.items else items_path(prompt_run)
    if not items_file.exists():
        raise SystemExit(f"no answer key at {items_file}: it is built on the server "
                         f"(data/processed/{prompt_run}/). Run this there, or pass --items <copy>.")
    sample = {_gid(r): r for path in sample_row_files(sdir, args.sample) for r in read_dicts(path)}
    files = sorted(judgments_dir(args.run).glob(f"{args.sample}*.jsonl"))
    if not files:
        raise SystemExit(f"no judgments for {args.sample} in {judgments_dir(args.run)}")
    best, _ = best_records(d for f in files for d in read_dicts(f))
    config = one_config(best.values())
    rows_meta = manifest["rows"]

    answers = outcomes(sample, rows_meta, _item_index(items_file), prompt_run)
    items = item_details(items_file)
    records = []
    for gid, row in sample.items():
        judged = {s: best.get((gid, s)) for s in ("trace", "explanation")}
        loops = is_incoherent(row, loop_score, LOOP_AT)
        records.append(joined_record(gid, row, rows_meta.get(gid, {}), items.get(row["item_id"]),
                                     judged, answers.get(gid), loops))
    records.sort(key=lambda r: (str(r["model"]), str(r["prompt_id"]), str(r["concept_id"]),
                                r.get("rank") if r.get("rank") is not None else 0,
                                str(r["item_id"])))

    out = Path(args.out_dir) if args.out_dir else analysis_dir(args.run) / "judge_sample" / args.sample
    out.mkdir(parents=True, exist_ok=True)
    with (out / "joined.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({**r, "judge_config": dict(zip(("concept", "framing", "mention"),
                                                              config))},
                               ensure_ascii=False) + "\n")
    cols = [k for k in records[0] if k not in LONG]
    with (out / "joined.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
                        for k, v in r.items() if k in cols})

    print(f"{len(records):,} generation(s) joined -> {out}/joined.jsonl (+ joined.csv, no trace)")
    print(f"judge configuration: concept {config[0]}, framing {config[1]}, mention {config[2]}")
    by_arm = Counter((r["arm"] or r.get("condition"), r["outcome"]) for r in records)
    for arm in sorted({a for a, _ in by_arm}, key=str):
        parts = ", ".join(f"{o}: {n}" for (a, o), n in sorted(by_arm.items(), key=str) if a == arm)
        print(f"  {arm:<11} {parts}")
    odd = Counter(r[f"{s}_stage1_0_but"] for r in records for s in ("trace", "explanation")
                  if r.get(f"{s}_stage1_0_but"))
    print(f"stage 1 = 0 but a later stage disagrees: {dict(odd) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
