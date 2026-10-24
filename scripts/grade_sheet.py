#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.identity import generation_id_of  # noqa: E402
from src.analysis.agreement import STAGES, compare, judge_labels, parse_grade_sheet  # noqa: E402
from src.analysis.judge_sample import one_config  # noqa: E402
from src.analysis.ratings import best_records  # noqa: E402
from src.analysis.spotcheck import has_surface  # noqa: E402
from src.analysis.surfaces import explanation_of, trace_of  # noqa: E402
from src.io import read_dicts, write_csv  # noqa: E402
from src.judge.prompts import (CONCEPT_SCORE, FRAMING_SCORE, MENTION_SCORE,  # noqa: E402
                               judge_concept_text)
from src.judge.run import verdict_text  # noqa: E402
from src.paths import analysis_dir, judge_samples_dir, judgments_dir, sample_row_files  # noqa: E402

_CONCEPT = "‹the Concept line of each entry›"
_NAMES = "‹the Names line of each entry›"
_PASSAGE = "‹the passage of each entry›"
_AXIS = {"present": "concept (present?)", "used": "framing (used?)", "named": "mention (named?)"}


def _passage(row: dict, surface: str) -> str:

    if surface == "explanation":
        return explanation_of(row.get("raw_output"))
    return trace_of(row.get("raw_output")) or row.get("raw_reasoning") or ""


def _sheet(run: str, sample: str, surface: str, seed: int, picked: list[str], rows: dict) -> str:
    q = [("1 · concept — is the concept present?  (every passage)",
          CONCEPT_SCORE.format(concept=_CONCEPT, sentence=_PASSAGE)),
         ("2 · framing — is its knowledge used, or not?  (every passage)",
          FRAMING_SCORE.format(concept=_CONCEPT, sentence=_PASSAGE)),
         ("3 · mention — is the concept named at all?  (every passage)",
          MENTION_SCORE.format(concept=_NAMES, sentence=_PASSAGE))]
    parts = [f"# Grade like the judge — {run} · {sample} · {len(picked)} {surface} passages "
             f"(random, seed {seed})\n\n",
             "Read each passage and answer the judge's three questions, exactly as the judge was "
             "asked them (below). Write your answers on the entry's GRADE line, on the judge's own "
             "scales, e.g. `GRADE concept=2 framing=0 mention=2`. Then run\n\n"
             f"    python scripts/grade_sheet.py --run {run} --sample {sample} --compare\n\n"
             "The judge's scores are not in this file. Ignore each question's last paragraph — "
             "the explanation and `Rating: [[score]]` format are for the judge; the GRADE line is "
             "yours.\n\n## The judge's instructions\n"]
    for title, text in q:
        parts.append(f"\n### {title}\n\n```text\n{text}\n```\n")
    for k, gid in enumerate(picked, 1):
        row = rows[gid]
        cid = row["concept_id"]
        parts.append(f"\n---\n\n## {k}\n\n<!-- row {gid} -->\n"
                     f"**Concept** (all three questions): {judge_concept_text(cid)}\n\n"
                     f"**Passage** — the {surface} the judge read:\n\n"
                     f"```text\n{_passage(row, surface).strip()}\n```\n\n"
                     f"GRADE concept=  framing=  mention=\n")
    return "".join(parts)


def make(args, d: Path, manifest: dict, sample: dict, seed: int) -> int:
    path = d / f"grade-{args.surface}-s{seed}-n{args.n}.md"
    if path.exists():
        print(f"{path} already exists — your grades live in it, so it is never rewritten.\n"
              f"Grade it, then: python scripts/grade_sheet.py --run {args.run} --sample "
              f"{args.sample} --compare")
        return 0
    population = sorted(g for g, m in manifest["rows"].items() if has_surface(m, args.surface))
    picked = random.Random(seed).sample(population, min(args.n, len(population)))
    d.mkdir(parents=True, exist_ok=True)
    path.write_text(_sheet(args.run, args.sample, args.surface, seed, picked, sample),
                    encoding="utf-8")
    print(f"{len(picked)} random {args.surface} passages -> {path}\n"
          f"⚠️  WMDP content — keep it out of git (results/ is ignored).")
    return 0


def _scores(rec: dict | None) -> tuple:
    if rec is None:
        return (None, None, None)
    return tuple(None if rec.get(f"{s}_score") in ("", None) else int(rec[f"{s}_score"])
                 for s in STAGES)


def _fmt(t: tuple) -> str:
    return "/".join("–" if v is None else str(v) for v in t)


def _drift(recs: list[dict]) -> list[str]:


    from src.judge.settings import judge_fingerprints

    judge = next((r.get("judge") for r in recs if r.get("judge")), None)
    if not judge:
        return []
    now = judge_fingerprints("rating", judge, quant="bitsandbytes")
    then = one_config(recs)
    if then[0] != now["concept"]:
        return []
    return [s for s, h in zip(("framing", "mention"), then[1:]) if h and h != now[s]]


def compare_all(args, d: Path, judged: dict) -> int:
    sheets = sorted(p for p in d.glob(f"grade-{args.surface}-*.md")
                    if not p.name.endswith(".compare.md"))
    if not sheets:
        raise SystemExit(f"no grading sheet in {d} — make one first (drop --compare)")
    by_rows = defaultdict(dict)
    for sheet in sheets:
        try:
            ids, human = parse_grade_sheet(sheet.read_text(encoding="utf-8"))
        except ValueError as e:
            raise SystemExit(f"{sheet.name}: {e}")
        by_rows[tuple(ids.values())][sheet.stem] = human
        recs = {k: judged.get((g, args.surface)) for k, g in ids.items()}
        judge = judge_labels([{"n": k, **{f"{s}_score": "" if r is None or r.get(f"{s}_score")
                                          is None else r[f"{s}_score"] for s in STAGES}}
                              for k, r in recs.items()])
        graded = sorted(set(human) & set(judge))
        cfg = one_config([recs[k] for k in graded]) if graded else (None, None, None)
        same = [k for k in graded if (human[k]["concept"], human[k]["framing"],
                                      human[k]["mention"]) == _scores(recs[k])]
        lines = [f"# {sheet.name} vs the judge\n\n",
                 f"judge configuration: concept `{cfg[0]}`, framing `{cfg[1]}`, mention "
                 f"`{cfg[2]}`\n\n",
                 f"**{len(graded)} entries graded by both · all three answers the same on "
                 f"{len(same)} ({100 * len(same) / max(len(graded), 1):.0f}%)**\n\n",
                 "| question | agree | κ | 2×2 (both yes / only you / only judge / both no) | "
                 "exact 0/1/2 |\n|---|---|---|---|---|\n"]
        print(f"\n{sheet.name} vs judge — {len(graded)} graded, all three the same on "
              f"{len(same)}")
        for r in compare(human, judge):
            k_ = "–" if r["kappa"] is None else f"{r['kappa']:.2f}"
            exact = "–" if r["exact_0_1_2"] is None else f"{100 * r['exact_0_1_2']:.0f}%"
            agree = f"{r['both'] + r['neither']}/{r['n']}"
            lines.append(f"| {_AXIS[r['axis']]} | {agree} | {k_} | {r['both']} / {r['a_only']} / "
                         f"{r['b_only']} / {r['neither']} | {exact} |\n")
            print(f"  {_AXIS[r['axis']]:<20} agree {agree:>7}   κ {k_:>5}   exact {exact:>5}")
        for stage in _drift([recs[k] for k in graded]):
            note = (f"⚠️  the judge's {stage} scores were made under an older {stage} rubric than "
                    f"the one in your sheet — re-run --compare after the re-judge")
            lines.append(f"\n{note}\n")
            print(f"  {note}")
        rows_csv = []
        lines.append("\n## Where you differ — yours vs the judge's (concept/framing/mention)\n")
        for k in graded:
            you = (human[k]["concept"], human[k]["framing"], human[k]["mention"])
            them, rec = _scores(recs[k]), recs[k]
            rows_csv.append({"entry": k, "generation_id": ids[k], "you": _fmt(you),
                             "judge": _fmt(them), "same": you == them})
            if you == them:
                continue
            lines.append(f"\n### entry {k}: you {_fmt(you)} · judge {_fmt(them)} · "
                         f"`{rec.get('prompt_id')}`\n")
            for s in STAGES:
                if rec.get(f"{s}_raw"):
                    v = verdict_text(rec[f"{s}_raw"], rec.get(f"{s}_finish_reason")).strip()
                    lines.append(f"\n**judge, {s}:** {v}\n")
        (d / f"{sheet.stem}.compare.md").write_text("".join(lines), encoding="utf-8")
        write_csv(d / f"{sheet.stem}.compare.csv", rows_csv,
                  ["entry", "generation_id", "you", "judge", "same"])
        print(f"  -> {d / (sheet.stem + '.compare.md')}  (each disagreement with the judge's "
              f"reasons)")
    for _, graders in by_rows.items():
        for a, b in combinations(sorted(graders), 2):
            print(f"\n{a} vs {b} — two humans, the same rows")
            for r in compare(graders[a], graders[b]):
                print(f"  {_AXIS[r['axis']]:<20} agree {r['both'] + r['neither']:>3}/{r['n']:<4}"
                      f" κ {'–' if r['kappa'] is None else format(r['kappa'], '.2f')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sample", required=True, help='Sample.')
    ap.add_argument("--n", type=int, default=30, help='N.')
    ap.add_argument("--seed", type=int, default=None, help='Seed.')
    ap.add_argument("--surface", choices=("trace", "explanation"), default="trace")
    ap.add_argument("--compare", action="store_true", help='Compare.')
    args = ap.parse_args(argv)

    sdir = judge_samples_dir(args.run)
    manifest = json.loads((sdir / f"{args.sample}.manifest.json").read_text(encoding="utf-8"))
    d = analysis_dir(args.run) / "judge_sample" / args.sample
    if args.compare:
        files = sorted(judgments_dir(args.run).glob(f"{args.sample}*.jsonl"))
        if not files:
            raise SystemExit(f"no judgments for {args.sample} under {judgments_dir(args.run)}")
        judged, _ = best_records(r for f in files for r in read_dicts(f))
        return compare_all(args, d, judged)
    seed = manifest["seed"] if args.seed is None else args.seed
    sample = {generation_id_of(r): r
              for path in sample_row_files(sdir, args.sample) for r in read_dicts(path)}
    return make(args, d, manifest, sample, seed)


if __name__ == "__main__":
    raise SystemExit(main())
