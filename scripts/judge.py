#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import read_dicts  # noqa: E402
from src.analysis.ratings import best_records  # noqa: E402
from src.analysis.surfaces import answer_of, explanation_of, trace_of  # noqa: E402
from src.data.schema import GenerationRow  # noqa: E402
from src.gen.env import make_llm, setup_vllm_env  # noqa: E402
from src.gen.runner import append_jsonl, read_jsonl  # noqa: E402
from src.judge.prompts import (concept_messages, framing_messages,  # noqa: E402
                               judge_concept_text, judge_mention_text, mention_messages)
from src.judge.run import concept_scores, framing_scores, mention_scores  # noqa: E402
from src.judge.settings import (JUDGE_CHAT_KWARGS, default_budget,  # noqa: E402
                                judge_fingerprints, judge_sampling, refuse_mixed_config)
from src.paths import generations_glob, judgments_dir  # noqa: E402
from src.provenance import git_commit  # noqa: E402


DEFAULT_CHUNK = 256


def _surface_text(row, surface: str):


    if surface == "answer":
        return answer_of(row.raw_output, expect_trace=row.think_state == "on")
    if surface == "explanation":
        return explanation_of(row.raw_output)
    return trace_of(row.raw_output) or row.raw_reasoning


_COLS = ["generation_id", "model", "concept_id", "concept_class", "prompt_id", "condition",
         "think_state", "surface", "concept_score", "framing_score", "mention_score"]

class Todo(NamedTuple):


    row: GenerationRow
    surface: str
    text: str
    done: dict | None
    concept_text: str
    mention_text: str


def _read_dicts(path: Path):

    if not path.exists():
        return
    yield from read_dicts(path)


def plan_file(gens_path, out_dir, suppressed_only, limit, surfaces=("explanation",),
              shard=None, stage="both", probe_types=None, fingerprints=None) -> dict:


    rows = list(read_jsonl(Path(gens_path), GenerationRow))
    if suppressed_only:
        rows = [r for r in rows if r.condition == "suppressed"]
    if probe_types:
        rows = [r for r in rows if r.probe_type in set(probe_types)]
    if limit:
        rows = rows[:limit]

    stem = Path(gens_path).stem
    if shard:
        i, n = shard
        rows = rows[i - 1::n]
        stem = f"{stem}.shard{i}of{n}"
    out_jsonl = Path(out_dir) / f"{stem}.jsonl"
    out_csv = Path(out_dir) / f"{stem}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)


    prior = list(_read_dicts(out_jsonl))
    best = best_records(prior)[0]
    table = [{k: d.get(k) for k in _COLS} for d in best.values()]

    def scored_here(done: dict, which: str) -> bool:
        return done.get(f"{which}_score") is not None and (
            fingerprints is None or done.get(f"{which}_config_hash") == fingerprints[which])

    todo = []
    for r in rows:
        for surface in surfaces:
            key = (r.generation_id, surface)
            done = best.get(key)
            if stage in ("concept", "both") and done is not None:
                continue
            if stage in ("framing", "mention") and done is None:
                continue
            if stage == "framing":
                c = done.get("concept_score")
                if not c or c <= 0:
                    continue
                if scored_here(done, "framing"):
                    continue
            if stage == "mention" and scored_here(done, "mention"):
                continue
            text = _surface_text(r, surface)
            if not (text or "").strip():
                continue
            todo.append(Todo(r, surface, text, done, judge_concept_text(r.concept_id),
                             judge_mention_text(r.concept_id)))
    if fingerprints and todo:
        refuse_mixed_config(prior, fingerprints["concept"], out_jsonl)
    return {"stem": stem, "out_jsonl": out_jsonl, "out_csv": out_csv, "best": best,
            "table": table, "todo": todo}


def write_csv(plan: dict) -> None:

    with plan["out_csv"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(plan["table"])


def _record(r, surface: str, concept_text: str, judge_name: str, meta: dict, scores: dict) -> dict:


    return {"generation_id": r.generation_id, "model": r.model, "item_id": r.item_id,
            "concept_id": r.concept_id, "concept_class": r.concept_class,
            "probe_type": r.probe_type, "prompt_id": r.prompt_id, "condition": r.condition,
            "think_state": r.think_state, "surface": surface, "judge": judge_name,
            "judge_concept": concept_text, **meta, **scores}


def run_plan(llm, sp, plan: dict, judge_name, stage="both", chat_kw=None,
             chunk: int = DEFAULT_CHUNK, meta: dict | None = None,
             framing_meta: dict | None = None, mention_meta: dict | None = None
             ) -> tuple[int, Path]:


    meta, framing_meta, mention_meta = meta or {}, framing_meta or {}, mention_meta or {}
    best, table, out_jsonl, todo = plan["best"], plan["table"], plan["out_jsonl"], plan["todo"]
    written = 0
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        if stage == "framing":


            pairs = [(t.concept_text, t.text) for t in part]
            recs = [{**t.done, **f, **framing_meta, "framing_git_commit": meta.get("git_commit")}
                    for t, f in zip(part, framing_scores(llm, sp, pairs, chat_kw))]
        elif stage == "mention":
            pairs = [(t.mention_text, t.text) for t in part]
            recs = [{**t.done, **m, **mention_meta, "judge_mention": t.mention_text,
                     "mention_git_commit": meta.get("git_commit")}
                    for t, m in zip(part, mention_scores(llm, sp, pairs, chat_kw))]
        else:
            pairs = [(t.concept_text, t.text) for t in part]
            scores = concept_scores(llm, sp, pairs, chat_kw)
            if stage == "both":
                leaked = [i for i, s in enumerate(scores) if (s["concept_score"] or 0) > 0]
                for i, f in zip(leaked, framing_scores(llm, sp, [pairs[i] for i in leaked],
                                                       chat_kw)):
                    scores[i].update({**f, **framing_meta})
            recs = [_record(t.row, t.surface, t.concept_text, judge_name, meta, s)
                    for t, s in zip(part, scores)]
        append_jsonl(out_jsonl, [_Row(rec) for rec in recs])
        for rec in recs:
            key = (rec["generation_id"], rec["surface"])
            best[key] = rec
            table[:] = [t for t in table if (t.get("generation_id"), t.get("surface")) != key]
            table.append({k: rec.get(k) for k in _COLS})
        written += len(recs)
        print(f"  {plan['stem']}: {written:,}/{len(todo):,} judged", flush=True)

    write_csv(plan)
    return written, out_jsonl


def _conversations(plans: list[dict], stage: str) -> tuple[list, list]:


    builders = {"concept": ((concept_messages, "concept_text"),),
                "framing": ((framing_messages, "concept_text"),),
                "mention": ((mention_messages, "mention_text"),),
                "both": ((concept_messages, "concept_text"),
                         (framing_messages, "concept_text"))}[stage]
    convs, labels = [], []
    for pl in plans:
        for t in pl["todo"]:
            for build, field in builders:
                convs.append(build(getattr(t, field), t.text))
                labels.append(f"{t.row.generation_id} [{t.surface}]")
    return convs, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help='Run.')
    ap.add_argument("--generations", nargs="+", default=None,
                    help='Generations.')
    ap.add_argument("--judge", required=True, help='Judge.')
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "./models"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quant", default=None, help='Quant.')
    ap.add_argument("--tp", type=int, default=None, metavar="N",
                    help='Tp.')
    ap.add_argument("--dtype", default=None, help='Dtype.')
    ap.add_argument("--suppressed-only", action="store_true")
    ap.add_argument("--probe-types", nargs="+", default=None,
                    choices=["forget", "neighbor", "retain", "utility", "dependent"],
                    help='Probe types.')
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help='Max tokens.')
    ap.add_argument("--max-model-len", type=int, default=None,
                    help='Max model len.')
    ap.add_argument("--gpu-util", type=float, default=None,
                    help='Gpu util.')
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                    help='Chunk.')
    ap.add_argument("--surfaces", nargs="+", default=["explanation"],
                    choices=["answer", "explanation", "trace"],
                    help='Surfaces.')
    ap.add_argument("--stage", default="both", choices=["concept", "framing", "mention", "both"],
                    help='Stage.')
    ap.add_argument("--shard", default=None, metavar="i/N",
                    help='Shard.')
    args = ap.parse_args()

    shard = None
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard must look like i/N (got {args.shard!r})")
        if not (1 <= i <= n):
            raise SystemExit(f"--shard i must be within 1..N (got {args.shard!r})")
        shard = (i, n)

    if not args.run and not args.generations:
        raise SystemExit("pass --run exp1|round1 (or --generations explicitly)")
    gen_globs = args.generations or [generations_glob(args.run)]
    out_dir = args.out_dir or judgments_dir(args.run)
    paths = [f for pat in gen_globs for f in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no files matched: {gen_globs}")


    fingerprints = judge_fingerprints("rating", args.judge, quant=args.quant, tp=args.tp,
                                      dtype=args.dtype, max_tokens=args.max_tokens,
                                      max_model_len=args.max_model_len)


    plans = [plan_file(p, out_dir, args.suppressed_only, args.limit, args.surfaces, shard,
                       args.stage, args.probe_types, fingerprints) for p in paths]
    outstanding = sum(len(pl["todo"]) for pl in plans)
    for pl in plans:
        write_csv(pl)
    if not outstanding:
        done = sum(len(pl["table"]) for pl in plans)
        print(f"nothing outstanding for stage={args.stage} across {len(paths)} file(s) "
              f"({done} judgments already on disk) — NOT loading the judge, no GPU used.")
        return 0
    print(f"{outstanding} judgment(s) outstanding across {len(paths)} file(s)")


    from src.gen.sweep import model_context_limit
    from src.gen.thinkstate import needs_raw_special
    from src.judge.preflight import load_tokenizer, run_preflight

    needs_raw_special(args.judge)
    model_dir = Path(args.models_dir) / args.judge
    convs, labels = _conversations(plans, args.stage)
    think = run_preflight(load_tokenizer(model_dir), convs,
                          args.max_model_len or model_context_limit(model_dir), labels)
    budget = args.max_tokens if args.max_tokens is not None else default_budget(model_dir)

    setup_vllm_env()
    kwargs = {k: v for k, v in (("quantization", args.quant), ("tensor_parallel_size", args.tp),
                                ("dtype", args.dtype), ("max_model_len", args.max_model_len),
                                ("gpu_memory_utilization", args.gpu_util)) if v}
    print(f"loading judge {args.judge}{' (' + args.quant + ')' if args.quant else ''}"
          f"{f' tp={args.tp}' if args.tp else ''} "
          f"for {len(paths)} file(s), surfaces={args.surfaces}, stage={args.stage}, "
          f"config concept {fingerprints['concept']} / framing {fingerprints['framing']} / "
          f"mention {fingerprints['mention']}")
    llm = make_llm(str(model_dir), **kwargs)
    sp, resolved = judge_sampling(llm, args.judge, budget)
    print("  judge decode (the model's own): " + ", ".join(f"{k}={v}" for k, v in resolved.items())
          + ("   [max_tokens=None = the rest of the window]" if budget is None else ""), flush=True)
    meta = {"judge_config_hash": fingerprints["concept"], "judge_sampling": resolved,
            "judge_think_default": think, "git_commit": git_commit()}
    framing_meta = {"framing_config_hash": fingerprints["framing"]}
    mention_meta = {"mention_config_hash": fingerprints["mention"]}

    total = 0
    for pl in plans:
        if not pl["todo"]:
            continue
        n, out = run_plan(llm, sp, pl, args.judge, args.stage, JUDGE_CHAT_KWARGS, args.chunk,
                          meta, framing_meta, mention_meta)
        total += n
        print(f"  {pl['stem']}: {n} judgments -> {out}")
    print(f"done — {total} judgments across {len(paths)} file(s)")
    return 0


class _Row:


    def __init__(self, d: dict):
        self._d = d

    def model_dump_json(self) -> str:
        return json.dumps(self._d, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
