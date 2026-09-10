#!/usr/bin/env python


from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.analysis.modes import derive_mode  # noqa: E402
from src.data.demo import demo_item  # noqa: E402
from src.data.schema import GenerationRow, Sampling  # noqa: E402
from src.gen.runner import read_jsonl, stub_generate, write_jsonl  # noqa: E402
from src.judge.run import stub_judge  # noqa: E402
from src.judge.schema import JudgmentRow, judge_json_schema  # noqa: E402
from src.provenance import config_hash, git_commit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "smoke"


def main() -> int:
    cell = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    sampling = Sampling(**cell["sampling"])


    item = demo_item()
    gen = stub_generate(item, cell, sampling)
    n_gen = write_jsonl(OUT / "generations.jsonl", [gen])


    gen_back = next(read_jsonl(OUT / "generations.jsonl", GenerationRow))
    assert gen_back.generation_id == gen.generation_id


    verdict = stub_judge(gen_back.raw_output, concept_label=item.concept_id)
    judgment = JudgmentRow(
        judge_model="stub/rule-judge",
        judge_config_hash=config_hash({"judge": "stub"}),
        git_commit=git_commit(),
        generation_id=gen_back.generation_id,
        surface="answer",
        raw_judge_json=verdict.model_dump_json(),
        parsed=verdict,
    )
    n_judge = write_jsonl(OUT / "judgments.jsonl", [judgment])
    judgment_back = next(read_jsonl(OUT / "judgments.jsonl", JudgmentRow))


    mode = derive_mode(judgment_back.parsed)

    print("smoke OK")
    print(f"  git_commit      {gen.git_commit}")
    print(f"  config_hash     {gen.config_hash}")
    print(f"  generation_id   {gen.generation_id}")
    print(f"  wrote           {n_gen} generation, {n_judge} judgment  -> {OUT}")
    print(f"  derived mode    {mode}")
    print(f"  judge schema    {len(judge_json_schema()['properties'])} primitives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
