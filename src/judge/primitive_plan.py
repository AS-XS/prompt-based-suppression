from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.analysis.surfaces import answer_of, explanation_of, trace_of
from src.data.schema import GenerationRow
from src.gen.runner import read_jsonl
from src.io import read_dicts
from src.provenance import select_one_config


@dataclass
class PrimitivePlan:
    out_jsonl: Path
    out_csv: Path
    prior: list[dict]
    todo: list[tuple[GenerationRow, str, str]]


def plan_file(gens_path, out_dir, suppressed_only, limit, surfaces, config_hash=None,
              shard=None, probe_types=None) -> PrimitivePlan:

    rows = list(read_jsonl(Path(gens_path), GenerationRow))
    rows = select_one_config([r.model_dump() for r in rows], config_hash,
                             what=f"generations in {Path(gens_path).name}")
    rows = [GenerationRow.model_validate(r) for r in rows]
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
    prior = list(read_dicts(out_jsonl)) if out_jsonl.exists() else []
    done = {(d.get("generation_id"), d.get("surface")) for d in prior}
    todo = []
    for r in rows:
        for surface in surfaces:
            if (r.generation_id, surface) in done:
                continue
            if surface == "answer":
                text = answer_of(r.raw_output)
            elif surface == "explanation":
                text = explanation_of(r.raw_output)
            else:
                text = trace_of(r.raw_output) or r.raw_reasoning
            if (text or "").strip():
                todo.append((r, surface, text))
    return PrimitivePlan(out_jsonl, out_jsonl.with_suffix(".csv"), prior, todo)
