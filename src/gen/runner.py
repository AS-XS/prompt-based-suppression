from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Type, TypeVar

from pydantic import BaseModel

from src.io import read_dicts
from src.data.schema import GenerationRow, Item, Sampling
from src.provenance import config_hash, git_commit

T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: str | Path, rows: Iterable[BaseModel]) -> int:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, rows: Iterable[BaseModel]) -> int:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[dict]:


    yield from read_dicts(Path(path))


def read_jsonl(path: str | Path, model: Type[T]) -> Iterator[T]:


    prev: str | None = None
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if prev is not None:
                yield model.model_validate_json(prev)
            prev = line
    if prev is not None:
        try:
            yield model.model_validate_json(prev)
        except ValueError as e:
            import warnings

            warnings.warn(f"read_jsonl: dropped a truncated final line in {path} "
                          f"(killed mid-write?) — its cell will regenerate. {str(e)[:80]}")


def stub_generate(item: Item, cell: dict, sampling: Sampling) -> GenerationRow:

    raw = f"[stub answer for {item.concept_id}::{item.probe_type}] {item.question}"
    return GenerationRow(
        git_commit=git_commit(),
        config_hash=config_hash(cell),
        config_id=cell.get("config_id", "stub"),
        model=cell.get("model", "stub/echo-tiny"),
        prompt_id=cell.get("prompt_id", "world-frame"),
        condition=cell.get("condition", "suppressed"),
        think_state=cell.get("think_state", "na"),
        sampling=sampling,
        item_id=item.item_id,
        concept_id=item.concept_id,
        concept_class=item.concept_class,
        concept_subtype=item.concept_subtype,
        probe_type=item.probe_type,
        raw_output=raw,
        raw_reasoning=None,
        finish_reason="stop",
    )


