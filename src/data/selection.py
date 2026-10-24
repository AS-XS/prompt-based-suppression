from __future__ import annotations

from pathlib import Path

from src.data.schema import Item
from src.gen.runner import read_jsonl


def load_items(path: Path, concepts, probe_types, max_items, sample: int | None = None) -> list[Item]:
    items = list(read_jsonl(path, Item))
    if concepts:
        items = [it for it in items if it.concept_id in set(concepts)]
    if probe_types:
        items = [it for it in items if it.probe_type in set(probe_types)]
    if sample:


        import random

        return random.Random(0).sample(items, min(sample, len(items)))
    return items[:max_items] if max_items else items
