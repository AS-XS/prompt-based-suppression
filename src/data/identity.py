from __future__ import annotations

from collections.abc import Mapping


def generation_id(config_hash, item_id, condition, think_state) -> str:
    return f"{config_hash}:{item_id}:{condition}:{think_state}"


def generation_id_of(row: Mapping, *, strict: bool = True) -> str:

    get = row.__getitem__ if strict else row.get
    return generation_id(*(get(k) for k in ("config_hash", "item_id", "condition", "think_state")))
