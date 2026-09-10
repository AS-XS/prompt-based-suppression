from __future__ import annotations

from src.data.schema import ConceptClass, Item, make_item_id

_RWKU = "jinzhuoran/RWKU"


_FORGET = {"forget_level1": "cloze", "forget_level2": "qa"}
_FORGET_EXTRA = {"forget_level3": "adversarial"}
_NEIGHBOR = {"neighbor_level1": "l1"}
_NEIGHBOR_EXTRA = {"neighbor_level2": "l2"}
CLASS: ConceptClass = "isolated"


def default_entities(n: int = 20) -> list[str]:

    from datasets import load_dataset

    ds = load_dataset(_RWKU, "forget_level2", split="test")
    return sorted(set(ds["subject"]))[:n]


def _rows(config: str):
    from datasets import load_dataset

    return load_dataset(_RWKU, config, split="test")


def load_rwku(entities: list[str] | None = None, n: int = 20, neighbor: bool = True,
              extra_forget: bool = False, extra_neighbor: bool = False) -> list[Item]:


    keep = set(entities) if entities is not None else set(default_entities(n))
    items: list[Item] = []

    forget_cfgs = {**_FORGET, **(_FORGET_EXTRA if extra_forget else {})}
    for cfg, tag in forget_cfgs.items():
        items += _from_config(cfg, keep, "forget",
                              {"level": tag, "adversarial": cfg == "forget_level3"})

    if neighbor:
        nbr_cfgs = {**_NEIGHBOR, **(_NEIGHBOR_EXTRA if extra_neighbor else {})}
        for cfg, tag in nbr_cfgs.items():
            items += _from_config(cfg, keep, "neighbor", {"level": tag})
    return items


def _from_config(cfg: str, keep: set[str], probe_type: str, meta: dict) -> list[Item]:
    return [_item(r, probe_type=probe_type, source=f"RWKU:{cfg}", meta=meta)
            for r in _rows(cfg) if r["subject"] in keep]


def _item(row: dict, probe_type: str, source: str, meta: dict) -> Item:
    entity = row["subject"]
    question = row["query"]
    return Item(

        item_id=make_item_id(entity, probe_type, f"{question}|{row.get('answer')}"),
        concept_id=entity,
        concept_class=CLASS,
        concept_subtype="entity",
        probe_type=probe_type,
        question=question,
        reference=row.get("answer"),
        source=source,
        meta=meta,
    )
