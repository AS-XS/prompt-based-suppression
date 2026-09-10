from __future__ import annotations

from src.data.schema import ConceptClass, Item, make_item_id

_WMDP = "cais/wmdp"
CLASS: ConceptClass = "general"
_LETTERS = ["A", "B", "C", "D"]


SUBSETS: dict[str, str] = {
    "wmdp-bio": "wmdp-bio",
    "wmdp-cyber": "wmdp-cyber",
    "wmdp-chem": "wmdp-chem",
}


EXPECTED_N: dict[str, int] = {"wmdp-bio": 1273, "wmdp-cyber": 1987, "wmdp-chem": 408}


def subset_rows(concept_id: str, split: str = "test"):


    from datasets import load_dataset

    return load_dataset(_WMDP, SUBSETS[concept_id], split=split)


def _content_key(row: dict, concept_id: str) -> str:


    return "|".join([row["question"], "mc", concept_id,
                     *[str(c) for c in row["choices"]], str(row["answer"])])


def wmdp_item(row: dict, concept_id: str) -> Item:

    return Item(
        item_id=make_item_id(concept_id, "forget", _content_key(row, concept_id)),
        concept_id=concept_id,
        concept_class=CLASS,
        concept_subtype="hazard",       
        probe_type="forget",
        question=row["question"],
        reference=row["choices"][int(row["answer"])],
        answer_key=_LETTERS[int(row["answer"])],
        source=f"WMDP:{SUBSETS[concept_id]}",
        meta={"format": "mc", "choices": list(row["choices"]), "wmdp_subset": SUBSETS[concept_id]},
    )


def load_wmdp(concept_id: str, limit: int | None = None, split: str = "test") -> list[Item]:

    if concept_id not in SUBSETS:
        raise KeyError(f"unknown WMDP subset {concept_id!r}; expected one of {sorted(SUBSETS)}")
    items = [wmdp_item(r, concept_id) for r in subset_rows(concept_id, split)]
    if limit is not None:
        items = items[:limit]
    return items


NEAR_RETAIN: dict[str, list[str]] = {
    "wmdp-bio": ["college_biology", "virology", "high_school_biology"],
    "wmdp-cyber": ["college_computer_science", "computer_security", "high_school_computer_science"],
    "wmdp-chem": ["college_chemistry", "high_school_chemistry"],
}


def load_near_retain(concept_id: str, limit_per_subject: int | None = 100) -> list[Item]:


    from src.data.mmlu import mc_item, subject_rows

    items: list[Item] = []
    for subject in NEAR_RETAIN[concept_id]:
        for i, row in enumerate(subject_rows(subject)):
            if limit_per_subject is not None and i >= limit_per_subject:
                break
            items.append(mc_item(row, concept_id, subject, probe_type="neighbor",
                                 extra_meta={"distance": "near"}))
    return items
