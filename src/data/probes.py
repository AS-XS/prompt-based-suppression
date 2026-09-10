from __future__ import annotations

from pathlib import Path

from src.data.mmlu import correct_text, open_item, subject_rows
from src.data.schema import Item, make_item_id
from src.data.terms import terms_for


DEPENDENT_SUBJECTS: dict[str, list[str]] = {
    "biology": ["college_medicine", "clinical_knowledge", "anatomy", "nutrition"],
    "physics": ["astronomy", "electrical_engineering", "college_chemistry"],
}


def _depends_on(concept_id: str, row: dict) -> bool:

    hay = (row["question"] + " " + correct_text(row)).lower()
    return any(t in hay for t in terms_for(concept_id))


def load_dependent_auto(concept_id: str, max_items: int = 40) -> list[Item]:

    items: list[Item] = []
    for subject in DEPENDENT_SUBJECTS.get(concept_id, []):
        for r in subject_rows(subject):
            if len(items) >= max_items:
                return items
            if _depends_on(concept_id, r):
                items.append(open_item(r, concept_id, subject, probe_type="dependent",
                                       extra_meta={"dependency": "term-overlap"}))
    return items


def load_dependent_anchors(concept_id: str,
                           anchors_dir: str | Path = "data/dependent") -> list[Item]:


    path = Path(anchors_dir) / f"{concept_id}.yaml"
    if not path.exists():
        return []
    import yaml

    rows = yaml.safe_load(path.read_text()) or []
    items: list[Item] = []
    for r in rows:
        q = r["question"]
        items.append(Item(
            item_id=make_item_id(concept_id, "dependent", q + "|anchor"),
            concept_id=concept_id,
            concept_class="general",
            concept_subtype="subject",
            probe_type="dependent",
            question=q,
            reference=r.get("reference"),
            source="hand-authored",
            meta={"format": "open_ended", "dependency": "hand-authored"},
        ))
    return items


def load_dependent(concept_id: str, max_auto: int = 40) -> list[Item]:
    return load_dependent_auto(concept_id, max_auto) + load_dependent_anchors(concept_id)
