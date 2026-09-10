from __future__ import annotations

from src.data.mmlu import mc_item, subject_rows
from src.data.schema import Item


STEM_SUBJECT: dict[str, str] = {
    "biology": "high_school_biology",
    "chemistry": "high_school_chemistry",
    "physics": "high_school_physics",
    "mathematics": "high_school_mathematics",
    "statistics": "high_school_statistics",
    "computer_science": "high_school_computer_science",
}


FAR_RETAIN: dict[str, str] = {
    "biology": "high_school_us_history",
    "chemistry": "professional_law",
    "physics": "high_school_us_history",
    "mathematics": "professional_law",
    "statistics": "high_school_us_history",
    "computer_science": "professional_law",
}


UTILITY_SUBJECTS: list[str] = [

    "world_religions", "marketing", "moral_scenarios", "philosophy", "management", "prehistory",


    "astronomy", "abstract_algebra", "college_mathematics", "conceptual_physics",
    "electrical_engineering", "high_school_physics",


    "formal_logic", "high_school_european_history", "international_law", "jurisprudence",

    "econometrics", "high_school_geography", "high_school_psychology", "sociology",

    "business_ethics", "clinical_knowledge", "nutrition", "professional_accounting",
]


def load_retain(concept_id: str, limit_per_subject: int | None = 50) -> list[Item]:


    items: list[Item] = []
    for other, subject in STEM_SUBJECT.items():
        if other == concept_id:
            continue
        for i, r in enumerate(subject_rows(subject)):
            if limit_per_subject is not None and i >= limit_per_subject:
                break
            items.append(mc_item(r, concept_id, subject, probe_type="retain",
                                 extra_meta={"distance": "cross"}))
    far = FAR_RETAIN[concept_id]
    for i, r in enumerate(subject_rows(far)):
        if limit_per_subject is not None and i >= limit_per_subject:
            break
        items.append(mc_item(r, concept_id, far, probe_type="retain",
                             extra_meta={"distance": "far"}))
    return items


def load_utility(concept_id: str, per_subject: int = 15) -> list[Item]:

    items: list[Item] = []
    for subject in UTILITY_SUBJECTS:
        for i, r in enumerate(subject_rows(subject)):
            if i >= per_subject:
                break
            items.append(mc_item(r, concept_id, subject, probe_type="utility"))
    return items
