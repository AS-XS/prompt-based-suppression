from __future__ import annotations

from src.data.schema import ConceptClass, Item, make_item_id

_MMLU = "cais/mmlu"
CLASS: ConceptClass = "general"
_LETTERS = ["A", "B", "C", "D"]


CONCEPT_SUBJECTS: dict[str, list[str]] = {
    "biology": ["high_school_biology", "college_biology"],
    "chemistry": ["high_school_chemistry", "college_chemistry"],
    "physics": ["high_school_physics", "college_physics", "conceptual_physics"],
    "mathematics": ["high_school_mathematics", "college_mathematics"],
    "statistics": ["high_school_statistics"],
    "computer_science": ["high_school_computer_science", "college_computer_science"],
}


def subjects_for(concept_id: str) -> list[str]:
    return CONCEPT_SUBJECTS[concept_id]


SUBJECT_TO_CONCEPT: dict[str, str] = {
    subject: concept for concept, subjects in CONCEPT_SUBJECTS.items() for subject in subjects
}


def subject_rows(subject: str, split: str = "test"):
    from datasets import load_dataset

    return load_dataset(_MMLU, subject, split=split)


def load_mmlu_concept(concept_id: str, open_ended: bool = False,
                      split: str = "test", limit_per_subject: int | None = None) -> list[Item]:

    items: list[Item] = []
    for subject in subjects_for(concept_id):
        rows = subject_rows(subject, split)
        for i, r in enumerate(rows):
            if limit_per_subject is not None and i >= limit_per_subject:
                break
            items.append(open_item(r, concept_id, subject) if open_ended
                         else mc_item(r, concept_id, subject))
    return items


def correct_text(row: dict) -> str:
    return row["choices"][int(row["answer"])]


def _content_key(row: dict, fmt: str, subject: str) -> str:


    return "|".join([row["question"], fmt, subject, *[str(c) for c in row["choices"]],
                     str(row["answer"])])


def mc_item(row: dict, concept_id: str, subject: str, probe_type: str = "forget",
            extra_meta: dict | None = None) -> Item:

    q = row["question"]
    meta = {"format": "mc", "choices": row["choices"], "mmlu_subject": subject}
    meta.update(extra_meta or {})
    return Item(
        item_id=make_item_id(concept_id, probe_type, _content_key(row, "mc", subject)),
        concept_id=concept_id,
        concept_class=CLASS,
        concept_subtype="subject",
        probe_type=probe_type,
        question=q,
        reference=correct_text(row),
        answer_key=_LETTERS[int(row["answer"])],
        source=f"MMLU:{subject}",
        meta=meta,
    )


def open_item(row: dict, concept_id: str, subject: str, probe_type: str = "forget",
              extra_meta: dict | None = None) -> Item:

    q = row["question"]
    meta = {"format": "open_ended", "choices": row["choices"], "mmlu_subject": subject}
    meta.update(extra_meta or {})
    return Item(
        item_id=make_item_id(concept_id, probe_type, _content_key(row, "open", subject)),
        concept_id=concept_id,
        concept_class=CLASS,
        concept_subtype="subject",
        probe_type=probe_type,
        question=q,
        reference=correct_text(row),
        answer_key=None,
        source=f"MMLU:{subject}",
        meta=meta,
    )
