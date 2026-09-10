from __future__ import annotations

from typing import Mapping


def eval_subject(item: Mapping) -> str | None:


    meta = item.get("meta") or {}
    subject = meta.get("mmlu_subject")
    if subject:
        return subject
    if item.get("probe_type") == "forget":
        return item.get("concept_id")
    return None


def tested_concept(scored: Mapping, subject_to_concept: Mapping[str, str]) -> str | None:


    subject = scored.get("mmlu_subject")
    concept_id = scored.get("concept_id")


    if concept_id:
        from src.data.wmdp import NEAR_RETAIN
        if subject in NEAR_RETAIN.get(concept_id, ()):
            return concept_id

    folded = subject_to_concept.get(subject)
    if folded:
        return folded
    if subject and subject == concept_id:
        return subject
    return None
