from __future__ import annotations

from src.data.wmdp import NEAR_RETAIN


NEAR_NEIGHBOUR: dict[str, tuple[str, ...]] = {
    "wmdp-bio": ("virology",),
    "wmdp-cyber": ("computer_security",),
}

TIERS: tuple[str, ...] = ("near", "general", "far")


def retain_tier(concept_id: str | None, subject: str | None) -> str:

    if subject in NEAR_NEIGHBOUR.get(concept_id, ()):
        return "near"
    if subject in NEAR_RETAIN.get(concept_id, ()):
        return "general"
    return "far"


def near_label(concept_id: str | None) -> str:

    subjects = NEAR_NEIGHBOUR.get(concept_id, ())
    return " + ".join(s.replace("_", " ").upper() for s in subjects) or "—"
