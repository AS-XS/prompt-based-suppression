from __future__ import annotations

from src.analysis.accuracy import extract_letter_tiered
from src.analysis.errors import ON_TARGET, classify
from src.analysis.surfaces import answer_of

OBJECTIVES = ("A", "B")


PROBE_TYPES: dict[str, tuple[str, ...]] = {
    "A": ("forget", "neighbor", "utility"),
    "B": ("forget",),
}


def _outcome(example, pred) -> tuple[str, bool]:

    text = getattr(pred, "response", None) if pred is not None else None
    letter, _ = extract_letter_tiered(answer_of(text))
    supp_correct = letter is not None and letter == example.answer_key
    outcome = classify(example.probe_type, example.baseline_correct, supp_correct,
                       supp_parsed=letter is not None)
    return outcome, supp_correct


def objective_A(example, pred, trace=None) -> float:


    outcome, _ = _outcome(example, pred)
    return 1.0 if outcome in {"held", "intact"} else 0.0


def objective_B(example, pred, trace=None) -> float:


    outcome, supp_correct = _outcome(example, pred)
    if outcome == "unparsed":
        return 0.0
    if example.probe_type == ON_TARGET:
        return 0.0 if supp_correct else 1.0
    return 1.0


def metric_for(objective: str):
    if objective not in OBJECTIVES:
        raise SystemExit(f"--objective must be one of {OBJECTIVES} (got {objective!r})")
    return {"A": objective_A, "B": objective_B}[objective]
