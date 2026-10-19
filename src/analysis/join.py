from __future__ import annotations

from typing import Mapping

from src.analysis.ratings import cell_of, mentioned, used
from src.analysis.surfaces import answer_of, explanation_of, trace_of
from src.judge.run import verdict_text

SURFACES = ("trace", "explanation")
STAGES = ("concept", "framing", "mention")
IDENTITY = ("generation_id", "run", "model", "base_model", "method", "arm", "condition",
            "prompt_id", "concept_id", "item_id", "think_state", "rank")


LONG = ("trace",)


def _judged(rec: Mapping | None, surface: str, incoherent: bool) -> dict:
    out = {}
    for s in STAGES:
        score = None if rec is None or rec.get(f"{s}_score") in ("", None) else int(rec[f"{s}_score"])
        raw = rec.get(f"{s}_raw") if rec else None
        out[f"{surface}_{s}"] = score
        out[f"{surface}_{s}_why"] = (verdict_text(raw, rec.get(f"{s}_finish_reason")).strip()
                                     if raw else None)
    if rec is None or rec.get("concept_score") in ("", None):
        out[f"{surface}_cell"] = out[f"{surface}_stage1_0_but"] = None
        return out
    r = {**rec, "incoherent": incoherent}
    out[f"{surface}_cell"] = cell_of(r)
    zero = int(rec["concept_score"]) == 0
    flags = [name for name, hit in (("used", rec.get("framing_score") not in ("", None) and used(r)),
                                    ("named", rec.get("mention_score") not in ("", None)
                                     and mentioned(r)))
             if zero and hit]
    out[f"{surface}_stage1_0_but"] = ",".join(flags) or None
    return out


def joined_record(gid: str, row: Mapping, meta: Mapping, item: Mapping | None,
                  judged: Mapping[str, Mapping], answer: Mapping | None,
                  incoherent: bool = False) -> dict:


    raw = row.get("raw_output")
    rec = {"generation_id": gid,
           **{k: meta.get(k, row.get(k)) for k in IDENTITY if k != "generation_id"},
           "question": (item or {}).get("question"), "choices": (item or {}).get("choices"),
           "answer_key": (item or {}).get("answer_key"),
           "answer_text": (item or {}).get("answer_text"),
           "model_answer": (answer or {}).get("pred"),
           "correct": (answer or {}).get("correct"),
           "baseline_answer": (answer or {}).get("baseline_pred"),
           "baseline_correct": (answer or {}).get("baseline_correct"),
           "outcome": (answer or {}).get("outcome"),
           "baseline_generation_id": (answer or {}).get("baseline_generation_id"),
           "finish_reason": row.get("finish_reason"), "incoherent": incoherent,
           "answer": answer_of(raw, expect_trace=row.get("think_state") == "on"),
           "explanation": explanation_of(raw) or None,
           "trace": trace_of(raw) or row.get("raw_reasoning")}
    for surface in SURFACES:
        rec.update(_judged(judged.get(surface), surface,
                           incoherent and surface == "trace"))
    return rec
