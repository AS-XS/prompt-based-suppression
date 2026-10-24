from pathlib import Path

from src.analysis.grouping import eval_subject
from src.io import read_dicts


def mc_item_index(path: Path) -> dict[str, dict]:
    idx = {}
    for it in read_dicts(path):
        meta = it.get("meta") or {}
        idx[it["item_id"]] = {"answer_key": it.get("answer_key"), "format": meta.get("format"),
                              "mmlu_subject": eval_subject(it),
                              "probe_type": it.get("probe_type"), "concept_id": it.get("concept_id")}
    return idx
