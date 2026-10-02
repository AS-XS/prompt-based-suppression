from __future__ import annotations

import re

from src.analysis.surfaces import split_think
from src.judge.schema import JudgeOutput

_RATING = re.compile(r"Rating:\s*\[\[\s*([0-2])\s*\]\]", re.IGNORECASE)


def extract_rating(text: str) -> int | None:


    ms = _RATING.findall(text or "")
    return int(ms[-1]) if ms else None


def verdict_text(raw: str | None, finish_reason: str | None = None) -> str:


    trace, answer = split_think(raw)
    if trace is not None:
        return answer
    return "" if finish_reason == "length" else answer


def _scored(stage: str, out) -> dict:


    o = out.outputs[0]
    fin = getattr(o, "finish_reason", None)
    ids = getattr(o, "token_ids", None)
    return {f"{stage}_score": extract_rating(verdict_text(o.text, fin)), f"{stage}_raw": o.text,
            f"{stage}_finish_reason": fin, f"{stage}_tokens": len(ids) if ids is not None else None}


def concept_scores(llm, sampling, pairs, chat_kw: dict | None = None) -> list[dict]:


    from src.judge.prompts import concept_messages

    if not pairs:
        return []
    outs = llm.chat([concept_messages(c, s) for c, s in pairs], sampling, **(chat_kw or {}))
    return [{**_scored("concept", o), "framing_score": None, "framing_raw": None} for o in outs]


def framing_scores(llm, sampling, pairs, chat_kw: dict | None = None) -> list[dict]:


    from src.judge.prompts import framing_messages

    if not pairs:
        return []
    outs = llm.chat([framing_messages(c, s) for c, s in pairs], sampling, **(chat_kw or {}))
    return [_scored("framing", o) for o in outs]


def mention_scores(llm, sampling, pairs, chat_kw: dict | None = None) -> list[dict]:


    from src.judge.prompts import mention_messages

    if not pairs:
        return []
    outs = llm.chat([mention_messages(c, s) for c, s in pairs], sampling, **(chat_kw or {}))
    return [_scored("mention", o) for o in outs]


def parse_primitives(raw_json: str) -> dict | None:


    import json

    try:
        return JudgeOutput.model_validate_json(raw_json).model_dump()
    except Exception:
        try:
            return JudgeOutput.model_validate(json.loads(raw_json)).model_dump()
        except Exception:
            return None


def run_primitive_judge(llm, sampling, concept: str, sentence: str, surface: str = "answer",
                        concept_class: str = "general") -> dict:


    from src.judge.prompts import primitive_messages

    o = llm.chat([primitive_messages(concept, sentence, surface, concept_class)],
                 sampling)[0].outputs[0]
    raw = o.text


    parsed = parse_primitives(verdict_text(raw, getattr(o, "finish_reason", None)))
    res = {"surface": surface, "raw_judge_json": raw, "parsed_ok": parsed is not None}
    for k in ("semantic_leak", "leak_type", "meta_reference", "fluent", "guessing"):
        res[k] = parsed.get(k) if parsed else None
    return res


def stub_judge(text: str, concept_label: str) -> JudgeOutput:


    leaked = concept_label.lower() in text.lower()
    return JudgeOutput(
        rationale=f"stub: concept term {'present' if leaked else 'absent'} in text",
        semantic_leak=leaked,
        leak_type="direct" if leaked else None,
        meta_reference="as if" in text.lower() or "restriction" in text.lower(),
        fluent=len(text.split()) >= 3,
        guessing="guess" in text.lower() or "not sure" in text.lower(),
    )


