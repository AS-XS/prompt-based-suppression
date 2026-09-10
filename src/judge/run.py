from __future__ import annotations

import re

from src.judge.schema import JudgeOutput

_RATING = re.compile(r"Rating:\s*\[\[\s*([0-2])\s*\]\]", re.IGNORECASE)


def extract_rating(text: str) -> int | None:


    ms = _RATING.findall(text or "")
    return int(ms[-1]) if ms else None


def judge_chat_kwargs(judge_name: str) -> dict:


    from src.gen.thinkstate import chat_kwargs

    try:
        return chat_kwargs(judge_name, "off")
    except Exception:
        return {}


def run_concept(llm, sampling, concept: str, sentence: str, chat_kw: dict | None = None) -> dict:


    from src.judge.prompts import concept_messages

    raw = llm.chat([concept_messages(concept, sentence)], sampling, **(chat_kw or {}))[0].outputs[0].text
    return {"concept_score": extract_rating(raw), "concept_raw": raw,
            "framing_score": None, "framing_raw": None}


def run_framing(llm, sampling, concept: str, sentence: str, chat_kw: dict | None = None) -> dict:

    from src.judge.prompts import framing_messages

    raw = llm.chat([framing_messages(concept, sentence)], sampling, **(chat_kw or {}))[0].outputs[0].text
    return {"framing_score": extract_rating(raw), "framing_raw": raw}


def run_judge(llm, sampling, concept: str, sentence: str, chat_kw: dict | None = None) -> dict:


    res = run_concept(llm, sampling, concept, sentence, chat_kw)


    if res["concept_score"] and res["concept_score"] > 0:
        res.update(run_framing(llm, sampling, concept, sentence, chat_kw))
    return res


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

    raw = llm.chat([primitive_messages(concept, sentence, surface, concept_class)],
                   sampling)[0].outputs[0].text
    parsed = parse_primitives(raw)
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


