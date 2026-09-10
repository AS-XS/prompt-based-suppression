from __future__ import annotations

from src.prompts.suppression import user_template


_FROM_USER_PROMPT: dict[str, str] = {
    "sys-scoped-ignorance": "scoped-ignorance",
    "sys-scoped-world": "scoped-world",
    "sys-scoped-no-meta": "scoped-no-meta",
}

SYSTEM_PROMPT_IDS: tuple[str, ...] = tuple(_FROM_USER_PROMPT)


_QUESTION_SUFFIX = "\n\n{question}"


def is_system_prompt(prompt_id: str) -> bool:

    return prompt_id in _FROM_USER_PROMPT


def instruction_for(prompt_id: str) -> str:

    source = _FROM_USER_PROMPT[prompt_id]
    template = user_template(source)
    if not template.endswith(_QUESTION_SUFFIX):
        raise ValueError(
            f"{source!r} no longer ends with {_QUESTION_SUFFIX!r}, so {prompt_id!r} cannot be "
            "derived from it. Fix the derivation rather than re-typing the wording — identical "
            "text is the whole basis for reading exp3 vs exp2 as a position effect."
        )
    return template[: -len(_QUESTION_SUFFIX)]


def build_system_messages(prompt_id: str, concept_label: str, question: str) -> list[dict[str, str]]:


    return [
        {"role": "system", "content": instruction_for(prompt_id).format(concept=concept_label)},
        {"role": "user", "content": question},
    ]


def system_fingerprint_payload() -> str:


    parts = []
    for pid in SYSTEM_PROMPT_IDS:
        messages = build_system_messages(pid, "{concept}", "{question}")
        parts.append(pid + "=" + "\x01".join(f"{m['role']}:{m['content']}" for m in messages))
    return "\x00".join(parts)
