from __future__ import annotations

from src.data.schema import Item

_LETTERS = ["A", "B", "C", "D"]

DEFAULT_ANSWER_FORMAT = "letter-why"

_MC_INSTRUCTIONS: dict[str, str] = {


    "letter-why": (
        "First answer with the single letter of the correct choice, then briefly say why, "
        "in exactly this format:\n"
        "Answer: <letter>\n"
        "Why: <one or two sentences>\n"
        "You must always pick one letter, even if you are unsure — guess if you have to."
    ),


    "letter-only": (
        "Answer with the single letter of the correct choice, in exactly this format:\n"
        "Answer: <letter>\n"
        "Give no explanation and write nothing else.\n"
        "You must always pick one letter, even if you are unsure — guess if you have to."
    ),


    "li-tags": "",
}

ANSWER_FORMATS: tuple[str, ...] = tuple(_MC_INSTRUCTIONS)


MC_INSTRUCTION = _MC_INSTRUCTIONS[DEFAULT_ANSWER_FORMAT]


def mc_instruction(answer_format: str = DEFAULT_ANSWER_FORMAT) -> str:


    try:
        return _MC_INSTRUCTIONS[answer_format]
    except KeyError:
        raise KeyError(f"unknown answer_format {answer_format!r}; "
                       f"expected one of {ANSWER_FORMATS}") from None


def render_question(item: Item, answer_format: str = DEFAULT_ANSWER_FORMAT) -> str:

    meta = item.meta or {}
    if meta.get("format") == "mc":
        choices = meta.get("choices") or []
        opts = "\n".join(f"{_LETTERS[i]}) {c}" for i, c in enumerate(choices))


        instruction = mc_instruction(answer_format)
        return f"{item.question}\n\n{opts}\n\n{instruction}" if instruction else \
               f"{item.question}\n\n{opts}"
    return item.question
