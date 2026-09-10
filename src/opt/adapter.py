from __future__ import annotations

import numpy  # Load before DSPy installs its lazy module proxy.
from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter

RESPONSE_FIELD = "response"
QUESTION_FIELD = "question"


def _is_ours(signature) -> bool:


    return (list(signature.input_fields) == [QUESTION_FIELD]
            and list(signature.output_fields) == [RESPONSE_FIELD])


def _demo_text(demo, field: str) -> str:

    if hasattr(demo, "get"):
        return str(demo.get(field, "") or "")
    return str(getattr(demo, field, "") or "")


def _format_with_demos(clause: str, question: str, demos) -> list[dict[str, str]]:


    turns: list[dict[str, str]] = []
    for i, demo in enumerate(demos):
        dq = _demo_text(demo, QUESTION_FIELD)
        da = _demo_text(demo, RESPONSE_FIELD)
        if not dq or not da:


            raise ValueError(
                f"demo {i} is missing {QUESTION_FIELD!r} or {RESPONSE_FIELD!r}. Labeled demos come "
                f"from the trainset, so every example needs a `response` -- see "
                f"opt/dataset.py::build_examples.")
        turns.append({"role": "user", "content": f"{clause}\n\n{dq}" if (clause and not turns) else dq})
        turns.append({"role": "assistant", "content": da})
    turns.append({"role": "user", "content": question if turns else
                  (f"{clause}\n\n{question}" if clause else question)})
    return turns


class VerbatimAdapter(Adapter):


    _fallback = ChatAdapter()

    def format(self, signature, demos, inputs) -> list[dict[str, str]]:
        if not _is_ours(signature):
            return self._fallback.format(signature, demos, inputs)
        clause = (signature.instructions or "").strip()
        question = inputs[QUESTION_FIELD]
        if not demos:


            content = f"{clause}\n\n{question}" if clause else question
            return [{"role": "user", "content": content}]
        return _format_with_demos(clause, question, demos)

    def parse(self, signature, completion: str) -> dict[str, str]:

        if not _is_ours(signature):
            return self._fallback.parse(signature, completion)
        return {RESPONSE_FIELD: completion}
