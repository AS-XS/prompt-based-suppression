from __future__ import annotations

from src.prompts.labels import label_for, split_prompt_id
from src.prompts.suppression import _TEMPLATES


def seed_clause(prompt_id: str, concept_id: str) -> str:


    template_id, scheme = split_prompt_id(prompt_id)
    label = label_for(scheme, concept_id) if scheme else concept_id


    from src.prompts.directed import build_directed_messages, is_directed_prompt, takes_examples
    if is_directed_prompt(template_id):
        if takes_examples(template_id):
            raise SystemExit(
                f"{prompt_id!r} renders worked examples inline, so its clause is not separable from "
                f"its demos — seeding from it would put examples inside the text the optimiser "
                f"rewrites. Seed from 'scoped-directed' and pass --demos to add examples, which "
                f"keeps the two interventions separable.")
        msgs = build_directed_messages(template_id, label, "")
        return msgs[0]["content"].strip()

    if template_id not in _TEMPLATES:
        raise SystemExit(f"unknown seed prompt_id {prompt_id!r}")
    return _TEMPLATES[template_id].format(concept=label, question="").strip()


def build_program(seed: str):

    import dspy

    class Suppressed(dspy.Signature):
        question: str = dspy.InputField(desc="the multiple-choice item, already carrying the "
                                             "frozen answer-format instruction")
        response: str = dspy.OutputField(desc="the model's raw answer, parsed downstream")

    return dspy.Predict(Suppressed.with_instructions(seed))


def clause_of(program) -> str:

    predictors = list(program.named_predictors())
    if len(predictors) != 1:
        raise AssertionError(f"expected exactly one predictor, got {len(predictors)}")
    return predictors[0][1].signature.instructions


def instrument_is_frozen(program, expected_desc: str) -> bool:


    field = clause_signature(program).output_fields["response"]
    return (field.json_schema_extra or {}).get("desc") == expected_desc


def clause_signature(program):
    return list(program.named_predictors())[0][1].signature
