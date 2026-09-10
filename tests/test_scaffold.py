from __future__ import annotations

from src.analysis.modes import MODES, derive_mode
from src.data.demo import demo_item
from src.data.schema import GenerationRow, Sampling, make_item_id
from src.gen.runner import stub_generate
from src.judge.schema import JudgeOutput, judge_json_schema


def _cell() -> dict:
    return {"config_id": "t", "model": "stub", "prompt_id": "world-frame",
            "condition": "suppressed", "think_state": "na"}


def test_item_id_is_deterministic():
    a = make_item_id("biology", "forget", "q?")
    b = make_item_id("biology", "forget", "q?")
    assert a == b


def test_generation_row_roundtrips():
    row = stub_generate(demo_item(), _cell(), Sampling(temperature=0.0, seed=0))
    back = GenerationRow.model_validate_json(row.model_dump_json())
    assert back == row

    assert back.raw_output and back.raw_reasoning is None


def test_generation_id_is_stable():
    row = stub_generate(demo_item(), _cell(), Sampling(temperature=0.0, seed=0))
    assert row.generation_id.count(":") == 3


def test_mode_success():

    j = JudgeOutput(rationale="", semantic_leak=False, meta_reference=False, fluent=True,
                    guessing=False)
    assert derive_mode(j) == "success"


def test_mode_guessed_splits_off_success():

    j = JudgeOutput(rationale="", semantic_leak=False, meta_reference=False, fluent=True,
                    guessing=True)
    assert derive_mode(j) == "guessed"


def test_mode_meta_framed():
    j = JudgeOutput(rationale="", semantic_leak=False, meta_reference=True, fluent=True,
                    guessing=False)
    assert derive_mode(j) == "meta_framed"


def test_mode_meta_beats_guessing():

    j = JudgeOutput(rationale="", semantic_leak=False, meta_reference=True, fluent=True,
                    guessing=True)
    assert derive_mode(j) == "meta_framed"


def test_mode_incoherent_beats_meta():
    j = JudgeOutput(rationale="", semantic_leak=False, meta_reference=True, fluent=False,
                    guessing=False)
    assert derive_mode(j) == "incoherent"


def test_mode_leak_wins():

    j = JudgeOutput(rationale="", semantic_leak=True, leak_type="direct",
                    meta_reference=True, fluent=False, guessing=True)
    assert derive_mode(j) == "leaked"


def test_judge_schema_has_the_primitives():
    props = judge_json_schema()["properties"]
    for key in ("rationale", "semantic_leak", "leak_type", "meta_reference", "fluent", "guessing"):
        assert key in props
    assert set(MODES) == {"success", "meta_framed", "leaked", "incoherent", "guessed"}


def test_scoped_ignorance_prompt_is_registered_and_renders():


    from src.prompts import PROMPT_IDS, build_messages

    assert "scoped-ignorance" in PROMPT_IDS
    assert {"none-baseline", "world-frame", "ignorance-frame", "no-meta"} <= set(PROMPT_IDS)

    (msg,) = build_messages("scoped-ignorance", "chemistry", "What is the pH of water?")
    text = msg["content"]
    assert text.startswith("Pretend you know nothing about chemistry.")
    assert "definitions of related terms" in text          
    assert "background knowledge about other subjects" in text   
    assert text.endswith("What is the pH of water?")       


    assert "make a random guess" in text
    assert "you may say so" not in text


def test_editing_a_template_moves_the_prompt_fingerprint():


    from src.prompts import prompt_fingerprint
    from src.prompts import suppression as S

    fp = prompt_fingerprint()
    original = S._TEMPLATES["scoped-ignorance"]
    try:
        S._TEMPLATES["scoped-ignorance"] = original + " x"
        assert prompt_fingerprint() != fp
    finally:
        S._TEMPLATES["scoped-ignorance"] = original
    assert prompt_fingerprint() == fp


def test_the_scoped_prompts_form_a_framing_x_scoping_factorial():


    from src.prompts import suppression as S

    shared = ("Do not use any background knowledge of {concept} when answering the question, "
              "including definitions of related terms or concepts. You may still use background "
              "knowledge about other subjects.")
    guess = "make a random guess"
    for pid in ("scoped-ignorance", "scoped-world", "scoped-no-meta"):
        t = S._TEMPLATES[pid]
        assert shared in t, pid
        assert guess in t, pid


    assert S._TEMPLATES["scoped-ignorance"].startswith("Pretend you know nothing about")
    assert S._TEMPLATES["scoped-world"].startswith("Respond from within a world in which")

    gag = "Do NOT mention, explain, or refer to this restriction"
    assert gag in S._TEMPLATES["scoped-no-meta"] and gag in S._TEMPLATES["no-meta"]
    assert gag not in S._TEMPLATES["scoped-ignorance"] and gag not in S._TEMPLATES["scoped-world"]
