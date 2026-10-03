from __future__ import annotations

import pytest

from src.prompts import (ALL_PROMPT_IDS, PROMPT_IDS, SYSTEM_PROMPT_IDS, build_messages,
                         prompt_fingerprint)
from src.prompts import system as sysmod
from src.prompts.suppression import _TEMPLATES


def test_exp3_does_not_move_the_exp1_or_exp2_fingerprint():


    base = prompt_fingerprint()
    for pid in PROMPT_IDS:
        assert prompt_fingerprint(pid) == base, f"{pid} would regenerate"
    for pid in SYSTEM_PROMPT_IDS:
        assert prompt_fingerprint(pid) != base


def test_the_system_templates_stay_OUT_of_the_user_turn_registry():


    assert set(_TEMPLATES) == set(PROMPT_IDS)
    assert not set(_TEMPLATES) & set(SYSTEM_PROMPT_IDS)


def test_editing_a_system_template_moves_only_exp3():
    unrelated = prompt_fingerprint("scoped-ignorance")
    before = prompt_fingerprint("sys-scoped-world")
    original = dict(sysmod._FROM_USER_PROMPT)
    try:
        sysmod._FROM_USER_PROMPT["sys-scoped-world"] = "no-meta"   
        assert prompt_fingerprint("sys-scoped-world") != before
        assert prompt_fingerprint("scoped-ignorance") == unrelated  
    finally:
        sysmod._FROM_USER_PROMPT.clear()
        sysmod._FROM_USER_PROMPT.update(original)
    assert prompt_fingerprint("sys-scoped-world") == before


def test_exp1_and_exp2_still_build_a_single_user_turn():

    for pid in PROMPT_IDS:
        msgs = build_messages(pid, "biology", "Q?")
        assert len(msgs) == 1 and msgs[0]["role"] == "user"


def test_a_system_prompt_builds_two_turns_with_a_bare_question():
    msgs = build_messages("sys-scoped-ignorance", "biology", "What is a ribosome?")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "What is a ribosome?"      
    assert "biology" in msgs[0]["content"]
    assert "{concept}" not in msgs[0]["content"]            
    assert "{question}" not in msgs[0]["content"]           


@pytest.mark.parametrize("sys_id,exp2_id", [("sys-scoped-ignorance", "scoped-ignorance"),
                                            ("sys-scoped-world", "scoped-world"),
                                            ("sys-scoped-no-meta", "scoped-no-meta")])
def test_the_wording_is_byte_identical_to_exp2(sys_id, exp2_id):


    (user_msg,) = build_messages(exp2_id, "biology", "What is a ribosome?")
    system_msg, probe = build_messages(sys_id, "biology", "What is a ribosome?")
    assert user_msg["content"] == system_msg["content"] + "\n\n" + probe["content"]


def test_a_broken_derivation_raises_instead_of_leaking_a_placeholder():


    original = _TEMPLATES["scoped-world"]
    try:
        _TEMPLATES["scoped-world"] = "no question at the end"
        with pytest.raises(ValueError, match="cannot be derived"):
            build_messages("sys-scoped-world", "biology", "Q?")
    finally:
        _TEMPLATES["scoped-world"] = original


def test_the_union_is_exported_and_an_unknown_id_names_every_option():


    from src.prompts import (CONDITIONAL_PROMPT_IDS, DIRECTED_PROMPT_IDS, LADDER_PROMPT_IDS, LI_PROMPT_IDS,
                             PUBLISHED_PROMPT_IDS)
    assert set(ALL_PROMPT_IDS) == (set(PROMPT_IDS) | set(SYSTEM_PROMPT_IDS)
                                   | set(PUBLISHED_PROMPT_IDS) | set(LADDER_PROMPT_IDS)
                                   | set(DIRECTED_PROMPT_IDS) | set(LI_PROMPT_IDS)
                                   | set(CONDITIONAL_PROMPT_IDS))
    with pytest.raises(KeyError, match="sys-scoped-ignorance"):
        build_messages("nonsense", "biology", "Q?")


class _FakeTok:


    def __init__(self, mode):
        self.mode = mode

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if self.mode == "reject" and any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        if self.mode == "drop":
            return "".join(f"<u>{m['content']}" for m in messages if m["role"] == "user")
        if self.mode == "merge":   
            glued = "\n\n".join(m["content"] for m in messages)
            return f"<u>{glued}"
        return "".join(f"<{m['role'][0]}>{m['content']}" for m in messages)


@pytest.mark.parametrize("mode,expected", [("separate", "SEPARATE"), ("merge", "MERGED"),
                                           ("drop", "DROPPED"), ("reject", "REJECTED")])
def test_check_system_classifies_each_way_a_template_can_ignore_the_role(mode, expected):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "check_system", Path(__file__).resolve().parents[1] / "scripts" / "check_system.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    verdict, _ = mod.classify(_FakeTok(mode))
    assert verdict == expected
