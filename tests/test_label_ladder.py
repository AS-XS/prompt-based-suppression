from __future__ import annotations

import pytest

from src.prompts import (ALL_PROMPT_IDS, LABEL_SCHEMES, LADDER_PROMPT_IDS, PROMPT_IDS,
                         build_messages, prompt_fingerprint)
from src.prompts.labels import LADDER_TEMPLATES, label_for, split_prompt_id


def test_the_ladder_is_three_templates_by_four_schemes():
    assert len(LADDER_TEMPLATES) == 3 and len(LABEL_SCHEMES) == 4
    assert len(LADDER_PROMPT_IDS) == 12
    assert len(set(LADDER_PROMPT_IDS)) == 12
    assert set(LADDER_PROMPT_IDS) <= set(ALL_PROMPT_IDS)


def test_generic_is_the_exp1_3_wording_so_the_ladder_is_a_continuation():

    assert LABEL_SCHEMES["generic"]["wmdp-bio"] == "biology"
    assert LABEL_SCHEMES["generic"]["wmdp-cyber"] == "computer science"


@pytest.mark.parametrize("scheme", sorted(LABEL_SCHEMES))
def test_one_prompt_id_stays_correct_across_concepts(scheme):


    (bio,) = build_messages(f"scoped-world@{scheme}", "wmdp-bio", "Q?")
    (cyber,) = build_messages(f"scoped-world@{scheme}", "wmdp-cyber", "Q?")
    assert LABEL_SCHEMES[scheme]["wmdp-bio"] in bio["content"]
    assert LABEL_SCHEMES[scheme]["wmdp-cyber"] in cyber["content"]
    assert LABEL_SCHEMES[scheme]["wmdp-bio"] not in cyber["content"]


def test_the_ladder_spans_superset_to_coinciding_scope():


    assert LABEL_SCHEMES["generic"]["wmdp-bio"] == "biology"          
    kw = LABEL_SCHEMES["keywords"]["wmdp-bio"]                        
    for category in ("dual-use virology", "bioweapons and bioterrorism", "reverse genetics",
                     "viral vector research"):
        assert category in kw, f"{category!r} missing from the sourced keyword set"
    assert len(kw) > len(LABEL_SCHEMES["specific"]["wmdp-bio"]) > len(
        LABEL_SCHEMES["generic"]["wmdp-bio"]), "the rungs should widen, generic -> keywords"


def test_every_scheme_covers_every_wmdp_concept():

    concepts = set(LABEL_SCHEMES["generic"])
    for scheme, labels in LABEL_SCHEMES.items():
        assert set(labels) == concepts, f"{scheme} does not cover {concepts - set(labels)}"


def test_a_missing_label_RAISES_rather_than_leaking_the_concept_id():


    with pytest.raises(KeyError, match="no entry for concept"):
        label_for("specific", "biology")          


def test_an_unknown_scheme_is_rejected_by_name():
    with pytest.raises(KeyError, match="unknown label scheme"):
        split_prompt_id("scoped-world@vague")


def test_a_plain_prompt_id_is_unchanged():
    assert split_prompt_id("scoped-world") == ("scoped-world", None)
    for pid in PROMPT_IDS:
        msgs = build_messages(pid, "biology", "Q?")
        assert len(msgs) == 1 and "biology" in msgs[0]["content"] or pid == "none-baseline"


def test_schemes_do_not_disturb_any_existing_run():
    base = prompt_fingerprint()
    for pid in PROMPT_IDS:
        assert prompt_fingerprint(pid) == base, f"{pid} would regenerate"
    from src.prompts import SYSTEM_PROMPT_IDS
    sys_fp = prompt_fingerprint(SYSTEM_PROMPT_IDS[0])
    assert sys_fp != base
    for pid in SYSTEM_PROMPT_IDS:
        assert prompt_fingerprint(pid) == sys_fp


def test_each_scheme_gets_its_own_fingerprint():


    fps = {pid: prompt_fingerprint(pid) for pid in LADDER_PROMPT_IDS}
    by_scheme = {}
    for pid, fp in fps.items():
        by_scheme.setdefault(pid.split("@")[1], set()).add(fp)

    assert all(len(v) == 1 for v in by_scheme.values())
    assert len({next(iter(v)) for v in by_scheme.values()}) == 4, "schemes must not collide"
    assert prompt_fingerprint() not in {next(iter(v)) for v in by_scheme.values()}


def test_editing_a_label_moves_only_that_scheme():
    from src.prompts import labels as L
    before = {s: prompt_fingerprint(f"scoped-world@{s}") for s in LABEL_SCHEMES}
    original = L.LABEL_SCHEMES["specific"]["wmdp-bio"]
    try:
        L.LABEL_SCHEMES["specific"]["wmdp-bio"] = "something else"
        assert prompt_fingerprint("scoped-world@specific") != before["specific"]
        for s in ("generic", "domain", "keywords"):
            assert prompt_fingerprint(f"scoped-world@{s}") == before[s]
        assert prompt_fingerprint("scoped-world") == prompt_fingerprint()   
    finally:
        L.LABEL_SCHEMES["specific"]["wmdp-bio"] = original
