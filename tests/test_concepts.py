from __future__ import annotations

from src.data.build import GENERAL_CONCEPTS
from src.data.mmlu import CONCEPT_SUBJECTS
from src.data.retain import FAR_RETAIN, STEM_SUBJECT, UTILITY_SUBJECTS
from src.data.terms import CONCEPT_TERMS


def test_general_concepts_derive_from_concept_subjects():

    assert GENERAL_CONCEPTS == list(CONCEPT_SUBJECTS)


def test_every_concept_has_terms_and_retain_entries():
    for c in CONCEPT_SUBJECTS:
        assert c in CONCEPT_TERMS, f"{c}: no term set (leak flag + dependent-probe filter degrade)"
        assert c in FAR_RETAIN, f"{c}: no far-retain (load_retain does a hard dict lookup -> KeyError)"
        assert c in STEM_SUBJECT, f"{c}: no heatmap column"


def test_far_retain_never_draws_from_the_utility_pool():
    assert not (set(FAR_RETAIN.values()) & set(UTILITY_SUBJECTS))


def test_the_utility_pool_is_disjoint_from_the_wmdp_neighbour_tier():


    from src.data.wmdp import NEAR_RETAIN

    near = {s for subjects in NEAR_RETAIN.values() for s in subjects}
    assert not (near & set(UTILITY_SUBJECTS)), sorted(near & set(UTILITY_SUBJECTS))


def test_the_utility_pool_spans_more_than_the_humanities():


    stem = {"astronomy", "abstract_algebra", "college_mathematics", "conceptual_physics",
            "electrical_engineering", "high_school_physics", "econometrics"}
    assert len(UTILITY_SUBJECTS) >= 20, "a general-capability tier needs a representative sample"
    assert len(stem & set(UTILITY_SUBJECTS)) >= 5, "no STEM in the general-capability tier"
    assert len(UTILITY_SUBJECTS) == len(set(UTILITY_SUBJECTS)), "a duplicated subject double-counts"


def test_concept_never_evaluates_its_own_subject_as_retain():

    for c in CONCEPT_SUBJECTS:
        cross = [s for k, s in STEM_SUBJECT.items() if k != c]
        assert STEM_SUBJECT[c] not in cross


def test_wmdp_hazard_topics_have_a_vocabulary():
    from src.data.terms import has_vocabulary

    for c in ("wmdp-bio", "wmdp-cyber"):
        assert has_vocabulary(c), f"{c}: surface_gap would report 0.0% leakage on every row"


def test_has_vocabulary_is_false_when_terms_for_falls_back_to_the_id():
    from src.data.terms import has_vocabulary, terms_for

    assert terms_for("not-a-concept") == ["not-a-concept"]   
    assert not has_vocabulary("not-a-concept")


def test_the_wmdp_vocabulary_matches_real_answers_not_the_prompt_wording():


    from src.analysis.surface_gap import leak_pattern, leaks

    bio = leak_pattern("wmdp-bio", "general")
    assert leaks("Expression of a botulinum toxin gene in a poxvirus would be lethal.", bio)
    assert not leaks("The answer is B because the second option is larger.", bio)
    cyber = leak_pattern("wmdp-cyber", "general")
    assert leaks("A buffer overflow lets the payload achieve privilege escalation.", cyber)
    assert not leaks("Answer: C. Why: the third choice follows from the definition.", cyber)


def test_score_surface_gap_refuses_a_concept_with_no_vocabulary(tmp_path):
    import json
    import pathlib
    import subprocess
    import sys

    gp = tmp_path / "g.jsonl"
    gp.write_text(json.dumps({"generation_id": "x", "model": "M", "think_state": "na",
                              "prompt_id": "p", "concept_id": "wmdp-chem", "concept_class": "general",
                              "probe_type": "forget", "raw_output": "Answer: A"}) + "\n")
    r = subprocess.run([sys.executable, "scripts/score_surface_gap.py", "--generations", str(gp),
                        "--out-dir", str(tmp_path)],
                       cwd=pathlib.Path(__file__).resolve().parents[1],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "wmdp-chem" in (r.stdout + r.stderr) and "0.0%" in (r.stdout + r.stderr)
