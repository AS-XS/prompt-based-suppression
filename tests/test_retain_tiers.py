from src.analysis.retain_tiers import NEAR_NEIGHBOUR, near_label, retain_tier
from src.data.wmdp import NEAR_RETAIN


def test_near_is_a_subset_of_the_declared_neighbours():
    for concept, near in NEAR_NEIGHBOUR.items():
        assert concept in NEAR_RETAIN, concept
        assert set(near) <= set(NEAR_RETAIN[concept]), (concept, near)


def test_near_plus_general_is_exactly_the_declared_set():

    for concept, declared in NEAR_RETAIN.items():
        split = {s for s in declared if retain_tier(concept, s) in ("near", "general")}
        assert split == set(declared), concept


def test_the_two_decided_near_neighbours():
    assert retain_tier("wmdp-cyber", "computer_security") == "near"
    assert retain_tier("wmdp-cyber", "college_computer_science") == "general"
    assert retain_tier("wmdp-bio", "virology") == "near"
    assert retain_tier("wmdp-bio", "college_biology") == "general"
    assert retain_tier("wmdp-bio", "marketing") == "far"

    assert retain_tier("wmdp-bio", "computer_security") == "far"


def test_a_concept_without_a_near_tier():
    assert retain_tier("wmdp-chem", "college_chemistry") == "general"
    assert near_label("wmdp-chem") == "—"
    assert near_label("wmdp-cyber") == "COMPUTER SECURITY"
