from __future__ import annotations

from src.analysis.dependency import _jaccard, pairwise_matrix


def test_jaccard():
    assert _jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert _jaccard({1, 2}, {1, 2}) == 1.0
    assert _jaccard(set(), {1}) == 0.0


def test_pairwise_matrix_all_unordered_pairs():
    rows = pairwise_matrix(["a", "b", "c"],
                           overlap_fn=lambda x, y: 0.5, distance_fn=lambda x, y: 2)
    assert {(r["a"], r["b"]) for r in rows} == {("a", "b"), ("a", "c"), ("b", "c")}
    assert all(r["subtree_overlap"] == 0.5 and r["tree_distance"] == 2 for r in rows)
