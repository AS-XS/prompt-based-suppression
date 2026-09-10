from __future__ import annotations

from collections import defaultdict

from src.data.mmlu import CONCEPT_SUBJECTS
from src.data.terms import terms_for


def ensure_wordnet() -> None:

    try:
        from nltk.corpus import wordnet as wn

        wn.synsets("test")
    except LookupError:
        import nltk

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def hyponym_subtree_size(term: str) -> int:

    from nltk.corpus import wordnet as wn

    seen: set = set()
    for syn in wn.synsets(term.replace(" ", "_"), pos=wn.NOUN):
        seen.update(syn.closure(lambda s: s.hyponyms()))
    return len(seen)


def _hyponym_closure(term: str) -> set:
    from nltk.corpus import wordnet as wn

    seen: set = set()
    for syn in wn.synsets(term.replace(" ", "_"), pos=wn.NOUN):
        seen.add(syn)
        seen.update(syn.closure(lambda s: s.hyponyms()))
    return seen


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return round(len(a & b) / union, 4) if union else 0.0


def subtree_overlap(term_a: str, term_b: str) -> float:


    return _jaccard(_hyponym_closure(term_a), _hyponym_closure(term_b))


def tree_distance(term_a: str, term_b: str):


    from nltk.corpus import wordnet as wn

    best = None
    for sa in wn.synsets(term_a.replace(" ", "_"), pos=wn.NOUN):
        for sb in wn.synsets(term_b.replace(" ", "_"), pos=wn.NOUN):
            d = sa.shortest_path_distance(sb)
            if d is not None and (best is None or d < best):
                best = d
    return best


def pairwise_matrix(terms: list[str], overlap_fn=None, distance_fn=None) -> list[dict]:


    if overlap_fn is None or distance_fn is None:
        ensure_wordnet()
    overlap_fn = overlap_fn or subtree_overlap
    distance_fn = distance_fn or tree_distance
    rows = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            a, b = terms[i], terms[j]
            rows.append({"a": a, "b": b,
                         "subtree_overlap": overlap_fn(a, b), "tree_distance": distance_fn(a, b)})
    return rows


def cross_subject_mention_fraction(terms: list[str], subject_questions: dict[str, list[str]],
                                   exclude: set[str] | None = None) -> float:

    exclude = exclude or set()
    low = [t.lower() for t in terms]
    subjects = {s: qs for s, qs in subject_questions.items() if s not in exclude}
    if not subjects:
        return 0.0
    hit = sum(any(any(t in q for t in low) for q in qs) for qs in subjects.values())
    return hit / len(subjects)


def mmlu_subject_questions(split: str = "test") -> dict[str, list[str]]:

    from datasets import load_dataset

    out: dict[str, list[str]] = defaultdict(list)
    for r in load_dataset("cais/mmlu", "all", split=split):
        out[r["subject"]].append(r["question"].lower())
    return dict(out)


def compute_rows(concepts: list[tuple[str, str]],
                 subject_questions: dict[str, list[str]]) -> list[dict]:

    ensure_wordnet()
    rows = []
    for cls, cid in concepts:
        terms = terms_for(cid) if cls == "general" else [cid]
        own = set(CONCEPT_SUBJECTS.get(cid, [])) if cls == "general" else set()
        rows.append({
            "concept_class": cls,
            "concept_id": cid,
            "wordnet_subtree": hyponym_subtree_size(cid),
            "cross_subject_frac": round(
                cross_subject_mention_fraction(terms, subject_questions, exclude=own), 4),
        })
    return rows
