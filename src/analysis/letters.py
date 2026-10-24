LETTERS = ("A", "B", "C", "D")
MIN_PARSED = 30


def top_share(r: dict) -> tuple[float | None, str | None]:

    counts = {x: int(r[x]) for x in LETTERS}
    parsed = sum(counts.values())
    if not parsed:
        return None, None
    best = max(LETTERS, key=lambda x: counts[x])
    return counts[best] / parsed, best
