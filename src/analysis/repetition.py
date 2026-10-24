SHINGLE = 12
TAIL = 4000
LOOP_AT = 0.5


def loop_score(text: str) -> float:

    words = text[-TAIL:].split()
    if len(words) < SHINGLE * 3:
        return 0.0
    grams = [" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE)]
    seen: set[str] = set()
    rep = 0
    for g in grams:
        if g in seen:
            rep += 1
        seen.add(g)
    return rep / len(grams)
