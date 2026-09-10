from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gen.runner import iter_jsonl  # noqa: E402


_ROWS = [{"item_id": "a", "question": "plain"},
         {"item_id": "b", "question": "before after"},
         {"item_id": "c", "question": "also plain"}]


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "items.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in _ROWS) + "\n",
                 encoding="utf-8")
    return p


def test_splitlines_really_does_corrupt_this_file(tmp_path):

    text = _write(tmp_path).read_text(encoding="utf-8")
    assert len(text.splitlines()) == len(_ROWS) + 1, "U+2028 adds a phantom line break"
    with pytest.raises(json.JSONDecodeError):
        [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def test_iter_jsonl_reads_it_correctly(tmp_path):
    rows = list(iter_jsonl(_write(tmp_path)))
    assert [r["item_id"] for r in rows] == ["a", "b", "c"]
    assert rows[1]["question"] == "before after", "the character must survive intact"


def test_no_scorer_reads_jsonl_with_splitlines():

    offenders = []
    for py in sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "src").rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if ".splitlines()" in line and ("json" in line or "jsonl" in text[:400].lower()):
                if "read_text().splitlines()" in line:
                    offenders.append(f"{py.relative_to(ROOT)}:{i}")
    assert not offenders, f"read JSONL with iter_jsonl, not splitlines(): {offenders}"
