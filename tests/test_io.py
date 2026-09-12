import json

import pytest

from src.io import read_dicts, write_csv


def test_jsonl_preserves_records_order_and_unicode(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"text": "before\u2028after\u0085é", "n": None}, {"text": "duplicate"},
            {"text": "duplicate"}, [1, True], None]
    path.write_text(" \r\n" + "\r\n\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
                    + "\n\t\n", encoding="utf-8")
    assert list(read_dicts(path)) == rows


def test_jsonl_opens_lazily_and_requires_the_file(tmp_path):
    rows = read_dicts(tmp_path / "missing.jsonl")
    with pytest.raises(FileNotFoundError):
        next(rows)


@pytest.mark.parametrize("suffix", ['{"unfinished":', 'bad json\n{"ok": 2}\n'])
def test_jsonl_raises_at_corruption_after_yielding_valid_rows(tmp_path, suffix):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"ok": 1}\n' + suffix, encoding="utf-8")
    rows = read_dicts(path)
    assert next(rows) == {"ok": 1}
    with pytest.raises(json.JSONDecodeError):
        next(rows)


@pytest.mark.parametrize("module", ["scripts.judge", "scripts.judge_primitives"])
def test_judge_resume_reader_still_allows_missing_files(tmp_path, module):
    from importlib import import_module

    assert list(import_module(module)._read_dicts(tmp_path / "missing.jsonl")) == []


def test_csv_preserves_column_order_quoting_and_missing_values(tmp_path):
    path = tmp_path / "nested" / "table.csv"
    write_csv(path, [{"text": 'é,"quoted"\nnext', "n": None, "extra": "ignored"},
                     {"n": 2}], ["n", "text"])
    assert path.read_bytes() == 'n,text\r\n,"é,""quoted""\nnext"\r\n2,\r\n'.encode("utf-8")


@pytest.mark.parametrize("columns,expected", [(["n", "text"], b"n,text\r\n"),
                                              ([], b"\r\n")])
def test_csv_empty_input_still_writes_header(tmp_path, columns, expected):
    path = tmp_path / "table.csv"
    write_csv(path, [], columns)
    assert path.read_bytes() == expected
