from __future__ import annotations

import warnings

import pytest
from pydantic import BaseModel

from src.gen.runner import read_jsonl


class _Row(BaseModel):
    gid: str
    val: int


def _write(tmp_path, *lines):
    p = tmp_path / "g.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_truncated_final_line_is_dropped_so_resume_works(tmp_path):
    p = _write(tmp_path,
               '{"gid":"a","val":1}',
               '{"gid":"b","val":2}',
               '{"gid":"c","val')                      
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rows = list(read_jsonl(p, _Row))
    assert [r.gid for r in rows] == ["a", "b"]         
    assert any("truncated final line" in str(x.message) for x in w)


def test_malformed_MIDDLE_line_raises_not_silently_skipped(tmp_path):
    p = _write(tmp_path,
               '{"gid":"a","val":1}',
               'GARBAGE',                              
               '{"gid":"c","val":3}')
    with pytest.raises(ValueError):
        list(read_jsonl(p, _Row))


def test_clean_file_reads_every_row(tmp_path):
    p = _write(tmp_path, '{"gid":"a","val":1}', '{"gid":"b","val":2}')
    assert [r.gid for r in read_jsonl(p, _Row)] == ["a", "b"]


def test_blank_lines_are_skipped(tmp_path):
    p = _write(tmp_path, '{"gid":"a","val":1}', '', '{"gid":"b","val":2}', '')
    assert [r.gid for r in read_jsonl(p, _Row)] == ["a", "b"]


MODEL = "Olmo-3-7B-Think"


class _Killed(Exception):
    pass


class _DyingLLM:


    def __init__(self, die_after):
        self.die_after, self.calls = die_after, 0

    def chat(self, messages, sampling, **kw):
        self.calls += 1
        if self.calls > self.die_after:
            raise _Killed("h_rt exceeded")
        out = type("X", (), {"text": "Answer: A", "token_ids": [1], "finish_reason": "stop"})()
        return [type("O", (), {"outputs": [out], "prompt_token_ids": [1, 2]})()
                for _ in messages]


def _gen_id(row: dict) -> str:

    return f"{row['config_hash']}:{row['item_id']}:{row['condition']}:{row['think_state']}"


def _items(n):
    from src.data.schema import Item
    return [Item(item_id=f"i{i}", concept_id="biology", concept_class="general",
                 concept_subtype="subject", probe_type="forget", question="q?",
                 answer_key="A", meta={"format": "mc", "choices": ["a", "b", "c", "d"]})
            for i in range(n)]


def test_a_job_killed_mid_wave_KEEPS_what_it_finished(tmp_path):

    import pytest
    from src.data.schema import Sampling
    from src.gen import sweep

    out = tmp_path / "gen.jsonl"
    samp = Sampling(temperature=0.0, top_p=1.0, max_tokens=10, seed=0)

    with pytest.raises(_Killed):
        sweep.run_sweep("Olmo-3-7B-Think", "path", _items(250), out, conditions=["suppressed"],
                        suppressed_prompt="no-meta", sampling=samp, llm=_DyingLLM(die_after=2),
                        chunk_size=100)

    kept = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(kept) == 200, "two chunks completed; both must be on disk"


def test_and_the_next_submission_resumes_from_there(tmp_path):
    import pytest
    from src.data.schema import Sampling
    from src.gen import sweep

    out = tmp_path / "gen.jsonl"
    samp = Sampling(temperature=0.0, top_p=1.0, max_tokens=10, seed=0)
    items = _items(250)

    with pytest.raises(_Killed):
        sweep.run_sweep(MODEL, "p", items, out, conditions=["suppressed"], suppressed_prompt="no-meta",
                        sampling=samp, llm=_DyingLLM(die_after=2), chunk_size=100)

    llm = _DyingLLM(die_after=99)
    n = sweep.run_sweep(MODEL, "p", items, out, conditions=["suppressed"],
                        suppressed_prompt="no-meta", sampling=samp, llm=llm, chunk_size=100)
    assert n == 50, "only the 50 outstanding rows were regenerated, not all 250"
    rows = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 250


    import json
    ids = [_gen_id(json.loads(l)) for l in rows]
    assert len(set(ids)) == 250, "no cell generated twice"


def test_chunking_does_not_change_what_is_written(tmp_path):

    import json
    from src.data.schema import Sampling
    from src.gen import sweep

    samp = Sampling(temperature=0.0, top_p=1.0, max_tokens=10, seed=0)

    def run(chunk):
        p = tmp_path / f"c{chunk}.jsonl"
        sweep.run_sweep(MODEL, "p", _items(120), p, conditions=["suppressed"],
                        suppressed_prompt="no-meta", sampling=samp,
                        llm=_DyingLLM(die_after=99), chunk_size=chunk)
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    small, big = run(10), run(1000)
    assert [_gen_id(r) for r in small] == [_gen_id(r) for r in big]
    assert [r["raw_output"] for r in small] == [r["raw_output"] for r in big]
