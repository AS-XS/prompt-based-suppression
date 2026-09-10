from __future__ import annotations

import pytest

from src.provenance import select_one_config

_S4096 = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096, "seed": 0}
_S1024 = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024, "seed": 0}


def _row(chash, prompt, sampling=None, model="Qwen3-1.7B"):
    return {"config_hash": chash, "model": model, "sampling": sampling or _S4096,
            "prompt_id": prompt, "item_id": "i1"}


def test_the_four_prompt_waves_are_ALLOWED_despite_3_config_hashes():

    rows = [_row("aaa", "none-baseline"), _row("aaa", "world-frame"),
            _row("bbb", "ignorance-frame"), _row("ccc", "no-meta")]
    assert select_one_config(rows) == rows


def test_mixed_sampling_is_REFUSED():
    rows = [_row("aaa", "world-frame", _S4096), _row("zzz", "world-frame", _S1024)]
    with pytest.raises(SystemExit) as e:
        select_one_config(rows)
    msg = str(e.value)
    assert "regime" in msg and "4096" in msg and "1024" in msg


def test_mixed_models_is_ALLOWED_because_model_is_a_grouping_key():


    rows = [_row("aaa", "world-frame"), _row("aaa", "world-frame", model="gemma-4-E2B-it")]
    assert select_one_config(rows) == rows


def test_one_model_with_TWO_samplings_is_REFUSED():


    rows = [_row("aaa", "world-frame", _S4096), _row("zzz", "world-frame", _S1024)]
    with pytest.raises(SystemExit) as e:
        select_one_config(rows)
    assert "MORE THAN ONE sampling" in str(e.value)


def test_single_regime_passes():
    rows = [_row("aaa", "none-baseline"), _row("aaa", "none-baseline")]
    assert select_one_config(rows) == rows


def test_chosen_hash_still_filters_by_hand():
    rows = [_row("aaa", "world-frame"), _row("bbb", "no-meta")]
    kept = select_one_config(rows, "bbb")
    assert len(kept) == 1 and kept[0]["config_hash"] == "bbb"


def test_chosen_hash_that_matches_nothing_is_an_error():
    with pytest.raises(SystemExit):
        select_one_config([_row("aaa", "world-frame")], "zzz")


def test_prompt_text_is_in_the_config_hash_so_a_template_edit_cannot_be_skipped():


    from src.prompts import prompt_fingerprint
    from src.prompts import suppression as S
    from src.provenance import config_hash

    fp1 = prompt_fingerprint()
    assert fp1 == prompt_fingerprint()

    def cfg(fp):
        return {"model": "M", "suppressed_prompt": "world-frame", "prompt_fingerprint": fp,
                "sampling": {"temperature": 0.0}}

    original = S._TEMPLATES["world-frame"]
    try:
        S._TEMPLATES["world-frame"] = original + " (edited)"
        fp2 = prompt_fingerprint()
        assert fp2 != fp1
        assert config_hash(cfg(fp1)) != config_hash(cfg(fp2))
    finally:
        S._TEMPLATES["world-frame"] = original
    assert prompt_fingerprint() == fp1
