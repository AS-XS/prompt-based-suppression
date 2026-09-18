from __future__ import annotations

from src.gen.env import _HF_OVERRIDES, _olmo3_rope_override


class _Cfg:


    def __init__(self, rope_parameters):
        self.rope_parameters = rope_parameters


_NESTED = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 500000.0},
    "full_attention": {"rope_type": "yarn", "attention_factor": 1.2079441541679836,
                       "beta_fast": 32.0, "beta_slow": 1.0, "factor": 8.0,
                       "original_max_position_embeddings": 8192, "rope_theta": 500000},
}


def test_flattens_to_the_full_attention_entry():
    cfg = _olmo3_rope_override(_Cfg(dict(_NESTED)))

    assert cfg.rope_parameters["rope_type"] == "yarn"
    assert cfg.rope_parameters["factor"] == 8.0
    assert cfg.rope_parameters["rope_theta"] == 500000


def test_sliding_reconstruction_matches_the_models_own_config():

    cfg = _olmo3_rope_override(_Cfg(dict(_NESTED)))

    rebuilt = {"rope_type": "default", "rope_theta": cfg.rope_parameters["rope_theta"]}
    original = _NESTED["sliding_attention"]
    assert rebuilt["rope_type"] == original["rope_type"]
    assert float(rebuilt["rope_theta"]) == float(original["rope_theta"])


def test_noop_on_an_already_flat_config():
    flat = {"rope_type": "yarn", "rope_theta": 500000}
    assert _olmo3_rope_override(_Cfg(dict(flat))).rope_parameters == flat


def test_noop_when_config_has_no_rope_parameters():
    class _Bare:
        pass

    bare = _Bare()
    assert _olmo3_rope_override(bare) is bare


def test_override_is_registered_for_olmo3_paths():
    assert any(k in "models/olmo-3-7b-instruct" for k in _HF_OVERRIDES)


def test_unset_decode_knobs_are_absent_from_config_hash():


    from src.data.schema import Sampling
    assert Sampling(max_tokens=16384).model_dump(exclude_none=True) == {"max_tokens": 16384}
    old = Sampling(temperature=0.0, top_p=1.0, max_tokens=8192, seed=0)
    assert old.model_dump(exclude_none=True) == {
        "temperature": 0.0, "top_p": 1.0, "max_tokens": 8192, "seed": 0}


def test_max_tokens_is_required_because_vllms_default_is_16():


    from src.data.schema import Sampling
    assert Sampling().max_tokens == 1024, "max_tokens must keep a real default, never None"


def test_generation_config_auto_is_pinned_not_inherited():


    import inspect

    from src.gen import env
    src = inspect.getsource(env.make_llm)
    assert 'setdefault("generation_config", "auto")' in src
