import re
from types import SimpleNamespace

import pytest

from src.unlearn.elm_targets import get_elm_peft_model


class Gemma4ClippableLinear:
    def __init__(self):
        self.linear = object()


class Block:
    def __init__(self, wrapped):
        self.proj = Gemma4ClippableLinear() if wrapped else object()

    def named_modules(self):
        yield "self_attn.q_proj", self.proj
        if hasattr(self.proj, "linear"):
            yield "self_attn.q_proj.linear", self.proj.linear
        yield "mlp.down_proj", object()
        yield "other", object()


@pytest.mark.parametrize("wrapped", [True, False])
def test_exact_decoder_targets_preserve_wrappers_and_layer_scope(wrapped):
    blocks = [Block(wrapped) for _ in range(3)]
    tower = SimpleNamespace(layers=blocks)
    model = SimpleNamespace(model=SimpleNamespace(language_model=tower)) if wrapped else SimpleNamespace(model=tower)
    prefix = "model.language_model.layers" if wrapped else "model.layers"
    config = SimpleNamespace(target_modules={"q_proj", "down_proj"}, layers_to_transform=[1], layers_pattern=None)
    original = blocks[1].proj
    captured = []
    result = get_elm_peft_model(model, config, lambda m, c: captured.append((m, c)) or m)
    assert result is model and captured == [(model, config)]
    suffix = ".linear" if wrapped else ""
    assert re.fullmatch(config.target_modules, f"{prefix}.1.self_attn.q_proj{suffix}")
    assert re.fullmatch(config.target_modules, f"{prefix}.1.mlp.down_proj")
    assert not re.fullmatch(config.target_modules, f"{prefix}.0.mlp.down_proj")
    assert not re.fullmatch(config.target_modules, "model.vision_tower.layers.1.mlp.down_proj")
    if wrapped:
        assert not re.fullmatch(config.target_modules, f"{prefix}.1.self_attn.q_proj")
    assert blocks[1].proj is original
    assert config.layers_to_transform is None


def test_empty_decoder_selection_fails_before_injection():
    model = SimpleNamespace(model=SimpleNamespace(layers=[Block(True)]))
    config = SimpleNamespace(target_modules={"missing_proj"}, layers_to_transform=[0])
    with pytest.raises(ValueError, match="No ELM projections"):
        get_elm_peft_model(model, config, lambda *_: pytest.fail("must not inject"))
