from __future__ import annotations

import re

from src.unlearn.rmu import decoder_layers_path


def get_elm_peft_model(model, config, get_peft_model):


    blocks, prefix = decoder_layers_path(model)
    targets = config.target_modules
    layers = config.layers_to_transform
    if not isinstance(targets, (list, set, tuple)) or not isinstance(layers, (list, tuple)):
        raise ValueError("ELM expects explicit projection names and decoder layer indices")
    paths = []
    for index in layers:
        if not 0 <= index < len(blocks):
            raise ValueError(f"ELM decoder layer out of range: {index}")
        selected = []
        for name, module in blocks[index].named_modules():
            if name.rsplit('.', 1)[-1] not in targets:
                continue
            path = f"{prefix}.{index}.{name}"
            if type(module).__name__ == "Gemma4ClippableLinear":
                if not hasattr(module, "linear"):
                    raise ValueError(f"Gemma clipping wrapper lacks linear child: {path}")
                path += ".linear"
            selected.append(path)
        if not selected:
            raise ValueError(f"No ELM projections found in decoder layer {index}")
        paths.extend(selected)
    if not paths:
        raise ValueError("No ELM adapter targets selected")

    config.target_modules = "(?:" + "|".join(re.escape(p) for p in paths) + ")"
    config.layers_to_transform = None
    config.layers_pattern = None
    print(f"ELM adapter targets ({len(paths)}): {paths}")
    return get_peft_model(model, config)
