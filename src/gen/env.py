from __future__ import annotations

import os
import shutil


def setup_vllm_env() -> None:

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    nvcc = shutil.which("nvcc")
    if nvcc and "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc))


    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_NO_FLASH_ATTN = ("gpt-oss", "gemma-4", "gemma-3n")


def _olmo3_rope_override(config):


    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and "full_attention" in rp:
        config.rope_parameters = dict(rp["full_attention"])
    return config


_HF_OVERRIDES = {"olmo-3": _olmo3_rope_override}


def make_llm(model: str, backend: str = "FLASH_ATTN", **kwargs):


    setup_vllm_env()

    from vllm import LLM

    name = model.lower()
    if "hf_overrides" not in kwargs:  
        for key, fix in _HF_OVERRIDES.items():
            if key in name:
                kwargs["hf_overrides"] = fix
                break
    if backend and any(k in name for k in _NO_FLASH_ATTN):
        backend = None
    if not backend:
        return LLM(model=model, **kwargs)
    try:
        return LLM(model=model, attention_backend=backend, **kwargs)
    except TypeError:
        return LLM(model=model, **kwargs)
    except Exception as e:


        print(f"[make_llm] {backend} rejected for {model} ({type(e).__name__}) — "
              f"retrying with vLLM's default backend", flush=True)
        return LLM(model=model, **kwargs)


def guided_sampling(schema: dict, temperature: float = 0.0, max_tokens: int = 400):


    from vllm import SamplingParams

    try:
        from vllm.sampling_params import StructuredOutputsParams

        try:
            so = StructuredOutputsParams(json=schema)
        except TypeError:
            so = StructuredOutputsParams(json_schema=schema)
        return SamplingParams(temperature=temperature, max_tokens=max_tokens, structured_outputs=so)
    except ImportError:
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              guided_decoding=GuidedDecodingParams(json=schema))
    except (ImportError, TypeError):
        return SamplingParams(temperature=temperature, max_tokens=max_tokens, guided_json=schema)
