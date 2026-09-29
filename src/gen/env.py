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


    kwargs.setdefault("generation_config", "auto")
    """Construct a vLLM `LLM`, forcing a **prebuilt** attention backend so nothing JIT-compiles
    at runtime (avoids FlashInfer's `nvcc` requirement on compute nodes). Falls back to vLLM's
    default selection if the build doesn't accept the kwarg, or if the model rejects the backend."""


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


RESOLVED_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                 "presence_penalty", "frequency_penalty", "max_tokens", "seed", "n")


def resolved_of(sp) -> dict | None:

    if sp is None:
        return None
    return {k: getattr(sp, k, None) for k in RESOLVED_KEYS}


def model_default_sampling(llm, *, strict: bool = False):


    try:
        return llm.get_default_sampling_params()
    except Exception as e:
        if strict:
            raise SystemExit(f"cannot read {type(llm).__name__}'s own decode defaults "
                             f"({type(e).__name__}: {e}); refusing to run at vLLM's bare defaults")
        print(f"[make_engine] get_default_sampling_params unavailable ({type(e).__name__}); "
              f"falling back to vLLM's bare defaults — decode will NOT be the model's own",
              flush=True)
        from vllm import SamplingParams

        return SamplingParams()


def constrain_to_schema(sp, schema: dict):


    try:
        from vllm.sampling_params import StructuredOutputsParams

        try:
            sp.structured_outputs = StructuredOutputsParams(json=schema)
        except TypeError:
            sp.structured_outputs = StructuredOutputsParams(json_schema=schema)
        return sp
    except ImportError:
        pass
    from vllm.sampling_params import GuidedDecodingParams

    sp.guided_decoding = GuidedDecodingParams(json=schema)
    return sp
