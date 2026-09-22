from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RMUConfig:


    layer: int
    update_layers: tuple[int, ...] = ()
    steering_coeff: float = 6.5
    alpha: float = 1200.0
    lr: float = 5e-5
    max_batches: int = 150
    batch_size: int = 4
    max_chars: int = 2000
    seed: int = 42

    def layers_to_update(self) -> tuple[int, ...]:
        return self.update_layers or tuple(range(max(self.layer - 2, 0), self.layer + 1))


@dataclass
class RMUReport:


    config: dict
    model: str
    concept: str
    n_forget: int
    n_retain: int
    losses: list[float] = field(default_factory=list)


    forget_losses: list[float] = field(default_factory=list)
    retain_losses: list[float] = field(default_factory=list)


    forget_activation_norm: float | None = None


    forget_activation_rms: float | None = None


    fluency_unique_ratio: float | None = None
    fluency_samples: list[str] = field(default_factory=list)


    git_commit: str | None = None


def activation_index(layer: int, n_hidden_states: int) -> int:


    idx = layer + 1
    if not 0 <= idx < n_hidden_states:
        raise SystemExit(
            f"--layer {layer} needs hidden_states[{idx}], but this model exposes "
            f"{n_hidden_states} (layers 0..{n_hidden_states - 2}). Layer indices are "
            f"architecture-specific.")
    return idx


def _hidden_at(model, tok, texts: list[str], layer: int, device, max_chars: int):

    import torch

    batch = tok([t[:max_chars] for t in texts], return_tensors="pt",
                padding=True, truncation=True, max_length=512).to(device)
    with torch.set_grad_enabled(model.training):
        out = model(**batch, output_hidden_states=True)
    return out.hidden_states[activation_index(layer, len(out.hidden_states))]


def run_rmu(model_path: str, out_dir: str | Path, *, concept: str,
            forget: list[str], retain: list[str], cfg: RMUConfig,
            dtype: str = "bfloat16", device: str = "cuda",
            frozen_dtype: str | None = None) -> RMUReport:


    src, dst = Path(model_path).resolve(), Path(out_dir).resolve()
    if dst == src or src in dst.parents or dst in src.parents:
        raise SystemExit(
            f"refusing to train: the output path is inside (or is) the source model.\n"
            f"  source {src}\n  output {dst}\n"
            f"Unlearned checkpoints belong in models/unlearned/<name>, never under MODELS_DIR.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    before = _fingerprint(src)
    torch.manual_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token


    frozen_dtype = frozen_dtype or dtype
    updated = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype)).to(device)
    frozen = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=getattr(torch, frozen_dtype)).to(device)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)


    for p in updated.parameters():
        p.requires_grad_(False)
    params = []
    layers = _decoder_layers(updated)
    for i in cfg.layers_to_update():
        if i >= len(layers):
            raise SystemExit(f"--layer {cfg.layer} implies layer {i}, but the model has "
                             f"{len(layers)} layers. Layer indices are model-specific.")
        proj = layers[i].mlp.down_proj
        proj.weight.requires_grad_(True)
        params.append(proj.weight)
    updated.train()


    hidden = hidden_size_of(updated.config)
    u = torch.rand(1, 1, hidden, dtype=getattr(torch, dtype), device=device)
    u = u / u.norm()
    control = u * cfg.steering_coeff

    opt = torch.optim.AdamW(params, lr=cfg.lr)
    report = RMUReport(config=vars(cfg) | {"layers_to_update": list(cfg.layers_to_update()),
                                           "dtype": dtype, "frozen_dtype": frozen_dtype},
                       model=model_path, concept=concept,
                       n_forget=len(forget), n_retain=len(retain),
                       git_commit=_git_commit())
    print(f"  code: commit {report.git_commit} · c={cfg.steering_coeff} alpha={cfg.alpha} "
          f"layers={list(cfg.layers_to_update())} read_at=hidden_states"
          f"[{activation_index(cfg.layer, len(_decoder_layers(updated)) + 1)}]", flush=True)

    n = min(cfg.max_batches, len(forget) // cfg.batch_size, len(retain) // cfg.batch_size)
    if n == 0:
        raise SystemExit(f"not enough corpus for one batch: {len(forget)} forget / {len(retain)} "
                         f"retain at batch_size {cfg.batch_size}")
    for step in range(n):
        s = slice(step * cfg.batch_size, (step + 1) * cfg.batch_size)
        fh = _hidden_at(updated, tok, forget[s], cfg.layer, device, cfg.max_chars)
        forget_loss = torch.nn.functional.mse_loss(fh, control.expand_as(fh))

        if step == 0:


            with torch.no_grad():
                h = fh.detach().float()
                report.forget_activation_norm = float(h.norm(dim=-1).mean())
                report.forget_activation_rms = float(h.pow(2).mean().sqrt() * hidden ** 0.5)


                h_term = float(h.pow(2).mean())
                c_term = cfg.steering_coeff ** 2 / hidden
            print(f"  scale: ||h_forget|| mean {report.forget_activation_norm:.2f} · "
                  f"rms {report.forget_activation_rms:.2f}  vs  ||c*u|| {cfg.steering_coeff:.2f}\n"
                  f"  forget_loss at step 0 ~= {h_term:.4f} (activations) + {c_term:.4f} (target); "
                  f"alpha*retain reaches this scale at retain MSE {h_term / cfg.alpha:.2e}",
                  flush=True)

        rh = _hidden_at(updated, tok, retain[s], cfg.layer, device, cfg.max_chars)
        with torch.no_grad():
            rh0 = _hidden_at(frozen, tok, retain[s], cfg.layer, device, cfg.max_chars)


        retain_loss = torch.nn.functional.mse_loss(rh, rh0.to(rh.dtype))

        loss = forget_loss + cfg.alpha * retain_loss
        opt.zero_grad(); loss.backward(); opt.step()
        report.losses.append(float(loss.detach()))
        report.forget_losses.append(float(forget_loss.detach()))
        report.retain_losses.append(float(retain_loss.detach()))
        if step % 10 == 0:
            print(f"  step {step:3d}/{n}  loss {float(loss):.4f}  "
                  f"(forget {float(forget_loss):.4f}  retain {float(retain_loss):.6f})", flush=True)

    _fluency_check(updated, tok, retain[:3], device, report)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    updated.save_pretrained(out); tok.save_pretrained(out)

    after = _fingerprint(src)
    if after != before:
        changed = sorted(set(before) ^ set(after)) or [k for k in before if before[k] != after.get(k)]
        raise SystemExit(
            f"THE SOURCE MODEL CHANGED during training — {src}\n"
            f"  differing entries: {changed[:8]}\n"
            f"The checkpoint at {out} may be fine, but the base model is no longer trustworthy: "
            f"re-download it before running anything else against it.")
    print(f"\nwrote checkpoint -> {out}")
    print(f"source model unchanged ({len(before)} files fingerprinted)")
    return report


def _fluency_check(model, tok, texts: list[str], device, report: RMUReport,
                   new_tokens: int = 40, floor: float = 0.35) -> None:


    import torch

    if not texts:
        return
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            batch = tok([t[:200] for t in texts], return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)
            out = model.generate(**batch, max_new_tokens=new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        ratios = []
        for i in range(out.shape[0]):
            new = out[i, batch["input_ids"].shape[1]:].tolist()
            if new:
                ratios.append(len(set(new)) / len(new))
                report.fluency_samples.append(tok.decode(new, skip_special_tokens=True)[:160])
        report.fluency_unique_ratio = sum(ratios) / len(ratios) if ratios else None
    except Exception as e:
        print(f"  [fluency check skipped: {type(e).__name__}: {e}]", flush=True)
        return
    finally:
        model.train(was_training)

    r = report.fluency_unique_ratio
    print(f"\n  fluency: unique-token ratio {r:.2f} over {len(texts)} retain continuations")
    for sample in report.fluency_samples:
        print(f"    | {sample!r}")
    if r is not None and r < floor:
        print(f"  ⚠️ THE MODEL IS DEGENERATE (ratio {r:.2f} < {floor}). It will hit the token cap on "
              f"every item and score as 'no answer' everywhere — which in an accuracy table is "
              f"indistinguishable from having forgotten. RMU steers toward noise and has no fluency "
              f"term, so this is what too small an --alpha buys. Do NOT spend a sweep on it.",
              flush=True)


def _git_commit() -> str | None:


    import subprocess

    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parents[2], timeout=10)
        return r.stdout.strip()[:12] if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _fingerprint(d: Path) -> dict[str, tuple[int, int]]:


    return {f.name: (f.stat().st_size, f.stat().st_mtime_ns)
            for f in sorted(d.iterdir()) if f.is_file()}


_LAYER_PATHS = ("model.language_model.layers", "language_model.model.layers", "language_model.layers",
                "model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers")


def hidden_size_of(config) -> int:


    for obj in (config, getattr(config, "text_config", None)):
        size = getattr(obj, "hidden_size", None)
        if isinstance(size, int):
            return size
    raise SystemExit(
        f"could not read hidden_size from {type(config).__name__} (or its text_config). RMU needs "
        f"the residual-stream width to build the steering vector; add this layout to "
        f"rmu.hidden_size_of().")


def decoder_layers_path(model) -> tuple[object, str]:


    for path in _LAYER_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if hasattr(obj, "__len__") and len(obj):
            return obj, path
    raise SystemExit(
        "could not locate the decoder layer list on this architecture.\n"
        f"  tried: {', '.join(_LAYER_PATHS)}\n"
        "  Find it with `python scripts/probe_model.py <model> --tree`, then add its path to "
        "rmu._LAYER_PATHS.")


def _decoder_layers(model):

    return decoder_layers_path(model)[0]
