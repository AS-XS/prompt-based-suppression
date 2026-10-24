from __future__ import annotations

from pathlib import Path


RUNS = ("exp1", "exp2", "exp3", "wmdp1", "wmdp2", "wmdp3", "wmdp4", "wmdp5", "wmdp5.1",
        "wmdp6", "wmdp7", "wmdp8", "rwku8", "rmu1", "rmu2", "rmu3", "rmu4", "elm1", "elm2", "elm3", "r2mu-e2b", "r2mu-g31", "round1",
        "dspy-exp2", "dspy-wmdp1", "dspy-wmdp4")  

DATA_ROOT = Path("data/processed")
RESULTS_ROOT = Path("results")


UNLEARNED_ROOT = Path("models/unlearned")
UNLEARN_REPORT = "unlearn_report.json"


def unlearn_report_path(model_dir: str | Path) -> Path:

    return Path(model_dir) / UNLEARN_REPORT


def items_path(run: str) -> Path:
    return DATA_ROOT / run / "items.jsonl"


def split_path(run: str) -> Path:


    return DATA_ROOT / run / "split.json"


def generations_dir(run: str) -> Path:
    return RESULTS_ROOT / run / "generations"


def judgments_dir(run: str) -> Path:
    return RESULTS_ROOT / run / "judgments"


def primitive_judgments_dir(run: str) -> Path:
    return RESULTS_ROOT / run / "judgments_primitive"


def judge_samples_dir(run: str) -> Path:


    return RESULTS_ROOT / run / "judge_samples"


def sample_row_files(directory: Path, name: str) -> list[Path]:


    return sorted(p for p in Path(directory).glob(f"{name}*.jsonl")
                  if p.name == f"{name}.jsonl" or p.name.startswith(f"{name}__"))


def judge_sample_files(run: str, name: str) -> list[Path]:
    return sample_row_files(judge_samples_dir(run), name)


def analysis_dir(run: str) -> Path:
    return RESULTS_ROOT / run / "analysis"


def presentation_dir(run: str) -> Path:


    return RESULTS_ROOT / run / "presentation"


def optimize_namespace(round_n: int, *, prompt_model: str | None = None, demos: int = 0) -> str:


    if int(round_n) < 1:
        raise ValueError(f"round must be >= 1, got {round_n!r}")
    parts = [f"dspy-r{int(round_n)}"]
    if not prompt_model:
        parts.append("self")
    if demos:
        parts.append(f"fs{int(demos)}")
    return "-".join(parts)


def optimize_dir(run: str) -> Path:


    return RESULTS_ROOT / run / "optimize"


def stale_analysis(run: str) -> list[Path]:


    gens = list(generations_dir(run).glob("*.jsonl"))
    if not gens:
        return []
    newest = max(g.stat().st_mtime for g in gens)
    return sorted(f for f in analysis_dir(run).glob("*.csv") if f.stat().st_mtime < newest)


def generations_glob(run: str) -> str:
    return str(generations_dir(run) / "*.jsonl")


def primitive_judgments_glob(run: str) -> str:
    return str(primitive_judgments_dir(run) / "*.jsonl")


def r2mu_training_dir(name: str) -> Path:
    return RESULTS_ROOT / "r2mu-training" / name


UNLEARN_BASES = {
    'rmu2': 'gemma-4-E2B-it', 'rmu3': 'Meta-Llama-3-8B-Instruct',
    'rmu4': 'gemma-4-31B-it', 'elm1': 'Meta-Llama-3-8B-Instruct',
    'elm2': 'gemma-4-E2B-it', 'elm3': 'gemma-4-31B-it',
    'r2mu-e2b': 'gemma-4-E2B-it', 'r2mu-e4b': 'gemma-4-E4B-it',
    'r2mu-g12': 'gemma-4-12B-it', 'r2mu-g31': 'gemma-4-31B-it',
    'r2mu-o7': 'Olmo-3-7B-Think', 'r2mu-o32': 'Olmo-3.1-32B-Think',
    'rmu-e4b': 'gemma-4-E4B-it', 'rmu-g12': 'gemma-4-12B-it',
    'elm-e4b': 'gemma-4-E4B-it', 'elm-g12': 'gemma-4-12B-it',
    'elm-o7t': 'Olmo-3-7B-Think', 'elm-o32t': 'Olmo-3.1-32B-Think',
    'rmu-o7': 'Olmo-3-7B-Instruct', 'elm-o7': 'Olmo-3-7B-Instruct',
    'rmu-o32': 'Olmo-3.1-32B-Instruct', 'elm-o32': 'Olmo-3.1-32B-Instruct',
}
