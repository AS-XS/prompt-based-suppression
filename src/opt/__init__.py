from __future__ import annotations

from pathlib import Path

from src.paths import optimize_dir


def result_path(run: str, model: str, objective: str, dry_run: bool) -> Path:


    suffix = ".dryrun" if dry_run else ""
    return optimize_dir(run) / f"{model}__{objective}{suffix}.json"
