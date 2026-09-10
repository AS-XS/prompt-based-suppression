"""Print or execute scheduler-independent experiment commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commands_for(name, recipe, stage, models, models_dir, tp=1, gpu_util=0.85):
    py = sys.executable
    if stage == "build":
        commands = [[py, "-m", "src.data.build", "--run", name, *recipe["build"]]]
        if recipe["split"]:
            commands.append([py, "scripts/build_split.py", "--run", name, "--seed", "0", "--val-frac", "0.25"])
        return commands
    if stage == "generate":
        commands = []
        for model in models:
            for wave in recipe.get("waves_by_model", {}).get(model, recipe["waves"]):
                baseline = wave == "none-baseline"
                commands.append([py, "scripts/generate.py", "--config", recipe["config"],
                                 "--model", model, "--models-dir", models_dir,
                                 "--tp", str(tp), "--gpu-util", str(gpu_util),
                                 "--conditions", "baseline" if baseline else "suppressed",
                                 "--prompt-suppressed", "world-frame" if baseline else wave])
        return commands
    if stage == "coverage":
        commands = []
        for model in models:
            waves = recipe.get("waves_by_model", {}).get(model, recipe["waves"])
            commands.append([py, "scripts/coverage.py", "--run", name, "--config", recipe["config"],
                             "--models", model, "--waves", *waves])
        return commands
    if stage == "score":
        splits = ["val", "test"] if recipe["split"] else [None]
        commands = [[py, "scripts/score_accuracy.py", "--run", name,
                     *(["--split", split] if split else [])] for split in splits]
        if not recipe["split"]:
            commands.append([py, "scripts/score_recall.py", "--run", name])
        return commands
    raise ValueError(stage)


def main():
    recipes = json.loads((ROOT / "experiments.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(recipes))
    parser.add_argument("--stage", choices=["build", "generate", "coverage", "score"], required=True)
    parser.add_argument("--model", action="append", help="Repeat to select several registered models.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    recipe = recipes[args.experiment]
    models = args.model or recipe["models"]
    if args.tp < 1 or not 0 < args.gpu_util <= 1:
        parser.error("--tp must be positive and --gpu-util must be in (0, 1]")
    if set(models) - set(recipe["models"]):
        parser.error("Unknown model for this recipe; use generate.py directly for a new experiment cell.")
    for cmd in commands_for(args.experiment, recipe, args.stage, models, args.models_dir, args.tp, args.gpu_util):
        print(shlex.join(cmd), flush=True)
        if args.execute:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
