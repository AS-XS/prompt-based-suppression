import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recipe", ROOT / "scripts/run_experiment.py")
recipe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe_module)


def test_recipe_controls_and_configs():
    import yaml
    from src.prompts import build_messages, is_directed_prompt, split_prompt_id, takes_examples
    recipes = json.loads((ROOT / "experiments.json").read_text())
    for name, recipe in recipes.items():
        config = yaml.safe_load((ROOT / recipe["config"]).read_text())
        assert config["run"] == name
        assert recipe["waves"][0] == "none-baseline"
        assert len(recipe["waves"]) == len(set(recipe["waves"]))
        for wave in recipe["waves"]:
            template = split_prompt_id(wave)[0]
            examples = [("Question?", "Answer: A")] if is_directed_prompt(template) and takes_examples(template) else None
            assert build_messages(wave, "wmdp-bio" if recipe["split"] else "biology", "Question?", examples)


def test_generate_and_coverage_use_the_same_grid_without_a_scheduler():
    recipes = json.loads((ROOT / "experiments.json").read_text())
    for name, recipe in recipes.items():
        for model in recipe["models"]:
            gen = recipe_module.commands_for(name, recipe, "generate", [model], "models")
            coverage = recipe_module.commands_for(name, recipe, "coverage", [model], "models")[0]
            waves = coverage[coverage.index("--waves") + 1:]
            assert len(gen) == len(waves)
            assert all("qsub" not in " ".join(cmd) for cmd in gen)
            assert gen[0][gen[0].index("--conditions") + 1] == "baseline"
            assert [cmd[-1] for cmd in gen[1:]] == waves[1:]


def test_ladder_keeps_both_label_controls():
    recipes = json.loads((ROOT / "experiments.json").read_text())
    assert recipes["wmdp5"]["waves"] == recipes["wmdp5.1"]["waves"]
    assert len(recipes["wmdp5"]["waves"]) == 13
    assert {"scoped-sandbag", "scoped-sandbag@specific"} <= set(recipes["wmdp5"]["waves"])
