from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _imported_modules(path: Path) -> set[str]:


    import ast
    mods: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_the_unlearn_package_never_imports_a_prompt_module():


    for f in (ROOT / "src" / "unlearn").glob("*.py"):
        assert not any(m.startswith("src.prompts") or m.startswith("src.opt")
                       for m in _imported_modules(f)), f


def test_no_prompt_arm_imports_the_unlearn_package():

    for pkg in ("prompts", "gen", "opt"):
        for f in (ROOT / "src" / pkg).glob("*.py"):
            assert not any(m.startswith("src.unlearn") for m in _imported_modules(f)), f


def test_the_measurement_config_sweeps_baseline_only():


    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "rmu1.yaml").read_text())
    assert cfg["conditions"] == ["baseline"]
    assert "prompt_suppressed" not in cfg
    assert cfg["run"] == "rmu1"

    assert cfg["probe_types"] == ["forget", "neighbor", "utility"]


def test_rmu_updates_only_the_paper_s_narrow_slice():


    from src.unlearn.rmu import RMUConfig
    assert RMUConfig(layer=7).layers_to_update() == (5, 6, 7)
    assert RMUConfig(layer=1).layers_to_update() == (0, 1)
    assert RMUConfig(layer=7, update_layers=(3,)).layers_to_update() == (3,)


def test_rmu_defaults_are_the_published_ones():

    from src.unlearn.rmu import RMUConfig
    c = RMUConfig(layer=7)
    assert (c.steering_coeff, c.alpha, c.lr, c.max_batches, c.batch_size) == (6.5, 1200.0, 5e-5, 150, 4)


def test_elm_refuses_rather_than_approximating():

    from src.unlearn.elm import run_elm
    with pytest.raises(SystemExit, match="not reimplemented"):
        run_elm()


def test_the_gated_bio_corpus_is_refused_not_substituted():
    r = subprocess.run([sys.executable, "scripts/unlearn.py", "--method", "rmu",
                        "--model", "x", "--concept", "wmdp-bio", "--layer", "7", "--name", "t"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "GATED" in (r.stdout + r.stderr)


def test_rmu_requires_an_explicit_layer():


    r = subprocess.run([sys.executable, "scripts/unlearn.py", "--method", "rmu",
                        "--model", "x", "--concept", "wmdp-cyber", "--name", "t"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "--layer is required" in (r.stdout + r.stderr)


def test_the_output_path_may_not_be_inside_the_source_model():


    from src.unlearn.rmu import RMUConfig, run_rmu
    with pytest.raises(SystemExit, match="refusing to train"):
        run_rmu("models/Meta-Llama-3-8B-Instruct",
                "models/Meta-Llama-3-8B-Instruct/sub", concept="wmdp-cyber",
                forget=["a"], retain=["b"], cfg=RMUConfig(layer=7))
    with pytest.raises(SystemExit, match="refusing to train"):
        run_rmu("models/x", "models/x", concept="wmdp-cyber",
                forget=["a"], retain=["b"], cfg=RMUConfig(layer=7))


def test_a_name_with_a_path_separator_is_refused():
    r = subprocess.run([sys.executable, "scripts/unlearn.py", "--method", "rmu", "--model", "x",
                        "--concept", "wmdp-cyber", "--layer", "7", "--name", "../Meta-Llama-3-8B"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "plain directory name" in (r.stdout + r.stderr)


def test_the_source_fingerprint_is_stat_only():

    import inspect
    from src.unlearn.rmu import _fingerprint
    src = inspect.getsource(_fingerprint)
    assert "read_bytes" not in src and "md5" not in src and "sha" not in src
    assert "st_size" in src and "st_mtime_ns" in src


def test_fetch_corpora_needs_no_name_or_model():


    src = (ROOT / "scripts" / "unlearn.py").read_text()
    i_fetch = src.index("if args.fetch_corpora:")
    assert i_fetch < src.index('if "/" in args.name')
    assert i_fetch < src.index("--layer is required")


def test_every_dataset_id_carries_a_namespace():


    from src.unlearn.corpora import _FORGET, _RETAIN
    ids = [name for name, _ in _FORGET.values()] + [_RETAIN[0]]
    for i in ids:
        assert "/" in i, f"{i!r} has no namespace — current huggingface_hub refuses it"


def test_rmu_does_not_use_device_map():


    import ast
    src = (ROOT / "src" / "unlearn" / "rmu.py").read_text()
    kwargs = {kw.arg for node in ast.walk(ast.parse(src))
              if isinstance(node, ast.Call) for kw in node.keywords if kw.arg}
    assert "device_map" not in kwargs
    assert "torch_dtype" not in kwargs          
    assert "dtype" in kwargs and ".to(device)" in src


TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


def _fake_pair(tmp_path, *, base="Meta-Llama-3-8B-Instruct", name="rmu-llama-cyber",
               ckpt_template=TEMPLATE, base_template=TEMPLATE, report=True):

    import json
    src = tmp_path / "models" / base
    src.mkdir(parents=True)
    (src / "tokenizer_config.json").write_text(json.dumps({"chat_template": base_template}))
    d = tmp_path / "unlearned" / name
    d.mkdir(parents=True)
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": ckpt_template}))
    if report:
        (d / "unlearn_report.json").write_text(
            json.dumps({"method": "rmu", "base_model": base, "model": str(src)}))
    return str(tmp_path / "unlearned"), name


def _states(root, name):
    from src.gen import thinkstate
    thinkstate._inherited.cache_clear()
    return thinkstate._inherited(name, root)


def test_an_unlearned_checkpoint_inherits_its_base_models_think_states(tmp_path):


    root, name = _fake_pair(tmp_path)
    base, entry = _states(root, name)
    assert base == "Meta-Llama-3-8B-Instruct"
    assert entry["states"] == ["na"]


def test_inheritance_is_verified_against_the_chat_template_not_assumed(tmp_path):


    root, name = _fake_pair(tmp_path, ckpt_template="{{ '<think>' }}" + TEMPLATE)
    with pytest.raises(SystemExit, match="chat templates DIFFER"):
        _states(root, name)


def test_a_checkpoint_without_provenance_still_raises(tmp_path):


    root, name = _fake_pair(tmp_path, report=False)
    assert _states(root, name) is None


def test_a_checkpoint_naming_an_unregistered_base_still_raises(tmp_path):

    root, name = _fake_pair(tmp_path, base="Some-Unaudited-7B")
    assert _states(root, name) is None


def test_think_states_for_reads_the_checkpoint_through_MODELS_DIR(tmp_path, monkeypatch):

    from src.gen import thinkstate
    root, name = _fake_pair(tmp_path)
    monkeypatch.setenv("MODELS_DIR", root)
    thinkstate._inherited.cache_clear()
    assert thinkstate.think_states_for(name) == ["na"]
    assert thinkstate.chat_kwargs(name, "na") == {}
    assert thinkstate.inherited_from(name) == "Meta-Llama-3-8B-Instruct"


def test_no_unlearned_checkpoint_is_hand_registered():

    from src.gen.thinkstate import registry
    for k in registry():
        assert not k.startswith(("rmu-", "elm-")), f"{k} belongs in unlearn_report.json, not here"


def test_both_writers_record_the_base_model_the_sweep_reads():


    src = (ROOT / "scripts" / "unlearn.py").read_text()
    assert src.count("_write_report(") == 3          
    assert "unlearn_report.json" not in src.split("def _write_report")[0], \
        "the filename belongs to src/paths.py — spelling it twice is how the two copies drift"


def test_adopting_a_checkpoint_never_writes_into_it(tmp_path):


    from src.unlearn.elm import register_checkpoint
    src = tmp_path / "external"
    src.mkdir()
    (src / "config.json").write_text("{}")
    dst = register_checkpoint(src, "elm-llama-bio", tmp_path / "unlearned")
    assert dst.is_dir() and not dst.is_symlink()
    (dst / "unlearn_report.json").write_text("{}")
    assert not (src / "unlearn_report.json").exists()


def test_the_template_check_tolerates_serialisation_whitespace(tmp_path):


    root, name = _fake_pair(tmp_path, ckpt_template="\n" + TEMPLATE + "\n\n")
    assert _states(root, name)[1]["states"] == ["na"]


def test_a_jinja_sidecar_is_read_the_same_as_an_inline_template(tmp_path):

    root, name = _fake_pair(tmp_path)
    d = tmp_path / "unlearned" / name
    (d / "tokenizer_config.json").unlink()
    (d / "chat_template.jinja").write_text(TEMPLATE)
    assert _states(root, name)[1]["states"] == ["na"]


def test_a_checkpoint_is_found_without_MODELS_DIR_being_set(tmp_path, monkeypatch):


    from src.gen import thinkstate
    root, name = _fake_pair(tmp_path)
    monkeypatch.delenv("MODELS_DIR", raising=False)
    monkeypatch.setattr(thinkstate, "UNLEARNED_ROOT", Path(root))
    thinkstate._inherited.cache_clear()
    assert thinkstate.think_states_for(name) == ["na"]


def _rows(model, correct_by_item, probe="forget"):
    return [{"model": model, "think_state": "na", "mmlu_subject": "s", "probe_type": probe,
             "prompt_id": "none-baseline", "concept_id": "c", "item_id": i,
             "parsed": True, "correct": c} for i, c in correct_by_item.items()]


def test_a_model_contrast_pairs_edited_against_unedited_on_the_same_item():

    from src.analysis.accuracy import delta_table
    base = _rows("Meta-Llama-3-8B-Instruct", {"a": True, "b": True, "c": True, "d": False})
    edit = _rows("rmu-llama-cyber", {"a": False, "b": False, "c": True, "d": False})
    rows, levels = delta_table(base + edit, by=("think_state", "mmlu_subject"),
                               baseline_prompt="Meta-Llama-3-8B-Instruct", contrast="model")
    assert levels == ["rmu-llama-cyber"]
    r = rows[0]
    assert r["baseline_acc"] == 0.75 and r["rmu-llama-cyber_acc"] == 0.25
    assert r["rmu-llama-cyber_drop"] == 0.5 and r["rmu-llama-cyber_n"] == 4


def test_the_contrast_field_may_not_also_be_a_grouping_key():


    from src.analysis.accuracy import delta_table
    with pytest.raises(ValueError, match="also a grouping key"):
        delta_table(_rows("m", {"a": True}), by=("model", "think_state"), contrast="model")


def test_the_prompt_contrast_is_unchanged_by_the_generalisation():

    from src.analysis.accuracy import delta_table
    rows = _rows("m", {"a": True, "b": True})
    supp = [{**r, "prompt_id": "scoped-sandbag", "correct": False} for r in rows]
    out, levels = delta_table(rows + supp, by=("model", "think_state", "mmlu_subject"))
    assert levels == ["scoped-sandbag"]
    assert out[0]["scoped-sandbag_drop"] == 1.0


def test_the_edited_row_renders_through_the_existing_table(tmp_path):


    from src.analysis.accuracy import delta_table
    from src.analysis.present import model_table
    base = _rows("Meta-Llama-3-8B-Instruct", {"a": True, "b": True}, probe="forget")
    edit = _rows("rmu-llama-cyber", {"a": False, "b": True}, probe="forget")
    rows, edited = delta_table(base + edit, by=("think_state", "probe_type"),
                               baseline_prompt="Meta-Llama-3-8B-Instruct", contrast="model")
    stamped = [{**r, "model": m} for r in rows for m in edited]
    header, body = model_table(stamped, "rmu-llama-cyber")
    assert body, "the checkpoint must render as a row, not an empty table"
    assert any("50" in c or "0.5" in c for r in body for c in r)


def test_scoring_refuses_a_baseline_model_that_was_never_swept_here():


    src = (ROOT / "scripts" / "score_accuracy.py").read_text()
    assert "has no rows in this run" in src
    assert "cannot be paired" in src


def test_the_checked_parameters_come_from_the_checkpoints_own_claim():


    from scripts.diff_checkpoint import updated_param_names
    assert updated_param_names({"config": {"layers_to_update": [5, 6, 7]}}) == [
        f"model.layers.{i}.mlp.down_proj.weight" for i in (5, 6, 7)]

    assert updated_param_names({"config": {"layer": 7}})[0] == "model.layers.5.mlp.down_proj.weight"
    assert len(updated_param_names({"config": {"layer": 1}})) == 2       
    with pytest.raises(SystemExit, match="no claim to check"):
        updated_param_names({"config": {}})


def _safetensors(path: Path, tensors: dict[str, bytes]) -> None:


    import json as _json
    import struct as _struct
    header, off = {}, 0
    for name, raw in tensors.items():
        header[name] = {"dtype": "F32", "shape": [len(raw) // 4], "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    blob = _json.dumps(header).encode()
    path.write_bytes(_struct.pack("<Q", len(blob)) + blob + b"".join(tensors.values()))


def test_a_single_file_checkpoint_is_read_like_a_sharded_one(tmp_path):

    from scripts.diff_checkpoint import shard_index
    d = tmp_path / "m"
    d.mkdir()
    (d / "model.safetensors.index.json").write_text(
        '{"weight_map": {"model.layers.5.mlp.down_proj.weight": "model-00001-of-00002.safetensors"}}')
    assert shard_index(d)["model.layers.5.mlp.down_proj.weight"].endswith(".safetensors")

    solo = tmp_path / "solo"
    solo.mkdir()
    _safetensors(solo / "model.safetensors", {"w": b"\x00" * 8})
    assert shard_index(solo) == {"w": "model.safetensors"}

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no model.safetensors"):
        shard_index(empty)


def test_the_container_is_parsed_without_mmapping_the_whole_shard(tmp_path):


    from scripts.diff_checkpoint import locate, read_header, shard_index
    d = tmp_path / "m"
    d.mkdir()
    _safetensors(d / "model.safetensors", {"a": b"\x01" * 8, "b": b"\x02" * 4})
    header, data_start = read_header(d / "model.safetensors")
    assert set(header) == {"a", "b"}
    path, dtype, shape, start, end = locate(d, shard_index(d), "b")
    assert (dtype, shape) == ("F32", [1])
    assert end - start == 4 and start == data_start + 8
    assert path.read_bytes()[start:end] == b"\x02" * 4


def test_equality_is_streamed_so_a_1gb_embedding_is_not_materialised(tmp_path):

    from scripts.diff_checkpoint import identical, locate, shard_index
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
    _safetensors(a / "model.safetensors", {"w": b"\x01" * 64})
    _safetensors(b / "model.safetensors", {"w": b"\x01" * 64})
    ia, ib = shard_index(a), shard_index(b)
    assert identical(locate(a, ia, "w"), locate(b, ib, "w"))
    _safetensors(b / "model.safetensors", {"w": b"\x01" * 63 + b"\x02"})
    assert not identical(locate(a, ia, "w"), locate(b, shard_index(b), "w"))


def test_the_report_records_both_loss_terms_not_just_the_total():


    from src.unlearn.rmu import RMUReport
    r = RMUReport(config={}, model="m", concept="c", n_forget=1, n_retain=1)
    assert r.forget_losses == [] and r.retain_losses == []
    src = (ROOT / "src" / "unlearn" / "rmu.py").read_text()
    assert "report.forget_losses.append" in src and "report.retain_losses.append" in src


def test_the_steered_activation_is_the_layers_OUTPUT_not_its_input():


    from src.unlearn.rmu import activation_index
    assert activation_index(7, 33) == 8
    assert activation_index(0, 33) == 1
    assert activation_index(31, 33) == 32
    with pytest.raises(SystemExit, match="architecture-specific"):
        activation_index(32, 33)


def test_every_updated_layer_is_upstream_of_the_read_point():


    from src.unlearn.rmu import RMUConfig, activation_index
    cfg = RMUConfig(layer=7)
    read_at = activation_index(cfg.layer, 33)
    for i in cfg.layers_to_update():
        assert activation_index(i, 33) <= read_at, f"layers[{i}] runs after the loss is taken"


def test_the_report_records_the_activation_scale_c_must_be_chosen_against():


    from src.unlearn.rmu import RMUReport
    r = RMUReport(config={}, model="m", concept="c", n_forget=1, n_retain=1)
    assert r.forget_activation_norm is None and r.forget_activation_rms is None
    src = (ROOT / "src" / "unlearn" / "rmu.py").read_text()
    assert "report.forget_activation_norm = float(" in src
    assert "report.forget_activation_rms = float(" in src, \
        "the loss is an MSE, so it responds to the mean of SQUARES — and LLM activations have a "\
        "heavy tail that pulls mean and RMS far apart (5.17 vs 17.6 on Llama-3-8B at layer 7). "\
        "Recording only the mean is recording the statistic that does NOT govern the objective."


def test_a_model_contrast_writes_one_row_per_group_not_per_checkpoint():


    from scripts.score_accuracy import _stamp_reference
    groups = [{"think_state": "na", "mmlu_subject": "wmdp-cyber", "baseline_acc": 0.4758,
               "rmu-llama-cyber_drop": -0.0027, "rmu-llama-cyber-v2_drop": -0.0041}]
    out = _stamp_reference(groups, "Meta-Llama-3-8B-Instruct")
    assert len(out) == len(groups)
    assert {r["model"] for r in out} == {"Meta-Llama-3-8B-Instruct"}, \
        "the row is ABOUT the unedited reference; the edits are its columns"
    assert out[0]["rmu-llama-cyber-v2_drop"] == -0.0041


def test_the_checkpoint_records_the_commit_that_trained_it():


    from src.unlearn.rmu import RMUReport, _git_commit
    assert RMUReport(config={}, model="m", concept="c", n_forget=1, n_retain=1).git_commit is None
    got = _git_commit()
    assert got is None or (len(got) == 12 and all(ch in "0123456789abcdef" for ch in got))


def test_the_diff_prints_the_hyperparameters_it_was_trained_with():


    src = (ROOT / "scripts" / "diff_checkpoint.py").read_text()
    for field in ("steering_coeff", "alpha", "layers_to_update", "git_commit"):
        assert f"'{field}'" in src or f'"{field}"' in src, f"{field} is not shown beside the deltas"
    assert "UNRECORDED" in src, "a checkpoint predating the field must say so, not show a blank"


def test_a_collapsed_model_is_caught_before_a_sweep_is_spent_on_it():


    import inspect
    from src.unlearn.rmu import RMUReport, _fluency_check
    r = RMUReport(config={}, model="m", concept="c", n_forget=1, n_retain=1)
    assert r.fluency_unique_ratio is None and r.fluency_samples == []
    src = inspect.getsource(_fluency_check)
    assert "do_sample=False" in src, "the survival check must be deterministic"
    assert "DEGENERATE" in src and "Do NOT spend a sweep" in src
    assert "except Exception" in src, "a diagnostic must never lose a trained checkpoint"

    whole = (ROOT / "src" / "unlearn" / "rmu.py").read_text()
    assert whole.index("_fluency_check(updated") < whole.index("updated.save_pretrained(out)")


def test_fluency_can_be_checked_without_retraining():


    src = (ROOT / "scripts" / "check_fluency.py").read_text()
    assert "from src.unlearn.rmu import RMUReport, _fluency_check" in src, \
        "it must reuse the training-time check, not reimplement the threshold"
    assert "--write" in src, "a checkpoint should end up carrying its own answer"
    assert "return 1 if r is not None and r < 0.35 else 0" in src, "collapse must be a nonzero exit"


def test_concurrent_resubmissions_do_not_multiply_a_models_rows():


    from scripts.score_accuracy import _one_row_per_cell
    row = {"model": "m", "prompt_id": "none-baseline", "think_state": "na",
           "condition": "baseline", "item_id": "i1", "raw_output": "A"}
    other = {**row, "raw_output": "B"}                 
    assert len(_one_row_per_cell([row, other, row])) == 1
    assert _one_row_per_cell([row, other])[0]["raw_output"] == "A", "first kept, deterministically"

    keep = [row, {**row, "item_id": "i2"}, {**row, "model": "m2"}, {**row, "think_state": "on"}]
    assert len(_one_row_per_cell(keep)) == 4
