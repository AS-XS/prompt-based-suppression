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
    assert "DEGENERATE" in src and "Do NOT sweep" in src


    assert "too small an --alpha buys" not in src
    assert "BASE model" in src, "a failing gate on a new architecture needs the base-model control"
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


def test_the_copied_baseline_is_only_valid_while_the_sampling_matches():


    import inspect
    from src.gen.sweep import run_sweep
    src = inspect.getsource(run_sweep)
    i = src.index("cfg = {")
    payload = src[i:src.index("config_hash(cfg)")]
    assert '"model"' in payload and '"sampling"' in payload and '"prompt_fingerprint"' in payload
    assert '"run"' not in payload, \
        "if the run name ever enters config_hash, rmu2's copied baseline becomes a different cell"


def test_the_gemma_E_series_layer_path_is_reachable():


    from src.unlearn.rmu import _LAYER_PATHS
    assert "model.language_model.layers" in _LAYER_PATHS
    assert _LAYER_PATHS.index("model.language_model.layers") < _LAYER_PATHS.index("model.layers"), \
        "a multimodal wrapper must not match a shorter path by accident"
    probe = (ROOT / "scripts" / "probe_model.py").read_text()
    assert "--tree" in probe, "there must be a way to find a path that is not in the list yet"


def test_the_probe_reports_RMS_and_a_c_suggestion_not_just_the_mean():


    probe = (ROOT / "scripts" / "probe_model.py").read_text()
    assert "RMS ||h||" in probe and "1.1 * rms" in probe
    assert "down_proj" in probe, "RMU is defined as editing that matrix; absence is disqualifying"
    assert "torch.cuda.is_available()" in probe, "check for a GPU before loading the model"


def test_hidden_size_is_read_from_nested_configs_too():


    import types
    from src.unlearn.rmu import hidden_size_of
    assert hidden_size_of(types.SimpleNamespace(hidden_size=4096)) == 4096
    nested = types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=1536))
    assert hidden_size_of(nested) == 1536
    with pytest.raises(SystemExit, match="could not read hidden_size"):
        hidden_size_of(types.SimpleNamespace())


def test_the_nested_config_rule_has_exactly_one_copy():


    probe = (ROOT / "scripts" / "probe_model.py").read_text()
    assert "hidden_size_of(cfg)" in probe
    assert "text_config" not in probe, "the rule belongs in rmu.hidden_size_of(), not in both files"


def test_the_fluency_check_pads_on_the_left_and_restores_the_tokenizer():


    import inspect
    from src.unlearn.rmu import _fluency_check
    src = inspect.getsource(_fluency_check)
    assert 'tok.padding_side = "left"' in src
    i_set, i_restore = src.index('tok.padding_side = "left"'), src.index("tok.padding_side = old_side")
    assert i_set < i_restore and "finally:" in src[i_set:i_restore], \
        "the restore must be in `finally`, or an exception saves a left-padded tokenizer"


def test_a_base_model_can_be_the_fluency_control():


    src = (ROOT / "scripts" / "check_fluency.py").read_text()
    assert "a base model; ratio reported, nothing written" in src
    assert 'raise SystemExit(f"no {p} to write into")' not in src


def test_the_diff_finds_nested_parameter_names_instead_of_assuming_llamas():


    from scripts.diff_checkpoint import updated_param_names
    report = {"config": {"layers_to_update": [6, 7, 8]}}
    gemma = {f"model.language_model.layers.{i}.mlp.down_proj.weight": "x.safetensors"
             for i in range(35)}
    assert updated_param_names(report, gemma) == [
        f"model.language_model.layers.{i}.mlp.down_proj.weight" for i in (6, 7, 8)]
    llama = {f"model.layers.{i}.mlp.down_proj.weight": "x" for i in range(32)}
    assert updated_param_names(report, llama)[0] == "model.layers.6.mlp.down_proj.weight"

    both = {**gemma, "model.vision_tower.layers.6.mlp.down_proj.weight": "v"}
    assert updated_param_names({"config": {"layers_to_update": [6]}}, both) == [
        "model.language_model.layers.6.mlp.down_proj.weight"]
    with pytest.raises(SystemExit, match="expected exactly one"):
        updated_param_names({"config": {"layers_to_update": [99]}}, gemma)


def test_training_never_sends_a_checkpoint_straight_to_a_sweep():


    import ast
    tree = ast.parse((ROOT / "scripts" / "unlearn.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_print_next")
    printed = " ".join(c.value for n in ast.walk(fn) for c in [n]
                       if isinstance(c, ast.Constant) and isinstance(c.value, str))
    assert "diff_checkpoint.py" in printed and "forget_losses" in printed, "the gates come first"
    assert "configs/rmu1.yaml" not in printed, "which run a checkpoint belongs in is not fixed"


def test_an_rmu_run_sets_no_decode_field_of_its_own():


    import yaml
    for run in ("rmu2", "rmu3"):
        s = yaml.safe_load((ROOT / "configs" / f"{run}.yaml").read_text())["sampling"]
        assert set(s) == {"max_tokens"}, f"{run} sets decode fields: {sorted(set(s) - {'max_tokens'})}"


def _bases(monkeypatch, mapping):

    import src.gen.thinkstate as ts
    monkeypatch.setattr(ts, "inherited_from", lambda m: mapping.get(m))


def test_a_contrast_of_one_family_is_allowed(monkeypatch):
    from src.analysis.checkpoints import refuse_cross_family
    _bases(monkeypatch, {"rmu-llama-cyber-v2": "Meta-Llama-3-8B-Instruct",
                         "rmu-llama-cyber-c20a300": "Meta-Llama-3-8B-Instruct"})
    refuse_cross_family(["Meta-Llama-3-8B-Instruct", "rmu-llama-cyber-v2",
                         "rmu-llama-cyber-c20a300"], "Meta-Llama-3-8B-Instruct")


def test_a_checkpoint_of_another_model_is_refused(monkeypatch):

    from src.analysis.checkpoints import refuse_cross_family
    _bases(monkeypatch, {"rmu-gemma-cyber": "gemma-4-E2B-it"})
    with pytest.raises(SystemExit, match="edited from 'gemma-4-E2B-it'"):
        refuse_cross_family(["Meta-Llama-3-8B-Instruct", "rmu-gemma-cyber"],
                            "Meta-Llama-3-8B-Instruct")


def test_a_second_base_model_in_the_run_is_refused(monkeypatch):

    from src.analysis.checkpoints import refuse_cross_family
    _bases(monkeypatch, {})
    with pytest.raises(SystemExit, match="not an edited checkpoint"):
        refuse_cross_family(["Meta-Llama-3-8B-Instruct", "gemma-4-E2B-it"],
                            "Meta-Llama-3-8B-Instruct")


def test_both_model_contrast_scorers_run_the_guard():

    for script in ("score_accuracy.py", "present_tables.py"):
        src = (ROOT / "scripts" / script).read_text()
        assert "refuse_cross_family" in src, f"{script} pairs models without checking their base"


def test_padding_after_a_stop_is_not_counted_as_output():


    import types
    from src.unlearn.rmu import _until_stop
    tok = types.SimpleNamespace(pad_token_id=0, eos_token_id=1)
    model = types.SimpleNamespace(generation_config=types.SimpleNamespace(eos_token_id=[1, 106]))
    assert _until_stop([5, 6, 1, 0, 0, 0], model, tok) == [5, 6]
    assert _until_stop([5, 106, 0, 0], model, tok) == [5]
    assert _until_stop([5, 6, 7], model, tok) == [5, 6, 7]
    nocfg = types.SimpleNamespace()
    assert _until_stop([5, 0, 0], nocfg, tok) == [5], "a model with no generation_config still stops"


def test_a_continuation_that_stopped_is_shown_but_not_scored():
    import inspect
    from src.unlearn.rmu import _MIN_SCORED, _fluency_check
    src = inspect.getsource(_fluency_check)
    assert _MIN_SCORED >= 5
    assert "if len(new) >= _MIN_SCORED" in src
    assert "every continuation stopped early" in src, "an all-stopped run must not crash the print"


def test_smoke_parse_reads_the_answer_surface_like_the_scorers_do():


    from scripts.smoke_parse import tally
    rows = [

        {"model": "m", "think_state": "on", "probe_type": "utility",
         "raw_output": "<think>maybe A or B</think>Answer: C", "finish_reason": "stop"},

        {"model": "m", "think_state": "on", "probe_type": "utility",
         "raw_output": "<think>option A seems plausible and then", "finish_reason": "length"},

        {"model": "m", "think_state": "off", "probe_type": "forget",
         "raw_output": "Answer: B\nWhy: because", "finish_reason": "stop"},

        {"model": "m", "think_state": "off", "probe_type": "forget",
         "raw_output": "@- p @- p @- p @- p", "finish_reason": "length"},
    ]
    agg = tally(rows)
    assert agg[("m", "on", "utility", "all")] == [2, 1, 1]
    assert agg[("m", "off", "forget", "?")] == [2, 1, 1]


def test_smoke_parse_splits_the_forget_tier_into_target_and_control():


    from scripts.smoke_parse import tally
    row = {"model": "m", "think_state": "off", "finish_reason": "stop"}
    rows = [{**row, "probe_type": "forget", "concept_id": "wmdp-cyber", "raw_output": "@- p @- p"},
            {**row, "probe_type": "forget", "concept_id": "wmdp-bio", "raw_output": "Answer: A"},
            {**row, "probe_type": "utility", "concept_id": "wmdp-bio", "raw_output": "Answer: B"},
            {**row, "probe_type": "utility", "concept_id": "wmdp-cyber", "raw_output": "Answer: C"}]
    agg = tally(rows)
    assert agg[("m", "off", "forget", "wmdp-cyber")] == [1, 0, 0]
    assert agg[("m", "off", "forget", "wmdp-bio")] == [1, 1, 0]
    assert agg[("m", "off", "utility", "all")] == [2, 2, 0], "utility is the health check: pooled"


def test_a_missing_model_directory_is_named_not_handed_to_vllm():


    import yaml
    cfg = ROOT / "configs" / "rmu2.yaml"
    assert yaml.safe_load(cfg.read_text())["run"] == "rmu2"
    r = subprocess.run([sys.executable, "scripts/generate.py", "--config", str(cfg),
                        "--model", "no-such-checkpoint", "--models-dir", "models/unlearned",
                        "--items", str(cfg)],          
                       capture_output=True, text=True, cwd=ROOT)
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert "no model directory at models/unlearned/no-such-checkpoint" in out, out[-600:]
    assert "LocalEntryNotFoundError" not in out


def test_a_checkpoint_gets_the_base_serving_files_it_lacks(tmp_path):
    from src.unlearn.rmu import complete_checkpoint
    src, dst = tmp_path / "base", tmp_path / "ckpt"
    src.mkdir(); dst.mkdir()
    for name in ("preprocessor_config.json", "processor_config.json", "config.json",
                 "model-00001-of-00002.safetensors", "model.safetensors.index.json", "pytorch_model.bin"):
        (src / name).write_text(f"base:{name}")
    (src / "subdir").mkdir()                                          
    (dst / "config.json").write_text("edited")                        
    copied = complete_checkpoint(src, dst)
    assert copied == ["preprocessor_config.json", "processor_config.json"]
    assert (dst / "config.json").read_text() == "edited", "never overwrite the edited model's own files"
    for w in ("model-00001-of-00002.safetensors", "model.safetensors.index.json", "pytorch_model.bin"):
        assert not (dst / w).exists(), f"{w} is the edit's territory — a base weight must never land here"
    assert complete_checkpoint(src, dst) == [], "a second call finds nothing left to copy"


def test_run_rmu_completes_the_checkpoint_before_the_source_is_rechecked():


    src = (ROOT / "src" / "unlearn" / "rmu.py").read_text()
    i_save = src.index("updated.save_pretrained(out)")
    i_complete = src.index("complete_checkpoint(src, out)")
    i_after = src.index("after = _fingerprint(src)")
    assert i_save < i_complete < i_after


def test_the_rmu_figure_draws_strict_accuracy_and_never_claims_forgetting():


    from scripts.plot_rmu import _strict
    rows = [{"model": "m", "mmlu_subject": "wmdp-cyber", "probe_type": "forget",
             "n": "100", "n_parsed": "10", "n_correct": "4"},
            {"model": "m", "mmlu_subject": "philosophy", "probe_type": "utility",
             "n": "30", "n_parsed": "30", "n_correct": "24"}]
    acc, n, p, c = _strict(rows, "m", subjects={"wmdp-cyber"})
    assert (round(acc, 6), n, p, c) == (4.0, 100, 10, 4), "strict = correct / ALL items, not / parsed"
    assert _strict(rows, "m", probe="utility")[0] == 80.0
    src = (ROOT / "scripts" / "plot_rmu.py").read_text()
    assert "trained to remove" in src and "forgotten," not in src


def _st(path, tensors: dict):

    import json as _json
    import struct as _struct
    header, off, blobs = {}, 0, []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw); blobs.append(raw)
    blob = _json.dumps(header).encode()
    path.write_bytes(_struct.pack("<Q", len(blob)) + blob + b"".join(blobs))


def _gemma_like(tmp_path):

    import json as _json
    base, ckpt = tmp_path / "base", tmp_path / "ckpt"
    base.mkdir(); ckpt.mkdir()
    edited = "model.language_model.layers.8.mlp.down_proj.weight"
    shared = {f"model.language_model.layers.{i}.self_attn.{p}.weight": ("BF16", [3], bytes([i, len(p), 7, 1, 2, 3]))   
              for i in (15, 16) for p in ("k_proj", "v_proj", "k_norm")}
    norm32 = {"model.language_model.norm.weight": ("F32", [2], b"\x01\x02\x03\x04\x05\x06\x07\x08")}
    _st(base / "model-00001-of-00002.safetensors",
        {edited: ("BF16", [2], b"\xaa\xbb\xcc\xdd"), **norm32})
    _st(base / "model-00002-of-00002.safetensors", shared)
    wm = {k: "model-00001-of-00002.safetensors" for k in (edited, *norm32)}
    wm.update({k: "model-00002-of-00002.safetensors" for k in shared})
    (base / "model.safetensors.index.json").write_text(_json.dumps({"weight_map": wm}))

    _st(ckpt / "model.safetensors", {edited: ("BF16", [2], b"\x11\x22\x33\x44"), **norm32})
    return base, ckpt, edited, shared


def test_missing_base_tensors_are_carried_over_byte_for_byte(tmp_path):
    from src.unlearn.tensors import CARRYOVER, carry_over_missing, missing_tensors, read_header
    base, ckpt, edited, shared = _gemma_like(tmp_path)
    assert missing_tensors(base, ckpt) == sorted(shared)
    assert carry_over_missing(base, ckpt) == sorted(shared)

    header, start = read_header(ckpt / CARRYOVER)
    data = (ckpt / CARRYOVER).read_bytes()[start:]
    for name, (_, _, raw) in shared.items():
        a, b = header[name]["data_offsets"]
        assert data[a:b] == raw, f"{name} is not a byte-exact copy of the base"
    assert missing_tensors(base, ckpt) == [], "every base tensor is now reachable through the index"


def test_the_carryover_shard_obeys_the_container_rules(tmp_path):


    from src.unlearn.tensors import _ITEMSIZE, CARRYOVER, carry_over_missing, read_header
    base, ckpt, _, _ = _gemma_like(tmp_path)

    import json as _json
    extra = {"model.language_model.layers.20.self_attn.k_norm.weight": ("F32", [1], b"\x09\x0a\x0b\x0c")}
    _st(base / "model-00003-of-00003.safetensors", extra)
    idx = _json.loads((base / "model.safetensors.index.json").read_text())
    idx["weight_map"].update({k: "model-00003-of-00003.safetensors" for k in extra})
    (base / "model.safetensors.index.json").write_text(_json.dumps(idx))
    carry_over_missing(base, ckpt)

    header, start = read_header(ckpt / CARRYOVER)
    assert start % 8 == 0, "the data section must start 8-byte aligned"
    spans = sorted((v["data_offsets"], k) for k, v in header.items() if k != "__metadata__")
    assert spans[0][0][0] == 0
    for (a, _), (b, _) in zip(spans, spans[1:]):
        assert a[1] == b[0], "no holes between tensors"
    total = len((ckpt / CARRYOVER).read_bytes()) - start
    assert spans[-1][0][1] == total, "the data section is fully indexed"
    for (a, _), name in spans:
        assert a % _ITEMSIZE[header[name]["dtype"]] == 0, f"{name} is misaligned"


def test_the_edit_is_never_overwritten_and_a_second_call_adds_nothing(tmp_path):


    from src.unlearn.tensors import carry_over_missing, read_header, shard_index
    base, ckpt, edited, _ = _gemma_like(tmp_path)
    carry_over_missing(base, ckpt)
    wm = shard_index(ckpt)
    assert wm[edited] == "model.safetensors"
    header, start = read_header(ckpt / "model.safetensors")
    a, b = header[edited]["data_offsets"]
    assert (ckpt / "model.safetensors").read_bytes()[start + a:start + b] == b"\x11\x22\x33\x44"
    assert carry_over_missing(base, ckpt) == [], "idempotent"


def test_the_repair_has_a_read_only_mode(tmp_path):


    from src.unlearn.rmu import complete_checkpoint
    from src.unlearn.tensors import missing_tensors
    base, ckpt, _, shared = _gemma_like(tmp_path)
    (base / "processor_config.json").write_text("{}")
    before = sorted(p.name for p in ckpt.iterdir())
    assert complete_checkpoint(base, ckpt, dry_run=True) == ["processor_config.json"]
    assert missing_tensors(base, ckpt) == sorted(shared)
    assert sorted(p.name for p in ckpt.iterdir()) == before, "a dry run must write nothing"


def test_run_rmu_saves_a_complete_checkpoint_before_rechecking_the_source():
    whole = (ROOT / "src" / "unlearn" / "rmu.py").read_text()
    i_save = whole.index("updated.save_pretrained(out)")
    i_carry = whole.index("carry_over_missing(src, out)")
    i_after = whole.index("after = _fingerprint(src)")
    assert i_save < i_carry < i_after, "carry-over only READS the source; the guard still follows it"


def test_the_diff_reports_tensors_the_checkpoint_lacks():


    src = (ROOT / "scripts" / "diff_checkpoint.py").read_text()
    assert "set(bi) - set(ci)" in src and "ABSENT from the checkpoint" in src


def test_the_rmu_figure_never_pools_think_states():


    from scripts.plot_rmu import _strict
    rows = [{"model": "g", "think_state": "off", "mmlu_subject": "wmdp-cyber", "probe_type": "forget",
             "n": "100", "n_parsed": "100", "n_correct": "80"},
            {"model": "g", "think_state": "on", "mmlu_subject": "wmdp-cyber", "probe_type": "forget",
             "n": "100", "n_parsed": "100", "n_correct": "20"}]
    assert _strict(rows, "g", think="off", subjects={"wmdp-cyber"})[0] == 80.0
    assert _strict(rows, "g", think="on", subjects={"wmdp-cyber"})[0] == 20.0
    src = (ROOT / "scripts" / "plot_rmu.py").read_text()
    assert 'f"_think-{think}"' in src, "one file per state"
    assert "--out names ONE file" in src, "a single --out with several states must refuse, not pool"


def test_the_rmu_table_highlights_each_checkpoints_own_target():


    from scripts.plot_rmu import _target_of
    assert _target_of("rmu-e2b-bio-c81a100") == "wmdp-bio"
    assert _target_of("rmu-gemma-cyber-lr25") == "wmdp-cyber"
    assert _target_of("rmu-llama-cyber-c20a300-s2") == "wmdp-cyber"
    assert _target_of("gemma-4-E2B-it") is None
    assert _target_of("rmu-bio-cyber-mixed") is None
    assert _target_of("rmu-e2b-biology-x") is None, "a name TOKEN, not a substring"


def test_the_rmu_figure_refuses_a_fourth_checkpoint_and_the_table_takes_the_grid(monkeypatch, tmp_path):


    import csv
    import scripts.plot_rmu as p
    four = ["rmu-a-cyber", "rmu-b-cyber", "rmu-c-cyber", "rmu-d-bio"]
    argv = ["plot_rmu.py", "--run", "rmu2", "--baseline-model", "base", "--checkpoints", *four]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="at most 3 checkpoints"):
        p.main()

    (tmp_path / "test").mkdir()
    with open(tmp_path / "test" / "accuracy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["model", "think_state", "mmlu_subject", "probe_type", "n", "n_parsed",
                               "n_correct"])
        w.writeheader()
        for m in ["base", *four]:
            w.writerow({"model": m, "think_state": "off", "mmlu_subject": "wmdp-cyber",
                        "probe_type": "forget", "n": 10, "n_parsed": 10, "n_correct": 5})
    drawn = {}
    monkeypatch.setattr(p, "analysis_dir", lambda run: tmp_path)
    monkeypatch.setattr(p, "_draw_table", lambda rows, args, models, labels, states, out:
                        drawn.update(models=models, states=states, out=out))
    monkeypatch.setattr(sys, "argv", [*argv, "--table", "--name", "rmu2_grid"])
    assert p.main() == 0
    assert drawn == {"models": ["base", *four], "states": ["off"],
                     "out": Path("png") / "rmu2_grid.png"}


def test_the_rmu_views_name_the_method_and_shorten_names_only_where_rows_are_grouped():


    from scripts.plot_rmu import _method_of, _pct, _row_label, _short_label
    assert _method_of(["elm-llama-bio", "elm-llama-cyber"]) == "ELM"
    assert _method_of(["rmu-e2b-cyber-c81a100"]) == "RMU"
    assert _method_of(["rmu-a-cyber", "elm-b-bio"]) == "Unlearning"
    assert _short_label("rmu-llama-cyber-c20a300-s2") == "c20a300-s2"
    assert _short_label("elm-llama-bio") == "elm-llama-bio", "nothing follows the target: keep the name"
    assert _row_label("rmu-g31-bio-r11a1000") == "bio · r11a1000"
    assert _row_label("elm-e2b-cyber") == "elm-e2b-cyber"
    assert (_pct(-0.2), _pct(37.4), _pct(-4.0)) == ("0%", "37%", "-4%"), "no '-0%'"


def _answers_run(root, monkeypatch, *, unanswered_in_text: int):


    import json
    import scripts.plot_rmu as p
    tables = {}
    for split, (n, parsed, correct) in {"val": (4, 3, 2), "test": (6, 5, 3)}.items():
        tables[split] = [{"model": "m", "think_state": "na", "mmlu_subject": "wmdp-cyber",
                          "probe_type": "forget", "n": str(n), "n_parsed": str(parsed),
                          "n_correct": str(correct)}]
    gen = {"probe_type": "forget", "concept_id": "wmdp-cyber", "think_state": "na"}
    rows = ([{**gen, "raw_output": "Answer: B", "finish_reason": "stop"}] * (10 - unanswered_in_text)
            + [{**gen, "raw_output": "I cannot help with that.", "finish_reason": "stop"}]
            + [{**gen, "raw_output": "the the the", "finish_reason": "length"}] * (unanswered_in_text - 1))
    root.mkdir(parents=True)
    (root / "m__none-baseline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                                 encoding="utf-8")
    monkeypatch.setattr(p, "generations_dir", lambda run: root)
    return p, tables


def test_the_answers_view_splits_non_answers_by_how_they_ended_and_refuses_a_mismatch(tmp_path,
                                                                                    monkeypatch):


    p, tables = _answers_run(tmp_path / "a", monkeypatch, unanswered_in_text=2)
    assert p._answer_split("r", tables, "m", "wmdp-cyber", "na") == {
        "n": 10, "correct": 5, "wrong": 3, "stopped": 1, "capped": 1}
    p, tables = _answers_run(tmp_path / "b", monkeypatch, unanswered_in_text=3)
    with pytest.raises(SystemExit, match="Rescore"):
        p._answer_split("r", tables, "m", "wmdp-cyber", "na")


def test_the_grid_views_take_every_checkpoint_when_none_are_named(monkeypatch, tmp_path):


    import csv
    import scripts.plot_rmu as p
    (tmp_path / "test").mkdir()
    with open(tmp_path / "test" / "accuracy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["model", "think_state", "mmlu_subject", "probe_type", "n", "n_parsed",
                               "n_correct"])
        w.writeheader()
        for m in ("base", "rmu-x-cyber-c1", "rmu-x-bio-c2"):
            w.writerow({"model": m, "think_state": "na", "mmlu_subject": "wmdp-cyber",
                        "probe_type": "forget", "n": 10, "n_parsed": 10, "n_correct": 5})
    drawn = {}
    monkeypatch.setattr(p, "analysis_dir", lambda run: tmp_path)
    monkeypatch.setattr(p, "_draw_answers", lambda args, models, labels, states, out:
                        drawn.update(models=models, labels=labels, out=out))
    monkeypatch.setattr(sys, "argv", ["plot_rmu.py", "--run", "r", "--baseline-model", "base",
                                      "--answers"])
    assert p.main() == 0
    assert drawn == {"models": ["base", "rmu-x-bio-c2", "rmu-x-cyber-c1"],
                     "labels": ["unedited", "c2", "c1"], "out": Path("png") / "r_answers.png"}
    monkeypatch.setattr(sys, "argv", ["plot_rmu.py", "--run", "r", "--baseline-model", "base"])
    with pytest.raises(SystemExit, match="--checkpoints is required"):
        p.main()
