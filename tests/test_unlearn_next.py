import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_likelihood_keeps_forget_holdout_and_splits_duplicate_retain_consistently():
    prepare = script("prepare_unlearn_likelihood").prepare
    def item(ident, tier, concept="wmdp-bio"):
        return dict(item_id=ident, probe_type=tier, concept_id=concept,
                    question="Example question?", meta={"choices": ["one", "two", "three", "four"]},
                    answer_key="C")
    items = [item("f1", "forget"), item("f2", "forget"), item("r1", "utility"),
             item("r2", "utility", "wmdp-cyber")]
    split = {"val": {"wmdp-bio": ["f1"]}}
    val, test = (prepare(items, split, name) for name in ("val", "test"))
    def ids(groups):
        return {r["item_id"] for rows in groups.values() for r in rows}
    assert not ids(val) & ids(test)
    assert ids(val) | ids(test) == {"f1", "f2", "r1", "r2"}
    assert "f1" in ids(val) and "f2" in ids(test)
    assert ("r1" in ids(val)) == ("r2" in ids(val))
    row = val["wmdp_bio_forget"][0]
    assert row["answer"] == 2 and row["prompt"].endswith("D. four\nAnswer:")
    with pytest.raises(ValueError, match="absent"):
        prepare(items, {"val": {"x": ["missing"]}}, "val")
    with pytest.raises(ValueError, match="duplicate"):
        prepare(items + [items[0]], split, "val")


def test_grid_requires_bio_corpus_and_allocates_two_gpus_for_31b():
    module = script("plan_rmu")
    row = module.candidates("gemma-31b", "wmdp-bio")[0]
    with pytest.raises(ValueError, match="forget-file"):
        module.command(row, None)
    command = module.command(row, "data/raw/bio.jsonl")
    assert "gpus=2" in command and "FROZEN_DEVICE=cuda:1" in command
    assert "STEERING_RATIO=1.1" in command and "BATCH_SIZE=1" in command
    assert "EARLY_STOP=1" in command
    with pytest.raises(ValueError, match="commas"):
        module.command(row, "x,y")
    all_rows = [r for f in ["llama", "gemma-e2b", "gemma-31b"]
                for c in ["wmdp-bio", "wmdp-cyber"] for r in module.candidates(f, c)]
    assert len({r["name"] for r in all_rows}) == len(all_rows)


def test_prefix_hook_matches_full_forward_and_gradients_and_cleans_up():
    torch = pytest.importorskip("torch")
    from src.unlearn.activations import block_output
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])
            self.suffix_calls = 0

        def forward(self, input_ids, use_cache=False):
            h = input_ids
            outputs = []
            for i, block in enumerate(self.blocks):
                h = block(h)
                outputs.append(h)
                if i == 3:
                    self.suffix_calls += 1
            return outputs
    torch.manual_seed(7)
    model = Tiny()
    inputs = torch.randn(2, 3, 4)
    full = model(inputs)[2]
    full.square().mean().backward()
    grads = [b.weight.grad.clone() for b in model.blocks[:3]]
    model.zero_grad()
    prefix = block_output(model, model.blocks[2], {"input_ids": inputs})
    prefix.square().mean().backward()
    torch.testing.assert_close(full, prefix)
    for b, grad in zip(model.blocks[:3], grads):
        torch.testing.assert_close(b.weight.grad, grad)
    assert model.suffix_calls == 1
    assert not model.blocks[2]._forward_hooks
    assert model.blocks[3].weight.grad is None
    with pytest.raises(RuntimeError, match="not executed"):
        block_output(model, torch.nn.Linear(4, 4), {"input_ids": inputs})


def test_elm_rejects_unreviewed_source():
    from src.unlearn.elm_upstream import adapted_tree
    with pytest.raises(ValueError, match="hash mismatch"):
        adapted_tree("print('different revision')")


def test_elm_adapter_keeps_training_loop_and_excludes_post_training_evaluation(monkeypatch):
    import ast
    import hashlib
    from src.unlearn import elm_upstream
    source = '''
def train_elm(args):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    model = get_peft_model(model, lora_config)
    for batch in batches:
        loss = erase_loss + retain_loss + consistency_loss
        loss.backward()
        optimizer.step()
        model.save_pretrained(intermediate)
    model.save_pretrained(filename)
    evaluate_on_test()
if __name__ == "__main__":
    run_everything()
'''
    monkeypatch.setattr(elm_upstream, "SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest())
    before = ast.parse(source)
    after = elm_upstream.adapted_tree(source)
    assert len(after.body) == 1
    original_loop = next(n for n in before.body[0].body if isinstance(n, ast.For))
    adapted_loop = next(n for n in after.body[0].body if isinstance(n, ast.For))
    assert ast.dump(original_loop) == ast.dump(adapted_loop)
    assert "evaluate_on_test" not in ast.unparse(after)
    assert "_load_model(model_id, dtype)" in ast.unparse(after)
    assert "_get_elm_peft_model(model, lora_config, get_peft_model)" in ast.unparse(after)
