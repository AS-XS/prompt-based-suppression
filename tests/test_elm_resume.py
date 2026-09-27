import ast
import random

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from src.unlearn.elm_resume import Recovery, TrainingPaused, lock_work
from src.unlearn.elm_resume_ast import add_recovery


def model_and_optimizer(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(0.25))
    return model, torch.optim.AdamW(model.parameters(), lr=0.01)


def train(model, optimizer, recovery, stop=None):
    start, counter, positions, losses = recovery.restore(model, optimizer)
    for idx in range(start, 12):
        counter += 1
        positions[0] = positions.get(0, -1) + 1
        x = torch.randn(3) + random.random() + np.random.random()
        loss = model(x).square().mean()
        loss.backward()
        losses.setdefault('erase', []).append(loss.item())
        if (counter + 1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()
        if idx + 1 == stop:
            recovery.seconds = 1
            recovery.started = 0
        recovery.boundary(model, optimizer, idx + 1, counter, positions, losses, 4, 12)
    return losses


def test_interrupted_training_matches_uninterrupted_including_rng_and_optimizer(tmp_path):
    full = tmp_path / 'full'; full.mkdir()
    split = tmp_path / 'split'; split.mkdir()
    model, optimizer = model_and_optimizer(42)
    expected = train(model, optimizer, Recovery(full, {'seed': 42}, resume=False, seconds=0))
    weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    model, optimizer = model_and_optimizer(42)
    with pytest.raises(TrainingPaused, match='4/12'):
        train(model, optimizer, Recovery(split, {'seed': 42}, resume=False), stop=3)
    saved = torch.load(split / 'resume.pt', weights_only=False)
    assert saved['next_idx'] == 4 and saved['dataset_cntr'] == {0: 3}
    model, optimizer = model_and_optimizer(999)
    result = train(model, optimizer, Recovery(split, {'seed': 42}, resume=True, seconds=0))
    assert result == expected
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, weights[name])

    model, optimizer = model_and_optimizer(0)
    assert train(model, optimizer, Recovery(split, {'seed': 42}, resume=True)) == expected
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, weights[name])


def test_mismatch_and_interrupted_write_keep_last_checkpoint(tmp_path, monkeypatch):
    model, optimizer = model_and_optimizer(42)
    recovery = Recovery(tmp_path, {'lr': 0.01}, resume=False, seconds=0)
    train(model, optimizer, recovery)
    saved = recovery.path.read_bytes()
    with pytest.raises(RuntimeError, match='mismatch'):
        Recovery(tmp_path, {'lr': 0.02}, resume=True).restore(model, optimizer)
    with pytest.raises(RuntimeError, match='enable --resume'):
        recovery.restore(model, optimizer)
    def fail_save(state, handle):
        handle.write(b'incomplete')
        raise OSError('interrupted write')
    monkeypatch.setattr(torch, 'save', fail_save)
    with pytest.raises(OSError):
        recovery.boundary(model, optimizer, 12, 11, {0: 11}, {}, 4, 12)
    assert recovery.path.read_bytes() == saved


def test_duplicate_job_lock(tmp_path):
    first = lock_work(tmp_path)
    try:
        with pytest.raises(RuntimeError, match='Another ELM job'):
            lock_work(tmp_path)
    finally:
        first.close()
    lock_work(tmp_path).close()


def test_ast_preserves_inner_loop_and_places_save_after_update():
    function = ast.parse('''
def train_elm(args):
    if os.path.exists(filename):
        return filename
    with tqdm(total=batchsize) as pbar:
        for idx in range(0, batchsize):
            if args.save_every is not None:
                model.save_pretrained(filename)
            for data_idx in dataset_idxs:
                iter_cnt += 1
                loss.backward()
                if (iter_cnt + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            pbar.update(1)
''').body[0]
    original = ast.dump(function.body[1].body[0].body[1])
    add_recovery(function)
    ast.fix_missing_locations(function)
    text = ast.unparse(function)
    assert 'os.path.exists' not in text and 'save_pretrained' not in text
    loop = function.body[1].body[1]
    assert ast.dump(loop.body[0]) == original
    assert ast.unparse(loop.iter) == 'range(_resume_start, batchsize)'
    assert '_recovery.restore' in ast.unparse(function.body[0])
    assert '_recovery.boundary' in ast.unparse(loop.body[-1])
    with pytest.raises(ValueError, match='pinned training loop'):
        add_recovery(ast.parse('def train_elm(args):\n    pass').body[0])
