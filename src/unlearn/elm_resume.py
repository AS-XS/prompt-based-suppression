from __future__ import annotations

import fcntl
import os
import random
import time
from pathlib import Path


class TrainingPaused(Exception):
    pass


def lock_work(work: Path):
    work.mkdir(parents=True, exist_ok=True)
    handle = (work / '.training.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f'Another ELM job is using {work}') from None
    return handle


class Recovery:
    def __init__(self, work, identity, *, resume, every=100, seconds=37800, started=None, label="ELM"):
        self.label = label
        self.path = Path(work) / 'resume.pt'
        self.identity = identity
        self.resume = resume
        self.every = every
        self.seconds = seconds
        self.started = time.monotonic() if started is None else started
        self.last = 0

    def restore(self, model, optimizer):
        import numpy as np
        import torch

        if not self.path.exists():
            return 0, -1, {}, {}
        if not self.resume:
            raise RuntimeError('Recovery exists; enable --resume')

        state = torch.load(self.path, map_location='cpu', weights_only=False)
        if state['identity'] != self.identity or state['version'] != 1:
            raise RuntimeError('ELM recovery configuration/corpus/base mismatch')
        parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if parameters.keys() != state['parameters'].keys():
            raise RuntimeError('ELM recovery trainable parameter names differ')
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = state['parameters'][name]
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise RuntimeError(f'ELM recovery parameter mismatch: {name}')
                parameter.copy_(value)
        optimizer.load_state_dict(state['optimizer'])
        optimizer.zero_grad()
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng']:
            if len(state['cuda_rng']) != torch.cuda.device_count():
                raise RuntimeError('ELM recovery CUDA device count differs')
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        self.last = state['next_idx']
        print(f'{self.label} resumed at {self.last} samples from {self.path}', flush=True)
        return self.last, state['iter_cnt'], state['dataset_cntr'], state['losses']

    def boundary(self, model, optimizer, next_idx, iter_cnt, dataset_cntr, losses,
                 accumulation, total):
        import numpy as np
        import torch

        if (iter_cnt + 1) % accumulation:
            return
        expired = bool(self.seconds and time.monotonic() - self.started >= self.seconds)
        if not (expired or next_idx == total or next_idx - self.last >= self.every):
            return
        state = dict(
            version=1, identity=self.identity, next_idx=next_idx, iter_cnt=iter_cnt,
            dataset_cntr=dataset_cntr, losses=losses,
            parameters={n: p.detach().cpu().clone() for n, p in model.named_parameters()
                        if p.requires_grad}, optimizer=optimizer.state_dict(),
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        )
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        self.last = next_idx
        print(f'{self.label} recovery saved: {next_idx}/{total} samples -> {self.path}', flush=True)
        if expired:
            raise TrainingPaused(f'{self.label} paused at {next_idx}/{total}; resubmit with RESUME=1')
