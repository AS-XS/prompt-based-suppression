#!/usr/bin/env python3

import json
import shutil
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unlearn.r2mu_author import DEFAULT_SOURCE, HASHES, REVISION, sha
from src.paths import items_path, split_path


def main():
    DEFAULT_SOURCE.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        dst = DEFAULT_SOURCE / name
        if dst.exists():
            if sha(dst) != expected:
                raise ValueError(f'Differing existing file: {dst}')
        else:
            url = f'https://raw.githubusercontent.com/OPTML-Group/Unlearn-R2MU/{REVISION}/{name}'
            tmp = dst.with_suffix(dst.suffix + '.tmp')
            urllib.request.urlretrieve(url, tmp)
            if sha(tmp) != expected:
                tmp.unlink()
                raise ValueError(f'Download checksum mismatch: {name}')
            tmp.replace(dst)
    from datasets import load_dataset
    fingerprints = {}
    for name, config, split in [('wikitext', 'wikitext-2-raw-v1', 'test'), ('GAIR/LIMO', None, 'train')]:
        ds = load_dataset(name, config, split=split)
        fingerprints[name] = ds._fingerprint
    (DEFAULT_SOURCE/'datasets.json').write_text(json.dumps(fingerprints, indent=2))
    for run in ('r2mu-author-ds8', 'r2mu-author-g12'):
        for source, target in [(items_path('wmdp8'), items_path(run)), (split_path('wmdp8'), split_path(run))]:
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(f'Differing frozen evaluation input: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
    print('Verified upstream files, cached WikiText/LIMO, and copied frozen WMDP8 evaluation inputs.')


if __name__ == '__main__':
    main()
