#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unlearn.rmu_author import FILES, ROOT, MODELS, REVISION, sha
from src.unlearn.rmu_author_data import write_bundle
from src.paths import items_path, split_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT)
    ap.add_argument('--forget-file', type=Path, default=Path('data/raw/bio-forget-corpus.jsonl'))
    args = ap.parse_args()
    for path in (args.forget_file, items_path('wmdp8'), split_path('wmdp8')):
        if not path.is_file():
            ap.error(f'Missing {path}; see RMU_AUTHOR.md setup prerequisites')
    args.source.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in FILES.items():
        dst = args.source/name
        if not dst.exists():
            tmp = dst.with_suffix(dst.suffix+'.tmp')
            urllib.request.urlretrieve(url, tmp)
            if sha(tmp) != digest:
                tmp.unlink()
                raise ValueError(f'Download hash mismatch: {name}')
            tmp.replace(dst)
        if sha(dst) != digest:
            raise ValueError(f'Existing source hash mismatch: {name}')
    write_bundle(args.source, args.forget_file)
    for variant in MODELS:
        run = f'rmu-author-{variant}'
        for src, dst in [(items_path('wmdp8'), items_path(run)),
                         (split_path('wmdp8'), split_path(run))]:
            if dst.exists() and dst.read_bytes() != src.read_bytes():
                raise ValueError(f'Differing frozen evaluation input: {dst}')
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists(): shutil.copyfile(src, dst)
    manifest = dict(upstream_revision=REVISION, source_hashes={k:v[1] for k,v in FILES.items()},
                    data_hashes={k:sha(args.source/k) for k in ('corpora.json','corpora-manifest.json')})
    (args.source/'setup.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Verified author files, frozen training corpora, and copied WMDP8 evaluation inputs.')


if __name__ == '__main__': main()
