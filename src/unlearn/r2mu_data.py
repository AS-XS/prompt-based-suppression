import hashlib
import json
from pathlib import Path

from src.analysis.surfaces import split_think

AUTHOR_REVISION = '3529e557a9c5d72ef53b7ee2b677a3aa8d2c8b06'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trace_text(row):

    if '<|channel>' not in row['raw_output']:
        raise ValueError(f"No native Gemma reasoning channel for row {row['index']}")
    trace, _ = split_think(row['raw_output'], expect_trace=True)
    if not trace or not trace.strip():
        raise ValueError(f"No reasoning trace for training row {row['index']}")
    return trace


def load_bundle(directory, model, concept):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['model'] != model or manifest['concept'] != concept:
        raise ValueError('R2MU data model/concept mismatch')
    rows = read_rows(root / 'traces.jsonl')
    if len(rows) != manifest['samples'] or [r['index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Incomplete or duplicate R2MU training traces; rerun preparation')
    if any(r['config_hash'] != manifest['config_hash'] for r in rows):
        raise ValueError('Mixed R2MU generation configurations')
    if any(digest(r['forget']) != r['source_hash'] for r in rows):
        raise ValueError('R2MU source text hash mismatch')
    usable, traces = [], []
    for row in rows:
        try:
            trace = trace_text(row)
        except ValueError:
            continue
        usable.append(row); traces.append(trace)
    print(f'R2MU data: {len(usable)}/{len(rows)} rows have explicit reasoning channels; '
          f'{sum(r["finish_reason"] == "length" for r in rows)} hit the generation cap')
    return manifest, usable, traces


def resume_rows(path):

    path = Path(path)
    if not path.exists():
        return []
    blob = path.read_bytes()
    lines = blob.splitlines(keepends=True)
    rows, offset = [], 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(lines)-1 or line.endswith(b'\n'):
                raise ValueError('Corrupt trace JSONL; inspect before resuming') from None
            with path.open('r+b') as handle:
                handle.truncate(offset)
            break
        rows.append(row); offset += len(line)
    if rows and offset == len(blob) and not blob.endswith(b'\n'):
        with path.open('ab') as handle:
            handle.write(b'\n')
    return rows
