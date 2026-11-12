import json
from pathlib import Path
from .rmu_author import sha

BATCHES = 150
BATCH_SIZE = 4


def eligible(texts):

    return [str(t) for t in texts if len(t) > 0]


def write_bundle(root, bio_file):
    from datasets import load_dataset
    root = Path(root)
    wiki = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    cyber = load_dataset('cais/wmdp-cyber-forget-corpus', split='train')
    with Path(bio_file).open() as f:
        bio = [json.loads(line)['text'] for line in f if line.strip()]
    payload = {}
    for key, texts in [('bio', bio), ('cyber', cyber['text']), ('retain', wiki['text'])]:
        rows = eligible(texts)
        if len(rows) < BATCHES * BATCH_SIZE:
            raise ValueError(f'{key}: need 600 eligible rows, found {len(rows)}')
        payload[key] = rows[:BATCHES * BATCH_SIZE]
    metadata = dict(bio_file_sha256=sha(bio_file),
                    cyber_dataset='cais/wmdp-cyber-forget-corpus', cyber_fingerprint=cyber._fingerprint,
                    retain_dataset='Salesforce/wikitext', retain_config='wikitext-2-raw-v1',
                    retain_split='test', retain_fingerprint=wiki._fingerprint,
                    min_len=0, batch_size=4, order=['bio', 'cyber'],
                    note='150 alternating updates use 75 batches (300 documents) per domain')
    for name, value in [('corpora.json', payload), ('corpora-manifest.json', metadata)]:
        dst = root / name
        content = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
        if dst.exists() and dst.read_text() != content:
            raise ValueError(f'Differing frozen input: {dst}; use a separate source directory')
        dst.write_text(content)


def load_bundle(root):
    root = Path(root)
    manifest = json.loads((root/'setup.json').read_text())
    if set(manifest['data_hashes']) != {'corpora.json', 'corpora-manifest.json'}:
        raise ValueError('Frozen data hash manifest is incomplete')
    for name, digest in manifest['data_hashes'].items():
        if sha(root/name) != digest:
            raise ValueError(f'Frozen training data changed: {name}')
    data = json.loads((root/'corpora.json').read_text())
    def batches(key):
        values = data[key]
        if len(values) != 600 or any(not isinstance(t, str) or not t for t in values):
            raise ValueError(f'Invalid frozen {key} corpus')
        return [values[i:i+4] for i in range(0, 600, 4)]
    return [batches('bio'), batches('cyber')], [batches('retain'), batches('retain')], manifest
