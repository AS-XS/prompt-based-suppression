import json
from pathlib import Path
from .rmu_author import sha

BATCHES = 150
BATCH_SIZE = 4


def eligible(texts):

    return [str(t) for t in texts if len(t) > 0]


def write_bundle(root, bio_file, cyber_source='author'):
    from datasets import load_dataset
    root = Path(root)
    wiki = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    if cyber_source == 'author':
        cyber_id, cyber_config = 'cais/wmdp-cyber-forget-corpus', None
        cyber = load_dataset(cyber_id, split='train')
    elif cyber_source == 'cached-corpora':
        from datasets import config
        cyber_id, cyber_config = 'cais/wmdp-corpora', 'cyber-forget-corpus'
        previous = config.HF_DATASETS_OFFLINE
        try:
            config.HF_DATASETS_OFFLINE = True
            cyber = load_dataset(cyber_id, cyber_config, split='train')
        finally:
            config.HF_DATASETS_OFFLINE = previous
    else:
        raise ValueError(f'Unknown cyber source: {cyber_source}')
    print(f'Cyber input: {cyber_id} / {cyber_config or "default"} ({cyber_source})')
    with Path(bio_file).open() as f:
        bio = [json.loads(line)['text'] for line in f if line.strip()]
    payload = {}
    for key, texts in [('bio', bio), ('cyber', cyber['text']), ('retain', wiki['text'])]:
        rows = eligible(texts)
        if len(rows) < BATCHES * BATCH_SIZE:
            raise ValueError(f'{key}: need 600 eligible rows, found {len(rows)}')
        payload[key] = rows[:BATCHES * BATCH_SIZE]
    metadata = dict(bio_file_sha256=sha(bio_file),
                    cyber_dataset=cyber_id, cyber_fingerprint=cyber._fingerprint,
                    retain_dataset='Salesforce/wikitext', retain_config='wikitext-2-raw-v1',
                    retain_split='test', retain_fingerprint=wiki._fingerprint,
                    min_len=0, batch_size=4, order=['bio', 'cyber'],
                    note='150 alternating updates use 75 batches (300 documents) per domain')
    if cyber_source != 'author':
        metadata.update(cyber_config=cyber_config, cyber_source=cyber_source,
                        author_dataset_equivalence='not verified; explicitly reused existing corpus')
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
