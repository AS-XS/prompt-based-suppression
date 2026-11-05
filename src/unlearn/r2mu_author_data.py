import json
from pathlib import Path
from src.unlearn.r2mu_author import sha, verified_source
from src.unlearn.r2mu_data import load_bundle, digest
from src.unlearn.rmu import _fingerprint


def batches(rows, size=4):
    return [rows[i:i+size] for i in range(0, len(rows), size)]


def training_data(source, forget_file, bundle, model, base, work, samples=2000):
    from datasets import load_dataset
    wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    retain = [str(row['text']) for row in wiki if len(row['text']) > 0][:samples]
    meta = {'wikitext_fingerprint': wiki._fingerprint}
    if bundle:
        manifest, rows, traces = load_bundle(bundle, model, 'wmdp-bio')
        if digest(manifest['base_fingerprint']) != digest(_fingerprint(base)):
            raise ValueError('Prepared traces belong to different model files')
        if len(rows) < samples:
            raise ValueError(f'Need {samples} usable native traces, found {len(rows)}')
        rows, traces = rows[:samples], traces[:samples]
        forget = [r['forget'] for r in rows]
        path = work / 'native-traces-author-format.jsonl'
        path.write_text(''.join(json.dumps({'512': t})+'\n' for t in traces))
        meta.update(preparation=manifest, selected_indices=[r['index'] for r in rows],
                    trace_sha256=sha(path), forget_sha256=digest(forget))
    else:
        verified_source(source, 'generated_all_wmdp.jsonl')
        path = Path(source) / 'generated_all_wmdp.jsonl'
        forget = []
        with Path(forget_file).open() as handle:
            for line in handle:
                text = json.loads(line)['text']
                if len(text) > 0:
                    forget.append(str(text))
                if len(forget) == samples:
                    break
        meta.update(trace_sha256=sha(path), forget_file=str(forget_file),
                    forget_sha256=digest(forget),
                    limitation='Author trace provenance and exact original private forget dataset are not supplied')
    if len(forget) < samples or len(retain) < samples:
        raise ValueError('Insufficient forget or WikiText rows for 500 batches of four')
    return [batches(forget)], [batches(retain)], path, meta
