#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unlearn.r2mu_data import R2MU_MODELS, digest, resume_rows
from src.unlearn.rmu import _git_commit, _fingerprint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fetch-limo', action='store_true')
    ap.add_argument('--limo-file', type=Path, default=Path('data/raw/r2mu-limo.jsonl'))
    ap.add_argument('--model', choices=R2MU_MODELS)
    ap.add_argument('--concept', choices=['wmdp-bio', 'wmdp-cyber'], default='wmdp-bio')
    ap.add_argument('--models-dir', default='models')
    ap.add_argument('--forget-file', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--samples', type=int, default=800)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.fetch_limo:
        if args.limo_file.exists():
            print(f'Using existing {args.limo_file}'); return
        from datasets import load_dataset
        ds = load_dataset('GAIR/LIMO', split='train')
        args.limo_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.limo_file.with_suffix('.tmp')
        with tmp.open('w') as handle:
            for i, row in enumerate(ds):
                handle.write(json.dumps(dict(question=row['question'], solution=row['solution'],
                    source='GAIR/LIMO', index=i, dataset_fingerprint=ds._fingerprint)) + '\n')
        tmp.replace(args.limo_file)
        print(f'Cached {len(ds)} LIMO rows: {args.limo_file}'); return
    if not args.model or not args.out or args.samples <= 0 or not 0 < args.max_tokens <= 3072:
        ap.error('GPU preparation requires model, out and positive samples and max-tokens in 1..3072')
    if args.concept == 'wmdp-bio' and not args.forget_file:
        ap.error('bio requires the authorized local --forget-file')
    from src.unlearn.corpora import load_forget
    from src.gen.env import setup_vllm_env, make_llm
    from src.gen.thinkstate import chat_kwargs
    from src.unlearn.elm_resume import lock_work
    setup_vllm_env()
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    base = Path(args.models_dir) / args.model
    tokenizer = AutoTokenizer.from_pretrained(base)
    corpus = load_forget(args.concept, local_path=args.forget_file, limit=args.samples)
    if len(corpus) < args.samples:
        ap.error('not enough training corpus')

    segments = [tokenizer.decode(tokenizer.encode(t, add_special_tokens=False)[:512],
                                 skip_special_tokens=True) for t in corpus]
    if any(not t.strip() for t in segments):
        ap.error('empty training segment')
    kw = chat_kwargs(args.model, 'on').get('chat_template_kwargs', {})
    messages = [[{'role': 'user', 'content': 'Explain this passage step by step:\n\n' + t}]
                for t in segments]
    prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
               for m in messages]
    spec = dict(model=args.model, concept=args.concept, samples=args.samples, seed=args.seed,
                max_tokens=args.max_tokens, temperature=0.6, top_p=0.95, top_k=-1,
                corpus_hash=digest(segments), prompts_hash=digest(prompts),
                base_fingerprint=_fingerprint(base), version=1)
    spec['config_hash'] = digest(spec)
    _lock = lock_work(args.out)
    manifest = args.out / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['config_hash'] != spec['config_hash']:
        ap.error('existing preparation differs; use another output directory')
    manifest.write_text(json.dumps(spec, indent=2))
    path = args.out / 'traces.jsonl'
    rows = resume_rows(path)
    if [r['index'] for r in rows] != list(range(len(rows))) or any(
            r['config_hash'] != spec['config_hash'] for r in rows) or len(rows) > args.samples:
        ap.error('mixed/duplicate trace file; inspect before resuming')
    if len(rows) == args.samples:
        print('R2MU training traces already complete'); return
    llm = make_llm(str(base), dtype='bfloat16', gpu_memory_utilization=0.90,
                   max_model_len=4096, max_num_seqs=1)
    commit = _git_commit()
    with path.open('a') as handle:
        for i in range(len(rows), args.samples):
            sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.6, top_p=0.95,
                                      top_k=-1, seed=args.seed+i, skip_special_tokens=False)
            output = llm.generate([prompts[i]], sampling, use_tqdm=False)[0].outputs[0]
            row = dict(index=i, model=args.model, concept=args.concept, forget=segments[i],
                source_hash=digest(segments[i]), config_hash=spec['config_hash'],
                git_commit=commit, seed=args.seed+i, prompt=prompts[i], raw_output=output.text,
                raw_token_ids=list(output.token_ids), finish_reason=output.finish_reason)
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush(); os.fsync(handle.fileno())
            print(f'R2MU training traces: {i+1}/{args.samples}', flush=True)


if __name__ == '__main__':
    main()
