#!/usr/bin/env python3

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plan_wmdp8_unlearn import PROFILES, commands as original

RUNS = [f'{method}-{code}' for code,method,_,gpus in PROFILES if gpus==2]


def commands(stage):
    if stage=='setup':
        return ['python scripts/setup_unlearn_runs.py '+' '.join(r+'-mp80' for r in RUNS)]
    if stage=='train':
        output=[]
        for line in original('train'):
            parts=shlex.split(line)
            if 'gpus=2,gpu_memory=140G' not in parts:continue
            parts[parts.index('-l')+1]='gpus=2,gpu_memory=80G,h_rt=48:00:00'
            env=dict(v.split('=',1) for v in parts[parts.index('-v')+1].split(','))
            env['NAME']+='-mp80'
            env['MIXED_PRECISION']=1
            if parts[-1]!='scripts/qsub_unlearn.sh':env['MAX_SECONDS']=169200
            if parts[-1]!='scripts/qsub_r2mu.sh':env['DTYPE']='bfloat16'
            parts[parts.index('-N')+1]=env['NAME'].replace('-','_')
            parts[parts.index('-v')+1]=','.join(f'{k}={v}' for k,v in env.items())
            output.append(shlex.join(parts))
        return output
    output=[]
    for line in original(stage):
        if not any(f'configs/{r}.yaml' in line or f'--run {r} ' in line for r in RUNS):continue

        for run in RUNS:
            line=line.replace(f'configs/{run}.yaml', f'configs/{run}-mp80.yaml')
            line=line.replace(f'--run {run} ', f'--run {run}-mp80 ')
            for domain in ('bio','cyber'):
                for suffix in ('t01','control','p01'):
                    name=f'{run}-{domain}-{suffix}'
                    line=line.replace(name, name+'-mp80')
        output.append(line)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['setup','train','generate','score'])
    args=parser.parse_args()
    print('\n'.join(commands(args.stage)))
