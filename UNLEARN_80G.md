# Independent 2x80G / 48-hour large-model variants

Keep the original 140G jobs queued. These 14 jobs use NEW `-mp80` checkpoint names,
recovery directories, and evaluation namespaces. No original jobs are cancelled.
Both domains: R2MU Gemma31B and OLMo32B Think (each with control), RMU OLMo32B
Instruct, ELM OLMo32B Instruct/Think. Previously completed Gemma31B ELM is not rerun.

This is an explicit precision variant, not a bitwise-equivalent memory optimization.
No quantization. RMU/R2MU frozen backbone and reference are bf16; trainable master
weights and Adam state are float32. Updated forward uses bf16 autocast, losses use
float32. One copy per GPU. ELM uses a bf16 backbone balanced across two GPUs with
40GiB placement budgets per GPU and float32 LoRA adapters; PEFT handles adapter
input casting. Original non-mixed paths retain their existing precision/placement.
All final exports use bf16. R2MU/ELM recovery identities record the precision change
and cannot resume original precision checkpoints. Original R2MU traces are reused.

Training resources: 2 GPUs, each at least 80 GB, with a 48-hour limit. ELM/R2MU pause after 47 hours with recovery;
ordinary RMU has no resume timer. Actual GPU peak memory and queue eligibility are
not validated locally. No smoke jobs, per owner. Do not repeat this submission
if the same mp80 jobs are queued/running or their checkpoints already exported.

L4 has 24 GB and does not qualify for 80G. SCC gpu_memory is per GPU. GPU type is
not pinned; qgpus -v reports current eligible hardware. A 48-hour request may exclude
buy-in queues limited to 12 hours, so smaller memory does not guarantee faster scheduling.
Sources: https://www.nvidia.com/en-au/data-center/l4/ and
https://www.bu.edu/tech/support/research/software-and-programming/gpu-computing/ and
https://www.bu.edu/tech/support/research/system-usage/running-jobs/resources-jobs/

Run setup and train after git pull; generation only after checkpoint exports;
coverage before validation scoring. Do not pool old and mp80 results.

## setup

```bash
python scripts/setup_unlearn_runs.py r2mu-g31-mp80 r2mu-o32-mp80 rmu-o32-mp80 elm-o32-mp80 elm-o32t-mp80
```

## train

```bash
qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N r2mu_g31_bio_t01_mp80 -v MODEL=gemma-4-31B-it,NAME=r2mu-g31-bio-t01-mp80,CONCEPT=wmdp-bio,DATA=data/processed/r2mu-training/g31-bio,LAYER=auto,STEERING_RATIO=1.1,GAMMA=300,LR=5e-05,BATCH_SIZE=1,STEPS=150,TRACE_WEIGHT=0.1,COT_WEIGHT=1,RESUME=1,FROZEN_DEVICE=cuda:1,MIXED_PRECISION=1,MAX_SECONDS=169200 scripts/qsub_r2mu.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N r2mu_g31_bio_control_mp80 -v MODEL=gemma-4-31B-it,NAME=r2mu-g31-bio-control-mp80,CONCEPT=wmdp-bio,DATA=data/processed/r2mu-training/g31-bio,LAYER=auto,STEERING_RATIO=1.1,GAMMA=300,LR=5e-05,BATCH_SIZE=1,STEPS=150,TRACE_WEIGHT=0,COT_WEIGHT=0,RESUME=1,FROZEN_DEVICE=cuda:1,MIXED_PRECISION=1,MAX_SECONDS=169200 scripts/qsub_r2mu.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N r2mu_g31_cyber_t01_mp80 -v MODEL=gemma-4-31B-it,NAME=r2mu-g31-cyber-t01-mp80,CONCEPT=wmdp-cyber,DATA=data/processed/r2mu-training/g31-cyber,LAYER=auto,STEERING_RATIO=1.1,GAMMA=300,LR=5e-05,BATCH_SIZE=1,STEPS=150,TRACE_WEIGHT=0.1,COT_WEIGHT=1,RESUME=1,FROZEN_DEVICE=cuda:1,MIXED_PRECISION=1,MAX_SECONDS=169200 scripts/qsub_r2mu.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N r2mu_g31_cyber_control_mp80 -v MODEL=gemma-4-31B-it,NAME=r2mu-g31-cyber-control-mp80,CONCEPT=wmdp-cyber,DATA=data/processed/r2mu-training/g31-cyber,LAYER=auto,STEERING_RATIO=1.1,GAMMA=300,LR=5e-05,BATCH_SIZE=1,STEPS=150,TRACE_WEIGHT=0,COT_WEIGHT=0,RESUME=1,FROZEN_DEVICE=cuda:1,MIXED_PRECISION=1,MAX_SECONDS=169200 scripts/qsub_r2mu.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N r2mu_o32_bio_t01_mp80 -v MODEL=Olmo-3.1-32B-Think,NAME=r2mu-o32-bio-t01-mp80,CONCEPT=wmdp-bio,DATA=data/processed/r2mu-training/o32-bio,LAYER=auto,STEERING_RATIO=1.1,GAMMA=300,LR=5e-05,BATCH_SIZE=1,STEPS=150,TRACE_WEIGHT=0.1,COT_WEIGHT=1,RESUME=1,FROZEN_DEVICE=cuda:1,MIXED_PRECISION=1,MAX_SECONDS=169200 scripts/qsub_r2mu.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N r2mu_o32_bio_control_mp80 -v MODEL=Olmo-3.1-32B-Think,NAME=r2mu-o32-bio-control-mp80,CONCEPT=wmdp-bio,DATA=data/processed/r2mu-training/o32-bio,LAYER=auto,STEERING_RATIO=1.1,GAMMA=300,LR=5e-05,BATCH_SIZE=1,STEPS=150,TRACE_WEIGHT=0,COT_WEIGHT=0,RESUME=1,FROZEN_DEVICE=cuda:1,MIXED_PRECISION=1,MAX_SECONDS=169200 scripts/qsub_r2mu.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N r2mu_o32_cyber_t01_mp80 -v MODEL=Olmo-3.1-32B-Think,NAME=r2mu-o32-cyber-t01-mp80,CONCEPT=wmdp-cyber,DATA=data/processed/r2mu-training/o32-cyber,LAYER=auto,STEERING_RATIO=1.1,GAMMA=300,LR=5e-05,BATCH_SIZE=1,STEPS=150,TRACE_WEIGHT=0.1,COT_WEIGHT=1,RESUME=1,FROZEN_DEVICE=cuda:1,MIXED_PRECISION=1,MAX_SECONDS=169200 scripts/qsub_r2mu.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N r2mu_o32_cyber_control_mp80 -v MODEL=Olmo-3.1-32B-Think,NAME=r2mu-o32-cyber-control-mp80,CONCEPT=wmdp-cyber,DATA=data/processed/r2mu-training/o32-cyber,LAYER=auto,STEERING_RATIO=1.1,GAMMA=300,LR=5e-05,BATCH_SIZE=1,STEPS=150,TRACE_WEIGHT=0,COT_WEIGHT=0,RESUME=1,FROZEN_DEVICE=cuda:1,MIXED_PRECISION=1,MAX_SECONDS=169200 scripts/qsub_r2mu.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N rmu_o32_bio_p01_mp80 -v MODEL=Olmo-3.1-32B-Instruct,NAME=rmu-o32-bio-p01-mp80,CONCEPT=wmdp-bio,LAYER=auto,STEERING_RATIO=1.1,ALPHA=300,LR=5e-05,MAX_BATCHES=150,BATCH_SIZE=1,DTYPE=bfloat16,FROZEN_DTYPE=bfloat16,EARLY_STOP=1,FROZEN_DEVICE=cuda:1,FORGET_FILE=data/raw/bio-forget-corpus.jsonl,MIXED_PRECISION=1 scripts/qsub_unlearn.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N rmu_o32_cyber_p01_mp80 -v MODEL=Olmo-3.1-32B-Instruct,NAME=rmu-o32-cyber-p01-mp80,CONCEPT=wmdp-cyber,LAYER=auto,STEERING_RATIO=1.1,ALPHA=300,LR=5e-05,MAX_BATCHES=150,BATCH_SIZE=1,DTYPE=bfloat16,FROZEN_DTYPE=bfloat16,EARLY_STOP=1,FROZEN_DEVICE=cuda:1,MIXED_PRECISION=1 scripts/qsub_unlearn.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N elm_o32_bio_p01_mp80 -v MODEL=Olmo-3.1-32B-Instruct,NAME=elm-o32-bio-p01-mp80,CONCEPT=wmdp-bio,LAYERS=auto,SAMPLES=3000,ETA=500,LR=5e-05,RESUME=1,PARALLEL=1,FORGET_FILE=data/raw/bio-forget-corpus.jsonl,MIXED_PRECISION=1,MAX_SECONDS=169200,DTYPE=bfloat16 scripts/qsub_elm.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N elm_o32_cyber_p01_mp80 -v MODEL=Olmo-3.1-32B-Instruct,NAME=elm-o32-cyber-p01-mp80,CONCEPT=wmdp-cyber,LAYERS=auto,SAMPLES=3000,ETA=500,LR=5e-05,RESUME=1,PARALLEL=1,MIXED_PRECISION=1,MAX_SECONDS=169200,DTYPE=bfloat16 scripts/qsub_elm.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N elm_o32t_bio_p01_mp80 -v MODEL=Olmo-3.1-32B-Think,NAME=elm-o32t-bio-p01-mp80,CONCEPT=wmdp-bio,LAYERS=auto,SAMPLES=3000,ETA=500,LR=5e-05,RESUME=1,PARALLEL=1,FORGET_FILE=data/raw/bio-forget-corpus.jsonl,MIXED_PRECISION=1,MAX_SECONDS=169200,DTYPE=bfloat16 scripts/qsub_elm.sh

qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N elm_o32t_cyber_p01_mp80 -v MODEL=Olmo-3.1-32B-Think,NAME=elm-o32t-cyber-p01-mp80,CONCEPT=wmdp-cyber,LAYERS=auto,SAMPLES=3000,ETA=500,LR=5e-05,RESUME=1,PARALLEL=1,MIXED_PRECISION=1,MAX_SECONDS=169200,DTYPE=bfloat16 scripts/qsub_elm.sh
```

## generate

```bash
MODELS_DIR=models/unlearned GPU_MEM=80 GPU_UTIL=0.92 WAVES="none-baseline" CONFIG=configs/r2mu-g31-mp80.yaml bash scripts/generate_all.sh r2mu-g31-bio-t01-mp80 r2mu-g31-bio-control-mp80 r2mu-g31-cyber-t01-mp80 r2mu-g31-cyber-control-mp80

MODELS_DIR=models/unlearned GPU_MEM=80 GPU_UTIL=0.92 SHARDS=8 WAVES="none-baseline" CONFIG=configs/r2mu-o32-mp80.yaml bash scripts/generate_all.sh r2mu-o32-bio-t01-mp80 r2mu-o32-bio-control-mp80 r2mu-o32-cyber-t01-mp80 r2mu-o32-cyber-control-mp80

MODELS_DIR=models/unlearned GPU_MEM=80 GPU_UTIL=0.92 WAVES="none-baseline" CONFIG=configs/rmu-o32-mp80.yaml bash scripts/generate_all.sh rmu-o32-bio-p01-mp80 rmu-o32-cyber-p01-mp80

MODELS_DIR=models/unlearned GPU_MEM=80 GPU_UTIL=0.92 WAVES="none-baseline" CONFIG=configs/elm-o32-mp80.yaml bash scripts/generate_all.sh elm-o32-bio-p01-mp80 elm-o32-cyber-p01-mp80

MODELS_DIR=models/unlearned GPU_MEM=80 GPU_UTIL=0.92 SHARDS=8 WAVES="none-baseline" CONFIG=configs/elm-o32t-mp80.yaml bash scripts/generate_all.sh elm-o32t-bio-p01-mp80 elm-o32t-cyber-p01-mp80
```

## score

```bash
python scripts/coverage.py --run r2mu-g31-mp80 --config configs/r2mu-g31-mp80.yaml --models gemma-4-31B-it r2mu-g31-bio-t01-mp80 r2mu-g31-bio-control-mp80 r2mu-g31-cyber-t01-mp80 r2mu-g31-cyber-control-mp80

python scripts/score_accuracy.py --run r2mu-g31-mp80 --split val --baseline-model gemma-4-31B-it

python scripts/coverage.py --run r2mu-o32-mp80 --config configs/r2mu-o32-mp80.yaml --models Olmo-3.1-32B-Think r2mu-o32-bio-t01-mp80 r2mu-o32-bio-control-mp80 r2mu-o32-cyber-t01-mp80 r2mu-o32-cyber-control-mp80

python scripts/score_accuracy.py --run r2mu-o32-mp80 --split val --baseline-model Olmo-3.1-32B-Think

python scripts/coverage.py --run rmu-o32-mp80 --config configs/rmu-o32-mp80.yaml --models Olmo-3.1-32B-Instruct rmu-o32-bio-p01-mp80 rmu-o32-cyber-p01-mp80

python scripts/score_accuracy.py --run rmu-o32-mp80 --split val --baseline-model Olmo-3.1-32B-Instruct

python scripts/coverage.py --run elm-o32-mp80 --config configs/elm-o32-mp80.yaml --models Olmo-3.1-32B-Instruct elm-o32-bio-p01-mp80 elm-o32-cyber-p01-mp80

python scripts/score_accuracy.py --run elm-o32-mp80 --split val --baseline-model Olmo-3.1-32B-Instruct

python scripts/coverage.py --run elm-o32t-mp80 --config configs/elm-o32t-mp80.yaml --models Olmo-3.1-32B-Think elm-o32t-bio-p01-mp80 elm-o32t-cyber-p01-mp80

python scripts/score_accuracy.py --run elm-o32t-mp80 --split val --baseline-model Olmo-3.1-32B-Think
```
