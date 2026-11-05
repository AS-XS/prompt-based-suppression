#!/bin/bash -l
# Full author-code reference/adaptation, not a smoke job.
#$ -P cs505am
#$ -l gpus=2
#$ -l gpu_c=8.0
#$ -l gpu_memory=80G
#$ -l h_rt=48:00:00
#$ -j y
#$ -o logs/
set -euo pipefail
cd "${SGE_O_WORKDIR:?Submit from the concept_removal project root}"
module load cuda 2>/dev/null || true
source venv/bin/activate
source scripts/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
printf 'R2MU author checkout %s\n' "$(git rev-parse --short HEAD)"
if [ "${MODE:-train}" = traces ]; then
    python scripts/prepare_r2mu.py --model gemma-4-12B-it --models-dir "$MODELS_DIR" \
        --concept wmdp-bio --forget-file data/raw/bio-forget-corpus.jsonl \
        --out data/processed/r2mu-author-training/g12-bio --samples 2400
elif [ "${MODE:-train}" = train ]; then
    extra=()
    if [ "${VARIANT:?ds8 or g12}" = g12 ]; then
        extra=(--bundle data/processed/r2mu-author-training/g12-bio)
    fi
    if [ -n "${NAME:-}" ]; then extra+=(--name "$NAME"); fi
    python scripts/train_r2mu_author.py --variant "$VARIANT" --models-dir "$MODELS_DIR" "${extra[@]}"
else
    echo 'MODE must be train or traces' >&2
    exit 2
fi
