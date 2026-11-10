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
trap 'rc=$?; printf "R2MU job failed (exit %s) at line %s: %s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR
cd "${SGE_O_WORKDIR:?Submit from the concept_removal project root}"
module load cuda 2>/dev/null || true
if [ -n "${VENV_DIR:-}" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo 'No .venv or venv found; set VENV_DIR to your environment directory.' >&2
    exit 2
fi
# The source checkout has a personal cache helper; the advisor export does not.
# Without it, retain the same default HF cache used during setup.
if [ -f scripts/env.sh ]; then source scripts/env.sh; fi
export MODELS_DIR="${MODELS_DIR:-./models}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
printf 'R2MU author checkout %s\n' "$(git rev-parse --short HEAD)"
if [ "${MODE:-train}" = traces ]; then
    python scripts/prepare_r2mu.py --model gemma-4-12B-it --models-dir "$MODELS_DIR" \
        --concept wmdp-bio --forget-file data/raw/bio-forget-corpus.jsonl \
        --out data/processed/r2mu-author-training/g12-bio --samples 2400
elif [ "${MODE:-train}" = train ]; then
    args=(--variant "${VARIANT:?ds8 or g12}" --models-dir "$MODELS_DIR")
    if [ "$VARIANT" = g12 ]; then
        args+=(--bundle data/processed/r2mu-author-training/g12-bio)
    fi
    if [ -n "${NAME:-}" ]; then args+=(--name "$NAME"); fi
    python scripts/train_r2mu_author.py "${args[@]}"
else
    echo 'MODE must be train or traces' >&2
    exit 2
fi
