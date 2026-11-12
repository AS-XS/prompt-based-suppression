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
trap 'rc=$?; printf "RMU job failed (exit %s) at line %s: %s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR
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
printf 'RMU author checkout %s\n' "$(git rev-parse --short HEAD)"
args=(--variant "${VARIANT:?zephyr, g12 or llama}" --models-dir "$MODELS_DIR")
if [ -n "${NAME:-}" ]; then args+=(--name "$NAME"); fi
python scripts/train_rmu_author.py "${args[@]}"
