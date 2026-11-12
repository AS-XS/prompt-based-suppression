# RMU author-code reference: Zephyr, then Gemma 12B and Llama 8B Instruct

This is a separate experiment from the earlier local RMU implementation. Start with
**HuggingFaceH4/zephyr-7b-beta**, a published WMDP RMU reference, then review its results
before submitting the two adaptations. No smoke stage is required.

## What is preserved and what changes

The [published Zephyr notebook](https://github.com/centerforaisafety/wmdp/blob/c0b6c12bb0de3decf9c13bb13f9f1aa15754a132/run_rmu_zephyr.ipynb)
is pinned at `c0b6c12bb0de3decf9c13bb13f9f1aa15754a132`, together with its
`rmu/unlearn.py` and `rmu/utils.py`. Setup checks SHA256 before execution.
We execute the actual `run_rmu` function. The legacy Transformers 4.38.2 AdamW class
is downloaded and hash-checked too, rather than replacing it with PyTorch AdamW.
Its epsilon is 1e-6, weight decay 0, and bias correction enabled.

The recipe trains **bio and cyber jointly**, alternating domains for **150 total
updates**, batch size 4, learning rate 5e-5, steering 6.5 for each domain, alpha 1200
for each domain, seed 42, target layer 7, updated layers 5/6/7, parameter index 6.
Each domain receives 75 updates / 300 documents. The upstream batch-count guard
requires at least 150 available batches per corpus, so setup freezes 600 documents
from each in the original order. Retain is WikiText-2 raw **test**, not WikiText-103.
The author tokenizer caps forget at 512 bio / 768 cyber tokens and retain at 512.
Their `max_len=2000` data argument is unused; we do not silently truncate by characters.

Portability changes: local authorized bio JSONL, frozen local dataset bundle,
modern model runtime, one model copy per GPU, input placement, metrics and checkpoint
registration. The original training loop and optimizer update are unchanged for
Zephyr and Llama. Gemma additionally needs nested hidden-size/decoder access,
explicit down-projection selection, and layer 7/32 scaled to its decoder depth.
Gemma and Llama reuse the Zephyr hyperparameters initially; they are **untuned
adaptations**, not published recipes for those models.

This does not promise identical paper numbers: runtime versions and dataset revisions
can differ. WMDP8 chat generation below is our evaluation, not the paper's likelihood
protocol. GPU memory fit and scientific effectiveness have not been validated locally.

## 1. Existing SCC checkout: setup once, online

From `concept_removal`, with your existing `venv` activated:

```bash
git pull
source scripts/env.sh
hf download HuggingFaceH4/zephyr-7b-beta --local-dir models/zephyr-7b-beta
python scripts/setup_rmu_author.py
```

Prerequisites: `data/raw/bio-forget-corpus.jsonl` (authorized bio training corpus),
`data/processed/wmdp8/items.jsonl` and matching `split.json`. Setup checks these first,
fetches the pinned source and official `cais/wmdp-cyber-forget-corpus`, caches
`Salesforce/wikitext` / `wikitext-2-raw-v1` test, freezes the training bundle, and copies
evaluation inputs into the three new namespaces. It does not alter old inputs/results.
Repeated setup refuses differing inputs. Compute jobs read the frozen bundle offline.
The official cyber corpus and WikiText-2 are required to match this reference recipe;
previous WikiText-103 caching is not a substitute.

For a fresh advisor clone, install `pip install -e '.[gpu,analysis]' accelerate
sentencepiece` in `.venv`, obtain the authorized corpus using the dataset conversion
section of `RUN_R2MU_AUTHOR.md`, then build inputs once:

```bash
python -m src.data.build --run wmdp8 --wmdp bio cyber
python scripts/build_split.py --run wmdp8
```

The advisor clone does not need `scripts/env.sh`: skip that line. Its job wrapper
supports `.venv` or `venv` (or `VENV_DIR`) and defaults to `models`. If you customized
cache locations, preserve them between setup and jobs. Override the script's SCC
project using `qsub -P YOUR_PROJECT`; `cs505am` is the original owner's default.

## 2. Published-model reference: train Zephyr

```bash
mkdir -p logs
qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N rmu_author_zephyr -v VARIANT=zephyr scripts/qsub_rmu_author.sh
```

Training uses two GPUs, one updated model and one frozen reference. A single-GPU
submission is refused. The 48-hour allocation is a budget, not a runtime estimate.

```bash
tail -n 60 logs/rmu_author_zephyr.o*
tail -n 3 results/rmu-author-training/rmu-author-zephyr-joint/losses.jsonl
```

Success prints `RMU author checkpoint saved:` and writes
`models/unlearned/rmu-author-zephyr-joint/unlearn_report.json`. The report records
source/data hashes, exact hyperparameters, runtime versions, selected parameter names,
completed steps and counts of changed parameter elements. Check that all three selected
matrices changed. This verifies updates, not successful unlearning.

There is **no optimizer resume or periodic checkpoint** in this author loop. A failed
attempt is retained; inspect it before choosing a fresh `NAME` in qsub. Do not delete
old artifacts to force a retry. Subsequent generation must use the chosen retry name.

## 3. Zephyr: our generation and analysis

After the final checkpoint exists, from the original SCC checkout:

```bash
MODELS_DIR=models GPU_MEM=80 GPU_UTIL=0.92 SHARDS=2 WAVES=none-baseline CONFIG=configs/rmu-author-zephyr.yaml bash scripts/generate_all.sh zephyr-7b-beta
MODELS_DIR=models/unlearned GPU_MEM=80 GPU_UTIL=0.92 SHARDS=2 WAVES=none-baseline CONFIG=configs/rmu-author-zephyr.yaml bash scripts/generate_all.sh rmu-author-zephyr-joint
```

Once both generation jobs' shards finish:

```bash
python scripts/coverage.py --run rmu-author-zephyr --models zephyr-7b-beta rmu-author-zephyr-joint --waves none-baseline && python scripts/score_accuracy.py --run rmu-author-zephyr --split val --baseline-model zephyr-7b-beta
```

Read `results/rmu-author-zephyr/analysis/val/accuracy.csv`, `accuracy_delta.csv`,
`side_effect.csv`: inspect both bio/cyber, parsing, adjacent subjects, and general utility.
Both domains are edited here; neither is an untouched-domain control. The published
model uses a different evaluation protocol; do not equate our chat accuracy with it.

Advisor users can run the same generation without unexported SCC helpers, on an
allocated GPU, once per model (change `--models-dir` for the edited checkpoint):

```bash
python scripts/generate.py --config configs/rmu-author-zephyr.yaml --model zephyr-7b-beta --models-dir models --conditions baseline --gpu-util 0.92
python scripts/generate.py --config configs/rmu-author-zephyr.yaml --model rmu-author-zephyr-joint --models-dir models/unlearned --conditions baseline --gpu-util 0.92
```

## 4. After reviewing Zephyr: train both adaptations

Reuse the models already installed on SCC. Only if missing:

```bash
hf download google/gemma-4-12B-it --local-dir models/gemma-4-12B-it
hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir models/Meta-Llama-3-8B-Instruct
```

```bash
qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N rmu_author_g12 -v VARIANT=g12 scripts/qsub_rmu_author.sh
qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N rmu_author_llama -v VARIANT=llama scripts/qsub_rmu_author.sh
```

These write `rmu-author-g12-joint` and `rmu-author-llama-joint` in `models/unlearned`.
No reasoning-trace preparation is needed: this is RMU, not R²MU.

After each final checkpoint exists:

```bash
MODELS_DIR=models GPU_MEM=80 GPU_UTIL=0.92 SHARDS=4 WAVES=none-baseline CONFIG=configs/rmu-author-g12.yaml bash scripts/generate_all.sh gemma-4-12B-it
MODELS_DIR=models/unlearned GPU_MEM=80 GPU_UTIL=0.92 SHARDS=4 WAVES=none-baseline CONFIG=configs/rmu-author-g12.yaml bash scripts/generate_all.sh rmu-author-g12-joint
MODELS_DIR=models GPU_MEM=80 GPU_UTIL=0.92 SHARDS=2 WAVES=none-baseline CONFIG=configs/rmu-author-llama.yaml bash scripts/generate_all.sh Meta-Llama-3-8B-Instruct
MODELS_DIR=models/unlearned GPU_MEM=80 GPU_UTIL=0.92 SHARDS=2 WAVES=none-baseline CONFIG=configs/rmu-author-llama.yaml bash scripts/generate_all.sh rmu-author-llama-joint
```

For advisor clones use `scripts/generate.py` as above with each model/config pair;
it needs no scheduler helper. Then, once coverage passes:

```bash
python scripts/coverage.py --run rmu-author-g12 --models gemma-4-12B-it rmu-author-g12-joint --waves none-baseline && python scripts/score_accuracy.py --run rmu-author-g12 --split val --baseline-model gemma-4-12B-it
python scripts/coverage.py --run rmu-author-llama --models Meta-Llama-3-8B-Instruct rmu-author-llama-joint --waves none-baseline && python scripts/score_accuracy.py --run rmu-author-llama --split val --baseline-model Meta-Llama-3-8B-Instruct
```

Review validation before changing settings; keep separate names for new experiments.
After fixing the checkpoint, rerun the same scoring commands with `--split test`.
Retain is shared between val/test; it is not an independent held-out retention check.

## Published evaluation protocol (separate from WMDP8 chat scoring)

The authors specify `lm-evaluation-harness==0.4.2`, tasks `mmlu,wmdp`, HF likelihood
scoring, without chat-template prompting. To compare under that protocol, use a
**separate Python 3.11 environment** so the historical dependencies do not downgrade
your working generation environment. On a network-enabled machine:

```bash
python3.11 -m venv .venv-rmu-eval
.venv-rmu-eval/bin/pip install 'numpy<2' 'torch==2.1.2' 'transformers==4.38.2' 'lm-eval==0.4.2' 'datasets==2.18.0' 'pyarrow<21' 'accelerate==0.28.0' sentencepiece
```

On an allocated 80GB GPU with dataset network access/cache available, from the repo root:

```bash
.venv-rmu-eval/bin/lm-eval --model hf --model_args pretrained=models/zephyr-7b-beta --tasks mmlu,wmdp --batch_size 32 --output_path results/rmu-author-paper-eval/zephyr-base --log_samples
.venv-rmu-eval/bin/lm-eval --model hf --model_args pretrained=models/unlearned/rmu-author-zephyr-joint --tasks mmlu,wmdp --batch_size 32 --output_path results/rmu-author-paper-eval/zephyr-edited --log_samples
```

This historical evaluation environment has not been installed/tested here. Offline SCC
nodes require its task datasets to be cached in advance. Gemma's modern architecture
is not supported by that old Transformers environment; use the project pipeline for
the adaptations. Even matching the released recipe does not guarantee bitwise or
numerical reproduction under different hardware/runtime/data revisions.
