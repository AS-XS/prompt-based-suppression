# R²MU author-code reference, then Gemma 12B

This executes `run_rmu` from OPTML-Group/Unlearn-R2MU revision
`3529e557a9c5d72ef53b7ee2b677a3aa8d2c8b06`, verified by SHA256. It does not call
our reimplementation in `src/unlearn/r2mu.py`. Sources and released traces are
fetched into the ignored `data/external/r2mu-author/`, not vendored into this repo.

## What the reference establishes

DeepSeek-R1-Distill-Llama-8B runs the authors' training loop: AdamW, batch4, 500
updates, seed42, lr7.5e-5, alpha1.4, steering6.5, assistant loss1, raw/retain cap512,
trace cap100, assistant/reasoning cap150, unmasked MSE, layer7 and parameter index6
of layers5/6/7. Parameter names are checked to be three down-projection weights.
The shipped model precision is retained, including optimizer precision; no silent
FP32-master conversion. Changed-element counts expose a possible zero-update run.

Necessary operational repairs: local corpus and cache paths; the upstream LIMO
loader references an undefined `assistant_msg`, repaired by reading that row's
assistant message; JSONL metrics instead of W&B; one copy per GPU with inputs moved
to the corresponding device; separate evaluation instead of private paths and
automatic upstream evaluation. Objective and backward/update statements stay intact.

The released 2,000 traces have only a `512` text field, without source IDs or
complete generation provenance. The author's private `bio_remove_dataset.jsonl`
is not released. We use the first2,000 nonempty rows of the existing authorized
bio forget corpus (no shuffling, as in their loader), WikiText2 raw test, and LIMO.
This is an author-code reference with a disclosed corpus substitution, NOT an
exact reproduction of paper numbers or a proven contamination-free benchmark.
Keep its results separate. Raw author traces are retained unchanged and hashed.

The upstream trainer has intermediate weight exports at200/400 updates but no
optimizer recovery. The 48-hour job runs the full500 updates. Existing output
is refused; do not use the old adaptation's RESUME flag here. If interrupted,
inspect the saved intermediate weights and logs before choosing a new run.
`unlearn_report.json` marks the final completed export only. Base files are guarded.

## First: download/setup on the login node

Use your current activated venv and usual `scripts/env.sh`. No new Python package
installation is required beyond the project's installed training dependencies.
The model download reuses files if already present. Setup verifies downloads and
caches the upstream datasets for offline compute nodes; never run setup on a GPU.

```bash
git pull
hf download deepseek-ai/DeepSeek-R1-Distill-Llama-8B --local-dir models/DeepSeek-R1-Distill-Llama-8B
python scripts/setup_r2mu_author.py
qsub -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N r2mu_author_ds8_bio -v VARIANT=ds8 scripts/qsub_r2mu_author.sh
```

## Evaluate the DeepSeek reference after final export

These are our existing WMDP8 generation/scoring instruments, not the authors'
likelihood/API evaluation. Compare against a fresh unedited baseline under the
same instrument. No API judge messages are sent. Validation is used for diagnosis;
final-answer accuracy alone does not establish removal of reasoning-trace leakage.

```bash
MODELS_DIR=models GPU_MEM=80 GPU_UTIL=0.92 SHARDS=4 WAVES=none-baseline CONFIG=configs/r2mu-author-ds8.yaml bash scripts/generate_all.sh DeepSeek-R1-Distill-Llama-8B
MODELS_DIR=models/unlearned GPU_MEM=80 GPU_UTIL=0.92 SHARDS=4 WAVES=none-baseline CONFIG=configs/r2mu-author-ds8.yaml bash scripts/generate_all.sh r2mu-author-ds8-bio
```

After generation:

```bash
python scripts/coverage.py --run r2mu-author-ds8 --models DeepSeek-R1-Distill-Llama-8B r2mu-author-ds8-bio --waves none-baseline && python scripts/score_accuracy.py --run r2mu-author-ds8 --split val --baseline-model DeepSeek-R1-Distill-Llama-8B
```

Inspect loss finiteness, selected-weight changes, WMDP bio accuracy, parse/format
rates, cyber control and retained subjects before claiming success. Review raw
reasoning traces too; this workflow does not certify the paper's trace-leakage metric.

## Then: Gemma 12B adaptation (submit after reviewing the DeepSeek reference)

Reuse your already downloaded Gemma12 weights and authorized bio corpus. Prepare
2,400 native traces to allow filtering; training refuses fewer than2,000 usable
traces rather than silently running fewer updates. Preparation is resumable; its
raw prompts, outputs, token IDs and fingerprints are retained. It uses our existing
512-token corpus segments and native thinking generation (.6 temperature/.95 top-p,
2048 output tokens). This is an explicit data adaptation, not author trace reproduction.

Only the architecture-dependent access changes: nested hidden size and decoder
blocks, named down-projection weights, layer scaled by depth (7/32), Gemma reasoning
delimiters. The actual upstream training loop remains in use, with the same loss
weights, learning rate, batch and token caps as the DeepSeek reference.

```bash
qsub -l gpus=1,gpu_memory=80G,h_rt=48:00:00 -N r2mu_author_g12_traces -v MODE=traces scripts/qsub_r2mu_author.sh
qsub -hold_jid r2mu_author_g12_traces -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N r2mu_author_g12_bio -v VARIANT=g12 scripts/qsub_r2mu_author.sh
```

After final export:

```bash
MODELS_DIR=models GPU_MEM=80 GPU_UTIL=0.92 SHARDS=4 WAVES=none-baseline CONFIG=configs/r2mu-author-g12.yaml bash scripts/generate_all.sh gemma-4-12B-it
MODELS_DIR=models/unlearned GPU_MEM=80 GPU_UTIL=0.92 SHARDS=4 WAVES=none-baseline CONFIG=configs/r2mu-author-g12.yaml bash scripts/generate_all.sh r2mu-author-g12-bio
```

After generation:

```bash
python scripts/coverage.py --run r2mu-author-g12 --models gemma-4-12B-it r2mu-author-g12-bio --waves none-baseline && python scripts/score_accuracy.py --run r2mu-author-g12 --split val --baseline-model gemma-4-12B-it
```

No smoke jobs. GPU memory/runtime and scientific success require the server runs.
Sources: https://github.com/OPTML-Group/Unlearn-R2MU and https://arxiv.org/abs/2506.12963
