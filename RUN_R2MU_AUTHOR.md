# Run the R²MU author-code reference and Gemma 12B adaptation

Run all commands from this repository's root. Start with DeepSeek-R1-Distill-Llama-8B;
review its training and evaluation results before starting Gemma 12B. This guide uses
Python entry points included in this repository, without the original project's SCC
helper scripts. Request GPUs through your own scheduler before running GPU commands.

## What code runs

`scripts/train_r2mu_author.py` executes the authors' `run_rmu` function from
[OPTML-Group/Unlearn-R2MU](https://github.com/OPTML-Group/Unlearn-R2MU), pinned to
`3529e557a9c5d72ef53b7ee2b677a3aa8d2c8b06`. Source files and released traces are
SHA256-verified. This is separate from our R²MU reimplementation in
`src/unlearn/r2mu.py`, including its `-paper-*` hyperparameter variants.

The DeepSeek run retains the upstream objective, backward pass and optimizer step:
500 updates, batch size 4, learning rate 7.5e-5, alpha 1.4, steering coefficient 6.5,
assistant-loss weight 1, seed 42, layer 7, and parameter index 6 in layers 5–7.
It retains the model's shipped precision and checks the selected parameter names.

Documented repairs replace private paths and W&B logging, place one model copy on
each GPU, move inputs to the corresponding GPU, and fix an undefined `assistant_msg`
in the upstream LIMO loader. Evaluation runs separately through our WMDP8 pipeline.

The authors did not release the private forget-data file referenced by their script.
Their released 2,000 traces lack source IDs and complete generation provenance.
We substitute our available bio forget corpus. This is an author-code reference,
not an exact reproduction of paper numbers or a verified contamination-free evaluation.
See [R2MU_AUTHOR.md](R2MU_AUTHOR.md) for additional implementation details; use the
portable commands below when working from this advisor repository.

## 1. Environment and required data

Start from a fresh clone in any directory you own. No files from the researcher's
computer or cluster are needed. Repository access is required if it is private.
Use Linux, Python 3.12, and NVIDIA drivers compatible with the pinned GPU stack.
Allow several hundred GB for models, intermediate checkpoints and outputs.

```bash
git clone https://github.com/AS-XS/prompt-based-suppression.git
cd prompt-based-suppression
```

Create the environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[gpu,analysis]' accelerate sentencepiece
```

For an existing compatible environment, activate it and skip installation. Dependencies
are specified in `pyproject.toml`; local CPU checks do not establish CUDA compatibility.
Put Hugging Face caches on a filesystem with enough space if your home quota is small.

In each new terminal/job, return to your clone and activate `.venv/bin/activate`.
Use your cluster's allocation mechanism before GPU commands; CPU preparation and
analysis need no GPU. No particular username, scheduler or absolute path is assumed.

### Download the training corpus

Request access and accept the terms on
[the bio forget dataset page](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus).
Authenticate with the approved account:

```bash
hf auth login
```

On a network-enabled machine, download and convert all training shards, once:

```bash
python - <<'PYDATA'
import json
from pathlib import Path
from datasets import load_dataset

dst = Path("data/raw/bio-forget-corpus.jsonl")
if dst.exists():
    raise SystemExit(f"Already exists: {dst}; keep it rather than overwrite it.")
ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
if "text" not in ds.column_names:
    raise ValueError(f"Expected text column, got {ds.column_names}")
texts = [r["text"] for r in ds if isinstance(r["text"], str) and r["text"].strip()]
if len(texts) < 2400:
    raise ValueError(f"Need at least 2400 nonempty documents; got {len(texts)}")
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("x", encoding="utf-8") as f:
    for text in texts:
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
print(f"Wrote {len(texts)} documents to {dst}")
PYDATA
```

Authentication does not itself grant dataset access; approval is an external
prerequisite. This training corpus is separate from the evaluation questions.

### Build evaluation inputs from public data

Run once before setup/generation:

```bash
python -m src.data.build --run wmdp8 --wmdp bio cyber
python scripts/build_split.py --run wmdp8
```

The builder downloads `cais/wmdp` and MMLU subjects from `cais/mmlu`. It writes
`data/processed/wmdp8/items.jsonl` with bio/cyber questions, adjacent MMLU subjects
and broad utility controls. The split command writes `data/processed/wmdp8/split.json`:
seed 0, 25% validation / 75% test for forget items. Retain items are shared between
reports, not independent held-out retention evidence. Preserve both files for all
baseline/checkpoint comparisons; do not rebuild after generation starts.

This makes a standalone experiment using the available dataset revisions, not a
certified byte-identical copy of historical project data. Archive these inputs and
dataset cache revisions with your results. Record the environment and code:

```bash
mkdir -p results/r2mu-author-training
pip freeze > results/r2mu-author-training/environment.txt
git rev-parse HEAD > results/r2mu-author-training/checkout.txt
```

## 2. DeepSeek: download and setup (network-enabled host)

```bash
hf download deepseek-ai/DeepSeek-R1-Distill-Llama-8B --local-dir models/DeepSeek-R1-Distill-Llama-8B
python scripts/setup_r2mu_author.py
```

Setup downloads the pinned author files into `data/external/r2mu-author/`, caches
WikiText-2 raw test and GAIR/LIMO, and copies the frozen evaluation inputs into the
two separate `r2mu-author-*` namespaces. Repeated setup verifies existing files.
Do this before moving to an offline compute node.

## 3. DeepSeek: full training (two allocated GPUs)

Request **two GPUs with at least 80 GB each and a 48-hour wall-clock limit**.
Memory fit and runtime still require server validation. Activate the environment in
the allocation; retain the scheduler's `CUDA_VISIBLE_DEVICES` assignment.

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train_r2mu_author.py --variant ds8 --models-dir models
```

On SCC, you can submit the same training through the included wrapper instead
of running the Python command interactively. Set `SCC_PROJECT` to your own authorized
allocation (the script's historical default is `cs505am`; override it):

```bash
mkdir -p logs
qsub -P "${SCC_PROJECT:?Set SCC_PROJECT to your authorized SCC allocation}" -l gpus=2,gpu_memory=80G,h_rt=48:00:00 -N r2mu_author_ds8_bio -v VARIANT=ds8 scripts/qsub_r2mu_author.sh
```

The wrapper activates `.venv/` (or `venv/`, or an explicit `VENV_DIR`) and works
without `scripts/env.sh`. Errors include the failing command and line in
`logs/r2mu_author_ds8_bio.o<job-id>`. It preserves default cache locations; if you
used custom HF cache variables for setup, export the same variables in the job
(for example with qsub `-v HF_HOME` when HF_HOME is exported). Do not switch caches
between online setup and offline training.

This runs all 500 updates, with no smoke stage. Progress and losses are printed and
saved to `results/r2mu-author-training/r2mu-author-ds8-bio/losses.jsonl`.
The trainer also writes `request.json` with configuration and provenance.

A successful final export prints `R2MU author checkpoint saved:` and creates:

```text
models/unlearned/r2mu-author-ds8-bio/unlearn_report.json
```

The report includes selected-parameter changed-element counts. Zero changes means
training did not change those weights at stored precision; do not call it successful
unlearning. Finite losses and a final export alone also do not establish unlearning.

The author loop saves intermediate weights at 200 and 400 updates, but has **no
optimizer resume**. A scheduler kill can leave intermediate weights without a final
report. Inspect those artifacts before restarting. Existing run directories are
refused; an explicitly chosen fresh retry can use `--name r2mu-author-ds8-bio-retry1`.
Use that name in later generation commands too. No existing outputs are overwritten.

## 4. DeepSeek: generation and validation

On one allocated GPU with at least 80 GB, run these sequentially. They load the
unedited and edited models separately, so the training allocation is not required.

```bash
python scripts/generate.py --config configs/r2mu-author-ds8.yaml --model DeepSeek-R1-Distill-Llama-8B --models-dir models --conditions baseline --gpu-util 0.92
python scripts/generate.py --config configs/r2mu-author-ds8.yaml --model r2mu-author-ds8-bio --models-dir models/unlearned --conditions baseline --gpu-util 0.92
```

For shorter allocations, add `--shard 1/4` and run all four shard indices (1/4 through
4/4), each in its own allocation, for each model. Choose either sharded or unsharded
output from the outset; do not mix both layouts for one model/run. Repeating the same
generation command resumes saved rows. Runtime depends on generated trace lengths.

After generation, these CPU commands check full coverage before scoring:

```bash
python scripts/coverage.py --run r2mu-author-ds8 --models DeepSeek-R1-Distill-Llama-8B r2mu-author-ds8-bio --waves none-baseline && python scripts/score_accuracy.py --run r2mu-author-ds8 --split val --baseline-model DeepSeek-R1-Distill-Llama-8B
```

Read `results/r2mu-author-ds8/analysis/val/accuracy.csv`, `accuracy_delta.csv`, and
`side_effect.csv`. Check bio accuracy, parsing/format rates, cyber control, and retained
subjects together. Review saved reasoning traces for leakage; these commands do not
reproduce the paper's API-based trace evaluation. Keep test results for a fixed final
comparison after selecting settings on validation.

## 5. Gemma 12B: prepare native traces, then train

Proceed after reviewing the DeepSeek reference. Download the model only if missing;
complete any Hugging Face access requirements before downloading.

```bash
hf download google/gemma-4-12B-it --local-dir models/gemma-4-12B-it
```

On one allocated GPU with at least 80 GB, prepare native thinking traces:

```bash
python scripts/prepare_r2mu.py --model gemma-4-12B-it --models-dir models --concept wmdp-bio --forget-file data/raw/bio-forget-corpus.jsonl --out data/processed/r2mu-author-training/g12-bio --samples 2400
```

Preparation is resumable. It retains raw prompts, outputs, token IDs and provenance.
It uses our 512-token corpus segments, temperature 0.6, top-p 0.95 and a 2,048-token
output cap. Training requires at least 2,000 usable native traces and refuses a
silently shortened run.

After preparation completes, request **two GPUs with at least 80 GB each for 48 hours**:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train_r2mu_author.py --variant g12 --models-dir models --bundle data/processed/r2mu-author-training/g12-bio
```

The adaptation uses the upstream loop with Gemma's nested model layout, named down
projections, depth-scaled layer selection and native reasoning delimiters. Training
hyperparameters and loss/token caps match the DeepSeek reference. The final report is:

```text
models/unlearned/r2mu-author-g12-bio/unlearn_report.json
```

## 6. Gemma 12B: generation and validation

On one allocated GPU with at least 80 GB, after final export:

```bash
python scripts/generate.py --config configs/r2mu-author-g12.yaml --model gemma-4-12B-it --models-dir models --conditions baseline --gpu-util 0.92
python scripts/generate.py --config configs/r2mu-author-g12.yaml --model r2mu-author-g12-bio --models-dir models/unlearned --conditions baseline --gpu-util 0.92
```

The same optional four-shard procedure applies. Then score on CPU:

```bash
python scripts/coverage.py --run r2mu-author-g12 --models gemma-4-12B-it r2mu-author-g12-bio --waves none-baseline && python scripts/score_accuracy.py --run r2mu-author-g12 --split val --baseline-model gemma-4-12B-it
```

Results are in `results/r2mu-author-g12/analysis/val/`. This is a Gemma adaptation,
not a model/configuration evaluated by the original R²MU paper. Compare its own
unedited baseline and retain the separate reference/adaptation namespaces.

## 7. Interpret analysis and report the final comparison

| File | What to inspect |
| --- | --- |
| `accuracy.csv` | Subject accuracy and parsing rate for edited and original models, matched by thinking state. |
| `accuracy_delta.csv` | Paired changes relative to the explicitly supplied original model. |
| `side_effect.csv` | Retained-subject performance; separate adjacent knowledge from general utility. |

A useful bio edit moves accuracy toward four-choice chance (25%) while preserving
parsing and retained performance. Collapse in parsing or general ability is not
selective unlearning. Cyber is a control here: this guide trains bio edits only.
Inspect raw outputs under `results/<run>/generations/` for correct reasoning with
incorrect final letters. These tables do not establish knowledge removal or reproduce
the paper's trace-leakage judge.

Print the validation tables on CPU:

```bash
python - <<'PYTABLES'
from pathlib import Path
import pandas as pd
for run in ("r2mu-author-ds8", "r2mu-author-g12"):
    folder = Path("results") / run / "analysis" / "val"
    for name in ("accuracy.csv", "accuracy_delta.csv", "side_effect.csv"):
        path = folder / name
        if path.exists():
            print(f"\n{path}")
            print(pd.read_csv(path).to_string(index=False))
PYTABLES
```

Once validation fixes your checkpoint/settings, score the same saved generations
on test. No additional GPU generation is needed:

```bash
python scripts/score_accuracy.py --run r2mu-author-ds8 --split test --baseline-model DeepSeek-R1-Distill-Llama-8B
python scripts/score_accuracy.py --run r2mu-author-g12 --split test --baseline-model gemma-4-12B-it
```

Final tables are in `results/r2mu-author-ds8/analysis/test/` and
`results/r2mu-author-g12/analysis/test/`. Do not choose checkpoints by their test
results. Preserve checkpoint reports, training losses, raw generations, frozen
items/splits and analysis CSVs together. GPU execution and scientific effectiveness
remain to be validated; this guide does not claim a successful training run.
