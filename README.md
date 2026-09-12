# Prompt-based suppression

Executable experiments measuring language-model behavior under concept-suppression prompts.
The prompting experiments use frozen target models. RMU is a separate weight-editing comparison;
the ELM interface imports an externally produced checkpoint. Prefix tuning is not implemented.

## Installation

Use Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.lock
pip install --no-deps -e .
python -m pytest -q
```

The test lock records the CPU validation environment, including DSPy/Optuna. GPU execution
requires a separate Linux/CUDA environment:

```bash
pip install -e '.[gpu,analysis,opt,dev]'
```

The GPU versions in `pyproject.toml` are the serving configuration; GPU execution has not been
validated by this repository's CPU CI. Record `pip freeze`, GPU details, model revisions, and
the repository commit with each run. Targets are not quantized. Judge quantization is optional.

## Running an experiment

`experiments.json` describes the supported generation grids. The default action prints the
commands without executing them; `--execute` runs them sequentially on the available hardware.
There is no scheduler submission or GPU allocation in this repository.

```bash
python scripts/run_experiment.py wmdp5 --stage build
python scripts/run_experiment.py wmdp5 --stage build --execute
python scripts/run_experiment.py wmdp5 --stage generate --model Meta-Llama-3-8B-Instruct
python scripts/run_experiment.py wmdp5 --stage generate --model Meta-Llama-3-8B-Instruct --execute
python scripts/run_experiment.py wmdp5 --stage score --execute
```

For generation, download each selected model into `models/<model-name>` or pass `--models-dir`.
The Hugging Face identifiers are in `scripts/models.txt`. Respect dataset/model access conditions.
For a large target pass `--tp N` after allocating N GPUs; `--gpu-util` sets the serving memory
fraction. Inspect a new model with `scripts/check_think.py` before a reasoning-enabled sweep.

The build stage downloads data as needed. WMDP builds include the complete selected WMDP subsets,
up to 100 adjacent MMLU items per subject, and 15 utility items per subject. The current utility
mapping is defined in `src/data/retain.py`. The split generator uses seed 0 and 25% validation
for forget items. Other tiers are not split.

Original server-side split files and dataset/model revision manifests are not bundled. These
recipes create fresh runs using the current loaders; they do not promise exact reproduction of
historical results. Preserve `items.jsonl`, its checksum, source dataset revisions, and `split.json`
with a completed run. Do not overwrite a frozen split to make results agree.

Useful direct commands:

```bash
python scripts/coverage.py --run wmdp5 --config configs/wmdp5.yaml --models Meta-Llama-3-8B-Instruct
python scripts/check_surfaces.py 'results/wmdp5/generations/*.jsonl' --examples 0
python scripts/present_tables.py --run wmdp5
```

Consult each command's `--help` for options. Build and generation stages must precede scoring.
`present_tables.py` reads the whole run; use the scored `analysis/test/` CSVs for held-out results.
The generation recipe includes all listed prompt waves; selecting one model does not select a
statistically representative subset of the study.

## Optional evaluation and optimization

The rating judge and primitive judge are different instruments. The rating judge scores concept
presence and framing; the primitive judge supplies independent labels for derived mode statistics.
Judge agreement must be evaluated separately from output-format validity.

```bash
python scripts/judge.py --run wmdp5 --judge Qwen3.6-27B --quant bitsandbytes --stage concept
python scripts/judge_primitives.py --run wmdp5 --judge Qwen3.6-27B --quant bitsandbytes
```

DSPy connects to an already running OpenAI-compatible local model server. It does not start a
server or allocate GPUs. The model and reasoning setting must match the baseline generation.

```bash
python scripts/optimize_prompt.py --run wmdp5 --round 1 --model Meta-Llama-3-8B-Instruct \
  --concept wmdp-cyber --think-state na --seed-prompt scoped-directed@specific \
  --objective A --max-tokens 8192 --dry-run
```

Remove `--dry-run` to perform a search. Use a new `--round` for every search: the inherited optimizer
currently banks its seed before checking the overwrite guard. A searched prompt must be registered
and evaluated on held-out items before reporting its performance. Historical search grids are not
reconstructed by `experiments.json`.

For RMU, use `scripts/unlearn.py --help` to specify the base model, concept, layer, hyperparameters,
and a new checkpoint name. Bio requires an explicitly supplied forget corpus. Evaluate the edited
and unedited model on the same `rmu1` item set, then pass `--baseline-model` to `score_accuracy.py`.
ELM requires `--from-checkpoint` and its base model; no ELM optimization is implemented here.

## Outputs and interpretation

- Inputs: `data/processed/<run>/items.jsonl` and `split.json`.
- Generations, judgments, metrics, and figures: `results/<run>/` (ignored by Git).
- Raw outputs may contain model reasoning. No real generated outputs are committed here.
- Accuracy over parsed responses and accuracy counting missing answers as wrong are distinct.
  Paired delta tables currently use items parsed in both conditions. Report coverage alongside them.
- Lexical matches are vocabulary proxies, not validated semantic knowledge judgments.
- Keep concepts, prompt conditions, and reasoning settings separate when comparing results.

Known inherited limitations: interrupted JSONL tails require repair before appending; few-shot
demonstrations must be frozen before changing item selection or sharding. The supplied recipes
do not shard few-shot waves. A clean export does not resolve these scientific/execution limitations.

## Verification and provenance

```bash
make test
make smoke
make clean-smoke
```

`make smoke` uses synthetic inputs, a fake generator, and a fake judge. It verifies plumbing only.
The test suite exercises scoring, prompts, and mocked execution without model downloads. Scheduler
tests are excluded because scheduler launchers are not distributed. `.export-state.json` records
the source revision and exported file hashes without importing source Git history.

This repository is a one-way code export. Local edits stop synchronization instead of being
overwritten. Updates are validated locally before automatic publication from the source checkout. The export tool
and machine-specific hook configuration are maintained outside this repository.

## Attribution

- RWKU: `jinzhuoran/RWKU`; MMLU: `cais/mmlu`; WMDP: `cais/wmdp` on Hugging Face.
- Thaker et al., *Guardrail Baselines for Unlearning in LLMs*, arXiv:2403.03329.
- Li, Phuong and Siegel, *LLMs Can Covertly Sandbag on Capability Evaluations Against
  Chain-of-Thought Monitoring*, arXiv:2508.00943, Appendix A.
- RMU: Li et al., *The WMDP Benchmark*, arXiv:2403.03218.
- ELM: Gandikota et al., *Erasing Conceptual Knowledge from Language Models*, arXiv:2410.02760.
- The concept judge follows the AxBench concept-scoring instrument, arXiv:2501.17148, Appendix J.3.

No project-wide redistribution license has been selected. Referenced models, data, and methods
retain their own terms and attribution requirements.
