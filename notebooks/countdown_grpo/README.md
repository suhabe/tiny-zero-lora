# Countdown GRPO implementation lab

A ten-section marimo notebook for students who know Sutton & Barto, PPO and GRPO,
and want to connect those ideas to an LLM training implementation.

The walkthrough inspects `suhabe/llm-post-training/train.py` at commit
`594aa060a431a7edc464fd1202a679d23e9230f4`. The bundled upstream file is unchanged
and hash-checked. No GitHub repository changes are needed.

## Start with the offline notebook

Run the commands below from `notebooks/countdown_grpo/` and keep the Python files together. Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ui.txt
python -m marimo edit countdown_grpo.py
```

The light environment runs dataset-viewer examples, the exact source prompt
function with a message recorder, exact reward functions, an interactive reward
editor, and a finite-difference-checked token-policy update. It does not download
the full dataset/model. Static HTML exports provide a reading copy; interactive edits, stage launches and live chart polling require marimo.
Generate a static reading copy with the export command in the Verification section. The HTML viewer may need internet access to load marimo's frontend assets.

## Run the real model

Use a Linux machine with a BF16-capable NVIDIA CUDA GPU and compatible drivers.
The upstream Modal environment uses CUDA 13.1.1 and a B300 GPU. This project does
not provision a remote GPU or launch a paid cloud job. On an existing GPU machine,
install one training profile into the **same environment** that runs marimo:

```bash
# Smaller exercise: 0.5B model, 320 tasks, 50 updates, HF generation.
python -m pip install -r requirements-classroom.txt
python -m pip check
python -m marimo edit countdown_grpo.py
```

For the original 3B/vLLM stack instead:

```bash
python -m pip install -r requirements-gpu.txt
python -m pip check
python -m marimo edit countdown_grpo.py
```

The training pins are taken from the inspected repository, including PyTorch
2.13.0, Transformers 5.19.0 and TRL 1.17.1. Do not silently substitute a different
TRL version: defaults, batching and loss normalization have changed across releases.
CUDA/vLLM setup needs a compatible driver/toolchain; the repository's pinned
`modal_train.py` is the reference for its full CUDA image. The notebook worker
supports one process/GPU. No API tokens are embedded or required for the public
model/dataset under their current access settings.

In the notebook launcher, keep the same preset and run folder and submit:

1. **Prepare** — load the actual dataset/tokenizer, pin Hub revisions, map/filter,
   split, save rows/IDs and report task overlap. Needs internet and ML dependencies.
2. **Preflight** — report packages and CUDA/BF16 support.
3. **Baseline** — evaluate the initial instruction model, saving per-example results.
4. **Step** — run one additional optimizer update and save a complete checkpoint.
5. **Train** — resume that checkpoint to the fixed horizon.
6. **Final** — reload the saved adapter and evaluate the same held-out rows.

Wait for each worker to exit before submitting the next stage. The subprocess
keeps the notebook responsive; chart polling reads flushed JSONL lines. An empty
chart means no real metric has been recorded. Raw errors and console logs are
visible in the notebook. Changing settings requires a new run folder. Reopening
the notebook can read existing results; it does not resume training automatically.

Each Step uses the **full planned scheduler horizon**, then stops via a callback.
Train restores optimizer/scheduler/RNG state. Repeated Step submissions advance
one update each. Do not change `max_steps` between Step and Train.

## CLI alternative

Run from `notebooks/countdown_grpo/`, in the configured GPU environment:

```bash
python training_worker.py --stage prepare --preset classroom --run-dir runs/classroom
python training_worker.py --stage preflight --preset classroom --run-dir runs/classroom
python training_worker.py --stage baseline --preset classroom --run-dir runs/classroom
python training_worker.py --stage step --preset classroom --run-dir runs/classroom
python training_worker.py --stage train --preset classroom --run-dir runs/classroom
python training_worker.py --stage final --preset classroom --run-dir runs/classroom
```

Use `--preset repository --run-dir runs/repository` for the original model/budget.
The notebook can read a CLI run by selecting that preset/folder in its launcher;
submitting Preflight is a convenient read-only way to select an existing run.
Ctrl-C stops a foreground CLI worker. A notebook-launched subprocess continues
when the browser closes; its PID is displayed. Completed checkpoints survive
interruption; the unfinished update does not. Only one rotating checkpoint is kept.

## What is original and what the lab changes

The exact source functions generate prompts and rewards. `repository` preserves
all explicit GRPO/LoRA values and the data-processing order. Operational additions
are per-run output paths, a save interval of 10 rather than the inherited 500,
forced saves at one-step/terminal boundaries, explicit seed/data_seed 42, pinned
model/dataset revisions, audit row IDs, immutable manifests, JSONL logging and
per-completion evaluation capture. All differences are explained in the notebook.

`classroom` changes model size to 0.5B, dataset selection to 320 rows, max_steps to
50, microbatch to 2, generations to 4, completion limit to 256, eval batch to 4,
enables gradient checkpointing and disables vLLM. It is a smaller **different
experiment**, with no guarantee that 50 updates improve solve rate.

The notebook explicitly explains that TRL 1.17.1 defaults to DAPO token-normalized
loss rather than original sequence-normalized GRPO, and that effective batch size
counts completions, not distinct prompts. It covers every explicit source setting,
important inherited defaults, performance tradeoffs and the logged metrics.

The final evaluation reuses the validation split used for monitoring, so it is
not an untouched test result. Equivalent number permutations can cross the source
random split; an overlap audit is shown rather than silently changing the task.
Rates use exact per-example denominators, with Wilson intervals and paired
new-solves/regressions. These do not establish statistical significance of a gain.

## Files and outputs

| File | Purpose |
|---|---|
| `countdown_grpo.py` | Main reactive marimo notebook. |
| `lab_core.py` | Exact-function extraction, presets, numeric trace, process/result helpers. |
| `training_worker.py` | Real dataset preparation, GRPO training, callbacks, resume/evaluation. |
| `lesson_tables.py` | Configuration and metric explanations. |
| `upstream_train.py` | Unmodified inspected source; never starts automatically. |
| `requirements-*.txt` | Light UI, smaller GPU, original vLLM dependency profiles. |
| `test_lab.py` | Offline behavior and source-integrity checks. |
| `VALIDATION.md` | Actual authoring-time verification and execution limitations. |

Runs save `data.json`, `manifest.json`, materialized splits/tokenizer, resolved
GRPO configuration, `events.jsonl`, `console.log`, optimizer checkpoints,
`adapter/`, `baseline.json` and `final.json`. LoRA adapter files require the saved
base-model revision to reload; they are not a full base-model checkpoint.
`latest_rollouts.json` preserves the most recent training generation batch with
reward components and diagnostic group advantages for the one-step walkthrough.

## Verification

```bash
python -m unittest -v test_lab
python -m marimo check countdown_grpo.py
python -m marimo export html countdown_grpo.py -o local_preview.html --no-include-code
```

No Qwen training/evaluation was executed in the authoring environment: no CUDA
GPU or training stack was available. The lab contains no fabricated model
metrics or pretrained/trained model artifacts. See `VALIDATION.md` for the exact
checks that were run.
