# Run every lesson on Modal

Run these commands from the repository root. Your local machine needs Python and
the Modal client; the pinned ML stack, model downloads and GPU work run remotely.

```bash
python -m pip install 'modal==1.6.3'
modal setup
```

`modal_lessons.py` reuses the CUDA image and `llm-post-training` volume from
`modal_train.py`. The first invocation builds the image; subsequent commands can
reuse cached layers. Commands stream the original lesson's output to your terminal.
No deployment is needed; each `modal run` invokes the named function directly.

## One command for each lesson

Run the commands in order and wait for each to finish. Defaults share one
experiment called `tutorial`. The reading/data lessons use CPU containers;
preflight and model-execution lessons request the original B300 GPU resources.

| Topic | Command | Compute |
|---|---|---|
| Check the remote environment | `modal run modal_lessons.py::preflight` | B300; checks CUDA/BF16, no model run |
| 1. Dataset | `modal run modal_lessons.py::dataset` | CPU |
| 2. Prompt generation | `modal run modal_lessons.py::prompts` | CPU |
| 3. Reward functions | `modal run modal_lessons.py::rewards` | CPU |
| 4. Initial-model baseline | `modal run modal_lessons.py::baseline` | B300 |
| 5. Training code and settings | `modal run modal_lessons.py::configuration` | CPU |
| 6. One optimizer step | `modal run modal_lessons.py::step` | B300 |
| 7. Inspect actual metrics | `modal run modal_lessons.py::metrics` | CPU |
| 8. Performance explanation | `modal run modal_lessons.py::performance` | CPU; explanatory text, not a benchmark |
| 9. Continue training | `modal run modal_lessons.py::training` | B300 |
| 10. Final saved-adapter evaluation | `modal run modal_lessons.py::final_evaluation` | B300 |

`configuration` prints the actual source; the full parameter-by-parameter
explanations and metric dictionary remain in [the walkthrough](README.md).
`performance` prints that guide's performance section directly.

## Persistent state between commands

All commands use `/data/runs/tutorial` on the named volume, including the prepared
dataset/tokenizer, `baseline.json`, `training/checkpoint-N/`, metrics, rollout traces,
`adapter/` and `final.json`. The Hub cache also persists at `/data/huggingface`.
Each invocation reloads the volume before selecting its inputs and commits it
after its child process exits, including Python/subprocess failures. The lesson
itself runs in a fresh subprocess, so model allocations and CLI arguments do not
carry over to the next invocation.

Use `--run-name <name>` on stateful commands to create a separate experiment.
Names can contain letters, digits, `_`, `.` and `-`, starting with a letter/digit.
Use the same name throughout an experiment. Reading-only `preflight`, `rewards`,
`configuration` and `performance` do not need a run name.
Do not run two stateful commands for the same experiment simultaneously.

`step` and `training` default to `--resume latest`. They select the numerically
highest checkpoint containing trainer, optimizer, scheduler and RNG files;
incomplete checkpoint directories are ignored when a complete one exists.
With no checkpoints, they start from the initial policy. An existing incomplete
run with no complete checkpoint requires inspection before proceeding.

`--max-steps 0` (the Modal default) reuses the saved training horizon, or uses
2,000 updates for a new experiment. One-step execution still preserves that full
schedule. `training` continues to that **total** number of updates, not that many
additional ones. The underlying lesson checks package/config compatibility.

To select a particular checkpoint:

```bash
modal run modal_lessons.py::step --resume checkpoint-10
modal run modal_lessons.py::training --resume checkpoint-11
```

Only complete checkpoints still present on the volume can be selected; the
original retention setting keeps one rotating numbered checkpoint. `--resume none`
requests a fresh start, and the underlying lesson rejects an existing checkpointed
run rather than overwriting it. Use a new run name for a new experiment.

## A shorter first experiment

This changes the training horizon to 50 updates and evaluates 20 rows to check
the workflow. It is a smaller experiment, not evidence that quality will improve.

```bash
modal run modal_lessons.py::dataset --run-name quick
modal run modal_lessons.py::baseline --run-name quick --limit 20
modal run modal_lessons.py::step --run-name quick --max-steps 50
modal run modal_lessons.py::metrics --run-name quick
modal run modal_lessons.py::training --run-name quick
modal run modal_lessons.py::final_evaluation --run-name quick
```

Final evaluation defaults to the baseline's saved row count, so you do not need
to repeat `--limit 20`. An explicit conflicting limit is rejected by the
evaluation protocol check. Baseline/final JSON files are not overwritten.
For an extra evaluation after a step, choose a distinct result name:

```bash
modal run modal_lessons.py::final_evaluation --run-name quick --name after_step
```

For a long training invocation, `modal run --detach modal_lessons.py::training`
allows the remote job to continue after the local connection ends. Wait for that
job to finish before starting metrics/final commands for the same run.

## Download the results

Volume paths in the CLI omit the container's `/data` mount prefix:

```bash
modal volume ls llm-post-training /runs/tutorial
modal volume get llm-post-training /runs/tutorial ./modal-results
```

Downloading a directory includes its contents recursively. You can instead get
an individual result, e.g. `/runs/tutorial/final.json`.
These educational entrypoints save runs on the volume; the existing
`modal_train.py::train` entrypoint retains its original behavior.

## Execution status

The Modal 1.6.3 client imports these functions and generates CLI help for all
entrypoints successfully. Local tests cover all command mappings, persistent
paths, checkpoint/horizon selection, baseline-size reuse, invalid paths and
commit-on-failure behavior. Actual Modal image builds, cloud invocations and
GPU learning outcomes have not been executed in the authoring environment.

API references: [function CLI](https://modal.com/docs/cli/latest/run),
[volume commits/reloads](https://modal.com/docs/guide/volumes),
[source files in images](https://modal.com/docs/reference/modal.Image#add_local_dir),
[volume downloads](https://modal.com/docs/cli/latest/volume).
