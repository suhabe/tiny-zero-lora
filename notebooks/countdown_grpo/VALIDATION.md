# Validation and execution status

Verified on Python 3.12.14, with marimo 0.25.2 and Altair 6.3.0.

## Completed checks

- `python -m unittest -v test_lab`: all four contract tests passed, covering reward
  edge cases, exact prompt-function options, source/preset integrity, group signal,
  masked-token update gradients, immutable run manifests and partial JSONL reads.
- `python -m py_compile`: notebook, helper, worker, explanations and upstream source
  compile successfully.
- `python -m marimo check countdown_grpo.py`: no findings after fixes.
- `python -m marimo export html ... --no-include-code`: exit 0. The saved HTML
  contains 33 executed cells and zero error outputs.
- Programmatic `app.run()`: 33 cells executed; expected reward totals were
  `[2, 1, 1, 1, 2, 1, 0]`; no background worker launched automatically.
- Tiny token-policy gradient exercise: analytic versus central finite-difference
  maximum absolute error approximately `9.36e-13`; actual surrogate objective
  changed from `-0.2226536163` to `-0.2263716670` after one gradient step.
  This is a mechanics exercise, not an LLM training result.
- Programmatic result-panel/chart verification with explicitly labeled unit-test
  fixtures exercised the populated baseline/final and live-metric branches and
  validated the Altair schema. Fixtures were held in memory only and are not
  included as model results or preloaded chart history.
- A real background **preflight** process completed successfully, wrote flushed
  `job_started → preflight → job_finished` JSONL records, and the notebook's reader
  consumed them. This checks process/logging plumbing, not GPU training.
- The bundled `upstream_train.py` has the exact Git blob SHA
  `bbd35a87e896ca5ac34ed53b88a7106eaa6d5fab`, matching the inspected repository.
- Important defaults/implementation details were checked against TRL 1.17.1
  `grpo_config.py` / `grpo_trainer.py` and Transformers 5.19.0 trainer source,
  including the checkpoint API used for complete terminal saves.

## Not executed here

No CUDA GPU, PyTorch, Transformers, TRL, datasets, PEFT, Accelerate or vLLM was
installed in the authoring environment. The actual preflight is included at
`validation/preflight.json`.

Therefore full dataset preparation, actual Qwen-tokenizer rendering, initial-model
baseline, GPU one-step training, checkpoint resume on a GPU, the complete training
run and final model evaluation have **not** been executed here. Their code paths
are included and source-reviewed, but end-to-end GPU compatibility, memory fit,
throughput and learning quality remain unverified. The notebook labels these
panels as pending until the student runs the corresponding stages.

Browser screenshot inspection could not be completed: the local headless browser
was absent and its download failed. Headless notebook execution and HTML export
were verified; no screenshot-based layout validation is claimed.

There are no fabricated baseline/final scores, training curves, or trained model
weights in this archive. The original 3B/vLLM preset and smaller classroom preset
are separate experiments, and no accuracy improvement is promised for either.
