# Execution status

Authoring environment: Python 3.12.14, no CUDA device, and no installed torch,
Transformers, datasets, TRL, PEFT, Accelerate or vLLM. Lesson 00 verifies these
facts without launching a model.

Executed successfully:

- `python -m unittest discover -s tests -v`: eight offline contract tests, covering
  real reward behavior, prompt continuation arguments, group-normalized advantages,
  reward lesson CLI execution, paired evaluation identity, fixed resume horizon
  explicit Hub revision pinning across the manifest/preparation boundary, and
  preservation of the original config record when a changed config is rejected.
- `python -m compileall -q train.py countdown.py modal_train.py education tests`.
- `python -m education.00_preflight`.
- `python -m education.03_rewards`: ten hand-built cases using shared production
  functions, plus mixed/constant group-advantage calculations.
- `python -m education.05_code_and_config`: reads and prints actual source.
- `python -m education.07_metrics`: correctly reports that no model metrics exist.

Prompt and reward function bodies were preserved during extraction. Original
explicit GRPO/LoRA settings remain unchanged for `python train.py`. Source review
used the repo's pinned TRL 1.17.1 and Transformers 5.19.0 implementations for
batching, reward normalization, loss normalization and callback/checkpoint behavior.

Not executed here: real Hub dataset preparation, actual Qwen tokenizer rendering,
initial-model evaluation, GPU optimizer update, checkpoint resume, full training
and final saved-adapter evaluation. Those commands require the configured model
environment and GPU. Their runtime compatibility, VRAM fit, throughput and
learning outcomes remain unverified; passing the offline tests does not establish
any of those. There are no fabricated model results or trained adapters in this PR.
