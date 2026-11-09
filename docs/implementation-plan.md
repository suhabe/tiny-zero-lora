# Educational scripts implementation plan

Goal: teach students who know PPO/GRPO how the existing Countdown LLM code
implements them, using runnable scripts that call shared production functions.

Approach: numbered Python lessons plus a reading guide. Extract the existing
pure task functions to `countdown.py`, re-export them from `train.py`, and expose
data/trainer factories without changing the default experiment. Avoid a parallel
training implementation or a notebook UI dependency.

- [x] Extract task functions; test actual scoring and prompt continuation contracts.
- [x] Expose data/trainer setup; retain all existing default GRPO/LoRA values.
- [x] Add dataset, tokenizer, reward, config and group-advantage exercises.
- [x] Add a persistent held-out snapshot, baseline/final evaluation and one-step/full
  training commands, with real rollout/metric artifacts and checkpoint resume.
- [x] Explain every explicit setting, inherited algorithm defaults, metrics,
  performance tradeoffs, and promising/failing run signatures.
- [x] Run offline tests and scripts; compile everything and review GPU paths against
  pinned library source. Report GPU execution as unverified if CUDA is unavailable.

Constraints: Python 3.12; original dependency pins; no fabricated model results;
no cloud GPU launch. The selected validation split is reused for monitoring and
final comparison, so it must not be described as an untouched test set.

Review focus: prompt's existing `<think>` prefix; completion/prompt token slicing;
duplicate task leakage; complete optimizer/scheduler checkpoints after one step;
identical held-out rows and generation policy for before/after comparisons.
