# A Countdown task, from a row to a LoRA update

You already know the policy gradient and PPO/GRPO objective. These lessons show
how `train.py` turns those ideas into a language-model training run. Run commands
from the repository root with Python 3.12. Read the scripts alongside this guide;
they are short entrypoints, not a second trainer implementation.

To run each lesson remotely without installing the ML stack locally, use the
separate commands in **[Run every lesson on Modal](MODAL.md)**. They share a
persistent experiment directory and automatically resume complete checkpoints.

| Concept from RL | Concrete object here |
|---|---|
| Context / initial state | Tokenized chat prefix containing `nums` and `target` |
| Action | A vocabulary token sampled from the model |
| Trajectory | The newly generated completion tokens, ending at EOS or the length limit |
| Terminal reward | Two binary Python verifiers, summed with equal weight |
| Baseline for an advantage | Mean reward of responses to the same prompt |
| Policy parameters | Frozen Qwen base weights plus trainable LoRA matrices |
| PPO-style update | Ratios of token probabilities, clipping, and a reference-policy KL term |
| Optimizer step | Accumulated completion-token gradients followed by one LoRA update |

There is no learned critic, environment simulator, supervised reasoning trace or
reference answer string. The policy receives a prompt, samples a sequence and
receives a scalar reward from arithmetic and formatting checks.

## Setup and route through the lessons

First run the standard-Python exercises:

```bash
python -m education.00_preflight
python -m education.03_rewards
python -m education.05_code_and_config
python -m unittest discover -s tests -v
```

For actual data/tokenizer inspection and model runs, install `requirements.txt`
into a Python 3.12 environment on Linux with compatible NVIDIA CUDA drivers:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m education.00_preflight
```

The reference `modal_train.py` uses a B300 GPU, CUDA 13.1.1 development image,
PyTorch 2.13.0 and vLLM 0.32.0. BF16 support is required by this experiment.
The 3B model, training activations and colocated vLLM engine need substantial
VRAM. These scripts do not claim a smaller GPU will fit. Downloads require Hub
access. Do not change pinned TRL versions casually: batching, loss defaults and
metrics differ across releases. No additional notebook/UI dependency is needed.

Use the same `--run-dir` throughout. The default is `runs/tutorial`.
Use a new directory for a new experiment. Run one command at a time; the lesson
recorder supports one Python process and one GPU, without `torchrun`.

| Order | Command | Requires / produces |
|---|---|---|
| 1 | `python -m education.01_dataset` | Downloads real data/tokenizer; saves prepared splits |
| 2 | `python -m education.02_prompts` | Prints actual Qwen chat text, token IDs and attention masks |
| 3 | `python -m education.03_rewards` | Offline reward cases and a numerical advantage example |
| 4 | `python -m education.04_evaluate` | GPU; saves initial-policy `baseline.json`, including completions |
| 5 | `python -m education.05_code_and_config` | Offline source walkthrough; read the parameter tables below |
| 6 | `python -m education.06_train --one-step` | GPU; one real optimizer update and complete `checkpoint-1` |
| 7 | `python -m education.07_metrics` | Reads real JSONL metrics and all 16 responses in one group |
| 8 | Read performance section below | Explains where time/memory go and the actual optimizations |
| 9 | `python -m education.06_train --resume runs/tutorial/training/checkpoint-1` | GPU; continues the same run to 2,000 total updates |
| 10 | `python -m education.04_evaluate --adapter runs/tutorial/adapter` | GPU; reloads adapter, saves `final.json` and paired changes |

Baseline and final default to the entire saved validation split. For a quicker
smoke check, use `--limit 20` in **both** evaluation commands. Such a small sample
is useful for plumbing checks, not a reliable quality estimate. A fresh result
will not overwrite an existing JSON file; `--name after_step` allows an extra
evaluation without losing the baseline/final files.

## 1. Dataset: tasks rather than demonstrations

`Jiayi-Pan/Countdown-Tasks-3to4` supplies a `train` split whose relevant fields
are `nums` (three or four integers) and `target` (an integer). Lesson 01 prints
the current real schema, raw count and five actual rows, then calls
`train.prepare_data()` and prints processed examples from both splits.

For a **hand-built illustration**, `{"nums": [1, 2, 3], "target": 6}` admits
`1 + 2 + 3`. Another, `{"nums": [2, 2, 3], "target": 7}`, requires both copies
of 2. These are illustrations, not claims about which Hub rows you will see.

Preprocessing follows the original order exactly:

1. Load the Hub training split.
2. Shuffle with seed 42, then select at most 2,000 rows.
3. Load the model's tokenizer; left-pad and set PAD to EOS.
4. `map` each row through `generate_r1_prompt`; retain `nums` and `target`.
5. Drop rows whose rendered prompt exceeds 256 tokens, counting without adding
   extra special tokens. This filters rows; it does not truncate their text.
6. Split the surviving rows 90/10 with seed 42.

Therefore 1,800/200 is an upper bound on train/validation sizes, not a promise.
TRL uses `prompt` as policy input and forwards `nums`/`target` to reward functions.
It does not train cross entropy against `target`.

The lesson saves this `DatasetDict` and tokenizer once. Subsequent commands load
those same rows rather than drawing a new split. `data.json` records fingerprints,
sizes, model revision and equivalent-task overlap: `(sorted(nums), target)` can
occur in both splits even when source row IDs differ. The lesson reports this
without silently changing the repo experiment. A stricter generalization study
would deduplicate/group tasks before splitting. The held-out split used for
monitoring is a **validation set**, including when evaluated at the end.

## 2. Prompt generation: keep the assistant turn open

Read `countdown.generate_r1_prompt`, re-exported from `train.py`. It constructs
three messages: a system instruction, the Countdown user task, and an assistant
prefix `Let me solve this step by step.\n<think>`.

`apply_chat_template(tokenize=False, continue_final_message=True)` returns a
string using Qwen's real role/special tokens. It keeps the assistant message
unfinished. A generated response should begin with reasoning text, **after** the
opening `<think>` already supplied by the prompt. Adding another assistant turn
or another opening tag changes both generation and format scoring.

Lesson 02 renders three actual tasks, prints trailing token IDs/pieces and the
padding attention mask. Decoder-only batched generation uses left padding so
every row ends at its generation boundary. PAD=EOS provides a valid padding ID;
the attention mask still distinguishes padding positions from actual text.
Passing `add_special_tokens=False` avoids wrapping the already rendered chat
text in a second set of tokenizer tokens.

**Try:** change the sample numbers in a call to `generate_r1_prompt`, print the
result, and count tokens. Then temporarily omit `continue_final_message` in a
scratch call and observe the changed ending. Keep the production function intact.

## 3. Reward: exactly what is checked

Lesson 03 imports and calls the production functions directly. It exercises
correct answers, missing tags, wrong targets, reused operands, duplicate input
numbers, division by zero, exponentiation, an equals sign and a duplicated
`<think>` tag. It needs no ML packages or network.

| Component | Rules | What it does not establish |
|---|---|---|
| `format_reward_func` | Prepend the prompt's `<think>` to the completion; match thinking section, closing `</think>`, a newline and an `<answer>…</answer>` section | Truth or usefulness of the reasoning; arithmetic correctness |
| `equation_reward_func` | Extract the first answer; strip whitespace; extract digit runs and compare their multiset with `nums`; allow digits, `+ - * / ( )` and whitespace; reject `**` and `//`; evaluate with no builtins and require absolute error `< 1e-5` | Format correctness outside the answer; semantic quality of reasoning |

Each returns 0 or 1. TRL's inherited equal weights sum them: total reward is
0, 1 or 2. A correct expression with no thinking tags can earn 1; so can a
perfectly formatted wrong answer. Missing/extra/reused integers, decimals,
an equals sign, nonallowed characters or evaluation errors fail the equation
check. Parentheses and intermediate negative/fractional values are allowed.
Number matching counts literal occurrences, including duplicate inputs.

This is an inexpensive verifier with implementation-specific edge cases, not a
general mathematics oracle. The `<think>` text is not graded. Compare equation
and format rewards separately to avoid mistaking formatting progress for solving.

For one prompt with sampled totals `[0,1,1,2]`, group mean is 1 and the sample
standard deviation is `sqrt(2/3)`. With pinned TRL's default normalization,
`A_i = (R_i - mean(R)) / (std(R) + 1e-4)`, approximately
`[-1.22459, 0, 0, +1.22459]`. `[1,1,1,1]` and `[2,2,2,2]` both yield zero
reward advantages. The KL term can still contribute gradients when reward
advantages are zero. Lesson 03's small groups illustrate the arithmetic; real
training uses groups of 16.

## 4. Establish the initial-policy baseline

Run lesson 04 before updating weights. It loads the initial Qwen model at the
revision recorded with the tokenizer and samples one response for every saved
validation row. It saves completions and component rewards, exact numerator/
denominator rates, lengths, truncation and the decoding protocol in
`baseline.json`. Read several successes and failures, not just its summary.

Evaluation uses an explicit Transformers `GenerationConfig`: sampling at
temperature 1, top-p 1, top-k 0, up to 1,024 new tokens, batch size 1, and per-row
seed `42 + index`. This avoids inheriting Qwen's model-specific sampling filters.
It slices off all prompt tokens before decoding/scoring, so the example
`<answer>` embedded in the instruction cannot be rewarded by mistake.

This is **sampled single-response solve rate**, not greedy accuracy or pass@16.
It measures the instruction-tuned initial model, not an untrained network.
Periodic trainer evaluation also samples one response/task, but uses vLLM and
batched generation, so individual completions/seeds and aggregate estimators
need not match lesson 04 exactly. Before/after comparisons use lesson 04 on both
sides, with an identical saved protocol. Fixed seeds aid comparison but do not
guarantee identical numerical behavior across GPU kernels/hardware.

## 5. Walk through the training code

Lesson 05 prints the actual source of the three main entrypoints with line
locations. `prepare_data` owns preprocessing. `build_trainer` constructs the
original GRPO/LoRA settings and trainer. `main` trains, evaluates the terminal
policy if needed, and saves. Pure task functions live in `countdown.py` so reward
examples can import them without loading CUDA libraries; `train.py` re-exports
the same functions. Their scoring/prompt logic is unchanged.

At runtime TRL performs this bridge from equations to tensors:

1. Repeat each selected prompt 16 times and sample completion token IDs.
2. Decode only completions and call both reward functions, forwarding the task
   columns along with trainer metadata.
3. Sum reward components, group responses by prompt and compute scalar advantages.
4. Concatenate prompt and completion IDs, with attention masks; compute current,
   rollout/old and reference-policy token log-probabilities as needed.
5. Broadcast each sequence's scalar advantage across its generated-token positions.
   Prompt and post-EOS padding positions do not contribute to the policy loss.
6. Form the clipped token-probability-ratio surrogate plus `beta * KL`, accumulate
   gradients across two microbatches, clip gradient norm and update LoRA weights.
7. Advance scheduler, log metrics, periodically evaluate and checkpoint.

For the pinned TRL 1.17.1, the inherited `loss_type` is **`dapo`**, which divides
the summed masked token losses by active completion tokens in the global
accumulated batch. It is not the original GRPO averaging of per-sequence token
means. This difference changes length weighting. Other inherited choices include
`scale_rewards="group"`, sum-then-normalize reward aggregation,
`importance_sampling_level="token"`, `epsilon=0.2` and `num_iterations=1`.
There is no critic loss. A centered surrogate loss near zero can have a useful
gradient; minimizing its displayed value is not a quality objective.

The reference policy is the initial base policy. With LoRA, TRL can obtain its
log-probabilities with adapters disabled instead of keeping a second fully
trainable model. This differs from the rollout/old policy used in the PPO ratio.
vLLM rollout-versus-training log-probability correction is enabled by the pinned
library's defaults; it is separate from the PPO clipping ratio.

## 6. One real optimizer update and all settings

```bash
python -m education.06_train --one-step
python -m education.07_metrics
```

This calls `build_trainer()` and `trainer.train()`. A callback stops after one
additional **optimizer** update and requests the normal full Trainer checkpoint.
It does not set `max_steps=1`: doing so would compress a linear learning-rate
schedule to one update, making the exercise unlike the beginning of a long run.
The intended horizon remains 2,000. `checkpoint-1` includes LoRA weights,
optimizer, scheduler, RNG and trainer state. Resume that directory to continue.
`adapter/` alone is sufficient for inference but not an equivalent training resume.

The reward wrapper observes the real generation batch while returning the
original equation rewards unchanged. `step_1_rollouts.json` shows prompt,
completion, token length, both rewards, total and a diagnostic advantage for each
response. `latest_rollouts.json` updates during full training. The diagnostic
recomputes the pinned default normalization for inspection; TRL computes the
actual training tensors. No custom loss implementation replaces TRL.

### Data, model and token settings

The rationale column explains the role/tradeoff of the existing choices; these
are sensible starting settings, not claims that the values are optimal.

| Setting | Existing value and reason | If omitted or misconfigured |
|---|---|---|
| `model_id` | `Qwen/Qwen2.5-3B-Instruct`: instruction-following base with chat template and a manageable 3B scale on the reference GPU | Another model changes ability, template, memory needs and valid LoRA module names |
| Dataset | `Jiayi-Pan/Countdown-Tasks-3to4`: programmatically verifiable arithmetic contexts | Different fields/task break prompt/reward assumptions |
| `dataset_size` | 2,000 before filtering: bounds data preparation and task diversity | All rows increases work/diversity; too few makes repeated tasks/overfitting more likely |
| Shuffle/split seeds | 42: repeatable subset/order and 90/10 validation membership | Unseeded splits make before/after comparisons confounded |
| `max_prompt_length` | 256: filter overly long rendered prompts before splitting | Too small removes tasks; skipping the filter can exceed the 1,280-token vLLM context budget |
| `max_completion_length` | 1,024: allow reasoning while bounding generation/activation/KV memory | Too low cuts off answers and rewards; too high costs time/memory and can favor verbose rollouts |
| Left padding / PAD=EOS | Align decoder-only batch endings and provide a padding ID | Right-padding generation or missing attention masks can produce incorrect continuations; missing PAD can prevent batching |
| `continue_final_message` | True: continue the supplied assistant prefix | Closed/reopened turn or duplicate tags changes task distribution/scoring |

### Every explicit GRPO/Trainer setting

| Parameter | Value and why it is set | What removal or a bad value changes |
|---|---|---|
| `output_dir` | `qwen-r1-aha-moment`; lessons use `<run-dir>/training` to keep outputs together | Reusing directories confuses checkpoints/experiments; unwritable paths prevent saving |
| `model_init_kwargs.dtype` | `bfloat16`: lower parameter memory and suitable dynamic range | FP32 raises memory/compute cost; unsupported BF16 hardware fails or needs a deliberate alternative |
| `model_init_kwargs.attn_implementation` | `sdpa`: PyTorch optimized scaled dot-product attention | Fallback may materialize attention scores and cost more memory/time; backend choice still depends on shape/device |
| `learning_rate` | `2e-5`: conservative adapter update scale | Too high can destroy useful behavior/raise KL; too low gives little movement; omitting uses a different library default |
| `max_steps` | 2,000 optimizer updates: explicit budget | Counts updates, not samples; overrides epochs; too many may overfit, too few may show no change |
| `lr_scheduler_type` | `linear`: decay learning rate toward zero across that horizon | Different horizon/scheduler changes every update; changing max_steps on resume spoils the intended schedule |
| `num_train_epochs` | 1: explicit inherited Trainer setting, but inactive while max_steps is positive | Removing max_steps makes epochs the stopping rule; one epoch would be a much shorter/different run |
| `logging_steps` | 1: expose each update for this small teaching run | Coarser logs hide early failures; very frequent logs add modest overhead |
| `eval_strategy` | `steps`: monitor fixed update intervals | `no` hides held-out behavior; an unsupported value is rejected |
| `eval_steps` | 10: monitor frequently enough to see changes | Smaller costs more generation time; larger delays discovery of regressions |
| `eval_on_start` | True: trainer-side initial-policy measurement | Omitting loses its initial reference point; standalone lesson 04 still saves a separate paired baseline |
| `per_device_eval_batch_size` | 100: evaluate many tasks in parallel on the reference GPU | Too large OOMs; lower values reduce peak memory and may reduce throughput |
| `num_generations_eval` | 1: one response/task, cheap sampled solve rate | Inherits training count if omitted; 16 responses greatly increases evaluation work and changes interpretation; evaluation need not estimate group advantages |
| `disable_tqdm` | True: readable callback lines without progress bars | Removal adds display clutter, not an algorithm change |
| `per_device_train_batch_size` | 16 **completions** per microbatch | Larger increases activation memory; smaller is possible only with compatible generation batching |
| `gradient_accumulation_steps` | 2: effective batch of 32 completions/GPU without holding all training activations at once | 1 halves the effective batch; excessive accumulation delays updates and adds sequential work |
| `num_generations` | 16 responses/prompt: estimate within-prompt reward contrast and explore multiple solutions | Must be at least 2 for training, and divide the generation batch; 1 cannot estimate GRPO contrast |
| `gradient_checkpointing` | False: reference GPU trades memory for faster backward passes | True recomputes forward activations in backward, reducing stored activations at extra compute cost; not enabled here |
| `bf16` | True: BF16 training/autocast path consistent with model loading | Unsupported hardware fails; toggling only this while leaving model dtype unchanged is not a complete precision configuration |
| `temperature` | 1.0: preserve diverse stochastic rollouts | Near-zero discourages within-group variation; overly high sampling makes reward discovery noisier |
| `use_vllm` | True: accelerated autoregressive rollout generation | False uses the Transformers generation path, often slower; missing compatible vLLM/CUDA dependencies prevents startup |
| `vllm_mode` | `colocate`: engine shares the training GPU/process resources | Server mode needs a separate configured service; colocate can contend for VRAM |
| `vllm_gpu_memory_utilization` | 0.3: reserve a bounded share for the generation engine, leaving headroom for training | It is not a total-process memory cap; too high competes with activations/optimizer, too low can starve engine capacity or fail initialization |
| `vllm_max_model_length` | `256 + 1024 = 1280`: match maximum admitted prompt plus completion | Too small cannot fit sequences; unnecessarily large context provisioning increases KV-cache pressure |
| `beta` | 0.001: weak regularization toward the initial policy | 0 removes reference KL and saves its forward work, but permits more drift; too large can suppress reward-driven improvement |
| `save_total_limit` | 1: keep at most one rotating numbered checkpoint | Does not set save frequency; more retained checkpoints use disk; deletion of older checkpoints limits rollback |
| `report_to` | `none`: no hosted experiment tracker required | Enabling a tracker adds dependency/auth/logging setup; lessons save local JSONL instead |

On **one GPU**, defaults imply `16 * 2 = 32` completions per accumulated update
and `32 / 16 = 2` prompt groups in a generated batch. With multiple processes,
global generation-batch size also multiplies by world size; groups can be
distributed across ranks. These educational commands intentionally support one
process so each recorded local group is complete. With the default
`steps_per_generation=gradient_accumulation_steps`, the generated batch must be
divisible by `num_generations`. For example, changing the microbatch to 4 and
keeping accumulation 2 yields 8 completions, which cannot hold groups of 16.

### Every explicit LoRA setting

| Parameter | Value and rationale | If omitted or misconfigured |
|---|---|---|
| `r` | 16: rank of low-rank `B @ A` updates, balancing capacity and memory | Lower rank saves state but restricts updates; higher rank adds trainable parameters and optimizer work |
| `lora_alpha` | 32: standard LoRA scaling `alpha/r = 2` for the adapter contribution | Changing rank without considering alpha changes update scale; excessive scaling destabilizes training |
| `lora_dropout` | 0.05: regularize adapter inputs during training | 0 removes this regularization; too much dropout makes the reward gradient noisy; disabled in evaluation |
| `target_modules` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`: attention and MLP linear projections | Misspelled/nonexistent names fail or omit intended adaptation; fewer targets save state but reduce capacity |
| `task_type` | `CAUSAL_LM`: PEFT wrapper for next-token autoregressive generation | Wrong task wrapper can misroute forward/generation behavior |

LoRA keeps base weights frozen. It greatly reduces parameter-gradient and
optimizer-state memory; it does **not** eliminate base-model forward/backward
activation costs. `adapter/` stores the learned deltas and config, not a complete
copy of Qwen's base weights.

### Inherited settings and lesson-only additions

`resolved_grpo_config.json` saves the actual installed config, including inherited
defaults. Important ones in the pinned source are `loss_type=dapo`,
`scale_rewards=group`, equal `reward_weights`, `num_iterations=1`, `epsilon=0.2`,
token-level importance sampling, and `mask_truncated_completions=False`.
Truncated responses therefore remain eligible for the loss; inspect clipping
metrics before concluding the model cannot solve a task. Trainer also supplies
an AdamW optimizer, gradient norm clipping (`max_grad_norm=1`), and scheduler
defaults; these are not hand-coded in this repository. Inspect the saved config
before changing libraries or comparing runs.

Lessons add explicit `seed=42` and `data_seed=42`, resolve and pin the base revision
before loading the tokenizer, materialize the selected rows and set `save_steps=10` (the
original inherits 500). Those are reproducibility/inspection/checkpoint choices.
The one-step stop forces a full checkpoint at its terminal update. These changes
do not alter the explicit source reward, model size, sampling budget, LoRA setup,
learning rate or 2,000-update horizon. The CLI's `--max-steps` lets you choose a
different horizon **before starting**; use the same value on resume. Changed
plans are rejected in an existing run directory. The lesson records package
versions and compares the full resolved GRPO and LoRA configs before resuming;
the original records are preserved. `run_name=countdown-tutorial` gives the
experiment an explicit label; it does not change the training objective.

## 7. Metrics: meaning and how to use them

`ReadableLoggingCallback` prints a compact subset. Lesson 06 saves all callback
logs in `metrics.jsonl`; lesson 07 prints the latest entries and one full response
group. Trainer logs aggregate over batches/logging windows and evaluation can
average batch means. For final reporting, lesson 04 computes exact per-example
rates over its saved rows. Do not assume every logged average is a single-example
measurement or pool it with a different run's denominator.

| Metric | Meaning | Practical interpretation |
|---|---|---|
| `loss` / `train_loss` | Current-window / final average of the masked surrogate plus regularization | Not supervised NLL or accuracy; can be near zero/negative despite useful updates; do not rank checkpoints by this alone |
| `grad_norm` | Global trainable-gradient norm before configured clipping | Nonfinite values are a failure; persistent zero with unsolved tasks suggests no signal; spikes warrant checking KL/rewards/lr |
| `learning_rate` | Scheduler-reported LR at the logging boundary | Should follow the intended linear horizon; at checkpoint-1 it should be near the initial LR, not already zero |
| `step_time` | Time summed over training microsteps for an optimizer update in TRL | Includes generation/training work in that path; excludes some outer-loop evaluation/checkpoint overhead; compare with wall time for a budget |
| `rewards/equation_reward_func/mean` | Fraction of sampled responses passing arithmetic verification | Main task-quality signal; compare training and held-out values to detect overfitting |
| `rewards/format_reward_func/mean` | Fraction passing formatting check | Useful initial progress; saturation alone does not imply better solving |
| Reward component `/std` | Spread of binary per-component outcomes | High near mixed successes/failures; low near saturation or uniform failure |
| `reward` | Mean summed weighted reward, here in `[0,2]` | Can rise purely through formatting; decompose it |
| `reward_std` | Spread of summed rewards across the sampled batch | Not the same as the within-group std used in each advantage |
| `frac_reward_zero_std` | Fraction of prompt groups with no reward variation (equal-size groups make the repeated-row fraction equivalent) | All 0s, all 1s or all 2s each give no reward-policy contrast; inspect component rewards and completions to tell which |
| `completions/mean_length`, `/min_length`, `/max_length` | Generated completion token-length summaries | Increasing length may be useful reasoning or wasted output; compare solves and truncation |
| `completions/clipped_ratio` | Fraction of responses without a normal stopping EOS (length-capped) | High values suggest missing closing answers or insufficient token budget; unrelated to PPO probability-ratio clipping |
| `completions/mean_terminated_length`, `/min_terminated_length`, `/max_terminated_length` | Length summaries for normally terminated completions | Separate ordinary responses from capped ones; inspect availability when few terminate |
| `kl` | Mean masked nonnegative per-token KL estimator against reference policy | Raw divergence, not multiplied by beta in the printed metric; sustained rapid rise can indicate excessive policy drift |
| `entropy` | Mean token-distribution entropy in nats over active completion positions | Falling entropy can mean learned consistency or collapse; judge with held-out solves, variation and length |
| `clip_ratio/region_mean` | Fraction of active positions where the clipped PPO branch limits an advantage-improving update | High sustained values suggest large/off-policy updates; low values are common with one iteration and are not proof of no learning |
| `clip_ratio/low_mean`, `/high_mean`, `/low_min`, `/high_max` | Lower/upper clipping aggregates and extrema | Distinguish excessive reduction of negative-advantage tokens from excessive increase of positive-advantage tokens |
| `num_tokens` / `num_input_tokens_seen` when present | Accumulated tokens observed by the trainer, including prompt/generation accounting | Budget/throughput context; not distinct tasks or optimizer steps |
| `epoch` | Fractional progress through the dataloader | Can exceed 1 because positive max_steps overrides the epoch setting |
| `train_runtime`, `train_steps_per_second`, `train_samples_per_second` | End-of-run wall-time and throughput summaries | Hardware/workload measures; sample accounting is not unique-prompt quality |
| `eval_*` | Evaluation-prefixed versions of reward/loss/generation metrics; `eval_runtime` is wall time for evaluation | Use held-out equation/format metrics to track generalization; eval group-zero-std is uninformative with one generation |
| vLLM sampling-correction metrics, when emitted | Rollout/training distribution mismatch and correction statistics | Inspect source/version-specific names if mismatch grows; separate them from reference KL and PPO clipping |

GPU callback lines are memory observations, not model-quality metrics:

| Printed field | Interpretation |
|---|---|
| Device used / free / total | Whole-device snapshot, including other processes and vLLM allocations |
| Torch allocated | Bytes occupied by live PyTorch tensors in this process |
| Torch reserved | PyTorch caching allocator's reserved blocks; includes reusable free space |
| Torch peak allocated / reserved | Peaks since the optimizer-step reset; not the entire run's or entire device's peak |

Large reserved-minus-allocated memory is not automatically a leak. Device usage
can exceed PyTorch's numbers because vLLM and other components allocate memory
outside that allocator. Log both memory and time while changing one setting.

## 8. Performance: which code helps and why

| Code/setting | Mechanism | Tradeoff / what it does not solve |
|---|---|---|
| LoRA `peft_config` | Freeze billions of base weights; store gradients and Adam moments mainly for small adapters | Activations and base-weight matmuls still cost substantial memory/compute; less adaptation capacity than full finetuning |
| BF16 model loading and training | Smaller weight/activation storage and efficient Tensor Core operations on supported GPUs | Requires suitable hardware and can change numerical behavior |
| SDPA attention | Select optimized attention kernels; eligible fused paths avoid materializing large attention matrices | Exact speed/memory benefit depends on kernel dispatch, masks and shape |
| vLLM rollouts | Optimized autoregressive decoding, batching and KV-cache management reuse attention keys/values for generated prefixes | Separate engine state and weight synchronization consume memory/time; gains depend on rollout workload |
| Colocate mode / 0.3 engine fraction | Share one device and avoid maintaining a separate generation server; allocate a bounded generation budget | Does not isolate memory from training; excessive reservation causes OOMs |
| 1,280 max vLLM length | Match admitted prompt+completion budget, avoiding unnecessarily broad context provisioning | Raising completion lengths requires coordinated context/memory changes |
| Microbatch 16 plus accumulation 2 | Train an effective 32-completion batch with only microbatch training activations resident | Accumulation is primarily a memory tradeoff, not guaranteed faster; generation may still allocate for its larger rollout batch |
| Prompt filter / 1,024 completion cap | Bound prefill, decoding, attention/KV and activation work | Can discard contexts or cut off valid solutions; changes accessible task distribution |
| `gradient_checkpointing=False` | Avoid recomputation overhead when VRAM permits | Enable it to reduce stored activations if training OOMs, paying extra backward compute |
| Batched evaluation / eval every 10 steps | Amortize generation overhead and avoid evaluating after every update | Large eval batches may OOM; eval still consumes material wall time |
| `save_total_limit=1` | Bound disk usage of optimizer checkpoints | This is a disk optimization, not a GPU compute optimization; limits recovery choices |

Reward evaluation is simple CPU string/arithmetic work; sampling and model
forward/backward passes dominate typical LLM runs. `beta=0.001` is an algorithmic
choice, not a speed optimization: reference log-probability computation adds work.
GPU/logging callbacks observe behavior; they do not accelerate the model.
Do not claim FlashAttention, quantization, `torch.compile` or gradient
checkpointing is enabled when it is not configured by this code.

To investigate OOMs, distinguish model initialization, rollout generation,
training forward/backward and evaluation peaks. Lower an eval batch to address
eval OOM; reduce compatible train/generation batching or enable checkpointing
for training activation pressure; reconsider the vLLM memory fraction/context
for engine capacity. A smaller microbatch alone may violate group divisibility.
Treat each such change as a new experiment and record its resolved settings.

## 9. Continue the run and decide whether it is promising

```bash
python -m education.06_train --resume runs/tutorial/training/checkpoint-1
python -m education.07_metrics
```

This resumes optimizer, scheduler and RNG state and targets **2,000 total**
updates, not 2,000 additional ones. For a shorter planned experiment, choose e.g.
`--max-steps 50` for both the original one-step command and the resume command.
That changes the learning-rate trajectory and is a different experiment.
Repeated one-step commands can advance a checkpoint by one update at a time:
use `--one-step --resume <latest-checkpoint>` with the same horizon. Only the
newest rotating numbered checkpoint is retained.

During a promising run, look for these patterns across several evaluations:

- Held-out **equation** solve rate rises beyond early fluctuation, alongside
  improvements in actual sampled expressions. Early format gains are useful but
  insufficient. With a small held-out set, several percentage points can be noise.
- Training and held-out solving improve together. If only training improves,
  inspect task overlap, repeated contexts and overfitting.
- Groups contain differing rewards, so some responses have positive/negative
  advantages. High zero-std with high equation rewards can mean tasks are already
  easy; high zero-std with poor solving means reward discovery has stalled.
- Completion lengths remain useful and truncated fraction is manageable. Longer
  output without more solved tasks can consume the budget without progress.
- Gradient norms, KL and entropy stay finite and evolve smoothly enough to
  interpret. A fast KL spike plus falling held-out solves warrants intervention.
  An entropy drop without quality gain can indicate collapse.
- Memory headroom remains sufficient; step time is stable after startup/kernel
  warmup. Include evaluation/checkpoint time in your wall-time budget.

At the one-step boundary, inspect all 16 completions in the first group, not
only the best. Find the arithmetic checks that failed and correlate totals with
advantage signs. Then inspect the LR, gradient norm and saved global step.
One step verifies mechanics; it cannot establish a trend. No particular target
solve rate or guaranteed improvement is assumed by these lessons.

## 10. Final evaluation: load what you actually saved

```bash
python -m education.04_evaluate --adapter runs/tutorial/adapter
```

The process loads the pinned base model plus the saved adapter from disk and
uses the same saved validation rows and explicit sampling protocol as baseline.
It verifies metadata/protocol compatibility before running. `final.json`
contains every completion, a summary, and paired counts of newly solved tasks
and regressions alongside solve-rate delta. Read those changed cases: unchanged
aggregate accuracy can hide both gains and losses.

Report model/base revision, training horizon, exact data size, sampling policy,
solve and format rates, truncation and paired outcomes. A single sampled
response per task has sampling noise; fixed seeds do not make it a proof of
statistical significance. The validation set was repeatedly monitored, and
equivalent tasks may cross the row split, so this is not a pristine test-set
generalization claim. For stronger claims, add a separate deduplicated test set
and repeated seeds after making all training choices.

## Files you can inspect

| File under the run directory | Contents |
|---|---|
| `data.json`, `dataset/`, `tokenizer/` | Exact prepared row snapshot, fingerprints, overlap and base revision |
| `baseline.json`, `final.json` | Actual standalone sampled evaluations and completions |
| `training_plan.json` | Fixed scheduler horizon and experiment metadata |
| `resolved_grpo_config.json` | Explicit plus inherited settings from installed TRL/Trainer |
| `resolved_lora_config.json` | Full resolved PEFT settings checked again on resume |
| `metrics.jsonl` | Actual callback metric events, each tagged with global step |
| `step_1_rollouts.json`, `latest_rollouts.json` | Real completions and reward/advantage inspection data |
| `training/checkpoint-N/` | Complete resumable training state; older numbered checkpoints rotate away |
| `adapter/` | Final LoRA delta, tokenizer and tutorial metadata for fresh evaluation |

## Source references and verification limits

Implementation details were checked against the pinned official source:
[TRL 1.17.1 config](https://github.com/huggingface/trl/blob/v1.17.1/trl/trainer/grpo_config.py),
[TRL trainer](https://github.com/huggingface/trl/blob/v1.17.1/trl/trainer/grpo_trainer.py),
and [Transformers 5.19.0 training arguments](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/training_args.py).
Read [VALIDATION.md](VALIDATION.md) for the checks actually run. No baseline score,
Qwen update, learning curve or final solve rate is prepopulated or fabricated.
