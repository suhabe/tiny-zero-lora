# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo==0.25.2", "altair==6.3.0"]
# ///
"""Countdown: from a terminal reward to an LLM policy update.

Keep the complete project together when opening this notebook; lab_core.py,
lesson_tables.py, training_worker.py and upstream_train.py are sibling files.
Run: python -m marimo edit countdown_grpo.py
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Countdown · GRPO implementation lab")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import inspect
    import json
    import math
    from pathlib import Path
    import lab_core as core
    import lesson_tables as tables
    import training_worker as worker
    def code_block(text, language="python"):
        return mo.md("```" + language + "\n" + str(text) + "\n```")
    return Path, alt, code_block, core, inspect, json, math, mo, tables, worker


@app.cell(hide_code=True)
def _(core, mo):
    mo.md(rf"""
    # From GRPO equations to an LLM training loop
    **A guided reading of [`train.py`]({core.SOURCE_URL}) · source commit `{core.COMMIT[:12]}`**

    You know PPO and GRPO. Here we connect a dataset row to a chat prefix, sampled
    token trajectories, a terminal reward, a masked surrogate loss, and a LoRA update.
    Read sections 1–5, run a baseline, inspect one update in section 6, then continue
    training and compare held-out outcomes.

    **Execution status at delivery:** the offline examples and numerical update are
    executable on CPU. This authoring environment has no CUDA GPU; no Qwen baseline,
    GPU update, full training run, or final model result has been measured here.
    The result panels fill only from a real worker run. Empty charts mean no data.

    **Two presets:** `repository` preserves the 3B model, 2,000 sampled rows,
    2,000 updates, 16 generations and vLLM; `classroom` uses the same task/rewards
    with a 0.5B model, 320 rows, 50 updates and Transformers generation. The smaller
    preset is a different experiment, not a reproduction of 3B results.

    From this directory, open the project with `python -m marimo edit countdown_grpo.py`.
    The light UI environment needs only `requirements-ui.txt`. For real training,
    install `requirements-classroom.txt` or `requirements-gpu.txt` in the **same Python
    environment** as marimo. See `README.md` for exact commands and CUDA setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Dataset: tasks, not labeled demonstrations

    [`Jiayi-Pan/Countdown-Tasks-3to4`](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
    exposes a `train` split with `target: int64` and `nums: list[int]` of length 3 or 4.
    A row contains no reasoning trace or target token sequence. The integers are
    inputs to a programmatic verifier, not labels passed to cross entropy.

    | Field | Example | Role |
    |---|---|---|
    | `nums` | `[44, 19, 35]` | Each occurrence must appear exactly once in the expression. |
    | `target` | `98` | Desired numeric result; `44 + 19 + 35` is one valid answer. |
    | `prompt` | Added by `map` | Tokenizer-rendered chat prefix sent to the policy. |
    | `example_id` | Added only by this lab | Stable original-row index for paired evaluation/auditing. |

    The cell below uses actual dataset-viewer rows bundled for offline inspection.
    **Prepare** downloads the real dataset and executes the same preprocessing as the source:
    shuffle with seed 42 → select at most 2,000 rows → map to prompts → discard prompts
    over 256 tokens → seeded 90/10 split. Thus 1,800/200 is an upper bound, not a
    guaranteed final size. `nums` survives `map` for the reward callback.

    A random row split can contain equivalent tasks with reordered numbers. Prepare
    reports overlap using `(sorted(nums), target)`; it does not silently change the
    source split. The held-out split is reused during training and is a **validation
    set**, not an untouched test set.
    """)
    return


@app.cell
def _(core, mo):
    dataset_examples = core.VIEWER_EXAMPLES
    mo.ui.table(dataset_examples, selection=None, pagination=False, label="Actual dataset-viewer examples (offline)")
    return (dataset_examples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Prompt generation: continue an assistant prefix

    The source constructs system, user, and assistant messages. The assistant
    message already contains `Let me solve this step by step.\n<think>`.
    `continue_final_message=True` keeps that assistant message open; a new assistant
    turn or an extra opening `<think>` would change both the distribution and scoring.

    `tokenize=False` produces a **string** now. GRPOTrainer later tokenizes it. The
    exact chat-template special tokens depend on the model revision; use the real
    tokenizer, not manual role-token concatenation. Left padding aligns variable-length
    prompts for batched decoder-only generation. Reusing EOS as PAD supplies a valid
    padding ID; the attention mask must still distinguish padding from actual tokens.
    """)
    return


@app.cell
def _(code_block, core, mo):
    format_reward, equation_reward, generate_prompt = core.upstream_functions()
    recorder = core.MessageRecorder()
    prompt_audit = generate_prompt(recorder, [44, 19, 35], 98)
    mo.vstack([
        code_block(core.source_for("generate_r1_prompt"), language="python"),
        mo.md("**Runnable message audit.** This calls the exact function with a recorder; it does not claim to show Qwen special tokens."),
        mo.ui.table(recorder.messages, selection=None, pagination=False),
        mo.md(f"Template options: `{recorder.options}`. The prepared snapshot below will show the actual rendered prompt and token count."),
    ])
    return equation_reward, format_reward, generate_prompt, prompt_audit, recorder


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Reward: two independent binary tests

    For completion $y$ and task $x$, the default equal weights give
    $r(x,y)=r_{\mathrm{format}}(y)+r_{\mathrm{equation}}(x,y)\in\{0,1,2\}$.

    **Format:** prepend `<think>` to the *generated suffix*, then match the source
    regex. It requires a closing `</think>`, a newline, and an `<answer>…</answer>`
    block. It checks structure, not the truth of the explanation. Python regex `$`
    also permits a single final newline; this is not a strict XML parser.

    **Equation:** extract the first answer block; compare the multiset of decimal
    digit runs with `nums`; allow only digits, whitespace, `+ - * / ( )`; explicitly
    reject `**` and `//`; evaluate with builtins disabled; award 1 if the numeric
    error is below $10^{-5}$. Missing tags, division by zero, invalid syntax, extra
    numbers and wrong results receive 0. Negative intermediate values and fractional
    division are allowed. Decimal literals are rejected. Duplicate supplied numbers
    must be used with the correct multiplicity.

    The equation check does not require format reward to pass. A correct answer
    without reasoning tags can earn 1; a beautifully formatted wrong answer can
    also earn 1. This is why we track solve rate and joint success separately.
    The restricted `eval` below is the exact source behavior, not a general-purpose
    expression sandbox. A production verifier should use a bounded arithmetic parser.
    """)
    return


@app.cell
def _(code_block, core, equation_reward, format_reward, mo):
    reward_cases = [
        {"case":"both pass", "suffix":"Sum them.</think>\n<answer>44+19+35</answer>", "nums":[44,19,35], "target":98},
        {"case":"correct; no reasoning tags", "suffix":"<answer>44+19+35</answer>", "nums":[44,19,35], "target":98},
        {"case":"format only", "suffix":"Guess.</think>\n<answer>44-19+35</answer>", "nums":[44,19,35], "target":98},
        {"case":"extra number", "suffix":"Work.</think>\n<answer>44+19+35+0</answer>", "nums":[44,19,35], "target":98},
        {"case":"duplicate inputs used correctly", "suffix":"Cancel.</think>\n<answer>58+5+23-23</answer>", "nums":[23,58,5,23], "target":63},
        {"case":"forbidden floor division", "suffix":"Work.</think>\n<answer>44//19+35</answer>", "nums":[44,19,35], "target":98},
        {"case":"missing answer", "suffix":"I cannot solve it.</think>", "nums":[44,19,35], "target":98},
    ]
    scored_cases = [dict(case=_row['case'], completion=_row['suffix'],
        format=format_reward([_row['suffix']])[0],
        equation=equation_reward([_row['suffix']],[_row['target']],[_row['nums']])[0]) for _row in reward_cases]
    for _row in scored_cases:
        _row['total'] = _row['format'] + _row['equation']
    assert [row['total'] for row in scored_cases] == [2, 1, 1, 1, 2, 1, 0]
    mo.vstack([
        mo.accordion({"Exact reward implementations":mo.vstack([code_block(core.source_for("format_reward_func"),language="python"), code_block(core.source_for("equation_reward_func"),language="python")])}),
        mo.ui.table(scored_cases, selection=None, pagination=False, label="Executed reward cases"),
    ])
    return reward_cases, scored_cases


@app.cell
def _(mo):
    completion_editor = mo.ui.text_area(
        value="Add all three.</think>\n<answer>44+19+35</answer>",
        label="Try a generated suffix for nums=[44,19,35], target=98",
        full_width=True,
    )
    completion_editor
    return (completion_editor,)


@app.cell
def _(completion_editor, equation_reward, format_reward, mo):
    _text = completion_editor.value
    _f = format_reward([_text])[0]
    _e = equation_reward([_text], [98], [[44,19,35]])[0]
    mo.md(f"Executed score: **format {_f:g} + equation {_e:g} = {_f + _e:g}**.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Baseline on the initial model

    Use the launcher in this order: **Prepare → Preflight → Baseline → Step → Train → Final**.
    Submit once per stage and wait for the worker to exit before the next stage.
    Preparation needs internet access and the training Python dependencies, but no GPU.
    Baseline and later stages need a BF16-capable CUDA GPU. Start the classroom
    preset on a GPU with comfortable memory headroom; the original repo uses a B300.
    A hardware name alone does not guarantee a particular memory budget will fit.

    Baseline runs `GRPOTrainer.evaluate()` before any optimizer update, using one
    sampled completion per held-out prompt at temperature 1.0 and the configured
    completion limit. It is sampled solve rate, **not greedy accuracy or pass@16**.
    The wrapper records each completion and the exact reward components, then
    computes rates with an explicit per-example denominator and Wilson intervals.

    Keep the run folder and preset fixed through the experiment. A manifest refuses
    incompatible settings; model/dataset revisions and the prepared split are saved.
    Opening the notebook never downloads a model or launches training automatically.
    """)
    return


@app.cell
def _(mo):
    job_form = mo.ui.dictionary({
        "stage":mo.ui.dropdown(["prepare","preflight","baseline","step","train","final"], value="prepare", label="Stage"),
        "preset":mo.ui.dropdown(["classroom","repository"], value="classroom", label="Preset"),
        "folder":mo.ui.text(value="runs/classroom",label="Run folder (new folder for a new experiment)",full_width=True),
    }).form(submit_button_label="Launch selected stage")
    refresh = mo.ui.refresh(options=[2,5,10],default_interval=2,label="Refresh results")
    mo.vstack([job_form, refresh])
    return job_form, refresh


@app.cell
def _(core):
    jobs = core.JobManager()
    return (jobs,)


@app.cell
def _(core, job_form, jobs, mo):
    if job_form.value is not None:
        _request = job_form.value
        _folder = core.ROOT / _request['folder']
        _message = jobs.launch(_request['stage'],_request['preset'],_folder)
    else:
        _message = "No stage submitted. Offline examples above are ready to run."
    mo.md(_message)
    return


@app.cell
def _(core, job_form, jobs, refresh):
    _ = refresh.value
    selected_run = job_form.value or {'preset':'classroom','folder':'runs/classroom','stage':'prepare'}
    run_dir = (core.ROOT / selected_run['folder']).expanduser().resolve()
    run_events = core.read_events(run_dir)
    data_snapshot = core.load_json(run_dir/'data.json')
    baseline = core.load_json(run_dir/'baseline.json')
    final = core.load_json(run_dir/'final.json')
    preflight = core.load_json(run_dir/'preflight.json')
    parameter_counts = core.load_json(run_dir/'trainable_parameters.json')
    latest_rollouts = core.load_json(run_dir/'latest_rollouts.json')
    checkpoint = core.latest_checkpoint(run_dir)
    checkpoint_state = core.load_json(checkpoint/'trainer_state.json') if checkpoint else None
    console_tail = (run_dir/'console.log').read_text(errors='replace')[-9000:] if (run_dir/'console.log').exists() else 'No worker console output yet.'
    job_status = jobs.status()
    return baseline, checkpoint_state, console_tail, data_snapshot, final, job_status, latest_rollouts, parameter_counts, preflight, run_dir, run_events, selected_run


@app.cell(hide_code=True)
def _(code_block, console_tail, data_snapshot, job_status, json, mo, preflight, run_dir, run_events):
    _errors = [event['error'] for event in run_events if event['kind']=='error']
    _items = [mo.md(f"**{job_status}** · reading `{run_dir}`")]
    if _errors:
        _items.append(mo.callout(f"Latest worker error: {_errors[-1]}",kind="danger"))
    if preflight:
        _items.append(mo.accordion({"Hardware and package preflight":code_block(json.dumps(preflight,indent=2),language="json")}))
    if data_snapshot:
        _items.extend([
            mo.md(f"Prepared **{data_snapshot['train_rows']} train / {data_snapshot['eval_rows']} validation rows**; filtered out {data_snapshot['filtered_out']}; canonical task overlap: **{data_snapshot['canonical_train_eval_overlap']}**."),
            mo.ui.table([{k:v for k,v in row.items() if k!='prompt'} for row in data_snapshot['examples']],selection=None,label="Real prepared dataset rows"),
            mo.accordion({"Actual tokenizer-rendered prompt":code_block(data_snapshot['examples'][0]['prompt'],language="text"),
                "Prepared revisions and split provenance":code_block(json.dumps({k:v for k,v in data_snapshot.items() if k!='examples'},indent=2),language="json")}),
        ])
    else:
        _items.append(mo.md("**No prepared dataset yet.** Prepare runs the real `load_dataset`, prompt mapping, token-length filter and seeded split."))
    _items.append(mo.accordion({"Worker console (tail)":code_block(console_tail,language="text")}))
    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(baseline, mo):
    if baseline:
        _summary = baseline['summary']
        _interval = _summary['solve_rate_95pct_wilson']
        _display = mo.vstack([
            mo.md(f"### Measured initial-model baseline\nModel: `{baseline['model_id']}` · n={_summary['n']} · sampled solve rate **{_summary['equation']:.1%}** (95% Wilson {_interval[0]:.1%}–{_interval[1]:.1%}); format {_summary['format']:.1%}; joint {_summary['joint']:.1%}; truncation {_summary['truncated']:.1%}."),
            mo.ui.table(baseline['records'][:6],selection=None, label="Actual initial-model completions"),
        ])
    else:
        _display = mo.callout("Initial-model baseline has not been run. No baseline score is assumed or fabricated.",kind="info")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Read the training loop and its settings

    `GRPOTrainer` supplies the rollout/optimization machinery; `train.py` supplies
    the task, verifier, model, adapter and experiment configuration. This is not
    supervised training on a list of correct equations.

    | RL object | Concrete implementation |
    |---|---|
    | State | Prompt tokens followed by generated tokens so far. |
    | Action | Next token sampled from the causal LM. |
    | Trajectory | One completion, ending at EOS or the length cap. |
    | Reward | Sequence-level format + arithmetic score, evaluated after generation. |
    | Baseline | Mean reward across G completions of the same prompt; no learned value head. |
    | Advantage | One scalar per completion, broadcast to its active generated-token positions. |
    | Old policy | Policy that produced the rollout; log-probabilities are detached/frozen for ratios. |
    | Reference policy | Initial instruction model, used for KL; with fresh LoRA, disabling adapters gives the base reference. |
    | Trainable policy | Base model plus low-rank adapters; only adapter parameters receive optimizer updates. |

    The initial instruction model already knows language and tag syntax. GRPO
    redistributes probability among sampled behaviors using task feedback. It does
    not differentiate through the arithmetic verifier, and it does not validate
    the truth of the text inside the thinking block.
    """)
    return


@app.cell
def _(core, mo, selected_run, tables):
    source_settings = core.original_settings()
    active_preset = core.preset(selected_run['preset'])
    _grpo_rows = [{"parameter":key,"source value":repr(value),
        "why it is configured this way":tables.GRPO[key][0],"if removed or misconfigured":tables.GRPO[key][1]}
        for key,value in source_settings['grpo'].items()]
    _lora_rows = [{"parameter":key,"source value":repr(value),
        "why it is configured this way":tables.LORA[key][0],"if removed or misconfigured":tables.LORA[key][1]}
        for key,value in source_settings['lora'].items()]
    mo.vstack([
        mo.md("### Every explicit GRPOConfig argument\nValues are read from the bundled source AST, not retyped from memory. The rationales explain engineering tradeoffs; the author has not supplied ablation evidence that these values are optimal."),
        mo.ui.table(_grpo_rows,selection=None,pagination=False,wrapped_columns=["why it is configured this way","if removed or misconfigured"]),
        mo.md("### Every explicit LoRA argument\nFor a projection $W\\in\\mathbb{R}^{d_{out}\\times d_{in}}$, LoRA adds $(\\alpha/r)BA$: $r(d_{in}+d_{out})$ trainable parameters instead of $d_{in}d_{out}$. The frozen base weights still occupy memory and participate in forward/backward computation."),
        mo.ui.table(_lora_rows,selection=None,pagination=False,wrapped_columns=["why it is configured this way","if removed or misconfigured"]),
        mo.md("### Defaults that change the meaning of the algorithm\nThese refer to pinned TRL 1.17.1 / Transformers 5.19.0. Inspect `resolved_grpo_config.json` after model initialization; version drift is an error rather than an implicit algorithm change."),
        mo.ui.table([{"parameter":key,"resolved/default value":value,"meaning":explanation} for key,value,explanation in tables.DEFAULTS],selection=None,pagination=False,wrapped_columns=["meaning"]),
    ])
    return active_preset, source_settings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Other source choices matter too: the 3B model controls capacity/memory; the
    2,000-row sample controls task diversity; the 256-token prompt **filter** bounds
    input length without truncating task instructions. Removing that filter is
    different from increasing a trainer context limit. Changing the tokenizer or
    dropping `continue_final_message=True` changes the generation prefix and reward
    assumptions. Removing the callbacks removes observability, not the loss.

    `processing_class=tokenizer` handles text/token conversion and padding.
    `reward_funcs=[format, equation]` sets the two objectives. `train_dataset` provides
    prompts and reward metadata; `eval_dataset` is reserved for evaluation. Omitting
    `peft_config` would train all model parameters and require substantially more
    optimizer/gradient memory. `report_to="none"` does not disable custom logging.
    """)
    return


@app.cell
def _(code_block, core, inspect, mo, worker):
    mo.accordion({
        "Original main(): exact training code":code_block(core.source_for("main"),language="python"),
        "Lab worker: prepare real data":code_block(inspect.getsource(worker.prepare),language="python"),
        "Lab worker: real evaluation, callbacks, training and checkpoint continuation":code_block(inspect.getsource(worker.run_model),language="python"),
        "Original memory and readable logging callbacks":mo.vstack([
            code_block(core.source_for("GPUMemoryCallback"),language="python"),
            code_block(core.source_for("ReadableLoggingCallback"),language="python"),
        ]),
    })
    return


@app.cell(hide_code=True)
def _(active_preset, core, mo, selected_run, source_settings):
    _changes = []
    for _key in ['model_id','dataset_size','max_prompt_length']:
        _changes.append({'setting':_key,'repository':repr(source_settings[_key]),'active preset':repr(active_preset[_key])})
    for _key in ['max_steps','per_device_train_batch_size','gradient_accumulation_steps','num_generations','max_completion_length','per_device_eval_batch_size','gradient_checkpointing','use_vllm']:
        _changes.append({'setting':_key,'repository':repr(source_settings['grpo'][_key]),'active preset':repr(active_preset['grpo'][_key])})
    mo.vstack([
        mo.md(f"### Active preset: `{selected_run['preset']}`"),
        mo.ui.table(_changes,selection=None,pagination=False),
        mo.md(r"""
        The classroom preset uses less model/activation memory, only four rollouts per
        prompt, shorter completions and a shorter run. These changes reduce cost and
        also change exploration, capacity and truncation behavior. Transformers generation
        avoids vLLM setup/duplicate engine memory but is usually slower per token.

        Both presets use the source rewards, LoRA settings, LR, temperature, KL coefficient,
        seeded data-processing order and 90/10 split. Lab-only operational changes:
        save every 10 steps (instead of the original inherited 500), force a checkpoint
        after Step, explicitly set `seed=data_seed=42`, resolve/pin Hub revisions, add
        audit IDs, log JSONL, and save per-example evaluations. These are recorded in
        the run manifest. The GPU worker intentionally supports one process/device.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · One optimizer update, from rewards to token gradients

    On one GPU, $B=\text{microbatch}\times\text{accumulation}$ counts **completion
    sequences**. The source has $B=16\times2=32$ and $G=16$, hence only two distinct
    prompts per generation batch/update. The classroom preset has $B=2\times2=4$
    and $G=4$, hence one prompt. Grouping requires $B\bmod G=0$. Increasing $G$ while
    holding $B$ fixed reduces distinct-prompt diversity.

    For each prompt group, TRL's default `sum_then_normalize` gives
    $A_i=(r_i-\bar r)/(s_r+10^{-4})$, with a **sample** standard deviation. The terminal
    scalar advantage is shared by every generated token in that completion. There
    is no per-token verifier reward and no value-function bootstrap/GAE in this loop.

    For active completion token $t$:
    \[
      \rho_{it}=\exp(\log\pi_\theta(y_{it}|x,y_{i,<t})-\log\pi_{old}(y_{it}|x,y_{i,<t})),
      \quad \ell_{it}=-\min(\rho_{it}A_i,
                    \operatorname{clip}(\rho_{it},0.8,1.2)A_i)+\beta\hat k_{it}.
    \]
    A base sampled KL term is $\exp(d)-d-1$ with
    $d=\log\pi_{ref}-\log\pi_\theta$. The pinned implementation applies an additional
    current/old ratio correction to that term. vLLM runs also correct rollout/training
    log-probability mismatch. The compact exercise below isolates the surrogate and
    base KL estimator; read the displayed worker/TRL source for those corrections.

    `loss_type="dapo"` sums masked token losses and normalizes by active tokens
    across the accumulated global batch. Prompt and padding positions contribute
    zero to this objective; prompt tokens still condition generation and receive
    gradient through their contextual computation. EOS is an active generated token.
    Do not replace this with supervised loss on the prompt or assume every sequence
    gets the same length-normalized weight.

    **A zero loss can still have a nonzero gradient.** When the old and current
    policy match, $\rho=1$, but $\nabla_\theta\rho$ need not vanish. The same sampled
    tokens are teacher-forced through the policy to obtain differentiable log-probs;
    generation and rewards themselves run without gradient tracking.
    """)
    return


@app.cell
def _(alt, core, mo):
    numeric_trace = core.numeric_step()
    _rows = numeric_trace['rows']
    _plot_rows = [dict(completion=str(row['completion']),position=row['position'],
        stage=stage,probability=row[key]) for row in _rows if row['mask']
        for stage,key in [('before','p_before'),('after','p_after')]]
    _chart = alt.Chart(alt.Data(values=_plot_rows)).mark_point(size=80).encode(
        x=alt.X('position:O',title='Token position'),y=alt.Y('probability:Q',scale=alt.Scale(zero=False)),
        color='stage:N',column='completion:N',tooltip=['completion:N','position:O','stage:N','probability:Q'],
    ).properties(width=150,height=150,title='Executed tiny token-policy update (not an LLM result)')
    mo.vstack([
        mo.md(f"**Runnable numerical trace.** Rewards `{numeric_trace['rewards']}`, lengths `{numeric_trace['lengths']}`; advantages `{[round(a,4) for a in numeric_trace['advantages']]}`. Each position has a Bernoulli logit for its sampled token. One gradient step with LR=0.2 changes the actual objective from **{numeric_trace['loss_before']:.6f}** to **{numeric_trace['loss_after']:.6f}**. Central finite differences agree with analytic gradients to `{numeric_trace['max_gradient_error']:.2e}`."),
        mo.ui.table(_rows,selection=None,pagination=False,label="Inspect completion masks, token gradients and probability changes"),
        mo.as_html(_chart),
        mo.md("This small policy demonstrates the update mechanics. It is not a substitute for running the Qwen step. KL gradients initially vanish here; negative-advantage token probabilities fall, positive ones rise, and masked positions do not change."),
    ])
    return (numeric_trace,)


@app.cell(hide_code=True)
def _(checkpoint_state, latest_rollouts, mo, parameter_counts):
    _parts = [mo.md(r"""
    **Run one real update:** return to the launcher and submit **Step** after Baseline.
    It calls `trainer.train()` with the full preset horizon (50 or 2,000), stops after
    one additional optimizer step, and saves model, optimizer, scheduler and RNG state.
    Do **not** create a 1-step scheduler and later extend it: that changes the learning
    rate schedule. Train resumes this complete checkpoint with the original horizon.
    Step can be submitted again to advance one more update.

    Inspect `grad_norm`, reward variation, completion lengths and generated examples
    before scaling up. Equal rewards make every task advantage zero; KL may still
    contribute a gradient away from the reference. At initialization both task and
    KL gradients can be zero for an uninformative group. One update is a wiring check,
    not evidence of a learned arithmetic capability.
    """)]
    if checkpoint_state:
        _parts.append(mo.md(f"Latest complete checkpoint: **optimizer step {checkpoint_state['global_step']}**. A resumed run retains the preset’s scheduler horizon."))
    if parameter_counts:
        _parts.append(mo.ui.table([parameter_counts],selection=None,pagination=False,label="Actual initialized parameter counts and batching"))
    if latest_rollouts:
        _parts.append(mo.md(f"**Actual latest training rollout batch, for update {latest_rollouts['for_update']}.** Compare siblings in each group: format, equation, summed reward and group-relative advantage. These diagnostic advantages are recomputed from the rewards; TRL forms its own training tensor."))
        _parts.append(mo.ui.table(latest_rollouts['records'],selection=None,label="Real sampled trajectories and their group signal"))
    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def _(mo, tables):
    mo.vstack([
        mo.md("## 7 · Metrics and live charts\nTraining callbacks flush one JSONL record at a time. The separate notebook timer reads completed lines every two seconds, so chart updates do not wait for training to return. GPU metrics arrive at step end; reward/policy metrics arrive at logging events. Evaluation points are labeled separately. No synthetic training history is preloaded."),
        mo.ui.table([{'metric':name,'definition':definition,'how to use it':use} for name,definition,use in tables.METRICS],selection=None,pagination=False,wrapped_columns=['definition','how to use it']),
    ])
    return


@app.cell
def _(alt):
    def metric_chart(events,keys,title,zero=True):
        points=[]
        for event in events:
            if event.get('kind')!='metrics':
                continue
            for key in keys:
                value=event.get(key)
                if isinstance(value,(int,float)):
                    points.append({'step':event['step'],'value':value,'metric':key})
        if not points:
            return None
        return alt.Chart(alt.Data(values=points)).mark_line(point=True).encode(
            x=alt.X('step:Q',title='Optimizer step'),
            y=alt.Y('value:Q',title=title,scale=alt.Scale(zero=zero)),
            color=alt.Color('metric:N',legend=alt.Legend(orient='bottom',labelLimit=330)),
            tooltip=['step:Q','metric:N','value:Q'],
        ).properties(width=420,height=190,title=title).interactive()
    return (metric_chart,)


@app.cell(hide_code=True)
def _(metric_chart, mo, run_events):
    _specs = [
        (['rewards/equation_reward_func/mean','rewards/format_reward_func/mean','eval_rewards/equation_reward_func/mean','eval_rewards/format_reward_func/mean'],'Solve / format rates'),
        (['reward','eval_reward'],'Total reward'),
        (['frac_reward_zero_std','completions/clipped_ratio','clip_ratio/region_mean'],'Zero-signal groups / truncation / clipping'),
        (['kl'],'Reference KL'),
        (['entropy'],'Token entropy (nats)'),
        (['completions/mean_length','completions/min_length','completions/max_length'],'Completion tokens'),
        (['loss'],'Training surrogate loss'),
        (['grad_norm'],'Gradient norm before clipping'),
        (['learning_rate'],'Learning rate'),
        (['step_time','wall_step_seconds'],'Step timing (seconds)'),
        (['gpu_allocated_gib','gpu_reserved_gib','gpu_peak_allocated_gib','gpu_device_used_gib'],'GPU memory (GiB)'),
    ]
    _charts = [metric_chart(run_events,keys,title) for keys,title in _specs]
    _charts = [chart for chart in _charts if chart is not None]
    if _charts:
        _display = mo.vstack([mo.hstack([mo.as_html(chart) for chart in _charts[i:i+2]],justify='start') for i in range(0,len(_charts),2)])
    else:
        _display = mo.callout("No real trainer metrics yet. Run Baseline or Step on a configured GPU to populate the charts.",kind="info")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Where the performance gains come from

    | Code/setting | Mechanism | Cost or limitation |
    |---|---|---|
    | `peft_config=LoraConfig(...)` | Store gradients and Adam moments only for low-rank matrices; avoid optimizing billions of base weights. | Frozen base weights, activations and generation KV cache remain. Capacity differs from full finetuning. |
    | Fresh LoRA + reference KL | Compute reference log-probs with adapters disabled rather than maintaining a second trainable base model. | Still requires reference forward computation when beta > 0. |
    | BF16 load + `bf16=True` | Smaller tensors/bandwidth demand and supported tensor-core arithmetic; wider exponent range than FP16. | Hardware support and numeric behavior matter; this is not 4-bit quantization. |
    | `attn_implementation="sdpa"` | Dispatch optimized scaled-dot-product attention; eligible fused kernels avoid materializing a full attention matrix. | Kernel selection depends on device, shapes and masks; SDPA does not guarantee a particular FlashAttention kernel. |
    | `use_vllm=True` | Specialized generation scheduling and paged KV-cache management improve autoregressive rollout throughput. | Extra engine/model/cache memory and weight synchronization overhead. Benchmark actual step time. |
    | `vllm_mode="colocate"`, utilization 0.3 | Share one GPU between inference and training, avoiding a separate inference service/GPU. | Memory contention. The 0.3 fraction is an inference budget, not a whole-process VRAM cap. |
    | Prompt filter + bounded context | Bound input activations and KV cache without truncating arithmetic instructions. | Filtering changes the task distribution; audit dropped rows. |
    | Completion cap | Bound rollout latency and cache growth, often a dominant RL cost. | Too short prevents rewardable answers; it changes the effective task. |
    | Batched generation + left padding | Amortize model work over many completion trajectories and align decoder prefixes. | More concurrent sequences increase memory; padding still wastes computation. |
    | Gradient accumulation | Fit the intended effective batch with smaller training microbatches. | Saves activation memory relative to processing B at once; it does not reduce total model FLOPs and can be slower. |
    | Checkpointing off in source; on in classroom | Off saves recomputation time; on trades extra forward work for lower activation storage. | This is a deliberate speed/memory tradeoff, not a universal speedup. |
    | G_eval=1; eval every 10 | Limit held-out rollout cost while getting regular feedback. | Noisy sampled solve rate. Evaluating 100 completions at once can still OOM. |
    | GPU/readable callbacks; `report_to="none"` | Expose timing/memory/reward failures and avoid external logging integrations. | Observability rather than a model-speed algorithm; logging has a small overhead. |

    The source does not enable quantization, explicit `torch.compile`, a learned
    critic, or a separate reward model. Do not attribute its speed to those.
    vLLM and Transformers sampling can differ numerically even at the same seed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Continue the run and judge progress

    Submit **Train** in the launcher. It loads the latest complete checkpoint and
    runs to the fixed preset horizon. Intermediate validation is every 10 steps;
    charts also refresh on each training update. The source's `max_steps=2000`
    overrides `num_train_epochs=1`, so expect repeated exposure to sampled tasks.

    | Observation | Interpretation and next action |
    |---|---|
    | Validation solve rate improves across several checkpoints; examples use each number exactly once | Promising task learning. Check the magnitude against n and random-seed variation. |
    | Format quickly approaches 1, total reward rises, equation score stays flat | Formatting learning. Inspect arithmetic failures; do not claim reasoning improvement. |
    | Most groups have zero reward standard deviation | No within-group task signal. Distinguish all-wrong from all-correct groups; inspect sampling diversity and task difficulty. |
    | Long outputs repeatedly hit the cap; format reward drops | Incomplete answers may dominate. Inspect suffixes before increasing the cap or changing the prompt. |
    | Training solves rise while validation solves flatten/fall | Possible overfitting/reward exploitation; stop and inspect per-task cases, overlap and decoding consistency. |
    | Entropy collapses, repeated outputs, flat solve rate | Exploration failure is plausible; compare temperature, learning rate and KL. Entropy alone is not a success criterion. |
    | KL/gradients spike with reward regression or nonfinite values | Stop and inspect the numerical logs/configuration. Lower LR or check length/masking before spending a full run. |
    | Memory peaks approach device capacity; step time grows | Inspect rollout length and allocator/engine usage. Lower microbatch while preserving grouping; use checkpointing or the classroom preset. |

    These are diagnostic patterns, not universal numeric thresholds. Make one
    controlled change in a **new run folder**, keep the evaluation protocol fixed,
    and compare multiple seeds for a claim of improvement. The 50-step classroom
    run may show no measurable gain; its purpose is to make the pipeline inspectable.
    A detached subprocess continues until its stage finishes; watch the console/PID
    before closing marimo. CLI Ctrl-C stops a foreground worker. Completed
    checkpoints survive interruption, but the current unfinished step does not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Final evaluation: compare behavior, not just loss

    Submit **Final** after the training worker exits. It loads the saved adapter on
    the exact base-model revision, evaluates the same held-out rows with the same
    tokenizer, completion budget, sampling parameters and seed as Baseline, and
    saves every completion with both reward components. No training occurs in Final.
    The final-stage KL/loss uses an evaluation reference constructed at reload and
    should not be compared to the training reference-KL trajectory; compare verifier
    rates, lengths, truncation and output text here.

    Same seed reduces one source of variation; changed lengths and numerical
    backends mean it does not force identical random token draws. The Wilson
    intervals below describe each binary rate, not a paired confidence interval for
    the improvement. New solves versus regressions are paired by original row ID.
    Related/duplicate tasks and repeated validation use weaken statistical claims.
    A publishable result needs a fresh, task-deduplicated test set and repeated seeds.
    """)
    return


@app.cell(hide_code=True)
def _(baseline, final, mo):
    if baseline and final:
        _summary_rows = []
        for _name,_key in [('Sampled solve rate','equation'),('Format rate','format'),('Joint success','joint'),('Mean total reward','reward'),('Truncation fraction','truncated'),('Mean completion tokens','tokens')]:
            _before = baseline['summary'][_key]
            _after = final['summary'][_key]
            _summary_rows.append({'metric':_name,'baseline':_before,'final':_after,'change':_after-_before})
        _before_by_id = {row['example_id']:row for row in baseline['records']}
        _paired = [dict(example_id=row['example_id'],nums=row['nums'],target=row['target'],
            before=_before_by_id[row['example_id']]['equation'],after=row['equation'],
            baseline_completion=_before_by_id[row['example_id']]['completion'],final_completion=row['completion']) for row in final['records']]
        _new = sum(row['after']>row['before'] for row in _paired)
        _lost = sum(row['after']<row['before'] for row in _paired)
        _ci_b = baseline['summary']['solve_rate_95pct_wilson']
        _ci_f = final['summary']['solve_rate_95pct_wilson']
        _display = mo.vstack([
            mo.md(f"### Measured before/after results\nModel `{final['model_id']}` · final optimizer step **{final['step']}** · n={final['summary']['n']}. Solve-rate Wilson intervals: baseline **{_ci_b[0]:.1%}–{_ci_b[1]:.1%}**; final **{_ci_f[0]:.1%}–{_ci_f[1]:.1%}**. **{_new} new solves, {_lost} regressions** on paired rows."),
            mo.ui.table(_summary_rows,selection=None,pagination=False),
            mo.ui.table(_paired,selection=None,label="Paired completions: inspect new solves, regressions and verifier exploitation"),
            mo.md("Report whether the final solve-rate change is positive, how many tasks improved/regressed, whether format/truncation changed, and which failures remain. A positive training reward curve by itself does not establish improvement."),
        ])
    else:
        _display = mo.callout("Final comparison is pending real Baseline and Final evaluations. No before/after scores are supplied without a model run.",kind="info")
    _display
    return


@app.cell(hide_code=True)
def _(core, mo):
    mo.md(rf"""
    ### Source trail and reproducibility

    - [Inspected train.py]({core.SOURCE_URL}), git blob `{core.TRAIN_BLOB}`. The notebook
      checks the bundled file against this hash; reward and prompt examples execute
      the exact three pure functions extracted from it.
    - [Repository environment and B300 launcher](https://github.com/suhabe/llm-post-training/blob/{core.COMMIT}/modal_train.py)
      and [original dependency pins](https://github.com/suhabe/llm-post-training/blob/{core.COMMIT}/requirements.txt).
    - [TRL 1.17.1 GRPOConfig](https://github.com/huggingface/trl/blob/v1.17.1/trl/trainer/grpo_config.py)
      and [GRPOTrainer](https://github.com/huggingface/trl/blob/v1.17.1/trl/trainer/grpo_trainer.py):
      defaults, grouping, masks, reward normalization and loss/metric implementation.
    - [Dataset viewer](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
      for the bundled example rows; the full dataset is loaded only by Prepare.
    - [marimo refresh API](https://docs.marimo.io/api/inputs/refresh/) for live polling.

    Each run saves `manifest.json`, Hub revisions/split IDs in `data.json`, the
    materialized dataset/tokenizer, `resolved_grpo_config.json`, JSONL metrics,
    console logs, rotating optimizer checkpoints, a reusable adapter, and
    `baseline.json` / `final.json` with per-example records. To reload for generation,
    load the recorded base revision, then `PeftModel.from_pretrained(base, "…/adapter")`;
    the adapter directory is not a complete standalone base model.

    **Suggested exercises:** find a response earning equation=1/format=0; create a
    zero-advantage group; predict which masked-token probabilities change; compare
    G=4 versus G=8 at a valid fixed effective batch; explain a reward increase caused
    only by formatting; propose a held-out task split that eliminates canonical overlap.
    """)
    return


if __name__ == "__main__":
    app.run()
