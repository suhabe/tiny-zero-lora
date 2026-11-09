# Tiny Zero LoRA: from GRPO to an LLM implementation

For students who understand Sutton & Barto, PPO and GRPO and want to understand
the tensors, data, generation, rewards and engineering behind an LLM run.

Start with **[the educational walkthrough](education/README.md)**. Its numbered
scripts exercise the same prompt functions, verifiers, data preparation and
trainer used by `train.py`.

**[Run every lesson on Modal](education/MODAL.md)** using only the Modal client
locally. Each lesson has a separate command, and one shared volume preserves
data, baseline results, checkpoints and the saved adapter between invocations:

```bash
modal run modal_lessons.py::dataset
modal run modal_lessons.py::baseline
modal run modal_lessons.py::step
modal run modal_lessons.py::training
modal run modal_lessons.py::final_evaluation
```

```bash
# These lessons need only Python 3.12; no model download.
python -m education.00_preflight
python -m education.03_rewards
python -m education.05_code_and_config
```

The remaining lessons need the pinned training stack and a compatible NVIDIA
GPU for model execution. The walkthrough includes actual dataset inspection,
tokenizer rendering, baseline evaluation, one optimizer update, resumable
training, metrics interpretation and final evaluation of a reloaded adapter.

```bash
# On a configured Linux CUDA machine; see modal_train.py for the reference image.
python -m pip install -r requirements.txt
python -m pip check
python -m education.01_dataset
python -m education.02_prompts
python -m education.04_evaluate
python -m education.06_train --one-step
python -m education.07_metrics
python -m education.06_train --resume runs/tutorial/training/checkpoint-1
python -m education.04_evaluate --adapter runs/tutorial/adapter
```

`python train.py` still runs the original 3B/2,000-update experiment.
`modal_train.py` retains its original training entrypoint and now includes the
shared task module in its image. No paid GPU job is launched by the lessons
automatically. See [execution status](education/VALIDATION.md) for what was
actually verified while authoring these additions.
