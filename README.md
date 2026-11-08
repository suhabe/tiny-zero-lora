# Countdown GRPO on Modal

Train Qwen2.5-3B-Instruct with LoRA on one B300:

```bash
python -m pip install modal==1.6.3
python -m modal setup
modal run --detach modal_train.py
```

Run a one-step smoke test first:

```bash
modal run modal_train.py --max-steps 1 --dataset-size 32
```

The launcher uploads the local `main.py` and `reward.py`, so it also works with this private repository. No GitHub or Hugging Face token is needed for the public model and dataset.

B300 requires CUDA 13.1 or newer. The launcher uses NVIDIA's `26.01-py3` image with CUDA 13.1.1 and retains its preinstalled PyTorch. The Hugging Face versions are pinned in `requirements.txt`. LoRA uses BF16 weights and PyTorch SDPA attention; no bitsandbytes or Flash Attention installation is needed.

The `llm-post-training` Modal volume stores the Hugging Face cache under `/huggingface` and the saved LoRA adapter and tokenizer under `/qwen-r1-aha-moment`. Download the result with:

```bash
modal volume get llm-post-training /qwen-r1-aha-moment ./qwen-r1-aha-moment
```

The saved adapter requires the base `Qwen/Qwen2.5-3B-Instruct` model when loading it for inference. Runs use the same output directory.

For local checks in Python 3.12, install a suitable PyTorch build first, then:

```bash
python -m pip install -r requirements.txt modal==1.6.3 pytest mypy
python -m pytest -q
python -m mypy --follow-imports=silent --ignore-missing-imports --check-untyped-defs main.py reward.py modal_train.py reward_test.py
```

References: [Modal GPU requirements](https://modal.com/docs/guide/gpu), [NVIDIA image release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-01.html), and [pinned TRL API](https://huggingface.co/docs/trl/v0.25.1/en/grpo_trainer).
