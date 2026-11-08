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

B300 requires CUDA 13.1 or newer. The launcher uses NVIDIA's `nvcr.io/nvidia/vllm:26.01-py3` image with CUDA 13.1.1, PyTorch, and vLLM 0.13.0. TRL 1.0.0 supports this vLLM version. The actual image reports `0.13.0+faa43dbf.nv26.1.cu131`; the NVIDIA release notes list a different version. Keep these preinstalled GPU builds; do not install the PyPI vLLM wheel over them. The Hugging Face versions are pinned in `requirements.txt`. LoRA training uses BF16 weights and PyTorch SDPA attention. Generation uses vLLM colocated on the same GPU, reserving 30% of GPU memory and a 1280-token context. Prompts exceeding 256 tokens are filtered before training; the deprecated trainer truncation option is no longer used. Gradient checkpointing is enabled to reduce activation memory. No additional bitsandbytes or Flash Attention installation is needed.

The `llm-post-training` Modal volume stores the Hugging Face cache under `/huggingface` and the saved LoRA adapter and tokenizer under `/qwen-r1-aha-moment`. Download the result with:

```bash
modal volume get llm-post-training /qwen-r1-aha-moment ./qwen-r1-aha-moment
```

The saved adapter requires the base `Qwen/Qwen2.5-3B-Instruct` model when loading it for inference. Runs use the same output directory.

For local checks in Python 3.12, install a suitable PyTorch build first, then:

```bash
python -m pip install -r requirements.txt modal==1.6.3 pytest mypy
python -m pytest -q
python -m mypy --follow-imports=silent --ignore-missing-imports --check-untyped-defs main.py reward.py modal_train.py reward_test.py training_test.py
```

References: [Modal GPU requirements](https://modal.com/docs/guide/gpu), [NVIDIA image release notes](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-01.html), and [pinned TRL API](https://huggingface.co/docs/trl/v1.0.0/en/grpo_trainer).

Compare generation backends over several steps:

```bash
modal run modal_train.py --max-steps 10 --dataset-size 128
modal run modal_train.py --max-steps 10 --dataset-size 128 --no-use-vllm
```

vLLM compiles kernels and captures CUDA graphs during startup. Compare warmed-up `step_time` and completion lengths, rather than one-step total runtime. A fresh vLLM run can take longer to start than the Transformers backend. Both paths use 16 completions per prompt, 16 gradient accumulation steps, temperature 1.0, the 1024-token completion limit, and KL penalty 0.001. The learning rate is 2e-5 with the default linear schedule; logging runs every five steps and only one checkpoint is retained. Calling `main()` directly trains for one epoch by default (`max_steps=-1`); the Modal launcher retains its explicit 100-step default.

For throughput experiments, increase the per-device batch size:

```bash
modal run modal_train.py --max-steps 10 --dataset-size 128 --batch-size 8
```

With 16 gradient accumulation steps and 16 generations per prompt, batch size eight processes eight prompt groups per optimizer step instead of two. Compare completions or tokens per second: more work per step does not guarantee fewer seconds per step. Watch equation reward and truncation as well as timing. Backend changes can produce different completion lengths, so latency comparisons are approximate.

The local tests validate both backend configurations and the reward functions. Running with `use_vllm=False` also allows a tiny-model CPU smoke test without installing vLLM. The actual NVIDIA image and B300 speedup require a Modal GPU run to verify.

The image build runs `pip check` and imports Pydantic, vLLM, and the GRPO trainer before starting the training function. `transformers==4.57.1` matches the NVIDIA vLLM package's exact dependency. `typing_extensions==4.15.0` restores `Sentinel`, which Pydantic needs, and `aiohttp==3.13.3` restores the networking version expected by the TRL vLLM integration after Modal's legacy bootstrap.

Modal also recommends updating the workspace's [Image Builder](https://modal.com/settings/image-config). The explicit dependency pins above repair the observed legacy-builder downgrades without requiring a workspace-wide settings change.

Training prints a compact summary instead of raw dictionaries and an overlapping progress bar:

```text
Step 1/100 | 18.16 s | loss -0.2459 | grad 0.0841 | lr 5.00e-07
  Rewards | equation 0.0% | format 50.0% | total 0.500 | zero-std 0.0%
  Outputs | mean 262.0 tokens | range 102-422 | truncated 0.0%
  Policy  | KL 0.0000 | entropy 1.0414 | clipped 0.0%
```

All metrics retain their original precision in `log_history` inside `trainer_state.json`, saved alongside the adapter when training finishes. This includes sampling diagnostics omitted from the console summary. Download it with:

```bash
modal volume get llm-post-training /qwen-r1-aha-moment/trainer_state.json ./trainer_state.json
```
