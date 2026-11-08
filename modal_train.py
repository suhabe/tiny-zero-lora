from pathlib import Path

import modal


SOURCE_DIR = Path(__file__).parent
app = modal.App("llm-post-training")
volume = modal.Volume.from_name("llm-post-training", create_if_missing=True)

# Keep NVIDIA's matching CUDA 13.1.1, PyTorch, and vLLM 0.13.0 builds.
image = modal.Image.from_registry("nvcr.io/nvidia/vllm:26.01-py3")
image = image.pip_install_from_requirements(str(SOURCE_DIR / "requirements.txt"))
image = image.run_commands("python -m pip check", "python -c 'from typing_extensions import Sentinel; from pydantic import BaseModel; from vllm import LLM, SamplingParams; from trl import GRPOConfig, GRPOTrainer'")
image = image.env({"HF_HOME": "/data/huggingface"}).workdir("/app")
image = image.add_local_file(SOURCE_DIR / "main.py", "/app/main.py")
image = image.add_local_file(SOURCE_DIR / "reward.py", "/app/reward.py")


@app.function(image=image, gpu="B300", cpu=8, memory=32768, timeout=86400, volumes={"/data": volume})
def train(max_steps: int = 100, dataset_size: int = 50000, batch_size: int = 2, use_vllm: bool = True) -> None:
    from main import main

    main(output_dir="/data/qwen-r1-aha-moment", max_steps=max_steps, dataset_size=dataset_size, batch_size=batch_size, use_vllm=use_vllm)
    volume.commit()

