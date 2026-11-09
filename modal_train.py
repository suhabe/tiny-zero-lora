from pathlib import Path

import modal


SOURCE_DIR = Path(__file__).parent
app = modal.App("llm-post-training")
volume = modal.Volume.from_name("llm-post-training", create_if_missing=True)

# NVIDIA vLLM 26.10 includes CUDA 13.4.3 and vLLM 0.31.0.
image = modal.Image.from_registry("nvcr.io/nvidia/vllm:26.10-py3")
image = image.pip_install_from_requirements(str(SOURCE_DIR / "requirements.txt"))
image = image.run_commands("python -m pip check", "python -c 'from typing_extensions import Sentinel; from pydantic import BaseModel; from vllm import LLM, SamplingParams; from trl import GRPOConfig, GRPOTrainer'")
image = image.env({"HF_HOME": "/data/huggingface"}).workdir("/app")
image = image.add_local_file(SOURCE_DIR / "train.py", "/app/train.py")


@app.function(image=image, gpu="B300", cpu=8, memory=32768, timeout=86400, volumes={"/data": volume})
def train() -> None:
    from train import main

    main()
    volume.commit()

