from pathlib import Path

import modal


SOURCE_DIR = Path(__file__).parent
app = modal.App("llm-post-training")
volume = modal.Volume.from_name("llm-post-training", create_if_missing=True)

# Use vLLM's version-matched CUDA image. requirements.txt pins its torch and
# transformers dependencies to the versions supported by vLLM 0.32.0.
# vLLM provides python3 (but not python), while Modal's installer invokes python.
# Clear vLLM's "vllm serve" entrypoint so Modal can launch the training function.
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.32.0")
    .entrypoint([])
    .run_commands(
        "ln -sf /usr/bin/python3 /usr/local/bin/python",
        "python -m pip --version",
    )
)
image = image.pip_install_from_requirements(str(SOURCE_DIR / "requirements.txt"))
image = image.run_commands(
    "python -m pip check",
    "python -c 'from importlib.metadata import version; expected = {\"torch\": \"2.13.0\", \"transformers\": \"5.19.0\", \"vllm\": \"0.32.0\"}; assert all(version(p) == v for p, v in expected.items()), {p: version(p) for p in expected}'",
    "python -c 'from typing_extensions import Sentinel; from pydantic import BaseModel; from vllm import LLM, SamplingParams; from trl import GRPOConfig, GRPOTrainer'",
)
# Legacy Modal image builders can skip WORKDIR, so keep train.py on PYTHONPATH.
image = image.env({"HF_HOME": "/data/huggingface", "PYTHONPATH": "/app"}).workdir("/app")
image = image.add_local_file(SOURCE_DIR / "train.py", "/app/train.py")


@app.function(image=image, gpu="B300", cpu=8, memory=32768, timeout=86400, volumes={"/data": volume})
def train() -> None:
    from train import main

    main()
    volume.commit()

