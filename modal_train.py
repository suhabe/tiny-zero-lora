from pathlib import Path

import modal


SOURCE_DIR = Path(__file__).parent
app = modal.App("llm-post-training")
volume = modal.Volume.from_name("llm-post-training", create_if_missing=True)

image = (
    modal.Image
        .from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12")
        .apt_install("libnuma1")
        .pip_install_from_requirements(str(SOURCE_DIR / "requirements.txt"))
        .env({"HF_HOME": "/data/huggingface", "PYTHONPATH": "/app"}).workdir("/app")
        .add_local_file(SOURCE_DIR / "train.py", "/app/train.py")
        .add_local_file(SOURCE_DIR / "train_callback.py", "/app/train_callback.py")
)


@app.function(image=image, gpu="B300", cpu=8, memory=32768, timeout=86400, volumes={"/data": volume})
def train() -> None:
    from train import main

    main()
    volume.commit()
