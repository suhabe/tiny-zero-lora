"""Run individual educational lessons: modal run modal_lessons.py::<lesson>.

Only the Modal client is needed locally. All model/data dependencies run remotely.
"""
from pathlib import Path

import modal

from modal_train import image as training_image, volume

SOURCE_DIR = Path(__file__).parent
app = modal.App("tiny-zero-lora-lessons")
image = training_image.add_local_dir(
    SOURCE_DIR / "education", "/app/education", ignore=["**/__pycache__/**", "**/*.pyc"]
)
# The entrypoint imports this module to reuse the image/volume definitions.
image = image.add_local_file(SOURCE_DIR / "modal_train.py", "/app/modal_train.py")
image = image.add_local_file(SOURCE_DIR / "requirements.txt", "/app/requirements.txt")
CPU = dict(image=image, cpu=2, memory=8192, timeout=3600, volumes={"/data": volume})
GPU = dict(image=image, gpu="B300", cpu=8, memory=32768, timeout=86400, volumes={"/data": volume})


def _run(lesson: str, run_name: str = "tutorial", **options) -> None:
    from education.modal_runner import run_lesson

    run_lesson(lesson, volume=volume, run_name=run_name, **options)


@app.function(**GPU)
def preflight() -> None:
    """Inspect the actual B300/CUDA environment; do not run a model."""
    _run("preflight")


@app.function(**CPU)
def dataset(run_name: str = "tutorial") -> None:
    """Lesson 1: inspect real rows and save prepared data/tokenizer on the volume."""
    _run("dataset", run_name)


@app.function(**CPU)
def prompts(run_name: str = "tutorial") -> None:
    """Lesson 2: render actual Qwen prompts and inspect token IDs/masks."""
    _run("prompts", run_name)


@app.function(**CPU)
def rewards() -> None:
    """Lesson 3: exercise the production reward functions on hand-built examples."""
    _run("rewards")


@app.function(**GPU)
def baseline(run_name: str = "tutorial", limit: int = 0) -> None:
    """Lesson 4: sample the initial model; limit=0 evaluates all validation rows."""
    _run("baseline", run_name, limit=limit)


@app.function(**CPU)
def configuration() -> None:
    """Lesson 5: print the actual training code and configuration."""
    _run("configuration")


@app.function(**GPU)
def step(run_name: str = "tutorial", max_steps: int = 0, resume: str = "latest") -> None:
    """Lesson 6: one additional optimizer update; reuse saved horizon or default 2000."""
    _run("step", run_name, max_steps=max_steps, resume=resume)


@app.function(**CPU)
def metrics(run_name: str = "tutorial") -> None:
    """Lesson 7: inspect persisted real metrics and sampled rollout groups."""
    _run("metrics", run_name)


@app.function(**CPU)
def performance() -> None:
    """Lesson 8: explain the training code's memory/throughput tradeoffs."""
    _run("performance")


@app.function(**GPU)
def training(run_name: str = "tutorial", max_steps: int = 0, resume: str = "latest") -> None:
    """Lesson 9: resume the latest complete checkpoint and saved total horizon by default."""
    _run("training", run_name, max_steps=max_steps, resume=resume)


@app.function(**GPU)
def final_evaluation(run_name: str = "tutorial", limit: int = 0, name: str = "final") -> None:
    """Lesson 10: reload the adapter; limit=0 reuses the baseline evaluation size."""
    _run("final", run_name, limit=limit, name=name)
