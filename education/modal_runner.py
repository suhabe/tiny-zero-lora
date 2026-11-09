"""Run the existing lesson CLIs against a shared Modal volume.

This module has no Modal/ML imports so command routing can be tested locally.
"""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

MODULES = {
    "preflight": "00_preflight", "dataset": "01_dataset", "prompts": "02_prompts",
    "rewards": "03_rewards", "baseline": "04_evaluate", "configuration": "05_code_and_config",
    "step": "06_train", "metrics": "07_metrics", "performance": "08_performance",
    "training": "06_train", "final": "04_evaluate",
}
CHECKPOINT_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")


def checkpoint_path(training_dir: Path, resume: str) -> Path | None:
    if resume == "none":
        return None
    if resume == "latest":
        candidates = [p for p in training_dir.glob("checkpoint-*")
                      if re.fullmatch(r"checkpoint-\d+", p.name) and p.is_dir()]
        complete = [p for p in candidates if all((p / name).is_file() for name in CHECKPOINT_FILES)]
        if complete:
            return max(complete, key=lambda p: int(p.name.split("-")[-1]))
        if candidates:
            raise ValueError("No complete checkpoint found; inspect this run before resuming")
        return None
    if not re.fullmatch(r"checkpoint-\d+", resume):
        raise ValueError("resume must be 'latest', 'none', or a checkpoint-N directory name")
    path = training_dir / resume
    if not all((path / name).is_file() for name in CHECKPOINT_FILES):
        raise ValueError(f"Incomplete or missing checkpoint: {path}")
    return path


def lesson_command(lesson: str, *, run_name: str = "tutorial", max_steps: int = 0,
                   limit: int = 0, resume: str = "latest", name: str = "final",
                   runs_root: Path = Path("/data/runs")) -> list[str]:
    if lesson not in MODULES:
        raise ValueError(f"Unknown lesson: {lesson}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("run_name must start with a letter/digit and contain only letters, digits, _, . or -")
    if max_steps < 0 or limit < 0:
        raise ValueError("max_steps and limit must be nonnegative (0 means use the saved/default setting)")
    run_dir = runs_root / run_name
    command = [sys.executable, "-m", "education." + MODULES[lesson]]
    if lesson not in {"preflight", "rewards", "configuration", "performance"}:
        command += ["--run-dir", str(run_dir)]
    if lesson in {"step", "training"}:
        if not max_steps:
            plan = run_dir / "training_plan.json"
            max_steps = json.loads(plan.read_text())["max_steps"] if plan.exists() else 2000
        command += ["--max-steps", str(max_steps)]
        checkpoint = checkpoint_path(run_dir / "training", resume)
        if checkpoint is not None:
            command += ["--resume", str(checkpoint)]
        if lesson == "step":
            command.append("--one-step")
    if lesson == "final":
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("name must be a simple result filename stem")
        command += ["--adapter", str(run_dir / "adapter"), "--name", name]
        baseline = run_dir / "baseline.json"
        if not limit and baseline.exists():
            limit = json.loads(baseline.read_text())["protocol"]["n"]
    if lesson in {"baseline", "final"} and limit:
        command += ["--limit", str(limit)]
    return command


def run_lesson(lesson: str, *, volume, source_dir: Path = Path("/app"),
               runs_root: Path = Path("/data/runs"), **options) -> None:
    # A warm container must refresh to see the previous lesson's committed files.
    volume.reload()
    try:
        command = lesson_command(lesson, runs_root=runs_root, **options)
        print("Running:", shlex.join(command), flush=True)
        print("Persistent experiment:", runs_root / options.get("run_name", "tutorial"), flush=True)
        subprocess.run(command, cwd=source_dir, check=True,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
    finally:
        # The child has exited/closed its files, even after a Python error. Preserve
        # completed checkpoints/diagnostics without converting failures to success.
        volume.commit()
