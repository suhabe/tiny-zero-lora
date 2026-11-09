"""Small shared lesson utilities. ML dependencies are imported only when needed."""
import argparse
import json
import statistics
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"


def group_advantages(rewards: list[float]) -> list[float]:
    """Diagnostic for TRL's default sum-then-group-normalize, not a training loss."""
    if len(rewards) < 2:
        raise ValueError("A training group needs at least two completions")
    mean, std = statistics.mean(rewards), statistics.stdev(rewards)
    return [(reward - mean) / (std + 1e-4) for reward in rewards]


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--run-dir", type=Path, default=Path("runs/tutorial"),
                        help="One experiment's saved data, tokenizer, scores and checkpoints")
    return result


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def check_training_plan(path: Path, plan: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Training settings changed; use a new --run-dir. Keep the scheduler horizon fixed.")
        return
    write_json(path, plan)


def compare_evaluations(baseline: dict, final: dict) -> dict:
    if baseline["protocol"] != final["protocol"]:
        raise ValueError("Evaluation protocols differ; use identical data, revision and decoding settings")
    before, after = baseline["rows"], final["rows"]
    identity = lambda rows: [(r["index"], r["nums"], r["target"]) for r in rows]
    if not before or identity(before) != identity(after):
        raise ValueError("Evaluation rows differ or are empty")
    n = len(before)
    b = sum(r["equation"] for r in before) / n
    f = sum(r["equation"] for r in after) / n
    return {"n": n, "baseline_solve_rate": b, "final_solve_rate": f, "delta": f-b,
            "new_solves": sum(a["equation"] > b["equation"] for b,a in zip(before, after)),
            "regressions": sum(a["equation"] < b["equation"] for b,a in zip(before, after))}


def load_data(run_dir: Path):
    """Create once, then always reuse the exact prepared split and tokenizer."""
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    manifest_path = run_dir / "data.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer", padding_side="left")
        return tokenizer, load_from_disk(str(run_dir / "dataset")), manifest

    from train import prepare_data
    from huggingface_hub import HfApi
    # Transformers 5.19 no longer preserves tokenizer.init_kwargs['_commit_hash'].
    # Resolve first and load explicitly at that SHA, including for the tokenizer.
    model_revision = HfApi().model_info(MODEL_ID).sha
    if not model_revision:
        raise RuntimeError("The Hub did not return a base model revision")
    tokenizer, split = prepare_data(model_revision=model_revision)
    run_dir.mkdir(parents=True, exist_ok=True)
    split.save_to_disk(str(run_dir / "dataset"))
    tokenizer.save_pretrained(run_dir / "tokenizer")
    task_key = lambda row: (tuple(sorted(row["nums"])), row["target"])
    train_keys = {task_key(row) for row in split["train"]}
    overlap = sum(task_key(row) in train_keys for row in split["test"])
    manifest = {
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "dataset_id": DATASET_ID,
        "split_fingerprints": {name: data._fingerprint for name, data in split.items()},
        "sizes": {name: len(data) for name, data in split.items()},
        "validation_rows_with_equivalent_training_task": overlap,
        "note": "Saved rows define this experiment. Validation is reused during training.",
    }
    write_json(manifest_path, manifest)
    return tokenizer, split, manifest
