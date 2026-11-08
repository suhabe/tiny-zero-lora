from pathlib import Path
from typing import Any
from unittest.mock import Mock

from datasets import Dataset
import pytest
from trl import GRPOConfig
from transformers import TrainerControl, TrainerState, TrainingArguments

import main as training


@pytest.mark.parametrize(("batch_size", "use_vllm", "max_steps"), [(2, True, -1), (8, True, 1), (2, False, 1)])
def test_training_backend_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, batch_size: int, use_vllm: bool, max_steps: int) -> None:
    dataset = Dataset.from_dict({"nums": [[1, 2]] * 15 + [[3, 4]], "target": [3] * 15 + [7]})
    tokenizer = Mock(eos_token="<eos>")
    tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: "overlong" if "[3, 4]" in messages[1]["content"] else "prompt"
    tokenizer.encode.side_effect = lambda prompt, **kwargs: [1] * (257 if prompt == "overlong" else 125)
    create_trainer = Mock()

    def cpu_config(**kwargs: Any) -> GRPOConfig:
        kwargs.update(bf16=False, use_cpu=True)
        return GRPOConfig(**kwargs)

    monkeypatch.setattr(training, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(training, "GRPOConfig", cpu_config)
    monkeypatch.setattr(training, "GRPOTrainer", create_trainer)

    if max_steps == -1:
        training.main(output_dir=str(tmp_path), dataset_size=16, batch_size=batch_size, use_vllm=use_vllm)
    else:
        training.main(output_dir=str(tmp_path), max_steps=max_steps, dataset_size=16, batch_size=batch_size, use_vllm=use_vllm)

    config = create_trainer.call_args.kwargs["args"]
    train_dataset = create_trainer.call_args.kwargs["train_dataset"]
    eval_dataset = create_trainer.call_args.kwargs["eval_dataset"]
    assert len(train_dataset) + len(eval_dataset) == 15
    assert set(list(train_dataset["prompt"]) + list(eval_dataset["prompt"])) == {"prompt"}
    assert config.use_vllm is use_vllm
    assert config.vllm_mode == "colocate"
    assert config.vllm_max_model_length == training.MAX_PROMPT_LENGTH + config.max_completion_length
    assert config.per_device_train_batch_size == batch_size
    assert config.learning_rate == 2e-5
    assert config.lr_scheduler_type == "linear"
    assert config.num_train_epochs == 1
    assert config.max_steps == max_steps
    assert config.gradient_accumulation_steps == 16
    assert config.num_generations == 16
    assert config.max_completion_length == 1024
    assert config.beta == 0.001
    assert config.temperature == 1.0
    assert config.vllm_gpu_memory_utilization == 0.3
    assert config.gradient_checkpointing is False
    assert config.logging_steps == 5
    assert config.save_total_limit == 1
    assert config.disable_tqdm is True

    peft_config = create_trainer.call_args.kwargs["peft_config"]
    assert peft_config.r == 16
    assert peft_config.lora_alpha == 32
    assert peft_config.lora_dropout == 0.05
    assert peft_config.target_modules == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert peft_config.task_type == "CAUSAL_LM"

    create_trainer.return_value.train.assert_called_once()
    create_trainer.return_value.save_model.assert_called_once_with(str(tmp_path))
    create_trainer.return_value.save_state.assert_called_once()


def test_readable_step_log(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    logs = {"loss": -0.2459, "grad_norm": 0.0840676, "learning_rate": 5e-7, "step_time": 18.15729, "reward": 0.5, "rewards/format_reward_func/mean": 0.5, "rewards/equation_reward_func/mean": 0.0, "frac_reward_zero_std": 0.0, "completions/mean_length": 262.0, "completions/min_length": 102.0, "completions/max_length": 422.0, "completions/clipped_ratio": 0.0, "kl": 0.0, "entropy": 1.0414076, "clip_ratio/region_mean": 0.0}
    original_logs = logs.copy()
    state = TrainerState(global_step=1, max_steps=100, is_world_process_zero=True)

    training.ReadableLoggingCallback().on_log(TrainingArguments(output_dir=str(tmp_path), report_to="none"), state, TrainerControl(), logs=logs)

    output = capsys.readouterr().out
    assert "Step 1/100 | 18.16 s" in output
    assert "equation 0.0% | format 50.0% | total 0.500" in output
    assert "mean 262.0 tokens | range 102-422 | truncated 0.0%" in output
    assert "KL 0.0000 | entropy 1.0414 | clipped 0.0%" in output
    assert "'loss':" not in output
    assert logs == original_logs


def test_readable_final_log(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    logs = {"train_runtime": 22.15, "train_loss": -0.2459}
    state = TrainerState(global_step=1, max_steps=1, is_world_process_zero=True)

    training.ReadableLoggingCallback().on_log(TrainingArguments(output_dir=str(tmp_path), report_to="none"), state, TrainerControl(), logs=logs)

    assert capsys.readouterr().out.strip() == "Finished | 22.15 s total | training loss -0.2459"
