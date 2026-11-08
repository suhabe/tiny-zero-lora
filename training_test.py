from pathlib import Path
from typing import Any
from unittest.mock import Mock

from datasets import Dataset
import pytest
from trl import GRPOConfig, GRPOTrainer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainerControl, TrainerState, TrainingArguments
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch

import main as training


@pytest.mark.parametrize(("batch_size", "use_vllm", "max_steps"), [(2, True, -1), (8, True, 1), (2, False, 1)])
def test_training_backend_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, batch_size: int, use_vllm: bool, max_steps: int) -> None:
    dataset = Dataset.from_dict({"nums": [[1, 2]] * 15 + [[3, 4]], "target": [3] * 15 + [7]})
    tokenizer = Mock(eos_token="<eos>")
    tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: "overlong" if "[3, 4]" in messages[1]["content"] else "prompt"
    tokenizer.encode.side_effect = lambda prompt, **kwargs: [1] * (257 if prompt == "overlong" else 125)
    create_trainer = Mock()
    create_trainer.return_value.state = TrainerState(global_step=1)

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
    assert config.logging_steps == 1
    assert config.eval_strategy == "steps"
    assert config.eval_steps == 10
    assert config.eval_on_start is True
    assert config.num_generations_eval == 1
    assert config.per_device_eval_batch_size == 1
    assert config.save_total_limit == 1
    assert config.disable_tqdm is True

    peft_config = create_trainer.call_args.kwargs["peft_config"]
    assert peft_config.r == 16
    assert peft_config.lora_alpha == 32
    assert peft_config.lora_dropout == 0.05
    assert peft_config.target_modules == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert peft_config.task_type == "CAUSAL_LM"

    create_trainer.return_value.train.assert_called_once()
    create_trainer.return_value.evaluate.assert_called_once_with()
    create_trainer.return_value.save_model.assert_called_once_with(str(tmp_path))
    create_trainer.return_value.save_state.assert_called_once()


@pytest.mark.parametrize("max_steps", [10, 11])
def test_held_out_evaluation_schedule(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], max_steps: int) -> None:
    torch.set_num_threads(1)
    dataset = Dataset.from_dict({"nums": [[number, 1] for number in range(40)], "target": list(range(1, 41))})
    backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2, "puzzle": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>", chat_template="{{ messages[-1]['content'] }}")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=64, n_embd=8, n_layer=1, n_head=1, eos_token_id=1, pad_token_id=1))
    model.config.name_or_path = str(tmp_path / "base-model")
    model.save_pretrained(model.config.name_or_path)
    trainers: list[GRPOTrainer] = []

    def cpu_config(**kwargs: Any) -> GRPOConfig:
        kwargs.update(bf16=False, use_cpu=True, use_vllm=False, model_init_kwargs=None, gradient_accumulation_steps=1, num_generations=2, max_completion_length=4, save_strategy="no", dataloader_pin_memory=False)
        return GRPOConfig(**kwargs)

    def cpu_trainer(**kwargs: Any) -> GRPOTrainer:
        kwargs.update(model=model, peft_config=None)
        trainer = GRPOTrainer(**kwargs)
        trainers.append(trainer)
        return trainer

    monkeypatch.setattr(training, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(training, "GRPOConfig", cpu_config)
    monkeypatch.setattr(training, "GRPOTrainer", cpu_trainer)
    monkeypatch.setattr(training, "EVAL_SIZE", 3)

    training.main(output_dir=str(tmp_path), max_steps=max_steps, dataset_size=40, use_vllm=False)

    trainer = trainers[0]
    evaluation_logs = [entry for entry in trainer.state.log_history if "eval_rewards/equation_reward_func/mean" in entry]
    assert [entry["step"] for entry in evaluation_logs] == ([0, 10] if max_steps == 10 else [0, 10, 11])
    assert all(entry["eval_rewards/equation_reward_func/mean"] == 0.0 for entry in evaluation_logs)
    assert isinstance(trainer.eval_dataset, Dataset)
    assert isinstance(trainer.train_dataset, Dataset)
    assert len(trainer.eval_dataset) == 3
    assert set(trainer.train_dataset["target"]).isdisjoint(trainer.eval_dataset["target"])
    assert trainer.generation_config.temperature == 1.0
    assert trainer.generation_config.max_new_tokens == 4
    assert TrainerState.load_from_json(str(tmp_path / "trainer_state.json")).log_history == trainer.state.log_history
    assert "Held-out step 0 | solve rate 0.0%" in capsys.readouterr().out


def test_readable_evaluation_log(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    logs = {"eval_rewards/equation_reward_func/mean": 0.25, "eval_rewards/format_reward_func/mean": 0.75, "eval_runtime": 12.0}
    state = TrainerState(global_step=10, max_steps=100, is_world_process_zero=True)

    training.ReadableLoggingCallback().on_log(TrainingArguments(output_dir=str(tmp_path), report_to="none"), state, TrainerControl(), logs=logs)

    assert capsys.readouterr().out.strip() == "Held-out step 10 | solve rate 25.0% | format 75.0% | 12.00 s"


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
