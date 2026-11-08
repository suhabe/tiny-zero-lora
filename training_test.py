from pathlib import Path
from typing import Any
from unittest.mock import Mock

from datasets import Dataset
import pytest
from trl import GRPOConfig

import main as training


@pytest.mark.parametrize(("batch_size", "use_vllm"), [(2, True), (8, True), (2, False)])
def test_training_backend_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, batch_size: int, use_vllm: bool) -> None:
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

    training.main(output_dir=str(tmp_path), max_steps=1, dataset_size=16, batch_size=batch_size, use_vllm=use_vllm)

    config = create_trainer.call_args.kwargs["args"]
    train_dataset = create_trainer.call_args.kwargs["train_dataset"]
    eval_dataset = create_trainer.call_args.kwargs["eval_dataset"]
    assert len(train_dataset) + len(eval_dataset) == 15
    assert set(list(train_dataset["prompt"]) + list(eval_dataset["prompt"])) == {"prompt"}
    assert config.use_vllm is use_vllm
    assert config.vllm_mode == "colocate"
    assert config.vllm_max_model_length == training.MAX_PROMPT_LENGTH + config.max_completion_length
    assert config.per_device_train_batch_size == batch_size
    assert config.num_generations == 2
    assert config.max_completion_length == 1024
    assert config.beta == 0.001
    assert config.gradient_checkpointing is False
    create_trainer.return_value.train.assert_called_once()
    create_trainer.return_value.save_model.assert_called_once_with(str(tmp_path))
