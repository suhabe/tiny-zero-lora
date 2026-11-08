from collections.abc import Callable
from typing import cast

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward import equation_reward_func, format_reward_func


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 1024


def generate_r1_prompt(tokenizer: PreTrainedTokenizerBase, numbers: list[int], target: int) -> dict[str, str | int]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Think through the problem before answering."},
        {"role": "user", "content": f"Using the numbers {numbers}, create an expression that equals {target}. Use only basic arithmetic (+, -, *, /) and each number exactly once. Show your work in <think> </think> tags. Return only the expression, without an equals sign, in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."},
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    return {"prompt": prompt, "target": target}


def main(output_dir: str = "qwen-r1-aha-moment", max_steps: int = 100, dataset_size: int = 50000, batch_size: int = 2, use_vllm: bool = True) -> None:
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    dataset = dataset.shuffle(seed=42).select(range(min(dataset_size, len(dataset))))

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(lambda row: generate_r1_prompt(tokenizer, row["nums"], row["target"]))
    dataset = dataset.filter(lambda row: len(tokenizer.encode(row["prompt"], add_special_tokens=False)) <= MAX_PROMPT_LENGTH)
    datasets = dataset.train_test_split(test_size=0.1, seed=42)

    training_args = GRPOConfig(
        output_dir=output_dir,
        model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        logging_steps=1,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        bf16=True,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=2,
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.1,
        vllm_max_model_length=MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH,
        beta=0.001,
        report_to="none",
    )
    # TRL's callback type also allows None for skipped rewards.
    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=[format_reward_func, equation_reward_func],
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        processing_class=tokenizer,
        peft_config=LoraConfig(task_type="CAUSAL_LM"),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

