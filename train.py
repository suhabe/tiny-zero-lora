from typing import cast

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import PrinterCallback
from trl import GRPOConfig, GRPOTrainer
import re


# TRL forwards trainer metadata and every dataset column to both callbacks.
def format_reward_func(completions: list[str], **kwargs: object) -> list[float | None]:
    """Reward the expected thinking and answer tags; the prompt already ends in <think>."""
    pattern = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)</think>\n<answer>([\s\S]*?)</answer>$"
    return [1.0 if re.search(pattern, "<think>" + completion, re.DOTALL) else 0.0 for completion in completions]


def equation_reward_func(completions: list[str], target: list[int], nums: list[list[int]], **kwargs: object) -> list[float | None]:
    """Reward expressions that use every supplied number once and evaluate to the target."""
    rewards: list[float | None] = []

    for completion, expected, numbers in zip(completions, target, nums):
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match is None:
            rewards.append(0.0)
            continue

        expression = match.group(1).strip()
        used_numbers = [int(number) for number in re.findall(r"\d+", expression)]
        if sorted(used_numbers) != sorted(numbers) or re.fullmatch(r"[0-9+\-*/()\s]+", expression) is None or "**" in expression or "//" in expression:
            rewards.append(0.0)
            continue

        try:
            result = eval(expression, {"__builtins__": None}, {})
            rewards.append(1.0 if abs(float(result) - expected) < 1e-5 else 0.0)
        except (ArithmeticError, SyntaxError, TypeError, ValueError):
            rewards.append(0.0)

    return rewards



def generate_r1_prompt(tokenizer: PreTrainedTokenizerBase, numbers: list[int], target: int) -> dict[str, str | int]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Think through the problem before answering."},
        {"role": "user", "content": f"Using the numbers {numbers}, create an expression that equals {target}. Use only basic arithmetic (+, -, *, /) and each number exactly once. Show your work in <think> </think> tags. Return only the expression, without an equals sign, in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."},
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    return {"prompt": cast(str, prompt), "target": target}


class GPUMemoryCallback(TrainerCallback):
    """Log device memory snapshots and this process's PyTorch peaks per optimizer step."""

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object) -> None:
        if state.is_world_process_zero and args.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(args.device)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object) -> None:
        if not state.is_world_process_zero or args.device.type != "cuda" or not torch.cuda.is_available():
            return

        gib = 1024**3
        device = args.device
        free, total = torch.cuda.mem_get_info(device)
        print(
            f"  GPU step {state.global_step}"
            f" | device used {(total - free) / gib:.1f}/{total / gib:.1f} GiB"
            f" | device free {free / gib:.1f} GiB"
            f" | torch allocated {torch.cuda.memory_allocated(device) / gib:.1f} GiB"
            f" | torch reserved {torch.cuda.memory_reserved(device) / gib:.1f} GiB"
            f" | torch peak allocated {torch.cuda.max_memory_allocated(device) / gib:.1f} GiB"
            f" | torch peak reserved {torch.cuda.max_memory_reserved(device) / gib:.1f} GiB",
            flush=True,
        )


class ReadableLoggingCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict[str, float] | None = None, **kwargs: object) -> None:
        if logs is None or not state.is_world_process_zero:
            return

        if "eval_rewards/equation_reward_func/mean" in logs:
            print(f"Held-out step {state.global_step} | solve rate {logs['eval_rewards/equation_reward_func/mean']:.1%} | format {logs['eval_rewards/format_reward_func/mean']:.1%} | {logs['eval_runtime']:.2f} s", flush=True)
            return

        if "train_runtime" in logs:
            print(f"Finished | {logs['train_runtime']:.2f} s total | training loss {logs['train_loss']:.4f}", flush=True)
            return

        if "loss" not in logs:
            return

        lines = [
            f"Step {state.global_step}/{state.max_steps} | {logs['step_time']:.2f} s | loss {logs['loss']:.4f} | grad {logs['grad_norm']:.4f} | lr {logs['learning_rate']:.2e}",
            f"  Rewards | equation {logs['rewards/equation_reward_func/mean']:.1%} | format {logs['rewards/format_reward_func/mean']:.1%} | total {logs['reward']:.3f} | zero-std {logs['frac_reward_zero_std']:.1%}",
            f"  Outputs | mean {logs['completions/mean_length']:.1f} tokens | range {logs['completions/min_length']:.0f}-{logs['completions/max_length']:.0f} | truncated {logs['completions/clipped_ratio']:.1%}",
            f"  Policy  | KL {logs['kl']:.4f} | entropy {logs['entropy']:.4f} | clipped {logs['clip_ratio/region_mean']:.1%}",
        ]
        print("\n".join(lines), flush=True)


def main() -> None:

    output_dir = "qwen-r1-aha-moment"
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    max_prompt_length = 256
    max_completion_length = 1024
    dataset_size = 2000


    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    dataset = dataset.shuffle(seed=42).select(range(min(dataset_size, len(dataset))))

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(lambda row: generate_r1_prompt(tokenizer, row["nums"], row["target"]))
    dataset = dataset.filter(lambda row: len(tokenizer.encode(row["prompt"], add_special_tokens=False)) <= max_prompt_length)
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    eval_steps = 10

    grpo_config: GRPOConfig = GRPOConfig(
        output_dir=output_dir,
        model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
        learning_rate=2e-5,
        max_steps=2000,
        lr_scheduler_type="linear",
        num_train_epochs=1,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=10,
        eval_on_start=True,
        per_device_eval_batch_size=100,
        num_generations_eval=1,
        disable_tqdm=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_generations=16,
        gradient_checkpointing=False,
        bf16=True,
        max_completion_length=max_completion_length,
        temperature=1.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=max_prompt_length + max_completion_length,
        beta=0.001,
        save_total_limit=1,
        report_to="none",
    )

    peft_config: LoraConfig = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=[format_reward_func, equation_reward_func],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[ReadableLoggingCallback(), GPUMemoryCallback()],
    )
    trainer.remove_callback(PrinterCallback)

    trainer.train()

    if trainer.state.global_step % eval_steps != 0:
        trainer.evaluate()

    trainer.save_state()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()