from typing import cast

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import PrinterCallback
from trl import GRPOConfig, GRPOTrainer

from reward import equation_reward_func, format_reward_func


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 1024
EVAL_STEPS = 10
EVAL_SIZE = 100


def generate_r1_prompt(tokenizer: PreTrainedTokenizerBase, numbers: list[int], target: int) -> dict[str, str | int]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Think through the problem before answering."},
        {"role": "user", "content": f"Using the numbers {numbers}, create an expression that equals {target}. Use only basic arithmetic (+, -, *, /) and each number exactly once. Show your work in <think> </think> tags. Return only the expression, without an equals sign, in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."},
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    return {"prompt": cast(str, prompt), "target": target}


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


def main(output_dir: str = "qwen-r1-aha-moment", max_steps: int = -1, dataset_size: int = 50000, batch_size: int = 2, use_vllm: bool = True) -> None:
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    dataset = dataset.shuffle(seed=42).select(range(min(dataset_size, len(dataset))))

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(lambda row: generate_r1_prompt(tokenizer, row["nums"], row["target"]))
    dataset = dataset.filter(lambda row: len(tokenizer.encode(row["prompt"], add_special_tokens=False)) <= MAX_PROMPT_LENGTH)
    datasets = dataset.train_test_split(test_size=0.1, seed=42)
    eval_dataset = datasets["test"].select(range(min(EVAL_SIZE, len(datasets["test"]))))

    grpo_config: GRPOConfig = GRPOConfig(
        output_dir=output_dir,
        model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        eval_on_start=True,
        per_device_eval_batch_size=1,
        num_generations_eval=1,
        disable_tqdm=True,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=16,
        gradient_checkpointing=False,
        bf16=True,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=16,
        temperature=1.0,
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH,
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
        model=MODEL_ID,
        reward_funcs=[format_reward_func, equation_reward_func],
        args=grpo_config,
        train_dataset=datasets["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[ReadableLoggingCallback()],
    )
    trainer.remove_callback(PrinterCallback)

    trainer.train()

    if trainer.state.global_step % EVAL_STEPS != 0:
        trainer.evaluate()

    trainer.save_state()
    trainer.save_model(grpo_config.output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
