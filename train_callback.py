import os
import sys
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class TrainingMonitorCallback(TrainerCallback):
    COLORS: dict[str, str] = {
        "primary": "\033[1;32m",   # Bold green: solve rate and reward
        "important": "\033[33m",   # Yellow: loss, gradients, and truncation
        "info": "\033[36m",        # Cyan: supporting metrics
        "muted": "\033[2m",        # Dim: diagnostics
        "critical": "\033[1;31m",  # Bold red: high GPU memory usage
    }

    def __init__(self, color: bool = True) -> None:
        self.color = color

    def _metric(self, value: float, spec: str, color: str = "muted") -> str:
        text = format(value, spec)
        return f"{self.COLORS[color]}{text}\033[0m" if self.color else text

    def _uses_cuda(self, args: TrainingArguments, state: TrainerState) -> bool:
        return state.is_world_process_zero and args.device.type == "cuda" and torch.cuda.is_available()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object) -> None:
        if self._uses_cuda(args, state):
            torch.cuda.reset_peak_memory_stats(args.device)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object) -> None:
        if not self._uses_cuda(args, state):
            return

        device = args.device
        gib = 1024**3
        free, total = torch.cuda.mem_get_info(device)
        used = total - free
        utilization = used / total
        memory_color = "critical" if utilization >= 0.90 else "important" if utilization >= 0.80 else "info"

        def memory(value: int, color: str = "muted") -> str:
            return f"{self._metric(value / gib, '.2f', color)} GiB"

        lines = [
            f"GPU · Step {state.global_step}",
            f"  Device     | used {memory(used, memory_color)} / {memory(total)} | free {memory(free)} | usage {self._metric(utilization, '.1%', memory_color)}",
            f"  PyTorch    | allocated {memory(torch.cuda.memory_allocated(device), 'info')} | reserved {memory(torch.cuda.memory_reserved(device))}",
            f"  Step peaks | allocated {memory(torch.cuda.max_memory_allocated(device), 'important')} | reserved {memory(torch.cuda.max_memory_reserved(device))}",
        ]
        print("\n".join(lines) + "\n", flush=True)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict[str, float] | None = None, **kwargs: object) -> None:
        if logs is None or not state.is_world_process_zero:
            return

        metric = self._metric

        if "eval_rewards/equation_reward_func/mean" in logs:
            lines = [
                f"Held-out evaluation · Step {state.global_step}",
                f"  Results    | solve rate {metric(logs['eval_rewards/equation_reward_func/mean'], '.1%', 'primary')} | format {metric(logs['eval_rewards/format_reward_func/mean'], '.1%', 'info')} | time {metric(logs['eval_runtime'], '.2f')} s",
            ]

        elif "train_runtime" in logs:
            lines = [
                "Training finished",
                f"  Summary    | total time {metric(logs['train_runtime'], '.2f', 'info')} s | training loss {metric(logs['train_loss'], '.4f', 'important')}",
            ]

        elif "loss" in logs:
            lines = [
                f"Training · Step {state.global_step}/{state.max_steps}",
                f"  Optimize   | loss {metric(logs['loss'], '.4f', 'important')} | grad norm {metric(logs['grad_norm'], '.4f', 'important')} | lr {metric(logs['learning_rate'], '.2e')} | time {metric(logs['step_time'], '.2f')} s",
                f"  Rewards    | solve rate {metric(logs['rewards/equation_reward_func/mean'], '.1%', 'primary')} | format {metric(logs['rewards/format_reward_func/mean'], '.1%', 'info')} | total {metric(logs['reward'], '.3f', 'primary')} | zero-std {metric(logs['frac_reward_zero_std'], '.1%')}",
                f"  Outputs    | mean {metric(logs['completions/mean_length'], '.1f', 'info')} tokens | range {metric(logs['completions/min_length'], '.0f')}–{metric(logs['completions/max_length'], '.0f')} | truncated {metric(logs['completions/clipped_ratio'], '.1%', 'important')}",
                f"  Policy     | KL {metric(logs['kl'], '.4f')} | entropy {metric(logs['entropy'], '.4f')} | clipped {metric(logs['clip_ratio/region_mean'], '.1%')}",
            ]

        else:
            return

        print("\n".join(lines) + "\n", flush=True)