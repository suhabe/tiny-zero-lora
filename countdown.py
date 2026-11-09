"""Shared Countdown prompt and verifiers; importing this module needs no ML stack."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


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
