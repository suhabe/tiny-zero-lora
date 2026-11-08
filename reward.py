import re


def format_reward_func(completions: list[str], **kwargs: object) -> list[float]:
    """Reward the expected thinking and answer tags; the prompt already ends in <think>."""
    pattern = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)</think>\n<answer>([\s\S]*?)</answer>$"
    return [1.0 if re.search(pattern, "<think>" + completion, re.DOTALL) else 0.0 for completion in completions]


def equation_reward_func(completions: list[str], target: list[int], nums: list[list[int]], **kwargs: object) -> list[float]:
    """Reward expressions that use every supplied number once and evaluate to the target."""
    rewards: list[float] = []

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
