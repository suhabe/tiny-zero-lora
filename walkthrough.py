"""Run with: python walkthrough.py (using the repo's training environment)."""

from transformers import AutoTokenizer

from train import equation_reward_func, format_reward_func, generate_r1_prompt


print("=== generate_r1_prompt ===")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
for numbers, target in [([1, 2, 3], 6), ([2, 3, 4], 14)]:
    result = generate_r1_prompt(tokenizer, numbers, target)
    print(f"\nNumbers: {numbers}, target: {result['target']}")
    print(result["prompt"])


print("\n=== format_reward_func and equation_reward_func ===")
# The prompt already supplies <think>, so completions start after that tag.
# Each scenario shows that formatting and arithmetic are scored independently.
scenarios = [
    (
        "Correct format and equation",
        "Add all three numbers.</think>\n<answer>1 + 2 + 3</answer>",
        [1, 2, 3], 6,
    ),
    (
        "Correct format, wrong result",
        "Subtract the last number.</think>\n<answer>1 + 2 - 3</answer>",
        [1, 2, 3], 6,
    ),
    (
        "Correct equation, missing thinking block",
        "<answer>1 + 2 + 3</answer>",
        [1, 2, 3], 6,
    ),
    (
        "Reaches the target but leaves out a number",
        "Use only two numbers.</think>\n<answer>1 + 2</answer>",
        [1, 2, 3], 3,
    ),
    (
        "Duplicate inputs: each occurrence must be used",
        "Add both twos and the three.</think>\n<answer>2 + 2 + 3</answer>",
        [2, 2, 3], 7,
    ),
    (
        "Division by zero",
        "Try division.</think>\n<answer>3 / (2 - 2)</answer>",
        [3, 2, 2], 0,
    ),
    (
        "Missing answer tags",
        "1 + 2 + 3",
        [1, 2, 3], 6,
    ),
]

for label, completion, numbers, target in scenarios:
    print(f"\n{label}")
    print(f"  nums={numbers}, target={target}")
    print(f"  completion={completion!r}")
    print("  format reward:", format_reward_func([completion]))
    print("  equation reward:", equation_reward_func([completion], target=[target], nums=[numbers]))
