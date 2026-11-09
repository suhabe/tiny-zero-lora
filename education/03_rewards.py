"""Run the production reward functions with hand-built completions (no downloads)."""
from countdown import equation_reward_func, format_reward_func
from education.common import group_advantages

EXAMPLES = [
    ("valid", "Add all numbers.</think>\n<answer>1 + 2 + 3</answer>", 6, [1,2,3]),
    ("correct equation, bad format", "<answer>1 + 2 + 3</answer>", 6, [1,2,3]),
    ("wrong target", "Add.</think>\n<answer>1 + 2 + 3</answer>", 7, [1,2,3]),
    ("reused operand", "Add.</think>\n<answer>1 + 2 + 2</answer>", 5, [1,2,3]),
    ("duplicate inputs are allowed", "Add.</think>\n<answer>2 + 2 + 3</answer>", 7, [2,2,3]),
    ("division by zero", "Divide.</think>\n<answer>1 / (2 - 2)</answer>", 1, [1,2,2]),
    ("forbidden exponent", "Power.</think>\n<answer>1 ** 2 + 3</answer>", 4, [1,2,3]),
    ("equals sign", "Add.</think>\n<answer>1 + 2 + 3 = 6</answer>", 6, [1,2,3]),
    ("extra opening think", "<think>Add.</think>\n<answer>1 + 2 + 3</answer>", 6, [1,2,3]),
    ("missing answer", "I cannot solve it.", 6, [1,2,3]),
]


def main():
    print("These are hand-built examples, not sampled model results.\n")
    for name, completion, target, nums in EXAMPLES:
        fmt = format_reward_func([completion])[0]
        eq = equation_reward_func([completion], [target], [nums])[0]
        print(f"{name}: format={fmt:g}, equation={eq:g}, total={fmt+eq:g}")
        print(" ", repr(completion))
    print("\nEach component is binary; equal weights sum to 0, 1 or 2.")
    print("Format checks tags/newline, not reasoning quality. Equation can pass without format.")
    print("Equation extracts the first answer, checks digit multiset and operator allowlist,")
    print("rejects ** and //, then evaluates with no builtins, tolerance 1e-5.")
    for totals in [[0.,1.,1.,2.], [1.,1.,1.,1.], [2.,2.,2.,2.]]:
        print("\nIllustrative group totals:", totals)
        print("Group advantage:", [round(a, 5) for a in group_advantages(totals)])
    print("Equal rewards give zero policy reward signal, whether all failed or all solved.")
    print("Edit EXAMPLES and rerun; do not prepend the prompt to a completion.")


if __name__ == "__main__":
    main()
