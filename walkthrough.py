"""Run and explain the three helpers in train.py (Python 3.10+).

    python walkthrough.py

The default uses only the standard library: no GPU, training, downloads, or
installed training dependencies. Its tokenizer is a clearly labeled readable
stand-in. To see a real model's chat template instead, install transformers and
run, for example:

    python walkthrough.py --model Qwen/Qwen2.5-3B-Instruct

That option may download tokenizer files, but does not load model weights.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
from typing import NamedTuple, cast


def load_helpers() -> dict:
    """Execute the actual helper definitions without importing the training stack.

    Importing train normally also imports torch, TRL, datasets, and PEFT. Select
    only these three definitions from the sibling file instead of duplicating
    their logic here. Postponed annotations avoid requiring transformers types.
    The helpers currently depend only on re and typing.cast; if they gain other
    dependencies, this loader must be updated too.
    """
    path = Path(__file__).resolve().with_name("train.py")
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"format_reward_func", "equation_reward_func", "generate_r1_prompt"}
    definitions = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions} != names:
        raise RuntimeError("Expected all three helper functions in train.py")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *definitions],
        type_ignores=[],
    )
    namespace = {"re": re, "cast": cast}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class PreviewTokenizer:
    """Readable chat-template stand-in, NOT an exact model tokenizer/template."""

    def apply_chat_template(self, messages, *, tokenize, continue_final_message):
        if tokenize is not False or continue_final_message is not True:
            raise AssertionError("Expected text output and an unfinished assistant turn")
        return "\n\n".join(
            f"[{message['role']}]\n{message['content']}" for message in messages
        )


def check(label, actual, expected):
    """Make the examples fail loudly if the repository's behavior changes."""
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def walkthrough_prompt(generate_r1_prompt, tokenizer):
    section("1. generate_r1_prompt: turn a puzzle into a chat prompt")
    print(
        "The function builds system, user, and assistant messages, then calls\n"
        "tokenizer.apply_chat_template(..., tokenize=False,\n"
        "                              continue_final_message=True).\n"
        "It returns a dict with the rendered prompt string and target.\n"
        "The assistant turn is left open so generation continues after <think>."
    )
    for numbers, target in [([1, 2, 3], 6), ([2, 3, 4], 14)]:
        print(f"\nInput: numbers={numbers}, target={target}")
        row = generate_r1_prompt(tokenizer, numbers, target)
        check("returned keys", set(row), {"prompt", "target"})
        check("target", row["target"], target)
        if not isinstance(row["prompt"], str) or "<think>" not in row["prompt"]:
            raise AssertionError("Expected a text prompt containing the assistant prefix")
        print(f"Returned target: {row['target']}")
        print(f"Returned prompt:\n{row['prompt']}")
    print(
        "\nIn train.py, dataset.map keeps the original nums column alongside these\n"
        "returned fields. The equation reward receives nums and target later."
    )


def walkthrough_format(format_reward_func):
    section("2. format_reward_func: check the completion's tag structure")
    print(
        "Input is a batch of completion strings, excluding the prompt.\n"
        "The checker prepends <think>, then checks for a thinking block,\n"
        "a newline, and an <answer>...</answer> block. It does not check math.\n"
        "repr() below makes newlines visible as \\n."
    )
    cases = [
        ("Expected continuation", "Add the numbers.</think>\n<answer>1 + 2 + 3</answer>", 1.0),
        ("Wrong math still has valid format", "Guess.</think>\n<answer>1 + 2 - 3</answer>", 1.0),
        ("Missing </think>", "Add the numbers.\n<answer>1 + 2 + 3</answer>", 0.0),
        ("Missing newline between blocks", "Add.</think><answer>1 + 2 + 3</answer>", 0.0),
        ("Repeating the prompt's opening tag", "<think>Add.</think>\n<answer>1 + 2 + 3</answer>", 0.0),
        ("Empty blocks also match the pattern", "</think>\n<answer></answer>", 1.0),
    ]
    completions = [completion for _, completion, _ in cases]
    # Trainer metadata is accepted through **kwargs and ignored by this helper.
    rewards = format_reward_func(completions, trainer_state=None)
    check("format batch", rewards, [expected for _, _, expected in cases])
    for (label, completion, _), reward in zip(cases, rewards):
        print(f"\n{label}\n  completion = {completion!r}\n  reward = {reward}")
    print(f"\nBatch result (one score per completion): {rewards}")


class EquationExample(NamedTuple):
    label: str
    expression: str
    numbers: list[int]
    target: int
    expected: float
    explanation: str


def walkthrough_equation(equation_reward_func):
    section("3. equation_reward_func: check numbers, operators, and the result")
    print(
        "Inputs are parallel batches: completions, target, and nums.\n"
        "For each row it extracts the first <answer>, checks the number\n"
        "multiset (including duplicates), allows basic arithmetic only, and\n"
        "evaluates the expression. A result within 1e-5 of target earns 1.0;\n"
        "a failed check or handled evaluation error earns 0.0."
    )
    cases = [
        EquationExample("Correct", "(1 + 2) * 3", [1, 2, 3], 9, 1.0,
                        "All numbers appear once; the expression evaluates to 9."),
        EquationExample("Wrong result", "1 + 2 - 3", [1, 2, 3], 6, 0.0,
                        "The numbers are right, but the result is 0, not 6."),
        EquationExample("Unused number", "1 + 2", [1, 2, 3], 3, 0.0,
                        "It reaches the target but omits the supplied 3."),
        EquationExample("Extra occurrence", "1 + 2 + 2 + 3", [1, 2, 3], 8, 0.0,
                        "The supplied 2 may only be used once."),
        EquationExample("Duplicate inputs are allowed", "2 + 2 + 3", [2, 2, 3], 7, 1.0,
                        "Two copies of 2 were supplied, and both are used."),
        EquationExample("Exponentiation rejected", "2 ** 3", [2, 3], 8, 0.0,
                        "** is explicitly rejected even though it is valid Python."),
        EquationExample("Floor division rejected", "4 // 2", [4, 2], 2, 0.0,
                        "// is explicitly rejected; ordinary / is allowed."),
        EquationExample("Equals sign rejected", "1 + 2 + 3 = 6", [1, 2, 3, 6], 6, 0.0,
                        "The answer must be an expression; '=' is not allowed."),
        EquationExample("Division by zero", "3 / (2 - 2)", [3, 2, 2], 0, 0.0,
                        "The arithmetic exception is caught and scores zero."),
        EquationExample("Invalid syntax", "1 + * 2 + 3", [1, 2, 3], 6, 0.0,
                        "Allowed characters alone do not guarantee a valid expression."),
        EquationExample("Floating point tolerance", "(1 / 49) * 49", [1, 49, 49], 1, 1.0,
                        "The result is 0.9999999999999999, within 1e-5 of 1."),
    ]
    completions = [f"<answer>{case.expression}</answer>" for case in cases]
    rewards = equation_reward_func(
        completions,
        target=[case.target for case in cases],
        nums=[case.numbers for case in cases],
        trainer_state=None,
    )
    check("equation batch", rewards, [case.expected for case in cases])
    for case, completion, reward in zip(cases, completions, rewards):
        print(f"\n{case.label}: nums={case.numbers}, target={case.target}")
        print(f"  completion = {completion!r}\n  reward = {reward}")
        print(f"  Why: {case.explanation}")
    missing = equation_reward_func(["1 + 2 + 3"], target=[6], nums=[[1, 2, 3]])
    check("missing answer tags", missing, [0.0])
    print(f"\nNo answer tags: completion='1 + 2 + 3' -> {missing}")
    print(
        "\nKeep the three batches aligned and equal in length: the helper uses\n"
        "zip(), so an unmatched trailing item would not be scored."
    )


def walkthrough_together(format_reward_func, equation_reward_func):
    section("4. Compare both rewards on the same completions")
    completions = [
        "Add.</think>\n<answer>1 + 2 + 3</answer>",
        "Guess.</think>\n<answer>1 + 2 - 3</answer>",
        "<answer>1 + 2 + 3</answer>",
        "1 + 2 + 3",
    ]
    formats = format_reward_func(completions)
    equations = equation_reward_func(completions, target=[6] * 4, nums=[[1, 2, 3]] * 4)
    check("combined format", formats, [1.0, 1.0, 0.0, 0.0])
    check("combined equation", equations, [1.0, 0.0, 1.0, 0.0])
    for completion, form, equation in zip(completions, formats, equations):
        print(f"\n{completion!r}\n  format={form}, equation={equation}")
    print(
        "\nThe two callbacks reward different properties. An answer-only response\n"
        "can earn the equation reward while failing the format reward. Neither\n"
        "callback checks whether the written reasoning is good."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", help="Optional Hugging Face tokenizer ID for a real chat template")
    args = parser.parse_args()
    helpers = load_helpers()
    if args.model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print(f"Using the real chat template from {args.model} (tokenizer only).")
    else:
        tokenizer = PreviewTokenizer()
        print("Using a readable tokenizer stand-in; [role] markers are illustrative.")
        print("Use --model Qwen/Qwen2.5-3B-Instruct for the actual model chat template.")
    print("Executing the three actual function definitions from sibling train.py.")
    walkthrough_prompt(helpers["generate_r1_prompt"], tokenizer)
    walkthrough_format(helpers["format_reward_func"])
    walkthrough_equation(helpers["equation_reward_func"])
    walkthrough_together(helpers["format_reward_func"], helpers["equation_reward_func"])
    print("\nAll example checks passed. No training or model generation was run.")


if __name__ == "__main__":
    main()
