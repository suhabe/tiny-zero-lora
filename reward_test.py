from collections.abc import Callable

import pytest

from reward import equation_reward_func, format_reward_func

correct_sample_1 = """We need to find an equation using the numbers 19, 36, 55, and 7
exactly once, with basic arithmetic operations, that equals 65. One possible
combination is 55 + 36 - 19 + 7... </think>
<answer> 55 + 36 - 7 - 19 </answer>"""

correct_sample_2 = """ ... </think>
<answer> 55 + 36 - 7 - 19 </answer>"""

wrong_format = """User: Using the numbers [19, 36, 55, 7], create an equation that equals 65."""

wrong_format_2 = """To find the equation that equals 79 using the numbers 95, 78, 6, 88, I'll start by adding 88 and 95:                      
95 + 88 = 183                                                                                                              
Now, let's subtract 104 from 183 to get 79:
183 - 104 = 79
<think> 183 - 104 = 79 </think><think> 183 - 104 = 79 </think><answer> 183 - 104 = 79 </answer>"""

wrong_result = """ ... </think>
<answer> 55 + 36 - 7 - 18 </answer>"""


def test_format_rewards() -> None:
    rewards = format_reward_func(completions=[correct_sample_1, correct_sample_2, wrong_format, wrong_format_2, wrong_result], target=[65] * 5, nums=[[19, 36, 55, 7]] * 5)
    assert rewards == [1.0, 1.0, 0.0, 0.0, 1.0]


def test_equation_rewards() -> None:
    rewards = equation_reward_func(completions=[correct_sample_1, correct_sample_2, wrong_format, wrong_format_2, wrong_result], target=[65] * 5, nums=[[19, 36, 55, 7]] * 5)
    assert rewards == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_invalid_equations_receive_zero_reward() -> None:
    completions = ["<answer>1 / 0</answer>", "<answer>2 + 2</answer>", "<answer>sum([1, 2])</answer>"]
    assert equation_reward_func(completions, target=[3, 3, 3], nums=[[1, 0], [1, 2], [1, 2]]) == [0.0, 0.0, 0.0]


def test_unsupported_operations_receive_zero_reward() -> None:
    completions = ["<answer>2**3</answer>", "<answer>5//2</answer>", "<answer>1.2*5</answer>"]
    assert equation_reward_func(completions, target=[8, 2, 6], nums=[[2, 3], [5, 2], [1, 2, 5]]) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("reward_func", [format_reward_func, equation_reward_func])
def test_reward_callbacks_accept_trl_metadata(reward_func: Callable[..., list[float | None]]) -> None:
    rewards = reward_func(prompts=["prompt"], completions=[correct_sample_2], completion_ids=[[1, 2]], target=[65], nums=[[19, 36, 55, 7]], trainer_state=None, log_extra=None, log_metric=None)
    assert rewards == [1.0]
