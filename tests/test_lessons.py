"""Offline contracts: actual task functions, numerical trace and CLI behavior."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LessonsTest(unittest.TestCase):
    def test_rewards_and_duplicate_number_accounting(self):
        # Catches accepting a reused/missing number or coupling the two rewards.
        from countdown import equation_reward_func, format_reward_func
        completions = [
            "Add them.</think>\n<answer>1 + 2 + 3</answer>",
            "<answer>1 + 2 + 3</answer>",
            "Add.</think>\n<answer>1 + 2 + 2</answer>",
            "Bad.</think>\n<answer>1 / (2 - 2)</answer>",
            "Power.</think>\n<answer>1 ** 2 + 3</answer>",
            "No answer.",
        ]
        self.assertEqual(format_reward_func(completions), [1., 0., 1., 1., 1., 0.])
        self.assertEqual(equation_reward_func(completions, [6]*6, [[1,2,3]]*6),
                         [1., 1., 0., 0., 0., 0.])
        self.assertEqual(equation_reward_func(["<answer>2 + 2 + 3</answer>"],
                                             [7], [[2,2,3]]), [1.])

    def test_prompt_keeps_assistant_turn_open(self):
        # The recorder isolates the external tokenizer; no invented chat tokens.
        from countdown import generate_r1_prompt
        class Recorder:
            def apply_chat_template(self, messages, *, tokenize, continue_final_message):
                self.messages = messages
                self.options = tokenize, continue_final_message
                return "rendered by tokenizer"
        recorder = Recorder()
        self.assertEqual(generate_r1_prompt(recorder, [1,2,3], 6),
                         {"prompt": "rendered by tokenizer", "target": 6})
        self.assertEqual(recorder.options, (False, True))
        self.assertEqual(recorder.messages[-1], {"role":"assistant", "content":
                         "Let me solve this step by step.\n<think>"})
        self.assertIn("[1, 2, 3]", recorder.messages[1]["content"])

    def test_group_advantages_and_zero_signal(self):
        # Hand calculated sample std of [0,1,1,2] is sqrt(2/3).
        from education.common import group_advantages
        a = group_advantages([0., 1., 1., 2.])
        self.assertAlmostEqual(a[0], -1.2245948898, places=7)
        self.assertEqual(a[1:3], [0.,0.])
        self.assertAlmostEqual(a[3], 1.2245948898, places=7)
        self.assertEqual(group_advantages([1.,1.,1.,1.]), [0.]*4)
        with self.assertRaises(ValueError):
            group_advantages([1.])

    def test_reward_lesson_runs_without_ml_dependencies(self):
        result = subprocess.run([sys.executable, "-m", "education.03_rewards"],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("total=2", result.stdout)
        self.assertIn("advantage", result.stdout)

    def test_comparison_counts_regressions_and_rejects_mismatched_rows(self):
        # Prevent a false gain caused by comparing different rows/decoding settings.
        from education.common import compare_evaluations
        base = {"protocol": {"seed":42}, "rows": [
            {"index":0, "nums":[1,2,3], "target":6, "equation":0},
            {"index":1, "nums":[2,3,4], "target":9, "equation":1}]}
        final = {"protocol": {"seed":42}, "rows": [
            {"index":0, "nums":[1,2,3], "target":6, "equation":1},
            {"index":1, "nums":[2,3,4], "target":9, "equation":0}]}
        self.assertEqual(compare_evaluations(base, final),
                         {"n":2, "baseline_solve_rate":0.5, "final_solve_rate":0.5,
                          "delta":0.0, "new_solves":1, "regressions":1})
        final["rows"][0]["target"] = 7
        with self.assertRaises(ValueError):
            compare_evaluations(base, final)

    def test_resume_requires_matching_horizon(self):
        # A one-step exercise must not silently shorten the intended LR schedule.
        from education.common import check_training_plan
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'plan.json'
            check_training_plan(path, {"max_steps":2000})
            check_training_plan(path, {"max_steps":2000})
            with self.assertRaises(ValueError):
                check_training_plan(path, {"max_steps":1})

    def test_changed_config_does_not_overwrite_original_record(self):
        from education.common import check_training_plan
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'resolved.json'
            check_training_plan(path, {"beta":0.001, "lora":{"r":16}})
            with self.assertRaises(ValueError):
                check_training_plan(path, {"beta":0.0, "lora":{"r":16}})
            self.assertEqual(__import__('json').loads(path.read_text()),
                             {"beta":0.001, "lora":{"r":16}})

    def test_snapshot_load_pins_resolved_hub_revision(self):
        # Transformers 5.19 discards _commit_hash; resolve revision before tokenizer loading.
        from unittest.mock import patch
        from types import ModuleType, SimpleNamespace
        from education.common import load_data
        datasets = ModuleType('datasets')
        datasets.load_from_disk = lambda _: None
        transformers = ModuleType('transformers')
        transformers.AutoTokenizer = object
        hub = ModuleType('huggingface_hub')
        hub.HfApi = lambda: SimpleNamespace(model_info=lambda _: SimpleNamespace(sha='a'*40))
        train = ModuleType('train')
        calls = []
        class Split(dict):
            def save_to_disk(self, _):
                pass
        class Rows(list):
            _fingerprint = 'fixture'
        split = Split(train=Rows([{'nums':[1,2,3], 'target':6}]),
                      test=Rows([{'nums':[2,3,4], 'target':9}]))
        tokenizer = SimpleNamespace(save_pretrained=lambda _: None, init_kwargs={})
        def prepare_data(**kwargs):
            calls.append(kwargs)
            return tokenizer, split
        train.prepare_data = prepare_data
        # Only external Hub/ML work is doubled; real manifest creation/identity logic runs.
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(sys.modules, {'datasets':datasets, 'transformers':transformers,
                                         'huggingface_hub':hub, 'train':train}):
                _, _, manifest = load_data(Path(folder))
        self.assertEqual(calls, [{'model_revision':'a'*40}])
        self.assertEqual(manifest['model_revision'], 'a'*40)


if __name__ == "__main__":
    unittest.main()
