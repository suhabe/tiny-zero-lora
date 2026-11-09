"""Modal command/lifecycle contracts without remote compute or ML downloads."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ModalLessonsTest(unittest.TestCase):
    def checkpoint(self, root, step, complete=True):
        directory = root / 'tutorial' / 'training' / f'checkpoint-{step}'
        directory.mkdir(parents=True)
        (directory / 'trainer_state.json').write_text(json.dumps({'global_step':step}))
        if complete:
            for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth']:
                (directory / name).touch()
        return directory

    def test_all_lessons_route_to_existing_modules(self):
        from education.modal_runner import lesson_command
        expected = {'preflight':'00_preflight', 'dataset':'01_dataset', 'prompts':'02_prompts',
                    'rewards':'03_rewards', 'baseline':'04_evaluate', 'configuration':'05_code_and_config',
                    'step':'06_train', 'metrics':'07_metrics', 'performance':'08_performance',
                    'training':'06_train', 'final':'04_evaluate'}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for lesson, module in expected.items():
                command = lesson_command(lesson, runs_root=root)
                self.assertEqual(command[:3], [sys.executable, '-m', 'education.'+module])
                self.assertTrue((ROOT / 'education' / (module+'.py')).exists())
            command = lesson_command('step', runs_root=root)
            self.assertIn('--one-step', command)
            self.assertEqual(command[command.index('--run-dir')+1], str(root/'tutorial'))

    def test_resume_uses_highest_complete_checkpoint_and_saved_horizon(self):
        from education.modal_runner import lesson_command
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.checkpoint(root, 2)
            latest = self.checkpoint(root, 10)
            self.checkpoint(root, 11, complete=False)
            (root/'tutorial'/'training_plan.json').write_text(json.dumps({'max_steps':50}))
            command = lesson_command('training', runs_root=root)
            self.assertEqual(command[command.index('--resume')+1], str(latest))
            self.assertEqual(command[command.index('--max-steps')+1], '50')
            command = lesson_command('step', resume='checkpoint-2', runs_root=root)
            self.assertTrue(command[command.index('--resume')+1].endswith('checkpoint-2'))

    def test_final_reuses_baseline_size_and_saved_adapter(self):
        from education.modal_runner import lesson_command
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'tutorial').mkdir()
            (root/'tutorial'/'baseline.json').write_text(json.dumps({'protocol':{'n':20}}))
            command = lesson_command('final', runs_root=root)
            self.assertEqual(command[command.index('--limit')+1], '20')
            self.assertEqual(command[command.index('--adapter')+1], str(root/'tutorial'/'adapter'))

    def test_rejects_paths_outside_run_and_incomplete_resume(self):
        from education.modal_runner import lesson_command
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.checkpoint(root, 3, complete=False)
            for options in [{'run_name':'../escape'}, {'run_name':'/tmp/escape'},
                            {'run_name':'..'}, {'resume':'../other/checkpoint-3'},
                            {'resume':'checkpoint-3'}, {'resume':'latest'}, {'max_steps':-1}]:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    lesson_command('training', runs_root=root, **options)
            with self.assertRaises(ValueError):
                lesson_command('baseline', limit=-1, runs_root=root)

    def test_actual_offline_lesson_and_volume_lifecycle(self):
        from education.modal_runner import run_lesson
        events = []
        class Volume:
            def reload(self): events.append('reload')
            def commit(self): events.append('commit')
        with tempfile.TemporaryDirectory() as folder:
            run_lesson('rewards', volume=Volume(), runs_root=Path(folder), source_dir=ROOT)
        self.assertEqual(events, ['reload', 'commit'])

    def test_child_failure_still_commits_and_propagates_error(self):
        from education.modal_runner import run_lesson
        events = []
        class Volume:
            def reload(self): events.append('reload')
            def commit(self): events.append('commit')
        with tempfile.TemporaryDirectory() as folder:
            def fail(command, **kwargs):
                # Represent a worker that has written an artifact before failing.
                artifact = Path(folder)/'tutorial'/'partial.json'
                artifact.parent.mkdir(parents=True)
                artifact.write_text('{}')
                raise subprocess.CalledProcessError(1, command)
            with patch('education.modal_runner.subprocess.run', side_effect=fail):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_lesson('dataset', volume=Volume(), runs_root=Path(folder), source_dir=ROOT)
            self.assertTrue((Path(folder)/'tutorial'/'partial.json').exists())
        self.assertEqual(events, ['reload', 'commit'])


if __name__ == '__main__':
    unittest.main()
