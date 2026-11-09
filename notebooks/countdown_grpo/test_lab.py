"""Offline contract checks: no model download or CUDA is needed."""
import tempfile
import unittest
from pathlib import Path


class LabContracts(unittest.TestCase):
    def test_source_functions_and_reward_edges(self):
        import lab_core as c
        f, e, p = c.upstream_functions()
        self.assertEqual(f(['done</think>\n<answer>44+19+35</answer>']), [1.0])
        self.assertEqual(e(['<answer>44+19+35</answer>'], [98], [[44,19,35]]), [1.0])
        for expr in ['44+19+19+35', '44+19', '44**19+35', '44//19+35', '44/(19-19)+35', '__import__("os")']:
            self.assertEqual(e([f'<answer>{expr}</answer>'], [98], [[44,19,35]]), [0.0])
        # Source behavior: format and correctness are independent; trailing newline is accepted by $.
        self.assertEqual(f(['x</think>\n<answer>44+19+35</answer>\n']), [1.0])
        self.assertEqual(f(['<answer>44+19+35</answer>']), [0.0])
        self.assertEqual(e(['<answer>44+19+35</answer>'], [98], [[44,19,35]]), [1.0])
        t = c.MessageRecorder()
        p(t, [44,19,35], 98)
        self.assertTrue(t.messages[-1]['content'].endswith('<think>'))
        self.assertEqual(t.options, {'tokenize':False, 'continue_final_message':True})

    def test_group_signal_and_token_mask(self):
        import lab_core as c
        self.assertEqual(c.group_advantages([1,1,1,1]), [0.0]*4)
        a = c.group_advantages([0,1,1,2])
        self.assertAlmostEqual(sum(a), 0)
        self.assertGreater(a[-1],0)
        self.assertLess(a[0],0)
        trace = c.numeric_step()
        self.assertLess(trace['loss_after'], trace['loss_before'])
        self.assertLess(trace['max_gradient_error'], 1e-6)

    def test_presets_and_source_fidelity(self):
        import lab_core as c
        repo = c.preset('repository')
        self.assertEqual(repo['grpo']['per_device_train_batch_size'],16)
        self.assertEqual(repo['grpo']['num_generations'],16)
        self.assertEqual(repo['grpo']['max_steps'],2000)
        for name in ['repository','classroom']:
            p = c.preset(name)
            g = p['grpo']
            self.assertEqual(g['per_device_train_batch_size']*g['gradient_accumulation_steps'] % g['num_generations'],0)
        self.assertFalse(c.preset('classroom')['grpo']['use_vllm'])

    def test_partial_jsonl_and_immutable_manifest(self):
        import lab_core as c
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            c.ensure_manifest(p, {'preset':'classroom'})
            c.ensure_manifest(p, {'preset':'classroom'})
            with self.assertRaises(ValueError):
                c.ensure_manifest(p, {'preset':'repository'})
            (p/'events.jsonl').write_text('{"step":1}\n{"step":')
            self.assertEqual(c.read_events(p),[{'step':1}])

if __name__ == '__main__':
    unittest.main()
