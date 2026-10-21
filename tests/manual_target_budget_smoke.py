"""Fast persistent-target budget guards; real files and no image pipeline."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from asset_identity import sha256_file
import photo_request_budget as budget
from creative_task_smoke import snapshot


class ManualTargetGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='manual-target-guard-'); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root/'source.png'; Image.new('RGB', (20, 30), 'gray').save(self.source)
        self.base = self.root/'base.png'; Image.new('RGB', (20, 30), 'blue').save(self.base)
        self.binding = budget.manual_target_key(source_sha256=sha256_file(self.source), base_sha256=sha256_file(self.base),
            scope_id='3'*64, design_id='4'*64, user_feedback='A specific new lighting goal', provider_target={'fixture': 'transport'})

    def session(self, run_id, *, kind='tool', image=1, vision=0):
        import photo_edit_session as session
        run = self.root/'outputs/runs'/run_id
        directory = run/'editing-session'; directory.mkdir(parents=True)
        source = {'path': str(self.source), 'sha256': sha256_file(self.source)}
        base = {'path': str(self.base), 'sha256': sha256_file(self.base)}
        context = {'images': [{'role': 'source', **source}, {'role': 'best', **base}]}
        costs = {key: 0 for key in session.COUNTERS}; costs.update(image_submissions=image, vision_requests=vision)
        binding = {'source': source, 'initial_usage': dict.fromkeys(session.COUNTERS, 0)}
        state = {'schema': session.SCHEMA, 'session_id': session.identity(binding), 'binding': binding,
            'scope': {'scope_id': self.binding['scope_id']}, 'design': {'design_id': self.binding['design_id']},
            'user_intent': {'user_feedback': self.binding['user_feedback']}, 'candidates': [{'image': base}],
            'used': costs, 'status': 'needs_attention',
            'attempts': [{'attempt_id': 'paid-attempt', 'kind': kind, 'status': 'unknown', 'costs': costs, 'context': context}]}
        (directory/'session.json').write_text(json.dumps(state), encoding='utf8')
        return run, state, context

    def image_receipts(self, run, context, *, terminal=False):
        output = run/'editing-tools/paid-attempt'; (output/'provider').mkdir(parents=True)
        tool = {'schema': 'cosluma_photo_edit_tool.v1', 'state': 'needs_attention',
            'binding': {'attempt_id': 'paid-attempt', 'context_sha256': budget.identity(context)},
            'generator_input': {'path': str(self.base), 'sha256': sha256_file(self.base)},
            'compiled_prompt': {'prompt': 'An exact checked model request'}}
        (output/'tool-attempt.json').write_text(json.dumps(tool), encoding='utf8')
        receipt = {'schema': 'reference_transport_attempt.v1', 'state': 'response_received' if terminal else 'failed',
            'source': {'sha256': self.binding['source_sha256']}, 'request': {'requested_prompt': tool['compiled_prompt']['prompt']},
            'events': [], 'error_type': 'TimeoutError'}
        path = output/'provider/edited.transport-actual.json'; path.write_text(json.dumps(receipt), encoding='utf8')
        return path

    def test_readonly_plan_and_atomic_same_target_claim(self):
        before = snapshot(self.root)
        budget.check_manual_target(self.root, self.binding)
        self.assertEqual(snapshot(self.root), before)
        claimed = threading.Event(); release = threading.Event()
        def first():
            with budget.manual_target_guard(self.root, self.binding, run_id='first-child'):
                claimed.set(); self.assertTrue(release.wait(5))
        with ThreadPoolExecutor(max_workers=2) as pool:
            worker = pool.submit(first); self.assertTrue(claimed.wait(5))
            try:
                with self.assertRaises(budget.ManualTargetBlocked):
                    with budget.manual_target_guard(self.root, self.binding, run_id='different-child-id'):
                        self.fail('An active identical target obtained a second owner')
                blocked_snapshot = snapshot(self.root)
                with self.assertRaises(budget.ManualTargetBlocked):
                    budget.check_manual_target(self.root, self.binding)
                self.assertEqual(snapshot(self.root), blocked_snapshot)
            finally:
                release.set(); worker.result(5)
        budget.check_manual_target(self.root, self.binding)

    def test_unknown_image_and_vision_remain_blocked_without_pid_or_missing_evidence_unlock(self):
        for kind, image, vision in (('tool', 1, 0), ('decide', 0, 1)):
            binding = {**self.binding, 'user_feedback': 'Goal for '+kind}
            previous = self.binding; self.binding = binding
            try:
                with budget.manual_target_guard(self.root, binding, run_id=kind):
                    run, _state, context = self.session(kind, kind=kind, image=image, vision=vision)
                    if kind == 'tool':
                        self.image_receipts(run, context)
                    else:
                        directory = run/'editing-decisions/paid-attempt'; directory.mkdir(parents=True)
                        (directory/'decision.json').write_text(json.dumps({'schema': 'cosluma_photo_session_decision.v1',
                            'binding': {'context_sha256': budget.identity(context)}, 'model_requests_attempted': 1,
                            'error': {'type': 'TimeoutError', 'message': 'Unknown actual outcome'}}), encoding='utf8')
                with self.assertRaises(budget.ManualTargetBlocked):
                    budget.check_manual_target(self.root, binding)
                with self.assertRaises(budget.ManualTargetBlocked):
                    budget.reconcile_manual_target(self.root, binding)
                (run/'editing-session/session.json').unlink()
                with self.assertRaises(budget.ManualTargetBlocked):
                    budget.reconcile_manual_target(self.root, binding)
                budget.check_manual_target(self.root, {**binding, 'user_feedback': 'A different explicit goal'})
            finally:
                self.binding = previous

    def test_actual_terminal_image_receipt_and_cpu_only_failure_do_not_lock_future_tasks(self):
        for name, images in (('returned-image', 1), ('cpu-only-error', 0)):
            binding = {**self.binding, 'user_feedback': name}; previous = self.binding; self.binding = binding
            try:
                with budget.manual_target_guard(self.root, binding, run_id=name):
                    run, _state, context = self.session(name, image=images)
                    if images:self.image_receipts(run, context, terminal=True)
                budget.check_manual_target(self.root, binding)
                with budget.manual_target_guard(self.root, binding, run_id=name+'-next'):
                    pass
                budget.check_manual_target(self.root, binding)
            finally:
                self.binding = previous

    def test_provider_identity_ignores_snapshot_paths_credentials_and_profile_readiness(self):
        def config(**options):
            return {'kind': 'comfyui', 'registry_id': 'qwen_image_2_1', 'model': options['model'],
                    'endpoint': 'http://local.invalid:8188', 'configured': False}
        with patch('reference_provider.provider_config', side_effect=config), \
                patch('vision_transport._connection', return_value=({'kind': 'openai-compatible'},
                    'https://vision.invalid/v1', 'private-key-never-persist', None)):
            first = budget.manual_provider_target({'model': 'C:/snapshot-a/qwen-image-edit.json'}, image_generation=False)
            second = budget.manual_provider_target({'model': 'D:/snapshot-b/qwen-image-edit.json'}, image_generation=False)
        self.assertEqual(first, second)
        self.assertNotIn('snapshot-', json.dumps(first))
        self.assertNotIn('private-key', json.dumps(first))


if __name__ == '__main__':
    unittest.main()
