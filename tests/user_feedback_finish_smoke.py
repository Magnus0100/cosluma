"""A passed saved photo is a baseline, never acceptance of new user feedback.

The public planner/start, coordinator, session, editing tools, native save and
publication run normally. Only actual model transports are synthetic; these
tiny pictures establish execution contracts rather than artistic quality.
"""
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from asset_identity import sha256_file
from backend_server import Backend
from creative_task_smoke import fixture
import scoped_production_smoke as production_fixture
import workbench_store


class UserFeedbackFinishTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='user-feedback-finish-')
        self.addCleanup(temporary.cleanup)
        temporary_root = Path(temporary.name).resolve()
        self.root = temporary_root/'project'; self.root.mkdir()
        registry, sources, _ = fixture(self.root)
        self.fx = production_fixture.ScopedProductionTests(methodName='runTest')
        self.fx.root = self.root; self.fx.registry = registry
        self.fx.sources = [Path(sources[0])]
        image = Image.new('RGB', (80, 120), (85, 110, 95))
        ImageDraw.Draw(image).rectangle((18, 26, 58, 84), fill=(170, 135, 120))
        image.save(self.fx.sources[0])
        self.fx.image_requests = []; self.fx.vision_requests = []
        self.fx.review_failure = None
        stack = ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {
            'COSPLAY_MODEL_REGISTRY': str(registry),
            'COSPLAY_PROVIDER_SETTINGS': str(self.root/'provider-settings.json'),
            'COSPLAY_MODEL_SETTINGS': str(self.root/'model-selection.json'),
            'COSPLAY_CONFIG_ROOT': str(self.root/'config'),
            'COSPLAY_CACHE_ROOT': str(temporary_root/'cache'),
            'COSPLAY_ASSET_IDENTITY_CACHE': str(temporary_root/'identity'),
            'COSPLAY_SOURCE_METADATA_CACHE': str(temporary_root/'source-metadata'),
            'SCOPED_FIXTURE_KEY': 'synthetic-not-transmitted'}))
        stack.enter_context(patch('vision_transport.request_vision_json', side_effect=self.vision))
        stack.enter_context(patch('reference_provider.request_edit', side_effect=self.generate))
        self.fx.backend = Backend(self.root, embedded=True)
        self.phase = 'parent'; self.outcome = 'better'; self.stop = False; self.timeout = False
        self.fail_save = False
        self.cancel_after_image = False; self.cancel_job_id = None
        self.job_available = threading.Event(); self.addCleanup(self.job_available.set)
        self.feedback = 'Balance the visible arm and prop lighting with the face while preserving this complete scene.'
        _, self.parent = self.fx.start(self.fx.request(self.fx.sources[0], run_id='feedback-parent'))
        self.assertTrue(self.parent['photo_edit_session']['automated_target_met'])
        self.parent_path = self.root/'outputs/runs/feedback-parent/manifest.json'
        self.parent_bytes = self.parent_path.read_bytes()
        self.parent_review = copy.deepcopy(self.parent['photo_edit_session']['independent_evaluation'])
        self.parent_png = Path(self.parent['artifacts']['auto_draft'])
        self.parent_sha = sha256_file(self.parent_png)
        self.initial_images = len(self.fx.image_requests)
        self.initial_vision = len(self.fx.vision_requests)
        self.phase = 'feedback'

    def vision(self, source, output, prompt, **kwargs):
        source = Path(source).resolve()
        if 'Select the single most useful available tool' in prompt:
            self.fx.vision_requests.append({'source': str(source), 'prompt': prompt})
            if self.phase != 'parent':self.assertIn(self.feedback, prompt)
            result = ({'action': 'stop', 'reason': 'No safe useful change is currently available.'} if self.stop else
                {'action': 'tool', 'tool': 'joint_edit', 'operations': ['relight'],
                 'instruction': self.feedback, 'reason': 'The explicit new feedback requires coherent light on related objects.'})
            return result, {'source': {'path': str(source), 'sha256': sha256_file(source)},
                'network_used': True, 'generation_used': False, 'model': 'synthetic-feedback-decision'}
        payload, provenance = self.fx.vision_request(source, output, prompt, **kwargs)
        if source.name == 'review-board.png':
            passed = self.phase == 'parent' or self.outcome == 'better'
            payload.update(credibility={'status': 'pass' if passed else 'fail',
                'findings': ['Synthetic visible geometry and lighting are credible.' if passed else 'The synthetic new light is less coherent.']},
                creative_benefit={'status': 'pass', 'findings': ['Synthetic scene intervention is visible.']},
                comparison=(payload['comparison'] if self.phase=='parent' else self.outcome),
                primary_issue='none' if passed else 'Keep the complete previous version.')
            if self.phase != 'parent':self.assertIn(self.feedback, prompt)
            if self.phase != 'parent' and self.fail_save:
                # A real filesystem failure after actual model review, rather
                # than a mocked renderer/save helper reporting a failure.
                blocker = self.root/'outputs/runs/feedback-child/render'
                self.assertFalse(blocker.exists())
                blocker.write_text('Synthetic file blocking native output directory creation.', encoding='utf-8')
        return payload, provenance

    def generate(self, *args, **kwargs):
        if self.phase != 'parent':
            self.assertIn(self.feedback, args[4], 'The actual joint-edit provider prompt lost the user feedback')
        if self.phase != 'parent' and self.timeout:
            self.fx.image_requests.append({'image': str(args[2]), 'prompt': args[4]})
            raise TimeoutError('Synthetic feedback image outcome is unknown')
        result = self.fx.image_request(*args, **kwargs)
        if self.phase != 'parent' and self.cancel_after_image:
            if not self.job_available.wait(5):raise AssertionError('Checked job identity was not exposed')
            self.fx.backend.cancel_job(self.cancel_job_id)
        return result

    def request(self):
        body = self.fx.request(self.fx.sources[0], run_id='feedback-child')
        body.pop('editing_autofinish')
        body['creative_task']['mode'] = 'existing_reference'
        body.update(parent_run_id=self.parent['run_id'], reference=str(self.parent_png),
            editing_iteration={'maximum_improvements': 1, 'reason': 'User requested another checked revision.',
                               'user_feedback': self.feedback})
        return body

    def plan(self):
        counts = len(self.fx.image_requests), len(self.fx.vision_requests)
        checked = self.fx.backend.task_plan(self.request())
        self.assertTrue(checked['plan']['ready'], checked['plan'].get('blockers'))
        self.assertEqual(counts, (len(self.fx.image_requests), len(self.fx.vision_requests)))
        return checked

    def start(self, checked=None):
        checked = self.plan() if checked is None else checked
        result = self.fx.backend.start_task({**checked['request'], 'plan_id': checked['plan']['plan_id'], 'async': False})
        self.assertTrue(result.get('ok'), str(result.get('stderr') or result)[-2000:])
        return result['run']

    def assert_parent_preserved(self):
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)
        self.assertEqual(sha256_file(self.parent_png), self.parent_sha)

    def state(self, child):
        return json.loads(Path(child['photo_edit_session']['path']).read_text('utf-8'))

    def assert_baseline_not_feedback_acceptance(self, child):
        state = self.state(child)
        self.assertEqual(state['best_candidate_id'], state['candidates'][0]['candidate_id'])
        self.assertEqual(state['candidates'][0]['review'], self.parent_review)
        self.assertEqual(state['candidates'][0]['review_objective'], 'baseline')
        self.assertFalse(child['photo_edit_session']['automated_target_met'])
        self.assertFalse(child['photo_edit_session']['objective_reviewed'])
        self.assertFalse(child['photo_edit_session']['human_approved'])
        self.assertEqual(sha256_file(child['artifacts']['auto_draft']), self.parent_sha)
        self.assert_parent_preserved()
        return state

    def test_passed_parent_feedback_reaches_real_decision_tool_and_current_review(self):
        child = self.start(); state = self.state(child)
        self.assertEqual([row['kind'] for row in state['attempts']], ['adopted_evaluation', 'decide', 'tool', 'evaluate'])
        self.assertEqual(state['binding']['user_intent']['user_feedback'], self.feedback)
        self.assertEqual(state['candidates'][0]['review'], self.parent_review)
        self.assertEqual(state['candidates'][0]['review_objective'], 'baseline')
        self.assertEqual(state['candidates'][1]['review_objective'], 'current')
        self.assertTrue(child['photo_edit_session']['automated_target_met'])
        self.assertTrue(child['photo_edit_session']['objective_reviewed'])
        self.assertFalse(child['photo_edit_session']['human_approved'])
        self.assertEqual(len(self.fx.image_requests)-self.initial_images, 1)
        self.assertEqual(len(self.fx.vision_requests)-self.initial_vision, 2)
        self.assertEqual(child['result_publication']['state'], 'committed')
        best = next(row for row in state['candidates'] if row['candidate_id']==state['best_candidate_id'])
        self.assertEqual(sha256_file(child['artifacts']['auto_draft']), best['image']['sha256'])
        self.assertNotEqual(best['image']['sha256'], self.parent_sha)
        self.assert_parent_preserved()

    def test_model_stop_with_zero_new_images_keeps_baseline_without_feedback_pass(self):
        self.stop = True
        child = self.start(); self.assert_baseline_not_feedback_acceptance(child)
        self.assertEqual(len(self.fx.image_requests)-self.initial_images, 0)
        self.assertEqual(len(self.fx.vision_requests)-self.initial_vision, 1)
        self.assertFalse(child['editing_iteration_result']['improved'])

    def test_worse_repair_keeps_old_pass_as_baseline_not_new_objective_pass(self):
        self.outcome = 'worse'
        child = self.start(); state = self.assert_baseline_not_feedback_acceptance(child)
        self.assertEqual(len(state['candidates']), 2)
        self.assertEqual(state['candidates'][1]['review_objective'], 'current')
        self.assertEqual(len(self.fx.image_requests)-self.initial_images, 1)
        self.assertEqual(len(self.fx.vision_requests)-self.initial_vision, 2)

    def test_unknown_image_cannot_report_feedback_complete_or_retry(self):
        self.timeout = True
        checked = self.plan(); child = self.start(checked)
        state = self.assert_baseline_not_feedback_acceptance(child)
        self.assertEqual(state['status'], 'needs_attention')
        self.assertEqual(state['attempts'][-1]['status'], 'unknown')
        self.assertEqual(child['editing_iteration_result']['failure']['type'], 'TimeoutError')
        self.assertEqual(len(self.fx.image_requests)-self.initial_images, 1)
        self.assertEqual(len(self.fx.vision_requests)-self.initial_vision, 1)
        counts = len(self.fx.image_requests), len(self.fx.vision_requests)
        try:self.fx.backend.start_task({**checked['request'], 'plan_id': checked['plan']['plan_id'], 'async': False})
        except (ValueError, FileExistsError, RuntimeError):pass
        self.assertEqual(counts, (len(self.fx.image_requests), len(self.fx.vision_requests)))

    def test_feedback_mutation_after_check_rejected_before_model_calls(self):
        checked = self.plan(); counts = len(self.fx.image_requests), len(self.fx.vision_requests)
        changed = copy.deepcopy(checked['request'])
        changed['editing_iteration']['user_feedback'] = 'A different unapproved objective.'
        with self.assertRaises(ValueError):
            self.fx.backend.start_task({**changed, 'plan_id': checked['plan']['plan_id'], 'async': False})
        self.assertEqual(counts, (len(self.fx.image_requests), len(self.fx.vision_requests)))
        self.assert_parent_preserved()

    def test_old_reason_alone_does_not_turn_parent_pass_into_new_feedback_permission(self):
        body = self.request(); body['editing_iteration'].pop('user_feedback')
        counts = len(self.fx.image_requests), len(self.fx.vision_requests)
        try:
            checked = self.fx.backend.task_plan(body)
        except ValueError:
            checked = None
        if checked is not None:self.assertFalse(checked['plan']['ready'])
        self.assertEqual(counts, (len(self.fx.image_requests), len(self.fx.vision_requests)))
        self.assert_parent_preserved()

    def test_late_cancellation_retains_real_baseline_and_does_not_claim_feedback_pass(self):
        self.cancel_after_image = True
        project_path = self.root/'project.json'; project_before = project_path.read_bytes()
        recent_path = self.root/'outputs/settings/recent_projects.json'; recent_before = recent_path.read_bytes()
        selection_before = workbench_store.saved_result_state(self.root)
        self.assertEqual(selection_before['latest_saved_run_id'], self.parent['run_id'])
        checked = self.plan()
        response = self.fx.backend.start_task({**checked['request'], 'plan_id': checked['plan']['plan_id'], 'async': True})
        self.cancel_job_id = response['job_id']; self.job_available.set()
        deadline = time.monotonic()+10
        while time.monotonic()<deadline:
            job = self.fx.backend.jobs.get(self.cancel_job_id)
            if job['status'] in {'cancelled','succeeded','failed','blocked'}:break
            time.sleep(.01)
        self.assertEqual(job['status'], 'cancelled')
        worker = self.fx.backend._job_workers.get(self.cancel_job_id)
        if worker is not None:
            worker.join(5); self.assertFalse(worker.is_alive())
        directory = self.root/'outputs/runs/feedback-child'
        state = json.loads((directory/'editing-session/session.json').read_text('utf-8'))
        self.assertFalse(state['automated_target_met'])
        self.assertEqual(state['candidates'][0]['review_objective'], 'baseline')
        self.assertEqual(len(state['candidates']), 1)
        self.assertEqual(sha256_file(state['candidates'][0]['image']['path']), self.parent_sha)
        child = json.loads((directory/'manifest.json').read_text('utf-8'))
        self.assertFalse(child.get('photo_edit_session',{}).get('automated_target_met',False))
        self.assertNotEqual(child.get('result_publication',{}).get('state'), 'committed')
        if child.get('artifacts',{}).get('auto_draft'):
            self.assertEqual(sha256_file(child['artifacts']['auto_draft']), self.parent_sha)
        self.assertEqual(len(self.fx.image_requests)-self.initial_images, 1)
        self.assertEqual(len(self.fx.vision_requests)-self.initial_vision, 1)
        self.assertEqual(project_path.read_bytes(), project_before)
        self.assertEqual(recent_path.read_bytes(), recent_before)
        self.assertEqual(workbench_store.saved_result_state(self.root), selection_before)
        self.assertNotIn('feedback-child', [row['run_id'] for row in workbench_store.list_recent(self.root)])
        self.assertEqual(self.fx.backend.recovery()['source_current']['run_id'], self.parent['run_id'])
        self.assert_parent_preserved()

    def test_native_save_failure_cannot_publish_the_unapplied_reviewed_best(self):
        self.fail_save = True
        checked = self.plan()
        selection_before = workbench_store.saved_result_state(self.root)
        recent_path = self.root/'outputs/settings/recent_projects.json'; recent_before = recent_path.read_bytes()
        project_path = self.root/'project.json'; project_before = project_path.read_bytes()
        try:
            response = self.fx.backend.start_task({**checked['request'], 'plan_id': checked['plan']['plan_id'], 'async': False})
        except (OSError, RuntimeError):
            response = None
        if response is not None:self.assertFalse(response.get('ok'), 'A failed native save became a successful task')
        directory = self.root/'outputs/runs/feedback-child'
        state = json.loads((directory/'editing-session/session.json').read_text('utf-8'))
        self.assertTrue(state['automated_target_met'], 'The test did not reach a real pass before save failure')
        best = next(row for row in state['candidates'] if row['candidate_id']==state['best_candidate_id'])
        self.assertEqual(best['review_objective'], 'current')
        child = json.loads((directory/'manifest.json').read_text('utf-8'))
        self.assertNotEqual(child.get('result_publication',{}).get('state'), 'committed')
        self.assertFalse(child.get('photo_edit_session',{}).get('automated_target_met',False))
        self.assertEqual(workbench_store.saved_result_state(self.root), selection_before)
        self.assertEqual(recent_path.read_bytes(), recent_before)
        self.assertEqual(project_path.read_bytes(), project_before)
        self.assertEqual(self.fx.backend.recovery()['source_current']['run_id'], self.parent['run_id'])
        self.assertNotIn('feedback-child', [row['run_id'] for row in workbench_store.list_recent(self.root)])
        self.assertEqual(len(self.fx.image_requests)-self.initial_images, 1)
        self.assertEqual(len(self.fx.vision_requests)-self.initial_vision, 2)
        self.assert_parent_preserved()


if __name__ == '__main__':
    unittest.main()
