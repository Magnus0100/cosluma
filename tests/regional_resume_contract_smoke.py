"""Pure preflight proves a failed region was never submitted; no service calls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import creative_task as task
from creative_direction_smoke import enhanced
from creative_task_smoke import fixture, snapshot
from task_model_preflight_smoke import mocked_task_capability


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='regional-resume-contract-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        registry, sources, reference = fixture(self.root)
        self.common = {'source': sources[0], 'model_registry': str(registry),
            'provider': 'qwen_image_2_1_uncensored_fast', 'character_name': '祝小红',
            'character_notes': '蝴蝶仅为情感意象，不是施法', 'input_prompt': 'Keep original scene'}
        self.enterContext(patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability))
        self.enterContext(patch('reference_provider.provider_health', return_value={'ready': True}))
        self.enterContext(patch('urllib.request.urlopen', side_effect=AssertionError('Preflight attempted network')))
        original = task.plan_task(self.root, {**self.common, 'reference': reference,
            'creative_task': enhanced('existing_reference',regional=True)})
        self.parent = {'kind': 'single_input_run', 'run_id': 'failed-parent', 'status': 'blocked',
            'source': original['plan']['inputs']['source'],
            'reference': {'path': reference, 'sha256': task.sha(Path(reference))},
            'task_intent': original['plan']['intent'], 'task_plan': original['plan'],
            'generation_budget': {'origin_run_id': 'failed-parent'}, 'options': copy.deepcopy(original['request'])}
        self.manifest = self.root/'outputs/runs/failed-parent/manifest.json'
        write(self.manifest, self.parent)
        self.reservation = {'schema': 'regional_scene_refinement.v1', 'state': 'failed_no_submission',
            'model_submissions': 0, 'maximum_submissions': 1,
            'binding': {'source': self.parent['source']['sha256'], 'reference': self.parent['reference']['sha256'],
                'core_box': [1, 2, 12, 16], 'checked_plan_id': original['plan']['plan_id'],
                'direction_id': original['plan']['creative_direction']['direction_id']}}
        self.ledger = self.root/'outputs/regional-reservations/failed-parent/regional-refinement.json'
        write(self.ledger, self.reservation)
        intent = enhanced('existing_reference', regional=False)
        intent['regional_refinement'] = {'maximum_submissions': 1, 'resume_from_run_id': 'failed-parent',
            'regions': [{'id': 'bg-1', 'core_box': [1, 2, 12, 16]}]}
        self.body = {**self.common, 'reference': reference, 'creative_task': intent}

    def test_verified_resume_budget_proof_and_read_only_roundtrip(self):
        before = snapshot(self.root)
        checked = task.plan_task(self.root, self.body)
        self.assertEqual(before, snapshot(self.root))
        budget = checked['plan']['image_budget']
        self.assertEqual(budget, {'initial_reference': 0, 'reference_correction': 0,
            'independent_effect': 0, 'regional_refinement': 1})
        self.assertFalse(checked['request']['generate'])
        services = checked['plan']['services']
        self.assertTrue(services['image_generation'])
        self.assertTrue(services['local_image_inference'])
        self.assertEqual(services['image_generation_location'], 'local_machine')
        self.assertEqual(services['regional_image_submissions_upper_bound'], 1)
        self.assertFalse(services['text_vision'])
        self.assertFalse(services['visual_review'])
        self.assertEqual(services['sent_content'], [])
        self.assertTrue(services['offline'])
        self.assertNotIn('生图费用另计', services['price'])
        proof = checked['plan']['regional_resume_authorization']
        self.assertEqual(proof, checked['request']['regional_resume_authorization'])
        self.assertEqual(proof['origin_run_id'], 'failed-parent')
        self.assertEqual(proof['previous_manifest_sha256'], task.sha(self.manifest))
        self.assertEqual(proof['reservation_sha256'], task.sha(self.ledger))
        self.assertEqual(proof, task.revalidate_regional_resume(self.root, checked['plan']))
        repeated = task.plan_task(self.root, checked['request'])
        self.assertEqual(checked['plan']['plan_id'], repeated['plan']['plan_id'])
        execute = task.execution_request(self.root, {**checked['request'], 'plan_id': checked['plan']['plan_id']})
        task.validate_task_plan(execute['task_plan'], execute['task_intent'])
        restored = task.saved_task_intent_for_replanning(checked['plan'], checked['plan']['intent'],
            source_sha256=self.parent['source']['sha256'], reference_sha256=self.parent['reference']['sha256'])
        self.assertEqual(restored, checked['plan']['intent'])
        from backend_server import Backend
        backend=Backend(self.root,embedded=True)
        request={**self.common,'source':self.parent['source']['path'],
            'reference':self.parent['reference']['path'],'generate':False,
            'input_prompt':self.common['input_prompt']}
        with patch.object(backend,'one_click',side_effect=lambda body: {'submitted':body}):
            result=backend._continue_checked_task(self.parent,request,resume_unsubmitted_region=True)
        submitted=result['submitted']
        self.assertFalse(submitted['generate'])
        self.assertEqual(submitted['task_plan']['image_budget']['regional_refinement'],1)
        self.assertEqual(submitted['regional_resume_authorization'],proof)

    def test_unknown_submitted_completed_started_and_ids_never_renew(self):
        for change in ({'state': 'unknown'}, {'state': 'reserved'}, {'state': 'submitted'},
                       {'state': 'completed'}, {'model_submissions': 1}, {'model_submissions': False},
                       {'request_started': True}, {'prompt_id': 'real-submission'},
                       {'provider_ids': ['real-submission']}):
            with self.subTest(change=change):
                write(self.ledger, {**self.reservation, **change})
                with self.assertRaises(ValueError):
                    task.plan_task(self.root, self.body)
        write(self.ledger, self.reservation)
        checked = task.plan_task(self.root, self.body)
        write(self.ledger, {**self.reservation, 'state': 'reserved', 'request_started': False})
        # Pure checked-plan validation does not reread the mutable consumed journal.
        task.validate_task_plan(checked['plan'], checked['plan']['intent'])
        with self.assertRaises(ValueError):
            task.revalidate_regional_resume(self.root, checked['plan'])

    def test_source_reference_role_brief_region_and_parent_must_match(self):
        for mutate in (
            lambda row: row.update(character_name='Other role'),
            lambda row: row.update(character_notes='Changed role notes'),
            lambda row: row.update(input_prompt='Changed original input'),
            lambda row: row['creative_task'].update(brief='Different narrative'),
            lambda row: row['creative_task']['constraints'].update(no_effects=True),
            lambda row: row['creative_task']['regional_refinement']['regions'][0].update(core_box=[2, 2, 13, 16])):
            body = copy.deepcopy(self.body); mutate(body)
            with self.assertRaises(ValueError):
                task.plan_task(self.root, body)
        for role in ('source', 'reference'):
            previous = copy.deepcopy(self.parent); previous[role]['sha256'] = 'f'*64
            write(self.manifest, previous)
            with self.assertRaises(ValueError):
                task.plan_task(self.root, self.body)
        for status in ('running', 'queued', 'needs_visual_review', 'accepted'):
            write(self.manifest, {**self.parent, 'status': status})
            with self.assertRaises(ValueError):
                task.plan_task(self.root, self.body)
        write(self.manifest, self.parent)
        previous = copy.deepcopy(self.reservation); previous['binding']['checked_plan_id'] = 'e'*64
        write(self.ledger, previous)
        with self.assertRaises(ValueError):
            task.plan_task(self.root, self.body)

    def test_active_jobs_missing_or_changed_proof_refused(self):
        checked = task.plan_task(self.root, self.body)
        own = self.root/'outputs/jobs/current/job.json'
        write(own, {'status': 'running', 'parent_run_id': 'failed-parent', 'task_plan': checked['plan']})
        with self.assertRaises(ValueError):
            task.plan_task(self.root, self.body)
        self.assertEqual(checked['plan']['regional_resume_authorization'],
            task.revalidate_regional_resume(self.root, checked['plan']))
        write(own, {'status': 'running', 'run_id': 'failed-parent'})
        with self.assertRaises(ValueError):
            task.revalidate_regional_resume(self.root, checked['plan'])
        write(own, {'status': 'failed', 'run_id': 'failed-parent'})
        self.ledger.write_text(self.ledger.read_text('utf8')+'\n', encoding='utf8')
        with self.assertRaises(ValueError):
            task.revalidate_regional_resume(self.root, checked['plan'])
        bad = copy.deepcopy(checked['plan']); bad.pop('regional_resume_authorization')
        with self.assertRaises(ValueError):
            task.validate_task_plan(bad, checked['plan']['intent'])
        with self.assertRaises(ValueError):
            task.plan_task(self.root, {**self.body, 'final_repair_only': True})

    def test_default_existing_and_unsafe_resume_never_grant_budget(self):
        for resume in ('../failed-parent', '', 'parent/run', 1):
            intent = copy.deepcopy(self.body['creative_task'])
            intent['regional_refinement']['resume_from_run_id'] = resume
            with self.assertRaises(ValueError):
                task.normalize_task_intent(intent)
        intent = copy.deepcopy(self.body['creative_task']); intent['regional_refinement'].pop('resume_from_run_id')
        self.assertEqual(task.image_budget(intent)['regional_refinement'], 1)
        intent['regional_refinement'] = {'maximum_submissions': 0, 'regions': []}
        self.assertEqual(sum(task.image_budget(intent).values()), 0)
        zero_services = task._services({}, task.normalize_task_intent(intent))
        self.assertEqual(zero_services, {'text_vision': False, 'visual_review': False,
            'image_generation': False, 'offline': True, 'configured': True,
            'network_probed': False, 'sent_content': [], 'text_vision_requests_upper_bound': 0,
            'price': '不调用外部服务'})
        intent = copy.deepcopy(self.body['creative_task']); intent['mode'] = 'create_reference'
        with self.assertRaises(ValueError):
            task.normalize_task_intent(intent)

    def test_explicit_independent_existing_region_is_a_new_checked_task(self):
        intent = copy.deepcopy(self.body['creative_task'])
        intent['regional_refinement'].pop('resume_from_run_id')
        body = {**self.body, 'creative_task': intent, 'run_id': 'new-independent-task'}
        # Old consumed journals are neither reset nor used to grant this new task.
        write(self.ledger, {**self.reservation, 'state': 'completed', 'model_submissions': 1,
            'prompt_id': 'old-submitted-request'})
        before = snapshot(self.root)
        checked = task.plan_task(self.root, body)
        self.assertEqual(before, snapshot(self.root))
        self.assertNotIn('regional_resume_authorization', checked['plan'])
        self.assertNotIn('regional_resume_authorization', checked['request'])
        self.assertEqual(checked['plan']['image_budget'], {'initial_reference': 0,
            'reference_correction': 0, 'independent_effect': 0, 'regional_refinement': 1})
        self.assertEqual(checked['plan']['generation_submissions_upper_bound'], 1)
        self.assertTrue(checked['plan']['services']['local_image_inference'])
        self.assertFalse(checked['request']['generate'])
        self.assertEqual(checked['request']['creative_direction']['background_mode'], 'enhance_source')
        execute = task.execution_request(self.root, {**checked['request'], 'plan_id': checked['plan']['plan_id']})
        self.assertEqual(execute['cosplay_correction_budget'], 0)
        task.validate_task_plan(execute['task_plan'], execute['task_intent'])
        other = task.plan_task(self.root, {**body, 'run_id': 'another-independent-task'})
        self.assertNotEqual(checked['plan']['plan_id'], other['plan']['plan_id'])
        for extra in ({'final_repair_only': True}, {'generate': True},
                      {'provider': 'openai-compatible'}, {'regional_resume_authorization': {'fake': True}}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                task.plan_task(self.root, {**body, **extra})
        default = copy.deepcopy(intent); default.pop('regional_refinement')
        zero = task.plan_task(self.root, {**body, 'creative_task': default})
        self.assertEqual(sum(zero['plan']['image_budget'].values()), 0)
        changed = copy.deepcopy(checked['request']); changed['creative_task']['regional_refinement']['regions'][0]['core_box'] = [2, 3, 13, 17]
        with self.assertRaises(ValueError):
            task.execution_request(self.root, {**changed, 'plan_id': checked['plan']['plan_id']})


if __name__ == '__main__':
    unittest.main()
