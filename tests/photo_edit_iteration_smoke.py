"""Real checked refinement, pixels and failure retention; only model boundaries fake.

Parent production, task planning/start, coordinator, editing tools, native IO and
independent review execute normally. Synthetic findings are contract fixtures,
not artistic acceptance of these deliberately tiny artificial photographs.
"""
import copy
from io import BytesIO
import json
from pathlib import Path
import threading
import time
import unittest

import numpy as np
from PIL import Image

from asset_identity import sha256_file
from creative_task_smoke import snapshot
import scoped_production_smoke as production_fixture


class EditingIterationTests(unittest.TestCase):
    def setUp(self):
        self.fx = production_fixture.ScopedProductionTests(methodName='runTest')
        self.fx.setUp(); self.addCleanup(self.fx.doCleanups)
        self.root = self.fx.root
        self.phase = 'parent'; self.repair_outcome = 'better'; self.image_failure = None
        self.selected_tool = 'edit_region'; self.generated_scale = 1; self.wrong_aspect = False
        self.upload_sizes = []; self.generated_sizes = []
        self.cancel_job_id = None; self.cancel_after_image = False
        self.job_available = threading.Event()
        self.addCleanup(self.job_available.set)
        self.fx.vision.side_effect = self.vision
        self.fx.generation.side_effect = self.generate
        _checked, self.parent = self.fx.start(self.fx.request(self.fx.sources[0], run_id='iteration-parent'))
        self.parent_path = self.root/'outputs/runs'/self.parent['run_id']/'manifest.json'
        self.assertEqual(self.parent['result_publication']['state'], 'committed')
        self.assertEqual(self.parent['photo_edit_session']['independent_evaluation']['credibility']['status'], 'fail')
        self.parent_bytes = self.parent_path.read_bytes()
        self.parent_draft = Path(self.parent['artifacts']['auto_draft'])
        self.parent_pixels = np.asarray(Image.open(self.parent_draft)).copy()
        self.phase = 'iteration'
        self.initial_images = len(self.fx.image_requests); self.initial_vision = len(self.fx.vision_requests)

    def vision(self, source, output, prompt, **kwargs):
        source = Path(source).resolve()
        if 'Select the single most useful available tool' in prompt:
            self.fx.vision_requests.append({'source': str(source), 'sha256': sha256_file(source), 'prompt': prompt})
            mapping=json.loads((source.parent/'review-board.json').read_text(encoding='utf8'))
            with Image.open(source) as board:
                self.assertEqual(board.size,(len(mapping['panels'])*524+12,828))
            result = {'action': 'tool', 'tool': self.selected_tool, 'operations': ['detail_repair'],
                'instruction': 'Repair the observed local material transition while preserving the current scene light.',
                'reason': 'The independently observed material seam is the main remaining defect.'}
            if self.selected_tool == 'edit_region':
                result['region'] = [.45, .10, .85, .50]
            return result, {'source': {'path': str(source), 'sha256': sha256_file(source)},
                'network_used': True, 'generation_used': False, 'model': 'actual-fixture-decision-model'}
        payload, provenance = self.fx.vision_request(source, output, prompt, **kwargs)
        if source.name == 'review-board.png':
            if self.phase == 'parent':
                mapping=json.loads((source.parent/'review-board.json').read_text(encoding='utf8'))
                payload['comparison']='better' if len(mapping['images'])==3 else 'first_candidate'
                payload['credibility'] = {'status': 'fail', 'findings': ['A synthetic local material seam is visible in the set.']}
                payload['primary_issue'] = 'Repair the visible local material transition while preserving the design.'
            else:
                payload['comparison'] = 'worse' if self.repair_outcome == 'worse' else 'better'
                finding = ('Synthetic repaired material blends plausibly.' if self.repair_outcome == 'better' else
                           'One synthetic transition improved, but a second visible material issue remains.'
                           if self.repair_outcome == 'partial' else
                           'Synthetic attempted repair is less credible than the saved best.')
                payload['credibility'] = {'status': 'pass' if self.repair_outcome == 'better' else 'fail', 'findings': [finding]}
                payload['creative_benefit'] = {'status': 'pass', 'findings': ['Synthetic original design remains visible.']}
                payload['primary_issue'] = ('none' if self.repair_outcome == 'better' else
                    'Repair the second visible synthetic material transition.' if self.repair_outcome == 'partial' else
                    'Keep the previous complete version.')
        return payload, provenance

    def generate(self, *args, **kwargs):
        if self.phase == 'iteration' and self.image_failure is not None:
            self.fx.image_requests.append({'image': str(args[2]), 'model': args[3], 'prompt': args[4]})
            raise self.image_failure
        result = self.fx.image_request(*args, **kwargs)
        if self.phase == 'iteration':
            with Image.open(args[2]) as uploaded:
                self.upload_sizes.append(uploaded.size)
            payload, data, mode = result
            with Image.open(BytesIO(data)) as generated:
                edited = generated.convert('RGB')
            if self.generated_scale != 1:
                edited = edited.resize((edited.width*self.generated_scale, edited.height*self.generated_scale))
            if self.wrong_aspect:
                edited = edited.resize((64, 128))
            self.generated_sizes.append(edited.size)
            stream = BytesIO(); edited.save(stream, format='PNG'); result = payload, stream.getvalue(), mode
        if self.phase == 'iteration' and self.cancel_after_image:
            if not self.job_available.wait(5):
                raise AssertionError('Asynchronous checked task did not expose its job identity')
            self.fx.backend.cancel_job(self.cancel_job_id)
        return result

    def request(self, *, run_id='iteration-child', parent=None):
        parent = self.parent if parent is None else parent
        body = self.fx.request(self.fx.sources[0], run_id=run_id)
        body.pop('editing_autofinish')  # Manual improvement owns a separate checked task.
        body['creative_task']['mode'] = 'existing_reference'
        body.update(parent_run_id=parent['run_id'], reference=str(parent['artifacts']['auto_draft']),
            editing_iteration={'maximum_improvements': 1, 'reason': '根据审片修复主要缺陷'})
        return body

    def plan(self, request=None):
        before = snapshot(self.root)
        images_before = len(self.fx.image_requests); vision_before = len(self.fx.vision_requests)
        result = self.fx.backend.task_plan(self.request() if request is None else request)
        self.assertTrue(result['plan']['ready'], result['plan'].get('blocking_reasons'))
        self.assertEqual(snapshot(self.root), before, 'Refinement planning wrote production files')
        self.assertEqual(len(self.fx.image_requests), images_before)
        self.assertEqual(len(self.fx.vision_requests), vision_before)
        return result

    def start(self, checked):
        return self.fx.backend.start_task({**checked['request'], 'plan_id': checked['plan']['plan_id'], 'async': False})

    def assert_started(self, response):
        self.assertTrue(response.get('ok'), {'exit_code': response.get('exit_code'),
            'stderr': str(response.get('stderr') or '')[-2000:],
            'run_status': (response.get('run') or {}).get('status')})

    def rejected_plan(self, request):
        try:
            result = self.fx.backend.task_plan(request)
        except (ValueError, FileNotFoundError):
            return
        self.assertFalse(result['plan']['ready'], 'Invalid parent evidence was authorized for model execution')

    def test_authorize_original_draft_scope_review_and_new_budget_without_history_charges(self):
        checked = self.plan(); plan = checked['plan']
        authorization = plan['editing_refinement_authorization']
        self.assertEqual(authorization['parent_manifest_sha256'], sha256_file(self.parent_path))
        self.assertEqual(authorization['base_image_sha256'], sha256_file(self.parent_draft))
        self.assertEqual(authorization['source_sha256'], sha256_file(self.fx.sources[0]))
        self.assertEqual(authorization['scope_id'], self.parent['task_plan']['editing_scope']['scope_id'])
        self.assertEqual(authorization['design_id'], self.parent['generation_design']['observed_design']['design_id'])
        self.assertEqual(plan['image_budget'], {'initial_reference': 0, 'reference_correction': 0,
                                              'independent_effect': 0, 'editing_refinement': 1})
        self.assertEqual(plan['services']['text_vision_requests_upper_bound'], 2)
        self.assertTrue(plan['services']['image_generation'])
        self.assertFalse(checked['request']['generate'])
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)

    def test_authorize_rejects_pending_parent_wrong_inputs_scope_and_observed_facts(self):
        variations = [
            ('pending publication', lambda value: value['result_publication'].update(state='pending')),
            ('different source hash', lambda value: value['source'].update(sha256='0'*64)),
            ('different reviewed pixels', lambda value: value['photo_edit_session']['independent_evaluation']['observed_images'][1].update(sha256='0'*64)),
            ('changed design content', lambda value: value['generation_design']['observed_design']['changes'][0].update(instruction='Unbound design edit')),
            ('invalid review axis', lambda value: value['photo_edit_session']['independent_evaluation']['credibility'].update(status='unverified_but_approved')),
        ]
        for name, mutate in variations:
            with self.subTest(name=name):
                changed = copy.deepcopy(self.parent); mutate(changed)
                self.parent_path.write_text(json.dumps(changed, ensure_ascii=False), encoding='utf-8')
                try:
                    self.rejected_plan(self.request())
                finally:
                    self.parent_path.write_bytes(self.parent_bytes)
        wrong = self.request(); wrong['reference'] = self.parent['reference']['path']; self.rejected_plan(wrong)
        wrong = self.request(); wrong['editing_request']['style'] = 'dramatic'; self.rejected_plan(wrong)
        external = self.root/'outside-runs'; external.mkdir(); (external/'manifest.json').write_bytes(self.parent_bytes)
        wrong = self.request(); wrong['parent_run_id'] = '../../outside-runs'; self.rejected_plan(wrong)
        self.assertEqual(len(self.fx.image_requests), self.initial_images)
        self.assertEqual(len(self.fx.vision_requests), self.initial_vision)

    def test_changed_parent_after_plan_is_rejected_before_any_new_model_call(self):
        checked = self.plan()
        changed = copy.deepcopy(self.parent)
        changed['photo_edit_session']['independent_evaluation']['primary_issue'] = 'Changed after checked planning'
        self.parent_path.write_text(json.dumps(changed, ensure_ascii=False), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.start(checked)
        self.assertEqual(len(self.fx.image_requests), self.initial_images)
        self.assertEqual(len(self.fx.vision_requests), self.initial_vision)

    def test_execute_selected_region_saves_complete_child_and_counts_only_new_calls(self):
        checked = self.plan(); response = self.start(checked)
        self.assert_started(response)
        child = response['run']
        self.assertEqual(child['parent_run_id'], self.parent['run_id'])
        self.assertEqual(child['result_publication']['state'], 'committed')
        self.assertEqual(child['source']['sha256'], self.parent['source']['sha256'])
        self.assertEqual(len(self.fx.image_requests)-self.initial_images, 1)
        self.assertEqual(len(self.fx.vision_requests)-self.initial_vision, 2)
        summary = child['photo_edit_session']; state = json.loads(Path(summary['path']).read_text(encoding='utf-8'))
        self.assertEqual(state['binding']['initial_usage'], dict.fromkeys(state['used'], 0))
        self.assertEqual(state['used']['image_submissions'], 1)
        self.assertEqual(state['used']['vision_requests'], 2)
        self.assertEqual(state['used']['local_repairs'], 1)
        choices = [row['response'] for row in state['attempts'] if row['kind'] == 'decide']
        self.assertEqual(choices[0]['tool'], 'edit_region')
        self.assertEqual(choices[0]['region'], [.45, .10, .85, .50])
        selected = next(row for row in state['candidates'] if row['candidate_id'] == state['best_candidate_id'])
        self.assertNotEqual(selected['candidate_id'], state['candidates'][0]['candidate_id'])
        self.assertFalse(summary['human_approved'])
        self.assertIsNone(child['review']['reviewer'])
        self.assertFalse(child['render']['render_precision']['generated_resolution']['native_generated_detail'])
        result = np.asarray(Image.open(child['artifacts']['auto_draft']))
        self.assertEqual(result.shape, self.parent_pixels.shape)
        self.assertFalse(np.array_equal(result, self.parent_pixels))
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)
        self.assertEqual(sha256_file(self.parent_draft), checked['plan']['inputs']['reference']['sha256'])

    def test_worse_last_candidate_preserves_best_complete_parent_pixels(self):
        self.repair_outcome = 'worse'
        response = self.start(self.plan())
        self.assert_started(response)
        child = response['run']; state = json.loads(Path(child['photo_edit_session']['path']).read_text(encoding='utf-8'))
        self.assertEqual(state['best_candidate_id'], state['candidates'][0]['candidate_id'])
        self.assertEqual(state['stop_reason'], 'no_verified_net_improvement')
        self.assertTrue(np.array_equal(np.asarray(Image.open(child['artifacts']['auto_draft'])), self.parent_pixels))
        self.assertFalse(child['photo_edit_session']['human_approved'])
        self.assertEqual(len(self.fx.image_requests)-self.initial_images, 1)
        self.assertEqual(len(self.fx.vision_requests)-self.initial_vision, 2)
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)

    def _assert_owned_child_survives_parent_assets_removal(self,outcome):
        from export_artifact import export_source_run
        from photo_appearance import validate_asset
        import shutil
        self.repair_outcome=outcome
        response=self.start(self.plan());self.assert_started(response);child=response['run']
        parent_directory=self.parent_path.parent.resolve()
        child_directory=(self.root/'outputs/runs'/child['run_id']).resolve()
        state=json.loads(Path(child['photo_edit_session']['path']).read_text(encoding='utf8'))
        self.assertTrue(Path(state['source_appearance']['path']).resolve().is_relative_to(child_directory))
        for candidate in state['candidates']:
            self.assertTrue(Path(candidate['appearance_asset']['path']).resolve().is_relative_to(child_directory))
        parent_assets={Path(asset['path']).resolve() for asset in
            (self.parent['initial_source_appearance'],self.parent['options']['photo_appearance'])}
        for directory in parent_assets:
            self.assertTrue(directory.is_relative_to(parent_directory))
            self.assertFalse(directory.is_relative_to(child_directory))
            shutil.rmtree(directory)
        binding={'source_sha256':child['source']['sha256'],
            'scope':child['task_plan']['editing_scope'],
            'design_id':child['generation_design']['observed_design']['design_id']}
        for asset in (child['initial_source_appearance'],child['options']['photo_appearance']):
            validate_asset(asset,**binding)
        exported=export_source_run(self.root,child['run_id'],directory=str(self.root/'owned-export'),format='PNG')
        self.assertEqual(sha256_file(exported['path']),sha256_file(child['artifacts']['auto_draft']))
        body=self.request(run_id='after-parent-assets-removed',parent=child)
        body['editing_iteration']['user_feedback']='Improve the visible local transition while keeping the current scene.'
        self.plan(body)
        self.assertEqual(self.parent_path.read_bytes(),self.parent_bytes)

    def test_improved_child_assets_survive_parent_asset_removal_and_allow_new_feedback(self):
        self._assert_owned_child_survives_parent_assets_removal('better')

    def test_retained_baseline_child_assets_survive_parent_asset_removal_and_allow_new_feedback(self):
        self._assert_owned_child_survives_parent_assets_removal('worse')

    def test_actual_joint_edit_uses_current_complete_best_without_a_region(self):
        self.selected_tool = 'joint_edit'
        response = self.start(self.plan()); self.assert_started(response)
        child = response['run']; state = json.loads(Path(child['photo_edit_session']['path']).read_text(encoding='utf-8'))
        choice = next(row['response'] for row in state['attempts'] if row['kind'] == 'decide')
        self.assertEqual(choice['tool'], 'joint_edit'); self.assertNotIn('region', choice)
        self.assertEqual(self.upload_sizes, [(self.parent_pixels.shape[1], self.parent_pixels.shape[0])])
        selected = next(row for row in state['candidates'] if row['candidate_id'] == state['best_candidate_id'])
        composition = json.loads(Path(selected['provenance']['composition_receipt']['path']).read_text(encoding='utf-8'))
        self.assertIsNone(composition['core_box'])
        self.assertFalse(composition['source_subject_RGB_restored'])
        self.assertEqual(state['used']['local_repairs'], 0)
        self.assertEqual(state['used']['image_submissions'], 1)
        self.assertEqual(state['used']['vision_requests'], 2)
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)

    def test_high_resolution_generated_crop_keeps_framing_and_resolution_lineage_honest(self):
        self.generated_scale = 8
        response = self.start(self.plan()); self.assert_started(response)
        child = response['run']; state = json.loads(Path(child['photo_edit_session']['path']).read_text(encoding='utf-8'))
        self.assertGreater(self.generated_sizes[0][0], self.upload_sizes[0][0])
        self.assertGreater(self.generated_sizes[0][1], self.upload_sizes[0][1])
        selected = next(row for row in state['candidates'] if row['candidate_id'] == state['best_candidate_id'])
        composition = json.loads(Path(selected['provenance']['composition_receipt']['path']).read_text(encoding='utf-8'))
        self.assertEqual(composition['generated_context_size'], list(self.generated_sizes[0]))
        from photo_appearance import validate_asset
        appearance=validate_asset(selected['appearance_asset'],source_sha256=child['source']['sha256'],
            scope=child['task_plan']['editing_scope'],design_id=child['generation_design']['observed_design']['design_id'])
        self.assertFalse(appearance['native_generated_detail'])
        self.assertEqual(appearance['source_coverage_semantics'],'one_minus_cumulative_generated_coverage_not_unprocessed_light')
        self.assertTrue(composition['outside_core_float_identical_to_current_best'])
        actual = np.asarray(Image.open(child['artifacts']['auto_draft']))
        x0, y0, x1, y1 = composition['core_box']
        outside = np.ones(actual.shape[:2], dtype=bool); outside[y0:y1, x0:x1] = False
        self.assertTrue(np.array_equal(actual[outside], self.parent_pixels[outside]))
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)

    def test_wrong_generated_crop_aspect_cannot_be_stretched_into_a_successful_edit(self):
        self.wrong_aspect = True
        response = self.start(self.plan()); self.assert_started(response)
        child = response['run']; state = json.loads(Path(child['photo_edit_session']['path']).read_text(encoding='utf-8'))
        self.assertEqual(state['best_candidate_id'], state['candidates'][0]['candidate_id'])
        self.assertEqual(len(state['candidates']), 1)
        self.assertEqual(state['used']['image_submissions'], 1)
        self.assertEqual(state['used']['vision_requests'], 1)
        self.assertTrue(np.array_equal(np.asarray(Image.open(child['artifacts']['auto_draft'])), self.parent_pixels))
        self.assertIn('framing', child['editing_iteration_result']['failure']['message'])
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)

    def test_partially_improved_three_image_review_can_authorize_a_further_checked_iteration(self):
        self.repair_outcome = 'partial'
        first_response = self.start(self.plan()); self.assert_started(first_response)
        first = first_response['run']; review = first['photo_edit_session']['independent_evaluation']
        self.assertEqual(len(review['observed_images']), 3)
        self.assertEqual(review['credibility']['status'], 'fail')
        self.assertEqual(review['comparison'], 'better')
        self.assertFalse(first['photo_edit_session']['automated_target_met'])
        first_path = self.root/'outputs/runs'/first['run_id']/'manifest.json'; first_bytes = first_path.read_bytes()
        checked = self.plan(self.request(run_id='iteration-grandchild', parent=first))
        self.assertEqual(checked['plan']['editing_refinement_authorization']['parent_run_id'], first['run_id'])
        images_before = len(self.fx.image_requests); vision_before = len(self.fx.vision_requests)
        self.repair_outcome = 'better'
        second_response = self.start(checked); self.assert_started(second_response)
        second = second_response['run']
        self.assertEqual(second['parent_run_id'], first['run_id'])
        self.assertEqual(len(self.fx.image_requests)-images_before, 1)
        self.assertEqual(len(self.fx.vision_requests)-vision_before, 2)
        self.assertTrue(second['photo_edit_session']['automated_target_met'])
        self.assertFalse(second['photo_edit_session']['human_approved'])
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)

    def test_unknown_image_outcome_retains_parent_and_same_operation_does_not_resubmit(self):
        self.image_failure = TimeoutError('Synthetic repair provider outcome unknown')
        checked = self.plan(); response = self.start(checked)
        self.assertEqual(len(self.fx.image_requests)-self.initial_images, 1)
        self.assertEqual(len(self.fx.vision_requests)-self.initial_vision, 1)
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)
        self.assertTrue(self.parent_draft.is_file())
        self.assertTrue(np.array_equal(np.asarray(Image.open(self.parent_draft)), self.parent_pixels))
        self.assertFalse((response.get('run') or {}).get('photo_edit_session', {}).get('human_approved', False))
        try:
            self.start(checked)
        except (ValueError, FileExistsError, RuntimeError):
            pass
        self.assertEqual(len(self.fx.image_requests)-self.initial_images, 1, 'Unknown request was implicitly resubmitted')

    def passed_parent(self):
        response=self.start(self.plan());self.assert_started(response)
        parent=response['run']
        self.assertTrue(parent['photo_edit_session']['automated_target_met'])
        return parent

    def test_explicit_user_feedback_on_passed_parent_reaches_checked_decision_review_and_save(self):
        parent=self.passed_parent();path=self.root/'outputs/runs'/parent['run_id']/'manifest.json'
        original=path.read_bytes();baseline_review=copy.deepcopy(parent['photo_edit_session']['independent_evaluation'])
        body=self.request(run_id='feedback-child',parent=parent)
        self.rejected_plan(body)  # A program-generated reason is not human feedback.
        body['editing_iteration']['user_feedback']='  '
        self.rejected_plan(body)
        feedback='Keep the original expression and soften the excessive green light on the raised arm.'
        body['editing_iteration']['user_feedback']='  '+feedback+'  '
        checked=self.plan(body)
        self.assertEqual(checked['plan']['editing_iteration']['user_feedback'],feedback)
        self.assertEqual(checked['plan']['image_budget']['editing_refinement'],1)
        self.assertEqual(checked['plan']['services']['text_vision_requests_upper_bound'],2)
        before=(len(self.fx.image_requests),len(self.fx.vision_requests))
        response=self.start(checked);self.assert_started(response);child=response['run']
        self.assertEqual(len(self.fx.image_requests)-before[0],1)
        self.assertEqual(len(self.fx.vision_requests)-before[1],2)
        state=json.loads(Path(child['photo_edit_session']['path']).read_text(encoding='utf-8'))
        self.assertEqual(state['binding']['user_intent']['user_feedback'],feedback)
        baseline=state['candidates'][0]
        self.assertEqual(baseline['review'],baseline_review)
        self.assertEqual(baseline['review_objective'],'baseline')
        adopted=next(row for row in state['attempts'] if row['kind']=='adopted_evaluation')
        parent_state=json.loads(Path(parent['photo_edit_session']['path']).read_text(encoding='utf-8'))
        actual=next(row for row in parent_state['attempts'] if row.get('response')==baseline_review)
        self.assertEqual(adopted['context'],actual['context'])
        for attempt in state['attempts']:
            if attempt['kind'] in {'decide','evaluate'}:
                self.assertEqual(attempt['context']['user_intent']['user_feedback'],feedback)
        self.assertTrue(all(feedback in row['prompt'] for row in self.fx.vision_requests[before[1]:]))
        self.assertTrue(child['photo_edit_session']['objective_reviewed'])
        self.assertEqual(child['photo_edit_session']['review_objective'],'current')
        self.assertTrue(child['photo_edit_session']['automated_target_met'])
        self.assertFalse(child['photo_edit_session']['human_approved'])
        self.assertEqual(path.read_bytes(),original)

    def test_worse_feedback_candidate_keeps_passed_baseline_without_claiming_new_target_met(self):
        parent=self.passed_parent();original=Path(parent['artifacts']['auto_draft']).read_bytes()
        body=self.request(run_id='feedback-worse-child',parent=parent)
        body['editing_iteration']['user_feedback']='Keep the original expression while improving scene depth.'
        self.repair_outcome='worse'
        response=self.start(self.plan(body));self.assert_started(response);child=response['run']
        self.assertEqual(Path(child['artifacts']['auto_draft']).read_bytes(),original)
        summary=child['photo_edit_session']
        self.assertEqual(summary['review_objective'],'baseline')
        self.assertFalse(summary['objective_reviewed']);self.assertFalse(summary['automated_target_met'])
        self.assertFalse(summary['human_approved'])
        from pipeline_stage_report import read_stage_report
        report=read_stage_report(self.root,child)
        self.assertEqual(report['whole_work']['execution_state'],'complete')
        self.assertEqual(report['whole_work']['result_state'],'needs_evidence')
        self.assertFalse(report['whole_work']['user_feedback']['objective_reviewed'])
        self.assertEqual(report['cosplay_post_diagnosis']['status'],'historical_baseline')
        self.assertFalse(report['cosplay_post_diagnosis']['checks'])
        self.assertEqual(report['cosplay_post_diagnosis']['historical_baseline_review'],
            parent['photo_edit_session']['independent_evaluation'])

    def test_user_feedback_cannot_change_after_checked_plan(self):
        parent=self.passed_parent()
        body=self.request(run_id='feedback-bound-child',parent=parent)
        body['editing_iteration']['user_feedback']='Keep the original expression.'
        checked=self.plan(body);before=(len(self.fx.image_requests),len(self.fx.vision_requests))
        checked['request']['editing_iteration']['user_feedback']='Replace the entire setting.'
        with self.assertRaises(ValueError):self.start(checked)
        self.assertEqual((len(self.fx.image_requests),len(self.fx.vision_requests)),before)

    def test_late_cancellation_retains_parent_and_returned_provider_evidence(self):
        self.cancel_after_image = True
        checked = self.plan()
        started = self.fx.backend.start_task({**checked['request'], 'plan_id': checked['plan']['plan_id'], 'async': True})
        self.cancel_job_id = started['job_id']; self.job_available.set()
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            job = self.fx.backend.jobs.get(self.cancel_job_id)
            if job['status'] in {'failed', 'blocked', 'cancelled', 'succeeded'}:
                break
            time.sleep(.01)
        self.assertEqual(job['status'], 'cancelled')
        worker = self.fx.backend._job_workers.get(self.cancel_job_id)
        if worker is not None:
            worker.join(5)
            self.assertFalse(worker.is_alive())
        self.assertEqual(len(self.fx.image_requests)-self.initial_images, 1)
        self.assertEqual(len(self.fx.vision_requests)-self.initial_vision, 1)
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)
        self.assertTrue(np.array_equal(np.asarray(Image.open(self.parent_draft)), self.parent_pixels))
        attempts = list((self.root/'outputs/runs'/checked['request']['run_id']).rglob('tool-attempt.json'))
        self.assertEqual(len(attempts), 1)
        saved = json.loads(attempts[0].read_text(encoding='utf-8'))
        self.assertEqual(saved['state'], 'interrupted')
        self.assertIn('generated', saved)


if __name__ == '__main__':
    unittest.main()
