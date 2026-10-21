"""Default checked production uses real session/tools/pixels; model boundaries fake.

Synthetic findings prove execution and preservation contracts, not visual quality.
"""
from io import BytesIO
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from asset_identity import sha256_file
import scoped_production_smoke as production_fixture


class DefaultAutofinishTests(unittest.TestCase):
    def setUp(self):
        self.fx = production_fixture.ScopedProductionTests(methodName='runTest')
        self.fx.setUp(); self.addCleanup(self.fx.doCleanups)
        self.root = self.fx.root
        self.initial_pass = False; self.repaired_pass = True; self.image_failure = None
        self.selected_tool = 'edit_region'
        self.fx.vision.side_effect = self.vision
        self.fx.generation.side_effect = self.generate

    def request(self, source=None, **extra):
        request = self.fx.request(source or self.fx.sources[0], **extra)
        if 'editing_autofinish' not in extra:
            request.pop('editing_autofinish')  # Exercise the actual canonical default.
        return request

    def stage(self,source):
        for parent in Path(source).parents:
            path=parent/'editing-session/session.json'
            if path.is_file():
                state=json.loads(path.read_text(encoding='utf8'))
                return (state['attempts'][-1]['context'].get('operation_stage') if state['attempts'] else 'baseline')
        return None

    def vision(self, source, output, prompt, **kwargs):
        source = Path(source).resolve()
        if 'Select the single most useful available tool' in prompt:
            self.fx.vision_requests.append({'source': str(source), 'sha256': sha256_file(source), 'prompt': prompt})
            result = {'action': 'tool', 'tool': self.selected_tool, 'operations': ['detail_repair'],
                'instruction': 'Repair the visible material seam in this region, preserving its current local illumination.',
                'reason': 'Independent review identified the visible material seam as the main defect.'}
            if self.selected_tool == 'edit_region':
                result['region'] = [.45, .10, .85, .50]
            return result, {'source': {'path': str(source), 'sha256': sha256_file(source)},
                'network_used': True, 'generation_used': False, 'model': 'actual-fixture-decision'}
        payload, provenance = self.fx.vision_request(source, output, prompt, **kwargs)
        if source.name == 'review-board.png':
            mapping = json.loads((source.parent/'review-board.json').read_text(encoding='utf-8'))
            stage=self.stage(source);correction=len(mapping['images'])==3
            passed=self.repaired_pass if stage=='improvement' else self.initial_pass if stage=='initial' else False
            payload['credibility'] = {'status': 'pass' if passed else 'fail',
                'findings': ['Synthetic material is coherent.' if passed else 'Synthetic local material seam is visible.']}
            payload['creative_benefit'] = {'status': 'fail' if stage=='baseline' else 'pass', 'findings': ['Synthetic scene changes are visibly useful.']}
            payload['primary_issue'] = 'none' if passed else 'Repair the visible local material seam.'
            payload['comparison'] = ('worse' if stage=='improvement' and not passed else 'better') if correction else 'first_candidate'
        return payload, provenance

    def generate(self, *args, **kwargs):
        if self.image_failure is not None and self.stage(args[2])=='improvement':
            self.fx.image_requests.append({'image': str(args[2]), 'model': args[3], 'prompt': args[4]})
            raise self.image_failure
        payload, pixels, mode = self.fx.image_request(*args, **kwargs)
        with Image.open(BytesIO(pixels)) as image:
            values = np.asarray(image.convert('RGB')).copy()
        values[:, :, 1] = np.uint8(min(255, 20+len(self.fx.image_requests)*30))
        output = BytesIO(); Image.fromarray(values).save(output, format='PNG')
        return payload, output.getvalue(), mode

    def start(self, **extra):
        return self.fx.start(self.request(**extra))

    def report(self, run):
        return run['editing_autofinish_result']['origin_budget']

    def test_first_edited_pass_on_two_originals_has_one_initial_decision_and_no_improvement(self):
        self.initial_pass = True
        for index, source in enumerate(self.fx.sources):
            self.fx.image_requests.clear(); self.fx.vision_requests.clear()
            checked, run = self.fx.start(self.request(source, run_id='auto-pass-'+str(index)))
            self.assertEqual(checked['plan']['editing_autofinish'], {'maximum_improvements': 2,'maximum_zero_image_finishes':1})
            self.assertEqual(sum(checked['plan']['image_budget'].values()), 3)
            self.assertEqual(checked['plan']['services']['text_vision_requests_upper_bound'], 10)
            self.assertEqual(len(self.fx.image_requests), 1)
            self.assertEqual(len(self.fx.vision_requests), 4)
            self.assertTrue(run['photo_edit_session']['automated_target_met'])
            self.assertFalse(run['photo_edit_session']['human_approved'])
            self.assertEqual(self.report(run)['reserved_image_requests'], 1)
            self.assertEqual(self.report(run)['reserved_vision_requests'], 4)
            self.assertEqual(self.report(run)['improvement_rounds_reserved'], 0)
            self.assertEqual(run['result_publication']['state'], 'committed')

    def test_failed_initial_review_uses_actual_region_decision_tool_and_independent_review(self):
        _checked, run = self.start(run_id='auto-repair')
        self.assertEqual(len(self.fx.image_requests), 2)
        self.assertEqual(len(self.fx.vision_requests), 6)
        self.assertTrue(run['photo_edit_session']['automated_target_met'], run.get('editing_autofinish_result'))
        self.assertEqual(self.report(run)['reserved_image_requests'], 2)
        self.assertEqual(self.report(run)['reserved_vision_requests'], 6)
        self.assertEqual(self.report(run)['improvement_rounds_reserved'], 1)
        state = json.loads(Path(run['photo_edit_session']['path']).read_text(encoding='utf-8'))
        decision = next(row for row in state['attempts'] if row['kind'] == 'decide')['response']
        self.assertEqual(decision['region'], [.45, .10, .85, .50])
        self.assertEqual([row['kind'] for row in state['attempts']], ['evaluate', 'decide', 'tool', 'evaluate', 'decide', 'tool', 'evaluate'])
        review = run['photo_edit_session']['independent_evaluation']
        self.assertEqual(review['comparison'], 'better')
        self.assertEqual(review['observed_images'][1]['sha256'], sha256_file(run['artifacts']['auto_draft']))
        self.assertEqual(run['result_publication']['state'], 'committed')
        self.assertFalse(run['photo_edit_session']['human_approved'])

    def test_last_candidate_fails_and_saved_output_remains_best_complete_initial(self):
        self.repaired_pass = False
        _checked, run = self.start(run_id='auto-keep-best',editing_autofinish={'maximum_improvements':1})
        self.assertEqual(len(self.fx.image_requests), 2)
        self.assertEqual(len(self.fx.vision_requests), 6)
        state = json.loads(Path(run['photo_edit_session']['path']).read_text(encoding='utf-8'))
        self.assertEqual(sha256_file(run['artifacts']['auto_draft']), state['candidates'][1]['image']['sha256'])
        self.assertNotEqual(state['candidates'][1]['image']['sha256'], state['candidates'][2]['image']['sha256'])
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertEqual(self.report(run)['improvement_rounds_reserved'], 1)
        self.assertEqual(run['result_publication']['state'], 'committed')

    def test_model_can_choose_joint_edit_instead_of_region_repair(self):
        self.selected_tool = 'joint_edit'
        _checked, run = self.start(run_id='auto-joint')
        self.assertEqual(len(self.fx.image_requests), 2)
        self.assertEqual(len(self.fx.vision_requests), 6)
        self.assertTrue(run['photo_edit_session']['automated_target_met'], run.get('editing_autofinish_result'))
        state = json.loads(Path(run['photo_edit_session']['path']).read_text(encoding='utf-8'))
        decision = next(row for row in state['attempts'] if row['kind'] == 'decide')['response']
        self.assertEqual(decision['tool'], 'joint_edit')
        self.assertNotIn('region', decision)
        self.assertEqual(self.report(run)['improvement_rounds_reserved'], 1)

    def test_timeout_retains_complete_initial_and_reselection_cannot_retry_tool(self):
        self.image_failure = TimeoutError('Synthetic image outcome is unknown')
        _checked, run = self.start(run_id='auto-timeout')
        self.assertEqual(len(self.fx.image_requests), 2)
        self.assertEqual(len(self.fx.vision_requests), 5)
        self.assertEqual(run['photo_edit_session']['status'], 'needs_attention')
        state = json.loads(Path(run['photo_edit_session']['path']).read_text(encoding='utf-8'))
        self.assertEqual(sha256_file(run['artifacts']['auto_draft']), state['candidates'][1]['image']['sha256'])
        self.assertEqual(self.report(run)['reserved_image_requests'], 2)
        self.assertEqual(self.report(run)['reserved_vision_requests'], 5)
        self.assertEqual(self.report(run)['unknown_operation_reservations'], 1)
        self.image_failure = None
        selected = self.fx.backend.select_source_variant(run['run_id'], {'variant_id': 'v1', 'async': False})
        self.assertTrue(selected.get('ok'), selected)
        child = selected['run']
        self.assertEqual(len(self.fx.image_requests), 2, 'A new revision retried an unknown tool request')
        self.assertEqual(len(self.fx.vision_requests), 5)
        self.assertEqual(self.report(child)['improvement_rounds_reserved'], 1)
        self.assertFalse(child['photo_edit_session']['human_approved'])

    def test_disabled_review_still_has_one_real_initial_decision_and_tool_without_qa(self):
        checked, run = self.start(run_id='auto-disabled', review=False)
        self.assertEqual(checked['plan']['editing_autofinish'], {'maximum_improvements': 0,'maximum_zero_image_finishes':0})
        self.assertEqual(len(self.fx.image_requests), 1)
        self.assertEqual(len(self.fx.vision_requests), 2)
        self.assertFalse(run['photo_edit_session']['objective_reviewed'])
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertIsNone(run['photo_edit_session']['independent_evaluation'])
        self.assertEqual(self.report(run)['reserved_vision_requests'],2)

    def test_reselecting_reviewed_repair_reuses_original_three_image_observation(self):
        _checked, run = self.start(run_id='auto-repaired-reopen')
        self.assertTrue(run['photo_edit_session']['automated_target_met'])
        parent_path = self.root/'outputs/runs'/run['run_id']/'manifest.json'
        parent_bytes = parent_path.read_bytes()
        selected = self.fx.backend.select_source_variant(run['run_id'], {'variant_id': 'v1', 'async': False})
        self.assertTrue(selected.get('ok'), selected)
        child = selected['run']
        self.assertEqual(child['run_id'], run['run_id'], 'Unchanged selected complete version needs no child')
        self.assertEqual(len(self.fx.image_requests), 2)
        self.assertEqual(len(self.fx.vision_requests), 6)
        self.assertTrue(child['photo_edit_session']['automated_target_met'], child.get('editing_autofinish_result'))
        self.assertEqual(child['photo_edit_session']['independent_evaluation']['observed_images'],
                         run['photo_edit_session']['independent_evaluation']['observed_images'])
        self.assertEqual(sha256_file(child['artifacts']['auto_draft']), sha256_file(run['artifacts']['auto_draft']))
        self.assertFalse(child['photo_edit_session']['human_approved'])
        self.assertEqual(parent_path.read_bytes(), parent_bytes)

    def test_three_image_cache_preserves_roles_and_rejects_a_changed_user_objective(self):
        import copy
        import photo_request_budget as budget
        _checked, run = self.start(run_id='auto-three-role-cache')
        connection = {key: run['options'].get(key) for key in
                      ('provider', 'vision_model', 'api_base', 'api_url', 'api_key_env')}
        hooks = budget.make_hooks(self.root, run, connection)
        from photo_appearance import load_asset
        original=load_asset(run['initial_source_appearance'],source_sha256=run['source']['sha256'],
            scope=run['task_plan']['editing_scope'],design_id=run['generation_design']['observed_design']['design_id'])
        context = {'source_original':run['source'],'scope': run['task_plan']['editing_scope'],
            'design': run['generation_design']['observed_design'],
            'user_intent': run['generation_design']['art_direction']['user_intent'],
            'images': [{'role':'source','path':str(original['preview_path']),'sha256':sha256_file(original['preview_path'])},
                       {'role': 'candidate', 'path': run['artifacts']['auto_draft'],
                        'sha256': sha256_file(run['artifacts']['auto_draft'])}]}
        cached = hooks.cached_review(context)
        self.assertEqual(cached, run['photo_edit_session']['independent_evaluation'])
        self.assertEqual([row['role'] for row in cached['observed_images']], ['source', 'candidate', 'previous_best'])
        changed = copy.deepcopy(context)
        changed['user_intent']['brief'] = 'A different user objective must not inherit this acceptance.'
        self.assertIsNone(hooks.cached_review(changed))
        self.assertEqual(len(self.fx.image_requests), 2)
        self.assertEqual(len(self.fx.vision_requests), 6)

    def test_three_candidate_selections_share_two_origin_correction_rounds(self):
        _checked, parent = self.start(run_id='auto-multi', variant_count=3, defer_reference_selection=True,
            editing_autofinish={'maximum_improvements':2})
        parent_path = self.root/'outputs/runs'/parent['run_id']/'manifest.json'
        parent_bytes = parent_path.read_bytes()
        first_response = self.fx.backend.select_source_variant(parent['run_id'], {'variant_id': 'v1', 'async': False})
        self.assertTrue(first_response.get('ok'), first_response)
        first = first_response['run']; self.assertTrue(first['photo_edit_session']['automated_target_met'])
        second_response = self.fx.backend.select_source_variant(first['run_id'], {'variant_id': 'v2', 'async': False})
        self.assertTrue(second_response.get('ok'), second_response)
        second = second_response['run']
        self.assertTrue(second['photo_edit_session']['automated_target_met'])
        third_response = self.fx.backend.select_source_variant(second['run_id'], {'variant_id': 'v3', 'async': False})
        self.assertTrue(third_response.get('ok'), third_response)
        third = third_response['run']
        self.assertEqual(len(self.fx.image_requests), 5)
        self.assertEqual(len(self.fx.vision_requests), 12)
        self.assertEqual(self.report(third)['reserved_image_requests'], 5)
        self.assertEqual(self.report(third)['reserved_vision_requests'], 12)
        self.assertEqual(self.report(third)['improvement_rounds_reserved'], 2)
        self.assertEqual(third['photo_edit_session']['stop_reason'], 'origin_improvement_budget_exhausted')
        self.assertEqual(parent_path.read_bytes(), parent_bytes)
        self.assertFalse(third['photo_edit_session']['human_approved'])


class ContinuousFusionTests(unittest.TestCase):
    def setUp(self):
        self.base = DefaultAutofinishTests(methodName='runTest')
        self.base.setUp(); self.addCleanup(self.base.doCleanups)
        self.fx = self.base.fx; self.root = self.fx.root
        self.first_tool = 'edit_region'; self.sigma = .015
        self.final_timeout = False; self.break_working_receipt = False; self.stop_after_working = False
        self.reviews = 0; self.decisions = 0
        self.fx.vision.side_effect = self.vision

    def vision(self, source, output, prompt, **kwargs):
        source = Path(source).resolve()
        if source.name != 'review-board.png':
            return self.fx.vision_request(source, output, prompt, **kwargs)
        self.fx.vision_requests.append({'source': str(source), 'sha256': sha256_file(source), 'prompt': prompt})
        provenance = {'source': {'path': str(source), 'sha256': sha256_file(source)},
            'network_used': True, 'generation_used': False, 'model': 'actual-synthetic-continuous-vision',
            'input_profile': kwargs.get('input_profile', 'standard')}
        if 'Select the single most useful available tool' in prompt:
            self.decisions += 1
            if self.decisions == 1:
                result = {'action': 'tool', 'tool': self.first_tool, 'operations': ['scene_edit'],
                    'instruction': 'Make the scoped environmental intervention visibly useful.',
                    'reason': 'The independent original comparison identifies a missing principal design benefit.'}
                if self.first_tool == 'edit_region':
                    result['region'] = [.45, .10, .85, .50]
            else:
                if self.decisions==3 and self.sigma==0:
                    return {'action':'stop','reason':'The optical no-op did not improve the image.'},provenance
                self.assertEqual(self.decisions, 2)
                packet=json.loads((Path(output).parent/'decision.json').read_text(encoding='utf8'))
                if 'integrate_generated_focus' not in packet['available_tools']:
                    return {'action':'stop','reason':'No admitted photographic integration tool fits the observed working candidate.'},provenance
                mapping = json.loads((source.parent/'review-board.json').read_text(encoding='utf8'))
                self.assertEqual([row['role'] for row in mapping['images']], ['source', 'best', 'working'])
                self.assertIn('"integrate_generated_focus"', prompt)
                self.assertIn('"inputs_available": true', prompt)
                if self.stop_after_working:
                    return {'action': 'stop', 'reason': 'No safe useful optical change is justified.'}, provenance
                result = {'action': 'tool', 'tool': 'integrate_generated_focus', 'operations': ['detail_repair'],
                    'parameters': {'focus_sigma': self.sigma},
                    'instruction': 'Match only the admitted generated increment to its photographic focus context.',
                    'reason': 'Preserve the useful working design and repair the independently observed integration defect.'}
            return result, provenance
        self.reviews += 1
        if self.reviews == 2 and self.break_working_receipt:
            mapping = json.loads((source.parent/'review-board.json').read_text(encoding='utf8'))
            candidate = mapping['images'][1]
            state = json.loads((Path(candidate['path']).parent.parent/'session.json').read_text(encoding='utf8'))
            row = next(row for row in state['candidates'] if row['image']['sha256'] == candidate['sha256'])
            receipt = Path(row['provenance']['composition_receipt']['path'])
            receipt.write_bytes(receipt.read_bytes()+b'\n')  # Simulate changed, untrusted delta evidence.
        if self.reviews == 3 and self.final_timeout:
            raise TimeoutError('Synthetic final independent-review outcome is unknown')
        if self.reviews == 3 and self.sigma == 0:
            raise AssertionError('An identical review endpoint must reuse evidence, not send again')
        return {'credibility': {'status': 'fail' if self.reviews == 2 else 'pass',
                    'findings': ['Synthetic generated-region focus needs integration.' if self.reviews == 2 else
                                 'Synthetic photographic interactions are coherent.']},
                'creative_benefit': {'status': 'fail' if self.reviews == 1 else 'pass',
                    'findings': ['Synthetic source-relative visual hierarchy is still missing.' if self.reviews == 1 else
                                 'Synthetic source-relative spatial and atmosphere benefit is visible.']},
                'comparison': 'first_candidate' if self.reviews == 1 else 'better',
                'primary_issue': 'none' if self.reviews == 3 else
                    'Integrate the visibly useful generated region with its photographic context.'}, provenance

    def start(self, source=None, **extra):
        return self.fx.start(self.base.request(source, **extra))

    def ledger(self, run):
        report = run['editing_autofinish_result']['origin_budget']
        return report, json.loads(Path(report['path']).read_text(encoding='utf8'))

    def test_default_two_round_loop_generates_then_integrates_on_two_originals(self):
        for index, source in enumerate(self.fx.sources):
            self.fx.image_requests.clear(); self.fx.vision_requests.clear()
            self.reviews = self.decisions = 0
            checked, run = self.start(source, run_id='default-continuous-'+str(index))
            self.assertEqual(checked['plan']['editing_autofinish'], {'maximum_improvements': 2,'maximum_zero_image_finishes':1})
            self.assertEqual(sum(checked['plan']['image_budget'].values()), 3)
            self.assertEqual(checked['plan']['services']['text_vision_requests_upper_bound'], 10)
            self.assertEqual(len(self.fx.image_requests), 1)
            self.assertEqual(len(self.fx.vision_requests), 6)
            self.assertTrue(run['photo_edit_session']['automated_target_met'], run.get('editing_autofinish_result'))
            state = json.loads(Path(run['photo_edit_session']['path']).read_text(encoding='utf-8'))
            reviews = [row['response']['provenance'] for row in state['attempts'] if row['kind'] == 'evaluate']
            self.assertEqual([row['input_profile'] for row in reviews], ['standard', 'detailed_board', 'detailed_board'])
            self.assertEqual([row['review_inspection_scope'] for row in reviews],
                             ['whole_image', 'whole_image_and_edit_region', 'whole_image_and_edit_region'])
            report, ledger = self.ledger(run)
            self.assertEqual(report['reserved_image_requests'], 1)
            self.assertEqual(report['confirmed_image_submissions'], 1)
            self.assertEqual(report['reserved_vision_requests'], 6)
            self.assertEqual(report['improvement_rounds_reserved'], 1)
            tools = [row for row in ledger['operations'] if row['kind'] == 'tool']
            self.assertEqual([row['costs']['image_submissions'] for row in tools], [1, 0])
            self.assertEqual(tools[1]['actual_image_submissions'], 0)
            self.assertEqual(run['photo_edit_session']['independent_evaluation']['observed_images'][1]['sha256'],
                             sha256_file(run['artifacts']['auto_draft']))
            self.assertEqual(run['result_publication']['state'], 'committed')
            self.assertFalse(run['photo_edit_session']['human_approved'])

    def test_global_working_is_not_admitted_and_model_explicitly_stops(self):
        self.first_tool = 'joint_edit'
        _checked, run = self.start(run_id='default-global-not-focusable')
        self.assertEqual(self.decisions, 2)
        self.assertEqual(len(self.fx.image_requests), 1)
        self.assertEqual(len(self.fx.vision_requests), 5)
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertEqual(run['photo_edit_session']['stop_reason'], 'No admitted photographic integration tool fits the observed working candidate.')
        report, _ledger = self.ledger(run)
        self.assertEqual(report['improvement_rounds_reserved'], 1)

    def test_zero_sigma_reuses_the_actual_identical_review_without_a_new_request(self):
        self.sigma = 0
        _checked, run = self.start(run_id='default-noop-cached-review')
        self.assertEqual(len(self.fx.image_requests), 1)
        self.assertEqual(len(self.fx.vision_requests), 6)
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        report, ledger = self.ledger(run)
        self.assertEqual(report['reserved_image_requests'], 1)
        self.assertEqual(report['reserved_vision_requests'], 6)
        self.assertEqual(report['confirmed_vision_requests'], 6)
        self.assertEqual(run['photo_edit_session']['reserved_budget_usage']['vision_requests'], 6)
        self.assertEqual(report['historical_design_requests'], 1)
        reused = [row for row in ledger['operations'] if row.get('reused_evidence')]
        self.assertEqual(len(reused), 1)
        self.assertEqual(reused[0]['costs']['vision_requests'], 0)
        self.assertEqual(reused[0]['actual_model_requests'], 0)
        self.assertEqual(reused[0]['result']['provenance']['model_requests'], 1,
                         'Retain the original observation provenance, never relabel it as a new call')

    def test_changed_delta_is_not_offered_and_model_stops_without_running_it(self):
        self.break_working_receipt = True
        _checked, run = self.start(run_id='default-invalid-working-evidence')
        self.assertEqual(self.decisions, 2)
        self.assertEqual(len(self.fx.image_requests), 1)
        self.assertEqual(len(self.fx.vision_requests), 5)
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertEqual(run['photo_edit_session']['stop_reason'], 'No admitted photographic integration tool fits the observed working candidate.')

    def test_explicit_zero_improvement_limit_stops_after_initial_even_with_eligible_working(self):
        checked, run = self.start(run_id='explicit-k0-limit', editing_autofinish={'maximum_improvements': 0})
        self.assertEqual(checked['plan']['editing_autofinish'], {'maximum_improvements': 0,'maximum_zero_image_finishes':0})
        self.assertEqual(self.decisions, 1)
        self.assertEqual(len(self.fx.image_requests), 1)
        self.assertEqual(len(self.fx.vision_requests), 4)
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertEqual(run['photo_edit_session']['stop_reason'], 'origin_improvement_budget_exhausted')
        self.assertEqual(self.ledger(run)[0]['improvement_rounds_reserved'], 0)

    def test_model_can_stop_instead_of_running_the_admitted_cpu_tool(self):
        self.stop_after_working = True
        _checked, run = self.start(run_id='default-model-stop-working')
        self.assertEqual(self.decisions, 2)
        self.assertEqual(len(self.fx.image_requests), 1)
        self.assertEqual(len(self.fx.vision_requests), 5)
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertEqual(run['photo_edit_session']['stop_reason'], 'No safe useful optical change is justified.')
        report, ledger = self.ledger(run)
        self.assertEqual(report['improvement_rounds_reserved'], 1)
        self.assertEqual(len([row for row in ledger['operations'] if row['kind'] == 'tool']), 1)

    def test_unknown_final_review_keeps_quota_and_cannot_be_retried(self):
        import photo_request_budget as budget
        self.final_timeout = True
        _checked, run = self.start(run_id='default-final-review-unknown')
        self.assertEqual(len(self.fx.image_requests), 1)
        self.assertEqual(len(self.fx.vision_requests), 6)
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertEqual(run['photo_edit_session']['status'], 'needs_attention')
        report, _ledger = self.ledger(run)
        self.assertEqual(report['reserved_image_requests'], 1)
        self.assertEqual(report['reserved_vision_requests'], 6)
        self.assertEqual(report['unknown_operation_reservations'], 1)
        state = json.loads(Path(run['photo_edit_session']['path']).read_text(encoding='utf8'))
        unknown = state['attempts'][-1]
        self.assertEqual(sha256_file(run['artifacts']['auto_draft']), state['candidates'][0]['image']['sha256'])
        with self.assertRaises(budget.ReviewPending):
            budget.reserve_operation(report, 'post_review', unknown['context'])
        self.assertEqual(len(self.fx.vision_requests), 6)
        self.assertEqual(budget.snapshot(report)['improvement_rounds_reserved'], 1)

    def test_ledger_cannot_reclassify_actual_generation_as_a_free_cpu_tool(self):
        import copy
        import photo_request_budget as budget
        _checked, run = self.start(run_id='default-immutable-tool-fees')
        report, ledger = self.ledger(run)
        path = Path(report['path']); original = path.read_bytes()
        altered = copy.deepcopy(ledger)
        generation = next(row for row in altered['operations'] if row['kind'] == 'tool'
                          and row['costs']['image_submissions'] == 1)
        generation['costs']['image_submissions'] = 0
        generation['tool_costs']['image_submissions'] = 0
        path.write_text(json.dumps(altered), encoding='utf8')
        with self.assertRaises(ValueError):
            budget.snapshot(report)
        path.write_bytes(original)
        self.assertEqual(budget.snapshot(report)['reserved_image_requests'], 1)
        self.assertEqual(len(self.fx.image_requests), 1)
        self.assertEqual(len(self.fx.vision_requests), 6)


class ZeroImageFinishTests(unittest.TestCase):
    def setUp(self):
        self.base=DefaultAutofinishTests(methodName='runTest');self.base.setUp();self.addCleanup(self.base.doCleanups)
        self.fx=self.base.fx;self.root=self.base.root;self.decisions=0;self.final_action='tool';self.early_pass=False
        self.fx.vision.side_effect=self.vision
        self.finish_context=None

    def vision(self,source,output,prompt,**kwargs):
        source=Path(source)
        if source.name!='review-board.png':return self.fx.vision_request(source,output,prompt,**kwargs)
        self.fx.vision_requests.append({'source':str(source),'sha256':sha256_file(source),'prompt':prompt})
        provenance={'source':{'path':str(source),'sha256':sha256_file(source)},'network_used':True,
            'generation_used':False,'model':'synthetic-bounded-zero-image-finish',
            'input_profile':kwargs.get('input_profile','standard')}
        if 'Select the single most useful available tool' in prompt:
            self.decisions+=1
            if self.decisions==4:
                self.finish_context=json.JSONDecoder().raw_decode(prompt.split('CONTEXT\n',1)[1])[0]
                self.assertTrue(self.finish_context['zero_image_only'])
                self.assertTrue(self.finish_context['tools'])
                self.assertTrue(all(tool['costs']['image_submissions']==0 for tool in self.finish_context['tools'].values()))
                self.assertNotIn('joint_edit',self.finish_context['tools']);self.assertNotIn('edit_region',self.finish_context['tools'])
                self.assertEqual(self.finish_context['unavailable_tools']['joint_edit']['reason'],'checked_zero_image_finish_only')
                if self.final_action=='unknown':raise TimeoutError('The final decision outcome is unknown')
                if self.final_action=='stop':return {'action':'stop','reason':'No useful zero-image finish remains.'},provenance
            tool='develop_photograph' if self.decisions in (1,4) else 'edit_region'
            payload={'action':'tool','tool':tool,'operations':['relight'],
                'instruction':'Correct the observed photographic balance.','reason':'Synthetic evidence selects this bounded operation.'}
            if tool=='develop_photograph':
                payload['parameters']={'adjustments':[{'region':[0,0,1,1],'exposure_ev':.2,
                    'warmth_ev':0.,'tint_ev':0.,'feather':0.}]}
            else:payload['region']=[.2,.2,.8,.8]
            return payload,provenance
        mapping=json.loads((source.parent/'review-board.json').read_text(encoding='utf8'))
        passed=self.early_pass and self.decisions>=2
        return {'credibility':{'status':'pass' if passed else 'fail','findings':['Synthetic whole-image credibility evidence.']},
            'creative_benefit':{'status':'pass','findings':['Synthetic creative evidence.']},
            'comparison':'better' if len(mapping['images'])==3 else 'first_candidate',
            'primary_issue':'Correct the remaining photographic imbalance.'},provenance

    def start(self):return self.fx.start(self.base.request(run_id='bounded-zero-image-finish'))

    def test_cpu_initial_two_images_then_cpu_finish_has_three_image_ten_vision_bounds(self):
        checked,run=self.start();report=run['editing_autofinish_result']['origin_budget']
        self.assertEqual(checked['plan']['generation_submissions_upper_bound'],3)
        self.assertEqual(checked['plan']['services']['text_vision_requests_upper_bound'],10)
        self.assertEqual(len(self.fx.image_requests),2);self.assertEqual(len(self.fx.vision_requests),10)
        self.assertEqual(report['reserved_image_requests'],2);self.assertEqual(report['reserved_vision_requests'],10)
        self.assertEqual(report['improvement_rounds_reserved'],3);self.assertEqual(report['zero_image_finish_rounds_reserved'],1)
        state=json.loads(Path(run['photo_edit_session']['path']).read_text(encoding='utf8'))
        tools=[row for row in state['attempts'] if row['kind']=='tool']
        self.assertEqual([row['response']['resources']['image_submissions'] for row in tools],[0,1,1,0])
        self.assertEqual(state['stop_reason'],'origin_improvement_budget_exhausted')
        self.assertEqual(run['result_publication']['state'],'committed')
        before=len(self.fx.vision_requests)
        response=self.fx.backend.select_source_variant(run['run_id'],{'variant_id':'v1','async':False})
        self.assertTrue(response.get('ok'),response);self.assertEqual(len(self.fx.vision_requests),before)

    def test_early_pass_does_not_spend_finish(self):
        self.early_pass=True;_,run=self.start()
        self.assertTrue(run['photo_edit_session']['automated_target_met'])
        self.assertEqual(len(self.fx.vision_requests),6);self.assertIsNone(self.finish_context)
        self.assertEqual(run['editing_autofinish_result']['origin_budget']['zero_image_finish_rounds_reserved'],0)

    def test_final_stop_and_unknown_decisions_consume_slot_without_retry(self):
        for action in ('stop','unknown'):
            with self.subTest(action=action):
                if action=='unknown':
                    self.base.doCleanups();self.setUp()
                self.final_action=action;_,run=self.start()
                report=run['editing_autofinish_result']['origin_budget']
                self.assertEqual(report['zero_image_finish_rounds_reserved'],1)
                self.assertEqual(report['reserved_vision_requests'],9)
                self.assertEqual(len(self.fx.image_requests),2)
                before=len(self.fx.vision_requests)
                response=self.fx.backend.select_source_variant(run['run_id'],{'variant_id':'v1','async':False})
                self.assertTrue(response.get('ok'),response);self.assertEqual(len(self.fx.vision_requests),before)
                self.assertEqual(report['unknown_operation_reservations'],int(action=='unknown'))


if __name__ == '__main__':
    unittest.main()
