"""Offline actual-QA replay, local worker, product and owned-child contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import automatic_subject_environment_completion as completion
import automatic_subject_detail_completion_smoke as detail_smoke
import single_source_workflow as workflow
import system_post_completion as system
from cosplay_post_review_smoke import response, verdict, multi_fake_transport
from cosplay_repair_plan import plan_final_repairs
from generation_design import DESIGN_VERSION
from pipeline_stage_report import build_stage_report, format_stage_report


def fixture(root, *, saved_run_binding=True):
    payload = response(True)
    payload['checks']['scene_integration']['final'] = verdict('final', status='fail',
        cause='retouch_missing', stage='scene_composition',
        action='Match locally observed subject hair and garment environmental response.',
        panels=['reference', 'final'])
    with patch.object(detail_smoke, 'multi_fake_transport', side_effect=lambda: multi_fake_transport(payload)):
        run = detail_smoke.fixture(root, saved_run_binding=saved_run_binding)
    run['generation_budget'] = {'max_image_submissions': 0, 'total_image_submissions': 0,
        'reference_correction_budget': 0, 'independent_effect_image_budget': 0}
    completion.write(Path(root)/'outputs/runs'/run['run_id']/'manifest.json', run)
    return run


def guard(root, run, diagnosis):
    paths = [Path(root)/'project.json', Path(root)/'outputs/runs'/run['run_id']/'manifest.json',
        Path(run['source']['path']), Path(run['reference']['path'])]
    paths += [Path(run['artifacts'][key]) for key in (*completion.ARTIFACTS, 'metadata')]
    paths += [Path(row['path']) for row in diagnosis['evidence_assets']]
    paths += list((Path(root)/'outputs/runs'/run['run_id']).glob('*/attempt.json'))
    return {str(path): completion.sha(path) for path in paths}


def main():
    checks = []
    with TemporaryDirectory(prefix='environment-coordinator-') as folder, \
            patch('native_subject_detail.validate_diagnosed_plate', side_effect=detail_smoke.plate):
        root = Path(folder); original_root = root/'original'; run = fixture(original_root)
        diagnosis = completion._diagnosis(original_root, run)
        assert diagnosis['actual_saved_evidence_verified'] and diagnosis['failed_criteria'] == ['scene_integration']
        classified = plan_final_repairs(run, current_review=True)
        task = next(row for row in classified['tasks'] if row['criterion'] == 'scene_integration')
        assert task['producer'] == completion.PRODUCER and task['pixel_repair_qualified']
        assert not task['generation_authorized_by_this_plan'] and classified['image_generation_budget_added'] == 0
        checks.append('actual_comprehensive_scene_response_QA_replayed_and_routed_without_image_budget')
        for name, mutate in (
            ('unknown', lambda value: value['cosplay_post_review']['checks']['scene_integration']['final'].update(status='unknown')),
            ('reference_stage', lambda value: value['cosplay_post_review']['checks']['scene_integration']['final'].update(correction_stage='reference_generation')),
            ('effects_criterion', lambda value: value['cosplay_post_review']['checks'].update(effects_space=value['cosplay_post_review']['checks'].pop('scene_integration')))):
            changed = copy.deepcopy(run); mutate(changed)
            assert completion._diagnosis(original_root, changed) is None
            assert not any(row['producer'] == completion.PRODUCER for row in plan_final_repairs(changed, current_review=True)['tasks'])
            checks.append(name+'_does_not_authorize_subject_environment_response')
        background = copy.deepcopy(run)
        background['cosplay_post_review']['checks']['scene_integration']['final']['cause'] = 'transfer_incomplete'
        assert completion._diagnosis(original_root, background) is None
        assert any(row['producer'] == 'background_consistency' for row in plan_final_repairs(background, current_review=True)['tasks'])
        checks.append('background_transfer_route_retains_its_separate_owner')
        for name, mutate in (
            ('revision', lambda value: value.update(revision=99)), ('run', lambda value: value.update(run_id='foreign')),
            ('master', lambda value: value['artifacts'].update(render_scene_master=value['artifacts']['render_master']))):
            stale = copy.deepcopy(run); mutate(stale)
            with patch.object(completion, 'prepare_subject_environment', side_effect=AssertionError('Stale QA cannot run')):
                rejected = completion.complete_subject_environment_once(original_root, stale, SimpleNamespace())
            assert rejected['status'] == 'subject_environment_evidence_unsupported'
            checks.append('stale_'+name+'_rejected_before_worker')

        original_review = original_root/'outputs/runs'/run['run_id']/'cosplay-post-review/review.json'
        original_bytes = original_review.read_bytes()
        def consumer(name, source=run, evidence=original_review):
            local_root = root/name; local_root.mkdir()
            value = copy.deepcopy(source); value['root'] = str(local_root)
            value[completion.EXTERNAL_REVIEW_KEY] = {'schema': completion.EXTERNAL_REVIEW_SCHEMA,
                'path': str(evidence), 'sha256': completion.sha(evidence)}
            completion.write(local_root/'outputs/runs'/value['run_id']/'manifest.json', value)
            completion.write(local_root/'project.json', {'selected': value['run_id']})
            return local_root, value
        consumer_root, external = consumer('external')
        consumed = completion._diagnosis(consumer_root, external)
        assert consumed['external_review_consumed_read_only'] and consumed['evidence'] == diagnosis['evidence']
        assert not (consumer_root/'outputs/runs'/run['run_id']/'cosplay-post-review').exists()
        checks.append('external_complete_QA_consumed_read_only_without_copy_or_new_request')
        for name, mutate in (
            ('schema', lambda value: value[completion.EXTERNAL_REVIEW_KEY].update(schema='other')),
            ('hash', lambda value: value[completion.EXTERNAL_REVIEW_KEY].update(sha256='0'*64)),
            ('native_identity', lambda value: value['cosplay_post_review']['native_selection']['run_binding'].update(run_id='other')),
            ('final_size', lambda value: value['cosplay_post_review']['native_selection']['final'].update(size=[1, 1])),
            ('final_path', lambda value: value['cosplay_post_review']['binding']['images']['final'].update(path=str(original_review))),
            ('verdict', lambda value: value['cosplay_post_review']['checks']['scene_integration']['final'].update(reason='invented'))):
            changed = copy.deepcopy(external); mutate(changed)
            assert completion.complete_subject_environment_once(consumer_root, changed, SimpleNamespace())['status'] == 'subject_environment_evidence_unsupported'
            assert original_review.read_bytes() == original_bytes
            checks.append('external_'+name+'_forgery_cannot_authorize_a_worker')
        # Rehashing an invented aggregate cannot replace actual saved boards.
        forged = completion.read(original_review)
        forged['checks']['scene_integration']['final']['reason'] = 'Invented aggregate despite unchanged raw boards'
        completion.write(original_review, forged)
        changed = copy.deepcopy(external); changed['cosplay_post_review'] = forged
        changed[completion.EXTERNAL_REVIEW_KEY]['sha256'] = completion.sha(original_review)
        try:
            assert completion.complete_subject_environment_once(consumer_root, changed, SimpleNamespace())['status'] == 'subject_environment_evidence_unsupported'
        finally:
            original_review.write_bytes(original_bytes)
        checks.append('rehashed_invented_aggregate_rejected_by_actual_board_replay')

        no_binding_root = root/'original-without-consumer-binding'
        producer = fixture(no_binding_root, saved_run_binding=False)
        producer_evidence = no_binding_root/'outputs/runs'/producer['run_id']/'cosplay-post-review/review.json'
        missing_binding_root, missing_binding = consumer('added-consumer-binding', producer, producer_evidence)
        assert 'run_binding' not in completion.read(producer_evidence)
        assert completion._diagnosis(missing_binding_root, missing_binding)['external_review_consumed_read_only']
        checks.append('external_original_without_run_binding_accepts_only_verified_consumer_binding')

        def worker(current, current_diagnosis, *, changed=True):
            return {'changed': changed, 'reason': 'controlled_source_owned_response',
                'receipt': {'state': 'prepared' if changed else 'unsupported', 'visual_acceptance': False,
                    'source_sha256': current['source']['sha256'], 'reference_sha256': current['reference']['sha256'],
                    'diagnosis': current_diagnosis['evidence'],
                    'qualification': {'qualified': True, 'actual_saved_evidence_verified': True, 'holdout_improved': True},
                    'checks': {'changed_pixels': 70 if changed else 0, 'alpha_changed': False,
                        'source_geometry_changed': False, 'reference_texture_imported': False, 'unknown_support_changed': False}},
                'design_reconstruction_plan': copy.deepcopy(current['options']['appearance_revision']['design_reconstruction_plan'])}
        def revision(local_root, run_id, *, snapshot, design_plan_override):
            assert snapshot is False
            current = completion.read(local_root/'outputs/runs'/run_id/'manifest.json')
            child = copy.deepcopy(current); child.update(run_id='environment-child', parent_run_id=run_id, revision=current['revision']+1)
            child['options']['appearance_revision']['design_reconstruction_plan'] = copy.deepcopy(design_plan_override)
            child['artifacts'] = {}
            return local_root/'outputs/runs'/child['run_id']/'manifest.json', child
        def render(local_root, run_id):
            path = local_root/'outputs/runs'/run_id/'manifest.json'; child = completion.read(path)
            assert child['cosplay_post_review']['status'] == 'unavailable'
            assert not child['cosplay_post_review']['actual_api_evidence_valid']
            assert completion.EXTERNAL_REVIEW_KEY not in child
            for key in completion.ARTIFACTS:
                target = path.parent/(key+'.synthetic'); target.write_bytes(('new-'+key).encode())
                child['artifacts'][key] = str(target)
            child['continuation_edit'].update(render_state='saved', rendered_run_id=run_id, rendered_revision=child['revision'])
            return child
        for prior in ('automatic-subject-detail-completion', 'automatic-scene-completion'):
            completion.write(consumer_root/'outputs/runs'/external['run_id']/prior/'attempt.json',
                {'status': 'previous_attempt_preserved', 'attempt': 1})
        protected = guard(consumer_root, external, consumed)
        with patch.object(completion, 'prepare_subject_environment', return_value=worker(external, consumed)) as cpu, \
                patch.object(workflow, '_source_revision', side_effect=revision), patch.object(workflow, 'adjust_source_run', side_effect=render):
            saved = completion.complete_subject_environment_once(consumer_root, external, SimpleNamespace())
            assert saved['status'] == 'subject_environment_child_saved', saved
            repeated = completion.complete_subject_environment_once(consumer_root, external, SimpleNamespace())
            assert repeated['status'] == 'already_attempted' and cpu.call_count == 1
        assert saved['run']['generation_budget'] == external['generation_budget']
        assert saved['run'][completion.RECORD_KEY]['original_draft_retained']
        assert saved['run'][completion.RECORD_KEY]['requires_new_visual_QA']
        assert protected == {name: completion.sha(name) for name in protected}
        checks.extend(['owned_child_saved_with_no_inherited_QA', 'original_assets_QA_budget_and_selection_exact',
            'persistent_single_attempt_does_not_repeat_or_reset_old_attempts'])

        for name, mutate, expected in (
            ('unsupported', lambda value: value.update(changed=False), 'subject_environment_evidence_unsupported'),
            ('zero_change', lambda value: value['receipt']['checks'].update(changed_pixels=0), 'subject_environment_rejected'),
            ('holdout_failure', lambda value: value['receipt']['qualification'].update(holdout_improved=False), 'subject_environment_rejected'),
            ('unknown_owner_change', lambda value: value['receipt']['checks'].update(unknown_support_changed=True), 'subject_environment_rejected'),
            ('foreign_receipt', lambda value: value['receipt'].update(source_sha256='foreign'), 'subject_environment_rejected'),
            ('foreign_diagnosis', lambda value: value['receipt'].update(diagnosis={'path': 'foreign', 'sha256': 'foreign'}), 'subject_environment_rejected'),
            ('network_attempt', lambda value: value['receipt'].update(text_vision_submissions=1), 'subject_environment_rejected'),
            ('artistic_approval', lambda value: value['receipt'].update(visual_acceptance=True), 'subject_environment_rejected')):
            case_root, case = consumer(name); case_diagnosis = completion._diagnosis(case_root, case)
            before = guard(case_root, case, case_diagnosis); prepared = worker(case, case_diagnosis)
            mutate(prepared)
            if not prepared['changed']: prepared['receipt']['state'] = 'unsupported'
            with patch.object(completion, 'prepare_subject_environment', return_value=prepared), \
                    patch.object(workflow, '_source_revision', side_effect=AssertionError('Rejected response cannot save child')):
                rejected = completion.complete_subject_environment_once(case_root, case, SimpleNamespace())
            assert rejected['status'] == expected, rejected
            assert rejected['run']['cosplay_post_review'] == case['cosplay_post_review']
            assert before == {key: completion.sha(key) for key in before}
            assert completion.complete_subject_environment_once(case_root, case, SimpleNamespace())['status'] == 'already_attempted'
            checks.append(name+'_retains_parent_failures_and_durable_attempt')

        case_root, case = consumer('cancelled'); case_diagnosis = completion._diagnosis(case_root, case)
        signal = case_root/'cancel.requested'; signal.write_bytes(b'cancel')
        with patch.object(completion, 'prepare_subject_environment', side_effect=AssertionError('Cancelled before worker')):
            cancelled = completion.complete_subject_environment_once(case_root, case, SimpleNamespace(cancel_request_path=signal))
        assert cancelled['status'] == 'subject_environment_cancelled' and cancelled['run'][completion.RECORD_KEY]['cancelled']
        signal.unlink(); assert completion.complete_subject_environment_once(case_root, case, SimpleNamespace())['status'] == 'already_attempted'
        checks.append('preworker_cancellation_keeps_durable_marker_and_parent')

        class Running:
            def __init__(self): self.terminated = False; self.returncode = None
            def poll(self): return self.returncode
            def terminate(self): self.terminated = True; self.returncode = -15
            def wait(self, timeout=None): return self.returncode
            def kill(self): self.returncode = -9
        running = Running(); calls = iter([False, True])
        with patch('current_model_registry.load_registry', return_value={'python': str(Path(__import__('sys').executable))}), \
                patch.object(completion.subprocess, 'Popen', return_value=running):
            try: completion.prepare_subject_environment(root, run, root/'cpu-cancel', cancel=lambda: next(calls))
            except InterruptedError: pass
            else: raise AssertionError('Running worker ignored cancellation')
        assert running.terminated
        checks.append('running_CPU_worker_terminated_without_a_submitted_service_request')

        observations = {'assets': {key: True for key in ('source', 'reference', 'draft', 'master', 'scene_master',
            'alpha', 'manifest', 'background', 'background_validity', 'reference_appearance_delta',
            'reference_appearance_colour_delta', 'reference_appearance_lightness_delta', 'reference_appearance_control_receipt')}, 'review': {}}
        current = copy.deepcopy(run); current['input_mode'] = 'single_source'; before = copy.deepcopy(current)
        report = build_stage_report(current, observations)
        assert current == before
        environment = report['whole_work']['subject_environment_repair']
        assert environment['candidate_from_saved_review'] and not environment['pixel_repair_authorized'], {
            'environment': environment, 'diagnosis': report['cosplay_post_diagnosis'], 'draft': report['draft_state'],
            'run_id': current['run_id'], 'revision': current['revision'], 'render': current.get('render')}
        assert not environment['visual_acceptance'] and not environment['assets_rehashed']
        assert '人物与环境光色协调' in format_stage_report(report)
        finding = next(row for row in report['whole_work']['open_findings'] if row['criterion'] == 'scene_integration')
        assert finding['status'] == 'fail' and finding['local_repair']['capability'] == completion.PRODUCER
        stale = copy.deepcopy(run); stale['cosplay_post_review']['run_binding']['revision'] = 99
        assert not build_stage_report(stale, observations)['whole_work']['subject_environment_repair']['candidate_from_saved_review']
        fake_saved = copy.deepcopy(run); fake_saved[completion.RECORD_KEY] = {
            'schema': completion.SCHEMA, 'status': 'subject_environment_child_saved',
            'initial_run_id': run['run_id'], 'result_run_id': run['run_id']}
        state = build_stage_report(fake_saved, observations)['whole_work']['subject_environment_repair']
        assert not state['saved_child_declared'] and '不能据此认为已应用' in state['last_outcome_detail']
        fake_saved[completion.RECORD_KEY].update(initial_run_id='foreign', result_run_id='foreign-child')
        state = build_stage_report(fake_saved, observations)['whole_work']['subject_environment_repair']
        assert not state['last_outcome']['applies_to_run']
        checks.extend(['shared_product_report_names_real_automatic_action_without_visual_pass',
            'stale_QA_and_foreign_or_unowned_outcome_cannot_claim_current_recovery'])

        # The system boundary is controlled here; diagnosis above replays real
        # saved request assets. A fresh endpoint is reviewed once after repair.
        system_root, initial = consumer('system'); initial['generation_design'].update(schema_version=DESIGN_VERSION, status='ready')
        initial.pop('system_post_completion', None); initial.pop('automatic_subject_detail_completion', None)
        completion.write(system_root/'outputs/runs'/initial['run_id']/'manifest.json', initial)
        calls = []
        def local_response(_root, current, args, *, bounded):
            assert bounded is True and system._review_current(current)
            calls.append('environment')
            child = copy.deepcopy(current); child.update(run_id='system-environment-child', parent_run_id=current['run_id'], revision=current['revision']+1)
            path = _root/'outputs/runs'/child['run_id']/'draft.synthetic'; path.parent.mkdir(parents=True); path.write_bytes(b'new-actual-endpoint')
            child['artifacts'] = {'auto_draft': str(path)}; child['cosplay_post_review'] = {'status': 'unavailable'}
            completion.write(path.parent/'manifest.json', child)
            return {'run': child, 'status': 'subject_environment_child_saved'}
        def fresh_review(_root, current, args):
            calls.append('new_QA')
            digest = completion.sha(current['artifacts']['auto_draft'])
            current['cosplay_post_review'] = {'status': 'reviewed', 'actual_api_evidence_valid': True,
                'run_binding': {'run_id': current['run_id'], 'revision': current['revision'],
                    'source_sha256': current['source']['sha256'], 'reference_sha256': current['reference']['sha256'],
                    'final_sha256': digest, 'render_artifacts': {'auto_draft': digest}},
                'binding': {'images': {'final': {'path': current['artifacts']['auto_draft'], 'sha256': digest}}},
                'checks': {'scene_integration': {'final': {'status': 'pass'}},
                    'native_detail': {'final': {'status': 'unknown', 'correction_stage': 'native_retouch'}}}}
        args = SimpleNamespace(final_repair_only=True, complete_post=True, generate=False,
            preserve_effective_prompt=False, cosplay_correction_budget=0, cancel_request_path=None)
        with patch.object(completion, 'complete_subject_environment_once', side_effect=local_response) as producer, \
                patch.object(workflow, 'diagnose_cosplay_post', side_effect=fresh_review) as qa, \
                patch('cosplay_post_completion.continue_cosplay_post', side_effect=AssertionError('Final repair cannot submit reference')), \
                patch('automatic_foreground_effects.complete_effects_once', side_effect=AssertionError('Final repair cannot submit effect')):
            result = system.complete_once(system_root, initial, args)
            assert calls == ['environment', 'new_QA'], result
            assert result['system_post_completion']['unknown_checks'] == ['native_detail']
            assert result['system_post_completion']['visual_state'] == 'findings_open'
            assert not result['system_post_completion']['automatic_visual_acceptance']
            assert result['system_post_completion']['reference_image_budget'] == 0
            assert result['system_post_completion']['independent_effect_image_budget'] == 0
            assert system.complete_once(system_root, initial, args)['run_id'] == result['run_id']
            assert producer.call_count == qa.call_count == 1
        assert original_review.read_bytes() == original_bytes
        checks.append('system_routes_local_response_then_new_QA_once_preserving_unknowns_and_zero_images')

    print(json.dumps({'status': 'passed', 'cases': checks, 'API_requests': 0,
        'image_generation_submissions': 0, 'model_inferences': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
