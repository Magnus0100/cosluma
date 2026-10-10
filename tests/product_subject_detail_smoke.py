"""Offline diagnostic persistence and product contracts; no service calls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import automatic_subject_detail_completion as completion
import single_source_workflow as workflow
from current_stack_fixture import prepare_fixture
from float_appearance import render_source_float
from native_source_baseline import source_for_plan
from pipeline_stage_report import (build_stage_report, format_stage_report,
    SUBJECT_DIAGNOSTIC_LABELS, SUBJECT_DIAGNOSTIC_ACTIONS)


def fixture(root):
    run = prepare_fixture(root)
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    _, operator = render_source_float(source_for_plan(plan), run)
    run.update(status='needs_visual_review',
        render={'run_id': run['run_id'], 'revision': run['revision'], 'operator': operator})
    run['generation_design'] = {'design_brief': {'primary_effect_plan': {'mode': 'none'}}}
    directory = Path(root)/'outputs/runs'/run['run_id']; directory.mkdir(parents=True)
    run['artifacts'] = {}
    for name in completion.ARTIFACTS:
        path = directory/(name+'.synthetic'); path.write_bytes(('frozen-'+name).encode())
        run['artifacts'][name] = str(path)
    hashes = {name: completion.sha(run['artifacts'][name]) for name in completion.ARTIFACTS}
    failed = {'status': 'fail', 'cause': 'retouch_missing', 'correction_stage': 'native_retouch',
        'reason': 'The current paired hand looks smoother.', 'action': 'Restore missing hand texture.',
        'evidence': [{'basis': 'visible_pixels', 'panels': ['native_0', 'native_1'],
            'observation': 'Current source and final hand differ.'}]}
    passed = {'status': 'pass', 'reason': 'Paired observation passes.', 'action': ''}
    run['cosplay_post_review'] = {'status': 'reviewed', 'actual_api_evidence_valid': True,
        'run_binding': {'run_id': run['run_id'], 'revision': run['revision'],
            'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
            'final_sha256': hashes['auto_draft'], 'render_artifacts': hashes},
        'checks': {key: {'final': copy.deepcopy(failed if key == 'native_detail' else passed)}
            for key in ('identity_makeup', 'concept_hierarchy', 'effects_space',
                        'skin_materials', 'native_detail', 'scene_integration')},
        'local_observations': [{'pair_id': 'left_hand', 'region': 'left_hand',
            'scope': 'matched_source_final_native_pair_only', 'source_panel': 'native_0',
            'final_panel': 'native_1', 'native_box': [2, 2, 20, 18],
            'checks': {'identity_makeup': copy.deepcopy(passed), 'skin_materials': copy.deepcopy(passed),
                       'native_detail': copy.deepcopy(failed)}}]}
    evidence = directory/'review.synthetic.json'; completion.write(evidence, run['cosplay_post_review'])
    diagnosis = {'evidence': {'path': str(evidence), 'sha256': completion.sha(evidence)},
        'final_sha256': hashes['auto_draft'], 'diagnostic_producer_run_id': run['run_id'],
        'consumer_run_id': run['run_id'], 'reviewed_windows': [{'native_box': [2, 2, 20, 18]}],
        'actual_saved_evidence_verified': True, 'evidence_assets': []}
    completion.write(directory/'manifest.json', run)
    completion.write(Path(root)/'project.json', {'selected': run['run_id']})
    return run, diagnosis


def summary(status):
    return {'status': status, 'owners': [{'semantic_class_id': 11, 'native_box': [2, 2, 20, 18],
        'status': status, 'permitted_pixels': 200, 'measured_blocks': 4,
        'fine_gain_medians': [1.0, 1.0], 'reason': 'synthetic_independent_measurement'}],
        'next_action': 'synthetic_algorithm_next_action', 'visual_qa_unchanged': True}


def record(run, diagnosis, status):
    return {'schema': completion.SCHEMA, 'status': 'subject_detail_rejected',
        'initial_run_id': run['run_id'], 'result_run_id': run['run_id'],
        'diagnosis': copy.deepcopy(diagnosis), 'diagnostic_summary': summary(status),
        'input_binding': {'run_id': run['run_id'], 'revision': run['revision'],
            'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
            'final_sha256': diagnosis['final_sha256']},
        'qualification': {'qualified': True}, 'visual_acceptance': False,
        'numerical_gate_grants_visual_acceptance': False}


def main():
    checks = []
    observations = {'assets': {key: True for key in ('source', 'reference', 'draft', 'master',
        'scene_master', 'alpha', 'manifest', 'background', 'background_validity',
        'reference_appearance_delta', 'reference_appearance_colour_delta',
        'reference_appearance_lightness_delta', 'reference_appearance_control_receipt')}, 'review': {}}
    with TemporaryDirectory(prefix='cosplay-product-subject-') as folder:
        root = Path(folder); run, diagnosis = fixture(root/'product')

        def report(value):
            before = copy.deepcopy(value)
            result = build_stage_report(value, observations)
            assert value == before, 'Report changed persisted source facts'
            state = result['whole_work']['native_subject_detail_repair']
            assert not state['pixel_repair_authorized'] and not state['assets_rehashed']
            assert not state['visual_acceptance'] and not state['numerical_gate_grants_visual_acceptance']
            assert not result['visual_approved']
            native = next(row for row in result['whole_work']['open_findings'] if row['criterion'] == 'native_detail')
            assert native['status'] == 'fail'
            return result, state

        for status in SUBJECT_DIAGNOSTIC_LABELS:
            attempted = copy.deepcopy(run); attempted[completion.RECORD_KEY] = record(run, diagnosis, status)
            result, state = report(attempted)
            text = format_stage_report(result)
            assert state['diagnostic_summary_current_binding_matches']
            assert state['last_outcome']['diagnostic_summary'] == summary(status)
            assert SUBJECT_DIAGNOSTIC_LABELS[status] in text and SUBJECT_DIAGNOSTIC_ACTIONS[status] in text
            assert SUBJECT_DIAGNOSTIC_ACTIONS[status] == result['next_action']
            assert '测量块 4' in text and '可测像素 200' in text
            assert result['whole_work']['result_state'] == 'needs_repair'
            checks.append(status+'_distinct_diagnosis_with_chinese_action_without_visual_pass')

        for name, mutate in (
            ('foreign_attempt', lambda value: value[completion.RECORD_KEY].update(initial_run_id='foreign', result_run_id='foreign-child')),
            ('old_revision', lambda value: value[completion.RECORD_KEY]['input_binding'].update(revision=99)),
            ('old_pixels', lambda value: value[completion.RECORD_KEY]['input_binding'].update(final_sha256='old-final')),
            ('old_diagnosed_pixels', lambda value: value[completion.RECORD_KEY]['diagnosis'].update(final_sha256='old-final')),
            ('unverified_diagnosis', lambda value: value[completion.RECORD_KEY]['diagnosis'].update(actual_saved_evidence_verified=False)),
            ('other_source', lambda value: value[completion.RECORD_KEY]['input_binding'].update(source_sha256='other')),
            ('other_reference', lambda value: value[completion.RECORD_KEY]['input_binding'].update(reference_sha256='other')),
            ('stale_review', lambda value: value['cosplay_post_review']['run_binding'].update(revision=99)),
            ('invalid_summary', lambda value: value[completion.RECORD_KEY]['diagnostic_summary'].update(visual_qa_unchanged=False)),
            ('artistic_claim', lambda value: value[completion.RECORD_KEY]['diagnostic_summary'].update(visual_acceptance=True)),
            ('incomplete_binding', lambda value: value[completion.RECORD_KEY].pop('input_binding'))):
            invalid = copy.deepcopy(run)
            invalid[completion.RECORD_KEY] = record(run, diagnosis, 'structured_loss_confirmed'); mutate(invalid)
            if name == 'stale_review':
                result = build_stage_report(invalid, observations)
                state = result['whole_work']['native_subject_detail_repair']
                assert result['whole_work']['result_state'] == 'needs_evidence'
            else:
                result, state = report(invalid)
            assert not state['diagnostic_summary_current_binding_matches']
            assert '历史' in state['diagnostic_summary_label']
            assert not state['saved_child_declared'] and not result['visual_approved']
            checks.append(name+'_cannot_describe_current_pixels_or_recovery_success')

        fake_saved = copy.deepcopy(run)
        fake_saved[completion.RECORD_KEY] = record(run, diagnosis, 'structured_loss_confirmed')
        fake_saved[completion.RECORD_KEY]['status'] = 'subject_detail_child_saved'
        _, state = report(fake_saved)
        assert not state['saved_child_declared'] and '不能据此认为恢复已应用' in state['last_outcome_detail']
        checks.append('same_parent_saved_status_without_owned_child_cannot_claim_recovery_applied')

        owned_child = copy.deepcopy(run)
        owned_child.update(run_id='owned-child', parent_run_id=run['run_id'], revision=run['revision']+1)
        owned_child['render'].update(run_id=owned_child['run_id'], revision=owned_child['revision'])
        owned_child['continuation_edit'] = {'render_state': 'saved', 'rendered_run_id': owned_child['run_id'],
            'rendered_revision': owned_child['revision']}
        owned_child[completion.RECORD_KEY] = record(run, diagnosis, 'structured_loss_confirmed')
        owned_child[completion.RECORD_KEY].update(status='subject_detail_child_saved', result_run_id='owned-child',
            candidate_run_id='owned-child', child_native_artifacts_owned=True)
        owned_child['cosplay_post_review'] = {'status': 'unavailable', 'actual_api_evidence_valid': False}
        result = build_stage_report(owned_child, observations)
        state = result['whole_work']['native_subject_detail_repair']
        assert state['saved_child_declared'] and not state['diagnostic_summary_current_binding_matches']
        assert result['whole_work']['result_state'] == 'needs_evidence' and not result['visual_approved']
        assert '新像素须重新审片' in state['last_outcome_detail']
        checks.append('owned_child_declares_save_but_parent_measurement_cannot_grant_new_visual_QA')

        coordinated = copy.deepcopy(run)
        coordinated[completion.RECORD_KEY] = record(run, diagnosis, 'detail_preserved_diagnostic_conflict')
        coordinated['system_post_completion'] = {'stages': [{'id': 'final_local_repair', 'outcomes': [{
            'producer': 'native_subject_detail', 'status': 'already_attempted',
            'input_run_id': run['run_id'], 'result_run_id': run['run_id']}]}]}
        _, state = report(coordinated)
        assert state['last_outcome']['status'] == 'already_attempted'
        assert state['last_outcome']['diagnostic_summary'] == summary('detail_preserved_diagnostic_conflict')
        assert '不自动重复' in state['last_outcome_detail']
        checks.append('shared_coordinator_outcome_retains_diagnostic_summary_on_no_repeat')

        declared = copy.deepcopy(run)
        plan = declared['options']['appearance_revision']['design_reconstruction_plan']
        plan.setdefault('provenance', {})['native_subject_detail'] = {'state': 'prepared', 'visual_acceptance': False}
        plan['layers'].append({'kind': 'source_subject_detail', 'assets': {
            'qualification_receipt': {'path': 'subject-detail.json', 'sha256': 'declared-receipt'}}})
        _, state = report(declared)
        assert state['prepared_asset_declared'] and state['requires_actual_subject_asset_validation']
        assert state['prepared_asset_declaration_basis'] == 'layer_qualification_receipt_path_and_sha256_only'
        plan['layers'][-1]['assets']['qualification_receipt'].pop('sha256')
        _, state = report(declared); assert not state['prepared_asset_declared']
        checks.append('actual_layer_receipt_declaration_without_nonexistent_producer_receipt_hash')

        for status in SUBJECT_DIAGNOSTIC_LABELS:
            case_root = root/('persist-'+status); case, case_diagnosis = fixture(case_root)
            protected = [case_root/'project.json', case_root/'outputs/runs'/case['run_id']/'manifest.json',
                Path(case['source']['path']), Path(case['reference']['path']),
                Path(case_diagnosis['evidence']['path']), *map(Path, case['artifacts'].values())]
            before = {str(path): completion.sha(path) for path in protected}
            worker = {'changed': False, 'qualification': {'qualified': True}, 'receipt': {
                'source_sha256': case['source']['sha256'], 'reference_sha256': case['reference']['sha256'],
                'diagnosis': case_diagnosis['evidence'], 'diagnostic_summary': summary(status),
                'reason': 'synthetic_no_change', 'numerical_gate': {'visual_acceptance': False}}}
            with patch.object(completion, '_diagnosis', return_value=case_diagnosis), \
                    patch.object(completion, 'prepare_subject_detail', return_value=worker) as cpu, \
                    patch.object(workflow, '_source_revision', side_effect=AssertionError('No child for unchanged diagnostics')):
                result = completion.complete_subject_detail_once(case_root, case, SimpleNamespace())
                assert result['run']['run_id'] == case['run_id']
                assert result['diagnostic_summary'] == summary(status)
                stored = completion.read(case_root/'outputs/runs'/case['run_id']/'automatic-subject-detail-completion/attempt.json')
                assert stored['diagnostic_summary'] == summary(status)
                assert stored['input_binding']['final_sha256'] == case_diagnosis['final_sha256']
                repeated = completion.complete_subject_detail_once(case_root, case, SimpleNamespace())
                assert repeated['status'] == 'already_attempted' and repeated['diagnostic_summary'] == summary(status)
                assert cpu.call_count == 1
            assert before == {str(path): completion.sha(path) for path in protected}
            assert result['run']['cosplay_post_review'] == case['cosplay_post_review']
            checks.append(status+'_persisted_on_unchanged_parent_and_repeat_without_requests_or_old_QA_mutation')

        case_root = root/'foreign-receipt'; case, case_diagnosis = fixture(case_root)
        worker['receipt']['diagnosis'] = {'path': 'foreign-review', 'sha256': 'foreign'}
        with patch.object(completion, '_diagnosis', return_value=case_diagnosis), \
                patch.object(completion, 'prepare_subject_detail', return_value=worker):
            rejected = completion.complete_subject_detail_once(case_root, case, SimpleNamespace())
        assert rejected['status'] == 'subject_detail_rejected'
        assert 'diagnostic_summary' not in rejected['run'][completion.RECORD_KEY]
        assert rejected['run']['cosplay_post_review'] == case['cosplay_post_review']
        checks.append('foreign_no_change_worker_diagnosis_cannot_install_a_current_summary')

        case_root = root/'invalid-artistic-summary'; case, case_diagnosis = fixture(case_root)
        worker['receipt'].update(source_sha256=case['source']['sha256'], reference_sha256=case['reference']['sha256'],
            diagnosis=case_diagnosis['evidence'], diagnostic_summary={**summary('structured_loss_confirmed'), 'visual_acceptance': True})
        with patch.object(completion, '_diagnosis', return_value=case_diagnosis), \
                patch.object(completion, 'prepare_subject_detail', return_value=worker):
            rejected = completion.complete_subject_detail_once(case_root, case, SimpleNamespace())
        assert rejected['status'] == 'subject_detail_rejected'
        assert 'diagnostic_summary' not in rejected['run'][completion.RECORD_KEY]
        assert rejected['run']['cosplay_post_review'] == case['cosplay_post_review']
        checks.append('numerical_summary_attempting_artistic_acceptance_is_rejected')

    print(json.dumps({'ok': True, 'checks': checks, 'model_calls': 0, 'qa_requests': 0,
        'image_requests': 0, 'network_requests': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
