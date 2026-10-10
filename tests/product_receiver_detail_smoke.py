"""Offline product regression for diagnosis-scoped receiver repair reporting."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from current_stack_fixture import prepare_fixture
from float_appearance import render_source_float
from native_source_baseline import source_for_plan
from pipeline_stage_report import build_stage_report, format_stage_report


def main():
    checks = []
    with TemporaryDirectory(prefix='cosplay-product-receiver-') as directory:
        run = prepare_fixture(Path(directory))
        plan = run['options']['appearance_revision']['design_reconstruction_plan']
        _, operator = render_source_float(source_for_plan(plan), run)
        run.update(status='needs_visual_review',
                   render={'run_id': run['run_id'], 'revision': run['revision'], 'operator': operator})
        run['generation_design'] = {'design_brief': {'primary_effect_plan': {'mode': 'none'}}}
        failure = {'status': 'fail', 'cause': 'retouch_missing', 'correction_stage': 'native_retouch',
                   'reason': 'Source shoe reflection detail was lost.', 'action': 'Restore observed floor detail.',
                   'evidence': [{'basis': 'visible_pixels', 'panels': ['native_0', 'native_1'],
                                 'observation': 'The final reflection is softer than the source.'}]}
        passed = {'status': 'pass', 'reason': 'Visible paired evidence passes.', 'action': ''}
        master_hashes = {'auto_draft': 'draft-sha', 'render_master': 'master-sha',
                         'render_scene_master': 'scene-sha'}
        run['cosplay_post_review'] = {
            'status': 'reviewed', 'actual_api_evidence_valid': True,
            'run_binding': {'run_id': run['run_id'], 'revision': run['revision'],
                            'source_sha256': run['source']['sha256'],
                            'reference_sha256': run['reference']['sha256'],
                            'final_sha256': master_hashes['auto_draft'], 'render_artifacts': master_hashes},
            'checks': {key: {'final': copy.deepcopy(failure if key in {'native_detail', 'scene_integration'}
                                                   else passed)} for key in
                       ('identity_makeup', 'concept_hierarchy', 'effects_space', 'skin_materials',
                        'native_detail', 'scene_integration')},
            'local_observations': [{'pair_id': 'left_shoe_contact', 'region': 'left_shoe_contact',
                'scope': 'matched_source_final_native_pair_only', 'source_panel': 'native_0',
                'final_panel': 'native_1', 'native_box': [2, 10, 14, 19],
                'checks': {'identity_makeup': copy.deepcopy(passed), 'skin_materials': copy.deepcopy(passed),
                           'native_detail': copy.deepcopy(failure), 'scene_integration': copy.deepcopy(failure)}}],
            'native_sampling_coverage': {'unsupported_regions': [{'region': 'lower_clothing',
                                        'status': 'unknown', 'reason': 'no_known_source_support'}]},
        }
        observations = {'assets': {key: True for key in
            ('source', 'reference', 'draft', 'master', 'scene_master', 'alpha', 'manifest',
             'background', 'background_validity', 'reference_appearance_delta',
             'reference_appearance_colour_delta', 'reference_appearance_lightness_delta',
             'reference_appearance_control_receipt')}, 'review': {}}

        def report(value):
            before = copy.deepcopy(value)
            result = build_stage_report(value, observations)
            assert value == before, 'Read-only product reduction changed saved facts'
            state = result['whole_work']['native_receiver_detail_repair']
            assert not state['pixel_repair_authorized'] and not state['assets_rehashed']
            assert state['requires_actual_receiver_asset_validation'] and not state['visual_acceptance']
            assert not result['visual_approved']
            return result, state

        result, state = report(run)
        assert result['whole_work']['result_state'] == 'needs_repair'
        assert state['producer'] == 'native_receiver_detail'
        assert state['qualification_state'] == 'requires_producer_validation'
        assert set(state['candidate_finding_ids']) == {'native_detail', 'scene_integration'}
        assert state['qualification_pair_ids'] == ['left_shoe_contact']
        findings = result['whole_work']['open_findings']
        assert all(row['local_repair']['capability'] == 'native_receiver_detail' for row in findings
                   if row['criterion'] in {'native_detail', 'scene_integration'})
        assert any(row['criterion'] == 'sampling_coverage' and row['status'] == 'unknown' for row in findings)
        repair = next(row for row in result['whole_work']['recovery_actions']
                      if row['action_id'] == 'repair_final_locally')
        assert repair['supported'] and repair['QA_required'] and not repair['generation_required']
        assert '接收面原尺寸恢复资格检查' in repair['reason']
        assert '接收面原尺寸恢复' in format_stage_report(result)
        checks.append('current_shoe_failure_names_candidate_without_granting_pixel_or_visual_authority')

        for change in ('missing_pair', 'non_shoe', 'identity_unknown', 'materials_fail', 'uncited_pair',
                       'unknown_detail', 'other_cause', 'stale_review'):
            invalid = copy.deepcopy(run)
            review = invalid['cosplay_post_review']
            local = review['local_observations'][0]
            if change == 'missing_pair':
                review['local_observations'] = []
            elif change == 'non_shoe':
                local.update(pair_id='eyes_0', region='eyes')
            elif change == 'identity_unknown':
                local['checks']['identity_makeup']['status'] = 'unknown'
            elif change == 'materials_fail':
                local['checks']['skin_materials']['status'] = 'fail'
            elif change == 'uncited_pair':
                local['checks']['native_detail']['evidence'][0]['panels'] = ['native_0']
            elif change == 'unknown_detail':
                local['checks']['native_detail']['status'] = 'unknown'
                for criterion in ('native_detail', 'scene_integration'):
                    review['checks'][criterion]['final']['status'] = 'unknown'
            elif change == 'other_cause':
                local['checks']['native_detail']['cause'] = 'unresolved'
            else:
                review['run_binding']['source_sha256'] = 'different-source'
            result, state = report(invalid)
            assert state['producer'] is None and not state['candidate_from_saved_review'], change
            if change == 'unknown_detail':
                assert state['qualification_state'] == 'needs_scoped_evidence'
            if change == 'stale_review':
                assert state['qualification_state'] == 'needs_current_pixel_bound_diagnosis'
            checks.append(change + '_cannot_report_receiver_qualified')

        inherited = copy.deepcopy(run)
        inherited['run_id'] = 'copied-child'
        inherited['revision'] += 1
        inherited['render'].update(run_id=inherited['run_id'], revision=inherited['revision'])
        copy_receipt = {'schema': 'saved_render_copy.v1', 'copied_from_run_id': run['run_id'],
                        'copied_from_manifest_sha256': 'parent-manifest-sha',
                        'rendered_run_id': run['run_id'], 'render_executed': False,
                        'copied_master_sha256': master_hashes}
        inherited['saved_render_copy'] = copy.deepcopy(copy_receipt)
        inherited['render']['saved_render_copy'] = copy.deepcopy(copy_receipt)
        result, state = report(inherited)
        assert result['cosplay_post_diagnosis']['binding_state'] == 'inherited_same_pixels'
        assert state['producer'] is None and state['qualification_state'] == 'needs_current_pixel_bound_diagnosis'
        assert all(row['local_repair']['qualification_state'] == 'needs_current_pixel_bound_diagnosis'
                   for row in result['whole_work']['open_findings']
                   if row['criterion'] in {'native_detail', 'scene_integration'})
        checks.append('inherited_same_pixel_findings_explain_fresh_diagnosis_without_repair_authority')

        for status in ('receiver_detail_child_saved', 'receiver_evidence_unsupported', 'receiver_detail_rejected'):
            attempted = copy.deepcopy(run)
            attempted['automatic_receiver_detail_completion'] = {
                'schema': 'automatic_receiver_detail_completion.v1', 'status': status,
                'initial_run_id': run['run_id'], 'result_run_id': run['run_id'],
                'reason': 'joint_receiver_support_missing' if status == 'receiver_evidence_unsupported' else status,
                'qualification': {'permitted_pixels': 0 if status == 'receiver_evidence_unsupported' else 40}}
            result, state = report(attempted)
            assert state['last_outcome']['status'] == status and state['last_outcome']['applies_to_run']
            assert state['last_outcome']['reason'] in format_stage_report(result)
            assert result['whole_work']['result_state'] == 'needs_repair'
            checks.append(status + '_shows_actual_outcome_and_keeps_failure_open')

        foreign = copy.deepcopy(attempted)
        foreign['automatic_receiver_detail_completion'].update(initial_run_id='other-parent', result_run_id='other-child')
        _, state = report(foreign)
        assert not state['last_outcome']['applies_to_run'] and '其它版本' in state['last_outcome_detail']
        checks.append('foreign_outcome_does_not_claim_current_execution')

        early_rejection = copy.deepcopy(run)
        early_rejection['system_post_completion'] = {'stages': [{'id': 'final_local_repair', 'outcomes': [{
            'producer': 'native_receiver_detail', 'status': 'receiver_evidence_unsupported',
            'reason': 'Actual native evidence cannot be reopened.',
            'input_run_id': run['run_id'], 'result_run_id': run['run_id']}]}]}
        result, state = report(early_rejection)
        assert state['last_outcome']['status'] == 'receiver_evidence_unsupported'
        assert state['last_outcome_source'] == 'system_post_completion.final_local_repair.outcomes'
        assert state['last_outcome']['reason'] in format_stage_report(result)
        checks.append('early_producer_rejection_visible_without_attempt_receipt')

        no_repeat = copy.deepcopy(attempted)
        no_repeat['system_post_completion'] = copy.deepcopy(early_rejection['system_post_completion'])
        no_repeat['system_post_completion']['stages'][0]['outcomes'][0].update(
            status='already_attempted', reason='persistent_local_attempt_marker')
        _, state = report(no_repeat)
        assert state['last_outcome']['status'] == 'already_attempted'
        assert '不自动重复' in state['last_outcome_detail']
        checks.append('latest_coordinator_no_repeat_outcome_precedes_older_attempt_status')

        prepared = copy.deepcopy(run)
        prepared_plan = prepared['options']['appearance_revision']['design_reconstruction_plan']
        prepared_plan.setdefault('provenance', {})['native_receiver_detail'] = {
            'state': 'prepared', 'receipt_sha256': 'receipt-sha', 'visual_acceptance': False}
        background = next(row for row in prepared_plan['layers'] if row['kind'] == 'background_design')
        background['assets']['native_receiver_detail'] = {'path': 'receiver-detail.json', 'sha256': 'receipt-sha'}
        result, state = report(prepared)
        assert state['prepared_asset_declared'] and result['whole_work']['result_state'] == 'needs_repair'
        assert not state['visual_acceptance']
        background['assets']['native_receiver_detail']['sha256'] = 'other-receipt'
        _, state = report(prepared)
        assert not state['prepared_asset_declared']
        checks.append('frozen_receipt_declaration_is_not_execution_or_quality_acceptance')

    print(json.dumps({'ok': True, 'checks': checks, 'model_calls': 0, 'qa_requests': 0,
                      'image_requests': 0, 'network_requests': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
