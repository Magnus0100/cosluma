"""Offline orchestration contracts; controlled worker, actual board replay, no API."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import automatic_subject_detail_completion as completion
import cosplay_post_review as visual
import single_source_workflow as workflow
from cosplay_native_review import prepare_cosplay_native_review
from cosplay_native_review_smoke import make_native_fixture
from cosplay_post_review_smoke import multi_fake_transport, verdict
from cosplay_repair_plan import plan_final_repairs
from pipeline_stage_report import native_subject_repair_state
from generation_design import DESIGN_VERSION
import system_post_completion as system


def fixture(root, *, saved_run_binding=True):
    root = Path(root).resolve(); directory = root/'outputs/runs/native-offline-fixture'
    run = make_native_fixture(directory/'inputs', native_size=(400, 500))
    run.update(root=str(root), kind='single_input_run', status='needs_visual_review')
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    for index, layer in enumerate(plan['layers']):
        layer.update(id='fixture_'+str(index), enabled=True, strength=1.)
    plan['mask_roles'] = {'foreground_alpha': copy.deepcopy(plan['source_baseline']['assets']['foreground_alpha'])}
    prepared = prepare_cosplay_native_review(run, directory/'native-selection', comprehensive=True)
    assert prepared['receipt']['status'] == 'ready'
    design = {'selected_style_id': 'offline', 'design_brief': {'summary': 'Synthetic subject coordinator',
              'primary_effect_plan': {'mode': 'none', 'description': 'No effects'}}}
    run['generation_design'] = design
    hand_pairs = {row['pair_id'] for row in prepared['receipt']['regions']
                  if row['region'] == 'left_hand'}
    assert hand_pairs
    base_transport = multi_fake_transport()
    def transport(*args, **kwargs):
        payload, provenance = base_transport(*args, **kwargs)
        for row in payload.get('pairs') or []:
            if row['pair_id'] in hand_pairs:
                panels = row['checks']['native_detail']['evidence'][0]['panels']
                row['checks']['native_detail'] = verdict(panels[-1], status='fail',
                    cause='retouch_missing', stage='native_retouch',
                    action='Restore observed original hand texture without changing colour.', panels=panels)
        return payload, provenance
    with patch.object(visual.vision_transport, 'request_vision_json', transport):
        review = visual.review_cosplay_post(prepared['source_path'], Path(run['reference']['path']),
            directory/'cosplay-post-review', design, final=prepared['final_path'],
            native_crops=prepared['native_crops'], native_selection=prepared['receipt'])
    assert review['status'] == 'reviewed', review.get('error')
    for name in ('render_master', 'render_scene_master'):
        path = directory/'inputs'/(name+'.tif'); path.write_bytes(('synthetic-'+name).encode())
        run['artifacts'][name] = str(path)
    review['run_binding'] = {**prepared['receipt']['run_binding'],
        'final_sha256': review['binding']['images']['final']['sha256'],
        'render_artifacts': {name: completion.sha(run['artifacts'][name]) for name in completion.ARTIFACTS}}
    run['cosplay_post_review'] = review
    metadata = {'run_id': run['run_id'], 'revision': run['revision'], 'operator': {
        'design_reconstruction_plan': {'working_domain': 'display_linear_sRGB',
            'foreground_alpha_sha256': plan['mask_roles']['foreground_alpha']['sha256'],
            'layers': copy.deepcopy(plan['layers'])}}}
    metadata_path = directory/'inputs/metadata.json'; completion.write(metadata_path, metadata)
    run['artifacts']['metadata'] = str(metadata_path); run['render'] = metadata
    if saved_run_binding:
        completion.write(directory/'cosplay-post-review/review.json', review)
    completion.write(directory/'manifest.json', run)
    completion.write(root/'project.json', {'selected': run['run_id']})
    return run


def plate(root, run):
    # Source donor/owner math has separate producer tests. Here that boundary
    # is controlled; actual aggregate, boards and master binding are replayed.
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    return {'metadata': {'path': run['artifacts']['metadata'], 'sha256': completion.sha(run['artifacts']['metadata'])},
            'evidence_assets': list(plan['source_baseline']['assets'].values()),
            'plate_matches_saved_render': True, 'source_baseline_matches_native_QA': True}


def preserved(root, run):
    paths = [root/'project.json', root/'outputs/runs'/run['run_id']/'manifest.json',
             root/'outputs/runs'/run['run_id']/'cosplay-post-review/review.json',
             Path(run['source']['path']), Path(run['reference']['path'])]
    paths += [Path(run['artifacts'][name]) for name in (*completion.ARTIFACTS, 'metadata')]
    paths += [Path(row['path']) for row in plate(root, run)['evidence_assets']]
    return {str(path): completion.sha(path) for path in paths}


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='subject-coordinator-offline-') as folder, \
            patch('native_subject_detail.validate_diagnosed_plate', side_effect=plate):
        root = Path(folder).resolve(); run = fixture(root)
        diagnosis = completion._diagnosis(root, run)
        assert diagnosis['reviewed_windows'] and all(row['region'] == 'left_hand' for row in diagnosis['reviewed_windows'])
        plan = plan_final_repairs(run, current_review=True)
        candidate = next(row for row in plan['tasks'] if row['criterion'] == 'native_detail')
        assert candidate['producer'] == 'native_subject_detail' and candidate['qualification_pair_ids']
        assert not any(row['generation_authorized_by_this_plan'] for row in plan['tasks'])
        state = native_subject_repair_state(run, {'current_render_binding_matches': True})
        assert state['candidate_from_saved_review'] and not state['pixel_repair_authorized'] and not state['visual_acceptance']
        checks.append('actual_paired_hand_boards_replayed_and_classified_without_visual_or_generation_authority')
        for name, change in (
                ('revision', lambda value: value.update(revision=99)),
                ('run', lambda value: value.update(run_id='other')),
                ('master', lambda value: value['artifacts'].update(render_scene_master=value['artifacts']['render_master']))):
            stale = copy.deepcopy(run); change(stale)
            with patch.object(completion, 'prepare_subject_detail', side_effect=AssertionError('stale diagnosis cannot run')):
                result = completion.complete_subject_detail_once(root, stale, SimpleNamespace())
            assert result['status'] == 'subject_evidence_unsupported', name
            checks.append('stale_'+name+'_rejected')
        unknown = copy.deepcopy(run)
        for row in unknown['cosplay_post_review']['local_observations']:
            row['checks']['native_detail']['status'] = 'unknown'
        assert not completion.complete_subject_detail_once(root, unknown, SimpleNamespace())['handled']
        checks.append('unknown_cannot_enter_worker')
        path = root/'outputs/runs'/run['run_id']/'cosplay-post-review/review.json'
        original = path.read_bytes(); forged = copy.deepcopy(run)
        for row in forged['cosplay_post_review']['local_observations']:
            if row['region'] == 'left_hand': row['native_box'][0] += 1
        completion.write(path, forged['cosplay_post_review'])
        assert completion.complete_subject_detail_once(root, forged, SimpleNamespace())['status'] == 'subject_evidence_unsupported'
        path.write_bytes(original)
        checks.append('aggregate_native_box_forgery_rejected_by_board_replay')
        # An independent consumer binds the same endpoint to the untouched
        # original receipt at its original path. It does not copy QA assets.
        producer_root = root/'external-producer'; producer = fixture(producer_root, saved_run_binding=False)
        consumer_root = root/'external-consumer'; consumer_root.mkdir()
        consumer = copy.deepcopy(producer); consumer['root'] = str(consumer_root)
        external_path = producer_root/'outputs/runs'/producer['run_id']/'cosplay-post-review/review.json'
        external_bytes = external_path.read_bytes()
        assert 'run_binding' not in completion.read(external_path)
        consumer[completion.EXTERNAL_REVIEW_KEY] = {'schema': completion.EXTERNAL_REVIEW_SCHEMA,
            'path': str(external_path), 'sha256': completion.sha(external_path)}
        completion.write(consumer_root/'outputs/runs'/consumer['run_id']/'manifest.json', consumer)
        completion.write(consumer_root/'project.json', {'selected': consumer['run_id']})
        external_diagnosis = completion._diagnosis(consumer_root, consumer)
        assert external_diagnosis['external_review_consumed_read_only']
        assert external_diagnosis['evidence']['path'] == str(external_path.resolve())
        assert not (consumer_root/'outputs/runs'/consumer['run_id']/'cosplay-post-review').exists()
        assert external_path.read_bytes() == external_bytes
        checks.append('external_original_receipt_strictly_replayed_with_only_added_consumer_run_binding')
        owned_external = copy.deepcopy(run)
        owned_path = root/'outputs/runs'/run['run_id']/'cosplay-post-review/review.json'
        owned_external[completion.EXTERNAL_REVIEW_KEY] = {'schema': completion.EXTERNAL_REVIEW_SCHEMA,
            'path': str(owned_path), 'sha256': completion.sha(owned_path)}
        assert completion._diagnosis(root, owned_external)['external_review_consumed_read_only']
        conflicting_binding = copy.deepcopy(owned_external)
        conflicting_binding['cosplay_post_review']['run_binding']['invented_field'] = True
        assert completion.complete_subject_detail_once(root, conflicting_binding, SimpleNamespace())['status'] == 'subject_evidence_unsupported'
        checks.append('external_existing_run_binding_accepted_only_when_entire_saved_binding_matches')
        for name, mutation in (
                ('hash', lambda value: value[completion.EXTERNAL_REVIEW_KEY].update(sha256='0'*64)),
                ('schema', lambda value: value[completion.EXTERNAL_REVIEW_KEY].update(schema='other')),
                ('verdict', lambda value: value['cosplay_post_review']['checks']['native_detail']['final'].update(reason='forged')),
                ('native_identity', lambda value: value['cosplay_post_review']['native_selection']['run_binding'].update(run_id='foreign')),
                ('final_path', lambda value: value['cosplay_post_review']['binding']['images']['final'].update(path=str(external_path))),
                ('final_size', lambda value: value['cosplay_post_review']['binding']['images']['final'].update(oriented_size=[1, 1])),
                ('masters', lambda value: value['artifacts'].update(render_scene_master=value['artifacts']['render_master']))):
            forged = copy.deepcopy(consumer); mutation(forged)
            with patch.object(completion, 'prepare_subject_detail', side_effect=AssertionError('Unsupported external QA cannot run')):
                result = completion.complete_subject_detail_once(consumer_root, forged, SimpleNamespace())
            assert result['status'] == 'subject_evidence_unsupported', name
            assert external_path.read_bytes() == external_bytes
            checks.append('external_'+name+'_tamper_rejected_without_touching_original_QA')
        # Hashing a changed aggregate cannot legalize invented QA semantics.
        forged = copy.deepcopy(consumer)
        changed_raw = completion.read(external_path)
        changed_raw['checks']['native_detail']['final']['reason'] = 'invented aggregate'
        completion.write(external_path, changed_raw)
        forged['cosplay_post_review'] = {**copy.deepcopy(changed_raw),
            'run_binding': copy.deepcopy(consumer['cosplay_post_review']['run_binding'])}
        forged[completion.EXTERNAL_REVIEW_KEY]['sha256'] = completion.sha(external_path)
        try:
            result = completion.complete_subject_detail_once(consumer_root, forged, SimpleNamespace())
            assert result['status'] == 'subject_evidence_unsupported'
        finally:
            external_path.write_bytes(external_bytes)
        checks.append('external_rehashed_forged_aggregate_rejected_by_actual_board_replay')
        with patch.object(completion, 'prepare_subject_detail', return_value={
                'changed': False, 'qualification': {'qualified': True},
                'receipt': {'reason': 'source_detail_already_preserved'}}), \
                patch.object(workflow, '_source_revision', side_effect=AssertionError('No loss cannot save child')):
            rejected = completion.complete_subject_detail_once(consumer_root, consumer, SimpleNamespace())
        assert rejected['status'] == 'subject_detail_rejected'
        assert rejected['run'][completion.RECORD_KEY]['diagnosis']['evidence'] == external_diagnosis['evidence']
        assert external_path.read_bytes() == external_bytes
        assert completion.read(producer_root/'outputs/runs'/producer['run_id']/'manifest.json') == producer
        checks.append('external_no_loss_rejection_records_original_QA_and_preserves_producer_manifest')
        protected = preserved(root, run)
        def worker(*args, **kwargs):
            return {'changed': True, 'qualification': {'qualified': True, 'permitted_pixels': 90},
                    'design_reconstruction_plan': copy.deepcopy(run['options']['appearance_revision']['design_reconstruction_plan']),
                    'receipt': {'state': 'prepared', 'reason': 'controlled_nonzero_subject_texture',
                        'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
                        'diagnosis': diagnosis['evidence'], 'checks': {'changed_pixels': 90}}}
        def revision(root, run_id, *, snapshot, design_plan_override):
            assert snapshot is False and run_id == run['run_id']
            child = copy.deepcopy(run); child.update(run_id='subject-child', parent_run_id=run_id)
            child['options']['appearance_revision']['design_reconstruction_plan'] = copy.deepcopy(design_plan_override)
            child['artifacts'] = {}
            return root/'outputs/runs/subject-child/manifest.json', child
        def render(root, run_id):
            path = root/'outputs/runs'/run_id/'manifest.json'; child = completion.read(path)
            assert child['options']['preserve_project_selection'] is True
            assert child['cosplay_post_review']['status'] == 'unavailable'
            for name in completion.ARTIFACTS:
                target = path.parent/(name+'.synthetic'); target.write_bytes(('owned-'+name).encode())
                child['artifacts'][name] = str(target)
            child['continuation_edit']['render_state'] = 'saved'
            return child
        with patch.object(completion, 'prepare_subject_detail', side_effect=worker) as cpu, \
                patch.object(workflow, '_source_revision', revision), patch.object(workflow, 'adjust_source_run', render):
            result = completion.complete_subject_detail_once(root, run, SimpleNamespace())
            assert result['status'] == 'subject_detail_child_saved', result
            assert completion.complete_subject_detail_once(root, run, SimpleNamespace())['status'] == 'already_attempted'
            assert cpu.call_count == 1
        assert result['run']['parent_run_id'] == run['run_id']
        record = result['run'][completion.RECORD_KEY]
        assert record['original_draft_retained'] and record['model_inferences'] == record['image_generation_submissions'] == 0
        assert not result['run']['cosplay_post_review']['actual_api_evidence_valid']
        assert protected == {name: completion.sha(name) for name in protected}
        checks.extend(['owned_child_requires_fresh_QA', 'all_parent_assets_and_selection_exact', 'persistent_single_attempt'])
        for name, output, expected in (
                ('unsupported', {'changed': False, 'qualification': {}, 'receipt': {'reason': 'owner_unknown'}}, 'subject_evidence_unsupported'),
                ('no_loss', {'changed': False, 'qualification': {'qualified': True}, 'receipt': {'reason': 'source_detail_already_preserved'}}, 'subject_detail_rejected'),
                ('forbidden_request', {'changed': False, 'qualification': {'qualified': True},
                    'receipt': {'model_inferences': 1}}, 'subject_detail_rejected'),
                ('false_change', {'changed': True, 'qualification': {}, 'receipt': {'state': 'prepared'}}, 'subject_detail_rejected')):
            case_root = root/name; case = fixture(case_root); before = preserved(case_root, case)
            with patch.object(completion, 'prepare_subject_detail', return_value=output), \
                    patch.object(workflow, '_source_revision', side_effect=AssertionError('No child for unsupported/no loss')):
                outcome = completion.complete_subject_detail_once(case_root, case, SimpleNamespace())
                assert outcome['status'] == expected, outcome
                assert completion.complete_subject_detail_once(case_root, case, SimpleNamespace())['status'] == 'already_attempted'
            assert before == {name: completion.sha(name) for name in before}
            assert outcome['run']['cosplay_post_review']['checks']['native_detail']['final']['status'] == 'fail'
            state = native_subject_repair_state(outcome['run'], {'current_render_binding_matches': True})
            assert state['last_outcome']['status'] == expected and not state['visual_acceptance']
            checks.append(name+'_retains_failure_and_parent_without_child_or_repeat')
        case_root = root/'cancelled'; case = fixture(case_root); before = preserved(case_root, case)
        signal = case_root/'cancel.requested'; signal.write_bytes(b'cancel')
        with patch.object(completion, 'prepare_subject_detail', side_effect=AssertionError('cancelled before worker')):
            outcome = completion.complete_subject_detail_once(case_root, case, SimpleNamespace(cancel_request_path=signal))
        assert outcome['status'] == 'subject_detail_rejected' and outcome['run'][completion.RECORD_KEY]['cancelled']
        signal.unlink()
        assert completion.complete_subject_detail_once(case_root, case, SimpleNamespace())['status'] == 'already_attempted'
        assert before == {name: completion.sha(name) for name in before}
        checks.append('pre_worker_cancellation_durable_and_parent_exact')
        class PendingProcess:
            returncode = None
            terminated = False
            def poll(self): return self.returncode
            def terminate(self): self.terminated = True
            def wait(self, timeout): self.returncode = -1
            def kill(self): self.returncode = -1
        process = PendingProcess(); cancellation = iter((False, True))
        with patch('current_model_registry.load_registry', return_value={'python': sys.executable}), \
                patch.object(completion.subprocess, 'Popen', return_value=process):
            try: completion.prepare_subject_detail(root, run, root/'cpu-cancel', cancel=lambda: next(cancellation))
            except InterruptedError: pass
            else: raise AssertionError('Running CPU cancellation ignored')
        assert process.terminated
        checks.append('running_CPU_worker_terminated_on_cancel')
        case_root = root/'system-order'; case = fixture(case_root)
        case['generation_design'].update(schema_version=DESIGN_VERSION, status='ready')
        review = case['cosplay_post_review']
        shoe_failure = {'status': 'fail', 'cause': 'retouch_missing', 'correction_stage': 'native_retouch',
                        'evidence': [{'basis': 'visible_pixels', 'panels': ['shoe-source', 'shoe-final']}]}
        review['checks']['scene_integration']['final'] = copy.deepcopy(shoe_failure)
        review['local_observations'].append({'pair_id': 'shoe-fixture', 'region': 'left_shoe_contact',
            'scope': 'matched_source_final_native_pair_only', 'source_panel': 'shoe-source', 'final_panel': 'shoe-final',
            'native_box': [0, 0, 20, 20], 'checks': {'native_detail': copy.deepcopy(shoe_failure),
                'scene_integration': copy.deepcopy(shoe_failure), 'identity_makeup': {'status': 'pass'},
                'skin_materials': {'status': 'pass'}}})
        completion.write(case_root/'outputs/runs'/case['run_id']/'manifest.json', case)
        order = []
        def reject_subject(_root, current, args, **kwargs):
            assert kwargs == {'bounded': True}; order.append('subject')
            return {'handled': True, 'run': current, 'status': 'subject_detail_rejected',
                    'reason': 'source detail already preserved'}
        def reject_receiver(_root, current, args, **kwargs):
            assert kwargs == {'bounded': True}; order.append('receiver')
            return {'handled': True, 'run': current, 'status': 'receiver_evidence_unsupported'}
        settings = SimpleNamespace(generate=False, complete_post=True, final_repair_only=True,
            cosplay_correction_budget=0, preserve_effective_prompt=False, cancel_request_path=None)
        project_sha = completion.sha(case_root/'project.json')
        with patch.object(completion, 'complete_subject_detail_once', side_effect=reject_subject), \
                patch('automatic_receiver_detail_completion.complete_receiver_detail_once', side_effect=reject_receiver), \
                patch.object(workflow, 'diagnose_cosplay_post', side_effect=AssertionError('Current pixels must reuse current QA')), \
                patch('cosplay_post_completion.continue_cosplay_post', side_effect=AssertionError('No image requests')), \
                patch('automatic_foreground_effects.complete_effects_once', side_effect=AssertionError('No effect requests')):
            retained = system.complete_once(case_root, case, settings)
            assert order == ['subject', 'receiver'], retained.get('system_post_completion')
            assert system.complete_once(case_root, case, settings)['run_id'] == case['run_id']
        receipt = retained['system_post_completion']
        assert receipt['local_repair_rounds'] == receipt['local_repair_round_budget'] == 1
        assert receipt['status'] == 'completed_with_findings' and receipt['visual_state'] == 'findings_open'
        assert 'native_detail' in receipt['failed_checks'] and 'scene_integration' in receipt['failed_checks']
        assert retained['generation_budget']['reference_correction_budget'] == 0
        assert retained['generation_budget']['independent_effect_image_budget'] == 0
        assert completion.sha(case_root/'project.json') == project_sha
        checks.append('subject_before_receiver_single_round_no_extra_images_no_fail_to_pass_or_repeat')
    print(json.dumps({'status': 'passed', 'cases': checks, 'API_requests': 0,
                      'image_generation_submissions': 0, 'model_inference_requests': 0}))


if __name__ == '__main__': main()
