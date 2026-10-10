"""Offline saved-diagnosis, no-repeat, cancellation and child ownership checks."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import automatic_receiver_detail_completion as completion
import cosplay_post_review as visual
import single_source_workflow as workflow
from cosplay_native_review import prepare_cosplay_native_review
from cosplay_native_review_smoke import make_native_fixture
from cosplay_post_review_smoke import multi_fake_transport, verdict


def fixture(root):
    root = Path(root).resolve(); directory = root/'outputs/runs/native-offline-fixture'
    run = make_native_fixture(directory/'inputs', native_size=(400, 500))
    run.update(root=str(root), kind='single_input_run', sample='receiver-fixture',
               input_mode='single_source', status='needs_visual_review', auto_draft=True)
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    semantics = plan['layers'][0]['assets']['source_shoe_semantics']
    with np.load(semantics['path'], allow_pickle=False) as resource:
        labels = resource['labels'].copy()
    labels[120:132, 38:51] = 9; labels[120:132, 58:71] = 18
    np.savez_compressed(semantics['path'], labels=labels)
    semantics['sha256'] = completion.sha(semantics['path'])
    plan['layers'][0]['provenance']['asset_bindings']['source_shoe_semantics_sha256'] = semantics['sha256']
    for identity in (plan['source_baseline']['foreground_bundle_identity'],
                     plan['provenance']['source_foreground']['identity']):
        identity['review_semantics_sha256'] = semantics['sha256']
    plan['layers'].append({'kind': 'background_design', 'assets': {
        'background': copy.deepcopy(plan['source_baseline']['assets']['old_environment']),
        'validity': copy.deepcopy(plan['source_baseline']['assets']['foreground_alpha'])}})
    for index, layer in enumerate(plan['layers']):
        layer.update(id='fixture_'+str(index), enabled=True, strength=1.)
    plan['mask_roles'] = {'foreground_alpha': copy.deepcopy(plan['source_baseline']['assets']['foreground_alpha'])}
    prepared = prepare_cosplay_native_review(run, directory/'native-selection', comprehensive=True)
    assert prepared['receipt']['status'] == 'ready'
    design = {'selected_style_id': 'offline', 'design_brief': {'summary': 'Synthetic receiver detail',
              'primary_effect_plan': {'mode': 'none', 'description': 'No effects'}}}
    run['generation_design'] = design
    shoe_pairs = {row['pair_id'] for row in prepared['receipt']['regions']
                  if row['region'] in ('left_shoe_contact', 'right_shoe_contact')}
    assert shoe_pairs
    base_transport = multi_fake_transport()
    def transport(*args, **kwargs):
        payload, provenance = base_transport(*args, **kwargs)
        for row in payload.get('pairs') or []:
            if row['pair_id'] in shoe_pairs:
                panels = row['checks']['native_detail']['evidence'][0]['panels']
                row['checks']['native_detail'] = verdict(panels[-1], status='fail',
                    cause='retouch_missing', stage='native_retouch',
                    action='Restore observed native floor detail beneath this shoe.', panels=panels)
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
    completion.write(directory/'cosplay-post-review/review.json', review)
    completion.write(directory/'manifest.json', run)
    completion.write(root/'project.json', {'state': {'selected': run['run_id']}})
    return run


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='receiver-completion-offline-') as folder:
        root = Path(folder).resolve(); run = fixture(root)
        diagnosis = completion._diagnosis(root, run)
        assert len(diagnosis['reviewed_windows']) == 2
        checks.append('actual_saved_comprehensive_native_pair_evidence_replayed')
        for name, mutation in (
            ('revision', lambda value: value.update(revision=99)),
            ('run', lambda value: value.update(run_id='other')),
            ('native_master', lambda value: value['artifacts'].update(render_scene_master=value['artifacts']['render_master']))):
            changed = copy.deepcopy(run); mutation(changed)
            result = completion.complete_receiver_detail_once(root, changed, SimpleNamespace())
            assert result['status'] == 'receiver_evidence_unsupported', name
            checks.append('stale_'+name+'_rejected_before_worker')
        unknown = copy.deepcopy(run)
        for row in unknown['cosplay_post_review']['local_observations']:
            row['checks']['native_detail']['status'] = 'unknown'
        assert completion.complete_receiver_detail_once(root, unknown, SimpleNamespace())['handled'] is False
        checks.append('unknown_diagnosis_does_not_enter_worker')
        path = root/'outputs/runs'/run['run_id']/'cosplay-post-review/review.json'
        original = path.read_bytes(); forged = copy.deepcopy(run)
        for row in forged['cosplay_post_review']['local_observations']:
            if row['region'] == 'left_shoe_contact': row['native_box'][0] += 1
        completion.write(path, forged['cosplay_post_review'])
        assert completion.complete_receiver_detail_once(root, forged, SimpleNamespace())['status'] == 'receiver_evidence_unsupported'
        path.write_bytes(original)
        checks.append('altered_aggregate_window_rejected_by_actual_board_replay')

        manifest = root/'outputs/runs'/run['run_id']/'manifest.json'
        manifest_bytes = manifest.read_bytes()
        for name, mutation in (
                ('unsaved_plan', lambda value: value['options']['appearance_revision']['design_reconstruction_plan'].update(enabled=False)),
                ('different_valid_plate', lambda value: value['options']['appearance_revision']['design_reconstruction_plan']['layers'][1]['assets'].update(background=copy.deepcopy(value['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']['assets']['foreground_premultiplied']))),
                ('executed_contact_asset', lambda value: value['options']['appearance_revision']['design_reconstruction_plan']['layers'][0]['assets']['source_shoe_semantics'].update(sha256='0'*64)),
                ('executed_alpha', lambda value: value['options']['appearance_revision']['design_reconstruction_plan']['mask_roles']['foreground_alpha'].update(sha256='0'*64)),
                ('reviewed_source_donor', lambda value: value['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']['assets']['old_environment'].update(sha256='0'*64))):
            changed = copy.deepcopy(run); mutation(changed)
            if name != 'unsaved_plan': completion.write(manifest, changed)
            assert completion.complete_receiver_detail_once(root, changed, SimpleNamespace())['status'] == 'receiver_evidence_unsupported', name
            manifest.write_bytes(manifest_bytes)
            checks.append(name+'_rejected_by_saved_execution_binding')
        metadata_path = Path(run['artifacts']['metadata']); metadata_bytes = metadata_path.read_bytes()
        metadata = completion.read(metadata_path); metadata['revision'] += 1; completion.write(metadata_path, metadata)
        assert completion.complete_receiver_detail_once(root, run, SimpleNamespace())['status'] == 'receiver_evidence_unsupported'
        metadata_path.write_bytes(metadata_bytes)
        checks.append('changed_render_receipt_rejected')

        protected = {str(path): completion.sha(path) for path in (
            root/'project.json', root/'outputs/runs'/run['run_id']/'manifest.json',
            Path(run['source']['path']), Path(run['reference']['path']),
            Path(run['artifacts']['metadata']),
            *(Path(run['artifacts'][name]) for name in completion.ARTIFACTS))}
        prepared_plan = copy.deepcopy(run['options']['appearance_revision']['design_reconstruction_plan'])
        def prepared(*args, **kwargs):
            return {'changed': True, 'design_reconstruction_plan': prepared_plan,
                    'qualification': {'qualified': True, 'permitted_pixels': 123},
                    'receipt': {'state': 'prepared', 'reason': 'synthetic_nonzero_receiver',
                        'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
                        'diagnosis': diagnosis['evidence'], 'checks': {'changed_pixels': 123}}}
        def child_revision(root, run_id, *, snapshot, design_plan_override):
            assert snapshot is False and run_id == run['run_id']
            child = copy.deepcopy(run); child.update(run_id='receiver-child', parent_run_id=run_id)
            child['options']['appearance_revision']['design_reconstruction_plan'] = copy.deepcopy(design_plan_override)
            child.pop('auto_draft', None); child['artifacts'] = {}
            path = root/'outputs/runs/receiver-child/manifest.json'
            return path, child
        def render_child(root, run_id):
            path = root/'outputs/runs'/run_id/'manifest.json'; child = completion.read(path)
            assert child['options']['preserve_project_selection'] is True
            assert child['cosplay_post_review']['diagnostic_gate'] == 'unknown'
            for name in completion.ARTIFACTS:
                output = path.parent/(name+'.synthetic'); output.write_bytes(('owned-'+name).encode())
                child['artifacts'][name] = str(output)
            child['continuation_edit']['render_state'] = 'saved'; child['auto_draft'] = True
            return child
        with patch.object(completion, 'prepare_receiver_detail', side_effect=prepared) as worker, \
                patch.object(workflow, '_source_revision', child_revision), \
                patch.object(workflow, 'adjust_source_run', render_child):
            result = completion.complete_receiver_detail_once(root, run, SimpleNamespace())
            assert result['status'] == 'receiver_detail_child_saved', result
            assert worker.call_count == 1
            repeated = completion.complete_receiver_detail_once(root, run, SimpleNamespace())
            assert repeated['status'] == 'already_attempted' and worker.call_count == 1
        assert result['run']['parent_run_id'] == run['run_id']
        assert result['run'][completion.RECORD_KEY]['model_inferences'] == 0
        assert not result['run']['cosplay_post_review']['actual_api_evidence_valid']
        for file, digest in protected.items(): assert completion.sha(file) == digest
        checks.extend(['owned_child_saved_without_inherited_final_acceptance',
                       'parent_source_reference_masters_project_exact', 'durable_attempt_blocks_duplicate_worker'])

        for name, outcome, expected in (
                ('unsupported', {'changed': False, 'qualification': {}, 'receipt': {'reason': 'no_joint_floor'}}, 'receiver_evidence_unsupported'),
                ('zero_change', {'changed': False, 'qualification': {'qualified': True}, 'receipt': {'reason': 'no_actual_change'}}, 'receiver_detail_rejected'),
                ('false_changed', {'changed': True, 'qualification': {}, 'receipt': {'state': 'prepared'}}, 'receiver_detail_rejected')):
            case_root = root/name; case = fixture(case_root)
            with patch.object(completion, 'prepare_receiver_detail', return_value=outcome), \
                    patch.object(workflow, '_source_revision', side_effect=AssertionError('No child allowed')):
                failed = completion.complete_receiver_detail_once(case_root, case, SimpleNamespace())
                assert failed['status'] == expected, failed
                assert completion.complete_receiver_detail_once(case_root, case, SimpleNamespace())['status'] == 'already_attempted'
            checks.append(name+'_retains_parent_without_child_or_retry')
        case_root = root/'cancelled'; case = fixture(case_root)
        signal = case_root/'cancel.requested'; signal.write_bytes(b'cancel')
        with patch.object(completion, 'prepare_receiver_detail', side_effect=AssertionError('Cancelled before worker')):
            cancelled = completion.complete_receiver_detail_once(case_root, case, SimpleNamespace(cancel_request_path=signal))
        assert cancelled['status'] == 'receiver_detail_rejected' and cancelled['run'][completion.RECORD_KEY]['cancelled']
        signal.unlink()
        assert completion.complete_receiver_detail_once(case_root, case, SimpleNamespace())['status'] == 'already_attempted'
        checks.append('cancel_before_worker_is_durable_and_preserves_parent')
        class PendingProcess:
            returncode = None
            terminated = False
            def poll(self): return self.returncode
            def terminate(self): self.terminated = True
            def wait(self, timeout): self.returncode = -1
            def kill(self): self.returncode = -1
        process = PendingProcess(); cancellation = iter((False, True))
        import current_model_registry
        import sys
        with patch.object(current_model_registry, 'load_registry', return_value={'python': sys.executable}), \
                patch.object(completion.subprocess, 'Popen', return_value=process):
            try: completion.prepare_receiver_detail(root, run, root/'cpu-cancel', cancel=lambda: next(cancellation))
            except InterruptedError: pass
            else: raise AssertionError('Active CPU worker cancellation was ignored')
        assert process.terminated
        checks.append('active_cpu_worker_terminated_on_cancel')
    print(json.dumps({'status': 'passed', 'cases': checks, 'API_requests': 0,
                      'image_generation_submissions': 0, 'model_inference_requests': 0}))


if __name__ == '__main__': main()
