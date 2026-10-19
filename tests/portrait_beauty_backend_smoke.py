"""CPU job lifecycle and real frozen beauty relocation; model sampling is mocked."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import types
from unittest.mock import patch

import numpy as np
from PIL import Image

from backend_server import Backend
from display_transform import source_transform_for_run, source_transform_for_precision
from native_source_baseline import prepare_source_only
from portable_project import export_project, import_project, validate_portrait_beauty
from portrait_beauty_layer import freeze_prepared, apply_for_run, sha256
from raw_support import open_linear, RGB_DECODE_VERSION
from revision_state import appearance_revision_patch, merge_appearance_revision
from session_draft import digest, editable_payload, binding_for_run, SessionDraftStore, validate_assets


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def project(root):
    root.mkdir(parents=True)
    source = root/'source.png'
    Image.new('RGB', (32, 24), (90, 110, 125)).save(source)
    run = {'kind': 'single_input_run', 'input_mode': 'single_source', 'run_id': 'beauty-parent',
        'sample': 'beauty-fixture', 'revision': 1, 'root': str(root), 'status': 'needs_visual_review',
        'source': {'path': str(source), 'sha256': sha256(source), 'oriented_size': [32, 24]}, 'reference': None,
        'options': {'render_mode': 'native_source_design', 'appearance_revision': {'params': {}}, 'manual_layers': {}}}
    values, precision = open_linear(source)
    values, plan = prepare_source_only(values, run, precision)
    run['options']['source_only_plan'] = plan
    alpha = root/'outputs/runs/beauty-parent/local-source/alpha.png'
    alpha.parent.mkdir(parents=True)
    Image.new('L', (32, 24), 255).save(alpha)
    run['options']['source_only_geometry'] = {'schema': 'source_only_geometry.v1',
        'source_sha256': run['source']['sha256'], 'native_size': [32, 24], 'backend_requested': 'auto',
        'path': str(alpha), 'sha256': sha256(alpha),
        'provenance': {'method': 'synthetic_CPU_geometry_fixture_not_model_inference'}}
    run['auto_draft'] = {'metadata': {'kind': 'local_source_render'}}
    write(root/'outputs/runs/beauty-parent/manifest.json', run)
    write(root/'project.json', {'kind': 'project', 'workflow_mode': 'single_source',
        'state': {'last_source_run': run['run_id']}})
    return run, values


def prepared(run, output, *, wrong_source=False):
    output = Path(output)
    output.mkdir(parents=True)
    box = [5, 4, 22, 19]
    generated = output/'generated-native.png'
    Image.new('RGB', (17, 15), (135, 125, 120)).save(generated)
    weight = np.zeros((15, 17), np.float32)
    weight[2:-2, 2:-2] = 1
    weights = output/'weights.npy'
    np.save(weights, weight, allow_pickle=False)
    record = {'schema': 'beautygrpo_layer_prepare.v1', 'working_domain': 'display_linear_sRGB',
        'source_path': run['source']['path'], 'source_sha256': 'wrong' if wrong_source else run['source']['sha256'],
        'native_size': [32, 24], 'source_precision': {'source_precision': 'raster8_srgb', 'decode_version': RGB_DECODE_VERSION},
        'source_display': source_transform_for_run(run), 'native_source_retouch': None,
        'model_identity': {'generator': 'CPU_contract_fixture_not_model_inference'},
        'faces': [{'id': 'face-01', 'native_box': box, 'generated_native_roi': str(generated),
            'generated_native_roi_sha256': sha256(generated), 'composition_weight': str(weights),
            'composition_weight_sha256': sha256(weights), 'model_output': str(generated)}]}
    path, layer = freeze_prepared(record, output/'frozen')
    return {'ok': True, 'result': str(path), 'layer': layer, 'cache_hit': False,
        'no_change': False, 'generation_inferences_this_attempt': 0}


def wait(backend, identifier):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        snapshot = backend.job(identifier)
        if snapshot['job']['status'] in {'succeeded', 'failed', 'cancelled', 'blocked'}:
            worker = backend._job_workers.get(identifier)
            if worker:
                worker.join(timeout=2)
            return snapshot
        time.sleep(.01)
    raise AssertionError('Beauty preparation did not finish')


def reject(call):
    try:
        call()
    except (ValueError, OSError):
        return
    raise AssertionError('Invalid frozen beauty state was accepted')


@contextmanager
def preparer_module(module):
    # Replacing only this entry leaves imported numerical extensions alive.
    name = 'beautygrpo_prepare'
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='cos-beauty-backend-') as temporary:
        base = Path(temporary)
        root = base/'workspace'
        run, source = project(root)
        parent = root/'outputs/runs/beauty-parent/manifest.json'
        project_path = root/'project.json'
        retained = parent.read_bytes(), project_path.read_bytes()
        backend = Backend(root, embedded=True)
        body = {'appearance_revision': {'params': {'global': {'exposure': .1}}}, 'manual_layers': {}}
        calls = []
        worker = types.ModuleType('beautygrpo_prepare')

        def forbidden(*args, **kwargs):
            raise AssertionError('A beauty prepare job must not run the CLI or generate a reference')

        backend.command = forbidden
        backend.one_click = forbidden

        def success(edited, project_root, output, options=None, *, cancel=None, progress=None):
            assert project_root == root and options == {} and cancel() is False
            assert not Path(output).exists() and Path(output).is_relative_to(root/'outputs/jobs')
            calls.append(copy.deepcopy(edited))
            progress('beauty_sampling', 'fixture stage', .5)
            return prepared(edited, output)

        with preparer_module(worker):
            worker.prepare_for_workbench = success
            started = backend.prepare_portrait_beauty(run['run_id'], body)
            assert started['job']['kind'] == 'prepare_portrait_beauty'
            finished = wait(backend, started['job_id'])
            assert finished['job']['status'] == 'succeeded', finished
            receipt = finished['portrait_beauty_result']
            assert receipt['input_sha256'] == digest(body)
            assert receipt['source_sha256'] == run['source']['sha256'] and receipt['run_id'] == run['run_id']
            assert receipt['preparation_contract'] == 'portrait_beauty_layer.v1'
            assert receipt['faces'] == [{'id': 'face-01', 'native_box': [5, 4, 22, 19]}]
            assert calls[0]['options']['appearance_revision'] == body['appearance_revision']
            assert (parent.read_bytes(), project_path.read_bytes()) == retained
            assert sorted(item.name for item in (root/'outputs/runs').iterdir()) == ['beauty-parent']
            assert Backend(root).job(started['job_id'])['portrait_beauty_result'] == receipt
            checks.append('source_only_async_prepare_durable_receipt_parent_and_edits_untouched')

            envelope = backend.jobs.path(started['job_id'])/'prepared-portrait-beauty.json'
            original = envelope.read_bytes()
            envelope.write_bytes(original+b' ')
            reject(lambda: backend.job(started['job_id']))
            envelope.write_bytes(original)
            changed = copy.deepcopy(receipt)
            changed['source_sha256'] = 'other'
            write(envelope, changed)
            changed_summary = copy.deepcopy(finished['job']['result_summary'])
            changed_summary['prepared_beauty_sha256'] = sha256(envelope)
            backend.jobs.update(started['job_id'], result_summary=changed_summary)
            reject(lambda: backend.job(started['job_id']))
            envelope.write_bytes(original)
            backend.jobs.update(started['job_id'], result_summary=finished['job']['result_summary'])
            checks.append('corrupt_and_resigned_wrong_source_receipts_rejected')

            def fail(*args, **kwargs):
                kwargs['progress']('beauty_sampling', 'failed fixture', .2)
                raise RuntimeError('fixture model unavailable')
            worker.prepare_for_workbench = fail
            failed = wait(backend, backend.prepare_portrait_beauty(run['run_id'], body)['job_id'])
            assert failed['job']['status'] == 'failed' and 'fixture model unavailable' in failed['job']['error']
            assert 'portrait_beauty_result' not in failed
            worker.prepare_for_workbench = lambda edited, project_root, output, **kwargs: prepared(edited, output, wrong_source=True)
            invalid = wait(backend, backend.prepare_portrait_beauty(run['run_id'], body)['job_id'])
            assert invalid['job']['status'] == 'failed'
            worker.prepare_for_workbench = lambda *args, **kwargs: {'ok': True, 'no_change': True,
                'result': None, 'layer': None, 'reason': 'no_visible_faces_available'}
            empty = wait(backend, backend.prepare_portrait_beauty(run['run_id'], body)['job_id'])
            assert empty['job']['status'] == 'succeeded'
            assert empty['portrait_beauty_result']['no_change'] and empty['portrait_beauty_result']['appearance_revision'] == {}
            checks.append('failure_wrong_source_and_no_visible_face_do_not_apply_a_layer')

            entered = threading.Event()
            def cancelling(*args, cancel=None, **kwargs):
                entered.set()
                while not cancel():
                    time.sleep(.005)
                raise InterruptedError('cancelled fixture')
            worker.prepare_for_workbench = cancelling
            identifier = backend.prepare_portrait_beauty(run['run_id'], body)['job_id']
            assert entered.wait(2)
            backend.cancel_job(identifier)
            assert wait(backend, identifier)['job']['status'] == 'cancelled'
            orphan = backend.jobs.create({'kind': 'prepare_portrait_beauty', 'preparation_kind': 'portrait_beauty'})
            assert Backend(root).job(orphan['job_id'])['job']['status'] == 'failed'
            assert (parent.read_bytes(), project_path.read_bytes()) == retained
            checks.append('cooperative_cancel_and_restart_interruption_keep_saved_parent')

            from current_stack_fixture import prepare_fixture
            paired = prepare_fixture(root/'paired-assets', native=(32, 24))
            paired.update(root=str(root), run_id='beauty-paired', sample='beauty-paired')
            paired['options']['manual_layers'] = {}
            baseline = paired['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']
            baseline['source_sha256'] = paired['source']['sha256']
            baseline['source_display'] = source_transform_for_precision('raster8_srgb')
            write(root/'outputs/runs/beauty-paired/manifest.json', paired)
            worker.prepare_for_workbench = success
            paired_receipt = wait(backend, backend.prepare_portrait_beauty(paired['run_id'], body)['job_id'])['portrait_beauty_result']
            paired['options']['appearance_revision']['portrait_beauty'] = paired_receipt['appearance_revision']['portrait_beauty']
            paired_binding = binding_for_run(root, paired)
            assert validate_assets(paired_binding) is None
            assert paired_binding['candidate_sha256'] == paired['reference']['sha256']
            assert any(asset['path'] == paired_receipt['appearance_revision']['portrait_beauty']['result']
                for asset in paired_binding['assets'])
            paired_store = SessionDraftStore(root)
            paired_store.checkpoint(paired_binding, {'portrait_beauty': paired['options']['appearance_revision']['portrait_beauty']})
            assert paired_store.restore(paired_binding)[0] is not None
            write(root/'outputs/runs/beauty-paired/manifest.json', paired)
            paired_package = base/'paired-beauty.cosplayproj'
            export_project(root, paired['run_id'], paired_package)
            imported_paired = import_project(base/'paired-relocated', paired_package, run_id='paired-relocated')['run']
            assert validate_portrait_beauty(imported_paired)[0]['source_sha256'] == paired['source']['sha256']
            assert validate_assets(binding_for_run(base/'paired-relocated', imported_paired)) is None
            checks.append('same_beauty_prepare_on_paired_reference_source_with_bound_session_and_portable_assets')

        control = receipt['appearance_revision']['portrait_beauty']
        edited = copy.deepcopy(run)
        edited['options']['appearance_revision']['portrait_beauty'] = control
        layer, path = validate_portrait_beauty(edited)
        after, evidence = apply_for_run(source, edited)
        assert evidence['applied'] and not np.array_equal(after, source)
        assert np.array_equal(after[:4], source[:4])
        off = copy.deepcopy(edited)
        off['options']['appearance_revision']['portrait_beauty']['disabled_faces'] = ['face-01']
        assert np.array_equal(apply_for_run(source, off)[0], source)
        assert merge_appearance_revision(edited['options']['appearance_revision'], {'portrait_beauty': {}})['portrait_beauty'] == {}
        assert appearance_revision_patch({'portrait_beauty': control}) == {'portrait_beauty': control}
        assert editable_payload({'portrait_beauty': control})['portrait_beauty'] == control
        checks.append('native_frozen_pixels_default_strength_and_face_restore_component_replacement')

        from workbench_store import save_template
        payload = {'appearance_revision': copy.deepcopy(edited['options']['appearance_revision']), 'manual_layers': {}}
        saved_template = save_template(root, 'beauty preset', payload)
        assert 'portrait_beauty' not in saved_template['parameters']['appearance_revision']
        assert payload['appearance_revision']['portrait_beauty'] == control
        checks.append('templates_never_carry_source_bound_generated_faces')

        from cos_workflow_graph import preset
        graph = preset('portrait')
        clean_template = save_template(root, 'Cos workflow', {'workflow_graph': graph, 'portrait_beauty': control})
        assert clean_template['parameters'] == {'workflow_graph': graph}
        reject(lambda: save_template(root, 'contaminated workflow', {'workflow_graph': {**graph, 'result': control['result']}}))
        checks.append('reusable_workflow_template_accepts_typed_intent_and_rejects_asset_locators')
        session = SessionDraftStore(root)
        binding = binding_for_run(root, run)
        assert binding['candidate_sha256'] is None and validate_assets(binding) is None
        session_payload = {'portrait_beauty': copy.deepcopy(control), 'workflow_graph': graph, 'manual_layers': {}}
        session.checkpoint(binding, session_payload)
        assert session.restore(binding)[0]['payload']['workflow_graph'] == graph
        original_result = path.read_bytes()
        path.write_bytes(original_result+b' ')
        assert session.restore(binding)[0] is None
        path.write_bytes(original_result)
        session.checkpoint(binding, session_payload)
        pixel_asset = Path(layer['faces'][0]['pixels']['path'])
        original_pixels = pixel_asset.read_bytes()
        pixel_asset.write_bytes(original_pixels+b'changed')
        assert session.restore(binding)[0] is None
        pixel_asset.write_bytes(original_pixels)
        session.checkpoint(binding, session_payload)
        assert session.restore(binding)[0]['payload']['portrait_beauty'] == control
        checks.append('source_only_session_recovers_unsaved_beauty_and_workflow_and_rejects_modified_assets')

        # Exercise the actual child save pipeline. A geometry inference is fatal:
        # frozen native geometry and generated pixels must be sufficient.
        with patch('source_alpha.source_silhouette', side_effect=AssertionError('No geometry/model inference during saved replay')):
            child = backend.adjust_source_run(run['run_id'], {'portrait_beauty': control, 'workflow_graph': graph})['run']
            assert child['continuation_edit']['render_state'] == 'saved'
            child_control = child['options']['appearance_revision']['portrait_beauty']
            assert Path(child_control['result']).is_relative_to(root/'outputs/runs'/child['run_id'])
            assert child['options']['appearance_revision']['workflow_graph'] == graph
            import tifffile
            np.testing.assert_array_equal(tifffile.imread(child['artifacts']['render_scene_master']), after)
            assert parent.read_bytes() == retained[0]
            assert validate_assets(binding_for_run(root, child)) is None
            jobs = root/'outputs/jobs'
            assert jobs.resolve().parent == root.resolve()/'outputs'
            shutil.rmtree(jobs)
            reopened_backend = Backend(root)
            reopened_child = reopened_backend.source_run(child['run_id'])['run']
            np.testing.assert_array_equal(apply_for_run(source, reopened_child)[0], after)
            half = copy.deepcopy(reopened_child['options']['appearance_revision']['portrait_beauty'])
            half['strength'] = .5
            saved_half = reopened_backend.adjust_source_run(child['run_id'], {'portrait_beauty': half})['run']
            expected_half = apply_for_run(source, saved_half)[0]
            np.testing.assert_array_equal(tifffile.imread(saved_half['artifacts']['render_scene_master']), expected_half)
            assert saved_half['continuation_edit']['render_state'] == 'saved'
        checks.append('actual_save_child_owned_beauty_and_geometry_reopen_after_jobs_deleted_strength_save_no_models')

        # Persist a real frozen pixel scene, package it, remove every original
        # path, and replay from the imported JSON/NPZ without producer modules.
        package = base/'beauty.cosplayproj'
        export_project(root, saved_half['run_id'], package)
        moved_root = base/'relocated-workspace'
        moved = import_project(moved_root, package, run_id='beauty-relocated')['run']
        moved_control = moved['options']['appearance_revision']['portrait_beauty']
        assert moved_control['result'] != control['result']
        assert Path(moved_control['result']).is_relative_to(moved_root)
        assert validate_portrait_beauty(moved)[0]['faces'][0]['pixels']['sha256'] == layer['faces'][0]['pixels']['sha256']
        # Only the verified task-owned temp workspace is deleted.
        assert root.resolve().parent == base.resolve() and root.name == 'workspace'
        shutil.rmtree(root)
        reopened = Backend(moved_root).source_run('beauty-relocated')['run']
        replay, _ = apply_for_run(source, reopened)
        np.testing.assert_array_equal(replay, expected_half)
        assert not any(name in sys.modules for name in ('torch', 'diffusers', 'beautygrpo_model_worker'))
        checks.append('real_portable_v2_relocation_after_original_assets_deleted_exact_float32_replay_no_models')
    print(json.dumps({'ok': True, 'checks': checks, 'gpu_calls': 0,
        'beauty_model_inferences': 0, 'scope': 'synthetic frozen pixels and mocked model preparation, not beauty quality'}))


if __name__ == '__main__':
    main()
