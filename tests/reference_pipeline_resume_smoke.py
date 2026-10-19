"""Verify interrupted reference jobs reuse only a validated stage prefix."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import reference_pipeline as pipeline


def prepare_stage_lock_contract(root):
    """Enter prepare's real lock/validation branch, stopping before pixel work."""
    import numpy as np
    from PIL import Image
    from display_transform import source_transform_for_precision
    from raw_support import RGB_DECODE_VERSION

    root.mkdir()
    def asset(name, *, raster=False):
        path = root/name
        if raster:
            Image.new('RGB', (4, 6), (40, 80, 120)).save(path)
        else:
            np.savez(path, values=np.ones((6, 4), np.float32))
        return {'path': str(path), 'sha256': pipeline._sha(path), 'size': [4, 6]}
    source, reference = asset('source.png', raster=True), asset('reference.png', raster=True)
    alpha, protection, background, validity, fp = [asset(name+'.npz') for name in (
        'alpha', 'protection', 'background', 'validity', 'foreground')]
    registry = {'sha256': 'CPU-installation-fixture', 'semantics_profile': 'sapiens2_seg_0.4b',
                'matting_profile': 'vitmatte_small_distinctions'}
    plan = {'schema_version': 'design_reconstruction_plan.v1', 'source_sha256': source['sha256'],
        'candidate_sha256': reference['sha256'], 'working_domain': 'display_linear_sRGB',
        'mask_roles': {'foreground_alpha': alpha, 'identity_protection': protection},
        'canvas': {'source_oriented_size': [4, 6], 'source_proxy_size': [4, 6],
            'candidate_mapping': {'candidate_original_to_final_original': [[1,0,0],[0,1,0],[0,0,1]]}},
        'source_baseline': {'schema': 'appearance_source_baseline.v1',
            'source_sha256': source['sha256'], 'source_precision': 'raster8_srgb',
            'decode_version': RGB_DECODE_VERSION, 'working_domain': 'display_linear_sRGB',
            'source_display': source_transform_for_precision('raster8_srgb'),
            'source_display_applied_once': True,
            'assets': {'foreground_alpha': alpha, 'foreground_premultiplied': fp}},
        'layers': [{'kind': 'background_design', 'assets': {'background': background, 'validity': validity}}],
        'provenance': {'local_model_registry': registry}}
    run = {'source': source, 'reference': reference,
           'options': {'appearance_revision': {'design_reconstruction_plan': plan}}}
    # Installation/model feature discovery is isolated; source, matte,
    # code identities and stage receipt validation below remain production code.
    with patch.object(pipeline, '_registry_identity', return_value=registry), \
            patch.object(pipeline, '_feature_identity', return_value={'available': False, 'reason': 'CPU fixture'}):
        identity = pipeline._input_identity(root, run)
        work = root/'outputs/cache/reference-pipeline'/pipeline._digest(identity)[:20]
        match = work/'matching'
        match.mkdir(parents=True)
        np.savez(match/'source-weights.npz', values=np.ones((6, 4), np.float32))
        binding = {'source': source['sha256'], 'reference': reference['sha256'], 'alpha': alpha['sha256'],
            'source_display': identity['source_display'], 'portrait_beauty': identity['portrait_beauty'],
            'native_source_retouch': identity.get('native_source_retouch'),
            'algorithm_sha256': pipeline._sha(pipeline._code_file('reference_colour_fit.py')),
            'evaluation_sha256': pipeline._sha(pipeline._code_file('evaluate_reference_colour.py')),
            'source_weights_sha256': pipeline._sha(match/'source-weights.npz')}
        pipeline._write(match/'parameters.json', {'binding': binding})
        pipeline._write(match/'validation.json', {'scope': 'CPU stage-resume fixture'})
        Image.new('RGB', (4, 6)).save(match/'anchors.jpg')
        assert pipeline._resume_fit_ready(root, run, match, identity)

        active_lock = []
        ready_calls = []
        actual_lock, actual_ready = pipeline.exclusive_file_lock, pipeline._resume_stage_ready
        @contextmanager
        def observed_lock(path, **kwargs):
            with actual_lock(path, **kwargs):
                active_lock.append(path)
                try:
                    yield path
                finally:
                    active_lock.pop()
        def observed_ready(root, run, directory, stage, current_identity, **kwargs):
            ready_calls.append((stage, bool(active_lock), current_identity))
            return actual_ready(root, run, directory, stage, current_identity, **kwargs)
        class StageBoundaryReached(InterruptedError):
            pass
        def stop_after_fit(stage, message, fraction):
            if stage == 'colour_fusion':
                raise StageBoundaryReached('fit reused; stop before colour work')
        with patch.object(pipeline, 'exclusive_file_lock', side_effect=observed_lock), \
                patch.object(pipeline, '_resume_stage_ready', side_effect=observed_ready), \
                patch.object(pipeline, '_process', side_effect=AssertionError('Validated fit was recomputed')):
            try:
                pipeline.prepare(root, run, progress=stop_after_fit)
            except StageBoundaryReached:
                pass
            else:
                raise AssertionError('Expected stage boundary was not reached')
        assert any(stage == 'fit' and locked and current == identity for stage, locked, current in ready_calls)
        state = pipeline._read(work/'preparation-state.json')
        assert state['completed_stages'] == ['fit']
        assert state['stage_metrics']['fit']['cache_hit'] is True
        assert (work/'fit.single-flight.lock').is_file()

        # A real binding mismatch must schedule the existing worker, even
        # when a preceding attempt left a completed fit in the journal.
        pipeline._write(match/'parameters.json', {'binding': {**binding, 'source': 'other-source'}})
        assert not pipeline._resume_fit_ready(root, run, match, identity)
        with patch.object(pipeline, '_process', side_effect=StageBoundaryReached('recompute invalid fit')) as worker:
            try:
                pipeline.prepare(root, run)
            except StageBoundaryReached:
                pass
            else:
                raise AssertionError('Invalid fit was reused')
        assert worker.call_count == 1
        assert Path(worker.call_args.args[1][0]).name == 'evaluate_reference_colour.py'
        assert pipeline._read(work/'preparation-state.json')['completed_stages'] == []


def main():
    with TemporaryDirectory(prefix='reference-resume-') as folder:
        root = Path(folder)
        work = root / 'outputs' / 'cache' / 'reference-pipeline' / 'fixture'
        matching = work / 'matching'; matching.mkdir(parents=True)
        worker = work / 'input-snapshot.json'; worker.write_text('{"fixture":true}\n', encoding='utf8')
        identity = {'schema': pipeline.INPUT_CONTRACT, 'code': {'fixture': 'current'},
                    'source': {'sha256': 'source'}, 'reference': {'sha256': 'reference'}}
        stages = ['fit', 'colour_fusion', 'native_baseline', 'scene_balance']
        state = {'schema': 'reference_pipeline_job.v1', 'status': 'preparing',
                 'identity': copy.deepcopy(identity), 'input_sha256': pipeline._sha(worker),
                 'completed_stages': stages[:],
                 'stage_metrics': {name: {'status': 'completed'} for name in stages}}
        with patch.object(pipeline, '_code_identity', return_value={'fixture': 'current'}), \
                patch.object(pipeline, '_resume_stage_ready', return_value=True) as ready:
            prefix = pipeline._resume_completed_stages(root, {}, work, state, identity,
                                                       pipeline._sha(worker))
            assert prefix == stages
            assert ready.call_count == len(stages)
            changed = copy.deepcopy(state); changed['identity']['source']['sha256'] = 'changed'
            assert pipeline._resume_completed_stages(root, {}, work, changed, identity,
                                                      pipeline._sha(worker)) == []
            failed = copy.deepcopy(state); failed['stage_metrics']['scene_balance']['status'] = 'failed'
            assert pipeline._resume_completed_stages(root, {}, work, failed, identity,
                                                      pipeline._sha(worker)) == stages[:3]
        prepare_stage_lock_contract(root/'real-prepare')
        print(json.dumps({'ok': True, 'same_identity_prefix_reused': stages,
                          'prepare_real_lock_resume_branch_exercised': True,
                          'actual_fit_binding_rechecked_and_invalid_fit_recomputed': True,
                          'changed_identity_rejected': True, 'failed_stage_stops_prefix': True}))


if __name__ == '__main__':
    main()
