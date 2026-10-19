"""Offline source CPU feature preparation: actual masks, strict negative cases."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import cosplay_native_review as native
import cosplay_post_review as visual
import source_feature_observation_worker as worker
from cosplay_native_review_smoke import make_native_fixture
from cosplay_post_review_smoke import multi_fake_transport, response
from current_model_registry import PATH_KEYS, PROFILES, SCHEMA
from material_anchor_review import file_record, load_observation, prepare_observation
from runtime_resources import code_path


def fixture(root):
    root = Path(root)
    run = make_native_fixture(root/'inputs', native_size=(330, 420))
    registry = {**PROFILES, 'schema': SCHEMA, 'ready': True}
    for key in PATH_KEYS:
        path = root/('registry-'+key)
        path.write_bytes(('synthetic fixture '+key).encode())
        registry[key] = str(path.resolve())
    registry['python'] = sys.executable
    registry_path = root/'registry.json'
    visual._write(registry_path, registry)
    run['options']['model_registry'] = str(registry_path)
    semantic = run['options']['appearance_revision']['design_reconstruction_plan']['layers'][0]['assets']['source_shoe_semantics']
    from local_semantics_worker import CLASS_NAMES
    identity = {'analysis_role': 'source', 'fixture_identity': 'current-source-cached-only'}
    directory = root/'outputs/cache/local-analysis'/run['source']['sha256'][:16]/'current'
    directory.mkdir(parents=True)
    analysis_path = directory/'analysis.json'
    visual._write(analysis_path, {'source': run['source'], 'analysis_role': 'source',
        'analysis_identity': identity, 'semantic_result': {**semantic, 'labels_path': semantic['path'],
            'class_names': CLASS_NAMES}})
    source_display = root/'canonical-source.png'
    Image.new('RGB', (330, 420), (170, 190, 200)).save(source_display)
    request = {'schema': 'source_feature_observation_request.v1', 'root': str(root),
        'run': {'source': run['source'], 'options': run['options']}, 'source': file_record(run['source']['path']),
        'source_display': file_record(source_display), 'source_semantics': {**semantic,
            'source_sha256': run['source']['sha256'], 'size': [110, 140]}, 'native_size': [330, 420]}
    return run, request, identity, analysis_path, registry


def fake_face(mode='available', *, cache_hit=False):
    def observe(root, image_path, semantics, *, registry_path=None):
        registry = json.loads(Path(registry_path).read_text())
        width, height = Image.open(image_path).size
        regions = []
        if mode != 'no_landmarks':
            for name, box in [('left_eye', [.36, .28, .41, .31]),
                              ('right_eye', [.50, .28, .55, .31]), ('lips', [.42, .36, .50, .39])]:
                x0, y0, x1, y1 = box
                regions.append({'region': name, 'polygon': [[x0*width, y0*height],
                    [x1*width, y0*height], [x1*width, y1*height], [x0*width, y1*height]],
                    'coordinate_space': 'source_proxy'})
            if mode == 'invalid_polygon':
                regions[0]['polygon'][0][0] = -1
            if mode == 'missing_eye':
                regions.pop(1)
        result = {'method': 'no_landmarks_protect_entire_face_neck' if mode == 'no_landmarks' else 'mediapipe_source_eyes_lips',
            'input_path': str(image_path), 'size': [width, height], 'protected_regions': regions,
            'model_sha256': visual._sha(registry['face_model_path']), 'cache_hit': cache_hit,
            'identity': {'input': visual._sha(image_path), 'labels': visual._sha(semantics['labels_path']),
                'model': visual._sha(registry['face_model_path']),
                'worker': visual._sha(code_path('local_face_protection_worker.py')),
                'parameters': {'num_faces': 1, 'semantic_crop': 'Face_Neck',
                    'detection_confidence': .3, 'presence_confidence': .3}}}
        key = hashlib.sha256(json.dumps(result['identity'], sort_keys=True).encode()).hexdigest()[:24]
        cache = Path(root)/'outputs/cache/face-protection'/key/'result.json'
        if cache_hit and cache.is_file():
            return {**json.loads(cache.read_text()), 'cache_hit': True}
        cache.parent.mkdir(parents=True, exist_ok=True)
        visual._write(cache, {**result, 'cache_hit': False})
        return result
    return observe


def display_derivative_cases(root, request, analysis_path, passed):
    """Large source displays must survive the actual strict material binding."""
    from raw_support import deterministic_srgb_profile
    directory = root/'display-derivatives'
    directory.mkdir()
    for name, size, maximum in (
            ('large-portrait', (1537, 9504), 2048),
            ('wide-nonintegral-ratio', (5001, 1443), 512),
            ('small-no-upscale', (311, 173), 2048)):
        width, height = size
        # Fine irregular pixels expose thumbnail's optional pre-reduction;
        # uniform fixtures cannot distinguish the two Lanczos data flows.
        rng = np.random.default_rng(2631)
        parent = Image.fromarray(rng.integers(0, 256, (height, width), dtype=np.uint8)).convert('RGB')
        displayed = directory/(name+'.png')
        parent.save(displayed, icc_profile=deterministic_srgb_profile())
        before = displayed.read_bytes()
        with patch('material_anchor_review.prepare_face_protection', side_effect=fake_face()), \
             patch('vision_transport.request_vision_json', side_effect=AssertionError('No API allowed')):
            spec, observation = prepare_observation(root, 'source', displayed,
                request['source']['path'], analysis_path, directory/name,
                maximum_edge=maximum, registry_path=request['run']['options']['model_registry'])
        canonical = visual._raster(displayed)
        old_thumbnail = canonical.copy()
        old_thumbnail.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
        actual = visual._raster(spec['binding']['image']['path'])
        expected = canonical.resize(actual.size, Image.Resampling.LANCZOS)
        assert actual.size == old_thumbnail.size
        assert np.array_equal(np.asarray(actual), np.asarray(expected))
        assert spec['binding']['display_parent'] == file_record(displayed)
        assert spec['binding']['display_derivative'] == 'oriented_srgb_lanczos'
        assert 'icc_profile' not in Image.open(spec['binding']['image']['path']).info
        assert displayed.read_bytes() == before
        assert load_observation(spec)['rgb'].shape == observation['rgb'].shape
        passed.append(name+'_canonical_srgb_single_lanczos_parent_binding')
        if name == 'small-no-upscale':
            assert actual.size == size
            assert np.array_equal(np.asarray(actual), np.asarray(canonical))
            continue
        assert not np.array_equal(np.asarray(old_thumbnail), np.asarray(expected))
        # Even freshly rehashed image/ownership receipts cannot authorize the
        # historical two-step pixels or one edited pixel under this parent.
        image_path = Path(spec['binding']['image']['path'])
        ownership_path = Path(spec['binding']['feature_ownership_receipt']['path'])
        image_bytes, ownership_bytes = image_path.read_bytes(), ownership_path.read_bytes()
        modified = np.asarray(actual).copy()
        modified[0, 0, 0] ^= 1
        for label, replacement in (('legacy_prereduction', old_thumbnail),
                                   ('altered_pixel', Image.fromarray(modified))):
            replacement.save(image_path, icc_profile=None)
            rejected = copy.deepcopy(spec)
            rejected['binding']['image'] = file_record(image_path)
            ownership = json.loads(ownership_bytes)
            ownership['input'] = rejected['binding']['image']
            visual._write(ownership_path, ownership)
            rejected['binding']['feature_ownership_receipt'] = file_record(ownership_path)
            try:
                load_observation(rejected)
            except ValueError as exc:
                assert 'explicit bound sRGB parent resize' in str(exc), str(exc)
                passed.append(name+'_'+label+'_rehashed_still_rejected')
            else:
                raise AssertionError(label+' derivative accepted under another bound parent')
            finally:
                image_path.write_bytes(image_bytes)
                ownership_path.write_bytes(ownership_bytes)


def main():
    passed = []
    with tempfile.TemporaryDirectory(prefix='source-features-smoke-') as temporary:
        root = Path(temporary)
        run, request, identity, analysis_path, registry = fixture(root)
        original = {path: path.read_bytes() for path in (root/'inputs').iterdir()}
        def execute(name, mutate=None, mode='available', cache_hit=False):
            current = copy.deepcopy(request)
            if mutate:
                mutate(current)
            path = root/(name+'-request.json'); visual._write(path, current)
            with patch('local_photo_analysis.analysis_identity', return_value=identity), \
                 patch('local_photo_analysis.prepare_local_appearance', side_effect=AssertionError('No segmentation allowed')), \
                 patch('vision_transport.request_vision_json', side_effect=AssertionError('No API allowed')), \
                 patch('material_anchor_review.prepare_face_protection', side_effect=fake_face(mode, cache_hit=cache_hit)):
                receipt = worker.prepare(path, root/name)
            return receipt, file_record(root/name/'preparation.json')

        prepared, record = execute('valid')
        assert prepared['status'] == 'prepared', prepared
        valid_cache_path = Path(prepared['face_cache_result']['path'])
        valid_cache_bytes = valid_cache_path.read_bytes()
        assert prepared['status'] == 'prepared', prepared
        assert prepared['cpu_face_landmark_inference_requests'] == 1
        spec = json.loads(Path(prepared['observation']['path']).read_text())
        with patch('local_photo_analysis.analysis_identity', return_value=identity), \
             patch('material_anchor_review.prepare_face_protection', side_effect=AssertionError('Saved checks must be read only')):
            checked, proof = worker.validate_saved_preparation(record, spec,
                source_sha256=run['source']['sha256'], source_display={**request['source_display'], 'size': [330, 420]},
                semantic_binding=request['source_semantics'])
        assert checked == prepared and proof
        passed.append('actual_prepare_observation_masks_and_read_only_saved_binding')
        display_derivative_cases(root, request, analysis_path, passed)
        cached, _ = execute('cache-hit', cache_hit=True)
        assert cached['status'] == 'prepared' and cached['cpu_face_landmark_inference_requests'] == 0
        assert cached['cpu_face_landmark_worker_attempted'] is False
        cached_owned = json.loads((root/'cache-hit/observation/feature-ownership.json').read_text())
        assert Path(cached_owned['face_observation']['input_path']).parent != root/'cache-hit/observation'
        passed.append('cpu_cache_hit_zero_inference_accounting')
        passed.append('repeated_output_reuses_historical_cache_path_with_exact_current_bytes')
        for name, mutation, mode in [
            ('wrong-source-sha', lambda r: r['source'].update(sha256='0'*64), 'available'),
            ('wrong-native-canvas', lambda r: r.update(native_size=[331, 420]), 'available'),
            ('wrong-semantic-coordinates', lambda r: r['source_semantics'].update(size=[111, 140]), 'available'),
            ('wrong-semantic-sha', lambda r: r['source_semantics'].update(sha256='0'*64), 'available'),
            ('wrong-source-owner', lambda r: r['source_semantics'].update(source_sha256='0'*64), 'available'),
            ('invalid-polygon', None, 'invalid_polygon'), ('missing-eye-topology', None, 'missing_eye')]:
            result, _ = execute(name, mutation, mode)
            assert result['status'] == 'unknown' and 'observation' not in result, (name, result)
            assert all(result[key] == 0 for key in ('segmentation_inference_requests', 'gpu_inference_requests',
                'image_generation_requests', 'text_vision_requests'))
            passed.append(name+'_unknown')
        original_analysis = analysis_path.read_bytes()
        altered = json.loads(original_analysis); altered['analysis_identity'] = {'retired': True}
        visual._write(analysis_path, altered)
        result, _ = execute('stale-current-identity')
        assert result['status'] == 'unknown' and not result['cpu_face_landmark_worker_attempted']
        analysis_path.write_bytes(original_analysis)
        passed.append('stale_current_semantics_fail_without_producer')
        analysis_path.rename(analysis_path.with_suffix('.saved'))
        result, _ = execute('cache-missing')
        assert result['status'] == 'unknown' and not result['cpu_face_landmark_worker_attempted']
        analysis_path.with_suffix('.saved').rename(analysis_path)
        passed.append('missing_current_cache_no_inference_fallback')
        registry_path = Path(run['options']['model_registry']); old_registry = registry_path.read_bytes()
        visual._write(registry_path, {**registry, 'semantics_profile': 'retired'})
        result, _ = execute('registry-wrong-profile')
        assert result['status'] == 'unknown' and not result['cpu_face_landmark_worker_attempted']
        registry_path.write_bytes(old_registry)
        passed.append('selected_registry_profile_strict')
        no_landmarks, _ = execute('no-landmarks', mode='no_landmarks')
        assert no_landmarks['status'] == 'prepared' and no_landmarks['landmark_status'] == 'unknown'
        with np.load(Path(no_landmarks['observation']['path']).parent/'eyes.npz') as data:
            assert np.count_nonzero(data['values']) == 0
        passed.append('no_landmarks_zero_eyes_and_explicit_unknown')
        try:
            worker.prepare(Path(prepared['request']['path']), root/'valid')
        except FileExistsError:
            passed.append('immutable_preparation')
        else:
            raise AssertionError('Existing observation was overwritten')

        launched = []
        def external_cpu(argv, **kwargs):
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
            assert Path(argv[0]).resolve() == Path(sys.executable).resolve()
            launched.append(argv)
            worker.prepare(Path(argv[argv.index('--request')+1]), Path(argv[argv.index('--output')+1]))
            return SimpleNamespace(returncode=0)
        with patch('local_photo_analysis.analysis_identity', return_value=identity), \
             patch('local_photo_analysis.prepare_local_appearance', side_effect=AssertionError('No Sapiens producer')), \
             patch('material_anchor_review.prepare_face_protection', side_effect=fake_face()), \
             patch('subprocess.run', side_effect=external_cpu):
            selected = native.prepare_cosplay_native_review(run, root/'native-default', feature_root=root)
        assert selected['receipt']['status'] == 'ready', selected['receipt']
        assert len(launched) == 1 and selected['receipt']['models_used'] is True
        assert {r['pair_id'] for r in selected['receipt']['regions']} >= {'eyes_0', 'eyes_1', 'lips'}
        assert all(row['status'] == 'available' for row in selected['receipt']['required_feature_evidence'].values())
        passed.append('default_external_cpu_path_selects_both_eyes_lips')
        design = {'design_brief': {'summary': 'Synthetic costume; source identity is mandatory.'}}
        with patch.object(visual.vision_transport, 'request_vision_json', side_effect=multi_fake_transport(response(True))):
            result = visual.review_cosplay_post(selected['source_path'], Path(run['reference']['path']), root/'qa', design,
                final=selected['final_path'], native_crops=selected['native_crops'], native_selection=selected['receipt'])
        with patch('local_photo_analysis.analysis_identity', return_value=identity), \
             patch('material_anchor_review.prepare_face_protection', side_effect=AssertionError('No model in saved QA')):
            saved, proof = visual.validate_saved_comprehensive_review(selected['source_path'], Path(run['reference']['path']), root/'qa/review.json', design)
        assert saved == result and any(row['path'] == selected['receipt']['source_feature_preparation']['path'] for row in proof)
        passed.append('full_saved_native_qa_binds_cpu_receipt_and_dependencies')
        with patch('local_photo_analysis.analysis_identity', return_value=identity), \
             patch('material_anchor_review.prepare_face_protection', side_effect=fake_face('invalid_polygon')), \
             patch('subprocess.run', side_effect=external_cpu):
            unknown_selected = native.prepare_cosplay_native_review(run, root/'native-unknown', feature_root=root)
        assert unknown_selected['receipt']['status'] == 'ready' and unknown_selected['receipt']['feature_binding'] is None
        assert unknown_selected['receipt']['cpu_face_landmark_inference_requests'] is None
        with patch.object(visual.vision_transport, 'request_vision_json', side_effect=multi_fake_transport(response(True))):
            unknown_result = visual.review_cosplay_post(unknown_selected['source_path'], Path(run['reference']['path']), root/'unknown-qa', design,
                final=unknown_selected['final_path'], native_crops=unknown_selected['native_crops'], native_selection=unknown_selected['receipt'])
        with patch('material_anchor_review.prepare_face_protection', side_effect=AssertionError('No saved models')):
            unknown_saved, unknown_proof = visual.validate_saved_comprehensive_review(unknown_selected['source_path'],
                Path(run['reference']['path']), root/'unknown-qa/review.json', design)
        assert unknown_saved == unknown_result and unknown_proof
        assert unknown_saved['checks']['identity_makeup']['final']['status'] == 'unknown'
        assert unknown_saved['checks']['skin_materials']['final']['status'] == 'unknown'
        passed.append('failed_cpu_receipt_read_only_revalidated_with_features_unknown')

        # Successful saved preparation must retain the current unique source analysis.
        valid_cache_path.write_bytes(valid_cache_bytes)
        def saved_reject(name, *, current_identity=identity, mutate=None):
            current_spec = copy.deepcopy(spec)
            if mutate:
                mutate(current_spec)
            with patch('local_photo_analysis.analysis_identity', return_value=current_identity):
                try:
                    worker.validate_saved_preparation(record, current_spec,
                        source_sha256=run['source']['sha256'],
                        source_display={**request['source_display'], 'size': [330, 420]},
                        semantic_binding=request['source_semantics'])
                except ValueError:
                    passed.append(name)
                else:
                    raise AssertionError(name+' was accepted')
        saved_reject('saved_current_analysis_identity_change_rejected', current_identity={'retired': True})
        saved_reject('saved_domain_changed_rejected', mutate=lambda s: s.update(domain='scene_linear_sRGB'))
        saved_reject('saved_other_source_identity_rejected', mutate=lambda s: s['binding']['identity_root'].update(sha256='0'*64))
        saved_reject('saved_wrong_display_parent_rejected', mutate=lambda s: s['binding']['display_parent'].update(sha256='0'*64))
        without = copy.deepcopy(selected['receipt'])
        without['feature_binding'] = None
        without.pop('source_feature_preparation')
        without['models_used'] = False
        without['required_feature_evidence'] = native._required_feature_evidence(without)
        assets = result['binding']['images']; crops = visual._native_crops(selected['native_crops'], assets, maximum=64)
        missing = visual._aggregate_multiple(result['reviews'], result['native_sampling_coverage']['requested_pairs'],
            assets, design, without, crops)
        assert missing['checks']['identity_makeup']['final']['status'] == 'unknown'
        assert missing['checks']['skin_materials']['final']['status'] == 'unknown'
        assert {row['region'] for row in missing['required_actions'] if row['criterion'] == 'feature_coverage'} == {'eyes', 'lips'}
        passed.append('whole_pass_cannot_fill_unowned_feature_coverage')

        def reject_owned(name, modify_face=None, modify_cache=None):
            valid_cache_path.write_bytes(valid_cache_bytes)
            current_spec = copy.deepcopy(spec)
            current_receipt = copy.deepcopy(prepared)
            path = Path(current_spec['binding']['feature_ownership_receipt']['path'])
            old = path.read_bytes(); document = json.loads(old)
            if modify_face:
                modify_face(document['face_observation'])
            if modify_cache:
                cached_result = json.loads(valid_cache_bytes)
                modify_cache(cached_result)
                visual._write(valid_cache_path, cached_result)
                current_receipt['face_cache_result'] = file_record(valid_cache_path)
            visual._write(path, document)
            current_spec['binding']['feature_ownership_receipt'] = file_record(path)
            try:
                worker._validate_cpu_observation(current_spec, current_receipt)
            except ValueError:
                passed.append(name)
            else:
                raise AssertionError(name+' was accepted')
            finally:
                path.write_bytes(old); valid_cache_path.write_bytes(valid_cache_bytes)
        reject_owned('lowered_cpu_threshold_rejected', lambda f: f['identity']['parameters'].update(detection_confidence=.2))
        reject_owned('cached_other_input_bytes_rejected', lambda f: f['identity'].update(input='0'*64))
        reject_owned('cached_other_label_bytes_rejected', lambda f: f['identity'].update(labels='0'*64))
        reject_owned('changed_actual_cache_polygons_rejected', modify_cache=lambda f: f['protected_regions'][0]['polygon'][0].__setitem__(0, 0))
        wrong_path = str(root/'a-different-source.png')
        reject_owned('fresh_result_other_path_rejected', lambda f: f.update(input_path=wrong_path),
            lambda f: f.update(input_path=wrong_path))

        # Rehashed mask/ownership/spec receipts still require actual polygon reconstruction.
        masked = copy.deepcopy(spec)
        valid_cache_path.write_bytes(valid_cache_bytes)
        eye_path = Path(masked['binding']['feature_masks']['eyes']['path'])
        old_eye = eye_path.read_bytes(); np.savez_compressed(eye_path, values=np.ones((420, 330), np.float32))
        masked['binding']['feature_masks']['eyes'] = file_record(eye_path)
        ownership_path = Path(masked['binding']['feature_ownership_receipt']['path'])
        old_ownership = ownership_path.read_bytes(); ownership = json.loads(old_ownership)
        ownership['feature_masks'] = masked['binding']['feature_masks']; visual._write(ownership_path, ownership)
        masked['binding']['feature_ownership_receipt'] = file_record(ownership_path)
        try:
            worker._validate_cpu_observation(masked, prepared)
        except ValueError as exc:
            assert 'actual CPU polygons' in str(exc)
            passed.append('rehashed_fabricated_eye_mask_rejected')
        else:
            raise AssertionError('Fabricated rehashed eye mask accepted')
        eye_path.write_bytes(old_eye); ownership_path.write_bytes(old_ownership)

        # Bound cache ancestry is the only supported fallback when run.root is absent.
        inferred_semantic = {**request['source_semantics'], 'path': str(analysis_path.parent/'labels.npz')}
        with patch('subprocess.run', return_value=SimpleNamespace(returncode=1)):
            output = root/'root-fallback'; output.mkdir()
            try:
                native._current_source_feature_observation(run, output, Path(request['source_display']['path']), inferred_semantic)
            except RuntimeError:
                pass
        inferred = json.loads((output/'source-feature-request.json').read_text())
        assert Path(inferred['root']).resolve() == root.resolve()
        passed.append('no_run_root_cache_ancestry_resolves_exact_project')
        assert all(path.read_bytes() == contents for path, contents in original.items())
        passed.append('frozen_source_inputs_unchanged')
    command = [sys.executable, '-B', '-c', "import sys; sys.path.insert(0, 'scripts'); import cosplay_native_review, cosplay_post_review, source_feature_observation_worker; assert not ({'cv2','torch','mediapipe'} & set(sys.modules))"]
    subprocess.run(command, check=True)
    passed.append('desktop_imports_no_cv2_torch_mediapipe')
    print(json.dumps({'schema': 'source_feature_observation_smoke.v1', 'status': 'passed',
        'cases_passed': len(passed), 'cases': passed, 'actual_network_requests': 0,
        'actual_model_inference_requests': 0, 'gpu_requests': 0, 'image_generation_requests': 0}))


if __name__ == '__main__':
    main()
