"""Freeze source-only feature location from cached semantics and CPU landmarks.

No segmentation producer, CUDA, generation, material qualification or text
vision is invoked. Run this module in the selected external Python runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

SCHEMA = 'source_feature_observation_preparation.v1'


def _validate_cpu_observation(spec, receipt):
    """Reconstruct feature masks from the bound own-canvas CPU polygons."""
    import numpy as np
    from PIL import ImageDraw
    from material_anchor_review import load_observation
    from cosplay_post_review import _sha, _raster
    observation = load_observation(spec)
    binding = spec['binding']
    ownership = json.loads(Path(binding['feature_ownership_receipt']['path']).read_text(encoding='utf-8-sig'))
    face = ownership['face_observation']
    if (type(face.get('cache_hit')) is not bool
            or face['model_sha256'] != receipt['face_model']['sha256']
            or face['size'] != list(_raster(binding['image']['path']).size)):
        raise ValueError('CPU face observation does not bind this displayed source or selected model')
    identity = face.get('identity', {})
    if any(identity.get(name) != expected for name, expected in {
        'input': binding['image']['sha256'], 'labels': binding['semantic_labels']['sha256'],
        'model': receipt['face_model']['sha256'], 'worker': receipt['face_worker']['sha256']}.items()):
        raise ValueError('CPU face cache identity differs from its actual input/model/worker bytes')
    if identity.get('parameters') != {'num_faces': 1, 'semantic_crop': 'Face_Neck',
            'detection_confidence': .3, 'presence_confidence': .3}:
        raise ValueError('CPU face observation changed the canonical detection/presence thresholds')
    request = json.loads(Path(receipt['request']['path']).read_text(encoding='utf-8-sig'))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    actual_cache_path = Path(request['root']).resolve()/'outputs/cache/face-protection'/key/'result.json'
    cache_record = receipt['face_cache_result']
    if (Path(cache_record['path']).resolve() != actual_cache_path
            or _sha(actual_cache_path) != cache_record['sha256']):
        raise ValueError('CPU source feature observation lacks its actual identity-keyed cached result')
    cached = json.loads(actual_cache_path.read_text(encoding='utf-8-sig'))
    if {**cached, 'cache_hit': face['cache_hit']} != face:
        raise ValueError('CPU source feature polygons differ from the actual cached worker result')
    # A cache hit can retain the first request path. Exact current raster/label
    # bytes, own-canvas size, model/worker identity and actual result remain bound.
    if (not face['cache_hit']
            and Path(face['input_path']).resolve() != Path(binding['image']['path']).resolve()):
        raise ValueError('Fresh CPU landmarks belong to another source observation path')
    names = [row['region'] for row in face['protected_regions']]
    if face['method'] == 'mediapipe_source_eyes_lips':
        if sorted(names) != ['left_eye', 'lips', 'right_eye']:
            raise ValueError('CPU landmarks lack the complete canonical eye/lip topology')
    elif face['method'] != 'no_landmarks_protect_entire_face_neck' or names:
        raise ValueError('Unknown CPU feature observation method or topology')
    masks = {'eyes': np.zeros(observation['labels'].shape, np.float32),
        'lips': np.isin(observation['labels'], [24, 25]).astype(np.float32)}
    for feature in face['protected_regions']:
        polygon = np.asarray(feature['polygon'], np.float64)
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                or not np.isfinite(polygon).all() or polygon.min() < 0
                or (polygon[:, 0] > face['size'][0]).any()
                or (polygon[:, 1] > face['size'][1]).any()):
            raise ValueError('CPU source feature polygon has invalid own-canvas coordinates')
        canvas = Image.new('L', tuple(face['size']))
        ImageDraw.Draw(canvas).polygon([tuple(point) for point in polygon], fill=255)
        name = 'eyes' if feature['region'] in ('left_eye', 'right_eye') else 'lips'
        masks[name] = np.maximum(masks[name], np.asarray(canvas, np.float32)/255)
    if any(not np.array_equal(masks[name], observation['feature_masks'][name]) for name in masks):
        raise ValueError('Source feature mask differs from the actual CPU polygons/semantic lips')
    return face


def validate_saved_preparation(record, spec, *, source_sha256, source_display, semantic_binding):
    """Read-only metadata/file closure validation; never starts CPU landmarks."""
    from cosplay_post_review import _sha
    from current_model_registry import load_registry, PROFILES
    from local_photo_analysis import analysis_identity
    from runtime_resources import code_path
    rows = []
    def read(row):
        if _sha(row['path']) != row['sha256']:
            raise ValueError('Source feature preparation file bytes changed')
        rows.append(row)
        return json.loads(Path(row['path']).read_text(encoding='utf-8-sig'))
    receipt = read(record)
    if (receipt.get('schema') != SCHEMA or receipt.get('status') not in ('prepared', 'unknown')
            or any(type(receipt.get(key)) is not int or receipt.get(key) != 0 for key in ('segmentation_inference_requests',
                'gpu_inference_requests', 'image_generation_requests', 'text_vision_requests'))
            or type(receipt.get('cpu_face_landmark_worker_attempted')) is not bool
            or (receipt.get('cpu_face_landmark_inference_requests') is not None
                and type(receipt.get('cpu_face_landmark_inference_requests')) is not int)
            or receipt.get('cpu_face_landmark_inference_requests') not in (None, 0, 1)):
        raise ValueError('No current source-only CPU feature preparation is evidenced')
    request = read(receipt['request'])
    if (request.get('schema') != 'source_feature_observation_request.v1'
            or request['source']['sha256'] != source_sha256
            or request['source_display']['sha256'] != source_display['sha256']
            or request['source_semantics']['sha256'] != semantic_binding['sha256']
            or request['native_size'] != source_display.get('oriented_size', source_display.get('size'))):
        raise ValueError('CPU feature preparation differs from the frozen source/canonical display/semantics')
    for name, bound in (('source_display', source_display), ('source_semantics', semantic_binding)):
        if Path(request[name]['path']).resolve() != Path(bound['path']).resolve():
            raise ValueError('CPU feature preparation aliases another source display or semantic asset')
    if (request['run']['source']['sha256'] != source_sha256
            or Path(request['source']['path']).resolve() != Path(request['run']['source']['path']).resolve()
            or request['source_semantics']['source_sha256'] != source_sha256
            or request['source_semantics']['size'] != semantic_binding['size']):
        raise ValueError('CPU feature request lacks its original source/semantic coordinates')
    for name in ('source', 'source_display', 'source_semantics'):
        row = request[name]
        if _sha(row['path']) != row['sha256']:
            raise ValueError('CPU feature request input bytes changed')
        rows.append(row)
        if name in receipt and (Path(receipt[name]['path']).resolve() != Path(row['path']).resolve()
                or receipt[name]['sha256'] != row['sha256']):
            raise ValueError('CPU feature receipt differs from its exact request inputs')
    if receipt['status'] == 'unknown':
        if spec is not None or not receipt.get('error') or ('observation' in receipt):
            raise ValueError('Unknown CPU preparation cannot grant source feature evidence')
        if receipt['cpu_face_landmark_inference_requests'] != (None if receipt['cpu_face_landmark_worker_attempted'] else 0):
            raise ValueError('Unknown CPU preparation has inconsistent request accounting')
        for key in ('registry', 'face_model', 'face_worker', 'face_orchestrator', 'face_cache_result', 'observation_producer', 'producer', 'cached_analysis'):
            if key in receipt:
                row = receipt[key]
                if _sha(row['path']) != row['sha256']:
                    raise ValueError('Failed CPU preparation dependency changed: '+key)
                rows.append(row)
        return receipt, rows
    if (receipt.get('cpu_face_landmark_inference_requests') not in (0, 1)
            or receipt['cpu_face_landmark_worker_attempted'] != bool(receipt['cpu_face_landmark_inference_requests'])
            or receipt.get('cpu_face_landmark_cache_hit') != (receipt['cpu_face_landmark_inference_requests'] == 0)
            or Path(receipt['registry']['path']).resolve() != Path(request['run']['options']['model_registry']).resolve()):
        raise ValueError('Successful CPU preparation has inconsistent model/cache accounting')
    registry = load_registry(request['root'], registry_path=receipt['registry']['path'])
    if not registry['ready'] or receipt['registry_profiles'] != PROFILES:
        raise ValueError('Source feature preparation uses another registry profile')
    analysis = read(receipt['cached_analysis'])
    expected = analysis_identity(request['root'], role='source', registry_path=registry['registry_path'])
    from automatic_material_completion import selected_cached_analysis
    if selected_cached_analysis(request['root'], request['run'], role='source', source_sha256=source_sha256) != Path(receipt['cached_analysis']['path']).resolve():
        raise ValueError('CPU feature preparation is not the unique current cached source analysis')
    if (receipt['cached_analysis_identity'] != expected or analysis['analysis_identity'] != expected
            or analysis['source']['sha256'] != source_sha256 or analysis['analysis_role'] != 'source'
            or analysis['source']['oriented_size'] != request['native_size']
            or _sha(analysis['semantic_result']['labels_path']) != semantic_binding['sha256']):
        raise ValueError('Source feature cached analysis identity is not current and source-bound')
    if read(receipt['observation']) != spec:
        raise ValueError('Native feature spec differs from the actual CPU prepared source observation')
    if (spec['role'] != 'source' or spec['domain'] != 'display_linear_sRGB'
            or spec['binding']['identity_root']['sha256'] != source_sha256
            or spec['binding']['display_parent']['sha256'] != source_display['sha256']
            or spec['binding']['analysis']['sha256'] != receipt['cached_analysis']['sha256']):
        raise ValueError('CPU feature spec lacks its exact canonical source parent')
    for key in ('source', 'source_display', 'source_semantics', 'registry', 'face_model', 'face_worker', 'face_orchestrator', 'face_cache_result', 'observation_producer', 'producer'):
        row = receipt[key]
        if _sha(row['path']) != row['sha256']:
            raise ValueError('Source feature preparation dependency bytes changed: '+key)
        rows.append(row)
    if (receipt['face_model']['sha256'] != _sha(registry['face_model_path'])
            or receipt['face_worker']['sha256'] != _sha(code_path('local_face_protection_worker.py'))
            or receipt['face_orchestrator']['sha256'] != _sha(code_path('local_face_protection.py'))
            or receipt['observation_producer']['sha256'] != _sha(code_path('material_anchor_review.py'))
            or receipt['producer']['sha256'] != _sha(code_path('source_feature_observation_worker.py'))):
        raise ValueError('Source feature preparation belongs to another CPU model or producer')
    face = _validate_cpu_observation(spec, receipt)
    if receipt['landmark_status'] != ('available' if face['method'] == 'mediapipe_source_eyes_lips' else 'unknown'):
        raise ValueError('CPU landmark availability differs from the actual source observation')
    for row in spec['binding'].values():
        if isinstance(row, dict) and row.get('path') and row.get('sha256'):
            rows.append(row)
    rows.extend(spec['binding']['feature_masks'].values())
    return receipt, rows


def prepare(request_path, output):
    from material_anchor_review import file_record, prepare_observation, write
    from current_model_registry import load_registry, PROFILES
    from automatic_material_completion import selected_cached_analysis
    from local_photo_analysis import analysis_identity
    from runtime_resources import code_path
    from cosplay_post_review import _sha, _raster
    output = Path(output).resolve()
    if (output/'preparation.json').exists():
        raise FileExistsError('Source feature preparation is immutable')
    output.mkdir(parents=True, exist_ok=True)
    request = json.loads(Path(request_path).read_text(encoding='utf-8-sig'))
    receipt = {'schema': SCHEMA, 'status': 'unknown', 'request': file_record(request_path),
        'cpu_face_landmark_worker_attempted': False, 'cpu_face_landmark_inference_requests': 0,
        'segmentation_inference_requests': 0, 'gpu_inference_requests': 0,
        'image_generation_requests': 0, 'text_vision_requests': 0,
        'visual_acceptance': False, 'feature_pigment_classification': False}
    try:
        if request.get('schema') != 'source_feature_observation_request.v1':
            raise ValueError('Unknown source feature observation request')
        root, run = Path(request['root']).resolve(), request['run']
        registry = load_registry(root, registry_path=run.get('options', {}).get('model_registry'))
        if not registry['ready']:
            raise ValueError('Selected current registry is not ready')
        source, displayed, semantic = [request[name] for name in ('source', 'source_display', 'source_semantics')]
        for row in (source, displayed, semantic):
            if _sha(row['path']) != row['sha256']:
                raise ValueError('Bound source feature input bytes changed')
        if (source['sha256'] != run['source']['sha256'] or semantic['source_sha256'] != source['sha256']
                or Path(source['path']).resolve() != Path(run['source']['path']).resolve()):
            raise ValueError('Source feature observation belongs to a different source')
        image = _raster(displayed['path'])
        if list(image.size) != request['native_size']:
            raise ValueError('Canonical source display is not the oriented source canvas')
        analysis_path = selected_cached_analysis(root, run, role='source', source_sha256=source['sha256'])
        analysis = json.loads(analysis_path.read_text(encoding='utf-8-sig'))
        expected = analysis_identity(root, role='source', registry_path=registry['registry_path'])
        if (analysis['analysis_identity'] != expected or analysis['analysis_role'] != 'source'
                or analysis['source']['oriented_size'] != request['native_size']
                or _sha(analysis['semantic_result']['labels_path']) != semantic['sha256']):
            raise ValueError('Current cached source semantics differ from the frozen native source binding')
        import numpy as np
        with np.load(semantic['path'], allow_pickle=False) as resource:
            if list(resource['labels'].shape[::-1]) != semantic['size']:
                raise ValueError('Source semantic proxy coordinates differ from the native selection')
        receipt.update(source=file_record(source['path']), source_display=file_record(displayed['path']),
            source_semantics=file_record(semantic['path']), native_size=request['native_size'],
            cached_analysis=file_record(analysis_path), cached_analysis_identity=expected,
            registry=file_record(registry['registry_path']), registry_profiles=PROFILES,
            face_model=file_record(registry['face_model_path']),
            face_worker=file_record(code_path('local_face_protection_worker.py')),
            face_orchestrator=file_record(code_path('local_face_protection.py')),
            observation_producer=file_record(code_path('material_anchor_review.py')),
            producer=file_record(Path(__file__)), cpu_face_landmark_worker_attempted=True)
        spec, _ = prepare_observation(root, 'source', displayed['path'], source['path'], analysis_path,
            output/'observation', registry_path=registry['registry_path'])
        owned = json.loads(Path(spec['binding']['feature_ownership_receipt']['path']).read_text(encoding='utf-8-sig'))
        cache_identity = owned['face_observation']['identity']
        cache_key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:24]
        receipt['face_cache_result'] = file_record(root/'outputs/cache/face-protection'/cache_key/'result.json')
        face = _validate_cpu_observation(spec, receipt)
        for row in (source, displayed, semantic):
            if _sha(row['path']) != row['sha256']:
                raise ValueError('Source feature inputs changed during CPU observation')
        receipt.update(status='prepared', observation=file_record(output/'observation/observation.json'),
            cpu_face_landmark_worker_attempted=face.get('cache_hit') is not True,
            cpu_face_landmark_cache_hit=face.get('cache_hit') is True,
            cpu_face_landmark_inference_requests=0 if face.get('cache_hit') else 1,
            landmark_status='available' if face['method'] == 'mediapipe_source_eyes_lips' else 'unknown',
            scope='source_feature_location_only; no iris pigment or makeup qualification')
    except Exception as exc:
        receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)[:600]}
        if receipt['cpu_face_landmark_worker_attempted']:
            receipt['cpu_face_landmark_inference_requests'] = None
    write(output/'preparation.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.request, args.output)
    print(json.dumps({'status': result['status'], 'landmark_status': result.get('landmark_status'),
        'segmentation_requests': 0, 'gpu_requests': 0, 'text_vision_requests': 0}))
