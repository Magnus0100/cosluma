"""Source-only proposals and evidence-gated native healing with frozen replay.

The detector proposes compact luminance outliers; it does not identify acne,
moles or lint. Two source-crop vision receipts must approve the defect and a
same-material donor before the existing log-linear healing brush is consumed.
No reference RGB, generated texture or low-frequency colour field is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

from clone_retouch import apply_clone_retouch, empty_clone_layer
from raw_support import RGB_DECODE_VERSION, linear_preview, open_linear


RECIPE = 'source-native-compact-outlier-two-crop-approval-v1'
CLASS_NAMES = [
    'Background', 'Apparel', 'Eyeglass', 'Face_Neck', 'Hair', 'Left_Foot',
    'Left_Hand', 'Left_Lower_Arm', 'Left_Lower_Leg', 'Left_Shoe', 'Left_Sock',
    'Left_Upper_Arm', 'Left_Upper_Leg', 'Lower_Clothing', 'Right_Foot',
    'Right_Hand', 'Right_Lower_Arm', 'Right_Lower_Leg', 'Right_Shoe',
    'Right_Sock', 'Right_Upper_Arm', 'Right_Upper_Leg', 'Torso',
    'Upper_Clothing', 'Lower_Lip', 'Upper_Lip', 'Lower_Teeth', 'Upper_Teeth', 'Tongue',
]
SKIN_CLASSES = (3, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 20, 21, 22)
CLOTH_CLASSES = (1, 13, 23)
FEATURE_CHECKS = ('eye_or_eye_makeup', 'contact_lens', 'lip_or_lip_makeup',
                  'tattoo_or_mole', 'hair_or_seam', 'low_frequency_colour_cast')
PARAMETERS = {
    'scale_native_short_side_reference': 2048,
    'sigma_multipliers': [1.0, 2.0, 4.0],
    'sigma_floor_px': 1.0, 'outlier_mad_multiplier': 6.0,
    'minimum_luminance_residual': .024, 'minimum_component_pixels': 3,
    'maximum_component_sigma_extent': 5.0, 'maximum_aspect_ratio': 4.0,
    'semantic_border_sigma_margin': 3.0, 'tile_size': 512,
    'context_minimum_px': 128, 'context_maximum_px': 768, 'context_radius_multiple': 12.0,
    'maximum_proposals': 24, 'donor_search_radius_multipliers': [4, 6, 9, 13, 18],
    'donor_angles': 24, 'maximum_brightness_difference_ev': .32,
    'maximum_log_chroma_difference': .12,
    'maximum_texture_scale_ratio': 2.0, 'minimum_vision_confidence': .92,
    'maximum_delta_fraction_of_source_luminance': .5,
}
VISION_SUFFIX = ('\nReturn only one JSON object, without markdown. Follow the requested JSON '
                 'schema and its uncertainty conventions. Do not guess unsupported identities.')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(values):
    value = np.asarray(values, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(value.shape), separators=(',', ':')).encode())
    for top in range(0, value.shape[0], 256):
        digest.update(np.ascontiguousarray(value[top:top+256]).tobytes())
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _write(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def _record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def _box(cx, cy, radius, width, height):
    return [max(0, int(math.floor(cx-radius))), max(0, int(math.floor(cy-radius))),
            min(width, int(math.ceil(cx+radius))+1), min(height, int(math.ceil(cy+radius))+1)]


def _slice(box):
    left, top, right, bottom = map(int, box)
    return slice(top, bottom), slice(left, right)


def _stable_srgb_profile():
    from raw_support import deterministic_srgb_profile
    return deterministic_srgb_profile()


def _crop_record(image, box, path):
    crop = image.crop(tuple(box))
    crop.save(path, 'PNG', icc_profile=_stable_srgb_profile())
    return {**_record(path), 'native_box': list(box), 'size': list(crop.size),
            'resampled': False, 'source_only': True,
            'rgb_sha256': hashlib.sha256(np.asarray(crop).tobytes()).hexdigest()}


def _validate_arrays(linear, labels, protection):
    linear = np.asarray(linear, np.float32)
    labels = np.asarray(labels, np.uint8)
    protection = np.asarray(protection, np.float32)
    if linear.ndim != 3 or linear.shape[2] != 3 or labels.shape != linear.shape[:2] or protection.shape != labels.shape:
        raise ValueError('native linear RGB, semantic labels and protection must share the native canvas')
    if not np.isfinite(linear).all() or np.any(linear < 0) or not np.isfinite(protection).all():
        raise ValueError('source linear RGB and protection must be finite and nonnegative')
    if labels.max(initial=0) >= 29 or np.any((protection < 0) | (protection > 1)):
        raise ValueError('unsupported semantic taxonomy or invalid protection')
    return linear, labels, protection


def propose_candidates(linear, labels, protection, *, maximum=None):
    """Return deterministic compact source-luminance proposals, never approvals.

    Finite filter halos and global native scale keep tiled detection independent
    of tile boundaries. Strong high-frequency texture and all semantic/protected
    boundaries are excluded; arbitrary high-frequency marks still need vision.
    """
    linear, labels, protection = _validate_arrays(linear, labels, protection)
    h, w = labels.shape
    scale = max(PARAMETERS['sigma_floor_px'], min(w, h)/PARAMETERS['scale_native_short_side_reference'])
    sigmas = np.asarray(PARAMETERS['sigma_multipliers'])*scale
    # Noise estimation filters an already filtered residual: their truncated
    # supports add, so each tile must include both halos.
    halo = int(math.ceil(sigmas.max()*(1.8*4+7*4)))+8
    tile = PARAMETERS['tile_size']
    # Native display Y detects compact blemishes without treating chromatic cast
    # as an instruction to clone. RAW16 is retained for subsequent healing.
    display = np.asarray(linear_preview(linear), np.float32)/255
    luminance = display[..., 0]*.2126+display[..., 1]*.7152+display[..., 2]*.0722
    protection_distance = distance_transform_edt(protection <= .01).astype(np.float32)
    candidates = []
    classes = sorted(set(np.unique(labels).tolist()).intersection(SKIN_CLASSES+CLOTH_CLASSES))
    for material in classes:
        interior = distance_transform_edt(labels == material).astype(np.float32)
        valid = (interior > sigmas.max()*PARAMETERS['semantic_border_sigma_margin']) & (protection_distance > sigmas.max()*3)
        if not valid.any():
            continue
        score = np.zeros((h, w), np.float32)
        scale_index = np.zeros((h, w), np.uint8)
        for top in range(0, h, tile):
            for left in range(0, w, tile):
                right, bottom = min(w, left+tile), min(h, top+tile)
                if not valid[top:bottom, left:right].any():
                    continue
                ext = [max(0, left-halo), max(0, top-halo), min(w, right+halo), min(h, bottom+halo)]
                patch = luminance[_slice(ext)]
                core = (slice(top-ext[1], bottom-ext[1]), slice(left-ext[0], right-ext[0]))
                for index, sigma in enumerate(sigmas):
                    residual = np.abs(gaussian_filter(patch, sigma*.45, truncate=4)-gaussian_filter(patch, sigma*1.8, truncate=4))
                    robust_noise = gaussian_filter(np.minimum(residual, .025), sigma*7, truncate=4)*1.4826
                    threshold = np.maximum(PARAMETERS['minimum_luminance_residual'], robust_noise*PARAMETERS['outlier_mad_multiplier'])
                    local = residual[core]/threshold[core]
                    chosen = local > score[top:bottom, left:right]
                    score[top:bottom, left:right][chosen] = local[chosen]
                    scale_index[top:bottom, left:right][chosen] = index
        blobs, count = label((score > 1) & valid)
        # Connected component slices avoid scanning an entire image per mark.
        from scipy.ndimage import find_objects
        for index, sl in enumerate(find_objects(blobs), 1):
            if sl is None:
                continue
            ys, xs = np.nonzero(blobs[sl] == index)
            if len(xs) < PARAMETERS['minimum_component_pixels']:
                continue
            xs, ys = xs+sl[1].start, ys+sl[0].start
            best = int(np.argmax(score[ys, xs])); x, y = int(xs[best]), int(ys[best])
            sigma = float(sigmas[int(scale_index[y, x])])
            extent_x, extent_y = int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
            if max(extent_x, extent_y) > sigma*PARAMETERS['maximum_component_sigma_extent']:
                continue
            if max(extent_x, extent_y)/max(1, min(extent_x, extent_y)) > PARAMETERS['maximum_aspect_ratio']:
                continue
            cx, cy = float((xs.max()+xs.min())/2), float((ys.max()+ys.min())/2)
            radius = max(2., float(np.max(np.hypot(xs-cx, ys-cy)))+sigma*.6)
            if interior[y, x] <= radius*1.4 or protection_distance[y, x] <= radius*1.4:
                continue
            candidate_id = hashlib.sha256(json.dumps([material, cx, cy, radius], separators=(',', ':')).encode()).hexdigest()[:16]
            candidates.append({'id': candidate_id, 'status': 'unapproved', 'center_native': [cx, cy],
                'radius_px': radius, 'native_box': _box(cx, cy, radius, w, h), 'sigma_px': sigma,
                'material_class': material, 'material_name': CLASS_NAMES[material],
                'material_family': 'skin' if material in SKIN_CLASSES else 'cloth',
                'outlier_score': float(score[y, x]), 'component_pixels': len(xs),
                'detection_only': True, 'defect_type': 'unknown'})
    candidates.sort(key=lambda x: (-x['outlier_score'], x['material_class'], x['center_native'][1], x['center_native'][0]))
    deduplicated = []
    for candidate in candidates:
        center = np.asarray(candidate['center_native'])
        if any(np.linalg.norm(center-np.asarray(old['center_native'])) < 3*(candidate['radius_px']+old['radius_px']) for old in deduplicated):
            continue
        deduplicated.append(candidate)
    # Metal/garment details often have the largest contrast. A fixed budget
    # must still inspect face and body proposals instead of spending all calls
    # on bright costume buttons. This changes proposal ordering, never approval.
    groups = [[], [], []]
    for candidate in deduplicated:
        group = 0 if candidate['material_class'] == 3 else 1 if candidate['material_family'] == 'skin' else 2
        groups[group].append(candidate)
    accepted = []
    for rank in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if rank < len(group):
                accepted.append(group[rank])
                if len(accepted) >= int(maximum or PARAMETERS['maximum_proposals']):
                    return accepted
    return accepted


def _candidate_prompt(candidate):
    box = candidate['target_crop']['native_box']
    target = candidate['native_box']
    local = [target[0]-box[0], target[1]-box[1], target[2]-box[0], target[3]-box[1]]
    checks = ', '.join('"'+key+'": false' for key in FEATURE_CHECKS)
    return (
        'Review a native camera-source crop for conservative Cos photo retouching. '
        'This is recognition only. No generated or reference pixels are shown. '
        f'Candidate ID {candidate["id"]}; image size {candidate["target_crop"]["size"]}; '
        f'candidate bounds [left,top,right,bottom] in this crop are {local}; '
        f'parser material is {candidate["material_name"]} (a proposal, not proof). '
        'Approve ONLY an unambiguous temporary small surface blemish on skin or lint/debris on cloth. '
        'Keep freckles, moles, tattoos, eye makeup, coloured contacts, lip makeup, hair, cloth seams and deliberate costume detail. '
        'Broad green/red colour casts and broad shadows are not defects for this operation. '
        'A pore, texture, dark point or coloured point alone does not prove temporary status. '
        'If permanence or makeup intent cannot be determined from this crop, return unknown. '
        'No global colour or skin smoothing. Return exactly this schema: '
        '{"schema_version":"native_retouch_candidate_approval.v1",'
        f'"candidate_id":"{candidate["id"]}",'
        '"decision":"approve|reject|unknown","defect_type":"temporary_surface_blemish|lint|unknown",'
        '"confidence":0.0,"target_location_verified":false,'
        f'"protected_feature_checks":{{{checks}}},"reason":"specific visual evidence"}}'
    )


def prepare_proposals(source_path, analysis_path, out_dir, *, face_protection_path=None, maximum=None):
    """Decode native source, bind existing source semantics, and write VLM jobs.

    Missing eye/lip landmarks protect the entire face; they cannot enable a
    guessed face repair. Semantic proxy is used only as a material eligibility
    field, never as the retouched RGB or as a native-detail measurement.
    """
    source_path, analysis_path, out_dir = map(lambda p: Path(p).resolve(), (source_path, analysis_path, out_dir))
    if (out_dir/'proposals.json').exists():
        raise FileExistsError('proposal manifest already exists; preserve its evidence and use a new directory')
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis = _json(analysis_path)
    source_record = _record(source_path)
    if analysis.get('analysis_role') != 'source' or analysis.get('source', {}).get('sha256') != source_record['sha256']:
        raise ValueError('retouch must use analysis bound to the actual camera source')
    if analysis.get('analysis_identity', {}).get('decode_version') != RGB_DECODE_VERSION:
        raise ValueError('source analysis decode identity is obsolete')
    semantic = analysis['semantic_result']
    if semantic.get('class_names') != CLASS_NAMES:
        raise ValueError('retouch requires the complete pinned 29-class semantic taxonomy')
    linear, decode = open_linear(source_path)
    h, w = linear.shape[:2]
    if [w, h] != analysis['source'].get('oriented_size'):
        raise ValueError('native decode differs from source analysis canvas')
    with np.load(semantic['labels_path']) as asset:
        proxy_labels = asset['labels']
    labels = np.asarray(Image.fromarray(proxy_labels.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST))
    protection = np.zeros((h, w), np.float32)
    face_evidence = {'method': 'missing_landmarks_protect_entire_face_neck'}
    if face_protection_path:
        face_path = Path(face_protection_path).resolve()
        face = _json(face_path)
        proxy = Path(analysis['analysis_directory'])/'source-srgb.png'
        if Path(face.get('input_path', '')).resolve() != proxy.resolve() or face.get('identity', {}).get('input') != sha256(proxy):
            raise ValueError('face protection is not bound to this exact source display proxy')
        if face.get('method') != 'mediapipe_source_eyes_lips':
            protection[labels == 3] = 1
        else:
            with np.load(face['field_path']) as asset:
                field = asset['protection']
            if list(field.shape[::-1]) != face.get('size'):
                raise ValueError('face protection canvas identity is invalid')
            protection = np.array(Image.fromarray(field.astype(np.float32), 'F').resize((w, h), Image.Resampling.BILINEAR))
        face_evidence = {**_record(face_path), 'field': _record(face['field_path']), 'method': face['method']}
    else:
        protection[labels == 3] = 1
    # Lip, teeth, hair and glasses remain excluded by taxonomy even when face
    # detection succeeds. Strong native edges need explicit vision rejection.
    protection[np.isin(labels, (2, 4, 24, 25, 26, 27, 28))] = 1
    eligibility_path = out_dir/'native-eligibility.npz'
    np.savez_compressed(eligibility_path, labels=labels, protection=protection)
    source_record.update(native_size=[w, h], linear_rgb_sha256=array_sha(linear), decode=decode,
                         coordinate_space='oriented_native_source')
    proposals = propose_candidates(linear, labels, protection, maximum=maximum)
    image = linear_preview(linear)
    requests = []
    for index, candidate in enumerate(proposals):
        directory = out_dir/'candidates'/candidate['id']
        directory.mkdir(parents=True, exist_ok=True)
        radius = min(PARAMETERS['context_maximum_px']/2,
                     max(PARAMETERS['context_minimum_px']/2, candidate['radius_px']*PARAMETERS['context_radius_multiple']))
        candidate['target_crop'] = _crop_record(image, _box(*candidate['center_native'], radius, w, h), directory/'source-context.png')
        prompt = _candidate_prompt(candidate)
        prompt_path = directory/'candidate-request.txt'
        prompt_path.write_text(prompt, encoding='utf8')
        candidate['request'] = {'source': candidate['target_crop']['path'], 'prompt_path': str(prompt_path),
            'expected_prompt_sha256': hashlib.sha256((prompt+VISION_SUFFIX).encode()).hexdigest(),
            'out_dir': str(directory/'candidate-vision'), 'approval_path': str(directory/'candidate-approval.json')}
        requests.append({'candidate_id': candidate['id'], **candidate['request']})
    manifest = {'schema_version': 'native_retouch_proposals.v1', 'recipe': RECIPE, 'parameters': PARAMETERS,
        'source': source_record, 'analysis': _record(analysis_path),
        'semantic_labels': _record(semantic['labels_path']), 'face_protection': face_evidence,
        'eligibility': _record(eligibility_path), 'producer_sha256': sha256(__file__),
        'candidates': proposals, 'vision_requests': requests, 'approved_count': 0,
        'source_rgb_changed': False, 'generation_used': False,
        'contract': 'source-only compact proposals; candidate AND donor evidence required before healing'}
    _write(out_dir/'proposals.json', manifest)
    return manifest


def _validate_file(record, label):
    path = Path(record['path'])
    if not path.is_file() or sha256(path) != record.get('sha256'):
        raise ValueError(label+' bytes differ from frozen evidence')
    return path


def _verify_source_crop(record, image):
    path = _validate_file(record, 'native source crop')
    with Image.open(path) as crop:
        actual = np.asarray(crop.convert('RGB'))
    expected = np.asarray(image.crop(tuple(record['native_box'])))
    if record.get('resampled') is not False or not np.array_equal(actual, expected):
        raise ValueError('vision crop does not reproduce from the exact native camera-source pixels')
    if list(expected.shape[1::-1]) != record.get('size') or hashlib.sha256(expected.tobytes()).hexdigest() != record.get('rgb_sha256'):
        raise ValueError('native crop pixel/size identity is invalid')


def _verify_source_pair(record, image):
    _verify_source_crop(record['target_crop'], image)
    _verify_source_crop(record['donor_crop'], image)
    path = _validate_file(record, 'native donor comparison')
    with Image.open(path) as pair:
        actual = np.asarray(pair.convert('RGB'))
    if list(actual.shape[1::-1]) != record['size'] or record.get('resampled') is not False:
        raise ValueError('donor comparison canvas identity is invalid')
    for key, origin in (('target_crop', record['target_panel_origin']), ('donor_crop', record['donor_panel_origin'])):
        with Image.open(record[key]['path']) as panel:
            expected = np.asarray(panel.convert('RGB'))
        x, y = origin
        if not np.array_equal(actual[y:y+expected.shape[0], x:x+expected.shape[1]], expected):
            raise ValueError('actual donor-comparison panel differs from its native source crop')


def validate_vision_receipt(approval, request, image_record):
    """Bind a response to the exact native crop and actual API raster receipt."""
    payload, provenance = approval.get('response'), approval.get('provenance')
    if not isinstance(payload, dict) or not isinstance(provenance, dict):
        raise ValueError('vision approval requires response and real transport provenance')
    if provenance.get('schema_version') != 'vision_transport.v1' or provenance.get('network_used') is not True or provenance.get('generation_used') is not False:
        raise ValueError('only actual recognition transport evidence can approve retouch')
    if not provenance.get('response_id') or not provenance.get('model') or not provenance.get('endpoint'):
        raise ValueError('vision response identity is missing')
    path = _validate_file(image_record, 'source crop')
    source = provenance.get('source', {})
    if Path(source.get('path', '')).resolve() != path.resolve() or source.get('sha256') != image_record['sha256']:
        raise ValueError('vision approval viewed a different crop')
    if provenance.get('prompt_sha256') != request['expected_prompt_sha256']:
        raise ValueError('vision approval used a different recognition contract')
    actual_prompt = Path(request['prompt_path']).read_text(encoding='utf8')
    if hashlib.sha256((actual_prompt+VISION_SUFFIX).encode()).hexdigest() != request['expected_prompt_sha256']:
        raise ValueError('recognition prompt changed after proposal preparation')
    api = provenance.get('api_input', {})
    _validate_file(api, 'actual API input')
    if provenance.get('input_sha256') != api.get('sha256') or api.get('source', {}).get('sha256') != image_record['sha256']:
        raise ValueError('actual API raster was not derived from the frozen native crop')
    if list(Image.open(path).size) != image_record['size']:
        raise ValueError('frozen native crop size differs from its receipt')
    if max(image_record['size']) > 1024 or api.get('size') != image_record['size']:
        raise ValueError('native recognition crop must reach the API without spatial resampling')
    return payload


def _approved_candidate(candidate, approval):
    payload = validate_vision_receipt(approval, candidate['request'], candidate['target_crop'])
    if payload.get('schema_version') != 'native_retouch_candidate_approval.v1' or payload.get('candidate_id') != candidate['id']:
        raise ValueError('candidate response schema/identity mismatch')
    if payload.get('decision') != 'approve':
        return False
    required_type = 'temporary_surface_blemish' if candidate['material_family'] == 'skin' else 'lint'
    checks = payload.get('protected_feature_checks', {})
    if payload.get('defect_type') != required_type or payload.get('target_location_verified') is not True:
        return False
    if not _qualified_confidence(payload.get('confidence')):
        return False
    if not all(checks.get(key) is False for key in FEATURE_CHECKS):
        return False
    return isinstance(payload.get('reason'), str) and bool(payload['reason'].strip())


def _qualified_confidence(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and PARAMETERS['minimum_vision_confidence'] <= value <= 1)


def _validate_recipe(manifest):
    if manifest.get('schema_version') != 'native_retouch_proposals.v1' or manifest.get('recipe') != RECIPE or manifest.get('parameters') != PARAMETERS:
        raise ValueError('retouch proposal algorithm/parameter identity differs from the approved recipe')


def _patch_metrics(linear, center, radius):
    h, w = linear.shape[:2]
    box = _box(*center, radius, w, h)
    patch = linear[_slice(box)]
    log_rgb = np.log(np.maximum(patch, 1e-6))
    yy, xx = np.mgrid[box[1]:box[3], box[0]:box[2]]
    distance = np.hypot(xx-center[0], yy-center[1])
    annulus = (distance > radius*.7) & (distance < radius)
    if annulus.sum() < 8:
        return None
    mean = np.median(log_rgb[annulus], axis=0)
    luminance = patch[..., 0]*.2126+patch[..., 1]*.7152+patch[..., 2]*.0722
    log_y = np.log(np.maximum(luminance, 1e-6))
    # Ratio of fine/coarse contrast captures texture scale independently of
    # absolute brightness; matching does not invent a pore field.
    texture_sigma = max(.65, radius*.07)
    fine = log_y-gaussian_filter(log_y, texture_sigma)
    coarse = gaussian_filter(log_y, texture_sigma)-gaussian_filter(log_y, texture_sigma*2.4)
    texture_scale = (float(np.sqrt(np.mean(fine[annulus]**2)))+1e-4)/(float(np.sqrt(np.mean(coarse[annulus]**2)))+1e-4)
    return {'log_rgb': mean.tolist(), 'log_luminance': float(np.median(np.log(np.maximum(luminance[annulus], 1e-6)))),
            'texture_scale': texture_scale, 'box': box}


def select_donor(linear, labels, protection, candidate, all_candidates):
    """Choose a photographed native same-class donor with matched light/scale.

    Semantic identity is necessary but does not prove the same material. The
    returned proposal requires a second source-crop visual approval.
    """
    linear, labels, protection = _validate_arrays(linear, labels, protection)
    h, w = labels.shape
    center = np.asarray(candidate['center_native'], np.float64)
    radius = float(candidate['radius_px'])
    margin = radius*1.5+2
    safe = (labels == candidate['material_class']) & (protection <= .01)
    for proposed in all_candidates:
        box = _box(*proposed['center_native'], proposed['radius_px']*1.75+2, w, h)
        safe[_slice(box)] = False
    safe_distance = distance_transform_edt(safe)
    target = _patch_metrics(linear, center, max(radius*2.5, 6.))
    if target is None:
        return None
    choices = []
    for multiplier in PARAMETERS['donor_search_radius_multipliers']:
        for angle in np.linspace(0, 2*np.pi, PARAMETERS['donor_angles'], endpoint=False):
            point = np.rint(center+max(radius, 3.)*multiplier*np.asarray([np.cos(angle), np.sin(angle)])).astype(int)
            x, y = map(int, point)
            if x < 0 or x >= w or y < 0 or y >= h or safe_distance[y, x] <= margin:
                continue
            metrics = _patch_metrics(linear, point, max(radius*2.5, 6.))
            if metrics is None:
                continue
            brightness = abs(metrics['log_luminance']-target['log_luminance'])/math.log(2)
            target_rgb, donor_rgb = np.asarray(target['log_rgb']), np.asarray(metrics['log_rgb'])
            chroma = float(np.max(np.abs((target_rgb-target_rgb.mean())-(donor_rgb-donor_rgb.mean()))))
            texture_ratio = max(metrics['texture_scale']/target['texture_scale'], target['texture_scale']/metrics['texture_scale'])
            if brightness > PARAMETERS['maximum_brightness_difference_ev'] or chroma > PARAMETERS['maximum_log_chroma_difference'] or texture_ratio > PARAMETERS['maximum_texture_scale_ratio']:
                continue
            score = brightness*4+chroma*6+abs(math.log(texture_ratio))+np.linalg.norm(point-center)/max(radius, 1)*.003
            choices.append((score, y, x, brightness, chroma, texture_ratio, metrics))
    if not choices:
        return None
    best = min(choices, key=lambda item: item[:3])
    return {'center_native': [best[2], best[1]], 'radius_px': radius,
            'material_class': candidate['material_class'], 'material_name': candidate['material_name'],
            'matching_score': float(best[0]), 'brightness_difference_ev': float(best[3]),
            'log_chroma_difference': float(best[4]), 'texture_scale_ratio': float(best[5]),
            'target_metrics': target, 'donor_metrics': best[6],
            'selection': 'same-source same-native-scale same-exact-semantic-class; visual same-material approval pending'}


def _donor_prompt(candidate, donor):
    return (
        'Review two camera-source native crops for conservative Cos photo healing. '
        'The left panel is the target; the right panel is a proposed photographed donor. '
        'Neither panel has been generated or resized. Each proposed region is at the exact center of its panel. '
        f'Candidate {candidate["id"]}; native healing radius {candidate["radius_px"]:.3f} pixels; '
        f'parser class {candidate["material_name"]}. '
        'Approve only if the donor region is an unblemished instance of the EXACT SAME photographed surface/material, '
        'with the same apparent texture scale and comparable light/shading. Skin from a different body area, '
        'different garment panels, material changes, moles, tattoo, makeup, hairs, seams or costume details are not valid donors. '
        'If you cannot establish this, return unknown. Return exactly this schema: '
        '{"schema_version":"native_retouch_donor_approval.v1",'
        f'"candidate_id":"{candidate["id"]}",'
        '"decision":"approve|reject|unknown","confidence":0.0,"same_material":false,'
        '"same_surface_scale":false,"same_illumination":false,'
        '"donor_has_no_protected_or_undesired_feature":false,"reason":"specific visual evidence"}'
    )


def prepare_donors(proposals_path, out_dir=None):
    """Consume candidate vision receipts and prepare the second recognition jobs."""
    proposals_path = Path(proposals_path).resolve()
    manifest = _json(proposals_path)
    _validate_recipe(manifest)
    _validate_file(manifest['source'], 'camera source')
    _validate_file(manifest['eligibility'], 'native eligibility')
    out_dir = Path(out_dir).resolve() if out_dir else proposals_path.parent
    if (out_dir/'donors.json').exists():
        raise FileExistsError('donor manifest already exists; do not overwrite recognition evidence')
    linear, _ = open_linear(Path(manifest['source']['path']))
    if array_sha(linear) != manifest['source']['linear_rgb_sha256']:
        raise ValueError('camera-source native decode changed')
    with np.load(manifest['eligibility']['path']) as asset:
        labels, protection = asset['labels'], asset['protection']
    image = linear_preview(linear)
    donors, statuses = [], []
    for candidate in manifest['candidates']:
        _verify_source_crop(candidate['target_crop'], image)
        approval_path = Path(candidate['request']['approval_path'])
        if not approval_path.is_file():
            statuses.append({'candidate_id': candidate['id'], 'status': 'no_candidate_approval'})
            continue
        approval = _json(approval_path)
        if not _approved_candidate(candidate, approval):
            statuses.append({'candidate_id': candidate['id'], 'status': 'candidate_not_approved'})
            continue
        donor = select_donor(linear, labels, protection, candidate, manifest['candidates'])
        if donor is None:
            statuses.append({'candidate_id': candidate['id'], 'status': 'no_qualified_photographed_donor'})
            continue
        directory = out_dir/'donors'/candidate['id']
        directory.mkdir(parents=True, exist_ok=True)
        # Symmetric small native panels guarantee no VLM input downsampling.
        radius = max(32., min(192., candidate['radius_px']*9))
        h, w = labels.shape
        target_box = _box(*candidate['center_native'], radius, w, h)
        donor_box = _box(*donor['center_native'], radius, w, h)
        target_record = _crop_record(image, target_box, directory/'target.png')
        donor_record = _crop_record(image, donor_box, directory/'donor.png')
        target_crop, donor_crop = image.crop(tuple(target_box)), image.crop(tuple(donor_box))
        gap, margin = 12, 24
        pair = Image.new('RGB', (target_crop.width+donor_crop.width+gap, max(target_crop.height, donor_crop.height)+margin), '#171717')
        pair.paste(target_crop, (0, margin)); pair.paste(donor_crop, (target_crop.width+gap, margin))
        draw = ImageDraw.Draw(pair); draw.text((4, 4), 'native source target', fill='white'); draw.text((target_crop.width+gap+4, 4), 'native source donor', fill='white')
        pair_path = directory/'source-pair.png'
        pair.save(pair_path, 'PNG', icc_profile=_stable_srgb_profile())
        pair_record = {**_record(pair_path), 'size': list(pair.size), 'source_only': True, 'resampled': False,
                       'target_crop': target_record, 'donor_crop': donor_record,
                       'target_panel_origin': [0, margin], 'donor_panel_origin': [target_crop.width+gap, margin]}
        prompt = _donor_prompt(candidate, donor)
        prompt_path = directory/'donor-request.txt'
        prompt_path.write_text(prompt, encoding='utf8')
        request = {'source': str(pair_path), 'prompt_path': str(prompt_path),
                   'expected_prompt_sha256': hashlib.sha256((prompt+VISION_SUFFIX).encode()).hexdigest(),
                   'out_dir': str(directory/'donor-vision'), 'approval_path': str(directory/'donor-approval.json')}
        donors.append({'candidate_id': candidate['id'], 'candidate_approval': _record(approval_path),
                       'donor': donor, 'source_pair': pair_record, 'request': request})
        statuses.append({'candidate_id': candidate['id'], 'status': 'donor_visual_approval_pending'})
    result = {'schema_version': 'native_retouch_donors.v1', 'recipe': RECIPE,
              'proposals': _record(proposals_path), 'source': manifest['source'],
              'donors': donors, 'statuses': statuses,
              'vision_requests': [{'candidate_id': d['candidate_id'], **d['request']} for d in donors],
              'source_rgb_changed': False, 'generation_used': False}
    _write(out_dir/'donors.json', result)
    return result


def _approved_donor(donor_record, approval):
    payload = validate_vision_receipt(approval, donor_record['request'], donor_record['source_pair'])
    if payload.get('schema_version') != 'native_retouch_donor_approval.v1' or payload.get('candidate_id') != donor_record['candidate_id']:
        raise ValueError('donor response schema/identity mismatch')
    return (payload.get('decision') == 'approve'
            and _qualified_confidence(payload.get('confidence'))
            and all(payload.get(key) is True for key in ('same_material', 'same_surface_scale',
                'same_illumination', 'donor_has_no_protected_or_undesired_feature'))
            and isinstance(payload.get('reason'), str) and bool(payload['reason'].strip()))


def stroke_for(candidate, donor, native_size):
    w, h = native_size
    return {'id': 'auto-native-'+candidate['id'], 'mode': 'heal', 'scope': 'foreground',
            'native_canvas_size': [w, h], 'points': [[candidate['center_native'][0]/max(1, w-1), candidate['center_native'][1]/max(1, h-1)]],
            'source_anchor': [donor['center_native'][0]/max(1, w-1), donor['center_native'][1]/max(1, h-1)],
            'radius_px': candidate['radius_px'], 'hardness': .35, 'opacity': 1.,
            'source_only': True, 'candidate_id': candidate['id']}


def freeze_approved_retouch(donors_path, out_dir=None):
    """Consume real double approvals, execute existing healing and freeze deltas."""
    donors_path = Path(donors_path).resolve()
    donors_manifest = _json(donors_path)
    proposals_path = _validate_file(donors_manifest['proposals'], 'proposal manifest')
    manifest = _json(proposals_path)
    _validate_recipe(manifest)
    _validate_file(manifest['source'], 'camera source')
    _validate_file(manifest['eligibility'], 'native eligibility')
    out_dir = Path(out_dir).resolve() if out_dir else donors_path.parent/'frozen'
    if (out_dir/'layer.json').exists():
        raise FileExistsError('frozen retouch layer already exists; retain its evidence')
    out_dir.mkdir(parents=True, exist_ok=True)
    linear, _ = open_linear(Path(manifest['source']['path']))
    if array_sha(linear) != manifest['source']['linear_rgb_sha256']:
        raise ValueError('native source decode differs from proposal evidence')
    with np.load(manifest['eligibility']['path']) as asset:
        labels, protection = asset['labels'], asset['protection']
    image = linear_preview(linear)
    layer = empty_clone_layer()
    receipts, statuses, patches, arrays = [], [], [], {}
    index = {candidate['id']: candidate for candidate in manifest['candidates']}
    # Compute every candidate against the immutable photographed source. Donor
    # selection excludes ALL candidate domains, so retouched pixels never donate.
    subject_alpha = np.isin(labels, SKIN_CLASSES+CLOTH_CLASSES).astype(np.float32)
    for donor_record in donors_manifest['donors']:
        candidate = index[donor_record['candidate_id']]
        _verify_source_crop(candidate['target_crop'], image)
        _verify_source_pair(donor_record['source_pair'], image)
        candidate_path = _validate_file(donor_record['candidate_approval'], 'candidate approval')
        if not _approved_candidate(candidate, _json(candidate_path)):
            raise ValueError('candidate no longer qualifies for healing')
        approval_path = Path(donor_record['request']['approval_path'])
        if not approval_path.is_file():
            statuses.append({'candidate_id': candidate['id'], 'status': 'no_donor_approval'})
            continue
        if not _approved_donor(donor_record, _json(approval_path)):
            statuses.append({'candidate_id': candidate['id'], 'status': 'donor_not_approved'})
            continue
        recomputed = select_donor(linear, labels, protection, candidate, manifest['candidates'])
        if recomputed != donor_record['donor']:
            raise ValueError('donor identity does not reproduce from native source and fixed parameters')
        stroke = stroke_for(candidate, recomputed, manifest['source']['native_size'])
        healed = apply_clone_retouch(linear.copy(), {'enabled': True, 'strokes': [stroke]},
                                    branch='foreground', subject_alpha=subject_alpha)
        radius = candidate['radius_px']+2
        box = _box(*candidate['center_native'], radius, *manifest['source']['native_size'])
        original = linear[_slice(box)]
        delta = healed[_slice(box)]-original
        support = np.any(delta != 0, axis=2)
        if np.any((labels[_slice(box)][support] != candidate['material_class']) | (protection[_slice(box)][support] > .01)):
            statuses.append({'candidate_id': candidate['id'], 'status': 'native_support_crosses_material_or_protection'})
            continue
        luminance = np.maximum(np.sum(original*np.asarray([.2126, .7152, .0722]), axis=2), 1e-4)
        magnitude = np.max(np.abs(delta), axis=2)/luminance
        if not np.isfinite(delta).all() or np.any(original+delta < 0) or magnitude.max(initial=0) > PARAMETERS['maximum_delta_fraction_of_source_luminance']:
            statuses.append({'candidate_id': candidate['id'], 'status': 'healing_delta_exceeds_photographic_bound'})
            continue
        if not support.any():
            statuses.append({'candidate_id': candidate['id'], 'status': 'approved_no_effect'})
            continue
        key = 'delta_'+candidate['id']
        arrays[key] = np.asarray(delta, np.float32)
        patches.append({'candidate_id': candidate['id'], 'native_box': box, 'array_key': key,
                        'changed_pixels': int(support.sum()), 'delta_array_sha256': array_sha(delta),
                        'original_patch_sha256': array_sha(original),
                        'maximum_delta_fraction_of_source_luminance': float(magnitude.max(initial=0)),
                        'stroke': stroke})
        layer['strokes'].append(stroke)
        receipts.append({'candidate_id': candidate['id'], 'candidate_approval': donor_record['candidate_approval'],
                         'donor_approval': _record(approval_path), 'source_pair': donor_record['source_pair'],
                         'matching': recomputed})
        statuses.append({'candidate_id': candidate['id'], 'status': 'approved_frozen'})
    delta_path = out_dir/'source-native-delta.npz'
    np.savez_compressed(delta_path, **arrays)
    result = {'schema_version': 'native_retouch_delta.v1', 'recipe': RECIPE,
              'enabled': bool(patches), 'source': manifest['source'], 'native_size': manifest['source']['native_size'],
              'donor_manifest': _record(donors_path), 'delta': _record(delta_path),
              'patches': patches, 'approvals': receipts, 'statuses': statuses,
              'clone_retouch': layer, 'source_rgb_donor_only': True,
              'reference_rgb_used': False, 'generation_used': False,
              'execution_stage': 'source_native_straight_linear_before_appearance_and_composition',
              'replay': 'frozen source difference; each pixel owns exactly one nonoverlapping patch'}
    _write(out_dir/'layer.json', result)
    replay = apply_frozen_retouch(linear, result, source_sha256=manifest['source']['sha256'], verify_original=True)
    preview = linear_preview(replay)
    preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    preview.save(out_dir/'retouch-preview.png')
    changed = np.any(replay != linear, axis=2)
    allowed = np.zeros(linear.shape[:2], bool)
    for patch in patches:
        allowed[_slice(patch['native_box'])] |= np.any(arrays[patch['array_key']] != 0, axis=2)
    verification = {'schema_version': 'native_retouch_verification.v1', 'approved_patches': len(patches),
        'source_file_preserved': sha256(manifest['source']['path']) == manifest['source']['sha256'],
        'unapproved_zero_change': bool(not np.any(changed & ~allowed)),
        'changed_pixels': int(sum(p['changed_pixels'] for p in patches)),
        'source_linear_sha256': array_sha(linear), 'replayed_linear_sha256': array_sha(replay),
        'frozen_delta_evidence_sha256': result['delta']['sha256'],
        'native_measurement': True, 'generation_used': False}
    _write(out_dir/'verification.json', verification)
    return result


def apply_frozen_retouch(values, layer, *, source_sha256, box=None,
                         canvas_size=None, premultiplied_alpha=None, verify_original=False):
    """Replay frozen native source differences on a full canvas or native ROI.

    This local consumer needs no filter halo and never resamples a donor. Use on
    the source baseline before tone/colour operations. ``verify_original`` is
    appropriate for straight-source replay; premultiplied foreground receives
    the same source difference once, multiplied by target alpha.
    """
    values = np.asarray(values, np.float32)
    if layer.get('schema_version') != 'native_retouch_delta.v1' or layer['source'].get('sha256') != source_sha256:
        raise ValueError('frozen retouch is not bound to the original camera source')
    native = tuple(layer['native_size'])
    if canvas_size is not None and tuple(canvas_size) != native:
        raise ValueError('frozen retouch requires the exact native canvas; proxy interpolation is not permitted')
    if box is None:
        box = [0, 0, native[0], native[1]]
    box = list(map(int, box))
    if values.shape[:2] != (box[3]-box[1], box[2]-box[0]) or values.shape[2:] != (3,):
        raise ValueError('frozen retouch ROI size does not match its native coordinate box')
    if box[0] < 0 or box[1] < 0 or box[2] > native[0] or box[3] > native[1]:
        raise ValueError('frozen retouch ROI lies outside the source native canvas')
    if not layer.get('enabled') or not layer.get('patches'):
        return values.copy()
    path = _validate_file(layer['delta'], 'frozen native retouch delta')
    result = values.copy()
    claimed = np.zeros(values.shape[:2], bool)
    association = None if premultiplied_alpha is None else np.asarray(premultiplied_alpha, np.float32)
    if association is not None and (association.shape != values.shape[:2] or not np.isfinite(association).all() or np.any((association < 0) | (association > 1))):
        raise ValueError('retouch target alpha must share the native replay ROI')
    with np.load(path) as asset:
        for patch in layer['patches']:
            pb = patch['native_box']
            overlap = [max(box[0], pb[0]), max(box[1], pb[1]), min(box[2], pb[2]), min(box[3], pb[3])]
            if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
                continue
            delta = asset[patch['array_key']]
            if delta.shape != (pb[3]-pb[1], pb[2]-pb[0], 3) or array_sha(delta) != patch['delta_array_sha256']:
                raise ValueError('frozen native retouch patch content/canvas is invalid')
            if verify_original and association is None and overlap == pb:
                original = values[overlap[1]-box[1]:overlap[3]-box[1], overlap[0]-box[0]:overlap[2]-box[0]]
                if array_sha(original) != patch['original_patch_sha256']:
                    raise ValueError('retouch is replaying against a changed source baseline')
            local = (slice(overlap[1]-box[1], overlap[3]-box[1]), slice(overlap[0]-box[0], overlap[2]-box[0]))
            selected = delta[overlap[1]-pb[1]:overlap[3]-pb[1], overlap[0]-pb[0]:overlap[2]-pb[0]]
            support = np.any(selected != 0, axis=2)
            if np.any(claimed[local] & support):
                raise ValueError('frozen retouch patches overlap and would apply source differences twice')
            claimed[local] |= support
            if association is not None:
                selected = selected*association[local][..., None]
            result[local] += selected
    return result


def frozen_working_domain(layer):
    if layer.get('schema_version')!='native_retouch_delta.v1' or layer.get('recipe')!=RECIPE:
        raise ValueError('Frozen retouch requires the current source-delta schema and recipe')
    if layer.get('source',{}).get('decode',{}).get('decode_version')!=RGB_DECODE_VERSION:
        raise ValueError('Frozen retouch requires the current explicit source decoder')
    precision = layer.get('source', {}).get('decode', {}).get('source_precision')
    if precision == 'raw16_linear':
        return 'scene_linear_sRGB'
    if precision == 'raster8_srgb':
        return 'display_linear_sRGB'
    raise ValueError('frozen retouch source precision does not establish its working domain')


def validate_frozen_retouch_layer(layer, *, source_sha256, native_size=None):
    """Validate the saved causal approval graph once when installing a layer.

    Native tile replay then validates its delta bytes/patch identities without
    rereading every API/crop receipt. An approved count is never substituted for
    source-crop evidence. Rejected observations are retained in donor statuses.
    """
    if (layer.get('schema_version') != 'native_retouch_delta.v1' or layer.get('recipe')!=RECIPE
            or layer.get('source', {}).get('sha256') != source_sha256):
        raise ValueError('frozen retouch layer source identity mismatch')
    if layer.get('source_rgb_donor_only') is not True or layer.get('reference_rgb_used') is not False or layer.get('generation_used') is not False:
        raise ValueError('retouch must contain photographed source differences only')
    if native_size is not None and list(native_size) != layer.get('native_size'):
        raise ValueError('frozen retouch layer native canvas mismatch')
    donors_path = _validate_file(layer['donor_manifest'], 'frozen donor manifest')
    donor_manifest = _json(donors_path)
    if donor_manifest.get('schema_version')!='native_retouch_donors.v1' or donor_manifest.get('recipe')!=RECIPE:
        raise ValueError('Frozen retouch donor algorithm/schema differs from the current contract')
    proposals_path = _validate_file(donor_manifest['proposals'], 'frozen proposal manifest')
    proposals = _json(proposals_path)
    _validate_recipe(proposals)
    if proposals['source'] != layer['source'] or donor_manifest['source'] != layer['source']:
        raise ValueError('retouch source differs across the frozen approval graph')
    candidate_index = {item['id']: item for item in proposals['candidates']}
    donor_index = {item['candidate_id']: item for item in donor_manifest['donors']}
    approval_index = {item['candidate_id']: item for item in layer['approvals']}
    patch_index = {item['candidate_id']: item for item in layer['patches']}
    if len(patch_index) != len(layer['patches']) or set(patch_index) != set(approval_index):
        raise ValueError('frozen patch/double-approval coverage differs')
    _validate_file(layer['delta'], 'frozen native retouch delta')
    for candidate_id in patch_index:
        if candidate_id not in candidate_index or candidate_id not in donor_index:
            raise ValueError('frozen patch has no deterministic proposal/donor evidence')
        candidate, donor, approval = candidate_index[candidate_id], donor_index[candidate_id], approval_index[candidate_id]
        candidate_receipt = _validate_file(approval['candidate_approval'], 'candidate approval')
        donor_receipt = _validate_file(approval['donor_approval'], 'donor approval')
        if approval['candidate_approval'] != donor['candidate_approval'] or approval['source_pair'] != donor['source_pair']:
            raise ValueError('frozen approval is not derived from its actual donor proposal')
        if not _approved_candidate(candidate, _json(candidate_receipt)) or not _approved_donor(donor, _json(donor_receipt)):
            raise ValueError('frozen patch no longer has two qualifying source-crop approvals')
        expected_stroke = stroke_for(candidate, donor['donor'], layer['native_size'])
        if patch_index[candidate_id]['stroke'] != expected_stroke:
            raise ValueError('frozen stroke does not reproduce from source-coordinate approved evidence')
    return {'ok': True, 'source_sha256': source_sha256, 'native_size': layer['native_size'],
            'working_domain': frozen_working_domain(layer), 'approved_patches': len(patch_index),
            'double_approved': bool(patch_index), 'generation_used': False,
            'source_display_binding': 'native source-crop RGB hashes + actual API raster hashes, derived from fixed root decode',
            'unapproved_zero_change': not patch_index}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    propose = commands.add_parser('propose')
    propose.add_argument('--source', required=True, type=Path)
    propose.add_argument('--analysis', required=True, type=Path)
    propose.add_argument('--face-protection', type=Path)
    propose.add_argument('--out-dir', required=True, type=Path)
    propose.add_argument('--maximum', type=int)
    donors = commands.add_parser('donors')
    donors.add_argument('--proposals', required=True, type=Path)
    donors.add_argument('--out-dir', type=Path)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--donors', required=True, type=Path)
    freeze.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    if args.command == 'propose':
        result = prepare_proposals(args.source, args.analysis, args.out_dir,
                                   face_protection_path=args.face_protection, maximum=args.maximum)
        summary = {'candidate_count': len(result['candidates']), 'out_dir': str(args.out_dir.resolve()), 'source_rgb_changed': False}
    elif args.command == 'donors':
        result = prepare_donors(args.proposals, args.out_dir)
        summary = {'donor_count': len(result['donors']), 'statuses': result['statuses'], 'source_rgb_changed': False}
    else:
        result = freeze_approved_retouch(args.donors, args.out_dir)
        summary = {'approved_patches': len(result['patches']), 'statuses': result['statuses'], 'delta': result['delta']}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
