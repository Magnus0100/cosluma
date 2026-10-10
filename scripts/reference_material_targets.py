"""Reviewed evidence for current material colour qualification.

All arrays use their own displayed canvas. Target pools require actual crop
review evidence; colour fitting never declares environmental colour unwanted.
The current solver and field producer live in reference_material_targets_v2.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.interpolate import PchipInterpolator

import cosplay_post_review as visual
from colour_space import to_lab, from_lab
from float_appearance import srgb_to_linear
from local_semantics_worker import CLASS_NAMES
from sapiens_model_profiles import profile

SCHEMA = 'qualified_material_targets.v2'
DOMAIN = 'display_linear_sRGB'
REGIONS = {'face': [3], 'left_arm': [7, 11], 'right_arm': [16, 20],
           'left_hand': [6], 'right_hand': [15], 'legs': [5, 8, 12, 14, 17, 21, 22]}
FEATURES = {'lips': [24, 25], 'teeth': [26, 27], 'tongue': [28]}
SETTINGS = {'maximum_ab': .04, 'quantiles': [.1, .3, .5, .7, .9],
            'minimum_cells': 6, 'minimum_pool_pixels': 24,
            'maximum_fold_disagreement': .012}


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _module_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(_canonical({'shape': list(value.shape), 'dtype': str(value.dtype)})+value.tobytes()).hexdigest()


def _file(record):
    path = Path(record['path']).resolve()
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if record['sha256'] != actual:
        raise ValueError('Material evidence bytes changed: '+str(path))
    return {'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size}


def _asset_array(record, name):
    item = _file(record)
    if Path(item['path']).suffix.lower() == '.npz':
        with np.load(item['path'], allow_pickle=False) as resource:
            return np.asarray(resource[record.get('array_name', name)])
    with Image.open(item['path']) as image:
        return np.asarray(image.convert('L'), np.float32)/255


def _resized(values, size, *, labels=False):
    if values.shape == size[::-1]:
        return values
    mode = Image.Resampling.NEAREST if labels else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(values.astype(np.int32) if labels else values.astype(np.float32)).resize(size, mode))


def _observation(observation):
    role = observation['role']
    if role not in ('source', 'reference') or observation.get('domain') != DOMAIN:
        raise ValueError('Material observation requires an explicit role and display-linear sRGB')
    rgb = np.asarray(observation['rgb'], np.float32)
    labels = np.asarray(observation['labels'])
    if (rgb.ndim != 3 or rgb.shape[-1] != 3 or not rgb.size or not np.isfinite(rgb).all()
            or rgb.min() < 0 or rgb.max() > 1 or labels.shape != rgb.shape[:2]
            or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() > 28):
        raise ValueError('RGB and complete semantic labels must occupy their own finite canvas')
    binding = observation['binding']
    image = _file(binding['image'])
    labels_record = _file(binding['semantic_labels'])
    parent_labels = _file(binding['semantic_labels']['parent']) if binding['semantic_labels'].get('parent') else labels_record
    actual_labels = _asset_array(binding['semantic_labels'], 'labels')
    if not np.array_equal(labels, _resized(actual_labels, rgb.shape[1::-1], labels=True)):
        raise ValueError('Observed labels differ from the actual bound semantic file')
    analysis_record = _file(binding['analysis'])
    analysis = _json(analysis_record['path'])
    semantic = analysis['semantic_result']
    specification = profile(semantic.get('model_profile'), 'segmentation')
    if (semantic.get('class_names') != CLASS_NAMES or semantic.get('model') != specification['repository']
            or semantic.get('model_revision') != specification['revision']
            or semantic.get('checkpoint_sha256') != specification['checkpoint_sha256']):
        raise ValueError('Material observations require the exact pinned complete semantic taxonomy')
    root = _file(binding['identity_root']) if binding.get('identity_root') else image
    display_parent = _file(binding['display_parent']) if binding.get('display_parent') else None
    if display_parent:
        parent_image = visual._raster(display_parent['path'])
        expected_image = parent_image.resize(rgb.shape[1::-1], Image.Resampling.LANCZOS)
        actual_image = visual._raster(image['path'])
        if binding.get('display_derivative') != 'oriented_srgb_lanczos' or not np.array_equal(np.asarray(expected_image), np.asarray(actual_image)):
            raise ValueError('Displayed material derivative must equal the explicit bound sRGB parent resize')
    parent_values = _asset_array(binding['semantic_labels']['parent'], 'labels') if binding['semantic_labels'].get('parent') else actual_labels
    if (analysis['source']['sha256'] != root['sha256']
            or analysis.get('analysis_role') != ('source' if role == 'source' else 'candidate')
            or Path(analysis['semantic_result']['labels_path']).resolve() != Path(parent_labels['path'])
            or not np.array_equal(actual_labels, _resized(parent_values, actual_labels.shape[::-1], labels=True))):
        raise ValueError('Material semantics do not belong to this actual identity root')
    displayed = srgb_to_linear(np.asarray(visual._raster(image['path']), np.float32)/255)
    if rgb.shape != displayed.shape or not np.allclose(rgb, displayed, rtol=0, atol=2e-7):
        raise ValueError('Material RGB buffers must be the actual bound displayed raster pixels')
    masks = {}; mask_files = {}
    for name, values in observation.get('feature_masks', {}).items():
        values = np.asarray(values, np.float32)
        if values.shape != labels.shape or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError('Feature ownership must be finite on its own material canvas')
        record = binding.get('feature_masks', {}).get(name)
        if record is None or not np.allclose(values, _resized(_asset_array(record, 'values'), rgb.shape[1::-1]), atol=1e-7, rtol=0):
            raise ValueError('Feature ownership lacks matching actual mask evidence: '+name)
        masks[name] = values; mask_files[name] = _file(record)
    if not any(name in masks for name in ('eyes', 'face_features')):
        raise ValueError('Skin target selection requires independent eye/face-feature ownership')
    feature_receipt = None
    if binding.get('feature_ownership_receipt'):
        feature_receipt = _file(binding['feature_ownership_receipt'])
        owned = _json(feature_receipt['path'])
        if (owned['input']['sha256'] != image['sha256']
                or owned['semantic_labels']['sha256'] != labels_record['sha256']
                or any(owned['feature_masks'][name]['sha256'] != item['sha256'] for name, item in mask_files.items())):
            raise ValueError('Feature producer receipt differs from its own canvas and material mask assets')
    # RGB buffers are explicitly bound in addition to the displayed raster and
    # root. They can be original float pixels, not quantized PNG reconstruction.
    provenance = {'role': role, 'domain': DOMAIN, 'coordinate_space': role+'_display',
        'canvas_size': list(rgb.shape[1::-1]), 'image': image, 'identity_root': root,
        'analysis': analysis_record, 'semantic_labels': labels_record, 'feature_masks': mask_files,
        'parent_semantic_labels': parent_labels,
        'display_parent': display_parent, 'display_derivative': binding.get('display_derivative'),
        'feature_ownership_receipt': feature_receipt,
        'rgb_array_sha256': array_sha(rgb), 'labels_array_sha256': array_sha(labels),
        'feature_array_sha256': {name: array_sha(values) for name, values in masks.items()}}
    return rgb, labels, masks, provenance


def _owners(labels, masks):
    excluded = np.isin(labels, [2, 24, 25, 26, 27, 28])
    for values in masks.values():
        excluded |= values > 1e-5
    owners = {name: np.isin(labels, ids) & ~excluded for name, ids in REGIONS.items()}
    # Features are separate targets; never inherit an estimated skin endpoint.
    owners.update({name: np.isin(labels, ids) for name, ids in FEATURES.items()})
    for name in ('eyes', 'makeup'):
        if name in masks:
            owners[name] = masks[name] > .5
    return owners


def candidate_material_pools(observation, *, subdivisions=3, minimum_pixels=24):
    """Automatic same-material spatial blocks; masks are cropped to each box.

    The returned mask excludes feature ownership and semantic boundaries.
    Outliers do not determine whether a block is a trustworthy colour target.
    """
    if type(subdivisions) is not int or not 1 <= subdivisions <= 8:
        raise ValueError('Material pool subdivision must be an integer between one and eight')
    rgb, labels, masks, binding = _observation(observation)
    pools = []
    for region, owner in _owners(labels, masks).items():
        interior = binary_erosion(owner, iterations=1)
        y, x = np.nonzero(interior)
        if not len(x):
            continue
        xs = np.linspace(x.min(), x.max()+1, subdivisions+1).astype(int)
        ys = np.linspace(y.min(), y.max()+1, subdivisions+1).astype(int)
        for row in range(subdivisions):
            for column in range(subdivisions):
                box = [int(xs[column]), int(ys[row]), int(xs[column+1]), int(ys[row+1])]
                left, top, right, bottom = box
                mask = interior[top:bottom, left:right].copy()
                if mask.sum() < minimum_pixels:
                    continue
                record = {'anchor_id': observation['role']+':'+region+':r'+str(row)+'c'+str(column),
                    'role': observation['role'], 'region': region,
                    'material_family': 'skin' if region in REGIONS else region,
                    'box': box, 'pixels': int(mask.sum()), 'mask_sha256': array_sha(mask),
                    'pool_profile': {'subdivisions': subdivisions, 'minimum_pixels': minimum_pixels},
                    'observation_sha256': _digest(binding), 'coordinate_space': observation['role']+'_display'}
                pools.append({**record, 'mask': mask})
    return pools


def _checked_diagnosis(diagnosis, source_binding):
    file = _file(diagnosis['evidence'])
    review = _json(file['path'])
    if (review.get('schema') != visual.COMPREHENSIVE_SCHEMA or review.get('status') != 'reviewed'
            or review.get('actual_api_evidence_valid') is not True
            or review['binding']['images']['source']['sha256'] not in {
                source_binding['image']['sha256'], (source_binding.get('display_parent') or {}).get('sha256')}):
        raise ValueError('Colour correction requires actual source-bound visual diagnosis')
    directory = Path(file['path']).parent
    import reference_refinement
    review, _, _ = reference_refinement._bound_review(Path(review['binding']['images']['source']['path']),
        Path(review['binding']['images']['reference']['path']), Path(file['path']), _json(directory/'design.json'))
    target = diagnosis.get('target', 'reference')
    if target not in ('reference','final'):
        raise ValueError('Unknown material diagnosis target')
    verdict = review['checks']['skin_materials'][target]
    if verdict['status'] != 'fail':
        raise ValueError('No actual diagnosed skin/material correction is available')
    if target == 'final':
        endpoint = diagnosis.get('appearance_endpoint') or {}
        binding = review.get('run_binding') or {}
        final_image = review['binding']['images'].get('final') or {}
        if (verdict.get('correction_stage') != 'subject_appearance'
                or verdict.get('cause') not in ('transfer_incomplete','composition_defect','reference_defect')
                or any(endpoint.get(key) != binding.get(key) for key in
                       ('run_id','revision','source_sha256','reference_sha256','final_sha256'))
                or endpoint.get('final_sha256') != final_image.get('sha256')
                or _file(final_image)['sha256'] != endpoint['final_sha256']
                or not endpoint.get('alpha_sha256') or not endpoint.get('automatic_delta_sha256')):
            raise ValueError('Final material diagnosis is not the current qualified appearance endpoint')
        from automatic_material_completion import final_skin_regions
        if not set(diagnosis['regions']).issubset(set(final_skin_regions(review))):
            raise ValueError('Final material regions exceed the actual failed native skin owners')
    regions = diagnosis['regions']
    if not isinstance(regions, list) or not regions or len(set(regions)) != len(regions) or any(name not in REGIONS for name in regions):
        raise ValueError('Diagnosed source owners require explicit supported skin regions')
    result = {'evidence': file, 'regions': regions, 'region_assignment': 'explicit_source_semantic_owners',
              'reason': verdict['reason']}
    if 'target' in diagnosis:
        result['target'] = target
    if target == 'final':
        result['appearance_endpoint'] = diagnosis['appearance_endpoint']
    return result


def _checked_anchor_review(review, pools, observations, diagnosis):
    if review.get('schema') != 'material_anchor_review.v1' or review.get('status') != 'reviewed':
        raise ValueError('Material targets require a completed actual anchor review')
    proof = {key: _file(review['binding'][key]) for key in ('diagnosis', 'comparison', 'prompt', 'response', 'provenance')}
    if proof['diagnosis']['sha256'] != diagnosis['evidence']['sha256']:
        raise ValueError('Anchor review belongs to another material diagnosis')
    prompt = Path(proof['prompt']['path']).read_text(encoding='utf8')
    visual._validate_provenance(_json(proof['provenance']['path']), Path(proof['comparison']['path']), prompt)
    response = _json(proof['response']['path'])
    actual = {row['anchor_id']: row for row in response['anchors']}
    if len(actual) != len(response['anchors']):
        raise ValueError('Anchor API verdicts have duplicate material pools')
    checked = []
    with Image.open(proof['comparison']['path']) as opened:
        sheet = opened.convert('RGB')
    for row in review['anchors']:
        pool = pools.get(row['anchor_id'])
        if pool is None or row['status'] not in ('qualified', 'rejected', 'unknown'):
            raise ValueError('Unknown reviewed material pool or qualification status')
        verdict = actual.get(row['anchor_id'])
        if verdict is None or row['status'] != verdict['status'] or row['reason'] != verdict['reason']:
            raise ValueError('Saved material qualification differs from the actual API verdict')
        crop = row['crop']; crop_file = _file(crop)
        binding = observations[pool['role']][3]
        if (crop['parent_sha256'] != binding['image']['sha256'] or crop['box'] != pool['box']
                or crop['coordinate_space'] != binding['coordinate_space']):
            raise ValueError('Anchor crop does not belong to this exact material pool canvas')
        # Crop bytes are compared with the observation display, then with the
        # reviewed sheet panel. Crops cannot be silently corrected before QA.
        rgb = observations[pool['role']][0]
        left, top, right, bottom = crop['box']
        display = np.where(rgb <= .0031308, 12.92*rgb, 1.055*np.maximum(rgb, 0)**(1/2.4)-.055)
        expected = np.uint8(np.clip(np.rint(display[top:bottom, left:right]*255), 0, 255))
        with Image.open(crop_file['path']) as opened:
            pixels = np.asarray(opened.convert('RGB'))
        if pixels.shape != expected.shape or not np.array_equal(pixels, expected):
            raise ValueError('Actually reviewed crop pixels differ from the declared material observation')
        if 'panel_box' in crop:
            panel = np.asarray(sheet.crop(tuple(crop['panel_box'])))
            if panel.shape != pixels.shape or not np.array_equal(panel, pixels):
                raise ValueError('Material crop is not present at its bound review-sheet panel')
        else:
            raise ValueError('Anchor review must locate its exact crop pixels on the actually reviewed sheet')
        checked.append({'anchor_id': pool['anchor_id'], 'status': row['status'], 'reason': row['reason'],
                        'crop': {**crop, **crop_file}})
    if len({row['anchor_id'] for row in checked}) != len(checked):
        raise ValueError('Saved anchor review repeats a material pool')
    return checked, proof


def _cells(rgb, mask, box, step):
    lab = to_lab(rgb); rows = []; positions = []
    left, top, right, bottom = box
    for y in range(top, bottom, step):
        for x in range(left, right, step):
            selected = mask[y:y+step, x:x+step]
            if selected.size and selected.mean() >= .75:
                values = lab[y:y+step, x:x+step][selected]
                if len(values) and .08 < np.median(values[:, 0]) < .96:
                    rows.append(np.median(values, axis=0)); positions.append([x, y])
    return np.asarray(rows).reshape(-1, 3), np.asarray(positions).reshape(-1, 2)


def reviewed_material_evidence(source, reference, diagnosis, anchor_review):
    """Return JSON-safe targets from reviewed same-material spatial pools."""
    observations = {'source': _observation(source), 'reference': _observation(reference)}
    diagnosis = _checked_diagnosis(diagnosis, observations['source'][3])
    pool_profile = anchor_review.get('pool_profile', {'subdivisions': 3, 'minimum_pixels': 24})
    pools = {row['anchor_id']: row for observation in (source, reference)
             for row in candidate_material_pools(observation, **pool_profile)}
    reviewed, proof = _checked_anchor_review(anchor_review, pools, observations, diagnosis)
    targets = {}; records = []
    for row in reviewed:
        pool = pools[row['anchor_id']]
        record = {key: value for key, value in pool.items() if key != 'mask'}
        record.update(status=row['status'], reason=row['reason'])
        records.append(record)
        if row['status'] != 'qualified':
            continue
        rgb = observations[pool['role']][0]
        mask = np.zeros(rgb.shape[:2], bool)
        left, top, right, bottom = pool['box']; mask[top:bottom, left:right] = pool['mask']
        step = max(2, min(rgb.shape[:2])//128)
        cells, xy = _cells(rgb, mask, pool['box'], step)
        if len(cells) < SETTINGS['minimum_cells']:
            record.update(target_status='unsupported', target_reason='insufficient_safe_independent_cells')
            continue
        targets.setdefault(pool['material_family'], []).append({'anchor_id': row['anchor_id'],
            'role': pool['role'], 'region': pool['region'], 'cells': cells.tolist(),
            'cell_positions': xy.tolist(), 'cell_size': step, 'target_status': 'qualified',
            'crop': row['crop'], 'pool': {key: value for key, value in pool.items() if key != 'mask'}})
    result = {'schema': SCHEMA, 'domain': DOMAIN, 'binding': {role: value[3] for role, value in observations.items()},
        'evidence_algorithm_sha256': _module_sha(),
        'diagnosis': diagnosis, 'anchor_review_proof': proof, 'reviewed_pools': records, 'targets': targets,
        'settings': SETTINGS.copy(), 'source_to_reference_mapping': False, 'paired_anchor_count': 0,
        'target_origin': 'actual_qualified_material_crops_only', 'environmental_colour_outlier_is_not_diagnosis': True}
    result['qualification_sha256'] = _digest(result)
    return result


def _conditional_endpoint(cells, lightness):
    # Equal mass per qualified pool is imposed by the caller, so one large
    # face region cannot suppress an independently reviewed smaller clean pool.
    order = np.argsort(cells[:, 0], kind='stable'); cells = cells[order]
    chunks = np.array_split(cells, min(len(SETTINGS['quantiles']), len(cells)))
    centers = np.array([np.median(chunk, axis=0) for chunk in chunks if len(chunk)])
    x, indices = np.unique(centers[:, 0], return_index=True); ab = centers[indices, 1:]
    if len(x) < 2:
        return np.broadcast_to(np.median(cells[:, 1:], axis=0), (*lightness.shape, 2)).copy()
    return PchipInterpolator(x, ab, axis=0, extrapolate=False)(np.clip(lightness, x[0], x[-1]))


def _limited(delta, maximum):
    magnitude = np.linalg.norm(delta, axis=-1, keepdims=True)
    return delta*np.minimum(1, maximum/np.maximum(magnitude, 1e-12))


def _gamut_dose(lab, delta):
    result = np.ones(lab.shape[:2], np.float32)
    active = np.any(delta != 0, axis=-1)
    if not active.any():
        return result
    base, shift = lab[active], delta[active]
    proposed = base.copy(); proposed[:, 1:] += shift
    rgb = from_lab(proposed)
    unsafe = (rgb.min(-1) < 0) | (rgb.max(-1) > 1)
    if not unsafe.any():
        return result
    base, shift = base[unsafe], shift[unsafe]
    low = np.zeros(len(base), np.float32); high = np.ones_like(low)
    for _ in range(20):
        dose = (low+high)/2; proposed = base.copy(); proposed[:, 1:] += shift*dose[:, None]
        rgb = from_lab(proposed)
        safe = (rgb.min(-1) >= 0) & (rgb.max(-1) <= 1)
        low = np.where(safe, dose, low); high = np.where(safe, high, dose)
    active_doses = np.ones(int(active.sum()), np.float32); active_doses[unsafe] = low
    result[active] = active_doses
    return result


def validate_material_evidence(source, reference, qualification, diagnosis, *, source_base=None):
    """Reopen current pixels, pools, verdicts and the source-owned float base."""
    observations = {'source': _observation(source), 'reference': _observation(reference)}
    body = {key: value for key, value in qualification.items() if key != 'qualification_sha256'}
    if (qualification.get('schema') != SCHEMA or qualification['qualification_sha256'] != _digest(body)
            or qualification['binding'] != {role: value[3] for role, value in observations.items()}
            or qualification.get('evidence_algorithm_sha256') != _module_sha()):
        raise ValueError('Qualified colour targets or their actual inputs changed')
    checked_diagnosis = _checked_diagnosis(diagnosis, observations['source'][3])
    if checked_diagnosis != qualification['diagnosis']:
        raise ValueError('Colour targets belong to another source material diagnosis')
    for record in qualification['anchor_review_proof'].values():
        _file(record)
    # Reopen the actual response so a self-consistent edited JSON target cannot
    # convert rejected/unknown crops into qualified colour endpoints.
    verdicts = {row['anchor_id']: row for row in _json(qualification['anchor_review_proof']['response']['path'])['anchors']}
    cells = []; anchor_ids = []; reopened_crops = []; reopened_pools = {}
    for target in qualification['targets'].get('skin', []):
        if verdicts.get(target['anchor_id'], {}).get('status') != 'qualified':
            raise ValueError('An unqualified actual crop entered the skin target endpoint')
        _file(target['crop'])
        pool = next((row for row in candidate_material_pools(source if target['role'] == 'source' else reference,
                                                           **target['pool']['pool_profile'])
                     if row['anchor_id'] == target['anchor_id']), None)
        if pool is None or target['pool'] != {key: value for key, value in pool.items() if key != 'mask'}:
            raise ValueError('Qualified target material ownership changed')
        if (pool['material_family'] != 'skin' or target.get('region') != pool['region']
                or target.get('target_status') != 'qualified'):
            raise ValueError('A different material family cannot enter a skin endpoint')
        reopened_pools[target['anchor_id']] = pool
        reopened_crops.append({'anchor_id':target['anchor_id'],'status':'qualified',
            'reason':verdicts[target['anchor_id']]['reason'],'crop':target['crop']})
        rgb = observations[target['role']][0]; mask = np.zeros(rgb.shape[:2], bool)
        if target.get('cell_size') != max(2, min(rgb.shape[:2])//128):
            raise ValueError('Current qualified material cell sampling changed')
        left, top, right, bottom = pool['box']; mask[top:bottom, left:right] = pool['mask']
        actual_cells, xy = _cells(rgb, mask, pool['box'], target['cell_size'])
        if actual_cells.tolist() != target['cells'] or xy.tolist() != target['cell_positions']:
            raise ValueError('Qualified endpoint colour cells differ from their reviewed pixels')
        # Deterministic quantile subsampling gives equal mass to each pool.
        order = np.argsort(actual_cells[:, 0], kind='stable')
        cells.append(actual_cells[order[np.linspace(0, len(order)-1, 32).astype(int)]])
        anchor_ids.append(target['anchor_id'])
    if len(set(anchor_ids)) != len(anchor_ids):
        raise ValueError('Qualified skin endpoints repeat a material pool')
    if reopened_crops:
        _checked_anchor_review({'schema':'material_anchor_review.v1','status':'reviewed',
            'binding':qualification['anchor_review_proof'],'anchors':reopened_crops},
            reopened_pools, observations, checked_diagnosis)
    rgb, labels, features, source_binding = observations['source']
    base_binding = {'basis': 'original_source_display', 'rgb_array_sha256': array_sha(rgb)}
    if source_base is not None:
        base = np.asarray(source_base['rgb'], np.float32)
        bound = source_base['binding']
        asset, recipe = _file(bound['asset']), _file(bound['recipe'])
        actual = _asset_array(bound['asset'], 'values')
        if (bound['parent_source_sha256'] != source_binding['image']['sha256']
                or base.shape != rgb.shape or actual.shape != base.shape
                or not np.array_equal(actual, base) or not np.isfinite(base).all()
                or base.min() < 0 or base.max() > 1):
            raise ValueError('Current colour base is not an exact source-bound frozen float asset')
        rgb = base
        base_binding = {'basis': 'frozen_current_colour_base', 'asset': asset, 'recipe': recipe,
                        'parent_source_sha256': source_binding['image']['sha256'], 'rgb_array_sha256': array_sha(rgb)}
    return {'observations': observations, 'source_colour_base': base_binding,
            'target_cells': np.concatenate(cells) if cells else np.empty((0, 3)),
            'diagnosis': checked_diagnosis, 'source_rgb': rgb}
