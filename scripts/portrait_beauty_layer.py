"""Frozen generated face RGB, replayed before colour and scene lighting.

The model's appearance is the editable layer. Composition weights handle
seams and visibility, not texture restoration or face-wide colour protection.
No model, GPU or network is used by this module.
"""
from __future__ import annotations

from functools import lru_cache
import copy
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from asset_identity import sha256_file

SCHEMA = 'portrait_beauty_layer.v1'
DOMAIN = 'display_linear_sRGB'
VISIBLE_FACE_POLICY = 'visible_complete_face_upward_forehead_inward_feather.v2'
HAIR_GUARD_POLICY = 'qualified_visible_face_vetoes_hair_correction.v1'


def sha256(path):
    return sha256_file(path)


def _write(path, record):
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf8')


def _own_bytes(source, target):
    """Own immutable cached bytes without a second full-file copy when possible."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        return
    try:
        target.unlink(missing_ok=True)
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _box(value, size):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(type(x) is not int for x in value)
            or not (0 <= value[0] < value[2] <= size[0]
                    and 0 <= value[1] < value[3] <= size[1])):
        raise ValueError('Beauty faces require integer native source coordinates')
    return tuple(value)


def freeze_prepared(record, directory):
    """Own generated pixels and float32 weights; discard runtime dependencies."""
    from float_appearance import srgb_to_linear
    if (record.get('schema') != 'beautygrpo_layer_prepare.v1'
            or record.get('working_domain') != DOMAIN or not record.get('faces')):
        raise ValueError('A completed BeautyGRPO preparation with visible faces is required')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    size = record['native_size']
    faces = []
    for index, face in enumerate(record['faces']):
        box = _box(face['native_box'], size)
        generated_path = Path(face['generated_native_roi'])
        weight_path = Path(face['composition_weight'])
        for path, digest in ((generated_path, face['generated_native_roi_sha256']),
                             (weight_path, face['composition_weight_sha256'])):
            if sha256(path) != digest:
                raise ValueError('Prepared face pixels or composition weights changed')
        with Image.open(generated_path) as opened:
            rgb = srgb_to_linear(np.asarray(opened.convert('RGB'), np.float32)/255)
        weight = np.load(weight_path, allow_pickle=False)
        shape = (box[3]-box[1], box[2]-box[0])
        if (rgb.shape != (*shape, 3) or weight.shape != shape
                or not np.isfinite(weight).all() or weight.min() < 0 or weight.max() > 1
                or not np.any(weight > 0)):
            raise ValueError('Prepared face must contain finite native RGB and nonempty weights')
        asset = directory/f'face-{index+1:02d}.npz'
        if asset.exists():
            raise FileExistsError('Choose a new beauty layer directory')
        np.savez_compressed(asset, generated_rgb=rgb.astype(np.float32),
                            composition_weight=weight.astype(np.float32))
        row = {'id': face['id'], 'native_box': list(box),
               'pixels': {'path': str(asset), 'sha256': sha256(asset)},
               'composition_policy': face.get('composition_policy', 'visible_complete_face_inward_feather'),
               'generated_detail_origin': 'BeautyGRPO_1024_sampling_not_camera_native_detail'}
        for key in ('composition','mapping','parent_native_box'):
            if key in face:
                row[key] = copy.deepcopy(face[key])
        # Keep the actual model output available for comparison after relocation.
        if face.get('model_output'):
            original = Path(face['model_output'])
            owned = directory/f'face-{index+1:02d}-model.png'
            shutil.copyfile(original, owned)
            row['model_output'] = {'path': str(owned), 'sha256': sha256(owned)}
        faces.append(row)
    result = {'schema': SCHEMA, 'state': 'prepared', 'working_domain': DOMAIN,
              'source_path': record['source_path'], 'source_sha256': record['source_sha256'],
              'native_size': size, 'source_precision': record['source_precision'],
              'source_display': record['source_display'],
              'native_source_retouch': record.get('native_source_retouch'),
              'model_identity': record['model_identity'], 'faces': faces,
              'default_strength': 1.0, 'alpha_changed': False,
              'model_geometry_preservation_guaranteed': False,
              'source_texture_restoration': False, 'replay_requires_models': False,
              'visual_acceptance': False}
    result['source_display_consumptions'] = 1
    path = directory/'portrait-beauty-layer.json'
    _write(path, result)
    return path, result


@lru_cache(maxsize=12)
def _read_json(path, modified, size):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load_result(path):
    path = Path(path).resolve()
    stat = path.stat()
    result = _read_json(str(path), stat.st_mtime_ns, stat.st_size)
    if (result.get('schema') != SCHEMA or result.get('state') != 'prepared'
            or result.get('working_domain') != DOMAIN or not result.get('faces')
            or result.get('alpha_changed') is not False):
        raise ValueError('Current frozen generated beauty layer is required')
    ids = [face['id'] for face in result['faces']]
    if len(set(ids)) != len(ids):
        raise ValueError('Beauty face identifiers must be distinct')
    for face in result['faces']:
        _box(face['native_box'], result['native_size'])
    return result, path


def identity_for_run(run):
    """Path-independent appearance identity for reference fitting and relocation."""
    control = ((run.get('options') or {}).get('appearance_revision') or {}).get('portrait_beauty')
    if not control:
        return None
    result, _ = load_result(control['result'])
    strength = float(control.get('strength', 1.))
    disabled = sorted(set(control.get('disabled_faces') or []))
    if (not math.isfinite(strength) or not 0 <= strength <= 1
            or set(disabled) - {face['id'] for face in result['faces']}):
        raise ValueError('Invalid beauty controls for reference fitting')
    return {'schema': SCHEMA, 'source_sha256': result['source_sha256'],
            'native_size': copy.deepcopy(result['native_size']),
            'working_domain': result['working_domain'],
            'source_display': copy.deepcopy(result['source_display']),
            'native_source_retouch': copy.deepcopy(result.get('native_source_retouch')),
            'strength': strength, 'disabled_faces': disabled,
            'faces': [{'id': face['id'], 'native_box': list(face['native_box']),
                       'pixels_sha256': face['pixels']['sha256']}
                      for face in result['faces']]}


def materialize_for_run(root, run, control):
    """Own the editable closure so deleting a prepare job cannot break a save."""
    if not control:
        return copy.deepcopy(control)
    result, origin = load_result(control['result'])
    if result['source_sha256'] != run['source']['sha256']:
        raise ValueError('Cannot save a beauty layer from another source')
    directory = Path(root)/'outputs/runs'/run['run_id']/'inputs/portrait-beauty'
    directory = directory.resolve();directory.mkdir(parents=True, exist_ok=True)
    layer = copy.deepcopy(result)
    for index, face in enumerate(layer['faces']):
        for key,suffix in (('pixels','.npz'),('model_output','-model.png')):
            if key not in face:
                continue
            asset = face[key];source = Path(asset['path'])
            source = source if source.is_absolute() else origin.parent/source
            if sha256(source) != asset['sha256']:
                raise ValueError('Frozen beauty asset changed before saving')
            target = directory/f'face-{index+1:02d}{suffix}'
            if source.resolve() != target.resolve():
                _own_bytes(source, target)
            asset['path'] = str(target)
    target = directory/'portrait-beauty-layer.json';_write(target,layer)
    owned = copy.deepcopy(control);owned['result']=str(target)
    return owned


@lru_cache(maxsize=12)
def _pixels(path, modified, size, digest, shape):
    if sha256(path) != digest:
        raise ValueError('Frozen beauty pixels changed')
    with np.load(path, allow_pickle=False) as fields:
        rgb, weight = fields['generated_rgb'], fields['composition_weight']
    if (rgb.dtype != np.dtype('float32') or weight.dtype != np.dtype('float32')
            or rgb.shape != (*shape, 3) or weight.shape != shape
            or not np.isfinite(rgb).all() or not np.isfinite(weight).all()
            or rgb.min() < 0 or rgb.max() > 1 or weight.min() < 0 or weight.max() > 1):
        raise ValueError('Invalid frozen native generated RGB or weight')
    rgb.setflags(write=False); weight.setflags(write=False)
    return rgb, weight


def _sample(values, extent, size):
    """Sample associated RGB and weights together to avoid dark edge halos."""
    left, top, right, bottom = extent
    if all(float(x).is_integer() for x in extent) and (right-left, bottom-top) == size:
        return values[int(top):int(bottom), int(left):int(right)]
    def plane(array):
        return np.asarray(Image.fromarray(array).transform(size, Image.Transform.EXTENT,
            extent, resample=Image.Resampling.BILINEAR), np.float32)
    return plane(values) if values.ndim == 2 else np.stack([plane(values[...,c]) for c in range(3)], -1)


def apply_result(values, result, *, parent=None, strength=1., disabled_faces=(),
                 canvas_size=None, box=None, premultiplied_alpha=None):
    """Replay full, reduced or native ROI RGB without changing coverage."""
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Beauty strength must lie in [0,1]')
    disabled = set(disabled_faces)
    if disabled - {face['id'] for face in result['faces']}:
        raise ValueError('A disabled beauty face does not belong to this layer')
    source = np.asarray(values, np.float32)
    height, width = source.shape[:2]
    canvas = tuple(canvas_size or (width, height))
    target = tuple(box or (0, 0, width, height))
    if source.ndim != 3 or source.shape[-1] != 3 or (target[2]-target[0], target[3]-target[1]) != (width, height):
        raise ValueError('Beauty replay extent must match its RGB buffer')
    if premultiplied_alpha is not None and premultiplied_alpha.shape != source.shape[:2]:
        raise ValueError('Beauty coverage must match the RGB buffer')
    if strength == 0:
        return values
    sx, sy = canvas[0]/result['native_size'][0], canvas[1]/result['native_size'][1]
    after = source.copy()
    for face in result['faces']:
        if face['id'] in disabled:
            continue
        l,t,r,b = face['native_box']
        x0,y0 = max(target[0], math.floor(l*sx)), max(target[1], math.floor(t*sy))
        x1,y1 = min(target[2], math.ceil(r*sx)), min(target[3], math.ceil(b*sy))
        if x0 >= x1 or y0 >= y1:
            continue
        path = Path(face['pixels']['path'])
        path = path if path.is_absolute() else Path(parent or '.')/path
        stat = path.stat()
        rgb, weight = _pixels(str(path), stat.st_mtime_ns, stat.st_size,
                              face['pixels']['sha256'], (b-t,r-l))
        extent = (x0/sx-l, y0/sy-t, x1/sx-l, y1/sy-t)
        sample_size = (x1-x0,y1-y0)
        w = np.clip(_sample(weight, extent, sample_size), 0, 1)*np.float32(strength)
        rgbw = _sample(rgb*weight[...,None], extent, sample_size)*np.float32(strength)
        region = np.s_[y0-target[1]:y1-target[1], x0-target[0]:x1-target[0]]
        if premultiplied_alpha is not None:
            rgbw *= premultiplied_alpha[region][...,None]
        changed = w > 0
        rendered = after[region]*(1-w[...,None]) + rgbw
        after[region][changed] = rendered[changed]
    return after


def validate_source_for_run(run, *, root=None):
    """Validate the source snapshot locked by a prepared generated face layer."""
    control = ((run.get('options') or {}).get('appearance_revision') or {}).get('portrait_beauty')
    if not control:
        raise ValueError('A prepared beauty revision is required')
    if not isinstance(control, dict) or set(control) - {'result', 'strength', 'disabled_faces'}:
        raise ValueError('Beauty revision requires result, strength and disabled_faces')
    result, path = load_result(control['result'])
    from display_transform import source_transform_for_run
    if (result['source_sha256'] != (run.get('source') or {}).get('sha256')
            or result['source_display'] != source_transform_for_run(run)):
        raise ValueError('Beauty layer belongs to a different source or display transform')
    plan = ((run.get('options') or {}).get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
    baseline = plan.get('source_baseline') or (run.get('options') or {}).get('source_only_plan') or {}
    native_size = baseline.get('native_size') or (plan.get('canvas') or {}).get('source_oriented_size')
    if native_size != result['native_size']:
        raise ValueError('Beauty layer belongs to a different native source canvas')
    from native_source_baseline import retouch_identity
    retouch = retouch_identity(run, root=root)
    stable = {key:retouch[key] for key in ('delta_sha256','source_sha256','working_domain')} if retouch else None
    if stable != result.get('native_source_retouch'):
        raise ValueError('Source retouch changed after beauty preparation; prepare beauty again')
    identity_for_run(run)
    return result, path, control


def hair_correction_guard_for_run(run, *, root=None):
    """Reuse qualified source visibility to veto hair corrections, not RGB.

    Beauty opacity and disabled faces are appearance controls, not changes to
    the independently observed source face. This guard never declares skin
    ownership, introduces a target, or modifies the saved semantic labels.
    """
    if not ((run.get('options') or {}).get('appearance_revision') or {}).get('portrait_beauty'):
        return None
    result, path, _ = validate_source_for_run(run, root=root)
    for face in result['faces']:
        composition = face.get('composition') or {}
        if (face.get('composition_policy') != VISIBLE_FACE_POLICY
                or composition.get('policy') != VISIBLE_FACE_POLICY
                or composition.get('qualification_status') not in {'full_visible_face', 'visible_upper_face_only'}
                or composition.get('nonface_material_excluded') is not True
                or any(composition.get(key) is not False for key in
                       ('skin_only', 'features_gate_applied', 'raw_probability_applied', 'source_texture_restoration'))):
            raise ValueError('Hair correction veto requires current qualified visible-face composition support')
    binding = {'policy': HAIR_GUARD_POLICY, 'source_sha256': result['source_sha256'],
        'native_size': copy.deepcopy(result['native_size']),
        'source_display': copy.deepcopy(result['source_display']),
        'native_source_retouch': copy.deepcopy(result.get('native_source_retouch')),
        'composition_policy': VISIBLE_FACE_POLICY,
        'scope': 'source_visible_face_soft_veto_of_hair_B_S_only; not_skin_truth_or_target',
        'faces': [{'id': face['id'], 'native_box': list(face['native_box']),
                   'pixels_sha256': face['pixels']['sha256']} for face in result['faces']]}
    return {'result': result, 'parent': path.parent, 'binding': binding}


def sample_visible_face_support(guard, box, canvas_size):
    """Sample the same frozen native visibility on proxy or native ROI grids."""
    left, top, right, bottom = _box(box, canvas_size)
    support = np.zeros((bottom-top, right-left), np.float32)
    if guard is None:
        return support
    result = guard['result']
    sx, sy = canvas_size[0]/result['native_size'][0], canvas_size[1]/result['native_size'][1]
    for face in result['faces']:
        l,t,r,b = face['native_box']
        x0,y0 = max(left, math.floor(l*sx)), max(top, math.floor(t*sy))
        x1,y1 = min(right, math.ceil(r*sx)), min(bottom, math.ceil(b*sy))
        if x0 >= x1 or y0 >= y1:
            continue
        path = Path(face['pixels']['path'])
        path = path if path.is_absolute() else guard['parent']/path
        stat = path.stat()
        _, weight = _pixels(str(path), stat.st_mtime_ns, stat.st_size,
                            face['pixels']['sha256'], (b-t,r-l))
        values = _sample(weight, (x0/sx-l,y0/sy-t,x1/sx-l,y1/sy-t), (x1-x0,y1-y0))
        region = np.s_[y0-top:y1-top,x0-left:x1-left]
        support[region] = np.maximum(support[region], np.clip(values,0,1))
    return support


def apply_for_run(values, run, *, canvas_size=None, box=None, premultiplied_alpha=None):
    """Consume the beauty revision on both local and prepared reference routes."""
    if not ((run.get('options') or {}).get('appearance_revision') or {}).get('portrait_beauty'):
        return values, {'enabled': False, 'applied': False}
    result, path, control = validate_source_for_run(run)
    strength = control.get('strength', 1.)
    disabled = control.get('disabled_faces') or []
    rendered = apply_result(values, result, parent=path.parent, strength=strength,
        disabled_faces=disabled, canvas_size=canvas_size, box=box,
        premultiplied_alpha=premultiplied_alpha)
    return rendered, {'enabled': True, 'applied': bool(strength and len(disabled) < len(result['faces'])),
        'result': str(path), 'strength': strength, 'disabled_faces': disabled,
        'execution_stage': 'display_source_before_reference_colour_and_scene_lighting',
        'alpha_changed': False, 'source_texture_restoration': False,
        'generated_detail_origin': 'BeautyGRPO_1024_sampling', 'replay_requires_models': False}
