"""Recover a frozen source observation from its unedited paired native bundle.

The identity is photographic observation, not a claim that F/B is uniquely
correct. Reconstruct before downsampling so the alpha/B product stays paired.
"""
from __future__ import annotations

from functools import lru_cache
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def retouch_identity(run, *, root=None):
    """Frozen source-stage identity; never starts recognition or repairs pixels."""
    record = (run.get('options') or {}).get('native_source_retouch')
    if not record:
        return None
    if record.get('schema') != 'automatic_native_retouch_binding.v1' or record.get('source_sha256') != run['source']['sha256']:
        raise ValueError('Frozen automatic retouch is bound to a different RAW root')
    layer_asset = record['layer']
    path = Path(layer_asset['path'])
    path = path if path.is_absolute() else Path(root or run.get('root') or '.')/path
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != layer_asset['sha256']:
        raise ValueError('Frozen native retouch layer bytes changed')
    layer = json.loads(path.read_text(encoding='utf8'))
    if layer['source']['sha256'] != run['source']['sha256'] or layer['native_size'] != record['native_size']:
        raise ValueError('Frozen retouch source/canvas differs from the installed binding')
    from native_retouch_producer import frozen_working_domain
    domain = frozen_working_domain(layer)
    if record['working_domain'] != domain:
        raise ValueError('Frozen retouch binding has an invalid working domain')
    delta_path = Path(layer['delta']['path'])
    delta_path = delta_path if delta_path.is_absolute() else path.parent/delta_path
    with delta_path.open('rb') as stream:
        delta_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    if delta_sha != layer['delta']['sha256']:
        raise ValueError('Frozen native retouch delta bytes changed')
    return {'schema': record['schema'], 'layer_sha256':digest,'delta_sha256':delta_sha,
            'source_sha256':record['source_sha256'],'native_size':record['native_size'],
            'working_domain':domain,'approved_patches':len(layer['patches']),
            'execution_stage':'before_display_alpha_decomposition_and_reference_TBS'}


def open_source_for_run(run, *, root=None):
    """Decode untouched root, consume its frozen source delta exactly once.

    This is for producer decoding before display/alpha decomposition. Saved
    native plans replay their already prepared Fp/B pair via source_for_plan;
    they must not add this difference to that pair again.
    """
    from raw_support import open_linear
    path = Path(run['source']['path'])
    path = path if path.is_absolute() else Path(root or run.get('root') or '.')/path
    values, precision = open_linear(path)
    record = (run.get('options') or {}).get('native_source_retouch')
    if not record:
        return values, precision
    identity = retouch_identity(run,root=root)
    from native_retouch_producer import apply_frozen_retouch, sha256
    if sha256(path) != run['source']['sha256']:
        raise ValueError('Original source changed before frozen retouch decode')
    layer_path = Path(record['layer']['path'])
    layer_path = layer_path if layer_path.is_absolute() else Path(root or run.get('root') or '.')/layer_path
    layer = json.loads(layer_path.read_text(encoding='utf8'))
    expected_domain = 'scene_linear_sRGB' if precision['source_precision']=='raw16_linear' else 'display_linear_sRGB'
    if identity['working_domain'] != expected_domain:
        raise ValueError('RAW/display domain mismatch before native retouch consumption')
    values = apply_frozen_retouch(values,layer,source_sha256=run['source']['sha256'],
        canvas_size=values.shape[1::-1],verify_original=True)
    return values,{**precision,'native_source_retouch':identity,'native_retouch_consumptions':1}


def retouch_baseline_record(run):
    """Carry owned closure into saved baseline without a replay-stage delta."""
    record = (run.get('options') or {}).get('native_source_retouch')
    return copy.deepcopy(record) if record else None


def source_for_reference_prepare(run, *, root=None, binding=None):
    """Decode approved source, apply its explicit native D once, then fit T/B/S.

    Every preparation stage starts from the same original and approved delta.
    This entry point never consumes a saved display-domain Fp/B pair.
    """
    from display_transform import source_transform_for_run, display_linear
    transform = source_transform_for_run(run)
    source, precision = open_source_for_run(run, root=root)
    baseline = run['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']
    if (precision['source_precision'] != baseline['source_precision']
            or precision['decode_version'] != baseline['decode_version']):
        raise ValueError('Decoded source precision/decoder differs from its preparation baseline')
    retouch = precision.get('native_source_retouch')
    if binding is not None:
        if (binding.get('source') != run['source']['sha256']
                or binding.get('decode_version') != precision['decode_version']
                or binding.get('source_display') != transform
                or 'native_source_retouch' not in binding
                or binding['native_source_retouch'] != retouch):
            raise ValueError('Preparation parameters belong to a different approved source or source transform')
    for top in range(0, len(source), 128):
        source[top:top+128] = display_linear(source[top:top+128], transform)
    return source, {**precision, 'source_display_consumptions': 1}, transform


def enabled(plan):
    return (plan.get('execution') or {}).get('source_reference') == 'paired_native_baseline'


@lru_cache(maxsize=2)
def _baseline(fp_path, fp_modified, alpha_path, alpha_modified, bg_path, bg_modified, target_size):
    from design_reconstruction import _planes
    fp = _planes(fp_path, fp_modified)['values']
    alpha = _planes(alpha_path, alpha_modified)['values'][0]
    background = _planes(bg_path, bg_modified)['values']
    width, height = alpha.size
    native = tuple(target_size) == (width, height)
    if native:
        original = np.empty((height, width, 3), np.float32)
    else:
        # Reconstruct the paired product at native resolution before reducing.
        # A proxy needs one native channel at a time, not an entire native RGB
        # image retained just to make a small preview.
        original = np.empty((target_size[1], target_size[0], 3), np.float32)
        plane = np.empty((height, width), np.float32)
    for channel in range(3):
        output = original[..., channel] if native else plane
        for top in range(0, height, 256):
            box = (0, top, width, min(height, top + 256))
            uncovered = 1 - np.asarray(alpha.crop(box), np.float32)
            output[top:box[3]] = np.asarray(fp[channel].crop(box), np.float32)
            output[top:box[3]] += uncovered * np.asarray(background[channel].crop(box), np.float32)
        if not native:
            original[..., channel] = np.asarray(Image.fromarray(plane).resize(
                target_size, Image.Resampling.BILINEAR), np.float32)
    return original


def source_for_plan(plan, size=None):
    baseline=plan.get('source_baseline') or {}
    if (baseline.get('schema')!='appearance_source_baseline.v1'
            or baseline.get('working_domain')!='display_linear_sRGB'):
        raise ValueError('Source replay requires the current explicit display-domain baseline')
    assets = (plan.get('source_baseline') or {}).get('assets')
    if not isinstance(assets,dict):raise ValueError('Current source replay requires its immutable original paired assets')
    fp, alpha, bg = (assets[key] for key in
        ('foreground_premultiplied','foreground_alpha','old_environment'))
    target = tuple(size or alpha['size'])
    files = [Path(asset['path']) for asset in (fp, alpha, bg)]
    return _baseline(str(files[0]), files[0].stat().st_mtime_ns,
        str(files[1]), files[1].stat().st_mtime_ns,
        str(files[2]), files[2].stat().st_mtime_ns, target)


def background_for_view(source_view, foreground_view, alpha_view):
    """Effective old B for the paired reduced view, not independent B resampling."""
    uncovered = 1 - alpha_view
    contribution = np.maximum(source_view - foreground_view, 0)
    return np.divide(contribution, uncovered[..., None], out=np.zeros_like(contribution),
                     where=uncovered[..., None] > 0)


def prepare_source_only(source_linear, run, precision):
    """Establish a current local source plan and consume its source display once."""
    from display_transform import source_transform_for_precision,validate_source_transform,display_linear
    from raw_support import RGB_DECODE_VERSION
    if run.get('reference') or ((run.get('options') or {}).get('appearance_revision') or {}).get('design_reconstruction_plan'):
        raise ValueError('Source-only preparation cannot inherit reference rendering assets')
    source=run.get('source') or {}
    if not source.get('sha256') or precision.get('decode_version')!=RGB_DECODE_VERSION:
        raise ValueError('Current source-only preparation requires a bound source and current decoder')
    if precision.get('source_precision') not in {'raw16_linear','raster8_srgb'}:
        raise ValueError('Unsupported source precision; decode the original through the current source decoder')
    existing=(run.get('options') or {}).get('source_only_plan')
    if existing:
        if (existing.get('schema')!='source_only_display_plan.v1' or existing.get('working_domain')!='display_linear_sRGB'
                or existing.get('source_sha256')!=source['sha256']
                or existing.get('decode_version')!=RGB_DECODE_VERSION
                or existing.get('source_precision')!=precision['source_precision']
                or existing.get('native_size')!=list(source_linear.shape[1::-1])
                or existing.get('reference_assets_used') is not False):
            raise ValueError('Source-only plan is not bound to the current source')
        transform=validate_source_transform(existing['source_display'], precision['source_precision'])
        if existing.get('source_display_applied_once') is not True:
            raise ValueError('Source-only baseline must record one source preparation consumption')
    else:transform=source_transform_for_precision(precision['source_precision'])
    values=display_linear(np.asarray(source_linear,np.float32),transform)
    plan={'schema':'source_only_display_plan.v1','working_domain':'display_linear_sRGB',
        'source_sha256':source['sha256'],'native_size':list(values.shape[1::-1]),
        'source_precision':precision['source_precision'],'decode_version':RGB_DECODE_VERSION,
        'source_display':transform,'source_display_applied_once':True,'reference_assets_used':False}
    return values,plan
