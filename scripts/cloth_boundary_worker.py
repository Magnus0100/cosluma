"""Automatically prepare an independent native shoulder/sleeve light layer."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

from cloth_boundary_runtime import KIND, SCHEMA, RECIPE, endpoint, endpoint_binding, validate_layer
from boundary_light_runtime import sanitize_endpoint, source_owners
from native_subject_detail_runtime import array_sha
from reference_cloth_boundary_transfer import prepare_cloth_boundary_light
from asset_identity import sha256_file

# The current cloth transfer contract is intentionally single-owner and uses
# a fixed reference-depth budget. Keep the worker's sampling gate independent
# from the producer module object so offline producer tests can substitute the
# producer without weakening the production contract.
CLOTH_OWNER = 23
CLOTH_MAX_DEPTH_REFERENCE_PIXELS = 24.0


def sha(path):
    return sha256_file(path)


PREFLIGHT_CACHE_SCHEMA = 'source_boundary_registration_preflight.v1'


def _without_paths(value):
    """Make a path-independent identity for a registration preflight."""
    if isinstance(value, dict):
        return {key: _without_paths(item) for key, item in sorted(value.items())
                if key not in {'path', 'root', 'analysis_directory', 'manifest', 'event_log'}}
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_without_paths(item) for item in value]
    return value


def _preflight_cache_key(root, run, plan):
    from runtime_resources import code_path
    code = {name: sha(code_path(name)) for name in (
        'cloth_boundary_worker.py', 'boundary_light_worker.py',
        'boundary_light_geometry_context.py', 'reference_cloth_boundary_transfer.py',
        'source_foreground_cache.py')}
    environment = next((row for row in plan.get('layers', [])
                        if row.get('kind') == 'source_environment_response'), {})
    environment_assets = environment.get('assets') or {}
    masks = plan.get('mask_roles') or {}
    relevant_masks = {name: (masks.get(name) or {}).get('sha256') for name in (
        'foreground_alpha', 'source_semantics', 'identity_protection',
        'candidate_occupancy')}
    registry = (run.get('options') or {}).get('model_registry')
    registry_path = Path(registry) if registry else None
    if registry_path is not None and not registry_path.is_absolute():
        registry_path = Path(root).resolve() / registry_path
    registry_sha = sha(registry_path) if registry_path is not None and registry_path.is_file() else None
    payload = {'schema': PREFLIGHT_CACHE_SCHEMA,
        'source_sha256': run['source']['sha256'],
        'reference_sha256': run['reference']['sha256'],
        'model_registry_sha256': registry_sha,
        'canvas': _without_paths(plan.get('canvas') or {}),
        'mask_roles': relevant_masks,
        'environment_observation': {
            key: value.get('sha256') for key, value in environment_assets.items()
            if key in {'qualification_receipt', 'observation_proof', 'reference_labels'}
            and isinstance(value, dict)},
        'code': code}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf8')
    return hashlib.sha256(encoded).hexdigest(), payload


def _preflight_cache_path(root, key):
    return Path(root).resolve() / 'outputs' / 'cache' / 'boundary-registration' / (key + '.json')


def _read_preflight_cache(root, key, payload):
    path = _preflight_cache_path(root, key)
    try:
        value = json.loads(path.read_text(encoding='utf8'))
        if (value.get('schema') != PREFLIGHT_CACHE_SCHEMA
                or value.get('key') != key or value.get('identity') != payload
                or not isinstance(value.get('result'), dict)):
            return None
        return value['result']
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _write_preflight_cache(root, key, payload, result):
    path = _preflight_cache_path(root, key)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps({'schema': PREFLIGHT_CACHE_SCHEMA, 'key': key,
            'identity': payload, 'result': result}, ensure_ascii=False,
            indent=2, allow_nan=False) + '\n', encoding='utf8')
        temporary.replace(path)
    except (OSError, TypeError, ValueError):
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def asset(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', 'utf8')


from boundary_light_worker import (observations, replacement_endpoint, replace_layer,
    identical_frozen_output, reuse_current_layer, save_current_snapshot, producer_code)


def native_geometry(root, run, plan, maps, proof, box, reference_size, observation_reference_size,
                    *, reuse_preflight=False):
    """Build the same native geometry once for this coordinated preparation."""
    from boundary_light_geometry_context import (ExactMaskDistance, NativeClothGeometry,
        current_geometry_context, geometry_key)
    from design_reconstruction import sample_asset
    key = geometry_key(root, run, plan, proof, box, reference_size, observation_reference_size, maps=maps)
    context = current_geometry_context()
    if reuse_preflight and key is not None:
        shared = context.take(key)
        if shared is not None:
            return shared
    native = tuple(plan['canvas']['source_oriented_size'])
    proxy = tuple(plan['canvas']['source_proxy_size'])
    a = sample_asset(plan['mask_roles']['foreground_alpha'], 'values', box, native)
    sl = source_owners(plan, box).astype(np.uint8)
    x = (np.arange(box[0], box[2], dtype=np.float32)+.5)*proxy[0]/native[0]-.5
    y = (np.arange(box[1], box[3], dtype=np.float32)+.5)*proxy[1]/native[1]-.5
    qy, qx = np.meshgrid(y, x, indexing='ij')
    native_maps = {name: map_coordinates(value, [qy, qx], order=0 if name == 'valid' else 1,
        mode='nearest', prefilter=False) for name, value in maps.items()}
    protection = sample_asset(plan['mask_roles']['identity_protection'], 'protection', box, native)
    background = next(row for row in plan['layers'] if row['kind'] == 'background_design')['assets']
    validity = sample_asset(background['validity'], 'values', box, native)
    geometry = NativeClothGeometry(a, sl, native_maps, protection, validity,
                                  ExactMaskDistance.compute(sl == CLOTH_OWNER))
    if key is not None and not reuse_preflight:
        context.store(key, geometry)
    return geometry


def registration_upper_bound(root, run, plan):
    """Reject only a proved impossible registration before native RGB work.

    Treat every source endpoint and reference F/B pixel as qualified. The
    exact production anchor sampler still needs 48 alpha/owner/map anchors;
    failure under this relaxed domain cannot be repaired by computing RGB.
    """
    import workflow_timing
    from source_foreground_cache import prepare_source_foreground
    from reference_cloth_boundary_transfer import _fit_affine
    from raw_support import open_rgb
    native = tuple(plan['canvas']['source_oriented_size'])
    if native[0]*native[1] < 1_000_000:
        return None
    cache_key, cache_identity = _preflight_cache_key(root, run, plan)
    with workflow_timing.span('boundary_preflight_cache_lookup', model_request=False, http_request=False):
        cached = _read_preflight_cache(root, cache_key, cache_identity)
    if cached is not None:
        cached = dict(cached)
        cached['preflight_cache_hit'] = True
        return cached
    with workflow_timing.span('boundary_preflight_observations', model_request=False, http_request=False):
        labels, ref_labels, cached_ref, maps, proof = observations(root, run, plan)
    with open_rgb(Path(run['reference']['path'])) as image:
        rw, rh = image.size
    ch, cw = cached_ref.shape[:2]
    if (rw, rh) != (cw, ch):
        maps = dict(maps)
        maps['ref_x'] = (maps['ref_x']+.5)*rw/cw-.5
        maps['ref_y'] = (maps['ref_y']+.5)*rh/ch-.5
        ref_labels = np.asarray(Image.fromarray(ref_labels).resize((rw, rh), Image.Resampling.NEAREST))
    with workflow_timing.span('boundary_preflight_reference_matte', model_request=False, http_request=False):
        matte = prepare_source_foreground(root, Path(run['reference']['path']), source_hash=run['reference']['sha256'],
            saliency_path=plan['mask_roles']['candidate_occupancy']['path'],
            semantics_path=proof['reference_labels']['path'], registry_path=run['options'].get('model_registry'))
    with np.load(matte['alpha_path'], allow_pickle=False) as data:
        ra = data['values']
    proxy = tuple(plan['canvas']['source_proxy_size'])
    yy, xx = np.where(labels == CLOTH_OWNER)
    if not len(xx):
        result = {'rejected':True, 'reason':'no_source_cloth_owner',
            'reference_matte_cache_hit':matte['cache_hit'],
            'model_tile_inferences_this_attempt':0 if matte['cache_hit'] else matte['tiles']['alpha_tiles']}
        _write_preflight_cache(root, cache_key, cache_identity, result)
        return result
    margin = int(np.ceil(100*max(native[0]/rw, native[1]/rh)))
    box = (max(0, int(xx.min()*native[0]/proxy[0])-margin), max(0, int(yy.min()*native[1]/proxy[1])-margin),
        min(native[0], int((xx.max()+1)*native[0]/proxy[0])+margin), min(native[1], int((yy.max()+1)*native[1]/proxy[1])+margin))
    with workflow_timing.span('boundary_preflight_geometry', model_request=False, http_request=False):
        geometry = native_geometry(root, run, plan, maps, proof, box, (rw, rh), (cw, ch))
    a, sl, native_maps = geometry.alpha, geometry.owners, geometry.maps
    protection, validity = geometry.protection, geometry.validity
    expected_scale = float(np.sqrt(rw*rh/max(native[0]*native[1], 1)))
    max_depth = int(np.ceil(CLOTH_MAX_DEPTH_REFERENCE_PIXELS*max(1., 1/max(expected_scale, 1e-6))))
    owner = sl == CLOTH_OWNER
    anchor_mask = owner & (geometry.owner_distance.for_mask(owner) <= max_depth)
    with workflow_timing.span('boundary_preflight_registration', model_request=False, http_request=False):
        affine, receipt = _fit_affine(native_maps, sl, a, validity, protection, ref_labels,
            ((ra >= .95)&(ref_labels == CLOTH_OWNER)).astype(np.float32),
            np.ones(a.shape, bool), anchor_mask)
    impossible = receipt.get('reason') in ('insufficient_opaque_same_owner_registration_anchors',
                                           'collinear_registration_anchors')
    result = {'rejected':impossible, 'registration':receipt, 'reason':receipt.get('reason'),
        'reference_matte_cache_hit':matte['cache_hit'],
        'model_tile_inferences_this_attempt':0 if matte['cache_hit'] else matte['tiles']['alpha_tiles'],
        'source_sha256':run['source']['sha256'], 'reference_sha256':run['reference']['sha256'],
        'endpoint_qualification_relaxed_to_true':True, 'reference_FB_qualification_relaxed_to_true':True}
    _write_preflight_cache(root, cache_key, cache_identity, result)
    return result


def prepare(root, run, output):
    from design_reconstruction import sample_asset, _manual_light_plans
    from subject_environment_worker import load_observations
    from source_foreground_cache import prepare_source_foreground
    from float_appearance import srgb_to_linear, linear_to_srgb
    from raw_support import open_rgb
    from native_roi_render import has_active_operations
    root, output = Path(root).resolve(), Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    original_plan, plan, previous = replacement_endpoint(run, KIND)
    # The measured endpoint is before manual finishing. Denoise/matte edits
    # alter its upstream pixels and need their own observation producer.
    if run['options'].get('denoise_strength') or plan['mask_roles'].get('edited_edge_light_applicability'):
        raise ValueError('Edited matte or denoise requires a new boundary observation')
    if has_active_operations(run, ('foreground', 'composite')) or any(
            value.get('enabled') and value.get('light_fields') for key, value in _manual_light_plans(run).items()
            if key in ('foreground', 'composite')) or any(
            value.get('enabled') or value.get('strokes') for value in (run['options'].get('manual_layers') or {}).values()):
        raise ValueError('Active downstream finishing requires independently observed final pixels')
    # Reusing a current frozen cloth layer avoids reopening the reference and
    # rerunning the native ROI correspondence when a coordinated retry has no
    # endpoint change.  The shared helper verifies the frozen assets and the
    # complete endpoint binding before allowing this fast path.
    import cloth_boundary_runtime as _runtime
    reused = reuse_current_layer(previous, plan, run, _runtime)
    if reused is not None:
        write(output/'qualification.json', reused['receipt'])
        return reused
    labels, ref_labels, cached_ref, maps, proof = observations(root, run, plan)
    reference = srgb_to_linear(np.asarray(open_rgb(Path(run['reference']['path'])), np.float32)/255)
    rh, rw = reference.shape[:2]; ch, cw = cached_ref.shape[:2]
    if (rw, rh) != (cw, ch):
        maps = dict(maps)
        maps['ref_x'] = (maps['ref_x']+.5)*rw/cw-.5; maps['ref_y'] = (maps['ref_y']+.5)*rh/ch-.5
        ref_labels = np.asarray(Image.fromarray(ref_labels).resize((rw, rh), Image.Resampling.NEAREST))
    matte = prepare_source_foreground(root, Path(run['reference']['path']), source_hash=run['reference']['sha256'],
        saliency_path=plan['mask_roles']['candidate_occupancy']['path'],
        semantics_path=proof['reference_labels']['path'], registry_path=run['options'].get('model_registry'))
    def values(key):
        with np.load(matte[key], allow_pickle=False) as data: return data['values'].copy()
    ra, fp, fb = values('alpha_path'), values('premultiplied_foreground_path'), values('unknown_edge_support_path')
    if ra.shape != (rh, rw): raise ValueError('Reference observations must use the original reference grid')
    bounded = np.all(fp <= ra[..., None]+2e-6, -1); fb[~bounded] = 0; fp[~bounded] = 0
    native = tuple(plan['canvas']['source_oriented_size']); proxy = tuple(plan['canvas']['source_proxy_size'])
    yy, xx = np.where(labels == 23)
    if not len(xx): raise ValueError('Source has no observed outer clothing')
    # Padding follows the largest reference-space measurement kernel.
    margin = int(np.ceil(100*max(native[0]/rw, native[1]/rh)))
    box = (max(0, int(xx.min()*native[0]/proxy[0])-margin), max(0, int(yy.min()*native[1]/proxy[1])-margin),
        min(native[0], int((xx.max()+1)*native[0]/proxy[0])+margin), min(native[1], int((yy.max()+1)*native[1]/proxy[1])+margin))
    geometry = native_geometry(root, run, plan, maps, proof, box, (rw, rh), (cw, ch), reuse_preflight=True)
    a, sl, native_maps = geometry.alpha, geometry.owners, geometry.maps
    protection, v = geometry.protection, geometry.validity
    p, endpoint_clipped_pixels = sanitize_endpoint(endpoint(plan, run, box, native))
    bg_assets = next(row for row in plan['layers'] if row['kind'] == 'background_design')['assets']
    # Reconstruct the immutable paired source photograph. F/B separation can
    # produce Fp > alpha, so dividing it is neither the original RGB nor a
    # valid material-colour observation. Match source_for_plan's exact product.
    original = plan['source_baseline']['assets']
    source_rgb = sample_asset(original['foreground_premultiplied'], 'values', box, native)
    original_alpha = sample_asset(original['foreground_alpha'], 'values', box, native)
    old_environment = sample_asset(original['old_environment'], 'values', box, native)
    source_rgb += (1-original_alpha[..., None])*old_environment
    # Use the observed cloth-owner contour for affine registration. The full
    # foreground alpha contour only contains the outside silhouette; on poses
    # where the garment meets skin, hair or another subject owner, that contour
    # has no owner-23 samples and can reject an otherwise source-bound pair.
    # Measuring depth inside owner 23 keeps anchors on the actual garment
    # boundary while retaining the existing owner, alpha, validity, opaque
    # reference and registration-error gates.
    source_inside = sl == CLOTH_OWNER
    source_depth = geometry.owner_distance.for_mask(source_inside)
    expected_scale = float(np.sqrt((rw * rh) / max(native[0] * native[1], 1)))
    max_native_depth = int(np.ceil(CLOTH_MAX_DEPTH_REFERENCE_PIXELS
                                  * max(1., 1. / max(expected_scale, 1e-6))))
    registration_anchor_mask = source_inside & (source_depth <= max_native_depth)
    result = prepare_cloth_boundary_light(p, a, reference, ra, sl, ref_labels, native_maps,
        source_rgb=source_rgb,
        validity=v, identity_protection=protection, reference_fp=fp, reference_fb_support=fb,
        registration_anchor_mask=registration_anchor_mask)
    delta = result['delta_p']; changed = int(np.any(delta != 0, -1).sum())
    receipt = {**result['receipt'], 'schema': SCHEMA, 'recipe': RECIPE, 'changed_pixels': changed,
        'producer_code': producer_code(KIND),
        'native_output': True, 'model_tile_inferences_this_attempt': 0 if matte['cache_hit'] else matte['tiles']['alpha_tiles'],
        'reference_matte_cache_hit': matte['cache_hit'], 'image_generation_submissions': 0,
        'artistic_acceptance': False, 'clothing_enabled': True, 'hair_changed': False, 'native_box': list(box),
        'endpoint_clipped_pixels': endpoint_clipped_pixels}
    components = receipt.get('components')
    if isinstance(components, list) and components and isinstance(components[0], dict):
        registration = components[0].setdefault('registration', {})
        registration['anchor_band_native_depth_pixels'] = max_native_depth
        registration['anchor_band_reference_depth_pixels'] = CLOTH_MAX_DEPTH_REFERENCE_PIXELS
        registration['anchor_scale_estimate'] = expected_scale
    write(output/'qualification.json', receipt)
    if not changed: return {'changed': False, 'receipt': receipt, 'design_reconstruction_plan': copy.deepcopy(original_plan)}
    # Crop unused padding from the frozen delta and its bound input patch.
    cy, cx = np.where(np.any(delta != 0, -1)); sub = (slice(cy.min(), cy.max()+1), slice(cx.min(), cx.max()+1))
    patch_box = (box[0]+int(cx.min()), box[1]+int(cy.min()), box[0]+int(cx.max())+1, box[1]+int(cy.max())+1)
    delta = np.ascontiguousarray(delta[sub]); target = endpoint(plan, run, patch_box, native)
    delta_path = output/'native-delta.npz'; np.savez_compressed(delta_path, patch_0=delta)
    binding = endpoint_binding(plan, run)
    frozen = {'schema': SCHEMA, 'recipe': RECIPE, 'binding': binding, 'delta': asset(delta_path),
        'producer_code': receipt['producer_code'],
        'patches': [{'native_box': list(patch_box), 'array_key': 'patch_0', 'delta_array_sha256': array_sha(delta),
            'changed_pixels': changed, 'target_patch_sha256': array_sha(target)}],
        'observation_assets': {'reference_alpha': asset(matte['alpha_path']), 'reference_fp': asset(matte['premultiplied_foreground_path']),
            'reference_FB_support': asset(matte['unknown_edge_support_path']), 'reference_FB_diagnostics': asset(matte['fb_diagnostics_path']),
            'reference_matte_producer': asset(matte['result_path'])}}
    write(output/'frozen.json', frozen)
    if identical_frozen_output(previous, frozen):
        return {'changed': False, 'receipt': {**receipt, 'state': 'no_change', 'reason': 'identical_frozen_pixels'},
            'design_reconstruction_plan': copy.deepcopy(original_plan)}
    layer = {'id': KIND, 'kind': KIND, 'label': '协调肩袖暖光', 'enabled': True, 'strength': 1.,
        'params': {'schema': SCHEMA, 'recipe': RECIPE, 'binding': binding},
        'assets': {'frozen_delta': asset(output/'frozen.json'), 'qualification_receipt': asset(output/'qualification.json')}}
    result_plan = replace_layer(plan, previous, layer)
    if validate_layer({**layer, 'enabled': True, 'strength': 1.}, result_plan, run)['bypass_reason']:
        raise ValueError('Prepared cloth light does not match its current native endpoint')
    # Full-size context boards are a human debugging aid, not part of the
    # frozen layer contract.  Encoding three native PNGs here adds avoidable
    # wall time and duplicates the review renderer's work, so keep them
    # opt-in for an explicit diagnostic run.
    if os.environ.get('COSPLAY_BOUNDARY_DEBUG_IMAGES') == '1':
        bg = sample_asset(bg_assets['background'], 'values', box, native)
        before = p+(1-a[..., None])*bg; after = before+result['delta_p']
        def image(rgb): return Image.fromarray(np.uint8(np.clip(np.rint(linear_to_srgb(rgb)*255), 0, 255)))
        image(before).save(output/'current-native-context.png'); image(after).save(output/'candidate-native-context.png')
        w, h = before.shape[1], before.shape[0]; sheet = Image.new('RGB', (2*w, h+32), '#222222')
        sheet.paste(image(before), (0, 32)); sheet.paste(image(after), (w, 32)); draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), 'current native', fill='white'); draw.text((w+8, 8), 'observed cloth boundary light', fill='white')
        sheet.save(output/'native-comparison.png')
    return {'changed': True, 'receipt': receipt, 'design_reconstruction_plan': result_plan}


def prepare_for_workbench(root, run, output, *, cancel=None, progress=None):
    """Isolate model preparation from the desktop runtime and saved parent."""
    from current_model_registry import load_registry
    from runtime_resources import code_path
    root, output = Path(root).resolve(), Path(output).resolve()
    if cancel and cancel(): raise InterruptedError('Cloth-light preparation cancelled')
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.with_name(output.name+'-input.json'); write(snapshot, run)
    python = load_registry(root, registry_path=run['options'].get('model_registry'))['python']
    log = output.with_name(output.name+'-worker.log')
    if progress: progress('cloth_boundary_light', '正在观察参考肩袖材质并准备原尺寸受光层；不新增生图。')
    with log.open('wb') as stream:
        process = subprocess.Popen([python, '-B', '-X', 'utf8', str(code_path('cloth_boundary_worker.py')),
            '--root', str(root), '--run-json', str(snapshot), '--output', str(output)], cwd=str(root),
            stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        def stop():
            if process.poll() is not None: return
            if os.name == 'nt':
                # The sole PID is the known task worker. Include its matting
                # child so cancellation releases GPU work before another job.
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            else: process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        try:
            while process.poll() is None:
                if cancel and cancel():
                    stop(); raise InterruptedError('Cloth-light preparation cancelled; saved parent retained')
                time.sleep(.25)
        finally: stop()
    if process.returncode:
        raise ValueError('Cloth-light preparation failed: '+log.read_text('utf8', errors='replace')[-2200:])
    if cancel and cancel(): raise InterruptedError('Cloth-light preparation cancelled; result not applied')
    result = json.loads((output/'result.json').read_text('utf8'))
    return {**result, 'ok': True, 'no_change': not result['changed'],
        'cache_hit': bool(result['receipt'].get('reference_matte_cache_hit')),
        'design_reconstruction_plan': result.get('design_reconstruction_plan', run['options']['appearance_revision']['design_reconstruction_plan'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--run-id'); inputs.add_argument('--run-json', type=Path)
    # Child revision publication belongs to the backend coordinator. The
    # worker only prepares and persists its isolated result receipt.
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import single_source_workflow as workflow
    run = json.loads(args.run_json.read_text('utf8')) if args.run_json else workflow.load_run(args.root, args.run_id)[1]
    from native_cuda_ndimage import cuda_ndimage_scope
    with cuda_ndimage_scope():
        result = prepare(args.root, run, args.output)
    write(args.output/'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'design_reconstruction_plan'}, ensure_ascii=False))


if __name__ == '__main__': main()
