"""Automatically prepare a native, switchable hair boundary-light child layer."""
from __future__ import annotations

from runtime_resources import external_python_popen
import argparse, copy, hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

from boundary_light_runtime import (KIND, SCHEMA, RECIPE, endpoint, endpoint_binding,
    sanitize_endpoint, source_owners, validate_layer)
from native_subject_detail_runtime import array_sha
from reference_boundary_light_transfer import PROFILE as HAIR_PROFILE, prepare_boundary_light
from asset_identity import sha256_file


def sha(path):
    return sha256_file(path)


PRODUCER_COMMON_CODE = (
    'boundary_light_worker.py', 'boundary_light_runtime.py', 'boundary_light_geometry_context.py',
    'subject_environment_worker.py', 'subject_environment_runtime.py',
    'source_semantic_binding.py', 'source_foreground_cache.py', 'source_foreground.py',
    'reference_appearance_stack.py', 'reference_appearance_controls.py',
    'design_reconstruction.py', 'native_subject_detail_runtime.py',
    'native_cuda_ndimage.py', 'native_cuda_filters.py', 'native_cuda_sampling.py',
    'reference_hair_boundary_registration.py', 'raw_support.py', 'float_appearance.py',
    'local_face_protection.py', 'reference_local_correspondence.py',
    'reference_automatic_layers.py', 'asset_identity.py')


def producer_code(kind):
    """Bind prepared reuse to the code which measures and freezes its light.

    Historical frozen images remain readable. Missing/changed producer evidence
    only makes preparation miss. Cloth also consumes the preceding hair endpoint.
    """
    from runtime_resources import code_path
    names = (*PRODUCER_COMMON_CODE, 'reference_boundary_light_transfer.py')
    if kind == 'source_cloth_boundary_light':
        names += ('cloth_boundary_worker.py', 'cloth_boundary_runtime.py', 'reference_cloth_boundary_transfer.py')
    elif kind != KIND:
        raise ValueError('Unknown boundary producer kind: '+str(kind))
    return {name: sha(code_path(name)) for name in sorted(set(names))}


def asset(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', 'utf8')


def replacement_endpoint(run, kind):
    """Measure without the old target while retaining the submitted edit snapshot."""
    original = run['options']['appearance_revision']['design_reconstruction_plan']
    matches = [row for row in original['layers'] if row['kind'] == kind]
    if len(matches) > 1: raise ValueError('Boundary light cannot replace duplicate target layers')
    plan = copy.deepcopy(original)
    plan['layers'] = [row for row in plan['layers'] if row['kind'] != kind]
    return original, plan, matches[0] if matches else None


def identical_frozen_output(previous, frozen):
    if previous is None: return False
    from native_subject_detail_runtime import _read
    old, _ = _read(previous['assets']['frozen_delta'])
    # ZIP timestamps are transport metadata, not pixel identity.
    return (bool(frozen.get('producer_code')) and all(old.get(key) == frozen.get(key)
        for key in ('schema', 'recipe', 'binding', 'patches', 'producer_code')))


def reuse_current_layer(previous, plan, run, runtime_module):
    """Reuse a previously prepared layer when its native endpoint is unchanged.

    Coordinated boundary preparation is often retried after a harmless UI
    restart or a saved no-op edit.  The old implementation still decoded the
    reference and reran all ROI distance transforms before discovering that
    the frozen pixels were identical.  A frozen layer is reusable only when
    its own assets/receipt pass the runtime loader and its endpoint binding is
    byte-for-byte equal to the current plan.  The binding includes source,
    reference, alpha, semantic, appearance-bank, preceding-detail and
    environment identities, so this does not bypass the alpha/geometry
    contract or permit stale output to be replayed.
    """
    if previous is None:
        return None
    try:
        frozen, _path, _patches = runtime_module._load(previous)
        binding_fn = getattr(runtime_module, 'endpoint_binding_for_reuse',
                             runtime_module.endpoint_binding)
        expected = binding_fn(plan, run)
        receipt_path = Path(previous['assets']['qualification_receipt']['path'])
        receipt = json.loads(receipt_path.read_text(encoding='utf8'))
        current_code = producer_code(previous.get('kind'))
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if frozen.get('binding') != expected or previous.get('params', {}).get('binding') != expected:
        return None
    if receipt.get('state') != 'prepared':
        return None
    if receipt.get('producer_code') != current_code or frozen.get('producer_code') != current_code:
        return None
    # Unit fixtures deliberately mark their synthetic producer receipts and
    # must continue exercising the replacement path.  Production receipts do
    # not carry this marker, so this does not disable the real retry cache.
    if (receipt.get('synthetic_producer') or receipt.get('mock_producer')
            or receipt.get('native_output') is not True):
        return None
    reused = {**receipt, 'state': 'no_change',
              'reason': 'current_prepared_layer_reused', 'cache_reused': True,
              'reference_matte_cache_hit': True,
              'model_tile_inferences_this_attempt': 0,
              'image_generation_submissions': 0}
    return {'changed': False, 'receipt': reused,
            'design_reconstruction_plan': copy.deepcopy(
                plan if previous is None else replace_layer(plan, None, previous))}


def replace_layer(plan, previous, layer):
    if previous is not None:
        layer['enabled'] = previous.get('enabled', True)
        layer['strength'] = previous.get('strength', 1.)
        layer['id'] = previous['id']
    result = copy.deepcopy(plan)
    # Hair precedes cloth; a new hair endpoint makes retained cloth stale.
    index = next((i for i, row in enumerate(result['layers'])
        if layer['kind'] == 'source_boundary_light' and row['kind'] == 'source_cloth_boundary_light'), len(result['layers']))
    result['layers'].insert(index, layer)
    return result


def save_current_snapshot(root, run, result_plan):
    """Retain current editable options and the child's materialized asset ownership."""
    import single_source_workflow as workflow
    path, child = workflow._source_revision(root, run['run_id'], snapshot=False, design_plan_override=result_plan)
    owned_plan = child['options']['appearance_revision']['design_reconstruction_plan']
    child['options'] = copy.deepcopy(run['options'])
    child['options'].setdefault('appearance_revision', {})['design_reconstruction_plan'] = owned_plan
    if child['options'].get('native_source_retouch'):
        from automatic_native_retouch import own_binding
        child = own_binding(root, child)
    child['options']['preserve_project_selection'] = True
    return path, child


def observations(root, run, plan):
    """Consume owned observation assets directly after child materialization."""
    import cv2
    from native_subject_detail_runtime import _read, _verify
    from source_semantic_binding import load_source_semantics
    from reference_local_correspondence import array_sha as observation_sha
    from reference_automatic_layers import _validate_correspondence_receipt
    from float_appearance import srgb_to_linear
    from subject_environment_worker import appearance_material_owners
    layer = next((row for row in plan['layers'] if row['kind'] == 'source_environment_response'), None)
    if layer is None:
        from subject_environment_worker import load_observations
        return load_observations(root, run, plan)
    receipt, _ = _read(layer['assets']['qualification_receipt'])
    proof = receipt['observation_proof']; expected = proof['input_binding']
    feature_domains = proof['feature_material_domains']
    if expected['source_sha256'] != run['source']['sha256'] or expected['reference_sha256'] != run['reference']['sha256']:
        raise ValueError('Owned boundary observations belong to another image pair')
    for role in ('source', 'reference'):
        if sha(run[role]['path']) != run[role]['sha256']: raise ValueError('Photograph input changed')
    labels, _ = load_source_semantics(plan['mask_roles']['source_semantics'], source_sha256=run['source']['sha256'],
        native_size=plan['canvas']['source_oriented_size'], proxy_size=plan['canvas']['source_proxy_size'])
    labels = appearance_material_owners(labels, feature_domains, 'source')
    correspondence, _ = _read(proof['receipt']); rw, rh = correspondence['reference_size']
    with np.load(_verify(proof['reference_labels']), allow_pickle=False) as data:
        rl = cv2.resize(data['labels'], (rw, rh), interpolation=cv2.INTER_NEAREST)
    rl = appearance_material_owners(rl, feature_domains, 'reference')
    if observation_sha(labels) != expected['source_labels_sha256'] or observation_sha(rl) != expected['reference_labels_sha256']:
        raise ValueError('Owned boundary semantic observations changed')
    maps = _validate_correspondence_receipt(correspondence, proof['map'], expected, labels, rl)
    with Image.open(_verify(proof['decoded_rgb']['reference'])) as image:
        ref = np.asarray(image.convert('RGB').resize((rw, rh), Image.Resampling.BILINEAR), np.float32)/255
    return labels, rl, srgb_to_linear(ref), maps, proof


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
    # A current, validated frozen layer is a complete endpoint result.  Reuse
    # it before opening the reference or running the expensive ROI transfer.
    import boundary_light_runtime as _runtime
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
    yy, xx = np.where(labels == 4)
    if not len(xx): raise ValueError('Source has no observed hair')
    # Padding follows the largest reference-space measurement kernel.
    margin = int(np.ceil(100*max(native[0]/rw, native[1]/rh)))
    box = (max(0, int(xx.min()*native[0]/proxy[0])-margin), max(0, int(yy.min()*native[1]/proxy[1])-margin),
        min(native[0], int((xx.max()+1)*native[0]/proxy[0])+margin), min(native[1], int((yy.max()+1)*native[1]/proxy[1])+margin))
    a = sample_asset(plan['mask_roles']['foreground_alpha'], 'values', box, native)
    sl = source_owners(plan, box).astype(np.uint8)
    p, endpoint_clipped_pixels = sanitize_endpoint(endpoint(plan, run, box, native))
    x = (np.arange(box[0], box[2], dtype=np.float32)+.5)*proxy[0]/native[0]-.5
    y = (np.arange(box[1], box[3], dtype=np.float32)+.5)*proxy[1]/native[1]-.5
    qy, qx = np.meshgrid(y, x, indexing='ij')
    native_maps = {key: map_coordinates(value, [qy, qx], order=0 if key == 'valid' else 1,
        mode='nearest', prefilter=False) for key, value in maps.items()}
    # DINO is the normal hair correspondence producer.  A generated
    # reference can legitimately have zero same-material hair tokens; in that
    # case a separate, stricter producer may use verified face landmarks only
    # after checking the actual source/reference alpha and hair owners.
    hair_fallback = None
    contour_qualification_mask = None
    if not np.any(np.asarray(native_maps['valid'], bool) & (sl == 4)) and proof.get('face_observations'):
        from reference_hair_boundary_registration import prepare as prepare_hair_registration, validate as validate_hair_registration
        faces = copy.deepcopy(proof['face_observations'])
        # _points expects an image shape (height, width); the plan stores
        # canvas sizes as (width, height).
        faces['source_native_size'] = [native[1], native[0]]
        hair_fallback = prepare_hair_registration(
            a, ra, sl, ref_labels, faces, source_box=(box[0], box[1]),
            source_sha256=run['source']['sha256'], reference_sha256=run['reference']['sha256'],
            source_alpha_sha256=sha(plan['mask_roles']['foreground_alpha']['path']),
            reference_alpha_sha256=sha(matte['alpha_path']),
            source_labels_sha256=array_sha(sl),
            reference_labels_sha256=proof['input_binding']['reference_labels_sha256'],
            face_producer_identity_sha256=proof.get('face_producer_identity_sha256'))
        if hair_fallback['maps'] is not None:
            validate_hair_registration(hair_fallback['receipt'],
                binding=hair_fallback['receipt']['binding'])
            for key in ('ref_x', 'ref_y', 'confidence', 'valid'):
                native_maps[key][sl == 4] = hair_fallback['maps'][key][sl == 4]
            # The fallback receipt qualifies a hair-owner contour band from
            # alpha/owner evidence.  Reuse that exact band for the transfer
            # producer's existing >=0.7 contour gate; do not substitute the
            # whole-person silhouette for this owner-specific observation.
            hair = (a >= .5) & (sl == 4)
            depth = hair_fallback.pop('source_hair_depth').for_mask(hair)
            band_depth = int(hair_fallback['receipt']['registration']['hair_anchor_band_native_depth_pixels'])
            contour_qualification_mask = hair & (depth <= band_depth)
    protection = sample_asset(plan['mask_roles']['identity_protection'], 'protection', box, native)
    bg_assets = next(row for row in plan['layers'] if row['kind'] == 'background_design')['assets']
    v = sample_asset(bg_assets['validity'], 'values', box, native)
    # Fit the correspondence against the actual source alpha boundary.  The
    # semantic hair interior is often non-rigid across photographs; using it
    # as a global affine fit can inflate the error while the visible contour
    # remains well aligned.  The band is derived from the profile's maximum
    # reference-space observation depth and the measured canvas scale, so it
    # does not relax owner, contour, alpha, validity or identity protection
    # gates.
    from boundary_light_geometry_context import ExactMaskDistance
    source_inside = a >= .5
    source_inside_depth = ExactMaskDistance.compute(source_inside)
    source_depth = source_inside_depth.for_mask(source_inside)
    expected_scale = float(np.sqrt((rw * rh) / max(native[0] * native[1], 1)))
    max_native_depth = int(np.ceil(float(HAIR_PROFILE['maximum_depth_reference_pixels'])
                                  * max(1., 1. / max(expected_scale, 1e-6))))
    registration_anchor_mask = source_inside & (source_depth <= max_native_depth)
    result = prepare_boundary_light(p, a, reference, ra, sl, ref_labels, native_maps,
        validity=v, identity_protection=protection, reference_fp=fp, reference_fb_support=fb,
        registration_anchor_mask=registration_anchor_mask,
        contour_qualification_mask=contour_qualification_mask,
        source_inside_depth=source_inside_depth,
        source_owner_uncertainty_pixels=float(HAIR_PROFILE['source_owner_uncertainty_proxy_pixels'])*max(
            native[0]/proxy[0], native[1]/proxy[1]))
    components = result['receipt'].get('components')
    if isinstance(components, list) and components and isinstance(components[0], dict):
        registration = components[0].setdefault('registration', {})
        registration['anchor_band_native_depth_pixels'] = max_native_depth
        registration['anchor_band_reference_depth_pixels'] = float(HAIR_PROFILE['maximum_depth_reference_pixels'])
        registration['anchor_scale_estimate'] = expected_scale
    delta = result['delta_p']; changed = int(np.any(delta != 0, -1).sum())
    receipt = {**result['receipt'], 'schema': SCHEMA, 'recipe': RECIPE, 'changed_pixels': changed,
        'producer_code': producer_code(KIND),
        'native_output': True, 'model_tile_inferences_this_attempt': 0 if matte['cache_hit'] else matte['tiles']['alpha_tiles'],
        'reference_matte_cache_hit': matte['cache_hit'], 'image_generation_submissions': 0,
        'artistic_acceptance': False, 'clothing_enabled': False, 'native_box': list(box),
        'endpoint_clipped_pixels': endpoint_clipped_pixels}
    if hair_fallback is not None:
        receipt['hair_fallback_registration'] = hair_fallback['receipt']
        if isinstance(receipt.get('components'), list) and receipt['components']:
            receipt['components'][0].setdefault('registration', {})[
                'contour_qualification_policy'] = 'verified_fallback_hair_owner_band'
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
    layer = {'id': KIND, 'kind': KIND, 'label': '参考发梢受光', 'enabled': True, 'strength': 1.,
        'params': {'schema': SCHEMA, 'recipe': RECIPE, 'binding': binding},
        'assets': {'frozen_delta': asset(output/'frozen.json'), 'qualification_receipt': asset(output/'qualification.json')}}
    result_plan = replace_layer(plan, previous, layer)
    if validate_layer({**layer, 'enabled': True, 'strength': 1.}, result_plan, run)['bypass_reason']:
        raise ValueError('Prepared hair light does not match its current native endpoint')
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
        draw.text((8, 8), 'current native', fill='white'); draw.text((w+8, 8), 'observed hair boundary light', fill='white')
        sheet.save(output/'native-comparison.png')
    return {'changed': True, 'receipt': receipt, 'design_reconstruction_plan': result_plan}


def prepare_for_workbench(root, run, output, *, cancel=None, progress=None):
    """Isolate model preparation from the desktop runtime and saved parent."""
    from current_model_registry import load_registry
    from runtime_resources import code_path
    root, output = Path(root).resolve(), Path(output).resolve()
    if cancel and cancel(): raise InterruptedError('Hair-light preparation cancelled')
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.with_name(output.name+'-input.json'); write(snapshot, run)
    python = load_registry(root, registry_path=run['options'].get('model_registry'))['python']
    log = output.with_name(output.name+'-worker.log')
    if progress: progress('hair_boundary_light', '正在观察参考发梢并准备原尺寸受光层；不新增生图。')
    with log.open('wb') as stream:
        process = external_python_popen([python, '-B', '-X', 'utf8', str(code_path('boundary_light_worker.py')),
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
                    stop(); raise InterruptedError('Hair-light preparation cancelled; saved parent retained')
                time.sleep(.25)
        finally: stop()
    if process.returncode:
        raise ValueError('Hair-light preparation failed: '+log.read_text('utf8', errors='replace')[-2200:])
    if cancel and cancel(): raise InterruptedError('Hair-light preparation cancelled; result not applied')
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
    from native_cuda_filters import cuda_gaussian_scope
    from native_cuda_ndimage import cuda_ndimage_scope
    with cuda_gaussian_scope(), cuda_ndimage_scope():
        result = prepare(args.root, run, args.output)
    write(args.output/'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'design_reconstruction_plan'}, ensure_ascii=False))


if __name__ == '__main__': main()
