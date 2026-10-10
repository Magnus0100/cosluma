"""Render a controlled native appearance-domain composite from fitted colours.

Reuses the selected background and alpha. Only the unknown source edge band
is re-estimated in appearance RGB; this does not alter the existing project.
Run with the registered model-worker Python (PyMatting and OpenCV).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from runtime_resources import code_path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from display_transform import display_linear
from raw_support import open_linear, linear_preview, save_render_master
from source_foreground import estimate_source_background
from reference_colour_fit import apply, REGIONS


def field_owners(receipt, shape):
    """New automatic fields own their proxy material/protection domains."""
    if not receipt.get('automatic_policy'):
        return None
    with np.load(receipt['asset']['path'], allow_pickle=False) as resource:
        owners = resource['owners']
    if owners.shape != tuple(shape) or not np.issubdtype(owners.dtype, np.integer):
        raise ValueError('Automatic field material domains do not match its grid')
    return owners


def sample_maps(values, box, proxy_size, native_size, *, owners=None):
    left, top, right, bottom = box
    xs = ((np.arange(left, right)+.5)*proxy_size[0]/native_size[0]-.5).astype(np.float32)
    ys = ((np.arange(top, bottom)+.5)*proxy_size[1]/native_size[1]-.5).astype(np.float32)
    xx, yy = np.meshgrid(xs, ys)
    if owners is not None:
        if (owners.shape != values.shape[:2] or not np.issubdtype(owners.dtype, np.integer)
                or tuple(proxy_size) != (values.shape[1], values.shape[0])):
            raise ValueError('Material-aware field sampling requires matching integer owners')
        xx = np.clip(xx, 0, values.shape[1]-1); yy = np.clip(yy, 0, values.shape[0]-1)
        x0 = np.floor(xx).astype(np.int32); y0 = np.floor(yy).astype(np.int32)
        x1 = np.minimum(x0+1, values.shape[1]-1); y1 = np.minimum(y0+1, values.shape[0]-1)
        domain = owners[np.floor(yy+.5).astype(np.int32), np.floor(xx+.5).astype(np.int32)]
        dx,dy = xx-x0, yy-y0
        result = np.zeros((*xx.shape,)+values.shape[2:], np.float32)
        total = np.zeros_like(xx)
        channels = (None,)*(values.ndim-2)
        for x,y,w in ((x0,y0,(1-dx)*(1-dy)),(x1,y0,dx*(1-dy)),
                      (x0,y1,(1-dx)*dy),(x1,y1,dx*dy)):
            dose = (w*(owners[y,x]==domain)).astype(np.float32)
            result += values[y,x]*dose[(...,)+channels]
            total += dose
        return result/np.maximum(total,1e-12)[(...,)+channels]
    return cv2.remap(values, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def donor_fields(source, alpha, weights, parameters, reference_blend=None, scene_balance=None, matched_baseline=None, chroma_weights=None):
    """A bounded affine extension avoids division by low edge alpha."""
    if (source.ndim != 3 or source.shape[-1] != 3 or alpha.shape != source.shape[:2]
            or weights.shape != (*source.shape[:2], len(REGIONS))
            or not all(np.isfinite(x).all() for x in (source, alpha, weights))
            or alpha.min() < 0 or alpha.max() > 1 or weights.min() < 0
            or weights.max() > 1+1e-6 or np.any(weights.sum(-1) > 1+1e-6)):
        raise ValueError('Invalid source/alpha/material donor inputs')
    material_weights = weights
    if parameters.get('automatic_material_palette') and chroma_weights is not None:
        material_weights = np.maximum(weights, chroma_weights)
        material_weights /= np.maximum(material_weights.sum(-1, keepdims=True), 1)
    eligible = (alpha >= .995) & (material_weights.sum(-1) > .8) & (source.max(-1) < .95)
    if matched_baseline is not None:
        if matched_baseline.shape != source.shape or not np.isfinite(matched_baseline).all():
            raise ValueError('Invalid native-matched donor baseline')
        matched = matched_baseline
    else:
        matched = apply(source, weights, parameters, canvas_size=(source.shape[1], source.shape[0]), chroma_weights=chroma_weights)
        if reference_blend is not None:
            from reference_colour_blend import apply_colour_layer
            matched = apply_colour_layer(matched, reference_blend)
    if scene_balance is not None:
        from reference_scene_balance import apply_scene_balance
        matched = apply_scene_balance(matched, scene_balance)
    if not np.isfinite(matched).all() or matched.min() < 0 or matched.max() > 1:
        raise ValueError('Donor target must be finite display SDR, including no-donor fallback')
    owner = material_weights.argmax(-1)
    ownership = material_weights.max(-1)
    extra_ev = float(np.max(np.abs(scene_balance[..., 0]))) if scene_balance is not None else 0.
    limit = 2**(parameters['settings']['maximum_exposure_ev']+extra_ev)
    if not np.isfinite(limit) or limit < 1:
        raise ValueError('Invalid donor exposure bound')
    fields = np.zeros((*source.shape[:2], 5), np.float32)
    fields[..., 0] = 1
    if not np.any(eligible):
        # A lack of donors is a valid degraded result; input effects were still
        # evaluated above so corrupt fields cannot be concealed by this path.
        return fields
    luma = np.array([.2126, .7152, .0722], np.float32)
    def observe(mask, select, maximum_distance, confidence):
        if not np.any(mask):
            return
        distance, positions = distance_transform_edt(~mask, return_indices=True)
        donor_source = source[tuple(positions)][select]
        donor_target = matched[tuple(positions)][select]
        gain = np.clip((donor_target @ luma)/np.maximum(donor_source @ luma, .003), 1/limit, limit)
        bias = donor_target-gain[..., None]*donor_source
        bias *= np.minimum(1, .08/np.maximum(np.linalg.norm(bias, axis=-1), 1e-8))[:, None]
        support = confidence*np.clip(1-distance[select]/maximum_distance, 0, 1)
        fields[select] = np.concatenate([gain[:, None], bias, support[:, None]], axis=-1)
    for index, name in enumerate(REGIONS):
        if (name not in parameters['operators'] and parameters.get('automatic_material_palette', {})
                .get('regions', {}).get(name, {}).get('status') != 'supported'):
            continue
        observe(eligible & (owner == index), (owner == index) & (ownership >= .25), 24, 1.)
    # Untyped edges retain their original premultiplied colour. Every extension
    # requires its own material owner; proximity alone cannot borrow skin/cloth.
    return fields


def bounded_edge_extension(fp, alpha, correction, report=None):
    """Reduce the proposed affine edge dose to the premultiplied SDR interval.

    One scalar dose preserves the correction's RGB direction. Existing source
    matting excursions are exposed separately and cannot be increased. This
    bounds the automatic operator before composition, not an output RGB clip.
    """
    if (fp.shape != correction.shape or fp.shape[:2] != alpha.shape or fp.shape[-1] != 3
            or not all(np.isfinite(x).all() for x in (fp, alpha, correction))
            or alpha.min() < 0 or alpha.max() > 1 or fp.min() < 0):
        raise ValueError('Invalid premultiplied edge colour or coverage')
    upper = np.maximum(alpha[..., None], fp)
    dose = np.ones_like(fp)
    positive, negative = correction > 0, correction < 0
    np.divide(upper-fp, correction, out=dose, where=positive)
    np.divide(-fp, correction, out=dose, where=negative)
    dose = np.clip(dose.min(-1), 0, 1)
    result = fp+dose[..., None]*correction
    if report is not None:
        proposed = fp+correction
        bad = np.any((proposed > upper+1e-6) | (proposed < -1e-6), axis=-1)
        source_excursion = np.maximum(fp-alpha[..., None], 0)
        report['policy'] = 'bound_new_affine_edge_delta_dose; no_final_RGB_clip; source_excursions_not_increased'
        report['requested_outside_sdr_pixels'] = report.get('requested_outside_sdr_pixels', 0)+int(bad.sum())
        report['source_outside_premultiplied_sdr_pixels'] = report.get('source_outside_premultiplied_sdr_pixels', 0)+int(
            np.any(source_excursion > 1e-6, axis=-1).sum())
        report['maximum_requested_excursion'] = max(report.get('maximum_requested_excursion', 0),
            float(np.maximum(proposed-upper, 0).max()), float(np.maximum(-proposed, 0).max()))
        report['maximum_source_excursion'] = max(report.get('maximum_source_excursion', 0), float(source_excursion.max()))
        report['minimum_applied_dose'] = min(report.get('minimum_applied_dose', 1), float(dose.min()))
        report['maximum_operator_increased_excursion'] = max(report.get('maximum_operator_increased_excursion', 0),
            float(np.maximum(result-upper, 0).max()))
    return result


def match_foreground(fp, alpha, weights, donors, edge_support, parameters, *, origin, canvas_size, reference_blend=None, scene_balance=None, chroma_weights=None, donor_gamut_report=None):
    # At high coverage unpremultiplication is stable; below .95 it is never used.
    safe = alpha >= .95
    source = np.divide(fp, alpha[..., None], out=np.zeros_like(fp), where=safe[..., None])
    matched = apply(source, weights, parameters, origin=origin, canvas_size=canvas_size, chroma_weights=chroma_weights)
    if reference_blend is not None:
        from reference_colour_blend import apply_colour_layer
        matched = apply_colour_layer(matched, reference_blend)
    if scene_balance is not None:
        from reference_scene_balance import apply_scene_balance
        matched = apply_scene_balance(matched, scene_balance)
    direct = matched*alpha[..., None]
    gain, bias, donor_support = donors[..., 0], donors[..., 1:4], donors[..., 4]
    confidence = donor_support*(.35+.65*np.clip(edge_support, 0, 1))
    correction = confidence[..., None]*((gain-1)[..., None]*fp+alpha[..., None]*bias)
    extended = bounded_edge_extension(fp, alpha, correction, donor_gamut_report)
    transition = np.clip((alpha-.95)/.05, 0, 1)
    transition = transition*transition*(3-2*transition)
    result = extended+(direct-extended)*transition[..., None]
    result = np.maximum(result, 0)
    result[alpha == 0] = 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--match-dir', type=Path, default=Path('outputs/inspection/reference-match-v2'))
    parser.add_argument('--reference-blend', type=Path, help='Bound selected low-frequency reference colour layer')
    parser.add_argument('--scene-balance', type=Path, help='Bound residual lightness and environment-colour balance layer')
    args = parser.parse_args()
    started = time.perf_counter()
    run = json.loads(args.run.read_text(encoding='utf8'))
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    directory = args.match_dir.resolve()
    parameters = json.loads((directory/'parameters.json').read_text(encoding='utf8'))
    if parameters['binding']['source'] != run['source']['sha256'] or parameters['binding']['reference'] != run['reference']['sha256']:
        raise ValueError('Match belongs to a different original/reference pair')
    if parameters['binding']['algorithm_sha256'] != hashlib.sha256(code_path('reference_colour_fit.py').read_bytes()).hexdigest():
        raise ValueError('Re-evaluate matching after an algorithm change')
    if parameters['binding']['alpha'] != plan['source_baseline']['assets']['foreground_alpha']['sha256']:
        raise ValueError('Match belongs to a different baseline alpha')
    if parameters.get('region_order') != list(REGIONS):
        raise ValueError('Re-evaluate matching after a region contract change')
    authored = parameters['binding'].get('authored_colour_edit')
    if authored and hashlib.sha256(Path(authored['path']).read_bytes()).hexdigest() != authored['sha256']:
        raise ValueError('Authored colour stroke changed after evaluation')
    if authored and hashlib.sha256(code_path('local_colour_brush.py').read_bytes()).hexdigest() != authored['stroke_algorithm_sha256']:
        raise ValueError('Authored stroke algorithm changed after evaluation')
    objects = parameters['binding'].get('object_masks')
    if objects:
        object_path = Path(objects['receipt_path'])
        if hashlib.sha256(object_path.read_bytes()).hexdigest() != objects['receipt_sha256']:
            raise ValueError('Accessory mask review changed after fitting')
        receipt = json.loads(object_path.read_text(encoding='utf8'))
        for role in ('source', 'reference'):
            asset = receipt['roles'][role]
            if hashlib.sha256(Path(asset['mask_path']).read_bytes()).hexdigest() != objects['masks'][role]:
                raise ValueError('Accessory mask changed after fitting')
    size, native = tuple(plan['canvas']['source_proxy_size']), tuple(plan['canvas']['source_oriented_size'])
    blend_field, blend_receipt = None, None
    if args.reference_blend:
        from reference_colour_blend import validate_layer
        blend_field, blend_receipt = validate_layer(args.reference_blend, parameters, native, size)
    blend_owners = field_owners(blend_receipt, blend_field.shape[:2]) if blend_field is not None else None
    scene_field, scene_receipt, donor_baseline = None, None, None
    if args.scene_balance:
        if not args.reference_blend:
            raise ValueError('Scene balance must follow its bound reference colour layer')
        from reference_scene_balance import validate_scene_layer
        scene_field, scene_receipt = validate_scene_layer(args.scene_balance, parameters,
            args.reference_blend, plan, native, size)
        with np.load(scene_receipt['asset']['path'], allow_pickle=False) as resource:
            donor_baseline = resource['baseline_rgb']
        if donor_baseline.shape != (*scene_field.shape[:2], 3) or not np.isfinite(donor_baseline).all():
            raise ValueError('Scene balance requires its native-matched proxy donor baseline')
    scene_owners = field_owners(scene_receipt, scene_field.shape[:2]) if scene_field is not None else None
    weights_hash = hashlib.sha256((directory/'source-weights.npz').read_bytes()).hexdigest()
    if parameters['binding'].get('source_weights_sha256') != weights_hash:
        raise ValueError('Re-evaluate matching after source weights change')
    with np.load(directory/'source-weights.npz', allow_pickle=False) as resource:
        weights = resource['values']
        chroma_weights = resource['chroma_values'] if 'chroma_values' in resource else None
    with np.load(plan['source_baseline']['assets']['foreground_alpha']['path'], allow_pickle=False) as resource:
        alpha = resource['values']
    print('Decoding native original into fixed appearance RGB', flush=True)
    from native_source_baseline import source_for_reference_prepare
    source, _, _ = source_for_reference_prepare(run, binding=parameters['binding'])
    proxy = cv2.resize(source, size, interpolation=cv2.INTER_LINEAR)
    donors = donor_fields(proxy, cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR), weights, parameters,
        blend_field, scene_field, donor_baseline, chroma_weights)
    # Alpha is reused. No ViTMatte, inpainting, reference generation or SR rerun.
    trimap = np.full(alpha.shape, 128, np.uint8)
    trimap[alpha == 0] = 0; trimap[alpha == 1] = 255
    print('Re-estimating only unknown appearance-domain edges', int(np.count_nonzero(trimap == 128)), flush=True)
    old_bg, foreground, edge_support, tiles, _, _ = estimate_source_background(source, alpha, trimap)
    maximum_reconstruction_error = 0.
    opaque_unchanged = True
    for top in range(0, len(source), 128):
        end = min(len(source), top+128)
        restored = foreground[top:end]+(1-alpha[top:end, ..., None])*old_bg[top:end]
        maximum_reconstruction_error = max(maximum_reconstruction_error, float(np.max(np.abs(restored-source[top:end]))))
        known = alpha[top:end] == 1
        opaque_unchanged &= bool(np.array_equal(foreground[top:end][known], source[top:end][known]))
    del source, trimap
    bg_layer = next(x for x in plan['layers'] if x['kind'] == 'background_design')
    with np.load(bg_layer['assets']['background']['path'], allow_pickle=False) as resource:
        background = resource['values']
    with np.load(bg_layer['assets']['validity']['path'], allow_pickle=False) as resource:
        validity_proxy = resource['values']
    associated = bg_layer['assets']['background'].get('validity_associated', False) or bg_layer['assets']['background'].get('representation') == 'sampling_validity_associated_RGB'
    # The selected plate is inverse-sRGB reference appearance, not RAW scene
    # radiance. Do not subject it to a second photographic display transform.
    print('Matching native Fp, selected reference colour fusion, and compositing once', flush=True)
    clipping, opaque_pixels, low_alpha_pixels = 0, 0, 0
    for top in range(0, len(foreground), 128):
        end = min(len(foreground), top+128); box = (0, top, native[0], end)
        w = sample_maps(weights, box, size, native)
        cw = sample_maps(chroma_weights, box, size, native) if chroma_weights is not None else None
        d = sample_maps(donors, box, size, native)
        colour = sample_maps(blend_field, box, size, native, owners=blend_owners) if blend_field is not None else None
        scene = sample_maps(scene_field, box, size, native, owners=scene_owners) if scene_field is not None else None
        foreground[top:end] = match_foreground(foreground[top:end], alpha[top:end], w, d, edge_support[top:end], parameters,
            origin=(0, top), canvas_size=native, reference_blend=colour, scene_balance=scene, chroma_weights=cw)
        if (background.shape[1], background.shape[0]) == native:
            target = background[top:end]
        else:
            target = sample_maps(background, box, (background.shape[1], background.shape[0]), native)
        validity = sample_maps(validity_proxy, box, (validity_proxy.shape[1], validity_proxy.shape[0]), native)
        bg = target+(1-validity[..., None])*old_bg[top:end] if associated else validity[..., None]*target+(1-validity[..., None])*old_bg[top:end]
        foreground[top:end] += (1-alpha[top:end, ..., None])*bg
        clipping += int(np.count_nonzero((foreground[top:end] < 0) | (foreground[top:end] > 1)))
        opaque_pixels += int(np.count_nonzero(alpha[top:end] == 1))
        low_alpha_pixels += int(np.count_nonzero((alpha[top:end] > 0) & (alpha[top:end] < .95)))
    del old_bg, background
    np.clip(foreground, 0, 1, out=foreground)
    master = directory/'appearance-composite.tif'
    save_render_master(foreground, master)
    linear_preview(cv2.resize(foreground, size, interpolation=cv2.INTER_AREA)).save(directory/'pure-composite.jpg', quality=97)
    previous = json.loads((directory/'validation.json').read_text(encoding='utf8'))
    for name, box in previous['native_crop_boxes'].items():
        left, top, right, bottom = box
        linear_preview(foreground[top:bottom, left:right]).save(directory/f'composite-native-{name}.png')
    report = {'schema': 'reference_colour_composite.appearance.v2', 'source_run_id': run['run_id'],
        'working_domain': 'display_linear_sRGB', 'output': str(master), 'size': list(native),
        'source_reference_binding': parameters['binding'], 'source_weights_sha256': weights_hash,
        'protection_policy': parameters['protection_policy'],
        'parameters_sha256': hashlib.sha256((directory/'parameters.json').read_bytes()).hexdigest(),
        'validation_sha256': hashlib.sha256((directory/'validation.json').read_bytes()).hexdigest(),
        'renderer_sha256': hashlib.sha256(code_path('render_reference_colour_composite.py').read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(master.read_bytes()).hexdigest(),
        'alpha_reused': True, 'appearance_edge_tiles': tiles, 'original_reconstruction_max_error': maximum_reconstruction_error,
        'edge_donors': 'same_semantic_region; untyped_very_near_edges_support_at_most_0.15',
        'opaque_original_unchanged_before_matching': opaque_unchanged,
        'opaque_source_pixels': opaque_pixels, 'low_alpha_division_avoided_pixels': low_alpha_pixels,
        'final_out_of_gamut_channels_before_clipping': clipping,
        'composition_count': 1, 'photographic_display_transform_after_reference': False,
        'extra_contact_rim_fog_bloom': False, 'existing_project_brushes_applied': False,
        'authored_colour_edit': authored, 'existing_reference_shadows_reflections_resolved': False,
        'reference_spatial_colour_used': blend_receipt is not None,
        'reference_high_frequency_texture_used': False,
        'reference_spatial_lightness_used': scene_receipt is not None,
        'scene_balance': {'receipt_path': str(args.scene_balance.resolve()),
            'receipt_sha256': hashlib.sha256(args.scene_balance.read_bytes()).hexdigest(),
            'asset_sha256': scene_receipt['asset']['sha256'],
            'donor_baseline': 'native_T_plus_colour_blend_reduced_to_proxy; opaque_donors_only',
            'joint_donor_exposure_limit_ev': parameters['settings']['maximum_exposure_ev']+float(np.max(np.abs(scene_field[..., 0]))),
            'background_curve': 'identity', 'physical_lighting_reconstructed': False,
            'regions': scene_receipt['regions']} if scene_receipt else None,
        'reference_colour_layer': {'receipt_path': str(args.reference_blend.resolve()),
            'receipt_sha256': hashlib.sha256(args.reference_blend.read_bytes()).hexdigest(),
            'asset_sha256': blend_receipt['asset']['sha256'], 'regions': blend_receipt['regions']} if blend_receipt else None,
        'geometry_or_texture_from_reference_person': False, 'project_pointer_modified': False,
        'visual_acceptance': False, 'elapsed_seconds': round(time.perf_counter()-started, 2)}
    (directory/'composite-receipt.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
