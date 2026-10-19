"""Fit reference colour parameters; final native rendering belongs downstream."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from runtime_resources import code_path
from asset_identity import sha256_file

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from display_transform import display_linear, source_transform_for_run
from float_appearance import srgb_to_linear
from local_photo_analysis import prepare_local_appearance
from local_face_protection import (prepare_face_protection_pair, source_feature_regions,
                                   prepare_feature_materials)
from raw_support import open_linear, open_rgb, linear_preview, RGB_DECODE_VERSION
from native_array_storage import save_proxy_arrays
from reference_colour_fit import (collect_anchors, collect_material_distributions,
    fit_material_distributions, fit, fit_tone_curves, fit_conditional_residual,
    fit_local_residual, region_weights, apply, REGIONS, to_lab,
    prepare_lab_observation, release_feature_colour)


def resize_plane(values, size, nearest=False):
    return cv2.resize(values, size, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def sheet(images, labels, maximum_width=1800):
    total = sum(i.width for i in images)
    scale = min(1., maximum_width/total)
    tiles = [i.resize((round(i.width*scale), round(i.height*scale)), Image.Resampling.LANCZOS) for i in images]
    out = Image.new('RGB', (sum(i.width for i in tiles), max(i.height for i in tiles)+30), '#171717')
    draw = ImageDraw.Draw(out)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    left = 0
    for i, label in zip(tiles, labels):
        out.paste(i, (left, 30)); draw.text((left+6, 6), label, font=font, fill='white'); left += i.width
    return out




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/inspection/reference-match-v2'))
    parser.add_argument('--object-masks', type=Path, help='Reviewed corresponding accessory masks')
    parser.add_argument('--authored-colour-edit', type=Path, help='Explicit reviewed source-coordinate colour stroke, independent of automatic fitting')
    args = parser.parse_args()
    root = args.root.resolve()
    run = json.loads(args.run.read_text(encoding='utf8'))
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    if any(row.get('kind') == 'source_reference_colour_match' for row in plan.get('layers', [])):
        raise ValueError('Retired colour matching layers cannot enter current preparation')
    source_display = source_transform_for_run(run)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    size = tuple(plan['canvas']['source_proxy_size']); native = tuple(plan['canvas']['source_oriented_size'])
    print('Preparing source/reference semantic observations', flush=True)
    analyses = [prepare_local_appearance(root, Path(run[role]['path']), source_hash=run[role]['sha256'],
        role='source' if role == 'source' else 'candidate',
        registry_path=run.get('options', {}).get('model_registry')) for role in ('source', 'reference')]
    label_values = []
    for analysis in analyses:
        with np.load(analysis['semantic_result']['labels_path'], allow_pickle=False) as resource:
            label_values.append(resource['labels'])
    source_labels = resize_plane(label_values[0], size, nearest=True)
    reference_image = open_rgb(Path(run['reference']['path']))
    ref = srgb_to_linear(np.asarray(reference_image, np.float32)/255)
    reference_material_labels = resize_plane(label_values[1], reference_image.size, nearest=True)
    matrix = np.diag([size[0]/native[0], size[1]/native[1], 1.]) @ np.asarray(plan['canvas']['candidate_mapping']['candidate_original_to_final_original'])
    ref_aligned = cv2.warpPerspective(ref, matrix, size, flags=cv2.INTER_LINEAR)
    reference_labels = cv2.warpPerspective(reference_material_labels, matrix, size, flags=cv2.INTER_NEAREST)
    object_binding = None
    if args.object_masks:
        objects = json.loads(args.object_masks.read_text(encoding='utf8'))
        if not objects.get('visual_reviewed') or objects.get('object') != 'bag' or objects.get('label') != 200:
            raise ValueError('Accessory masks require visual review and an explicit bag label')
        masks = {}
        for role in ('source', 'reference'):
            asset = objects['roles'][role]
            path = Path(asset['mask_path'])
            if asset['input_sha256'] != run[role]['sha256'] or sha256_file(path) != asset['mask_sha256']:
                raise ValueError('Accessory mask binding mismatch')
            masks[role] = np.asarray(Image.open(path).convert('L')) > 0
        source_labels[resize_plane(masks['source'].astype(np.uint8), size, nearest=True) > 0] = 200
        rm = resize_plane(masks['reference'].astype(np.uint8), reference_image.size, nearest=True)
        reference_material_labels[rm > 0] = 200
        rm = cv2.warpPerspective(rm, matrix, size, flags=cv2.INTER_NEAREST)
        reference_labels[rm > 0] = 200
        object_binding = {'receipt_path': str(args.object_masks.resolve()),
                          'receipt_sha256': sha256_file(args.object_masks),
                          'masks': {role: objects['roles'][role]['mask_sha256'] for role in masks}}
    validity = cv2.warpPerspective(np.ones(ref.shape[:2], np.float32), matrix, size)
    print('Loading fixed RAW appearance and alpha (no new background)', flush=True)
    from native_source_baseline import source_for_reference_prepare
    scene, source_precision, source_display = source_for_reference_prepare(
        run, root=root, apply_beauty=True, writable=False)
    source = resize_plane(scene, size)
    del scene
    # Keep one request-local Lab observation for the source and native
    # reference. The aligned reference is a separate canvas and therefore has
    # its own short-lived observation for anchor collection only.
    source_observation = prepare_lab_observation(source)
    reference_observation = prepare_lab_observation(ref)
    aligned_reference_observation = prepare_lab_observation(ref_aligned)
    with np.load(plan['source_baseline']['assets']['foreground_alpha']['path'], allow_pickle=False) as resource:
        alpha = resize_plane(resource['values'], size)
    with np.load(plan['mask_roles']['identity_protection']['path'], allow_pickle=False) as resource:
        protection = resize_plane(resource['protection'] if 'protection' in resource else resource['values'], size)
    source_face, face = prepare_face_protection_pair(root,
        Path(analyses[0]['analysis_directory'])/'source-srgb.png', analyses[0]['semantic_result'],
        Path(analyses[1]['analysis_directory'])/'source-srgb.png', analyses[1]['semantic_result'])
    with np.load(face['field_path'], allow_pickle=False) as resource:
        rp = resize_plane(resource['protection'], reference_image.size)
    semantic_hashes = {
        'source_labels': hashlib.sha256(source_labels.tobytes()).hexdigest(),
        'reference_labels': hashlib.sha256(reference_labels.tobytes()).hexdigest(),
        'reference_material_labels': hashlib.sha256(reference_material_labels.tobytes()).hexdigest(),
    }
    feature_domains = {
        'schema': 'feature_material_domains.v1',
        'source_regions': source_feature_regions(source_face, size),
        'reference_regions': source_feature_regions(face, reference_image.size),
        'source_face_identity': source_face['identity'],
        'reference_face_identity': face['identity'],
    }
    # Refine appearance materials once, before T fitting and weight creation.
    # Mislabelled skin must receive the same tone and colour as nearby skin;
    # shrinking a later lip field cannot undo a dark/cool T material seam.
    source_labels, protection, feature_domains['source_report'] = prepare_feature_materials(
        source_labels, protection, feature_domains['source_regions'])
    reference_material_labels, rp, feature_domains['reference_report'] = prepare_feature_materials(
        reference_material_labels, rp, feature_domains['reference_regions'])
    reference_labels = cv2.warpPerspective(reference_material_labels, matrix, size, flags=cv2.INTER_NEAREST)
    reference_protection = cv2.warpPerspective(rp, matrix, size)
    anchors, observations = collect_anchors(source, ref_aligned, source_labels, reference_labels, alpha,
                                           protection, reference_protection, validity,
                                           source_observation=source_observation,
                                           reference_observation=aligned_reference_observation)
    del aligned_reference_observation
    distributions = collect_material_distributions(source, ref, source_labels, reference_material_labels,
                                                     alpha, protection, rp,
                                                     source_observation=source_observation,
                                                     reference_observation=reference_observation)
    observations['material_distributions'] = distributions
    parameters = release_feature_colour(fit_material_distributions(fit(anchors), distributions))
    parameters = fit_tone_curves(anchors, parameters)
    parameters = fit_conditional_residual(anchors, parameters)
    parameters = fit_local_residual(anchors, parameters, size, source_labels=source_labels,
        source_lab=to_lab(source), source_alpha=alpha)
    from reference_colour_fit import preserve_source_hair_parameters
    parameters = preserve_source_hair_parameters(parameters, run)
    from reference_material_palette import fit_palette
    parameters['automatic_material_palette'] = fit_palette(source, ref, source_labels,
        reference_material_labels, alpha, parameters, source_protection=protection,
        reference_protection=rp, source_observation=source_observation,
        reference_observation=reference_observation)
    authored_binding = None
    if args.authored_colour_edit:
        edit = json.loads(args.authored_colour_edit.read_text(encoding='utf8'))
        if (edit.get('schema') != 'reference_authored_chroma.v1' or edit.get('working_domain') != 'display_linear_sRGB'
            or any(edit.get(f'{role}_sha256') != run[role]['sha256'] for role in ('source', 'reference'))):
            raise ValueError('Authored colour edit belongs to a different original/reference pair or domain')
        parameters['authored_chroma_adjustments'] = edit['adjustments']
        authored_binding = {'path': str(args.authored_colour_edit.resolve()),
            'sha256': sha256_file(args.authored_colour_edit),
            'stroke_algorithm_sha256': sha256_file(code_path('local_colour_brush.py')),
            'automatic_fit': False}
    parameters = preserve_source_hair_parameters(parameters, run)
    parameters['observations'] = observations
    parameters['feature_material_domains'] = feature_domains
    parameters['protection_policy']['mouth_lightness'] = (
        'ordered_mouth_receives_face_illumination; pigment_and_reference_fitting_domains_unchanged')
    parameters['support_policy'] = 'reference_support.v1'
    parameters['binding'] = {'source': run['source']['sha256'], 'reference': run['reference']['sha256'],
        'decode_version': RGB_DECODE_VERSION, 'source_display': source_display,
        'native_source_retouch': source_precision.get('native_source_retouch'),
        'portrait_beauty': source_precision.get('portrait_beauty'),
        'alpha': plan['source_baseline']['assets']['foreground_alpha']['sha256'],
        **semantic_hashes,
        'source_protection': plan['mask_roles']['identity_protection']['sha256'],
        'reference_protection': hashlib.sha256(reference_protection.tobytes()).hexdigest(),
        'reference_material_protection': hashlib.sha256(rp.tobytes()).hexdigest(),
        'algorithm_sha256': sha256_file(code_path('reference_colour_fit.py')),
        'evaluation_sha256': sha256_file(code_path('evaluate_reference_colour.py')),
        'mapping': matrix.tolist()}
    parameters['binding']['object_masks'] = object_binding
    parameters['binding']['authored_colour_edit'] = authored_binding
    parameters['region_order'] = list(REGIONS)
    parameters['anchors'] = anchors
    weights = region_weights(source_labels, protection, parameters, source_values=source,
                             source_observation=source_observation)
    chroma_weights = region_weights(source_labels, protection, parameters, chroma=True,
                                    source_values=source, source_observation=source_observation)
    save_proxy_arrays(output/'source-weights.npz', {'values': weights, 'chroma_values': chroma_weights})
    parameters['binding']['source_weights_sha256'] = sha256_file(output/'source-weights.npz')
    (output/'parameters.json').write_text(json.dumps(parameters, ensure_ascii=False, separators=(',', ':'))+'\n', encoding='utf8')
    print('Accepted anchors', {n: sum(a['region'] == n for a in anchors) for n in REGIONS}, flush=True)
    print('T evidence modes', {n: op['evidence']['mode'] for n, op in parameters['operators'].items()}, flush=True)
    anchor_image = linear_preview(source)
    draw = ImageDraw.Draw(anchor_image)
    colours = {'face': 'lime', 'left_arm': 'orange', 'right_arm': 'cyan'}
    for a in anchors:
        x, y = a['xy']; draw.rectangle((x-10, y-10, x+10, y+10), outline=colours.get(a['region'], 'white'), width=1)
    anchor_image.save(output/'anchors.jpg', quality=95)
    boxes = {}
    for label, ids in [('face', [3, 4]), ('arms', [6, 7, 11, 15, 16, 20]), ('bag', [200])]:
        selected = np.isin(source_labels, ids)
        yy, xx = np.nonzero(selected)
        if not len(xx):
            continue
        x0, x1, y0, y1 = max(0, xx.min()-20), min(size[0], xx.max()+21), max(0, yy.min()-15), min(size[1], yy.max()+16)
        boxes[label] = tuple(round(v*s) for v, s in zip((x0, y0, x1, y1), (native[0]/size[0], native[1]/size[1])*2))
    # Native crop coordinates are metadata for the one downstream renderer.
    # Proxy evaluation observes the fit only: nonlinear colour transforms do
    # not commute with resizing, so it cannot claim native output validation.
    source_preview = linear_preview(source)
    matched_proxy = apply(source, weights, parameters, canvas_size=size, chroma_weights=chroma_weights,
                          values_observation=source_observation)
    newly_clipped = int(np.count_nonzero((matched_proxy >= 1-1e-7) & (source < 1-1e-7)))
    # The diagnostic loop may contain thousands of accepted anchors. Convert
    # the fitted proxy to Lab once and take each anchor's masked median from
    # that read-only plane; converting the whole proxy once per anchor was
    # pure repeated work and did not change the report values.
    matched_proxy_lab = to_lab(matched_proxy)
    matched_preview = linear_preview(matched_proxy)
    matched_preview.save(output/'matched-preview.png')
    sheet([source_preview, linear_preview(ref_aligned), matched_preview], ['Source fit proxy', 'Reference', 'Parameter diagnostic proxy'], 2000).save(output/'whole-comparison.jpg', quality=96)
    values = []
    for a in anchors:
        left, top, right, bottom = a['box']; box = (slice(top, bottom), slice(left, right))
        valid = np.unpackbits(np.frombuffer(bytes.fromhex(a['sample_mask_bits']), dtype=np.uint8))[:(right-left)*(bottom-top)].reshape(bottom-top, right-left).astype(bool)
        new = np.median(matched_proxy_lab[box][valid], axis=0)
        values.append({'region': a['region'], 'xy': a['xy'], 'source': a['source'], 'reference': a['reference'], 'matched': new.tolist()})
    zero = apply(source, weights, parameters, strength=0, canvas_size=size,
                 chroma_weights=chroma_weights, values_observation=source_observation)
    report = {'schema': 'reference_colour_evaluation.v3', 'run_id': run['run_id'], 'scope': 'proxy_parameter_diagnostics',
        'working_domain': 'display_linear_sRGB', 'native_crop_boxes': boxes, 'accepted_anchor_count': len(anchors),
        'proxy_size': list(size), 'native_render_executed': False,
        'zero_strength_exact': bool(np.array_equal(zero, source)), 'finite_and_bounded': bool(np.isfinite(matched_proxy).all() and matched_proxy.min() >= 0 and matched_proxy.max() <= 1),
        'newly_clipped_channels': newly_clipped,
        'authored_colour_edit': authored_binding,
        'protection_policy': parameters['protection_policy'],
        'material_statistics': {n: op['evidence'] for n, op in parameters['operators'].items()},
        'reference_geometry_used_for_output': False, 'reference_texture_used_for_output': False,
        'professional_acceptance': False, 'anchor_observations': values}
    (output/'validation.json').write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':'))+'\n', encoding='utf8')
    print(json.dumps({'output': str(output), 'operators': parameters['operators'], 'matrix_fit': parameters['matrix_fit']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
