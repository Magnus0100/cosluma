"""Render one controlled, native-pixel reference colour comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from runtime_resources import code_path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from display_transform import display_linear, source_transform_for_run
from float_appearance import srgb_to_linear
from local_photo_analysis import prepare_local_appearance
from local_face_protection import prepare_face_protection
from raw_support import open_linear, open_rgb, linear_preview, RGB_DECODE_VERSION
from reference_colour_fit import collect_anchors, collect_material_distributions, fit_material_distributions, fit, fit_tone_curves, fit_conditional_residual, fit_local_residual, region_weights, apply, REGIONS, to_lab, release_feature_colour


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


def optional_review_boxes(validation, native_size):
    """Select available, nonempty source crops; evidence is not a model input."""
    width, height = native_size
    result = {}
    for name, box in (validation.get('native_crop_boxes') or {}).items():
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        values = np.asarray(box, np.float64)
        if not np.isfinite(values).all():
            continue
        x0, y0, x1, y1 = (int(value) for value in values)
        clipped = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            result[name] = clipped
    return result


def comparison_crops(plan, run, boxes, native, source_crops):
    """Compare current preparation with its captured native source appearance."""
    if any(row.get('kind') == 'source_reference_colour_match' for row in plan.get('layers', [])):
        raise ValueError('Retired colour matching layers cannot enter current preparation')
    return dict(source_crops), {'kind': 'source_display_appearance',
        'label': 'Original appearance baseline',
        'reason': 'current_source_before_matching', 'new_matching_applied': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/inspection/reference-match-v2'))
    parser.add_argument('--object-masks', type=Path, help='Reviewed corresponding accessory masks')
    parser.add_argument('--baseline-parameters', type=Path, help='Previous parameters for identical-anchor full-application comparison')
    parser.add_argument('--authored-colour-edit', type=Path, help='Explicit reviewed source-coordinate colour stroke, independent of automatic fitting')
    args = parser.parse_args()
    root = args.root.resolve()
    run = json.loads(args.run.read_text(encoding='utf8'))
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
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
            if asset['input_sha256'] != run[role]['sha256'] or hashlib.sha256(path.read_bytes()).hexdigest() != asset['mask_sha256']:
                raise ValueError('Accessory mask binding mismatch')
            masks[role] = np.asarray(Image.open(path).convert('L')) > 0
        source_labels[resize_plane(masks['source'].astype(np.uint8), size, nearest=True) > 0] = 200
        rm = resize_plane(masks['reference'].astype(np.uint8), reference_image.size, nearest=True)
        reference_material_labels[rm > 0] = 200
        rm = cv2.warpPerspective(rm, matrix, size, flags=cv2.INTER_NEAREST)
        reference_labels[rm > 0] = 200
        object_binding = {'receipt_path': str(args.object_masks.resolve()),
                          'receipt_sha256': hashlib.sha256(args.object_masks.read_bytes()).hexdigest(),
                          'masks': {role: objects['roles'][role]['mask_sha256'] for role in masks}}
    validity = cv2.warpPerspective(np.ones(ref.shape[:2], np.float32), matrix, size)
    print('Loading fixed RAW appearance and alpha (no new background)', flush=True)
    from native_source_baseline import source_for_reference_prepare
    scene, source_precision, source_display = source_for_reference_prepare(run, root=root)
    source = resize_plane(scene, size)
    with np.load(plan['source_baseline']['assets']['foreground_alpha']['path'], allow_pickle=False) as resource:
        alpha = resize_plane(resource['values'], size)
    with np.load(plan['mask_roles']['identity_protection']['path'], allow_pickle=False) as resource:
        protection = resize_plane(resource['protection'] if 'protection' in resource else resource['values'], size)
    face = prepare_face_protection(root, Path(analyses[1]['analysis_directory'])/'source-srgb.png', analyses[1]['semantic_result'])
    with np.load(face['field_path'], allow_pickle=False) as resource:
        rp = resize_plane(resource['protection'], reference_image.size)
    reference_protection = cv2.warpPerspective(rp, matrix, size)
    anchors, observations = collect_anchors(source, ref_aligned, source_labels, reference_labels, alpha,
                                           protection, reference_protection, validity)
    distributions = collect_material_distributions(source, ref, source_labels, reference_material_labels,
                                                    alpha, protection, rp)
    observations['material_distributions'] = distributions
    parameters = release_feature_colour(fit_material_distributions(fit(anchors), distributions))
    baseline_parameters = parameters
    parameters = fit_tone_curves(anchors, parameters)
    parameters = fit_conditional_residual(anchors, parameters)
    parameters = fit_local_residual(anchors, parameters, size)
    from reference_material_palette import fit_palette
    parameters['automatic_material_palette'] = fit_palette(source, ref, source_labels,
        reference_material_labels, alpha, parameters, source_protection=protection,
        reference_protection=rp)
    authored_binding = None
    if args.authored_colour_edit:
        edit = json.loads(args.authored_colour_edit.read_text(encoding='utf8'))
        if (edit.get('schema') != 'reference_authored_chroma.v1' or edit.get('working_domain') != 'display_linear_sRGB'
            or any(edit.get(f'{role}_sha256') != run[role]['sha256'] for role in ('source', 'reference'))):
            raise ValueError('Authored colour edit belongs to a different original/reference pair or domain')
        parameters['authored_chroma_adjustments'] = edit['adjustments']
        authored_binding = {'path': str(args.authored_colour_edit.resolve()),
            'sha256': hashlib.sha256(args.authored_colour_edit.read_bytes()).hexdigest(),
            'stroke_algorithm_sha256': hashlib.sha256(code_path('local_colour_brush.py').read_bytes()).hexdigest(),
            'automatic_fit': False}
    parameters['observations'] = observations
    parameters['support_policy'] = 'reference_support.v1'
    parameters['binding'] = {'source': run['source']['sha256'], 'reference': run['reference']['sha256'],
        'decode_version': RGB_DECODE_VERSION, 'source_display': source_display,
        'native_source_retouch': source_precision.get('native_source_retouch'),
        'alpha': plan['source_baseline']['assets']['foreground_alpha']['sha256'],
        'source_labels': hashlib.sha256(source_labels.tobytes()).hexdigest(),
        'reference_labels': hashlib.sha256(reference_labels.tobytes()).hexdigest(),
        'reference_material_labels': hashlib.sha256(reference_material_labels.tobytes()).hexdigest(),
        'source_protection': plan['mask_roles']['identity_protection']['sha256'],
        'reference_protection': hashlib.sha256(reference_protection.tobytes()).hexdigest(),
        'reference_material_protection': hashlib.sha256(rp.tobytes()).hexdigest(),
        'algorithm_sha256': hashlib.sha256(code_path('reference_colour_fit.py').read_bytes()).hexdigest(),
        'evaluation_sha256': hashlib.sha256(code_path('evaluate_reference_colour.py').read_bytes()).hexdigest(),
        'mapping': matrix.tolist()}
    parameters['binding']['object_masks'] = object_binding
    parameters['binding']['authored_colour_edit'] = authored_binding
    parameters['region_order'] = list(REGIONS)
    parameters['anchors'] = anchors
    weights = region_weights(source_labels, protection, parameters, source_values=source)
    chroma_weights = region_weights(source_labels, protection, parameters, chroma=True, source_values=source)
    baseline_weights = region_weights(source_labels, protection, baseline_parameters, source_values=source)
    baseline_chroma_weights = region_weights(source_labels, protection, baseline_parameters, chroma=True, source_values=source)
    np.savez_compressed(output/'source-weights.npz', values=weights, chroma_values=chroma_weights)
    parameters['binding']['source_weights_sha256'] = hashlib.sha256((output/'source-weights.npz').read_bytes()).hexdigest()
    (output/'parameters.json').write_text(json.dumps(parameters, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    previous_native = None
    if args.baseline_parameters:
        previous_parameters = json.loads(args.baseline_parameters.read_text(encoding='utf8'))
        if any(previous_parameters['binding'][role] != run[role]['sha256'] for role in ('source', 'reference')):
            raise ValueError('Comparison parameters belong to a different image pair')
        previous_weights = region_weights(source_labels, protection, previous_parameters, source_values=source)
        previous_chroma_weights = region_weights(source_labels, protection, previous_parameters, chroma=True, source_values=source)
        # A nonlinear colour transform does not commute with resizing. Apply
        # both parameter sets at native resolution before comparing patches.
        previous_native = np.empty_like(scene)
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
    # Capture original native crops before the owned array is matched in place.
    originals = {n: linear_preview(scene[b[1]:b[3], b[0]:b[2]]) for n, b in boxes.items()}
    reference_crops = {}
    for name, box in boxes.items():
        h = np.diag([native[0]/size[0], native[1]/size[1], 1.]) @ matrix
        h = np.array([[1, 0, -box[0]], [0, 1, -box[1]], [0, 0, 1.]]) @ h
        reference_crops[name] = linear_preview(cv2.warpPerspective(ref, h, (box[2]-box[0], box[3]-box[1])))
    source_preview = linear_preview(source)
    baseline_crops = {}
    for name, box in boxes.items():
        xs = ((np.arange(box[0], box[2])+.5)*size[0]/native[0]-.5).astype(np.float32)
        ys = ((np.arange(box[1], box[3])+.5)*size[1]/native[1]-.5).astype(np.float32)
        mapx, mapy = np.meshgrid(xs, ys)
        sampled = cv2.remap(baseline_weights, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sampled_chroma = cv2.remap(baseline_chroma_weights, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        baseline = apply(scene[box[1]:box[3], box[0]:box[2]], sampled, baseline_parameters,
                         origin=(box[0], box[1]), canvas_size=native, chroma_weights=sampled_chroma)
        baseline_crops[name] = linear_preview(baseline)
    current_crops, comparison_baseline = comparison_crops(plan, run, boxes, native, originals)
    newly_clipped = 0
    for top in range(0, len(scene), 128):
        end = min(len(scene), top+128)
        # Maps sampled in original coordinates; native person RGB is never resized.
        xs = ((np.arange(native[0])+.5)*size[0]/native[0]-.5).astype(np.float32)
        ys = ((np.arange(top, end)+.5)*size[1]/native[1]-.5).astype(np.float32)
        mapx, mapy = np.meshgrid(xs, ys)
        sampled = cv2.remap(weights, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sampled_chroma = cv2.remap(chroma_weights, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if previous_native is not None:
            old_sampled = cv2.remap(previous_weights, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            old_sampled_chroma = cv2.remap(previous_chroma_weights, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            previous_native[top:end] = apply(scene[top:end], old_sampled, previous_parameters,
                                             origin=(0, top), canvas_size=native, chroma_weights=old_sampled_chroma)
        matched = apply(scene[top:end], sampled, parameters, origin=(0, top), canvas_size=native,
                        chroma_weights=sampled_chroma)
        newly_clipped += int(np.count_nonzero((matched >= 1-1e-7) & (scene[top:end] < 1-1e-7)))
        scene[top:end] = matched
    matched_preview = linear_preview(resize_plane(scene, size))
    matched_preview.save(output/'matched-preview.png')
    for name, box in boxes.items():
        crop = linear_preview(scene[box[1]:box[3], box[0]:box[2]])
        crop.save(output/f'native-{name}.png')
        candidate_label = 'A+B + authored jaw colour' if authored_binding else 'Candidate A+B'
        sheet([originals[name], reference_crops[name], current_crops[name], crop], ['Original', 'Reference (colour target)', comparison_baseline['label'], candidate_label], 2200).save(output/f'{name}-comparison.jpg', quality=97)
        sheet([originals[name], baseline_crops[name], crop], ['Original', 'Candidate A', candidate_label], 2000).save(output/f'{name}-ab.jpg', quality=97)
    sheet([source_preview, linear_preview(ref_aligned), matched_preview], ['Original', 'Reference', 'Candidate (before composition)'], 2000).save(output/'whole-comparison.jpg', quality=96)
    values = []
    previous_values = []
    matched_proxy = resize_plane(scene, size)
    previous_proxy = resize_plane(previous_native, size) if previous_native is not None else None
    for a in anchors:
        left, top, right, bottom = a['box']; box = (slice(top, bottom), slice(left, right))
        valid = np.unpackbits(np.frombuffer(bytes.fromhex(a['sample_mask_bits']), dtype=np.uint8))[:(right-left)*(bottom-top)].reshape(bottom-top, right-left).astype(bool)
        new = np.median(to_lab(matched_proxy[box])[valid], axis=0)
        values.append({'region': a['region'], 'xy': a['xy'], 'source': a['source'], 'reference': a['reference'], 'matched': new.tolist()})
        if previous_proxy is not None:
            old = np.median(to_lab(previous_proxy[box])[valid], axis=0)
            previous_values.append({'region': a['region'], 'matched': old})
    zero = apply(source, weights, parameters, strength=0, canvas_size=size, chroma_weights=chroma_weights)
    report = {'schema': 'reference_colour_evaluation.v2', 'run_id': run['run_id'], 'scope': 'native_person_before_composition',
        'working_domain': 'display_linear_sRGB', 'native_crop_boxes': boxes, 'accepted_anchor_count': len(anchors),
        'zero_strength_exact': bool(np.array_equal(zero, source)), 'finite_and_bounded': bool(np.isfinite(scene).all() and scene.min() >= 0 and scene.max() <= 1),
        'newly_clipped_channels': newly_clipped,
        'authored_colour_edit': authored_binding, 'comparison_baseline': comparison_baseline,
        'protection_policy': parameters['protection_policy'],
        'material_statistics': {n: op['evidence'] for n, op in parameters['operators'].items()},
        'reference_geometry_used_for_output': False, 'reference_texture_used_for_output': False,
        'professional_acceptance': False, 'anchor_observations': values}
    if previous_values:
        compared = {}
        for name in REGIONS:
            selected = [i for i, row in enumerate(values) if row['region'] == name]
            if not selected:
                continue
            reference = np.array([values[i]['reference'] for i in selected])
            old = np.array([previous_values[i]['matched'] for i in selected])
            new = np.array([values[i]['matched'] for i in selected])
            compared[name] = {'identical_anchors': len(selected),
                'before_lightness_MAE': float(np.mean(abs(reference[:, 0]-old[:, 0]))),
                'after_lightness_MAE': float(np.mean(abs(reference[:, 0]-new[:, 0]))),
                'before_chroma_median_error': float(np.median(np.linalg.norm(reference[:, 1:]-old[:, 1:], axis=1))),
                'after_chroma_median_error': float(np.median(np.linalg.norm(reference[:, 1:]-new[:, 1:], axis=1)))}
        report['full_application_comparison'] = {'baseline_sha256': hashlib.sha256(args.baseline_parameters.read_bytes()).hexdigest(),
            'scope': 'same_accepted_masks; both_applied_at_native_resolution_then_resized; actual_protection_weights_fields_tone_and_gamut_mapping', 'regions': compared}
    (output/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'output': str(output), 'operators': parameters['operators'], 'matrix_fit': parameters['matrix_fit']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
