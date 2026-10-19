"""Prepare own-canvas material pools, review actual crops, qualify colour targets.

This invokes CPU face landmarks and optional text vision only. It never starts
segmentation, image generation or GPU inference. All large arrays remain files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import cosplay_post_review as visual
from float_appearance import srgb_to_linear
from local_face_protection import prepare_face_protection
import reference_material_evidence as targets
import reference_material_solver as current
import vision_transport

SCHEMA = 'material_target_inputs.v2'


def file_record(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def write(path, value):
    visual._write(path, value)


def load_observation(spec):
    """Load a JSON spec; reverify arrays, original raster and all bound files."""
    binding = spec['binding']
    for key in ('image', 'analysis', 'semantic_labels'):
        targets._file(binding[key])
    with np.load(binding['semantic_labels']['path'], allow_pickle=False) as resource:
        labels = np.asarray(resource['labels'])
    masks = {name: targets._asset_array(record, 'values') for name, record in binding['feature_masks'].items()}
    observation = {'role': spec['role'], 'domain': targets.DOMAIN,
        'rgb': srgb_to_linear(np.asarray(visual._raster(binding['image']['path']), np.float32)/255),
        'labels': labels, 'feature_masks': masks, 'binding': binding}
    targets._observation(observation)
    return observation


def load_inputs(path):
    document = targets._json(path)
    if document.get('schema') != SCHEMA:
        raise ValueError('Unknown material-target input schema')
    source, reference = [load_observation(document[name]) for name in ('source_observation', 'reference_observation')]
    qualification_record = targets._file(document['qualification'])
    review_record = targets._file(document['anchor_review'])
    qualification = targets._json(qualification_record['path'])
    review = targets._json(review_record['path'])
    recomputed = current.qualify_material_targets(source, reference, document['diagnosis'], review)
    if qualification != recomputed:
        raise ValueError('Saved material qualification differs from actual reopened crop evidence')
    return source, reference, qualification, document['diagnosis']


def prepare_observation(root, role, displayed, identity_root, analysis_path, output, *, maximum_edge=2048,
                        registry_path=None):
    displayed, identity_root, analysis_path, output = map(lambda value: Path(value).resolve(),
                                                       (displayed, identity_root, analysis_path, output))
    output.mkdir(parents=True, exist_ok=True)
    analysis = targets._json(analysis_path)
    if (analysis['source']['sha256'] != visual._sha(identity_root)
            or analysis.get('analysis_role') != ('source' if role == 'source' else 'candidate')):
        raise ValueError('Explicit semantic analysis belongs to another identity root or role')
    image = visual._raster(displayed)
    # Keep thumbnail's aspect-ratio geometry, but use one Lanczos resize from
    # the canonical sRGB parent. Its default reducing_gap pre-reduces large
    # images and produces different pixels from the strictly bound parent
    # resize checked by reference_material_evidence._observation.
    image.thumbnail((maximum_edge, maximum_edge), Image.Resampling.LANCZOS,
                    reducing_gap=None)
    # _raster has already converted the pixels to canonical sRGB. Its generated
    # ICC profile embeds a creation time, so omit it from this CPU cache input.
    image_path = output/'display.png'; image.save(image_path, icc_profile=None)
    with np.load(analysis['semantic_result']['labels_path'], allow_pickle=False) as resource:
        labels = np.asarray(Image.fromarray(np.asarray(resource['labels'], np.int32)).resize(image.size, Image.Resampling.NEAREST)).astype(np.uint8)
    labels_path = output/'labels.npz'; np.savez_compressed(labels_path, labels=labels)
    protection = prepare_face_protection(Path(root), image_path,
        {**analysis['semantic_result'], 'labels_path': str(labels_path)}, registry_path=registry_path)
    masks = {'eyes': np.zeros(labels.shape, np.float32), 'lips': np.isin(labels, [24, 25]).astype(np.float32),
             'teeth': np.isin(labels, [26, 27]).astype(np.float32), 'tongue': (labels == 28).astype(np.float32)}
    for feature in protection.get('protected_regions', []):
        name = 'eyes' if feature['region'] in ('left_eye', 'right_eye') else 'lips'
        canvas = Image.new('L', image.size)
        ImageDraw.Draw(canvas).polygon([tuple(point) for point in feature['polygon']], fill=255)
        masks[name] = np.maximum(masks[name], np.asarray(canvas, np.float32)/255)
    if protection['method'] == 'no_landmarks_protect_entire_face_neck':
        masks['face_features'] = (labels == 3).astype(np.float32)
    feature_records = {}
    for name, values in masks.items():
        path = output/(name+'.npz'); np.savez_compressed(path, values=values)
        feature_records[name] = file_record(path)
    protection_receipt = output/'feature-ownership.json'
    write(protection_receipt, {'schema': 'material_feature_ownership.v1', 'input': file_record(image_path),
        'semantic_labels': file_record(labels_path), 'face_observation': protection,
        'feature_masks': feature_records,
        'limitations': ['Canonical eye/lip topology, not independent iris/contact-lens or character-makeup classification.']})
    binding = {'image': file_record(image_path), 'identity_root': file_record(identity_root),
        'analysis': file_record(analysis_path), 'semantic_labels': {**file_record(labels_path),
            'parent': file_record(analysis['semantic_result']['labels_path'])},
        'feature_masks': feature_records, 'feature_ownership_receipt': file_record(protection_receipt),
        'display_parent': file_record(displayed), 'display_derivative': 'oriented_srgb_lanczos'}
    spec = {'role': role, 'domain': targets.DOMAIN, 'binding': binding}
    write(output/'observation.json', spec)
    return spec, load_observation(spec)


def prepare_sheet(source, reference, selected_pools, output):
    """Exact crops plus labelled display zooms and full-image context <=1024."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    observations = {'source': source, 'reference': reference}
    images = {role: visual._raster(value['binding']['image']['path']) for role, value in observations.items()}
    columns = 3; tile_width = 330; tile_height = 145; context_height = 500
    count = len(selected_pools)
    if count > 9:
        raise ValueError('At most nine exact material crops fit one bounded review sheet')
    sheet = Image.new('RGB', (990, context_height+((count+2)//3)*tile_height), '#171717')
    draw = ImageDraw.Draw(sheet)
    for index, (role, image) in enumerate(images.items()):
        context = image.copy(); context.thumbnail((320, context_height-25), Image.Resampling.LANCZOS)
        origin = (index*330, 25); sheet.paste(context, origin); draw.text((index*330+6, 7), role+' full context (display scaled)', fill='white')
    draw.text((670, 40), 'Exact crops and labelled zooms below.', fill='white')
    draw.text((670, 60), 'Black/white inset = selected material.', fill='white')
    anchors = []
    for index, pool in enumerate(selected_pools):
        crop = images[pool['role']].crop(tuple(pool['box']))
        if crop.width > 110 or crop.height > 110:
            raise ValueError('Exact material crop exceeds this review sheet; use a finer automatic subdivision')
        x, y = (index%columns)*tile_width, context_height+(index//columns)*tile_height
        draw.text((x+5, y+4), pool['anchor_id'], fill='white')
        panel_box = [x+5, y+25, x+5+crop.width, y+25+crop.height]
        sheet.paste(crop, (panel_box[0], panel_box[1]))
        zoom = crop.copy(); zoom = zoom.resize((max(1, round(crop.width*min(110/crop.width, 110/crop.height))),
                                                max(1, round(crop.height*min(110/crop.width, 110/crop.height)))), Image.Resampling.LANCZOS)
        sheet.paste(zoom, (x+115, y+25))
        selected = Image.fromarray(np.uint8(pool['mask'])*255)
        sheet.paste(selected, (x+230, y+25))
        crop_path = output/(pool['anchor_id'].replace(':', '-')+'.png'); crop.save(crop_path)
        anchors.append({'anchor_id': pool['anchor_id'], 'crop': {**file_record(crop_path), 'box': pool['box'],
            'panel_box': panel_box, 'parent_sha256': observations[pool['role']]['binding']['image']['sha256'],
            'coordinate_space': pool['coordinate_space']}})
    sheet_path = output/'comparison.png'; sheet.save(sheet_path)
    write(output/'crop-binding.json', {'sheet': file_record(sheet_path), 'anchors': anchors,
        'pools': [{key: value for key, value in row.items() if key != 'mask'} for row in selected_pools],
        'display_zoom_only': True, 'native_visual_acceptance': False})
    return sheet_path, anchors


def review_pools(source, reference, diagnosis, pools, output, *, provider, vision_model):
    output = Path(output)
    sheet, anchors = prepare_sheet(source, reference, pools, output)
    prompt = '''Judge EACH labelled material crop independently as a colour target for the same person's original-photo colour matching. This is not image generation or physical relighting. Full source and existing reference context are shown above. Each lower tile shows its exact crop, a labelled display zoom, and a black/white mask identifying the material pixels eligible for statistics; judge those selected material pixels only. Preserve intended environmental ambience; do not demand uniform neutral skin. A crop can be qualified even when another region of this reference is defective. Qualify a target only if its selected skin pixels have a plausible photographic material colour and smooth tonal transitions, with no unresolved severe green/red contamination, clipped waxy skin, generated HDR/embossed fake texture, boundary mixing, or makeup/eye/lip contamination. Rejected or unknown targets are never used. A merely smoother patch is not proof of correct skin colour. Use visible evidence and original makeup/material context; if ambiguous return unknown. This judgement qualifies colour statistics only, not exact identity, geometry, lighting recovery or native detail.
Return {"anchors":[{"anchor_id":"one exact supplied id","status":"qualified|rejected|unknown","reason":"specific visible evidence, concise"}]} with exactly one row for every supplied id:
'''+ '\n'.join(row['anchor_id'] for row in anchors)
    prompt_path = output/'prompt.txt'; prompt_path.write_text(prompt, encoding='utf8')
    try:
        payload, provenance = vision_transport.request_vision_json(sheet, output/'vision', prompt,
            provider=provider, vision_model=vision_model)
    except Exception as exc:
        write(output/'unavailable.json', {'schema': 'material_anchor_review.v1', 'status': 'unavailable',
            'comparison': file_record(sheet), 'prompt': file_record(prompt_path), 'model': vision_model,
            'error': {'type': type(exc).__name__, 'message': str(exc)[:500]},
            'qualified_targets_granted': False, 'image_generation_used': False})
        raise
    response_path, provenance_path = output/'response.json', output/'provenance.json'
    write(response_path, payload); write(provenance_path, provenance)
    visual._validate_provenance(provenance, sheet, prompt)
    actual = {row['anchor_id']: row for row in payload['anchors']}
    if len(actual) != len(anchors) or set(actual) != {row['anchor_id'] for row in anchors}:
        raise ValueError('Actual API response omitted or duplicated material pools')
    reviewed = [{**row, 'status': actual[row['anchor_id']]['status'], 'reason': actual[row['anchor_id']]['reason']} for row in anchors]
    result = {'schema': 'material_anchor_review.v1', 'status': 'reviewed', 'anchors': reviewed,
        'pool_profile': pools[0]['pool_profile'],
        'binding': {'diagnosis': diagnosis['evidence'], 'comparison': file_record(sheet),
                    'prompt': file_record(prompt_path), 'response': file_record(response_path),
                    'provenance': file_record(provenance_path)},
        'native_visual_acceptance': False, 'image_generation_used': False}
    write(output/'anchor-review.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    for name in ('source-display', 'original-source', 'source-analysis', 'reference', 'reference-analysis', 'diagnosis', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--provider', default='ccswitch_sub2api')
    parser.add_argument('--vision-model', default='gpt-6.1-sol')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--diagnosis-target', choices=('reference','final'), default='reference')
    parser.add_argument('--current-run', type=Path)
    args = parser.parse_args(argv); output = args.output.resolve()
    if (output/'material-target-inputs.json').exists():
        raise FileExistsError('Qualified material inputs already exist; preserve their actual review evidence')
    source_spec, source = prepare_observation(args.root, 'source', args.source_display, args.original_source,
                                             args.source_analysis, output/'source')
    reference_spec, reference = prepare_observation(args.root, 'reference', args.reference, args.reference,
                                                   args.reference_analysis, output/'reference')
    owners = targets._owners(source['labels'], source['feature_masks'])
    regions = [name for name in targets.REGIONS if owners[name].any()]
    diagnosis = {'evidence': file_record(args.diagnosis), 'regions': regions,
                 'target': args.diagnosis_target}
    if args.diagnosis_target == 'final':
        if args.current_run is None:
            raise ValueError('Final-only material qualification requires its current appearance endpoint')
        run = targets._json(args.current_run)
        plan = run['options']['appearance_revision']['design_reconstruction_plan']
        stack = next(row for row in plan['layers'] if row['kind']=='reference_appearance_stack')
        from automatic_material_completion import final_skin_regions
        actual_review=targets._json(args.diagnosis)
        diagnosis['regions']=[name for name in final_skin_regions(actual_review) if owners[name].any()]
        if not diagnosis['regions']:
            raise ValueError('Final material failure has no supported diagnosed source skin owner')
        diagnosis['appearance_endpoint'] = {
            'run_id':run['run_id'], 'revision':run['revision'],
            'source_sha256':run['source']['sha256'], 'reference_sha256':run['reference']['sha256'],
            'final_sha256':visual._sha(run['artifacts']['auto_draft']),
            'alpha_sha256':plan['mask_roles']['foreground_alpha']['sha256'],
            'automatic_delta_sha256':stack['assets']['automatic_delta']['sha256']}
    pools = targets.candidate_material_pools(reference)
    face_pools = [row for row in pools if row['region'] == 'face']
    if not face_pools:
        raise ValueError('No independent reference face material pools are available')
    if args.prepare_only:
        prepare_sheet(source, reference, face_pools[:9], output/'face-review')
        print(json.dumps({'status': 'prepared', 'source_observation': str(output/'source/observation.json'),
                         'reference_observation': str(output/'reference/observation.json')}))
        return 0
    review = review_pools(source, reference, diagnosis, face_pools[:9], output/'face-review',
                          provider=args.provider, vision_model=args.vision_model)
    qualification = current.qualify_material_targets(source, reference, diagnosis, review)
    review_path = output/'face-review/anchor-review.json'
    if not qualification['targets'].get('skin'):
        other_pools = [row for row in pools if row['material_family'] == 'skin' and row['region'] != 'face']
        # Large body pools use finer subdivisions to fit exact pixels. No
        # photographic coordinates or colour-ranked hand-picked patches.
        if any(row['box'][2]-row['box'][0] > 110 or row['box'][3]-row['box'][1] > 110 for row in other_pools):
            pools = targets.candidate_material_pools(reference, subdivisions=5)
            other_pools = [row for row in pools if row['material_family'] == 'skin' and row['region'] != 'face']
        for index in range(0, len(other_pools), 9):
            review_directory = output/('body-review-'+str(index//9+1))
            review = review_pools(source, reference, diagnosis, other_pools[index:index+9], review_directory,
                                  provider=args.provider, vision_model=args.vision_model)
            qualification = current.qualify_material_targets(source, reference, diagnosis, review)
            review_path = review_directory/'anchor-review.json'
            if qualification['targets'].get('skin'):
                break
    qualification_path = output/'qualification.json'; write(qualification_path, qualification)
    inputs = {'schema': SCHEMA, 'source_observation': source_spec, 'reference_observation': reference_spec,
        'diagnosis': diagnosis, 'qualification': file_record(qualification_path), 'anchor_review': file_record(review_path),
        'automatic_accepted': False, 'native_detail_status': 'unknown'}
    inputs_path = output/'material-target-inputs.json'; write(inputs_path, inputs)
    load_inputs(inputs_path)
    print(json.dumps({'status': 'qualified' if qualification['targets'].get('skin') else 'unsupported',
        'inputs': str(inputs_path), 'skin_target_pools': len(qualification['targets'].get('skin', [])),
        'image_generation_used': False, 'gpu_used': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
