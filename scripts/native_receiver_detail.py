"""Restore source-coordinate receiver detail inside actually reviewed windows.

QA localizes a lost detail observation, not a reflection segmentation. Learned
joint floor support, two native alphas and sampling validity independently limit
permission. No person pixels, source floor colour or synthetic reflection enter.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from native_receiver_preservation import (DOMAIN, array, sample_mask, sha,
    transfer_receiver_bands, write)

SCHEMA = 'native_receiver_detail.v1'
CONFIG = {'minimum_validity': .999, 'minimum_receiver': .5,
          'lowpass_candidate_pixels': 4., 'inward_feather_candidate_pixels': 2.,
          'maximum_native_roi_pixels': 2_000_000, 'minimum_permitted_pixels': 24,
          'maximum_below_sole_fraction_of_shoe_width': .5,
          'maximum_floor_gap_candidate_pixels': 16.,
          'maximum_receiver_alpha': .5, 'minimum_floor_run_candidate_pixels': 2.}
REGIONS = frozenset(('left_shoe_contact', 'right_shoe_contact'))


def reviewed_floor_continuation(floor, footwear, alpha, source_alpha, labels, *, pitch):
    """Continue observed floor into a bounded under-shoe segmentation gap.

    Photographic QA owns the checked window. A source opaque shoe locates its
    sole; nearby joint floor establishes a receiver seed. This does not infer
    a reflection shape, airborne contact, or receiver outside that context.
    """
    from scipy.ndimage import label
    floor = np.asarray(floor, bool)
    shoes = (footwear > .5) & (alpha >= .5) & (source_alpha >= .5)
    continuation = np.zeros_like(floor)
    yy, xx = np.indices(floor.shape)
    observations = []
    if not floor.any():
        return continuation, observations
    safe = (alpha < CONFIG['maximum_receiver_alpha']) & (source_alpha < CONFIG['maximum_receiver_alpha'])
    stable_floor = floor & safe
    run = max(2, int(np.ceil(CONFIG['minimum_floor_run_candidate_pixels']*pitch)))
    for offset in range(1, run):
        stable_floor[:-offset] &= (floor & safe)[offset:]
        stable_floor[-offset:] = False
    owners = []
    for semantic_id in (9, 18):
        components, count = label(shoes & (labels == semantic_id), np.ones((3, 3), np.uint8))
        owners.extend((semantic_id, component, components == component) for component in range(1, count+1))
    for semantic_id, component, owned in owners:
        ys, xs = np.nonzero(owned)
        if len(xs) < CONFIG['minimum_permitted_pixels']:
            continue
        width = int(xs.max()-xs.min()+1)
        depth = min(width*CONFIG['maximum_below_sole_fraction_of_shoe_width'],
                    CONFIG['maximum_floor_gap_candidate_pixels']*pitch)
        sole = np.full(floor.shape[1], -1, np.int32)
        np.maximum.at(sole, xs, ys)
        below = (sole[xx] >= 0) & (yy > sole[xx])
        endpoints = np.min(np.where(below & stable_floor,
                                    yy, floor.shape[0]), axis=0)
        bounded = (sole >= 0) & (endpoints < floor.shape[0]) & (endpoints-sole <= depth)
        obstacle = below & (yy < endpoints[xx]) & ~safe
        bounded &= ~np.any(obstacle, axis=0)
        pocket = (below & bounded[xx] & (yy < endpoints[xx]) & ~floor
                  & safe)
        continuation |= pocket
        observations.append({'source_shoe_semantic_id': semantic_id, 'source_opaque_shoe_component': component, 'shoe_width_native': width,
            'maximum_below_sole_native': float(depth), 'maximum_joint_floor_gap_native': float(depth),
            'continued_background_pixels': int(pocket.sum())})
    return continuation, observations


def qualified_review_windows(run):
    """Select explicit paired findings; callers still verify all actual bytes."""
    review = run.get('cosplay_post_review') or {}
    if review.get('status') != 'reviewed' or review.get('actual_api_evidence_valid') is not True:
        return []
    result = []
    for row in review.get('local_observations') or []:
        checks = row.get('checks') or {}
        detail = checks.get('native_detail') or {}
        box = row.get('native_box')
        if (row.get('region') not in REGIONS
                or row.get('scope') != 'matched_source_final_native_pair_only'
                or detail.get('status') != 'fail'
                or detail.get('correction_stage') != 'native_retouch'
                or detail.get('cause') != 'retouch_missing'
                or any((checks.get(key) or {}).get('status') != 'pass'
                       for key in ('identity_makeup', 'skin_materials'))
                or not isinstance(box, list) or len(box) != 4
                or any(type(value) is not int for value in box)):
            continue
        panels = {row.get('source_panel'), row.get('final_panel')}
        cited = {panel for evidence in detail.get('evidence') or []
                 if evidence.get('basis') == 'visible_pixels' for panel in evidence.get('panels') or []}
        if None in panels or not panels <= cited:
            continue
        result.append({'pair_id': row['pair_id'], 'region': row['region'],
                       'native_box': box.copy(), 'source_panel': row['source_panel'],
                       'final_panel': row['final_panel'], 'failed_criteria': [key for key in
                           ('native_detail', 'scene_integration')
                           if (checks.get(key) or {}).get('status') == 'fail'
                           and (checks.get(key) or {}).get('cause') == 'retouch_missing'
                           and (checks.get(key) or {}).get('correction_stage') == 'native_retouch']})
    return result


def validate_diagnosis(root, run):
    import single_source_workflow as workflow
    import system_post_completion as completion
    from cosplay_post_review import validate_saved_comprehensive_review
    root = Path(root).resolve()
    windows = qualified_review_windows(run)
    if not windows or not completion._review_current(run):
        raise ValueError('Receiver detail needs a current bound paired native detail failure')
    path = root/workflow.RUNS_DIR/run['run_id']/'cosplay-post-review/review.json'
    saved = json.loads(path.read_text(encoding='utf-8-sig'))
    if saved != run['cosplay_post_review']:
        raise ValueError('Receiver diagnosis differs from its saved actual receipt')
    images = saved['binding']['images']
    checked, _ = validate_saved_comprehensive_review(Path(images['source']['path']),
        Path(images['reference']['path']), path, run['generation_design'])
    if checked != saved:
        raise ValueError('Receiver native evidence cannot be reopened')
    validate_diagnosed_plate(root, run)
    return windows, {'path': str(path), 'sha256': sha(path)}


def validate_diagnosed_plate(root, run):
    """Bind the proposed donor/plate to the saved render actually diagnosed."""
    root = Path(root).resolve()
    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root/path).resolve()
    saved = json.loads((root/'outputs/runs'/run['run_id']/'manifest.json').read_text(encoding='utf-8-sig'))
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    if plan != saved['options']['appearance_revision']['design_reconstruction_plan']:
        raise ValueError('Receiver repair has unsaved or changed composition assets')
    metadata = json.loads(resolve(run['artifacts']['metadata']).read_text(encoding='utf-8-sig'))
    if (metadata != run.get('render') or metadata.get('run_id') != run['run_id']
            or metadata.get('revision') != run['revision']):
        raise ValueError('Receiver repair render receipt differs from the saved endpoint')
    execution = metadata['operator']['design_reconstruction_plan']
    if (execution.get('working_domain') != DOMAIN
            or execution.get('foreground_alpha_sha256') != plan['mask_roles']['foreground_alpha']['sha256']):
        raise ValueError('Receiver repair alpha differs from the diagnosed native render')
    executed_layers = {row['id']: row for row in execution['layers']}
    for row in plan['layers']:
        if row.get('kind') not in {'background_design', 'contact_shadow'}:
            continue
        recorded = executed_layers.get(row['id']) or {}
        if (recorded.get('kind') != row['kind'] or recorded.get('assets') != row.get('assets')
                or recorded.get('enabled') != bool(row.get('enabled'))
                or recorded.get('strength') != float(row.get('strength', 1))):
            raise ValueError('Receiver repair plate or floor assets differ from the diagnosed render')
    observed = run['cosplay_post_review']['native_selection']['source_baseline_assets']
    baseline = plan['source_baseline']['assets']
    for role in ('foreground_premultiplied', 'foreground_alpha', 'old_environment'):
        if baseline[role]['sha256'] != observed[role]['sha256'] or sha(resolve(baseline[role]['path'])) != observed[role]['sha256']:
            raise ValueError('Receiver detail source donor differs from the actual reviewed baseline')
    return {'metadata': {'path': str(resolve(run['artifacts']['metadata'])),
                          'sha256': sha(resolve(run['artifacts']['metadata']))},
            'plate_matches_saved_render': True, 'source_baseline_matches_native_QA': True}


def prepare_receiver_detail(root, run, output):
    """Prepare one union of reviewed windows and an immutable background asset."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    windows, diagnosis = validate_diagnosis(root, run)
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    result = copy.deepcopy(plan)
    receipt = {'schema': SCHEMA, 'state': 'unknown', 'config': CONFIG.copy(),
        'producer_sha256': sha(__file__), 'source_sha256': run['source']['sha256'],
        'reference_sha256': run['reference']['sha256'], 'diagnosis': diagnosis,
        'reviewed_windows': windows, 'visual_acceptance': False,
        'model_inferences': 0, 'image_generation_submissions': 0,
        'source_rgb_warped': False, 'synthetic_reflection_created': False}

    def finish(reason, changed=False):
        receipt['reason'] = reason
        write(output/'receiver-detail.json', receipt)
        prepared = {'ok': True, 'changed': changed, 'design_reconstruction_plan': result,
                    'receipt': receipt, 'qualification': receipt.get('qualification', {})}
        write(output/'prepared-plan.json', prepared)
        return prepared

    native = plan['canvas']['source_oriented_size']
    proxy = plan['canvas']['source_proxy_size']
    if (plan.get('working_domain') != DOMAIN or plan.get('source_sha256') != run['source']['sha256']
            or plan.get('candidate_sha256') != run['reference']['sha256']
            or plan['canvas'].get('candidate_mapping', {}).get('scene_registration_supported') is not True):
        return finish('registered_native_source_receiver_required')
    background_layer = next(row for row in result['layers'] if row['kind'] == 'background_design')
    assets = background_layer['assets']
    if assets.get('native_receiver_detail'):
        return finish('saved_receiver_detail_not_applied_twice')
    from reference_appearance_prepare import preserve_contact_layers
    contacts, _ = preserve_contact_layers(plan, run)
    if not contacts:
        return finish('current_joint_floor_observation_unavailable')
    joint_asset = contacts[0]['assets']['receiver_support']
    floor_asset = contacts[0]['assets']['source_support_surface']
    footwear_asset = contacts[0]['assets'].get('source_footwear')
    semantics_asset = contacts[0]['assets'].get('source_shoe_semantics')
    if (joint_asset.get('role') != 'joint_source_candidate_receiver_support'
            or floor_asset.get('role') != 'learned_source_receiver_support'
            or any(asset.get('coordinate_space') != 'final_source_proxy'
                   or asset.get('original_canvas_size') != native or asset.get('size') != proxy
                   for asset in (joint_asset, floor_asset))):
        raise ValueError('Receiver detail floor observation coordinates changed')
    baseline = plan['source_baseline']
    old = baseline['assets']['old_environment']
    observed_alpha = baseline['assets']['foreground_alpha']
    alpha = plan['mask_roles']['foreground_alpha']
    background, validity = assets['background'], assets['validity']
    if (baseline.get('working_domain') != DOMAIN
            or any(asset.get('size') != native or asset.get('coordinate_space') != 'source_canvas'
                   for asset in (old, observed_alpha, alpha, background, validity))
            or background.get('representation') != 'sampling_validity_associated_RGB'):
        raise ValueError('Receiver detail needs frozen native display-linear source and background')
    mapping = np.asarray(plan['canvas']['candidate_mapping']['candidate_original_to_final_original'], float)
    if (mapping.shape != (3, 3) or not np.isfinite(mapping).all() or np.linalg.det(mapping) <= 0):
        raise ValueError('Receiver detail registration is invalid')
    pitch = max(np.linalg.norm(mapping[:2, 0]), np.linalg.norm(mapping[:2, 1]), 1.)
    sigma = max(2., CONFIG['lowpass_candidate_pixels']*pitch)
    margin = int(np.ceil(4*sigma+pitch+2))
    boxes = [row['native_box'] for row in windows]
    if any(not (0 <= b[0] < b[2] <= native[0] and 0 <= b[1] < b[3] <= native[1]) for b in boxes):
        raise ValueError('Reviewed receiver window outside the native source canvas')
    box = [max(0, min(b[0] for b in boxes)-margin), max(0, min(b[1] for b in boxes)-margin),
           min(native[0], max(b[2] for b in boxes)+margin), min(native[1], max(b[3] for b in boxes)+margin)]
    left, top, right, bottom = box
    if (right-left)*(bottom-top) > CONFIG['maximum_native_roi_pixels']:
        return finish('reviewed_receiver_union_exceeds_native_roi_budget')
    sl = np.s_[top:bottom, left:right]
    native_to_proxy = np.diag([proxy[0]/native[0], proxy[1]/native[1], 1.])
    joint = sample_mask(array(joint_asset), native_to_proxy, box)
    floor = sample_mask(array(floor_asset), native_to_proxy, box)
    current_alpha, source_alpha, coverage = array(alpha)[sl], array(observed_alpha)[sl], array(validity)[sl]
    if any(np.any((v < 0) | (v > 1)) for v in (joint, floor, current_alpha, source_alpha, coverage)):
        raise ValueError('Receiver detail support outside the unit interval')
    receiver = ((joint > CONFIG['minimum_receiver']) & (floor > CONFIG['minimum_receiver'])
                & (current_alpha < CONFIG['maximum_receiver_alpha'])
                & (source_alpha < CONFIG['maximum_receiver_alpha']) & (coverage >= CONFIG['minimum_validity']))
    reviewed = np.zeros(receiver.shape, bool)
    for x0, y0, x1, y1 in boxes:
        reviewed[y0-top:y1-top, x0-left:x1-left] = True
    continued = np.zeros_like(receiver)
    continuation_observations = []
    if footwear_asset and semantics_asset:
        if (footwear_asset.get('role') != 'mapped_source_footwear_observation'
                or footwear_asset.get('coordinate_space') != 'final_source_proxy'
                or footwear_asset.get('original_canvas_size') != native or footwear_asset.get('size') != proxy):
            raise ValueError('Reviewed source shoe continuation coordinates changed')
        footwear = sample_mask(array(footwear_asset), native_to_proxy, box)
        if sha(semantics_asset['path']) != semantics_asset['sha256'] or semantics_asset.get('coordinate_space') != 'source_proxy':
            raise ValueError('Receiver source shoe semantic observation changed')
        with np.load(semantics_asset['path'], allow_pickle=False) as arrays:
            semantic_labels = arrays['labels']
        if semantic_labels.shape != (proxy[1], proxy[0]) or not np.issubdtype(semantic_labels.dtype, np.integer):
            raise ValueError('Receiver source shoe semantic grid changed')
        native_labels = sample_mask(semantic_labels, native_to_proxy, box)
        continued, continuation_observations = reviewed_floor_continuation(
            (joint > CONFIG['minimum_receiver']) & (floor > CONFIG['minimum_receiver']),
            footwear, current_alpha, source_alpha, native_labels, pitch=pitch)
        continued &= reviewed & (coverage >= CONFIG['minimum_validity'])
        receiver |= continued
    permitted = receiver & reviewed
    receipt['qualification'] = {'qualified': bool(permitted.sum() >= CONFIG['minimum_permitted_pixels']),
        'native_box': box, 'permitted_pixels': int(permitted.sum()),
        'reviewed_windows_unioned_before_transfer': True, 'sigma_native_pixels': float(sigma),
        'sampling_pitch_native': float(pitch), 'actual_paired_QA_verified': True,
        'floor_assets': {'joint_receiver': joint_asset, 'source_receiver': floor_asset},
        'source_footwear': copy.deepcopy(footwear_asset),
        'source_shoe_semantics': copy.deepcopy(semantics_asset),
        'floor_continuation': {'source_opaque_shoes': continuation_observations,
            'continued_reviewed_pixels': int(continued.sum()),
            'scope': 'reviewed_background_below_source_sole_to_stable_joint_floor; Fp_and_alpha_unchanged; not_reflection_shape_inference'},
        'scope': 'reviewed_source_coordinate_receiver_detail_only; no_complete_reflection_segmentation'}
    if permitted.sum() < CONFIG['minimum_permitted_pixels']:
        return finish('insufficient_reviewed_joint_receiver_support')
    source = array(old)[sl].copy()
    values = array(background)
    original = values[sl].copy()
    straight = np.divide(original, coverage[..., None], out=np.zeros_like(original),
                         where=coverage[..., None] > 1e-8)
    after, weight, checks = transfer_receiver_bands(source, straight, permitted, receiver,
        sigma=sigma, feather=max(1., pitch*CONFIG['inward_feather_candidate_pixels']))
    # Both learned floor and its short continuation can have a mixed matte.
    # Fade background confidence toward the opaque barrier uniformly; the
    # actual (1-A) over operation remains solely in the renderer.
    t = np.clip(1-np.maximum(current_alpha, source_alpha)/CONFIG['maximum_receiver_alpha'], 0, 1)
    confidence = t*t*(3-2*t)
    after = straight+(after-straight)*confidence[..., None]
    weight *= confidence
    from scipy.ndimage import gaussian_filter
    from native_receiver_preservation import LUMA
    denominator = np.maximum(gaussian_filter(receiver.astype(np.float32), sigma), 1e-8)
    def low_luminance(rgb):
        return gaussian_filter(np.where(receiver, rgb @ LUMA, 0), sigma)/denominator
    low_before, low_after = low_luminance(straight), low_luminance(after)
    drift = (low_after-low_before)[permitted]
    receipt['measured_low_frequency_luminance_drift'] = {
        'mean': float(drift.mean()), 'maximum_absolute': float(np.abs(drift).max()),
        'low_frequency_is_exact': False}
    active = weight > 0
    patch = original.copy()
    patch[active] = after[active]*coverage[active, None]
    changed = np.any(patch != original, axis=-1)
    if not changed.any():
        return finish('receiver_detail_produced_no_change')
    if np.any(changed & ~permitted) or np.any(changed & (current_alpha >= CONFIG['maximum_receiver_alpha'])):
        raise ValueError('Receiver detail crossed its reviewed background support')
    values[sl] = patch
    target = output/'receiver-detail-background.npz'
    np.savez_compressed(target, values=values)
    del values
    new_asset = {**background, 'path': str(target), 'sha256': sha(target),
                 'native_receiver_detail_origin': 'current_paired_QA_source_luminance_bands'}
    support_path = output/'receiver-detail-support.npz'
    np.savez_compressed(support_path, values=weight)
    receipt.update(state='prepared', reason='observed_native_receiver_detail_restored', input_background=copy.deepcopy(background),
        output_background=new_asset, source_environment=copy.deepcopy(old),
        validity=copy.deepcopy(validity), alpha=copy.deepcopy(alpha), observed_source_alpha=copy.deepcopy(observed_alpha),
        support_asset={'path': str(support_path), 'sha256': sha(support_path),
            'size': list(weight.shape[::-1]), 'coordinate_space': 'source_canvas_roi',
            'native_box': box, 'original_canvas_size': native},
        checks={**checks, 'changed_pixels': int(changed.sum()),
            'outside_reviewed_support_exact': True, 'outside_native_box_exact_by_construction': True,
            'opaque_foreground_exact': True, 'foreground_premultiplied_and_alpha_assets_unchanged': True,
            'partial_alpha_background_changed_only_in_reviewed_joint_floor_or_under_sole_continuation': True,
            'validity_and_contact_attenuation_unchanged': True},
        limitations=['Only reviewed joint floor windows qualify; whole-floor quality remains unverified.',
                     'Candidate low-frequency colour remains; original native luminance detail is restored.',
                     'A new saved child requires its own final visual review.'])
    receipt_path = output/'receiver-detail.json'
    write(receipt_path, receipt)
    background_layer['assets']['background'] = new_asset
    background_layer['assets']['native_receiver_detail'] = {'path': str(receipt_path), 'sha256': sha(receipt_path),
        'representation': 'diagnosis_bound_native_receiver_detail_provenance'}
    result.setdefault('provenance', {})['native_receiver_detail'] = {
        'state': 'prepared', 'receipt_sha256': sha(receipt_path), 'visual_acceptance': False}
    return finish('observed_native_receiver_detail_restored', changed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run = json.loads(args.run.read_text(encoding='utf-8-sig'))
    prepared = prepare_receiver_detail(args.root, run, args.output)
    print(json.dumps({'ok': True, 'changed': prepared['changed'],
        'qualification': prepared['qualification'], 'reason': prepared['receipt']['reason']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
