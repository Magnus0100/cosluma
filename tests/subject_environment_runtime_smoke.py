"""Synthetic full/ROI/preview, corruption, ownership and portable response proof."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

import design_reconstruction as design
from native_roi_render import render_native_roi
from native_subject_detail_runtime_smoke import fixture as detail_fixture, record, write, sha, closure
from portable_project import export_project, import_project
from subject_environment_runtime import (KIND, SCHEMA, RECIPE, DOMAIN, array_sha,
    apply_environment, foreground_controls_digest, preceding_detail_digest, scene_context_digest,
    _approved_retouch_identity)


def fixture(root, *, with_detail=False):
    run, target, detail_delta, alpha, background = detail_fixture(root)
    plan = design.plan_for_run(run); detail = plan['layers'][-1]
    semantic = plan['mask_roles']['source_semantics']
    semantic_path = root/'environment-owner.npz'
    labels = np.full((32, 48), 4, np.uint8); labels[alpha == 0] = 0
    np.savez_compressed(semantic_path, labels=labels)
    semantic = {**semantic, **record(semantic_path)}; plan['mask_roles']['source_semantics'] = semantic
    if with_detail:
        frozen = json.loads(Path(detail['assets']['frozen_delta']['path']).read_text('utf8'))
        frozen['owner_asset'] = semantic; frozen['binding']['semantic_sha256'] = semantic['sha256']
        qa = json.loads(Path(detail['assets']['qualification_receipt']['path']).read_text('utf8'))
        for window in qa['reviewed_windows']:
            window['region'] = 'hair'; window['owner_class_ids'] = [4]
        detail['params']['binding'] = frozen['binding']
        detail['assets']['frozen_delta'] = write(root/'preceding-detail.json', frozen)
        detail['assets']['qualification_receipt'] = write(root/'preceding-detail-qualification.json', qa)
        target = target+detail_delta
    else:
        plan['layers'].pop()
    protection = np.zeros((32, 48), np.float32); protection[7, 15] = 1
    protect_path = root/'identity-protection.npz'; np.savez_compressed(protect_path, protection=protection)
    protect = {**record(protect_path), 'size': [48, 32]}; plan['mask_roles']['identity_protection'] = protect
    baseline = plan['source_baseline']['assets']; stack = plan['layers'][1]
    binding = {'source_sha256': plan['source_sha256'], 'reference_sha256': plan['candidate_sha256'],
        'source_fp_sha256': baseline['foreground_premultiplied']['sha256'],
        'source_alpha_sha256': baseline['foreground_alpha']['sha256'],
        'alpha_sha256': plan['mask_roles']['foreground_alpha']['sha256'],
        'semantic_sha256': semantic['sha256'], 'native_size': [48, 32], 'working_domain': DOMAIN,
        'appearance_weights': design.reference_stack_execution_weights(stack)['bank_weights'],
        'appearance_banks': {'automatic_delta': stack['assets']['automatic_delta']['sha256']},
        'preceding_detail_digest': preceding_detail_digest(plan),
        'foreground_controls_digest': foreground_controls_digest(run),
        'scene_context_digest': scene_context_digest(plan),
        'identity_protection_sha256': protect['sha256']}
    full_delta = np.zeros_like(target); arrays = {}; patches = []
    for index, box in enumerate(([16, 8, 32, 24], [36, 20, 42, 26])):
        l, t, r, b = box
        delta = np.full((b-t, r-l, 3), .002, np.float32)*alpha[t:b, l:r, None]
        key = f'delta_{index}'; arrays[key] = delta; full_delta[t:b, l:r] = delta
        patches.append({'native_box': box, 'array_key': key, 'delta_array_sha256': array_sha(delta),
            'changed_pixels': int(np.any(delta != 0, axis=-1).sum()), 'owner_class_id': 4,
            'target_patch_sha256': array_sha(target[t:b, l:r])})
    delta_path = root/'environment-delta.npz'; np.savez_compressed(delta_path, **arrays)
    frozen = {'schema': SCHEMA, 'recipe': RECIPE, 'binding': binding, 'delta': record(delta_path),
        'patches': patches, 'donor_assets': baseline, 'owner_asset': semantic, 'approved_retouch_binding': None}
    receipt = {'schema': SCHEMA, 'recipe': RECIPE, 'state': 'prepared',
        'source_sha256': binding['source_sha256'], 'reference_sha256': binding['reference_sha256'],
        'qualification': {'qualified': True, 'actual_saved_evidence_verified': True, 'holdout_improved': True},
        'checks': {'changed_pixels': sum(row['changed_pixels'] for row in patches),
            'alpha_changed': False, 'source_geometry_changed': False, 'approved_source_retouch_reapplied': False,
            'unknown_support_changed': False, 'reference_texture_imported': False}}
    layer = {'id': KIND, 'kind': KIND, 'enabled': True, 'strength': 1.,
        'params': {'schema': SCHEMA, 'recipe': RECIPE, 'binding': binding},
        'assets': {'frozen_delta': write(root/'environment-frozen.json', frozen),
                   'qualification_receipt': write(root/'environment-qualification.json', receipt)}}
    plan['layers'].append(layer)
    return run, target, full_delta, alpha, background


def run_smoke():
    checks = 0
    def require(value):
        nonlocal checks
        assert value; checks += 1
    with tempfile.TemporaryDirectory(prefix='cos-environment-runtime-') as temporary:
        base = Path(temporary)
        with patch('reference_appearance_stack.validate_layer', return_value={}):
            for with_detail in (False, True):
                origin = base/('origin-detail' if with_detail else 'origin'); run, target, delta, alpha, background = fixture(origin, with_detail=with_detail)
                plan = design.plan_for_run(run); layer = plan['layers'][-1]
                values, receipt = apply_environment(target, layer, plan, run)
                require(np.array_equal(values, target+delta)); require(receipt['source_retouch_reapplied'] is False)
                require(np.array_equal(values[alpha == 0], target[alpha == 0]))
                roi = (21, 13, 29, 22)
                clipped, _ = apply_environment(target[13:22, 21:29], layer, plan, run, box=roi, canvas_size=(48, 32))
                require(np.array_equal(clipped, values[13:22, 21:29]))
                outside, _ = apply_environment(target[:4, :6], layer, plan, run, box=(0, 0, 6, 4), canvas_size=(48, 32))
                require(np.array_equal(outside, target[:4, :6]))
                preview_target = design._sample_channels(tuple(Image.fromarray(target[..., c]) for c in range(3)), (0, 0, 24, 16), (24, 16))
                expected_delta = np.stack([np.asarray(Image.fromarray(delta[..., c]).resize((24, 16), Image.Resampling.BILINEAR)) for c in range(3)], -1)
                reduced, _ = apply_environment(preview_target, layer, plan, run)
                require(np.allclose(reduced, preview_target+expected_delta, rtol=0, atol=1e-8))
                geometry = Image.fromarray(np.uint8(alpha*255))
                whole, execution = design.apply_plan(np.zeros_like(target), np.zeros_like(target), plan, geometry, run=run, include_optics=False)
                require(np.array_equal(whole, target+delta+(1-alpha[..., None])*background))
                require(next(row for row in execution['layers'] if row['kind'] == KIND)['applied'])
                native_roi, roi_execution = render_native_roi(run, roi)
                require(np.array_equal(native_roi, whole[13:22, 21:29]))
                require(next(row for row in roi_execution['layers'] if row['kind'] == KIND)['applied'])
                seen_manual = []
                def observe_manual(values, *args, **kwargs):
                    if kwargs.get('branch') == 'foreground': seen_manual.append(values.copy())
                    return values
                with patch('float_appearance.apply_manual_layers_float', side_effect=observe_manual):
                    design.apply_plan(np.zeros_like(target), np.zeros_like(target), plan, geometry, run=run, include_optics=False)
                require(len(seen_manual) == 1 and np.array_equal(seen_manual[0], target+delta))
                # Active effects can wrap operative numeric arrays in owned
                # JSON. Re-signing that wrapper during relocation is inert.
                effect_array = origin/'effect-emission.npz'; np.savez_compressed(effect_array, values=delta)
                effect_json = write(origin/'effect-original.json', {'owner_run_id': 'original',
                    'emission': record(effect_array), 'schema': 'numeric-effect-fixture.v1'})
                scene = copy.deepcopy(plan)
                scene['layers'].append({'id': 'effect-fixture', 'kind': 'independent_effect', 'enabled': True,
                    'strength': .3, 'params': {'coverage_once': True}, 'assets': {'frozen_numeric': effect_json}})
                original_digest = scene_context_digest(scene)
                relocated_array = origin/'effect-relocated.npz'; shutil.copyfile(effect_array, relocated_array)
                relocated_json = write(origin/'effect-relocated.json', {'owner_run_id': 'relocated',
                    'emission': record(relocated_array), 'schema': 'numeric-effect-fixture.v1'})
                scene['layers'][-1]['assets']['frozen_numeric'] = relocated_json
                require(effect_json['sha256'] != relocated_json['sha256'])
                require(scene_context_digest(scene) == original_digest)
                scene['layers'][-1]['strength'] = .4
                require(scene_context_digest(scene) != original_digest)
                scene['layers'][-1]['strength'] = .3
                np.savez_compressed(relocated_array, values=delta*.8)
                changed_json = write(origin/'effect-changed.json', {'owner_run_id': 'relocated',
                    'emission': record(relocated_array), 'schema': 'numeric-effect-fixture.v1'})
                scene['layers'][-1]['assets']['frozen_numeric'] = changed_json
                require(scene_context_digest(scene) != original_digest)
                disabled_scene = copy.deepcopy(plan)
                disabled_scene['layers'].append({'kind': 'bloom', 'enabled': False, 'strength': 1.,
                    'params': {'radius_fraction': .015}, 'assets': {}})
                require(scene_context_digest(disabled_scene) == scene_context_digest(plan))
                for mutation in ('dose', 'alpha', 'denoise', 'manual', 'appearance', 'identity', 'detail',
                        'background_strength', 'background_pixels', 'background_validity', 'background_disabled', 'new_effect'):
                    if mutation == 'detail' and not with_detail: continue
                    changed = copy.deepcopy(run); p = design.plan_for_run(changed)
                    if mutation == 'dose': p['layers'][1]['strength'] = .5
                    elif mutation == 'alpha':
                        path = origin/'edited-alpha.npz'; np.savez_compressed(path, values=alpha*.9)
                        p['mask_roles']['foreground_alpha'] = {**p['mask_roles']['foreground_alpha'], **record(path)}
                    elif mutation == 'denoise': changed['options']['denoise_strength'] = .2
                    elif mutation == 'manual': changed['options']['manual_layers'] = {'tone_curves': {'enabled': True, 'scope': 'subject'}}
                    elif mutation == 'appearance': changed['options']['appearance_revision']['params'] = {'subject': {'exposure': .1}}
                    elif mutation == 'identity':
                        path = origin/'changed-protection.npz'; np.savez_compressed(path, protection=np.zeros_like(alpha))
                        p['mask_roles']['identity_protection'] = {**p['mask_roles']['identity_protection'], **record(path)}
                    elif mutation == 'detail': p['layers'][-2]['strength'] = .5
                    elif mutation == 'background_strength': p['layers'][0]['strength'] = .5
                    elif mutation == 'background_disabled': p['layers'][0]['enabled'] = False
                    elif mutation in ('background_pixels', 'background_validity'):
                        role = 'background' if mutation == 'background_pixels' else 'validity'
                        path = origin/(mutation+'.npz')
                        data = background*.8 if role == 'background' else np.ones_like(alpha)*.9
                        np.savez_compressed(path, values=data)
                        p['layers'][0]['assets'][role] = {**p['layers'][0]['assets'][role], **record(path)}
                    elif mutation == 'new_effect':
                        p['layers'].append({'id': 'added-bloom', 'kind': 'bloom', 'enabled': True, 'strength': .2,
                            'params': {'source': 'visible_composite', 'threshold': .8, 'radius_fraction': .015}, 'assets': {}})
                    response_layer = next(row for row in p['layers'] if row['kind'] == KIND)
                    unchanged, state = apply_environment(target, response_layer, p, changed)
                    require(np.array_equal(unchanged, target) and state['bypass_reason'] is not None)
                for mutation in ('target', 'count', 'source', 'reference', 'owner', 'protected', 'nonopaque', 'qualification', 'holdout',
                        'unknown_support', 'reference_texture', 'current_foreground', 'negative_rgb', 'overflow_rgb', 'nonfinite', 'float64', 'overlap'):
                    changed = copy.deepcopy(run); p = design.plan_for_run(changed); bad_layer = p['layers'][-1]
                    frozen = json.loads(Path(layer['assets']['frozen_delta']['path']).read_text('utf8'))
                    qa = json.loads(Path(layer['assets']['qualification_receipt']['path']).read_text('utf8'))
                    if mutation == 'target': frozen['patches'][0]['target_patch_sha256'] = '0'*64
                    elif mutation == 'count': frozen['patches'][0]['changed_pixels'] += 1
                    elif mutation in ('source', 'reference'): changed[mutation]['sha256'] = '0'*64
                    elif mutation == 'owner': frozen['patches'][0]['owner_class_id'] = 6
                    elif mutation == 'protected':
                        protect = np.zeros_like(alpha); protect[8, 16] = 1
                        path = origin/'bad-protection.npz'; np.savez_compressed(path, protection=protect)
                        p['mask_roles']['identity_protection'] = {**p['mask_roles']['identity_protection'], **record(path)}
                        frozen['binding']['identity_protection_sha256'] = p['mask_roles']['identity_protection']['sha256']
                    elif mutation == 'nonopaque':
                        path = origin/'bad-source-alpha.npz'; np.savez_compressed(path, values=alpha*.9)
                        asset = {**p['source_baseline']['assets']['foreground_alpha'], **record(path)}
                        p['source_baseline']['assets']['foreground_alpha'] = asset; frozen['donor_assets']['foreground_alpha'] = asset
                        frozen['binding']['source_alpha_sha256'] = asset['sha256']
                    elif mutation == 'qualification': qa['qualification']['actual_saved_evidence_verified'] = False
                    elif mutation == 'holdout': qa['qualification']['holdout_improved'] = False
                    elif mutation == 'unknown_support': qa['checks']['unknown_support_changed'] = True
                    elif mutation == 'reference_texture': qa['checks']['reference_texture_imported'] = True
                    elif mutation == 'current_foreground':
                        path = origin/'changed-current-foreground.npz'; np.savez_compressed(path, values=target*.8)
                        p['layers'][0]['assets']['foreground_premultiplied'] = {
                            **p['layers'][0]['assets']['foreground_premultiplied'], **record(path)}
                    elif mutation == 'overlap': frozen['patches'][1]['native_box'] = frozen['patches'][0]['native_box']
                    elif mutation in ('nonfinite', 'float64', 'negative_rgb', 'overflow_rgb'):
                        with np.load(frozen['delta']['path'], allow_pickle=False) as archive:
                            arrays = {key: archive[key].copy() for key in archive.files}
                        key = frozen['patches'][0]['array_key']
                        if mutation == 'nonfinite': arrays[key][0, 0, 0] = np.nan
                        elif mutation == 'float64': arrays[key] = arrays[key].astype(np.float64)
                        elif mutation == 'negative_rgb': arrays[key][0, 0] = -2
                        elif mutation == 'overflow_rgb': arrays[key][0, 0] = 2
                        frozen['patches'][0]['delta_array_sha256'] = array_sha(arrays[key])
                        path = origin/(mutation+'-delta.npz'); np.savez_compressed(path, **arrays); frozen['delta'] = record(path)
                    bad_layer['params']['binding'] = copy.deepcopy(frozen['binding'])
                    bad_layer['assets']['frozen_delta'] = write(origin/('bad-'+mutation+'.json'), frozen)
                    bad_layer['assets']['qualification_receipt'] = write(origin/('bad-'+mutation+'-qa.json'), qa)
                    try: apply_environment(target, bad_layer, p, changed)
                    except ValueError: checks += 1
                    else: raise AssertionError('Unqualified/corrupt environment response accepted: '+mutation)
                if not with_detail:
                    from raw_support import RGB_DECODE_VERSION
                    # The real source-revision path makes an implicit absent
                    # retouch explicit, despite changing no approved pixels.
                    empty_delta = origin/'empty-native-retouch.npz'; np.savez_compressed(empty_delta)
                    retouch_layer = {'schema_version': 'native_retouch_delta.v1',
                        'recipe': 'source-native-compact-outlier-two-crop-approval-v1', 'enabled': False,
                        'source': copy.deepcopy(run['source']), 'native_size': [48, 32],
                        'delta': record(empty_delta), 'patches': [], 'approvals': [],
                        'clone_retouch': {'schema_version': 'clone_retouch.v1', 'enabled': True, 'strokes': []},
                        'source_rgb_donor_only': True, 'reference_rgb_used': False, 'generation_used': False}
                    retouch_layer['source']['decode'] = {'source_precision': 'raster8_srgb',
                        'working_precision': 'float32_linear', 'decode_version': RGB_DECODE_VERSION}
                    retouch_binding = {'schema': 'automatic_native_retouch_binding.v1',
                        'source_sha256': run['source']['sha256'], 'native_size': [48, 32], 'working_domain': DOMAIN,
                        'owner_run_id': 'explicit-no-op-child', 'approved_patches': 0,
                        'layer': write(origin/'empty-native-retouch.json', retouch_layer)}
                    plan['source_baseline']['native_source_retouch'] = copy.deepcopy(retouch_binding)
                    no_op_values, _ = apply_environment(target, layer, plan, run)
                    require(np.array_equal(no_op_values, target+delta))
                    zero_array_run = copy.deepcopy(run); zero_array_plan = design.plan_for_run(zero_array_run)
                    zero_array_layer = copy.deepcopy(retouch_layer)
                    zero_array_path = origin/'all-zero-native-retouch.npz'
                    np.savez_compressed(zero_array_path, delta_0=np.zeros((32, 48, 3), np.float32))
                    zero_array_layer['delta'] = record(zero_array_path)
                    zero_array_plan['source_baseline']['native_source_retouch']['layer'] = write(
                        origin/'all-zero-native-retouch.json', zero_array_layer)
                    zero_values, _ = apply_environment(target, zero_array_plan['layers'][-1], zero_array_plan, zero_array_run)
                    require(np.array_equal(zero_values, target+delta))
                    raw_layer = copy.deepcopy(retouch_layer)
                    raw_layer['source']['decode']['source_precision'] = 'raw16_linear'
                    raw_binding = copy.deepcopy(retouch_binding); raw_binding['working_domain'] = 'scene_linear_sRGB'
                    raw_binding['layer'] = write(origin/'raw-no-op-retouch.json', raw_layer)
                    require(_approved_retouch_identity(raw_binding, source_sha256=run['source']['sha256'],
                        native_size=[48, 32]) is None)
                    for mutation in ('nonzero_delta', 'nonfinite_delta', 'enabled', 'patch', 'approval', 'clone',
                            'approved_count', 'source', 'native', 'domain', 'mismatched_valid_domain',
                            'mismatched_layer_domain', 'unknown_precision', 'wrong_decoder', 'reference_donor', 'generation_donor'):
                        changed = copy.deepcopy(run); p = design.plan_for_run(changed)
                        binding = p['source_baseline']['native_source_retouch']; data = copy.deepcopy(retouch_layer)
                        if mutation in ('nonzero_delta', 'nonfinite_delta'):
                            values = np.zeros((2, 2, 3), np.float32)
                            values[0, 0, 0] = .001 if mutation == 'nonzero_delta' else np.nan
                            path = origin/('no-op-'+mutation+'.npz'); np.savez_compressed(path, delta_0=values)
                            data['delta'] = record(path)
                        elif mutation == 'enabled': data['enabled'] = True
                        elif mutation == 'patch': data['patches'] = [{'native_box': [0, 0, 2, 2]}]
                        elif mutation == 'approval': data['approvals'] = [{'candidate_id': 'unfrozen-new-approval'}]
                        elif mutation == 'clone': data['clone_retouch']['strokes'] = [{'source': [0, 0], 'target': [1, 1]}]
                        elif mutation == 'approved_count': binding['approved_patches'] = 1
                        elif mutation == 'source': binding['source_sha256'] = '0'*64
                        elif mutation == 'native': binding['native_size'] = [47, 32]
                        elif mutation == 'domain': binding['working_domain'] = 'scene_linear'
                        elif mutation == 'mismatched_valid_domain': binding['working_domain'] = 'scene_linear_sRGB'
                        elif mutation == 'mismatched_layer_domain': data['working_domain'] = 'scene_linear_sRGB'
                        elif mutation == 'unknown_precision': data['source']['decode']['source_precision'] = 'unestablished_linear'
                        elif mutation == 'wrong_decoder': data['source']['decode']['decode_version'] = 'unknown_decoder'
                        elif mutation == 'reference_donor': data['reference_rgb_used'] = True
                        elif mutation == 'generation_donor': data['generation_used'] = True
                        binding['layer'] = write(origin/('no-op-'+mutation+'.json'), data)
                        try: apply_environment(target, p['layers'][-1], p, changed)
                        except ValueError: checks += 1
                        else: raise AssertionError('Nonempty or unqualified retouch treated as absence: '+mutation)
                # Materialized and portable project bind immutable pixels across
                # asset path and ownership changes; original files are removed.
                saved = base/('saved-detail' if with_detail else 'saved'); owned_run = copy.deepcopy(run)
                owned_plan = design.materialize_plan(saved, owned_run, plan)
                owned_run['root'] = str(saved); owned_run['options']['appearance_revision']['design_reconstruction_plan'] = owned_plan
                owned_records = closure(owned_plan); require(owned_records > 12)
                write(saved/'outputs/runs/fixture/manifest.json', owned_run)
                archive = base/('detail.cosplay' if with_detail else 'plain.cosplay'); export_project(saved, 'fixture', archive)
                moved = import_project(base/('portable-detail' if with_detail else 'portable'), archive, run_id='relocated')['run']
                moved_plan = design.plan_for_run(moved); require(closure(moved_plan) >= owned_records)
                require(origin.resolve().is_relative_to(base.resolve()) and origin.name.startswith('origin'))
                shutil.rmtree(origin)
                for relocated in (owned_run, moved):
                    p = design.plan_for_run(relocated)
                    replay, state = apply_environment(target, p['layers'][-1], p, relocated)
                    require(np.array_equal(replay, target+delta) and state['applied'])
    return {'ok': True, 'checks': checks, 'appearance_validator_mocked': True,
        'full_roi_preview_shared_consumer': True, 'after_appearance_and_optional_detail_before_manual': True,
        'controls_changed_require_requalification': True, 'source_and_current_opaque_owner_verified': True,
        'identity_skin_unknown_background_excluded': True, 'source_retouch_reapplied': False,
        'native_support_before_preview_reduction': True, 'portable_after_original_root_removed': True,
        'scene_background_dose_pixels_and_effect_context_require_requalification': True,
        'scene_numeric_context_stable_under_json_locator_and_owner_rewrites': True,
        'absent_retouch_equals_only_verified_disabled_pixel_empty_noop': True,
        'nonzero_delta_new_approval_clone_or_source_binding_changes_rejected': True,
        'raw_scene_and_raster_display_retouch_domains_established_by_actual_frozen_source_decode': True,
        'illegal_domain_or_domain_decode_mismatch_rejected': True,
        'scope': 'Runtime replay validation; this fixture does not establish fitted field quality or real saved-image evidence.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path); args = parser.parse_args()
    report = run_smoke()
    if args.output: write(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
