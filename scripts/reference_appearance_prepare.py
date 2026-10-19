"""Freeze native T/B/S foreground contributions for the editable workbench.

This model-runtime producer owns analysis; the desktop consumer needs only
source-coordinate appearance Fp/B_old, alpha and a native premultiplied delta.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from runtime_resources import code_path, ordered_native_map
from asset_identity import sha256_file
import time

import numpy as np
from native_array_storage import save_native_array
import workflow_timing


from reference_recipe import RECIPE
STAGES = ['reference_colour_match', 'reference_colour_blend', 'reference_scene_balance', 'edge_donor_extension']


def sha(path):
    return sha256_file(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _materialize_reused_array_asset(path, reused, *, verified=None):
    """Bind the actual materialized bytes before publishing a reused asset."""
    import os
    import shutil
    from native_source_baseline import _native_asset_file_identity
    path, source_path = Path(path), Path(reused['path'])
    if verified is not None:
        verified.validate_asset(reused)
    source_identity = _native_asset_file_identity(source_path)
    linked = False
    if source_path.resolve() != path.resolve():
        if path.exists():
            path.unlink()
        try:
            os.link(source_path, path)
            linked = True
        except OSError:
            shutil.copy2(source_path, path)
    if _native_asset_file_identity(source_path)[:4] != source_identity[:4]:
        raise ValueError('Reused native source changed during materialization')
    target_identity = _native_asset_file_identity(path)
    digest = sha256_file(path, force=True)
    actual_target = _native_asset_file_identity(path)
    actual_source = _native_asset_file_identity(source_path)
    # Creating a hardlink can change POSIX inode ctime without changing its
    # payload. Only that same current inode, now fully SHA-verified through
    # the destination, may retain its proof across the link-count update.
    same_link = linked and actual_target[:2] == actual_source[:2]
    if (actual_source[:4] != source_identity[:4]
            or not same_link and actual_source != source_identity
            or actual_target != target_identity or digest != reused['sha256']):
        raise ValueError('Reused native asset bytes changed during materialization')
    if verified is not None and not same_link:
        verified.validate_asset(reused)
    return {**copy.deepcopy(reused), 'path': str(path)}


def provenance_only(value):
    """Keep content identities and numeric choices, never live cache paths."""
    if isinstance(value, dict):
        return {key: provenance_only(item) for key, item in value.items()
                if key not in {'path', 'manifest', 'output'} and not key.endswith(('_path', '_file'))}
    if isinstance(value, list):
        return [provenance_only(item) for item in value]
    return value


def _matching_donor_source(source, run, *, binding, precision, transform):
    """Reuse the bound straight Beauty endpoint without changing the source pair.

    The native baseline already publishes this endpoint for fitting. Probe the
    validated cache first so direct invocations and incomplete caches retain the
    existing CPU replay on ``source`` rather than decoding the original again.
    """
    from native_source_baseline import _load_reference_source_cache, source_for_reference_prepare
    from portrait_beauty_layer import apply_for_run, identity_for_run

    # ``precision`` belongs to the intentionally immutable pre-Beauty source
    # consumed by F/B.  The donor endpoint has its own path-independent
    # identity; never relabel the pre-Beauty array as the Beauty variant.
    beauty = precision.get('portrait_beauty') or identity_for_run(run)
    cached = (_load_reference_source_cache(run, root=None, transform=transform,
        beauty_identity=beauty, writable=False) if beauty is not None else None)
    if cached is not None and cached[2] == 'beauty':
        donor_source, donor_precision, _ = source_for_reference_prepare(
            run, binding=binding, apply_beauty=True, writable=False,
            _validated_cache=cached)
        del cached
        consumed = donor_precision.get('portrait_beauty') or {}
        return donor_source, {'enabled': True,
            'applied': bool(consumed.get('strength') and
                len(consumed.get('disabled_faces', ())) < len(consumed.get('faces', ()))),
            'source_endpoint_cache_hit': True, 'beauty_replay_performed': False}
    del cached
    donor_source, execution = apply_for_run(source, run)
    return donor_source, {**execution, 'source_endpoint_cache_hit': False,
        'beauty_replay_performed': bool(execution.get('applied'))}


def preserve_contact_layers(plan, run, *, changed_receiver_box=None):
    """Keep only qualified current contact, still owned by this native matte.

    Contact is dimensionless background attenuation. Its Q remains separate
    from A, and alpha belongs only to the consumer's final foreground over.
    No missing contact or historical FX is inferred or recreated here.
    """
    from design_scene_assets import CONTACT_SCHEMA, validate_contact_receiver_contract

    native = list(plan['canvas']['source_oriented_size'])
    proxy = list(plan['canvas']['source_proxy_size'])
    alpha = plan['mask_roles']['foreground_alpha']
    contacts, reports = [], []
    for layer in plan.get('layers', []):
        if layer.get('kind') != 'contact_shadow':
            continue
        report = {'layer_id': layer.get('id'), 'preserved': False}
        reports.append(report)
        if (layer.get('provenance') or {}).get('contract_schema') != CONTACT_SCHEMA:
            raise ValueError('Current contact receiver contract required')
        validation = validate_contact_receiver_contract(layer,
            source_sha256=run['source']['sha256'], candidate_sha256=run['reference']['sha256'],
            source_alpha_sha256=alpha['sha256'])
        qualification = layer['params']['receiver_qualification']
        if not validation['qualified_for_automatic_contact']:
            report['reason'] = 'receiver_not_qualified'
            continue
        if (qualification.get('native_canvas_size') != native or
                plan['canvas'].get('candidate_mapping', {}).get('scene_registration_supported') is not True):
            raise ValueError('Contact native canvas or current receiver mapping changed')
        if alpha.get('size') != native or sha(alpha['path']) != alpha['sha256']:
            raise ValueError('Contact requires the unchanged full native source alpha')
        for key, value in (('strength', layer.get('strength', 1.)), ('opacity', layer['params'].get('opacity'))):
            if value is None or not np.isfinite(value) or not 0 <= float(value) <= 1:
                raise ValueError('Contact control outside finite unit interval: ' + key)
        assets = layer['assets']
        native_box = qualification['native_box']
        for key, asset in assets.items():
            path = Path(asset['path'])
            if sha(path) != asset['sha256']:
                raise ValueError('Contact asset bytes changed: ' + key)
            with np.load(path, allow_pickle=False) as resource:
                values = resource['labels' if key == 'source_shoe_semantics' else 'values']
            if values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError('Contact asset must be a finite scalar grid: ' + key)
            actual = [values.shape[1], values.shape[0]]
            if asset.get('size', actual) != actual:
                raise ValueError('Contact asset grid metadata changed: ' + key)
            space = asset['coordinate_space']
            if space == 'source_canvas_roi':
                box = asset['native_box']
                if (box != native_box or asset.get('original_canvas_size') != native or
                        not 0 <= box[0] < box[2] <= native[0] or
                        not 0 <= box[1] < box[3] <= native[1]):
                    raise ValueError('Contact ROI is outside or differs from the source canvas: ' + key)
            elif space == 'source_canvas' and actual != native:
                raise ValueError('Contact alpha must use the native grid')
            elif space in {'source_proxy', 'final_source_proxy'}:
                if actual != proxy or asset.get('original_canvas_size', native) != native:
                    raise ValueError('Contact observation canvas changed: ' + key)
            if key == 'source_component_owner':
                legal = np.isin(values, [0, 9, 18]).all()
            elif key == 'source_shoe_semantics':
                legal = bool((values >= 0).all() and (values <= 28).all() and (values == np.floor(values)).all())
            else:
                legal = bool((values >= 0).all() and (values <= 1).all())
            if not legal:
                raise ValueError('Contact mask or categorical values invalid: ' + key)
        if changed_receiver_box is not None:
            l, t, r, b = native_box
            cl, ct, cr, cb = changed_receiver_box
            if max(l, cl) < min(r, cr) and max(t, ct) < min(b, cb):
                report['reason'] = 'repaired_receiver_overlaps_contact_requires_new_qualification'
                continue
        contacts.append(copy.deepcopy(layer))
        report.update(preserved=True, reason='qualified_native_contact_preserved',
            currently_enabled=layer.get('enabled') is True, asset_bytes_verified=True,
            validation=validation, execution_stage='background_receiver_attenuation_before_single_composition',
            receiver_support_consumed_once=True, foreground_coverage_consumption='final_over_only')
    return contacts, {'layers': reports, 'preserved_count': len(contacts),
        'new_contact_inferred': False, 'complete_cast_shadow': False, 'visual_acceptance': False}


from native_cuda_sampling import request_sampling_lease


@request_sampling_lease(('colour', 'colour_owners'), ('scene', 'scene_owners'))
def _freeze_matching_native(working_foreground, automatic_delta, alpha, edge_support,
                            parameters, colour, scene, colour_owners, scene_owners,
                            hair_guard, proxy_size, native, plain_proxy, plain_channels,
                            *, has_chroma, pre_scene_reader, workers=2):
    """Match original 128-row bands; commit deltas and QA in original order."""
    from render_reference_colour_composite import (sample_reference_field, sample_pre_scene_reference_field,
        sample_plain_stack, match_foreground, matching_tiles, _sample_grid, _sample_plan)
    from portrait_beauty_layer import sample_visible_face_support

    def compute(tile):
        top, end, box, region = tile
        left, first = box[:2]
        grid = _sample_grid(box, proxy_size, native)
        sample_plan = _sample_plan(grid, colour.shape[:2])
        sampled_plain = sample_plain_stack(plain_proxy, plain_channels,
                                           box, proxy_size, native, grid=grid)
        sampled_weights, sampled_donors = sampled_plain[:2]
        sampled_chroma = sampled_plain[2] if has_chroma else None
        working = working_foreground[region]
        visible_face_support = (sample_visible_face_support(hair_guard, box, native)
                                if hair_guard is not None else None)
        pre_scene, stats = (pre_scene_reader._read_band(box, working, alpha[region])
                           if pre_scene_reader is not None else (None, None))
        gamut = {}
        matched = match_foreground(working, alpha[region], sampled_weights,
            sampled_donors, edge_support[region], parameters,
            origin=(left,first), canvas_size=native, reference_blend=sample_pre_scene_reference_field(colour,box,proxy_size,native,
                owners=colour_owners,guard=hair_guard,grid=grid,
                visible_face_support=visible_face_support, sample_plan=sample_plan,
                pre_scene_valid=pre_scene[1] if pre_scene is not None else None),
            scene_balance=sample_reference_field(scene,box,proxy_size,native,
                owners=scene_owners,guard=hair_guard,grid=grid,
                visible_face_support=visible_face_support, sample_plan=sample_plan),
            chroma_weights=sampled_chroma,
            pre_scene_target=pre_scene[0] if pre_scene is not None else None,
            pre_scene_valid=pre_scene[1] if pre_scene is not None else None,
            donor_gamut_report=gamut, _trusted=True)
        zero_alpha = bool(np.all(matched[alpha[region] == 0] == 0))
        nonnegative = bool(np.isfinite(matched).all() and matched.min() >= 0)
        return top, end, region, matched-working, zero_alpha, nonnegative, gamut, stats

    zero_alpha_exact, nonnegative_target, edge_gamut = True, True, {}
    bands = None
    try:
        bands = ordered_native_map(compute, matching_tiles(alpha), workers=workers)
        for top, end, region, delta, zero_alpha, nonnegative, gamut, stats in bands:
            if pre_scene_reader is not None:
                pre_scene_reader._merge_stats(stats)
            zero_alpha_exact &= zero_alpha
            nonnegative_target &= nonnegative
            # Each worker uses a local gamut report. Fold it in the original
            # top order with the same count/min/max operations as the producer.
            edge_gamut['policy'] = gamut['policy']
            for key in ('requested_outside_sdr_pixels', 'source_outside_premultiplied_sdr_pixels'):
                edge_gamut[key] = edge_gamut.get(key, 0)+gamut[key]
            for key in ('maximum_requested_excursion', 'maximum_source_excursion',
                        'maximum_operator_increased_excursion'):
                edge_gamut[key] = max(edge_gamut.get(key, 0), gamut[key])
            key = 'minimum_applied_dose'
            edge_gamut[key] = min(edge_gamut.get(key, 1), gamut[key])
            automatic_delta[top:end] = np.float32(0)-working_foreground[top:end]
            automatic_delta[region] = delta
        reuse = dict(pre_scene_reader.stats) if pre_scene_reader is not None else {'available': False}
        return zero_alpha_exact, nonnegative_target, edge_gamut, reuse
    finally:
        # Join bounded tasks before closing their shared read-only mappings,
        # including a worker exception or a failed ordered output write.
        try:
            if bands is not None:
                bands.close()
        finally:
            if pre_scene_reader is not None:
                pre_scene_reader.close()


@workflow_timing.entry('reference_freeze', lambda run, match_directory, output: output)
def freeze(run, match_directory, output):
    import cv2
    from display_transform import display_linear, plan_for_run
    from raw_support import open_linear
    from render_reference_colour_composite import (sample_reference_field, donor_fields,
        field_owners,
        validate_material_inputs)
    from reference_support import donor_support_report
    from reference_colour_blend import validate_layer
    from reference_scene_balance import validate_scene_layer

    started = time.perf_counter()
    directory, output = Path(match_directory).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with workflow_timing.span('load', model_request=False, http_request=False):
        old_plan = run['options']['appearance_revision']['design_reconstruction_plan']
        parameters = read(directory/'parameters.json')
        validate_material_inputs(parameters)
        if parameters.get('automatic_material_palette'):
            from reference_material_palette import validate_palette
            validate_palette(parameters['automatic_material_palette'])
        receipt = read(directory/'composite-receipt.json')
        if receipt.get('scene_balance'):
            raise ValueError('Freeze requires the immutable pre-scene native baseline')
        native = tuple(old_plan['canvas']['source_oriented_size'])
        proxy_size = tuple(old_plan['canvas']['source_proxy_size'])
        for role in ('source', 'reference'):
            if sha(run[role]['path']) != parameters['binding'][role] or run[role]['sha256'] != parameters['binding'][role]:
                raise ValueError('Prepared colour target belongs to different photographs')
        if receipt['parameters_sha256'] != sha(directory/'parameters.json'):
            raise ValueError('Prepared native baseline parameters changed')
        if receipt['renderer_sha256'] != sha(code_path('render_reference_colour_composite.py')):
            raise ValueError('Prepared native baseline renderer changed')
        if parameters['binding']['algorithm_sha256'] != sha(code_path('reference_colour_fit.py')):
            raise ValueError('Reprepare colour fitting after algorithm changes')
        with np.load(directory/'source-weights.npz', allow_pickle=False) as resource:
            weights = resource['values']
            chroma_weights = resource['chroma_values'] if 'chroma_values' in resource else None
        if sha(directory/'source-weights.npz') != parameters['binding']['source_weights_sha256']:
            raise ValueError('Prepared source weights changed')
        colour, colour_receipt = validate_layer(directory/'reference-colour-layer.json', parameters, native, proxy_size)
        scene, scene_receipt, donor_baseline = validate_scene_layer(directory/'scene-balance-layer.json', parameters,
            directory/'reference-colour-layer.json', old_plan, native, proxy_size)
        colour_owners = field_owners(colour_receipt, colour.shape[:2])
        scene_owners = field_owners(scene_receipt, scene.shape[:2])
        from portrait_beauty_layer import hair_correction_guard_for_run, sample_visible_face_support
        hair_guard = hair_correction_guard_for_run(run)
        guard_binding = hair_guard['binding'] if hair_guard else None
        if any(layer.get('hair_correction_guard') != guard_binding for layer in (colour_receipt,scene_receipt)):
            raise ValueError('Frozen appearance fields have different source visible-face guards')
        alpha_asset = old_plan['mask_roles']['foreground_alpha']
        if sha(alpha_asset['path']) != parameters['binding']['alpha']:
            raise ValueError('The matte changed; prepare a new appearance foreground pair')
        with np.load(alpha_asset['path'], allow_pickle=False) as resource:
            alpha = resource['values']
    print('Preparing native appearance foreground and original environment', flush=True)
    with workflow_timing.span('source_decode', model_request=False, http_request=False):
        from native_source_baseline import (source_for_reference_prepare, retouch_baseline_record,
            display_pair_identity, prepare_display_pair_for_freeze, publish_display_pair_assets,
            VerifiedNativeArrayAsset)
        source, decode_precision, source_display = source_for_reference_prepare(
            run, binding=parameters['binding'], apply_beauty=False, writable=False)
    from portrait_beauty_layer import apply_for_run
    with workflow_timing.span('matching_donors', model_request=False, http_request=False):
        donor_source, beauty_execution = _matching_donor_source(source, run,
            binding=parameters['binding'], precision=decode_precision, transform=source_display)
        proxy_box=(0,0,proxy_size[0],proxy_size[1])
        donor_visible_face_support = (sample_visible_face_support(hair_guard, proxy_box, proxy_size)
                                      if hair_guard is not None else None)
        donor_scene=sample_reference_field(scene,proxy_box,proxy_size,proxy_size,
            owners=scene_owners,guard=hair_guard,
            visible_face_support=donor_visible_face_support)
        donors = donor_fields(cv2.resize(donor_source, proxy_size, interpolation=cv2.INTER_LINEAR),
            cv2.resize(alpha, proxy_size, interpolation=cv2.INTER_LINEAR), weights, parameters,
            None, donor_scene, donor_baseline, chroma_weights)
        del donor_source, donor_scene
        donor_support = donor_support_report(donors, weights, parameters)
    with workflow_timing.span('foreground_background', model_request=False, http_request=False):
        pair_identity = display_pair_identity(run, precision=decode_precision, transform=source_display,
            alpha_sha256=alpha_asset['sha256'], native_size=native)
        old_background, foreground, edge_support, tiles, reconstruction_error, reused_pair = prepare_display_pair_for_freeze(
            source, alpha, run, pair_identity)
        del source

    def array_asset(name, values, representation, *, reused=None):
        path = output/(name+'.npz')
        if isinstance(values, VerifiedNativeArrayAsset):
            if reused is None:
                raise ValueError('Verified native asset needs its original reuse binding')
        if reused is not None:
            return _materialize_reused_array_asset(path, reused,
                verified=values if isinstance(values, VerifiedNativeArrayAsset) else None)
        save_native_array(path, values)
        return {'path': str(path), 'sha256': sha(path), 'size': list(native),
                'coordinate_space': 'source_canvas', 'working_space': 'display_linear_sRGB',
                'representation': representation, 'sampling': 'bilinear'}

    print('Saving immutable source appearance pair', flush=True)
    with workflow_timing.span('source_assets_encoding', model_request=False, http_request=False):
        reused_assets = reused_pair['assets'] if reused_pair is not None else {}
        fp_asset = array_asset('appearance-source-foreground', foreground, 'premultiplied_RGB',
            reused=reused_assets.get('foreground_premultiplied'))
        old_bg_asset = array_asset('appearance-original-environment', old_background, 'straight_RGB',
            reused=reused_assets.get('old_environment'))
        del old_background
        edge_asset = array_asset('appearance-edge-support', edge_support, 'edge_confidence',
            reused=reused_assets.get('edge_applicability'))
        appearance_pair = publish_display_pair_assets(pair_identity,
            {'foreground_premultiplied': fp_asset, 'old_environment': old_bg_asset,
             'edge_applicability': edge_asset}, tiles=tiles, reconstruction_error=reconstruction_error)
    print('Freezing full T -> colour fusion -> scene balance -> final edge donors', flush=True)
    # Beauty replay is a frozen native layer.  Applying it once to the full
    # foreground avoids validating/loading the same generated face assets for
    # every 128-row matching tile.  Tile matching below still owns the output
    # writes and uses the exact same native canvas and premultiplied alpha,
    # preserving the prior pixel contract while removing repeated Python/PIL
    # interpolation work.
    with workflow_timing.span('matching_native', model_request=False, http_request=False):
        working_foreground, _beauty_execution = apply_for_run(
            foreground, run, canvas_size=native, premultiplied_alpha=alpha)
        plain_proxy = np.concatenate([weights, donors] +
                                     ([chroma_weights] if chroma_weights is not None else []), axis=-1)
        plain_channels = (weights.shape[-1], donors.shape[-1]) + ((chroma_weights.shape[-1],)
                                                                    if chroma_weights is not None else ())
        # Keep the actual Beauty working endpoint alive for control preparation.
        # With Beauty disabled or zero strength it aliases the original Fp; a new
        # delta buffer then avoids overwriting that endpoint. Active Beauty already
        # owns separate storage, so reuse the original foreground for the delta.
        automatic_delta = (np.empty_like(foreground) if np.shares_memory(
            working_foreground, foreground) else foreground)
        from reference_pre_scene_cache import open_reader
        pre_scene_reader = open_reader(parameters, colour_receipt, native, proxy_size)
        zero_alpha_exact, nonnegative_target, edge_gamut, pre_scene_reuse = _freeze_matching_native(
            working_foreground, automatic_delta, alpha, edge_support, parameters,
            colour, scene, colour_owners, scene_owners, hair_guard, proxy_size, native,
            plain_proxy, plain_channels, has_chroma=chroma_weights is not None,
            pre_scene_reader=pre_scene_reader)
    with workflow_timing.span('delta_encoding', model_request=False, http_request=False):
        delta_asset = array_asset('reference-appearance-delta', automatic_delta, 'premultiplied_delta')
        del foreground, edge_support, donors

    bg_layer = copy.deepcopy(next(x for x in old_plan['layers'] if x['kind'] == 'background_design'))
    bg_layer.update(enabled=True, strength=1., label='参考背景')
    bg_layer['assets'] = {key: copy.deepcopy(bg_layer['assets'][key]) for key in
        ('background','validity','native_receiver_preservation','compact_observation','compact_validity',
         'background_observation','background_consistency','native_receiver_detail') if key in bg_layer['assets']}
    bg_layer['assets'].update(foreground_premultiplied=fp_asset, old_environment=old_bg_asset)
    contact_layers, contact_receipt = preserve_contact_layers(old_plan, run)
    from native_receiver_preservation import prepare_native_receiver_preservation
    receiver_plan = copy.deepcopy(old_plan)
    receiver_plan['layers'] = [bg_layer, *contact_layers]
    receiver_plan['source_baseline'] = {'schema': 'appearance_source_baseline.v1',
        'source_sha256': run['source']['sha256'], 'working_domain': 'display_linear_sRGB',
        'source_display': source_display, 'source_display_applied_once': True,
        'source_precision': decode_precision['source_precision'], 'decode_version': decode_precision['decode_version'],
        'assets': {'foreground_premultiplied': fp_asset, 'foreground_alpha': alpha_asset, 'old_environment': old_bg_asset}}
    with workflow_timing.span('receiver_preservation', model_request=False, http_request=False):
        receiver_result = prepare_native_receiver_preservation(run, receiver_plan, output/'receiver-preservation')
        bg_layer = next(row for row in receiver_result['design_reconstruction_plan']['layers'] if row['kind'] == 'background_design')
        receiver_receipt = receiver_result['receiver_preservation']
    snapshot = {'schema':'reference_preparation_snapshot.v1', 'recipe':RECIPE,
        'parameters':provenance_only(parameters), 'colour_layer':provenance_only(colour_receipt),
        'scene_layer':provenance_only(scene_receipt), 'producer_code':{
            name:sha(code_path(name)) for name in ('reference_appearance_prepare.py',
            'reference_colour_fit.py','reference_colour_blend.py','reference_scene_balance.py','render_reference_colour_composite.py',
            'reference_local_correspondence.py','reference_local_correspondence_worker.py',
            'reference_adaptive_fusion.py','reference_automatic_layers.py','source_skin_domains.py',
            'reference_recipe.py','reference_material_palette.py','native_cuda_fields.py','native_cuda_sampling.py',
            'native_cuda_matting.py','design_scene_assets.py','native_sole_contact.py','native_receiver_preservation.py')},
        'source_display':source_display, 'prepared_stages':STAGES, 'donor_support':donor_support,
        'fit_beauty_identity': parameters['binding'].get('portrait_beauty'),
        'photo_specific_inputs':{'authored_colour_edit':bool(parameters['binding'].get('authored_colour_edit')),
            'accessory_masks':bool(parameters['binding'].get('object_masks'))},
        'checks':{'source_reconstruction_max_error':reconstruction_error,'zero_alpha_exact':zero_alpha_exact,
            'pre_scene_reuse': pre_scene_reuse,
            'immutable_display_source_pair_reused': reused_pair is not None,
            'finite_nonnegative_target':nonnegative_target,'composition_count':1,'reference_person_geometry_or_texture_imported':False,
            'edge_gamut':edge_gamut, 'contact_preservation':contact_receipt,
            'immutable_source_pair_contains_beauty': False,
            'working_beauty_applied_before_matching': beauty_execution.get('applied', False),
            'matching_donor_source': {'source_endpoint_cache_hit': beauty_execution['source_endpoint_cache_hit'],
                'beauty_replay_performed': beauty_execution['beauty_replay_performed']},
            'delta_anchor': 'beauty_working_premultiplied_foreground',
            'native_receiver_preservation': {'state': receiver_receipt['state'],
                'qualification': provenance_only(receiver_receipt['qualification']),
                'checks': receiver_receipt.get('checks'), 'visual_acceptance': False}},
        'visual_acceptance':False}
    snapshot_path = output/'preparation-snapshot.json'
    snapshot_path.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    snapshot_asset = {'path':str(snapshot_path),'sha256':sha(snapshot_path),'representation':'path_independent_preparation_provenance'}
    layer = {'id':'reference_appearance_stack','kind':'reference_appearance_stack','label':'人物与场景融合',
        'enabled':True,'strength':1., 'params':{'schema':'reference_appearance_stack.v2',
            'working_domain':'display_linear_sRGB','source_sha256':run['source']['sha256'],
            'reference_sha256':run['reference']['sha256'],'alpha_sha256':alpha_asset['sha256'],
            'source_fp_sha256':fp_asset['sha256'],'canvas_size':list(native),
            'fit_beauty_identity': parameters['binding'].get('portrait_beauty'),
            'prepared_stages':STAGES,'freeze_mode':'native_premultiplied_delta',
            'provenance_identity_sha256':snapshot_asset['sha256'],
            'strength_semantics':'0=source_person_appearance_with_current_background; 1=prepared_TBS_target; manual_edits_remain'},
        'assets':{'automatic_delta':delta_asset,'preparation_snapshot':snapshot_asset}}
    if snapshot.get('parameters', {}).get('source_hair_policy'):
        layer['params']['source_hair_policy'] = snapshot['parameters']['source_hair_policy']
    plan = copy.deepcopy(old_plan)
    plan.update(enabled=True, recipe=RECIPE, working_domain='display_linear_sRGB',
        summary='人物与场景融合已准备；可调整体强度、背景和局部笔刷，预览后保存新版本。',
        source_sha256=run['source']['sha256'], candidate_sha256=run['reference']['sha256'],
        layers=[layer,bg_layer,*contact_layers])
    # Scene/receiver editing still owns its removal and observation masks.
    # Keep their closure while replacing only the RGB domain and edge support.
    plan['mask_roles'] = copy.deepcopy(old_plan['mask_roles'])
    plan['mask_roles'].pop('edited_edge_light_applicability',None)
    plan['mask_roles']['edge_applicability'] = edge_asset
    plan['source_baseline'] = {'schema':'appearance_source_baseline.v1', 'working_domain':'display_linear_sRGB',
        'source_sha256':run['source']['sha256'], 'source_display_applied_once':True,
        'native_source_retouch':retouch_baseline_record(run),
        'source_precision':decode_precision['source_precision'],
        'decode_version':decode_precision['decode_version'],
        'source_display':source_display,
        'appearance_pair':{key: copy.deepcopy(value) for key, value in appearance_pair.items() if key != 'assets'},
        'assets':{'foreground_premultiplied':fp_asset,
        'foreground_alpha':copy.deepcopy(alpha_asset),'old_environment':old_bg_asset}}
    plan['execution'] = {'source_reference':'paired_native_baseline','receiver_context':'current_environment',
        'working_domain':'display_linear_sRGB','workflow':STAGES+(
            ['background_receiver_attenuation'] if contact_layers else [])+
            ['candidate_subject_light_evidence','source_environment_light',
             'single_composition','manual_refinement'],
        'automatic_subject_lighting':'candidate_subject_light_evidence_then_source_environment_light',
        'source_environment_light':True,
        'source_display_applied_at_native_prepare':True,'display_transform_after_reference':'clip_sdr_encode_only'}
    plan['provenance'] = {key:copy.deepcopy(value) for key,value in (old_plan.get('provenance') or {}).items()
        if key in {'local_model_registry','source_foreground','candidate_floor_contract','source_scene_enhancement'}}
    if receiver_result['design_reconstruction_plan'].get('provenance', {}).get('native_receiver_preservation'):
        plan['provenance']['native_receiver_preservation'] = copy.deepcopy(
            receiver_result['design_reconstruction_plan']['provenance']['native_receiver_preservation'])
    plan['provenance']['reference_pipeline'] = {'recipe':RECIPE,'prepared_stages':STAGES,
        'preparation_snapshot_sha256':snapshot_asset['sha256'],'producer_code':snapshot['producer_code'],
        'baseline_master_sha256':receipt['output_sha256'], 'visual_acceptance':False}
    plan.pop('source_appearance_responsibility',None)
    from reference_appearance_controls import prepare_owned_controls
    with workflow_timing.span('appearance_controls', model_request=False, http_request=False):
        plan = prepare_owned_controls(plan, output/'controls', run=run, native_inputs={
            'working_foreground': working_foreground, 'alpha': alpha,
            'automatic_delta': automatic_delta, 'beauty_execution': _beauty_execution})
    del working_foreground, automatic_delta
    controls_layer = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    controls_receipt = read(controls_layer['assets']['control_receipt']['path'])
    checks = {**snapshot['checks'], 'independent_controls': {
        'control_recipe': controls_layer['params']['control_recipe'],
        'endpoint_contract': controls_receipt['endpoint_contract'],
        'gamut_dose': controls_receipt['gamut_dose'],
        'original_endpoint_exceptions': controls_receipt['original_endpoint_exceptions']}}
    result = {'ok':True,'design_reconstruction_plan':plan,'cache_hit':False,
        'elapsed_seconds':round(time.perf_counter()-started,2),'prepared_stages':STAGES,
        'checks':checks,'new_rendered_revision':False,'project_pointer_updated':False,
        'workflow_timing': {'schema': workflow_timing.SCHEMA, 'path': str(workflow_timing.current().path),
            'scope': 'native_freeze_worker_body', 'clock': 'time.perf_counter'}}
    target = output/'prepared-plan.json'
    temporary = target.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    temporary.replace(target)
    print(json.dumps({'ok':True,'prepared_plan':str(target),'elapsed_seconds':result['elapsed_seconds'],'checks':snapshot['checks']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--match-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    freeze(read(args.run),args.match_dir,args.output)


if __name__ == '__main__':
    main()
