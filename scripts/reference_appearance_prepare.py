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
from runtime_resources import code_path
import time

import numpy as np


RECIPE = 'native-reference-appearance-freeze-v3-contact'
STAGES = ['reference_colour_match', 'reference_colour_blend', 'reference_scene_balance', 'edge_donor_extension']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def provenance_only(value):
    """Keep content identities and numeric choices, never live cache paths."""
    if isinstance(value, dict):
        return {key: provenance_only(item) for key, item in value.items()
                if key not in {'path', 'manifest', 'output'} and not key.endswith(('_path', '_file'))}
    if isinstance(value, list):
        return [provenance_only(item) for item in value]
    return value


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
            report['reason'] = 'historical_contact_contract_not_replayed'
            continue
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


def freeze(run, match_directory, output):
    import cv2
    from display_transform import display_linear, plan_for_run
    from raw_support import open_linear
    from source_foreground import estimate_source_background
    from render_reference_colour_composite import sample_maps, donor_fields, match_foreground, field_owners
    from reference_support import donor_support_report
    from reference_colour_blend import validate_layer
    from reference_scene_balance import validate_scene_layer

    started = time.perf_counter()
    directory, output = Path(match_directory).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    old_plan = run['options']['appearance_revision']['design_reconstruction_plan']
    parameters = read(directory/'parameters.json')
    receipt = read(directory/'composite-receipt.json')
    native = tuple(old_plan['canvas']['source_oriented_size'])
    proxy_size = tuple(old_plan['canvas']['source_proxy_size'])
    for role in ('source', 'reference'):
        if sha(run[role]['path']) != parameters['binding'][role] or run[role]['sha256'] != parameters['binding'][role]:
            raise ValueError('Prepared colour target belongs to different photographs')
    if receipt['parameters_sha256'] != sha(directory/'parameters.json'):
        raise ValueError('Prepared native target parameters changed')
    if receipt['renderer_sha256'] != sha(code_path('render_reference_colour_composite.py')):
        raise ValueError('Prepared native target renderer changed')
    if parameters['binding']['algorithm_sha256'] != sha(code_path('reference_colour_fit.py')):
        raise ValueError('Reprepare colour fitting after algorithm changes')
    with np.load(directory/'source-weights.npz', allow_pickle=False) as resource:
        weights = resource['values']
        chroma_weights = resource['chroma_values'] if 'chroma_values' in resource else None
    if sha(directory/'source-weights.npz') != parameters['binding']['source_weights_sha256']:
        raise ValueError('Prepared source weights changed')
    colour, colour_receipt = validate_layer(directory/'reference-colour-layer.json', parameters, native, proxy_size)
    scene, scene_receipt = validate_scene_layer(directory/'scene-balance-layer.json', parameters,
        directory/'reference-colour-layer.json', old_plan, native, proxy_size)
    colour_owners = field_owners(colour_receipt, colour.shape[:2])
    scene_owners = field_owners(scene_receipt, scene.shape[:2])
    with np.load(scene_receipt['asset']['path'], allow_pickle=False) as resource:
        donor_baseline = resource['baseline_rgb']
    alpha_asset = old_plan['mask_roles']['foreground_alpha']
    if sha(alpha_asset['path']) != parameters['binding']['alpha']:
        raise ValueError('The matte changed; prepare a new appearance foreground pair')
    with np.load(alpha_asset['path'], allow_pickle=False) as resource:
        alpha = resource['values']
    print('Preparing native appearance foreground and original environment', flush=True)
    from native_source_baseline import source_for_reference_prepare, retouch_baseline_record
    source, decode_precision, source_display = source_for_reference_prepare(run, binding=parameters['binding'])
    donors = donor_fields(cv2.resize(source, proxy_size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(alpha, proxy_size, interpolation=cv2.INTER_LINEAR), weights, parameters,
        colour, scene, donor_baseline, chroma_weights)
    donor_support = donor_support_report(donors, weights, parameters)
    trimap = np.full(alpha.shape, 128, np.uint8)
    trimap[alpha == 0], trimap[alpha == 1] = 0, 255
    old_background, foreground, edge_support, tiles, _, _ = estimate_source_background(source, alpha, trimap)
    reconstruction_error = 0.
    for top in range(0, len(source), 128):
        restored = foreground[top:top+128]+(1-alpha[top:top+128])[...,None]*old_background[top:top+128]
        reconstruction_error = max(reconstruction_error, float(np.max(np.abs(restored-source[top:top+128]))))
    del source, trimap

    def array_asset(name, values, representation):
        path = output/(name+'.npz')
        temporary = output/(name+'.tmp.npz')
        np.savez_compressed(temporary, values=values)
        temporary.replace(path)
        return {'path': str(path), 'sha256': sha(path), 'size': list(native),
                'coordinate_space': 'source_canvas', 'working_space': 'display_linear_sRGB',
                'representation': representation, 'sampling': 'bilinear'}

    print('Saving immutable source appearance pair', flush=True)
    fp_asset = array_asset('appearance-source-foreground', foreground, 'premultiplied_RGB')
    old_bg_asset = array_asset('appearance-original-environment', old_background, 'straight_RGB')
    del old_background
    edge_asset = array_asset('appearance-edge-support', edge_support, 'edge_confidence')
    print('Freezing full T -> colour fusion -> scene balance -> final edge donors', flush=True)
    zero_alpha_exact, nonnegative_target = True, True
    edge_gamut = {}
    for top in range(0, len(foreground), 128):
        end = min(len(foreground), top+128)
        box = (0, top, native[0], end)
        original = foreground[top:end].copy()
        matched = match_foreground(original, alpha[top:end], sample_maps(weights,box,proxy_size,native),
            sample_maps(donors,box,proxy_size,native), edge_support[top:end], parameters,
            origin=(0,top), canvas_size=native, reference_blend=sample_maps(colour,box,proxy_size,native,owners=colour_owners),
            scene_balance=sample_maps(scene,box,proxy_size,native,owners=scene_owners),
            chroma_weights=sample_maps(chroma_weights,box,proxy_size,native) if chroma_weights is not None else None,
            donor_gamut_report=edge_gamut)
        zero_alpha_exact &= bool(np.all(matched[alpha[top:end] == 0] == 0))
        nonnegative_target &= bool(np.isfinite(matched).all() and matched.min() >= 0)
        foreground[top:end] = matched-original
    delta_asset = array_asset('reference-appearance-delta', foreground, 'premultiplied_delta')
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
    receiver_result = prepare_native_receiver_preservation(run, receiver_plan, output/'receiver-preservation')
    bg_layer = next(row for row in receiver_result['design_reconstruction_plan']['layers'] if row['kind'] == 'background_design')
    receiver_receipt = receiver_result['receiver_preservation']
    snapshot = {'schema':'reference_preparation_snapshot.v1', 'recipe':RECIPE,
        'parameters':provenance_only(parameters), 'colour_layer':provenance_only(colour_receipt),
        'scene_layer':provenance_only(scene_receipt), 'producer_code':{
            name:sha(code_path(name)) for name in ('reference_appearance_prepare.py',
            'reference_colour_fit.py','reference_colour_blend.py','reference_scene_balance.py','render_reference_colour_composite.py',
            'reference_local_correspondence.py','reference_local_correspondence_worker.py',
            'reference_adaptive_fusion.py','reference_automatic_layers.py','reference_material_palette.py',
            'design_scene_assets.py','native_sole_contact.py','native_receiver_preservation.py')},
        'source_display':source_display, 'prepared_stages':STAGES, 'donor_support':donor_support,
        'photo_specific_inputs':{'authored_colour_edit':bool(parameters['binding'].get('authored_colour_edit')),
            'accessory_masks':bool(parameters['binding'].get('object_masks'))},
        'checks':{'source_reconstruction_max_error':reconstruction_error,'zero_alpha_exact':zero_alpha_exact,
            'finite_nonnegative_target':nonnegative_target,'composition_count':1,'reference_person_geometry_or_texture_imported':False,
            'edge_gamut':edge_gamut, 'contact_preservation':contact_receipt,
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
            'prepared_stages':STAGES,'freeze_mode':'native_premultiplied_delta',
            'provenance_identity_sha256':snapshot_asset['sha256'],
            'strength_semantics':'0=source_person_appearance_with_current_background; 1=prepared_TBS_target; manual_edits_remain'},
        'assets':{'automatic_delta':delta_asset,'preparation_snapshot':snapshot_asset}}
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
        'source_display':source_display, 'assets':{'foreground_premultiplied':fp_asset,
        'foreground_alpha':copy.deepcopy(alpha_asset),'old_environment':old_bg_asset}}
    plan['execution'] = {'source_reference':'paired_native_baseline','receiver_context':'current_environment',
        'working_domain':'display_linear_sRGB','workflow':STAGES+(
            ['background_receiver_attenuation'] if contact_layers else [])+['single_composition','manual_refinement'],
        'source_display_applied_at_native_prepare':True,'display_transform_after_reference':'clip_sdr_encode_only'}
    plan['provenance'] = {key:copy.deepcopy(value) for key,value in (old_plan.get('provenance') or {}).items()
        if key in {'source_foreground','candidate_floor_contract'}}
    if receiver_result['design_reconstruction_plan'].get('provenance', {}).get('native_receiver_preservation'):
        plan['provenance']['native_receiver_preservation'] = copy.deepcopy(
            receiver_result['design_reconstruction_plan']['provenance']['native_receiver_preservation'])
    plan['provenance']['reference_pipeline'] = {'recipe':RECIPE,'prepared_stages':STAGES,
        'preparation_snapshot_sha256':snapshot_asset['sha256'],'producer_code':snapshot['producer_code'],
        'diagnostic_master_sha256':receipt['output_sha256'], 'visual_acceptance':False}
    plan.pop('source_appearance_responsibility',None)
    from reference_appearance_controls import prepare_owned_controls
    plan = prepare_owned_controls(plan, output/'controls', run=run)
    controls_layer = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    controls_receipt = read(controls_layer['assets']['control_receipt']['path'])
    checks = {**snapshot['checks'], 'independent_controls': {
        'control_recipe': controls_layer['params']['control_recipe'],
        'endpoint_contract': controls_receipt['endpoint_contract'],
        'gamut_dose': controls_receipt['gamut_dose'],
        'original_endpoint_exceptions': controls_receipt['original_endpoint_exceptions']}}
    result = {'ok':True,'design_reconstruction_plan':plan,'cache_hit':False,
        'elapsed_seconds':round(time.perf_counter()-started,2),'prepared_stages':STAGES,
        'checks':checks,'new_rendered_revision':False,'project_pointer_updated':False}
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
