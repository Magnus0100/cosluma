"""Prepare an independently bound final-subject low-frequency response layer.

Reference observations and source correction guidance are filtered. Native
source RGB, geometry and alpha
are replayed unchanged up to the current appearance endpoint; a finite sparse
associated delta records the resulting pointwise appearance correction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'subject_environment_response.v1'
RECIPE = 'final_subject_low_frequency_residual_v1'
KIND = 'source_environment_response'
DOMAIN = 'display_linear_sRGB'
OWNERS = (1, 4, 13, 23)
# The response field is prepared on the bounded proxy canvas.  Native
# material texture is deliberately not used as an interpolation guide: it is
# the output owner's RGB and can contain zipper threads, weave and sensor
# grain at a frequency absent from the validated field.  Proxy guidance still
# gates interpolation across coarse material/semantic edges.
from subject_environment_field import GUIDANCE_POLICY


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf8'))


def asset(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def cache_roots(root, plan):
    """Use the current workspace and explicitly owned plan locators only."""
    roots = [Path(root).resolve()]
    for item in [*plan['source_baseline']['assets'].values(),
                 plan['mask_roles']['source_semantics'],
                 *next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')['assets'].values()]:
        for parent in Path(item['path']).resolve().parents:
            if parent.name == 'outputs' and parent.parent not in roots:
                roots.append(parent.parent)
                break
    return roots


def _verified(asset_record):
    from reference_appearance_stack import verify_asset
    verify_asset(asset_record)
    return Path(asset_record['path']).resolve()


def load_observations(root, run, plan):
    """Reopen exact independent correspondence, labels and RGB by frozen hashes.

    This does not trust a cache directory name or create new model evidence.
    Historical producer identity is verified against the immutable snapshot;
    no current inference installation or license service is needed for replay.
    """
    import cv2
    from PIL import Image
    from reference_local_correspondence import array_sha, canonical_sha
    from reference_automatic_layers import _receipt_provenance, _validate_correspondence_receipt
    from source_semantic_binding import load_source_semantics
    from float_appearance import srgb_to_linear
    for role in ('source', 'reference'):
        if sha(run[role]['path']) != run[role]['sha256']:
            raise ValueError('Environment photograph bytes changed: '+role)
    stack = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    snapshot_asset = stack['assets']['preparation_snapshot']
    snapshot = read(_verified(snapshot_asset))
    record = snapshot['colour_layer']['local_correspondence']
    expected = record['inputs']
    native, proxy = plan['canvas']['source_oriented_size'], plan['canvas']['source_proxy_size']
    labels, _ = load_source_semantics(plan['mask_roles']['source_semantics'],
        source_sha256=run['source']['sha256'], native_size=native, proxy_size=proxy)
    if (record.get('schema') != 'automatic_local_correspondence_record.v2'
            or expected['source_sha256'] != run['source']['sha256']
            or expected['reference_sha256'] != run['reference']['sha256']
            or expected['proxy_size'] != proxy
            or array_sha(labels) != expected['source_labels_sha256']
            or canonical_sha(record['receipt']) != record['receipt_content_sha256']):
        raise ValueError('Environment correspondence differs from the frozen source/reference/owner snapshot')
    roots = cache_roots(root, plan)
    receipt_path = None
    for base in roots:
        for path in sorted((base/'outputs/cache/reference-local-correspondence').glob('*/*/receipt.json')):
            if sha(path) == record['receipt_sha256']:
                receipt_path = path; break
        if receipt_path: break
    if receipt_path is None:
        raise ValueError('The exact independently bound local correspondence receipt is unavailable; prepare correspondence first')
    receipt = read(receipt_path)
    if _receipt_provenance(receipt) != record['receipt']:
        raise ValueError('Environment local correspondence retained producer evidence changed')
    map_record = receipt['asset']
    if map_record['sha256'] != record['asset']['sha256']:
        raise ValueError('Environment map differs from the appearance preparation snapshot')
    if not Path(map_record['path']).is_file():
        # Exact receipt bytes remain immutable; only the consumer's locator can
        # be relocated to the sibling map with the same certified byte hash.
        candidate = receipt_path.parent/'local-correspondence.npz'
        if sha(candidate) != map_record['sha256']:
            raise ValueError('Relocated environment map bytes do not match the producer')
        receipt = copy.deepcopy(receipt)
        receipt['asset']['path'] = str(candidate.resolve())
        map_record = receipt['asset']
    rw, rh = receipt['reference_size']
    reference_labels = None; label_asset = None
    rgb_assets = {}
    for role in ('source', 'reference'):
        digest = expected['cached_rgb_sha256'][role]
        for base in roots:
            candidates = (base/'outputs/cache/local-analysis'/run[role]['sha256'][:16]).glob('*/source-srgb.png')
            for path in sorted(candidates):
                if sha(path) == digest:
                    rgb_assets[role] = asset(path); break
            if role in rgb_assets: break
        if role not in rgb_assets:
            raise ValueError('Environment reference observations need the exact bound decoded RGB: '+role)
    for base in roots:
        candidates = (base/'outputs/cache/local-analysis'/run['reference']['sha256'][:16]).glob('*/semantics/labels.npz')
        for path in sorted(candidates):
            with np.load(path, allow_pickle=False) as data:
                candidate = cv2.resize(data['labels'], (rw, rh), interpolation=cv2.INTER_NEAREST)
            if array_sha(candidate) == expected['reference_labels_sha256']:
                reference_labels = candidate; label_asset = asset(path); break
        if reference_labels is not None: break
    if reference_labels is None:
        raise ValueError('Environment reference semantic observations do not match the independent map')
    for role, field in [('source', 'source_faces_sha256'), ('reference', 'reference_faces_sha256')]:
        if canonical_sha(record['face_observations'][role]) != expected[field]:
            raise ValueError('Environment correspondence face observation binding changed')
    maps = _validate_correspondence_receipt(receipt, map_record, expected, labels, reference_labels)
    with Image.open(rgb_assets['reference']['path']) as image:
        if image.mode != 'RGB':
            raise ValueError('Environment reference observation must be decoded RGB')
        reference = srgb_to_linear(np.asarray(image, np.float32)/255)
    if reference.shape[:2] != (rh, rw):
        reference = cv2.resize(reference, (rw, rh), interpolation=cv2.INTER_AREA)
    proof = {'snapshot': asset(snapshot_asset['path']), 'receipt': asset(receipt_path),
        'map': dict(map_record), 'decoded_rgb': rgb_assets, 'reference_labels': label_asset,
        'input_binding': expected, 'independent_geometry_verified': True,
        'model_inferences': 0, 'reference_texture_imported': False}
    return labels, reference_labels, reference, maps, proof


def endpoint(plan, run, box, size):
    from design_reconstruction import sample_asset
    from reference_appearance_stack import apply_stack
    background = next(row for row in plan['layers'] if row['kind'] == 'background_design')['assets']
    stack = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    rgb, _ = apply_stack(sample_asset(background['foreground_premultiplied'], 'values', box, size),
        stack, plan, run, box=box, canvas_size=size)
    for row in plan['layers']:
        if row['kind'] == 'source_subject_detail' and row.get('enabled') and row.get('strength', 1):
            from native_subject_detail_runtime import apply_detail
            rgb, _ = apply_detail(rgb, row, plan, run, box=box, canvas_size=size)
    return rgb


def guided_native_field(field, guide_lab, labels, box, native):
    """Upsample a low-frequency field without importing native texture.

    Compare proxy neighbours with the nearest proxy guide sample to preserve
    coarse source/material edges without sampling native piping or grain.
    """
    l,t,r,b = box; ph,pw = labels.shape
    xx = (np.arange(l,r, dtype=np.float32)+.5)*pw/native[0]-.5
    yy = (np.arange(t,b, dtype=np.float32)+.5)*ph/native[1]-.5
    x0=np.floor(xx).astype(int); y0=np.floor(yy).astype(int)
    fx=xx-x0; fy=yy-y0
    nx=np.clip(np.floor(xx+.5).astype(int),0,pw-1)
    ny=np.clip(np.floor(yy+.5).astype(int),0,ph-1)
    owners=labels[np.ix_(ny,nx)]
    nearest_active=field[np.ix_(ny,nx)][...,3]>0
    centre_lab=guide_lab[np.ix_(ny,nx)]
    total=np.zeros((b-t,r-l),np.float32)
    result=np.zeros((b-t,r-l,4),np.float32)
    for dy,wy in ((0,1-fy),(1,fy)):
        py=np.clip(y0+dy,0,ph-1)
        for dx,wx in ((0,1-fx),(1,fx)):
            px=np.clip(x0+dx,0,pw-1); sl=np.ix_(py,px)
            delta=np.linalg.norm(guide_lab[sl]-centre_lab,axis=-1)
            weight=(wy[:,None]*wx[None,:]).astype(np.float32)
            weight*=np.exp(-.5*(delta/.06)**2)*(delta<=.18)
            weight*=(labels[sl]==owners)&(field[sl][...,3]>0)&nearest_active
            result+=field[sl]*weight[...,None]; total+=weight
    np.divide(result,total[...,None],out=result,where=total[...,None]>1e-12)
    result[total<=1e-12]=0
    result[~np.isin(owners,OWNERS)]=0
    return result, owners


def prepare_environment(root, run, output):
    from automatic_subject_environment_completion import _diagnosis
    from native_subject_detail import validate_diagnosed_plate
    from design_reconstruction import sample_asset, reference_stack_execution_weights
    from reference_colour_fit import to_lab
    from reference_scene_balance import apply_scene_balance
    from subject_environment_field import prepare_environment_field, proxy_response_guidance
    from subject_environment_runtime import array_sha, preceding_detail_digest, foreground_controls_digest, scene_context_digest
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    base = {'schema':SCHEMA,'recipe':RECIPE,'source_sha256':run['source']['sha256'],
        'reference_sha256':run['reference']['sha256'],'state':'unsupported',
        'visual_acceptance':False,'image_generation_submissions':0,'model_inferences':0}
    from runtime_resources import code_path
    base['producer_code_sha256']={name:sha(code_path(name)) for name in (
        'subject_environment_worker.py','subject_environment_field.py',
        'subject_environment_runtime.py','reference_automatic_layers.py',
        'reference_scene_balance.py','reference_colour_fit.py')}
    def finish(receipt, changed=False, result_plan=None):
        write(output/'subject-environment.json',receipt)
        result={'ok':True,'changed':changed,'reason':receipt['reason'],'receipt':receipt}
        if result_plan is not None: result['design_reconstruction_plan']=result_plan
        write(output/'prepared-subject-environment.json',result)
        return result
    diagnosis = _diagnosis(root,run)
    if diagnosis is None:
        return finish({**base,'state':'not_required','reason':'no_current_final_subject_environment_failure'})
    base['diagnosis']=diagnosis['evidence']
    validate_diagnosed_plate(root,run)
    if any(row.get('kind')==KIND for row in plan['layers']):
        return finish({**base,'reason':'existing_environment_response_requires_a_new_endpoint_task'})
    # This producer observes the native endpoint before manual foreground
    # finishing. A diagnosed final with active downstream edits/effects cannot
    # be silently treated as that endpoint.
    from design_reconstruction import _manual_light_plans
    from native_roi_render import _operations
    from source_color import source_color_plan_for_run
    revision=run['options']['appearance_revision']
    controls=('exposure','contrast','temperature','tint','saturation','highlights','shadows','sharpness')
    appearance_active=any(spec.get('strength',1) and any(spec.get(key,0) for key in controls)
        for branch in ('foreground','composite') for spec in _operations(run,branch))
    manual_active=any(isinstance(spec,dict) and (spec.get('enabled') or spec.get('strokes'))
        for spec in (run['options'].get('manual_layers') or {}).values())
    light_active=any(branch.get('enabled') and branch.get('light_fields')
        for name,branch in _manual_light_plans(run).items() if name in ('foreground','composite'))
    color=source_color_plan_for_run(run)
    color_active=color.get('enabled') and revision.get('source_color_plan_origin') in {
        'manual','model_with_user_edits','local_model_with_user_edits','local_automatic_with_user_edits','user'}
    if (run['options'].get('denoise_strength',0)
            or manual_active or appearance_active or light_active or color_active
            or any(row.get('enabled') and row.get('strength',1) and row['kind'] in {
                'rim_light','atmosphere','particles','bloom','independent_effect'} for row in plan['layers'])):
        return finish({**base,'reason':'downstream_foreground_endpoint_not_independently_observed'})
    baseline=plan['source_baseline']['assets']
    current_fg=next(row for row in plan['layers'] if row['kind']=='background_design')['assets']['foreground_premultiplied']
    if (current_fg['sha256'] != baseline['foreground_premultiplied']['sha256']
            or plan['mask_roles']['foreground_alpha']['sha256'] != baseline['foreground_alpha']['sha256']
            or plan['mask_roles'].get('edited_edge_light_applicability')):
        return finish({**base,'reason':'changed_source_foreground_or_alpha_requires_new_qualification'})
    labels,reference_labels,reference,maps,proof=load_observations(root,run,plan)
    native=tuple(plan['canvas']['source_oriented_size']); proxy=tuple(plan['canvas']['source_proxy_size'])
    full=(0,0,*proxy); baseline=plan['source_baseline']['assets']
    alpha=sample_asset(plan['mask_roles']['foreground_alpha'],'values',full,proxy)
    guide_fp=sample_asset(baseline['foreground_premultiplied'],'values',full,proxy)
    guide=np.divide(guide_fp,alpha[...,None],out=np.zeros_like(guide_fp),where=alpha[...,None]>1e-6)
    current_fp=endpoint(plan,run,full,proxy)
    current=np.divide(current_fp,alpha[...,None],out=np.zeros_like(current_fp),where=alpha[...,None]>1e-6)
    current=np.clip(current,0,1);guide=np.clip(guide,0,1)
    protection_asset=plan['mask_roles'].get('identity_protection')
    if protection_asset:
        protected=sample_asset(protection_asset,'protection',full,proxy)>=.01
        alpha=alpha.copy();alpha[protected]=0
    field,report=prepare_environment_field(current,guide,reference,labels,reference_labels,alpha,maps)
    base.update(observation_proof=proof,field_report=report)
    field_path=output/'environment-field.npz'
    guide_lab=proxy_response_guidance(to_lab(guide),labels)
    np.savez_compressed(field_path,values=field,source_guide_lab=guide_lab)
    base['field_asset']=asset(field_path)
    if not np.any(field[...,:3]):
        return finish({**base,'state':'rejected','reason':'no_independently_qualified_low_frequency_residual'})
    stack=next(row for row in plan['layers'] if row['kind']=='reference_appearance_stack')
    weights=reference_stack_execution_weights(stack)['bank_weights']
    binding={'source_sha256':run['source']['sha256'],'reference_sha256':run['reference']['sha256'],
        'native_size':list(native),'working_domain':DOMAIN,
        'native_guidance_policy':GUIDANCE_POLICY,
        'source_fp_sha256':baseline['foreground_premultiplied']['sha256'],
        'source_alpha_sha256':baseline['foreground_alpha']['sha256'],
        'alpha_sha256':plan['mask_roles']['foreground_alpha']['sha256'],
        'semantic_sha256':plan['mask_roles']['source_semantics']['sha256'],
        'identity_protection_sha256':protection_asset['sha256'] if protection_asset else None,
        'appearance_weights':weights,'appearance_banks':{key:stack['assets'][key]['sha256']
            for key,value in weights.items() if value},
        'preceding_detail_digest':preceding_detail_digest(plan),
        'scene_context_digest':scene_context_digest(plan),
        'foreground_controls_digest':foreground_controls_digest(run)}
    patches=[]; arrays={};total=0
    changed_min=np.full(3,np.inf);changed_max=np.full(3,-np.inf)
    for top in range(0,native[1],512):
        for left in range(0,native[0],512):
            box=(left,top,min(native[0],left+512),min(native[1],top+512))
            # Avoid expensive native endpoint reads for tiles without any
            # observed owner support, without introducing a per-photo ROI.
            px0=max(0,int(left*proxy[0]/native[0])-1);px1=min(proxy[0],int(box[2]*proxy[0]/native[0])+2)
            py0=max(0,int(top*proxy[1]/native[1])-1);py1=min(proxy[1],int(box[3]*proxy[1]/native[1])+2)
            if not np.any(field[py0:py1,px0:px1,:3]):continue
            a=sample_asset(plan['mask_roles']['foreground_alpha'],'values',box,native)
            f,owners=guided_native_field(field,guide_lab,labels,box,native)
            allowed=(a>=.995)&np.isin(owners,OWNERS)
            if protection_asset: allowed &= sample_asset(protection_asset,'protection',box,native)<.01
            f[~allowed]=0
            if not np.any(f[...,:3]):continue
            target=endpoint(plan,run,box,native)
            straight=np.divide(target,a[...,None],out=np.zeros_like(target),where=a[...,None]>1e-6)
            if np.any(allowed & ((straight.min(-1)<0)|(straight.max(-1)>1))):
                raise ValueError('Current native environment target exceeds the observed display domain')
            corrected=apply_scene_balance(np.clip(straight,0,1),f)
            delta=((corrected-straight)*a[...,None]).astype(np.float32);delta[~allowed]=0
            count=int(np.any(delta!=0,axis=-1).sum())
            if not count:continue
            key='patch_'+str(len(patches));arrays[key]=delta;total+=count
            active=np.any(delta!=0,axis=-1)
            changed_min=np.minimum(changed_min,corrected[active].min(0));changed_max=np.maximum(changed_max,corrected[active].max(0))
            patches.append({'native_box':list(box),'array_key':key,'delta_array_sha256':array_sha(delta),
                'changed_pixels':count,'target_patch_sha256':array_sha(target)})
    if not patches:return finish({**base,'state':'rejected','reason':'no_native_supported_change'})
    delta_path=output/'environment-delta.npz';np.savez_compressed(delta_path,**arrays)
    frozen={'schema':SCHEMA,'recipe':RECIPE,'binding':binding,'delta':asset(delta_path),'patches':patches,
        'donor_assets':copy.deepcopy(baseline),'owner_asset':copy.deepcopy(plan['mask_roles']['source_semantics']),
        'approved_retouch_binding':copy.deepcopy(plan['source_baseline'].get('native_source_retouch'))}
    frozen_path=output/'frozen-subject-environment.json';write(frozen_path,frozen)
    receipt={**base,'state':'prepared','reason':'independently_qualified_current_subject_environment_residual',
        'qualification':{'qualified':True,'actual_saved_evidence_verified':True,'holdout_improved':True,
            'scope':'only_qualified_hair_and_clothing_correspondence_support; artistic_acceptance_requires_new_review'},
        'checks':{'changed_pixels':total,'alpha_changed':False,'source_geometry_changed':False,
            'approved_source_retouch_reapplied':False,'reference_texture_imported':False,
            'corrected_RGB_range':[changed_min.tolist(),changed_max.tolist()],
            'source_native_texture_is_output_owner':True,'unknown_support_changed':False,
            'native_guidance_policy':GUIDANCE_POLICY}}
    write(output/'subject-environment.json',receipt)
    result_plan=copy.deepcopy(plan)
    result_plan['layers'].append({'id':KIND,'kind':KIND,'label':'人物环境光色协调','enabled':True,'strength':1.,
        'params':{'schema':SCHEMA,'recipe':RECIPE,'binding':binding},
        'assets':{'frozen_delta':asset(frozen_path),'qualification_receipt':asset(output/'subject-environment.json')}})
    result_plan.setdefault('provenance',{})['subject_environment_response']={
        'schema':SCHEMA,'recipe':RECIPE,'changed_pixels':total,'visual_acceptance':False,
        'source_geometry_changed':False,'alpha_changed':False}
    return finish(receipt,True,result_plan)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args(argv)
    result=prepare_environment(args.root,read(args.run),args.output)
    print(json.dumps({'changed':result['changed'],'reason':result['reason']},ensure_ascii=True))


if __name__=='__main__':main()
