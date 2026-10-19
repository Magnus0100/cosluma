"""Native source scenery, bounded photographic adjustment and owned provenance."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from asset_identity import sha256_file
from cache_store import exclusive_file_lock
from native_array_storage import NativeArrayRowWriter

SCHEMA = 'source_scene_enhancement.v1'
PROFILE = {'maximum_exposure_EV': .5, 'maximum_channel_chroma_EV': .15,
           'inward_background_fade_proxy_pixels': 2., 'minimum_observed_pixels': 64,
           'adjustment': 'robust_global_low_frequency_gain; source_native_texture_owner'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pixels_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode()+value.tobytes()).hexdigest()


def asset(path, **metadata):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path), **metadata}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def mode(run):
    intent = run.get('task_intent') or (run.get('options') or {}).get('task_intent') or {}
    return (intent.get('constraints') or {}).get('background_mode') == 'enhance_source'


def background(plan):
    rows = [row for row in plan['layers'] if row.get('kind') == 'background_design']
    if len(rows) != 1: raise ValueError('Source enhancement requires one background owner')
    return rows[0]


def execution_stats(run, plan, regional=None, *, reused=False):
    if reused:
        receipt=validate_observation(plan,run)
        regional=receipt.get('regional_refinement')
    count=0 if reused or not regional else regional['new_model_submissions']
    ids=[]
    if regional:
        frozen=json.loads(Path(regional['receipt']['path']).read_text('utf8'))
        if not frozen.get('prompt_id'):raise ValueError('Regional execution lacks its actual provider task ID')
        ids=[frozen['prompt_id']]
    return {'new_model_submissions':count,'provider_ids':ids if count else [],
        'reused_saved_regions':int(bool(regional) and not count),'reused_provider_ids':ids if not count else [],
        'origin_run_id':(run.get('generation_budget') or {}).get('origin_run_id') or run.get('run_id'),
        'regional_state':('saved regional result reused' if reused and regional else
                          regional['state'] if regional else 'not_requested')}


def transform_for(plan):
    from display_transform import validate_source_transform
    baseline = plan['source_baseline']
    transform = validate_source_transform(baseline['source_display'], baseline['source_precision'])
    consumed = baseline.get('source_display_applied_once')
    domain = baseline.get('working_domain')
    if consumed is True and domain == 'display_linear_sRGB': return None
    if consumed is False and domain == transform['input_domain']: return transform
    raise ValueError('Source scenery has an ambiguous display-transform consumption')


def fit_gain(source, target, alpha, validity, *, registered):
    """A spatially constant low-frequency field cannot invent scene geometry.

    No histogram texture transfer or local observation-hull application mask.
    Unregistered references provide no automatic colour evidence.
    """
    permitted = (alpha == 0)&(validity > .999)&np.all(source > .003, -1)&np.all(target > .003, -1)
    if not registered or permitted.sum() < PROFILE['minimum_observed_pixels']:
        return np.ones(3, np.float32), {'state': 'source_retained_no_qualified_colour_evidence', 'samples': int(permitted.sum())}
    s, t = source[permitted], target[permitted]
    luma = np.asarray([.2126,.7152,.0722], np.float32)
    exposure = float(np.clip(np.median(np.log2((t@luma)/(s@luma))),
        -PROFILE['maximum_exposure_EV'], PROFILE['maximum_exposure_EV']))
    chroma = np.median(np.log2(t/s), axis=0)
    chroma -= float(chroma@luma)
    chroma = np.clip(chroma, -PROFILE['maximum_channel_chroma_EV'], PROFILE['maximum_channel_chroma_EV'])
    gain = np.exp2(exposure+chroma).astype(np.float32)
    return gain, {'state': 'bounded_reference_colour_evidence', 'samples': len(s),
                  'exposure_EV': exposure, 'channel_chroma_EV': chroma.tolist()}


def adjust(source, gain, weight):
    """Common RGB gamut dose; original spatial detail is never filtered."""
    change = source*(np.asarray(gain, np.float32)-1)*weight[..., None]
    scale = np.ones(weight.shape, np.float32)
    for channel in range(3):
        d, s = change[..., channel], source[..., channel]
        limit = np.ones_like(d)
        np.divide(1-s, d, out=limit, where=d > 0)
        np.divide(-s, d, out=limit, where=d < 0)
        scale = np.minimum(scale, limit)
    scale = np.clip(scale, 0, 1)
    scale = np.where(scale < 1, scale*(1-4*np.finfo(np.float32).eps), scale)
    return (source+change*scale[..., None]).astype(np.float32)


def validate_observation(plan, run):
    from reference_appearance_stack import verify_asset
    record = background(plan)['assets']['background_observation']; verify_asset(record)
    receipt = json.loads(Path(record['path']).read_text(encoding='utf-8-sig'))
    if (receipt.get('schema') != SCHEMA or receipt.get('source_sha256') != run['source']['sha256']
            or receipt.get('reference_sha256') != run['reference']['sha256']
            or receipt.get('native_size') != plan['canvas']['source_oriented_size']):
        raise ValueError('Source scenery receipt differs from source/reference/native canvas')
    for key in ('source_environment','source_alpha','base_background','output_background','validity','origin_map'):
        verify_asset(receipt[key])
    assets = background(plan)['assets']
    if any(assets[key]['sha256'] != receipt[role]['sha256'] for key,role in
           (('background','output_background'),('validity','validity'))):
        raise ValueError('Source scenery endpoint differs from its frozen provenance')
    regional = receipt.get('regional_refinement')
    if regional:
        from regional_scene_worker import validate_saved
        validate_saved(regional, run, plan)
    return receipt


def prepare(root, run, plan, output, cancel=None):
    if not mode(run): return plan
    from design_reconstruction import sample_asset
    from display_transform import display_linear
    started = time.perf_counter(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    result = copy.deepcopy(plan); native = tuple(plan['canvas']['source_oriented_size'])
    w,h = native; proxy = tuple(plan['canvas']['source_proxy_size'])
    baseline = plan['source_baseline']; source = baseline['assets']['old_environment']
    alpha_asset = plan['mask_roles']['foreground_alpha']; assets = background(plan)['assets']
    if alpha_asset['sha256']!=baseline['assets']['foreground_alpha']['sha256']:
        raise ValueError('Source-scene enhancement requires the original subject coverage; edited alpha exposes undefined scenery')
    transform = transform_for(plan)
    identity = {'source': run['source']['sha256'], 'reference': run['reference']['sha256'],
        'environment': source['sha256'], 'alpha': alpha_asset['sha256'], 'native': list(native),
        'transform': transform, 'colour_evidence': {k:assets[k]['sha256'] for k in ('background','validity')},
        'profile': PROFILE, 'producer': sha256_file(__file__)}
    directory = output/digest(identity)[:24]; directory.mkdir(exist_ok=True)
    with exclusive_file_lock(directory/'prepare.lock'):
        marker = directory/'source-scene.json'
        if marker.is_file():
            saved = json.loads(marker.read_text('utf8'))
            if saved.get('identity') != identity: raise ValueError('Source scenery cache identity mismatch')
            for key in ('base_background','output_background','validity','origin_map'):
                from reference_appearance_stack import verify_asset
                verify_asset(saved[key])
            base = saved
        else:
            carrier = directory/'source-display-background.npz'
            writer = NativeArrayRowWriter(carrier, (h,w,3))
            try:
                for top in range(0,h,128):
                    if cancel and cancel(): raise InterruptedError('Source scenery cancelled')
                    v = sample_asset(source,'values',(0,top,w,min(h,top+128)),native)
                    v = display_linear(v,transform) if transform is not None else v.copy()
                    if np.any((v < 0)|(v > 1)): raise ValueError('Source scenery must be display-linear SDR')
                    writer.write(v.astype(np.float32))
                writer.publish()
            finally: writer.close()
            meta = {'size':list(native),'native_size':list(native),'coordinate_space':'source_canvas',
                    'working_space':'display_linear_sRGB','sampling':'bilinear'}
            carrier_asset = asset(carrier, **meta, representation='straight_RGB')
            box=(0,0,*proxy)
            src = sample_asset(carrier_asset,'values',box,proxy)
            target = sample_asset(assets['background'],'values',box,proxy)
            v = sample_asset(assets['validity'],'values',box,proxy)
            target = np.divide(target,v[...,None],out=np.zeros_like(target),where=v[...,None]>1e-6)
            a = sample_asset(alpha_asset,'values',box,proxy)
            gain, evidence = fit_gain(src,target,a,v,registered=bool(
                plan['canvas'].get('candidate_mapping',{}).get('scene_registration_supported')))
            distance = distance_transform_edt(np.pad(a==0,1))[1:-1,1:-1]
            fade = np.clip((distance-1)/PROFILE['inward_background_fade_proxy_pixels'],0,1).astype(np.float32)
            fade *= fade*(3-2*fade)
            from native_array_storage import save_native_array
            fade_path=directory/'adjustment-support.npz';save_native_array(fade_path,fade)
            fade_asset=asset(fade_path,size=list(proxy))
            paths=[directory/name for name in ('background.npz','validity.npz','origin.npz')]
            writers=[NativeArrayRowWriter(paths[0],(h,w,3)), NativeArrayRowWriter(paths[1],(h,w)),
                     NativeArrayRowWriter(paths[2],(h,w))]
            try:
                for top in range(0,h,128):
                    if cancel and cancel(): raise InterruptedError('Source scenery cancelled')
                    box=(0,top,w,min(h,top+128));b=sample_asset(carrier_asset,'values',box,native)
                    a=sample_asset(alpha_asset,'values',box,native)
                    amount=sample_asset(fade_asset,'values',box,native);amount[a!=0]=0
                    enhanced=adjust(b,gain,amount)
                    # B is undefined behind opaque source; validity never claims observation there.
                    validity=(a<1).astype(np.float32)
                    origin=np.where(a==0,1,np.where(a<1,2,0)).astype(np.float32)
                    writers[0].write(enhanced*validity[...,None]);writers[1].write(validity);writers[2].write(origin)
                for writer in writers: writer.publish()
            finally:
                for writer in writers: writer.close()
            base={'schema':SCHEMA,'identity':identity,'source_sha256':run['source']['sha256'],
                'reference_sha256':run['reference']['sha256'],'native_size':list(native),
                'source_environment':copy.deepcopy(source),'source_alpha':copy.deepcopy(alpha_asset),
                'source_transform':copy.deepcopy(baseline['source_display']),
                'source_display_consumed_here':transform is not None,'profile':PROFILE,'colour_evidence':evidence,
                'base_background':asset(paths[0],**meta,representation='sampling_validity_associated_RGB',validity_associated=True),
                'output_background':asset(paths[0],**meta,representation='sampling_validity_associated_RGB',validity_associated=True),
                'validity':asset(paths[1],**meta,representation='sampling_coverage'),
                'origin_map':asset(paths[2],**meta,representation='categorical_origin'),
                'origin_codes':{'undefined':0,'source_observed':1,'source_FB_estimated':2,'regional_generated':3},
                'source_geometry_changed':False,'source_Fp_alpha_changed':False,'image_generation_submissions':0,
                'visual_acceptance':False,'elapsed_seconds':time.perf_counter()-started}
            write(marker,base)
    out_assets=background(result)['assets']
    for key in ('compact_observation','compact_validity','background_consistency','native_receiver_preservation',
                'native_receiver_detail','environment_completion','environment_origin'):
        out_assets.pop(key,None)
    out_assets.update(background=base['base_background'],validity=base['validity'])
    previous = ((run.get('options') or {}).get('appearance_revision') or {}).get('design_reconstruction_plan')
    saved_regional = None
    if previous and mode(run):
        old = background(previous)['assets'].get('background_observation')
        if old:
            from reference_appearance_stack import verify_asset
            verify_asset(old);receipt=json.loads(Path(old['path']).read_text(encoding='utf-8-sig'))
            if receipt.get('schema')==SCHEMA: saved_regional=receipt.get('regional_refinement')
    from regional_scene_worker import prepare as prepare_regional
    import workflow_timing
    with workflow_timing.span('regional_refinement',model_request=False,http_request=False) as measured:
        result, regional = prepare_regional(root,run,result,directory/'regional',saved=saved_regional,cancel=cancel)
        if measured:
            count=regional['new_model_submissions'] if regional else 0
            measured.annotate(model_request=bool(count),http_request=bool(count),model_submissions=count)
    final=copy.deepcopy(base)
    final['output_background']=copy.deepcopy(background(result)['assets']['background'])
    if regional:
        final['regional_refinement']=regional
        regional_receipt=json.loads(Path(regional['receipt']['path']).read_text('utf8'))
        core=regional_receipt['binding']['core_box']
        before=sample_asset(base['base_background'],'values',core,native)
        after=sample_asset(final['output_background'],'values',core,native)
        changed=np.any(before!=after,axis=-1)
        origin_path=directory/('origin-regional-'+regional['receipt']['sha256'][:16]+'.npz')
        writer=NativeArrayRowWriter(origin_path,(h,w))
        try:
            for top in range(0,h,128):
                end=min(h,top+128);values=sample_asset(base['origin_map'],'values',(0,top,w,end),native).copy()
                y0,y1=max(top,core[1]),min(end,core[3])
                if y0<y1:
                    patch=values[y0-top:y1-top,core[0]:core[2]]
                    patch[changed[y0-core[1]:y1-core[1]]]=3
                writer.write(values.astype(np.float32))
            writer.publish()
        finally:writer.close()
        final['origin_map']=asset(origin_path,**{k:v for k,v in base['origin_map'].items() if k not in {'path','sha256'}})
    # Keep source-only and regional observations independently immutable, not a rewritten cached base.
    final_path=directory/('source-scene-final-'+digest(final)[:16]+'.json');write(final_path,final)
    background(result)['assets']['background_observation']=asset(final_path,representation='source_native_scene_observation')
    result.setdefault('provenance',{})['source_scene_enhancement']={'schema':SCHEMA,'receipt_sha256':sha256_file(final_path),
        'origin':'source_native','regional_state':regional.get('state') if regional else 'not_requested',
        'visual_acceptance':False,**execution_stats(run,result,regional)}
    return result
