"""Bound observed floor reflections before cleanup; inference stays in its worker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_dilation, distance_transform_edt
from scipy.spatial import ConvexHull

from local_photo_analysis import file_identity
from local_scene_analysis import scene_model_identity, scene_runtime_paths
from runtime_resources import code_path


RECIPE = 'sam31-static-paired-floor-reflection-ownership-v3'
PARAMS = {'queries':['reflected legs','human legs'],
    'minimum_instance_confidence':.5,'maximum_leg_box_fraction':.6,
    'footwear_extent_radius':2.5,'sole_projection_margin':.04,'removal_dilation_pixels':2,
    'reflection_boundary_tolerance_pixels':2,'minimum_reflection_inliers':8,
    'minimum_candidate_overlap':.85,'minimum_source_overlap':.5,
    'minimum_inlier_hull_fraction':.1,'maximum_local_reprojection_pixels':2.5,
    'maximum_sole_gap_relative_to_footwear':.25}


def qualify_observed_reflections(source_fragment, candidate_fragment, source_floor, candidate_floor,
                                source_footwear, candidate_footwear, mapping, params=PARAMS,
                                *, candidate_proposals=None):
    """Protect a paired observed region, without copying or inventing RGB.

    A reflected-leg label alone is insufficient. Both learned regions must be
    on an observed floor and connected to real footwear, their masks must agree
    under supported scene registration, and distributed source/candidate feature
    inliers inside the region must verify its actual photographic structure.
    """
    from design_scene_assets import warp_values
    source_fragment, candidate_fragment = np.asarray(source_fragment,bool), np.asarray(candidate_fragment,bool)
    shape = candidate_fragment.shape
    protection = np.zeros(shape, bool)
    report = {'schema':'observed_floor_reflection_ownership.v1', 'qualified':False,
        'source_geometry_replayed':False, 'source_RGB_copied':False, 'synthetic_reflection_created':False,
        'mapping_supported':mapping.get('supported') is True, 'components':[],
        'complete_reflection_segmentation':False, 'protected_pixels':0}
    if mapping.get('supported') is not True:
        return protection, {**report, 'reason':'source_candidate_registration_unproven'}
    matrix = np.asarray(mapping.get('source_to_candidate_proxy'),np.float64)
    sp = np.asarray(mapping.get('inlier_source_points_scaled',[]),np.float64)
    cp = np.asarray(mapping.get('inlier_candidate_points',[]),np.float64)
    if (matrix.shape != (3,3) or not np.isfinite(matrix).all() or
            np.linalg.det(matrix) <= 0 or sp.ndim != 2 or sp.shape[1:] != (2,) or
            sp.shape != cp.shape or not np.isfinite(sp).all() or not np.isfinite(cp).all()):
        return protection, {**report, 'reason':'reflection_correspondence_evidence_unavailable'}
    sh,sw=source_fragment.shape;ch,cw=shape
    sp = sp * np.asarray([sw/cw,sh/ch])
    projected = np.column_stack([sp,np.ones(len(sp))]) @ matrix.T
    projected = projected[:,:2] / projected[:,2:]
    errors = np.linalg.norm(projected-cp,axis=1)
    sxy,cxy = np.rint(sp).astype(int),np.rint(cp).astype(int)
    valid = (sxy[:,0]>=0)&(sxy[:,0]<sw)&(sxy[:,1]>=0)&(sxy[:,1]<sh)&(
        cxy[:,0]>=0)&(cxy[:,0]<cw)&(cxy[:,1]>=0)&(cxy[:,1]<ch)&(
        errors<=params['maximum_local_reprojection_pixels'])

    def foot_owned(fragment, floor, footwear):
        fragment=fragment & np.asarray(floor,bool) & ~np.asarray(footwear,bool)
        ys,xs=np.nonzero(footwear)
        owned=np.zeros(fragment.shape,bool)
        if not len(xs):return owned
        extent=max(int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1))
        distances=distance_transform_edt(~np.asarray(footwear,bool))
        components,count=label(fragment,np.ones((3,3),bool))
        for index in range(1,count+1):
            part=components==index
            if part.sum()>=params['minimum_reflection_inliers'] and distances[part].min()<=max(
                    2,extent*params['maximum_sole_gap_relative_to_footwear']):
                owned |= part
        return owned

    source_owned=foot_owned(source_fragment,source_floor,source_footwear)
    candidate_owned=foot_owned(candidate_fragment,candidate_floor,candidate_footwear)
    mapped,validity=warp_values(source_owned.astype(np.float32),matrix,(cw,ch),order=0)
    mapped=(mapped>.5)&(validity>.5)&np.asarray(candidate_floor,bool)
    tolerance=int(params['reflection_boundary_tolerance_pixels'])
    allowed=binary_dilation(mapped,iterations=tolerance) if tolerance else mapped
    if candidate_proposals is None:
        components,count=label(candidate_owned,np.ones((3,3),bool))
        parts=[components==index for index in range(1,count+1)]
    else:
        # SAM proposals can overlap. Unioning them before qualification can
        # attach a false floor-material proposal to a real reflected leg.
        parts=[foot_owned(np.asarray(part,bool),candidate_floor,candidate_footwear)
               for part in candidate_proposals]
    for index,part in enumerate(parts):
        if not part.any():continue
        record={'candidate_component':index,'pixels':int(part.sum()),'qualified':False}
        report['components'].append(record)
        candidate_overlap=float((part&allowed).sum()/max(1,part.sum()))
        # Only source regions actually paired to this candidate component enter
        # its reciprocal overlap; separate legitimate leg components can match.
        source_near=mapped & binary_dilation(part,iterations=max(1,tolerance))
        source_overlap=float((source_near&part).sum()/max(1,source_near.sum()))
        record.update(candidate_overlap=candidate_overlap,source_overlap=source_overlap)
        selected=valid.copy()
        selected[valid] &= source_owned[sxy[valid,1],sxy[valid,0]] & part[cxy[valid,1],cxy[valid,0]]
        points=np.unique(np.round(cp[selected],3),axis=0)
        if len(points)>=3:
            try:hull=float(ConvexHull(points).volume)
            except Exception:hull=0.
        else:hull=0.
        fraction=hull/max(1,part.sum())
        record.update(verified_unique_inliers=int(len(points)),inlier_hull_fraction=fraction)
        if (candidate_overlap<params['minimum_candidate_overlap'] or
                source_overlap<params['minimum_source_overlap'] or
                len(points)<params['minimum_reflection_inliers'] or
                fraction<params['minimum_inlier_hull_fraction']):
            record['reason']='paired_mask_or_distributed_photographic_evidence_insufficient'
            continue
        # Preserve only mutually observed support plus the fixed segmentation
        # boundary tolerance, never the whole floor or a source-pose union.
        protected=binary_dilation(part,iterations=max(1,tolerance)) & allowed & np.asarray(candidate_floor,bool)
        protected &= ~np.asarray(candidate_footwear,bool)
        protection |= protected
        record.update(qualified=True,reason=None,protected_pixels=int(protected.sum()),
            source_inlier_points=sp[selected].tolist(),candidate_inlier_points=cp[selected].tolist(),
            median_reprojection_pixels=float(np.median(errors[selected])))
    report.update(qualified=bool(protection.any()),protected_pixels=int(protection.sum()),
        source_observed_fragment_pixels=int(source_owned.sum()),candidate_observed_fragment_pixels=int(candidate_owned.sum()),
        reason=None if protection.any() else 'no_qualified_observed_reflection')
    return protection,report


def registered_receiver_structure(mask, source_floor, source_reflection, mapping, params=PARAMS):
    """Counterevidence to deleting a broad learned 'leg' floor proposal.

    Distributed source-floor feature correspondences outside source reflection
    observations establish persistent receiver structure. They do not certify
    a clean floor or an exact human mask, so ambiguous proposals are withheld
    from cleanup rather than turned into a reflection layer.
    """
    shape=np.asarray(mask).shape;sh,sw=np.asarray(source_floor).shape;ch,cw=shape
    report={'cleanup_authorized':True,'state':'unmatched_learned_candidate_fragment',
        'verified_unique_inliers':0,'inlier_hull_fraction':0.}
    if mapping.get('supported') is not True:return report
    sp=np.asarray(mapping.get('inlier_source_points_scaled',[]),float)
    cp=np.asarray(mapping.get('inlier_candidate_points',[]),float)
    if sp.ndim!=2 or sp.shape[1:]!=(2,) or sp.shape!=cp.shape or not np.isfinite(sp).all() or not np.isfinite(cp).all():return report
    matrix=np.asarray(mapping.get('source_to_candidate_proxy'),float)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or np.linalg.det(matrix)<=0:return report
    sp=sp*np.asarray([sw/cw,sh/ch]);sxy,cxy=np.rint(sp).astype(int),np.rint(cp).astype(int)
    projected=np.column_stack([sp,np.ones(len(sp))])@matrix.T
    projected=projected[:,:2]/projected[:,2:]
    errors=np.linalg.norm(projected-cp,axis=1)
    valid=(sxy[:,0]>=0)&(sxy[:,0]<sw)&(sxy[:,1]>=0)&(sxy[:,1]<sh)&(
        cxy[:,0]>=0)&(cxy[:,0]<cw)&(cxy[:,1]>=0)&(cxy[:,1]<ch)&(
        errors<=params['maximum_local_reprojection_pixels'])
    selected=valid.copy()
    selected[valid] &= np.asarray(source_floor,bool)[sxy[valid,1],sxy[valid,0]] & (
        ~np.asarray(source_reflection,bool)[sxy[valid,1],sxy[valid,0]]) & np.asarray(mask,bool)[cxy[valid,1],cxy[valid,0]]
    points=np.unique(np.round(cp[selected],3),axis=0)
    if len(points)>=3:
        try:hull=float(ConvexHull(points).volume)
        except Exception:hull=0.
    else:hull=0.
    fraction=hull/max(1,np.asarray(mask,bool).sum())
    report.update(verified_unique_inliers=int(len(points)),inlier_hull_fraction=fraction)
    if len(points)>=params['minimum_reflection_inliers'] and fraction>=params['minimum_inlier_hull_fraction']:
        report.update(cleanup_authorized=False,state='persistent_source_receiver_structure_human_classification_ambiguous',
            source_inlier_points=sp[selected].tolist(),candidate_inlier_points=cp[selected].tolist())
    return report


def assemble_observed_floor_ownership(source_observation, candidate_observation,
                                     source_contact, candidate_contact, mapping, params=PARAMS):
    """CPU assembly shared by the GPU producer and saved-observation checks."""
    source_size=(source_observation or {}).get('size') or source_contact.get('support_surface',{}).get('size')
    candidate_size=candidate_observation['size']
    if not source_size:raise ValueError('Source floor observation sampling size unavailable')

    def observed(contact,key,size):
        asset=contact.get(key)
        if not asset:return np.zeros(size[::-1],bool)
        if file_identity(asset['path'])!=asset.get('sha256'):
            raise ValueError('Current paired floor observation mask changed: '+key)
        return read_spatial_mask(asset,size)

    def fragment(observation,size):
        asset=observation or {}
        path=asset.get('detected_fragments_path')
        if not path:return np.zeros(size[::-1],bool)
        if file_identity(path)!=asset.get('detected_fragments_sha256'):
            raise ValueError('Current learned reflection observation bytes changed')
        values=read_spatial_mask({'path':path},size)
        if list(Image.open(path).size)!=list(size):raise ValueError('Learned reflection observation grid changed')
        return values

    source_human=fragment(source_observation,source_size)
    candidate_human=fragment(candidate_observation,candidate_size)
    source_floor=observed(source_contact,'support_surface',source_size)
    candidate_floor=observed(candidate_contact,'support_surface',candidate_size)
    proposals=[item for item in candidate_observation['detections'] if item.get('used')]
    masks=[];bindings=[]
    for item in proposals:
        digest=file_identity(item['mask_path'])
        if item.get('mask_sha256',digest)!=digest:raise ValueError('Current candidate reflection proposal bytes changed')
        mask=read_spatial_mask({'path':item['mask_path']},candidate_size)
        if (item.get('model_profile')!='sam31_static_detector_v1'
                or item.get('prediction_representation')!='binary_instance_mask'
                or item['label'] not in params['queries']
                or item['instance_confidence']<params['minimum_instance_confidence']
                or item['box_fraction_of_roi']>=params['maximum_leg_box_fraction']):
            raise ValueError('Candidate reflection proposal no longer satisfies learned qualification')
        masks.append(mask);bindings.append(digest)
    union=np.logical_or.reduce(masks) if masks else np.zeros(candidate_human.shape,bool)
    if not np.array_equal(union,candidate_human):raise ValueError('Candidate reflection proposal union binding changed')
    protection,ownership=qualify_observed_reflections(source_human,candidate_human,
        source_floor,candidate_floor,observed(source_contact,'footwear',source_size),
        observed(candidate_contact,'footwear',candidate_size),mapping,params,candidate_proposals=masks)
    qualified={part['candidate_component'] for part in ownership['components'] if part['qualified']}
    cleanup=np.zeros(candidate_human.shape,bool);classifications=[]
    for index,mask in enumerate(masks):
        if index in qualified:
            structural={'cleanup_authorized':False,'state':'qualified_source_owned_reflection'}
        else:structural=registered_receiver_structure(mask,source_floor,source_human,mapping,params)
        classifications.append({'proposal_index':index,'mask_sha256':bindings[index],**structural})
        if structural['cleanup_authorized']:cleanup |= mask
    ownership['proposal_classification']=classifications
    ownership['observation_bindings']={'source_fragment_sha256':(source_observation or {}).get('detected_fragments_sha256'),
        'candidate_fragment_sha256':candidate_observation['detected_fragments_sha256'],
        'candidate_proposal_sha256':bindings,'qualification_params':params,
        'producer_sha256':file_identity(__file__)}
    return protection,cleanup,ownership


def protect_verified_reflections(removal, result, candidate_size, *, source_sha256, candidate_sha256):
    """Subtract only bound observed protection after every removal-mask union."""
    removal=np.asarray(removal,bool)
    asset=result.get('reflection_protection')
    report={'qualified':False,'protected_pixels':0,'changed_removal_pixels':0}
    if not asset:
        return removal.copy(),report
    contract=result.get('reflection_ownership') or {}
    binding=contract.get('binding') or {}
    if (contract.get('qualified') is not True or binding.get('source_sha256')!=source_sha256 or
            binding.get('candidate_sha256')!=candidate_sha256 or
            binding.get('protection_sha256')!=asset.get('sha256') or
            asset.get('coordinate_space')!='candidate_scene_proxy' or
            asset.get('role')!='paired_observed_floor_reflection_protection' or
            file_identity(asset['path'])!=asset.get('sha256')):
        raise ValueError('Observed floor reflection protection binding changed')
    if list(Image.open(asset['path']).size)!=asset.get('size'):
        raise ValueError('Observed reflection protection sampling grid changed')
    protected=read_spatial_mask(asset,candidate_size)
    if protected.shape!=removal.shape:
        raise ValueError('Observed reflection protection does not share removal coordinates')
    revised=removal & ~protected
    return revised,{'qualified':True,'protected_pixels':int(protected.sum()),
        'changed_removal_pixels':int((removal & protected).sum()),
        'validated_reflection_pixels_remaining_in_removal':int((revised&protected).sum()),
        'outside_protection_mask_unchanged':bool(np.array_equal(revised[~protected],removal[~protected])),
        'protection_sha256':asset['sha256']}


def classify_pollution_result(result):
    """Read the producer's explicit current state, never infer a historic upgrade."""
    if (not isinstance(result, dict) or result.get('analysis_status') not in ('unsupported','empty','partial')
            or result.get('clean_established') is not False
            or result.get('automatic_cleanup_applied') is not bool(result.get('mask'))
            or (result['analysis_status']=='partial') is not bool(result.get('mask'))):
        raise ValueError('Current explicit floor-cleanup state contract required')
    return {key:result[key] for key in ('analysis_status','clean_established','automatic_cleanup_applied')}


def read_spatial_mask(asset, size):
    path=Path(asset['path'])
    if path.suffix.lower()=='.npz':
        with np.load(path) as resource:
            values=resource['values']
        image=Image.fromarray(np.uint8(values>.5)*255)
    else:
        image=Image.open(path).convert('L')
    return np.asarray(image.resize(tuple(size),Image.Resampling.NEAREST))>0


def floor_search_roi(contact, size, params=PARAMS):
    """Use observed shoes/sole adjacency; absent evidence is a normal result."""
    if not contact.get('points'):
        return {'supported':False,'reason':'candidate_sole_contact_unavailable'}
    if not all(contact.get(key) for key in ('footwear','support_surface','contact_support')):
        return {'supported':False,'reason':'candidate_floor_or_footwear_unavailable'}
    shoe=read_spatial_mask(contact['footwear'],size)
    floor=read_spatial_mask(contact['support_surface'],size)
    if not shoe.any():
        return {'supported':False,'reason':'candidate_footwear_unavailable'}
    if not floor.any():
        return {'supported':False,'reason':'candidate_floor_unavailable'}
    components,count=label(shoe,structure=np.ones((3,3),np.uint8))
    areas=np.bincount(components.ravel(),minlength=count+1);areas[0]=0
    main=components==int(areas.argmax())
    fy,fx=np.nonzero(main);foot_center=np.asarray([fx.mean(),fy.mean()])
    point=np.asarray(max(contact['points'],key=lambda entry:entry['pixels'])['point'],np.float64)
    contact_size=contact['contact_support'].get('size',size)
    point *= np.asarray(size)/np.asarray(contact_size)
    sole=read_spatial_mask(contact['contact_support'],size)
    cy,cx=np.nonzero(sole)
    direction=point-foot_center
    if len(cx)>=2:
        eigenvalues,vectors=np.linalg.eigh(np.cov(np.column_stack([cx,cy]).T))
        if eigenvalues[-1]>eigenvalues[0]*1.1:
            tangent=vectors[:,-1];normal=np.asarray([-tangent[1],tangent[0]])
            direction=normal if np.dot(normal,direction)>=0 else -normal
    length=float(np.linalg.norm(direction))
    if length<1e-6:
        return {'supported':False,'reason':'candidate_sole_direction_unavailable'}
    direction /= length
    y,x=np.mgrid[:size[1],:size[0]]
    projection=(x-point[0])*direction[0]+(y-point[1])*direction[1]
    distance=np.hypot(x-point[0],y-point[1])
    extent=float(max(fx.max()-fx.min()+1,fy.max()-fy.min()+1))
    support=floor&~shoe&(projection>=-extent*params['sole_projection_margin'])&(
        distance<extent*params['footwear_extent_radius'])
    ys,xs=np.nonzero(support)
    if not len(xs):
        return {'supported':False,'reason':'candidate_proximal_floor_unavailable'}
    return {'supported':True,'box':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
        'support':support,'floor':floor,'footwear':shoe,'provenance':{
            'contact_point':point.tolist(),'observed_shoe_center':foot_center.tolist(),
            'direction_toward_observed_support':direction.tolist(),
            'radius_relative_to_observed_footwear_extent':params['footwear_extent_radius'],
            'scope':'proximal_foot_connected_floor; not_complete_floor_or_reflection_segmentation'}}


def prepare_candidate_floor_pollution(root,scene,*,registry_path=None):
    """Return a local candidate-only removal mask; no donor or LaMa dependency."""
    started=time.perf_counter();root=Path(root).resolve()
    proxy=scene['proxies']['candidate'];contact=scene.get('contact',{}).get('candidate',{})
    size=proxy['size'];original_size=scene['candidate']['oriented_size']
    base={'schema_version':'candidate_floor_pollution.sam31.v1','recipe':RECIPE,
        'supported':False,'mask':None,'size':list(size),'candidate_original_size':list(original_size),
        'coordinate_space':'candidate_scene_proxy','cache_hit':False,'gpt_used':False,
        'image_generation_used':False,'source_donor_required':False,'lama_used':False,
        'analysis_status':'unsupported','clean_established':False,
        'automatic_cleanup_applied':False}
    roi=floor_search_roi(contact,size)
    if not roi['supported']:
        return {**base,'reason':roi['reason'],'elapsed_seconds':round(time.perf_counter()-started,3),
                'provenance':{'candidate_sha256':scene['candidate']['sha256'],
                    'source_reflection_geometry_unioned':False,'source_registration_required':False}}
    paths=scene_runtime_paths(root,registry_path=registry_path)
    def bound_contact(role):
        observed=scene.get('contact',{}).get(role,{})
        return {'points':observed.get('points',[]),**{key:{**observed[key],
            'sha256':file_identity(observed[key]['path'])} for key in ('footwear','support_surface','contact_support')
            if observed.get(key)}}
    source_proxy=scene['proxies']['source']
    inputs={'candidate_proxy':{'path':proxy['path'],'size':list(size),'sha256':file_identity(proxy['path'])},
        'candidate_sha256':scene['candidate']['sha256'],'candidate_original_size':list(original_size),
        'contact':bound_contact('candidate'),
        'paired_source':{'candidate_proxy':{'path':source_proxy['path'],'size':list(source_proxy['size']),
            'sha256':file_identity(source_proxy['path'])},'candidate_sha256':scene['source']['sha256'],
            'candidate_original_size':scene['source']['oriented_size'],'contact':bound_contact('source'),
            'observation_role':'source','params':PARAMS},
        'background_mapping':scene['background_mapping'],
        'params':PARAMS}
    identity={'recipe':RECIPE,'inputs':inputs,'models':scene_model_identity(paths['models'],profile=paths['profile']),
        'worker_sha256':file_identity(code_path('sam3_floor_worker.py')),
        'helper_sha256':file_identity(__file__)}
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    directory=root/'outputs/cache/floor-pollution'/identity_hash
    result_path=directory/'analysis.json'
    if result_path.exists():
        saved=json.loads(result_path.read_text(encoding='utf-8'))
        if saved.get('identity')!=identity or saved.get('schema_version')!='candidate_floor_pollution.sam31.v1':
            raise ValueError('Outdated floor cleanup cache is unavailable to the current producer')
        return {**saved, **classify_pollution_result(saved),
                'cache_hit':True,'elapsed_seconds':round(time.perf_counter()-started,3)}
    directory.mkdir(parents=True,exist_ok=True)
    request_path=directory/'candidate-input.json'
    request_path.write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    log_path=directory/'worker.log'
    with log_path.open('w',encoding='utf-8') as log:
        completed=subprocess.run([str(paths['python']),'-X','utf8',
            str(code_path('sam3_floor_worker.py')),'--candidate-input',str(request_path),
            '--output-dir',str(directory),'--model-root',str(paths['models'])],
            stdout=log,stderr=log,timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
            env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    if completed.returncode:
        raise RuntimeError(f'Candidate floor detector failed; see {log_path}')
    detected=json.loads((directory/'candidate-floor-pollution.json').read_text(encoding='utf-8'))
    result={**base,**detected,**classify_pollution_result(detected),'identity':identity,
        'analysis_directory':str(directory.resolve()),
        'elapsed_seconds':round(time.perf_counter()-started,3)}
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result
