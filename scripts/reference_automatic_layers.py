"""Automatic local reference fields used by production colour and scene prep.

Correspondence is an independently bound observation. This module samples only
reference low frequencies and computes corrections in the unchanged source
canvas; no reference person pixels or texture are used as output pixels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from reference_colour_fit import REGIONS, SKIN
from asset_identity import sha256_file

POLICY = 'automatic_local_reference_fields.v1'
CONFIG = {'lowpass_region_fraction': 1/24, 'minimum_lowpass_support': .1,
          'maximum_colour_delta': .04, 'maximum_scene_colour_delta': .025,
          'minimum_region_pixels': 24, 'background_ring_fraction': [.006, .04],
          'observation_filter_iterations': 3, 'observation_edge_scale_floor': .012,
          'observation_edge_scale_ceiling': .06, 'observation_edge_cutoff_sigmas': 3.}
QUALIFIED_FACE_CORE = .99
_NUMBA_DOMAIN_SCAN = None


def sha(path):
    return sha256_file(path)


CORRESPONDENCE_RECORD = 'automatic_local_correspondence_record.v2'
MAP_REPRESENTATION = 'source_proxy_to_original_reference_pixel_centres_ref_x_ref_y_confidence_valid'


def _receipt_provenance(value):
    """Preserve evidence in frozen snapshots without live cache locators."""
    if isinstance(value, dict):
        return {key: _receipt_provenance(item) for key, item in value.items()
                if key not in {'path', 'manifest', 'output'} and not key.endswith(('_path', '_file'))}
    if isinstance(value, list):
        return [_receipt_provenance(item) for item in value]
    return value


def _correspondence_binding(root, source, reference, size, source_labels, reference_labels, faces,
                            *, algorithm_profile=None):
    from reference_local_correspondence import (array_sha, canonical_sha, _cached_rgb,
                                               local_correspondence_identity)
    size = list(size)
    if (len(size) != 2 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in size)
            or np.shape(source_labels) != tuple(reversed(size))):
        raise ValueError('Local correspondence source labels do not match the proxy canvas')
    for values in (source_labels, reference_labels):
        values = np.asarray(values)
        if (values.ndim != 2 or not values.size or not np.issubdtype(values.dtype, np.integer)
                or values.min() < 0):
            raise ValueError('Local correspondence requires exact integer semantic observations')
    identity = local_correspondence_identity(root, algorithm_profile=algorithm_profile) if algorithm_profile else local_correspondence_identity(root)
    cached = {role: _cached_rgb(root, {'sha256': digest}) for role, digest in
              (('source', source), ('reference', reference))}
    return {'source_sha256': source, 'reference_sha256': reference,
        'source_labels_sha256': array_sha(source_labels),
        'reference_labels_sha256': array_sha(reference_labels), 'proxy_size': size,
        'source_faces_sha256': canonical_sha(faces['source']),
        'reference_faces_sha256': canonical_sha(faces['reference']),
        'producer_identity_sha256': canonical_sha(identity),
        'cached_rgb_sha256': {role: value[2] if value else None for role, value in cached.items()}}


def _validate_correspondence_receipt(receipt, asset, expected, source_labels, reference_labels):
    from reference_local_correspondence import canonical_sha, validate_local_correspondence
    if (receipt.get('binding') != expected or receipt.get('proxy_size') != expected['proxy_size']
            or receipt.get('reference_size') != list(reversed(reference_labels.shape))
            or receipt.get('coordinate_space') != 'source_proxy'
            or receipt.get('reference_coordinate_space') != 'original_oriented_reference_pixels'
            or receipt.get('confidence_is_spatial_evidence_not_effect_dose') is not True
            or any(receipt.get(key) is not False for key in
                   ('person_geometry_warped', 'reference_texture_imported', 'background_H_used'))):
        raise ValueError('Local correspondence receipt input/domain binding mismatch')
    # The locator and its content must be the same asset the producer certified.
    if (canonical_sha(receipt.get('asset')) != canonical_sha(asset)
            or asset.get('representation') != MAP_REPRESENTATION):
        raise ValueError('Local correspondence receipt certifies a different map asset')
    maps = validate_local_correspondence(receipt, binding=expected)
    if any(maps[key].dtype != np.float32 for key in ('ref_x', 'ref_y', 'confidence')):
        raise ValueError('Local correspondence observations require float32 coordinates/confidence')
    with np.load(asset['path'], allow_pickle=False) as resource:
        owners = resource['material_labels']
    if (owners.dtype != np.int32 or owners.shape != source_labels.shape
            or not np.array_equal(owners, source_labels)):
        raise ValueError('Local correspondence map semantic owners changed')
    return maps


def pack_correspondence(result, run, size, *, root, source_labels, reference_labels,
                        source_faces=None, reference_faces=None, algorithm_profile=None):
    """Bind the actual producer evidence before admitting it to B/S preparation."""
    from reference_local_correspondence import canonical_sha, _face_observation, algorithm_profile_for_run
    for role in ('source', 'reference'):
        if sha(run[role]['path']) != run[role]['sha256']:
            raise ValueError('Local correspondence photograph bytes changed: '+role)
    receipt = json.loads(Path(result['receipt_path']).read_text(encoding='utf8'))
    if canonical_sha(receipt) != canonical_sha(result['receipt']):
        raise ValueError('Local correspondence returned receipt differs from its file')
    faces = {'source': _face_observation(source_faces), 'reference': _face_observation(reference_faces)}
    algorithm_profile = algorithm_profile if algorithm_profile is not None else algorithm_profile_for_run(run)
    expected = _correspondence_binding(root, run['source']['sha256'], run['reference']['sha256'],
        size, source_labels, reference_labels, faces, algorithm_profile=algorithm_profile)
    _validate_correspondence_receipt(receipt, result['asset'], expected, source_labels, reference_labels)
    if run['source'].get('oriented_size') and receipt.get('source_size') != run['source']['oriented_size']:
        raise ValueError('Local correspondence native source canvas changed')
    body = _receipt_provenance(receipt)
    return {'schema': CORRESPONDENCE_RECORD, 'algorithm_profile': algorithm_profile,
        'receipt_path': result['receipt_path'],
        'receipt_sha256': sha(result['receipt_path']), 'receipt': body,
        'receipt_content_sha256': canonical_sha(body), 'asset': dict(result['asset']),
        'inputs': expected, 'face_observations': faces, 'regions': body.get('regions', {})}


def load_maps(record, source_shape, reference_shape, parameters, *, root,
              source_labels, reference_labels, source_faces=None, reference_faces=None):
    """Preparation-only verification; frozen native replay never calls this loader."""
    from reference_local_correspondence import canonical_sha, _face_observation, DINO3_L_PROFILE
    if (record.get('schema') != CORRESPONDENCE_RECORD
            or np.shape(source_labels) != tuple(source_shape)
            or np.shape(reference_labels) != tuple(reference_shape)):
        raise ValueError('Local correspondence record or semantic canvas mismatch')
    faces = record['face_observations']
    for role, supplied in (('source', source_faces), ('reference', reference_faces)):
        if supplied is not None and _face_observation(supplied) != faces[role]:
            raise ValueError('Local correspondence current face observation changed: '+role)
    expected = _correspondence_binding(root, parameters['binding']['source'],
        parameters['binding']['reference'], [source_shape[1], source_shape[0]],
        source_labels, reference_labels, faces, algorithm_profile=record.get('algorithm_profile', DINO3_L_PROFILE))
    if record['inputs'] != expected or sha(record['receipt_path']) != record['receipt_sha256']:
        raise ValueError('Local correspondence input or receipt binding changed')
    receipt = json.loads(Path(record['receipt_path']).read_text(encoding='utf8'))
    body = _receipt_provenance(receipt)
    if (canonical_sha(body) != record['receipt_content_sha256'] or body != record['receipt']
            or record.get('regions') != body.get('regions', {})):
        raise ValueError('Local correspondence retained evidence changed')
    return _validate_correspondence_receipt(receipt, record['asset'], expected,
                                          source_labels, reference_labels)


def sample_reference(values, maps, *, nearest=False):
    import cv2
    values = np.asarray(values)
    if values.ndim not in (2,3) or not np.isfinite(values).all():
        raise ValueError('Reference sampling requires finite scalar or channel observations')
    return cv2.remap(values.astype(np.float32), maps['ref_x'].astype(np.float32),
        maps['ref_y'].astype(np.float32), cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def lowpass(values, mask):
    """Filter observations inside one owner using that image's OKLab edges.

    Domain-transform recursion smooths fine variation while stopping at strong
    colour boundaries. A Gaussian of the owner alone could pre-mix black and
    white costume panels before the correction solver sees their observations.
    This filters only observations, never the output person's RGB or texture.
    """
    sigma = max(1., np.sqrt(float(mask.sum()))*CONFIG['lowpass_region_fraction'])
    result=np.zeros_like(values);support=np.zeros(mask.shape,np.float32)
    if not mask.any():return result,support,sigma
    yy,xx=np.nonzero(mask)
    crop=(slice(int(yy.min()),int(yy.max())+1),slice(int(xx.min()),int(xx.max())+1))
    owner=mask[crop];guide=values[crop].astype(np.float32,copy=False)
    horizontal=np.linalg.norm(np.diff(guide,axis=1),axis=-1)
    vertical=np.linalg.norm(np.diff(guide,axis=0),axis=-1)
    same_h=owner[:,1:]&owner[:,:-1];same_v=owner[1:]&owner[:-1]
    observed=np.concatenate((horizontal[same_h],vertical[same_v]))
    edge_scale=float(np.clip(np.quantile(observed,.65) if observed.size else 0,
        CONFIG['observation_edge_scale_floor'],CONFIG['observation_edge_scale_ceiling']))
    # The range scale and hard cutoff match the correction solver's guide.
    # All images use the same policy; source and reference retain their own
    # edge positions, dimensions and spatial sigma before correspondence.
    horizontal=1+(sigma/edge_scale)*horizontal
    vertical=1+(sigma/edge_scale)*vertical
    cutoff=1+CONFIG['observation_edge_cutoff_sigmas']*sigma
    same_h&=horizontal<=cutoff;same_v&=vertical<=cutoff
    signal=np.concatenate((guide*owner[...,None],owner[...,None].astype(np.float32)),axis=-1)
    iterations=CONFIG['observation_filter_iterations']
    for iteration in range(iterations):
        spatial_sigma=sigma*np.sqrt(3)*2**(iterations-iteration-1)/np.sqrt(4**iterations-1)
        wh=np.exp(-np.sqrt(2)*horizontal/spatial_sigma)*same_h
        wv=np.exp(-np.sqrt(2)*vertical/spatial_sigma)*same_v
        _domain_transform_scan(signal, wh, wv)
    support[crop]=signal[...,-1]
    result[crop]=signal[...,:-1]/np.maximum(signal[...,-1,None],1e-6)
    return result,support,sigma


def _domain_transform_scan(signal, horizontal, vertical):
    """Run one recursive domain-transform pass in compiled native loops.

    The recurrence is intentionally not replaced with a Gaussian: each scan
    depends on its preceding pixel and must stop at semantic/OKLab edges.
    Numba parallelizes the independent rows/columns while preserving the
    four directional passes and float32 working buffer.  Desktop runtimes
    without Numba retain the exact reference recurrence as a fallback.
    """
    global _NUMBA_DOMAIN_SCAN
    if _NUMBA_DOMAIN_SCAN is None:
        try:
            from numba import njit, prange
        except ImportError:
            _NUMBA_DOMAIN_SCAN = False
        else:
            @njit(parallel=True, nogil=True, cache=False)
            def scan(values, wh, wv):
                height, width, channels = values.shape
                for y in prange(height):
                    for x in range(1, width):
                        weight = wh[y, x-1]
                        for channel in range(channels):
                            values[y, x, channel] += weight*(values[y, x-1, channel]-values[y, x, channel])
                    for x in range(width-2, -1, -1):
                        weight = wh[y, x]
                        for channel in range(channels):
                            values[y, x, channel] += weight*(values[y, x+1, channel]-values[y, x, channel])
                for x in prange(width):
                    for y in range(1, height):
                        weight = wv[y-1, x]
                        for channel in range(channels):
                            values[y, x, channel] += weight*(values[y-1, x, channel]-values[y, x, channel])
                    for y in range(height-2, -1, -1):
                        weight = wv[y, x]
                        for channel in range(channels):
                            values[y, x, channel] += weight*(values[y+1, x, channel]-values[y, x, channel])
            _NUMBA_DOMAIN_SCAN = scan
    if _NUMBA_DOMAIN_SCAN is False:
        for x in range(1, signal.shape[1]):
            signal[:,x]+=horizontal[:,x-1,None]*(signal[:,x-1]-signal[:,x])
        for x in range(signal.shape[1]-2,-1,-1):
            signal[:,x]+=horizontal[:,x,None]*(signal[:,x+1]-signal[:,x])
        for y in range(1,signal.shape[0]):
            signal[y]+=vertical[y-1,:,None]*(signal[y-1]-signal[y])
        for y in range(signal.shape[0]-2,-1,-1):
            signal[y]+=vertical[y,:,None]*(signal[y+1]-signal[y])
        return signal
    _NUMBA_DOMAIN_SCAN(signal, horizontal, vertical)
    return signal


def _observations(source_lab, reference_lab, sl, rl, alpha, maps, ids, excluded=None):
    sm = np.isin(sl,ids)&(alpha >= .995)
    if excluded is not None: sm &= ~excluded
    source_low=np.zeros_like(source_lab);target=np.zeros_like(source_lab)
    ss=np.zeros(sl.shape,np.float32);target_support=np.zeros(sl.shape,np.float32)
    mapped_labels=sample_reference(rl,maps,nearest=True).astype(np.int32)
    owner_scales={}
    compatible=mapped_labels==sl
    c = maps['confidence']*maps['valid']*sm*compatible
    # Filtering cannot create missing same-person correspondence. Unsupported
    # regions are rejected by both consumers before observing the arrays or
    # their scale details, so avoid their two independent material filters.
    if not np.any(c):
        return sm,source_low,target,c,{}
    # A report group can include several material owners. Filtering the union
    # before enforcing ownership would contaminate a neighbour's observation.
    for owner in ids:
        source_owner=sm&(sl==owner);reference_owner=rl==owner
        if not source_owner.any() or not reference_owner.any():continue
        current,support,sigma=lowpass(source_lab,source_owner)
        reference,rs,ref_sigma=lowpass(reference_lab,reference_owner)
        sampled_support=sample_reference(rs,maps)
        # Bilinear lookup near an owner edge must not mix the owner's colour
        # with zero outside its domain. Sample numerator and support together.
        sampled=sample_reference(reference*rs[...,None],maps)/np.maximum(sampled_support[...,None],1e-6)
        source_low[source_owner]=current[source_owner];ss[source_owner]=support[source_owner]
        target[source_owner]=sampled[source_owner];target_support[source_owner]=sampled_support[source_owner]
        owner_scales[str(owner)]={'source_sigma':sigma,'reference_sigma':ref_sigma}
    c *= (ss >= CONFIG['minimum_lowpass_support'])&(target_support >= CONFIG['minimum_lowpass_support'])
    return sm,source_low,target,c,{'material_owner_scales':owner_scales,
        'preblur_ownership':'each_exact_semantic_owner; no_region_union_filtering',
        'observation_filter':'normalized_three_iteration_domain_transform; independent_source_reference_OKLab_guides; strong_edge_cutoff_3sigmas'}


def _geometry(record, name, mask, confidence):
    eligible = int(np.count_nonzero(confidence))
    return {'status': 'partial' if eligible else 'unsupported', 'allowed': bool(eligible),
            'reason': 'independently_validated_local_person_correspondence' if eligible else
                'no_valid_local_person_correspondence',
            'source_pixels': int(mask.sum()), 'accepted_local_pixels': eligible,
            'source_support_fraction': eligible/max(int(mask.sum()),1),
            'background_H_is_person_correspondence': False,
            'dense_correspondence': False, 'local_correspondence': True,
            'local_observation': record.get('regions',{}).get(name,{})}


def _hair_face_core(support, labels):
    if support is None:
        return None
    support = np.asarray(support)
    if (support.shape != labels.shape or not np.isfinite(support).all()
            or support.min() < 0 or support.max() > 1):
        raise ValueError('Hair observation veto requires source-coordinate visible-face support')
    return (support >= QUALIFIED_FACE_CORE) & np.isin(labels, REGIONS['hair'])


def _hair_veto_report(core):
    return {'fully_qualified_core_threshold':QUALIFIED_FACE_CORE,
        'blocked_observation_and_solver_pixels':int(core.sum()) if core is not None else 0,
        'saved_semantic_labels_changed':False,'soft_dose_consumption':'once_after_coordinate_sampling'}


def colour_field(source_lab,reference_lab,source_labels,reference_labels,alpha,maps,record,
                 *,excluded=None,visible_face_support=None,preserve_hair=False,
                 allow_relaxed_continuation=False):
    from reference_adaptive_fusion import solve_field
    field = np.zeros((*source_labels.shape,3),np.float32)
    reports = {}
    face_core = _hair_face_core(visible_face_support, source_labels)
    regions = {**REGIONS,'lips':[24,25]}
    for name,ids in regions.items():
        if name == 'hair' and preserve_hair:
            reports[name] = {'status':'source_hair_preserved','source_hair_preserved':True,
                'eligible_pixels':0,'observed_pixels':0,'maximum_applied_chroma_shift':0.,
                'reference_colour_mode':'source_preserved_no_dedicated_hair_field'}
            continue
        blocked = face_core if name == 'hair' else None
        observation_excluded = excluded if blocked is None else blocked if excluded is None else excluded | blocked
        sm,current,target,c,scales = _observations(source_lab,reference_lab,source_labels,
            reference_labels,alpha,maps,ids,observation_excluded)
        geometry = _geometry(record,name,sm,c)
        if int(sm.sum()) < CONFIG['minimum_region_pixels'] or not np.any(c):
            reports[name] = {'eligible_pixels':0,'rejected':geometry['reason'],
                             'correspondence_support':geometry}
            if name == 'hair': reports[name]['visible_face_veto'] = _hair_veto_report(blocked)
            continue
        # Labels still partition black/white and feature boundaries through
        # the original guide; excluded features are distinct solver domains.
        labels = source_labels.astype(np.int32).copy()
        if excluded is not None: labels[excluded] = 1000
        if blocked is not None: labels[blocked] = 1001
        delta,solved = solve_field(target[...,1:]-current[...,1:],c,labels,source_lab,
                                  maximum_norm=CONFIG['maximum_colour_delta'],continuation_domain=sm,
                                  allow_relaxed_continuation=allow_relaxed_continuation)
        delta[~sm] = 0
        field[...,:2] += delta
        active = np.linalg.norm(delta,axis=-1)>0
        field[...,2] = np.maximum(field[...,2],np.where(active,np.maximum(c,.001),0))
        reports[name] = {**scales,'eligible_pixels':int(active.sum()),
            'observed_pixels':int(np.count_nonzero(c)),'correspondence_support':geometry,
            'maximum_applied_chroma_shift':float(np.linalg.norm(delta,axis=-1).max()),
            'reference_colour_mode':'local_correspondence_multiscale_evidence_solve',
            'source_lightness_retained':True,'automatic_solver':solved}
        if name == 'hair':
            reports[name]['visible_face_veto'] = _hair_veto_report(blocked)
    # Regions are disjoint source labels, so there is no dose stacking.
    if np.linalg.norm(field[...,:2],axis=-1).max() > CONFIG['maximum_colour_delta']+1e-6:
        raise ValueError('Automatic material colour fields exceeded their shared budget')
    return field,reports


def background_relation(source_lab,reference_lab,source_labels,reference_labels,alpha):
    # Background is already part of the frozen pre-S composite. Compare
    # exterior rings separately on both canvases; never include their people.
    scale=max(source_labels.shape)
    distance=distance_transform_edt(alpha < .01)
    lower,upper=(scale*x for x in CONFIG['background_ring_fraction'])
    sm=(alpha < .01)&(source_labels == 0)&(distance >= lower)&(distance <= upper)
    rp=reference_labels > 0
    rd=distance_transform_edt(~rp)
    rs=max(reference_labels.shape)
    rm=(reference_labels == 0)&(rd >= rs*CONFIG['background_ring_fraction'][0])&(
        rd <= rs*CONFIG['background_ring_fraction'][1])
    if min(int(sm.sum()),int(rm.sum())) < CONFIG['minimum_region_pixels']:
        return 0.,{'status':'unsupported','reason':'insufficient_background_ring_observations'}
    s=float(np.median(3*np.log2(np.maximum(source_lab[...,0][sm],.03))))
    r=float(np.median(3*np.log2(np.maximum(reference_lab[...,0][rm],.03))))
    return s-r,{'status':'partial','source_pixels':int(sm.sum()),'reference_pixels':int(rm.sum()),
        'source_background_log_L':s,'reference_background_log_L':r,
        'foreground_relation_offset_ev':s-r,'physical_light_direction_inferred':False}


def scene_field(source_lab,reference_lab,source_labels,reference_labels,alpha,protection,
                maps,record,*,visible_face_support=None,physical_domains=None,preserve_hair=False,
                allow_relaxed_continuation=False):
    from reference_adaptive_fusion import solve_field
    from reference_scene_balance import REGION_LIMITS
    field=np.zeros((*source_labels.shape,4),np.float32);reports={}
    joint_observations={}
    face_core = _hair_face_core(visible_face_support, source_labels)
    offset,background=background_relation(source_lab,reference_lab,source_labels,reference_labels,alpha)
    for name,ids in REGIONS.items():
        if name in {'bag','shoes'}: continue
        if name == 'hair' and preserve_hair:
            reports['head_hair'] = {'status':'source_hair_preserved','source_hair_preserved':True,
                'eligible_pixels':0,'observed_pixels':0,'spatial_colour_dose':0.,'lightness_dose':0.,
                'reference_mode':'source_preserved_no_dedicated_hair_field'}
            continue
        blocked = face_core if name == 'hair' else None
        sm,current,target,c,scales=_observations(source_lab,reference_lab,source_labels,
            reference_labels,alpha,maps,ids,blocked)
        geometry=_geometry(record,name,sm,c)
        if physical_domains is not None and name in {'left_hand','right_hand','left_arm','right_arm'} and np.any(c):
            joint_observations[name]=(sm,c,3*np.log2(np.maximum(target[...,0],.03)/np.maximum(current[...,0],.03))+offset)
        key='head_hair' if name=='hair' else name
        if int(sm.sum()) < CONFIG['minimum_region_pixels'] or not np.any(c):
            reports[key]={'eligible_pixels':0,'rejected':geometry['reason'],
                          'correspondence_support':geometry,'spatial_colour_dose':0.,'lightness_dose':0.}
            if name == 'hair': reports[key]['visible_face_veto'] = _hair_veto_report(blocked)
            continue
        desired=3*np.log2(np.maximum(target[...,0],.03)/np.maximum(current[...,0],.03))+offset
        limits=REGION_LIMITS['hair' if name=='hair' else 'clothing' if name=='clothing' else 'skin']
        dose=c.copy()
        solver_labels=source_labels
        if blocked is not None:
            solver_labels=source_labels.astype(np.int32).copy()
            solver_labels[blocked]=1001
        ev,solved=solve_field(desired,dose,solver_labels,source_lab,
            maximum_norm=max(abs(x) for x in limits),minimum_value=limits[0],maximum_value=limits[1],
            continuation_domain=sm, allow_relaxed_continuation=allow_relaxed_continuation)
        if name not in {'hair','clothing'}: ev *= 1-.85*protection
        ev[~sm]=0
        # After colour fusion, S changes appearance lightness once. The palette
        # and B have already used reference chroma; do not tint a second time.
        field[...,0] += ev
        active=ev != 0
        field[...,3]=np.maximum(field[...,3],np.where(active,np.maximum(dose,.001),0))
        reports[key]={**scales,'eligible_pixels':int(active.sum()),'observed_pixels':int(np.count_nonzero(c)),
            'correspondence_support':geometry,'applied_ev_range':[float(ev.min()),float(ev.max())],
            'maximum_ab':0.,'warm_colour_enabled':False,'preparation_ev_limits':limits,
            'protected_skin_L_dose':.15 if name not in {'hair','clothing'} else None,
            'reference_mode':'local_reference_lightness_and_observed_background_relationship',
            'automatic_solver':solved}
        if name == 'hair':
            reports[key]['visible_face_veto'] = _hair_veto_report(blocked)
    if physical_domains is not None:
        from source_skin_domains import POLICY as physical_policy,reference_evidence
        owners=physical_domains['owners'];guide=physical_domains['source_lab']
        if (owners.shape!=source_labels.shape or not np.issubdtype(owners.dtype,np.integer)
                or guide.shape!=source_lab.shape or not np.isfinite(guide).all()
                or physical_domains['report'].get('policy')!=physical_policy):
            raise ValueError('Scene physical continuation requires its original-source fine domains')
        joined=[]
        for group in physical_domains['groups']:
            domain=owners==group['owner']
            confidence=np.zeros(source_labels.shape,np.float32)
            desired=np.zeros(source_labels.shape,np.float32)
            for sm,c,r in joint_observations.values():
                selected=sm&domain
                confidence[selected]=c[selected];desired[selected]=r[selected]
            observed=int(np.count_nonzero(confidence))
            evidence=reference_evidence(confidence,domain)
            if not evidence['qualified']:
                # Do not broaden a one-point/cluster estimate to a whole arm.
                # Keep the original field and separate anatomical sampling
                # owners, while still isolating disconnected fine components.
                from scipy.ndimage import label as component_labels
                sampling=physical_domains.setdefault('sampling_owners',owners.copy())
                next_owner=int(sampling.max())+1
                for raw in group['source_classes']:
                    cc,count=component_labels(domain&(source_labels==raw))
                    selected=cc>0;sampling[selected]=cc[selected]+next_owner;next_owner+=count+1
                joined.append({**group,'status':'unsupported','observed_pixels':observed,
                    'accepted_joint_observed_pixels':0,'reference_evidence':evidence,
                    'reason':evidence['reason']+'; existing_field_and_anatomical_separation_retained'})
                continue
            limits=REGION_LIMITS['skin']
            ev,solved=solve_field(desired,confidence,owners,guide,
                maximum_norm=max(abs(v) for v in limits),minimum_value=limits[0],maximum_value=limits[1],
                continuation_domain=domain, allow_relaxed_continuation=allow_relaxed_continuation)
            ev*=1-.85*protection
            field[...,0][domain]=ev[domain]
            field[...,3][domain]=np.where(ev[domain]!=0,np.maximum(confidence[domain],.001),0)
            observed_by_class={str(raw):int(np.count_nonzero(confidence*(source_labels==raw)))
                for raw in group['source_classes']}
            estimated_by_class={str(raw):int(np.count_nonzero(domain&(source_labels==raw)&(confidence==0)&(ev!=0)))
                for raw in group['source_classes']}
            joined.append({**group,'status':'source_proven_illumination_continuation',
                'observed_pixels':observed,'reference_observed_by_source_class':observed_by_class,
                'reference_evidence':evidence,
                'estimated_without_direct_reference_by_source_class':estimated_by_class,
                'reference_correspondence_extended':False,'source_RGB_smoothed':False,
                'applied_EV_range':[float(ev[domain].min()),float(ev[domain].max())],
                'automatic_solver':solved})
            for name,ids in REGIONS.items():
                if name not in {'left_hand','right_hand','left_arm','right_arm'}:continue
                estimated=int(np.count_nonzero(domain&np.isin(source_labels,ids)&(confidence==0)&(ev!=0)))
                if estimated:
                    reports[name]['estimated_illumination_pixels']=reports[name].get('estimated_illumination_pixels',0)+estimated
                    reports[name]['reference_observation_status']='unchanged; estimate_is_not_new_correspondence'
        reports['physical_hand_arm_continuation']={'policy':physical_policy,
            'source_domain_evidence':physical_domains['report'],'groups':joined,
            'guarded_source_material_boundaries_respected':True,'legacy_other_regions_unchanged':True,
            'observation_confidence_extended':False,'visual_acceptance':False}
    return field,reports,background
