"""Bounded CPU backprojection to a qualified compact background observation."""
from __future__ import annotations

import copy
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from scene_stage_audit import (DOMAIN, array, bound, file_record, metrics, read, sha, write, _background)

SCHEMA = 'background_consistency.v1'
CONFIG = {'iterations': 3, 'sigma_compact': 1., 'maximum_linear_RGB_delta': .025,
          'minimum_relative_RMSE_improvement': .1, 'minimum_RMSE': 2e-4}


def bounded_update(current, delta, coverage):
    """A shared continuous RGB dose retains associated SDR and exact zeros."""
    dose = np.ones(current.shape[:2], np.float32)
    for channel in range(3):
        change = delta[..., channel]
        limit = np.full(change.shape, np.inf, np.float32)
        np.divide(coverage-current[..., channel], change, out=limit, where=change > 0)
        negative = change < 0
        np.divide(-current[..., channel], change, out=limit, where=negative)
        dose = np.minimum(dose, limit)
    return current+delta*np.clip(dose, 0, 1)[..., None]


def prepare_consistent_background(root, run, plan, output, *, audit, cancel=None):
    """Return new owned P only after measured improvement; do not render here."""
    started = time.perf_counter(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if audit.get('schema') != 'scene_stage_audit.v1' or audit.get('status') != 'repair_eligible':
        return {'ok': True, 'design_reconstruction_plan': copy.deepcopy(plan), 'changed': False,
                'receipt': {'schema': SCHEMA, 'status': 'audited_no_repair', 'reason': audit.get('reason')}}
    stored = read(bound(audit['receipt_asset']))
    if stored != {key: value for key, value in audit.items() if key != 'receipt_asset'}:
        raise ValueError('Scene audit differs from its actual saved receipt')
    assets = _background(plan)['assets']
    for role, key in (('input_background','background'), ('validity','validity')):
        if audit[role]['sha256'] != assets[key]['sha256']:
            raise ValueError('Background endpoint changed after its audit')
    if (audit['source_sha256'] != run['source']['sha256']
            or audit['reference_sha256'] != run['reference']['sha256']
            or audit['alpha']['sha256'] != plan['mask_roles']['foreground_alpha']['sha256']):
        raise ValueError('Scene audit source/reference/alpha binding changed')
    original = array(assets['background']); values = original.copy()
    validity = array(assets['validity']); alpha = array(plan['mask_roles']['foreground_alpha'])
    observation = audit['observation']; expected = array(observation['background'])
    weights = array(audit['support_asset']); size = tuple(observation['background']['size'])
    if (values.shape != (*validity.shape,3) or alpha.shape != validity.shape
            or np.any(original < 0) or np.any(original > validity[...,None]+2e-6)):
        raise ValueError('Background consistency requires finite associated SDR P/V')
    native_size = tuple(values.shape[1::-1]); h, w = values.shape[:2]
    native_support=cv2.resize((weights>0).astype(np.uint8),native_size,interpolation=cv2.INTER_NEAREST)>0
    receiver_box = None
    receiver_boxes = []
    for role in ('native_receiver_preservation', 'native_receiver_detail'):
        if not assets.get(role):
            continue
        receiver = read(bound(assets[role]))
        support = receiver.get('support_asset') or {}
        if receiver.get('state') == 'prepared':
            receiver_box = support.get('native_box') or receiver.get('native_box')
            if receiver_box is None:
                raise ValueError('Receiver protection has no native box')
            receiver_boxes.append(receiver_box)
    before = metrics(expected, cv2.resize(values,size,interpolation=cv2.INTER_AREA),weights)
    for iteration in range(CONFIG['iterations']):
        if cancel and cancel():
            raise InterruptedError('Background consistency cancelled')
        down = cv2.resize(values,size,interpolation=cv2.INTER_AREA)
        residual = gaussian_filter(expected-down, (CONFIG['sigma_compact'],CONFIG['sigma_compact'],0))
        residual *= weights[...,None]
        # Bound against the original endpoint over every iteration, not an
        # independently repeated layer budget that compounds on reopen.
        up = cv2.resize(residual, native_size, interpolation=cv2.INTER_LINEAR)
        for top in range(0,h,128):
            end=min(top+128,h); current=values[top:end]
            active=(alpha[top:end] == 0)&(validity[top:end] > .99999)&native_support[top:end]
            for protected_box in receiver_boxes:
                x0,y0,x1,y1=map(int,protected_box)
                if max(top,y0)<min(end,y1):
                    active[max(top,y0)-top:min(end,y1)-top,x0:x1]=False
            proposed=np.clip(current+up[top:end]-original[top:end],
                             -CONFIG['maximum_linear_RGB_delta'],CONFIG['maximum_linear_RGB_delta'])
            delta=proposed-(current-original[top:end]); delta[~active]=0
            updated=bounded_update(current,delta,validity[top:end])
            updated[~active]=current[~active]
            values[top:end]=updated
        del up
    after=metrics(expected,cv2.resize(values,size,interpolation=cv2.INTER_AREA),weights)
    changed=np.any(values != original,-1)
    identity_outside=bool(np.array_equal(values[alpha>0],original[alpha>0]))
    support_exact=bool(np.array_equal(values[~native_support],original[~native_support]))
    invalid_exact=bool(np.array_equal(values[validity<=.99999],original[validity<=.99999]))
    receiver_exact=True
    for protected_box in receiver_boxes:
        x0,y0,x1,y1=map(int,protected_box)
        receiver_exact &= bool(np.array_equal(values[y0:y1,x0:x1],original[y0:y1,x0:x1]))
    improvement=1-after['luminance_RMSE']/max(before['luminance_RMSE'],1e-12)
    accepted=bool(changed.any() and improvement>=CONFIG['minimum_relative_RMSE_improvement']
                  and identity_outside and receiver_exact and support_exact and invalid_exact
                  and np.isfinite(values).all())
    receipt={'schema':SCHEMA,'working_domain':DOMAIN,'status':'prepared' if accepted else 'rejected',
        'config':copy.deepcopy(CONFIG),'source_sha256':run['source']['sha256'],
        'reference_sha256':run['reference']['sha256'],'input_background':copy.deepcopy(assets['background']),
        'validity':copy.deepcopy(assets['validity']),'alpha':copy.deepcopy(plan['mask_roles']['foreground_alpha']),
        'observation':copy.deepcopy(observation),'audit':copy.deepcopy(audit['receipt_asset']),
        'before':before,'after':after,'relative_RMSE_improvement':float(improvement),
        'changed_native_pixels':int(changed.sum()),'person_coverage_exact':identity_outside,
        'outside_qualified_support_exact':support_exact,'invalid_sampling_exact':invalid_exact,
        'receiver_native_box_exact':receiver_exact,'validity_modified':False,'foreground_T_B_S_modified':False,
        'new_effect_generated':False,'physical_light_response':False,'visual_acceptance':False,
        'API_requests':0,'producer_sha256':sha(__file__),'elapsed_seconds':time.perf_counter()-started,
        'quality_scope':'registered_background_observation_data_consistency; not effect visual acceptance'}
    result=copy.deepcopy(plan)
    if accepted:
        target=output/'background-associated-consistent.npz';np.savez_compressed(target,values=values)
        replacement={**copy.deepcopy(assets['background']),**file_record(target),
            'data_state':DOMAIN,'working_domain':DOMAIN,
            'radiometric_origin':'qualified_compact_observation_consistent_SR_high_frequency',
            'consistency_observation_sha256':observation['background']['sha256']}
        receipt['output_background']=replacement
        output_assets=_background(result)['assets'];output_assets['background']=replacement
        output_assets['compact_observation']=copy.deepcopy(observation['background'])
        output_assets['compact_validity']=copy.deepcopy(observation['validity'])
    path=output/'background-consistency.json';write(path,receipt)
    if accepted:
        _background(result)['assets']['background_consistency']=file_record(path,representation='background_consistency_receipt')
        from scene_design import bind_scene_environment
        result['scene_design']=bind_scene_environment(result['scene_design'],{
            'background':_background(result)['assets']['background'],'validity':assets['validity'],
            'candidate_removal':result['mask_roles']['candidate_removal']})
        result.setdefault('provenance',{})['background_consistency']={
            'schema':SCHEMA,'receipt_sha256':sha(path),'status':receipt['status'],
            'relative_RMSE_improvement':float(improvement),'visual_acceptance':False}
    return {'ok':accepted,'design_reconstruction_plan':result,'receipt':receipt,'changed':accepted}
