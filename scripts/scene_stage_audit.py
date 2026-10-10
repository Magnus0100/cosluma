"""Read-only, content-bound background stage audit, without photo-specific ROIs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

SCHEMA = 'scene_stage_audit.v1'
DOMAIN = 'display_linear_sRGB'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def file_record(path, **kwargs):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path), **kwargs}


def bound(asset):
    path = Path(asset['path'])
    if not path.is_file() or sha(path) != asset['sha256']:
        raise ValueError('Scene audit asset bytes changed: '+str(path))
    return path


def array(asset):
    with np.load(bound(asset), allow_pickle=False) as resource:
        values = np.asarray(resource['values'], np.float32)
    if not np.isfinite(values).all() or list(values.shape[1::-1]) != list(asset['size']):
        raise ValueError('Scene audit asset grid/finite values invalid')
    return values


def _background(plan):
    rows = [r for r in plan.get('layers', []) if r.get('kind') == 'background_design']
    if len(rows) != 1:
        raise ValueError('One background owner is required')
    return rows[0]


def resolve_observation(root, run, plan):
    """Resolve explicit observation or the exact deterministic original cache."""
    assets = _background(plan)['assets']
    if (plan.get('source_sha256') != run['source']['sha256']
            or plan.get('candidate_sha256') != run['reference']['sha256']):
        raise ValueError('Scene plan belongs to another source/reference input pair')
    for role in ('source','reference'):
        item=run[role]; path=Path(item['path']); path=path if path.is_absolute() else Path(root)/path
        if sha(path) != item['sha256']:
            raise ValueError('Scene input bytes changed: '+role)
    explicit = bool(assets.get('compact_observation') and assets.get('compact_validity'))
    cached=plan; cache_path=None
    inherited={}; inherited_receipt={}
    if explicit and assets.get('background_consistency'):
        inherited_receipt=read(bound(assets['background_consistency']))
        inherited=inherited_receipt.get('observation') or {}
    elif explicit and assets.get('background_observation'):
        inherited=read(bound(assets['background_observation']))
        if (inherited.get('schema')!='background_observation.v1'
                or inherited.get('source_sha256')!=run['source']['sha256']
                or inherited.get('reference_sha256')!=run['reference']['sha256']):
            raise ValueError('Frozen background observation belongs to another input pair')
    sr=(plan.get('provenance') or {}).get('background_native_detail') or inherited.get('sr_provenance')
    if not sr:
        identity = plan.get('generation_identity')
        if not isinstance(identity, dict):
            raise ValueError('Compact observation is not bound to this saved plan')
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        cache_path = Path(root)/'outputs/cache/design-reconstruction'/key/'plan.json'
        cached = read(cache_path)
        if (cached.get('generation_identity') != identity
                or cached.get('source_sha256') != run['source']['sha256']
                or cached.get('candidate_sha256') != run['reference']['sha256']):
            raise ValueError('Deterministic observation cache belongs to another input pair')
        sr=(cached.get('provenance') or {}).get('background_native_detail') or {}
    ancestor = assets['background']
    pending = {role: read(bound(assets[role])) for role in
               ('background_consistency', 'native_receiver_preservation', 'native_receiver_detail')
               if assets.get(role)}
    while pending:
        matching = [role for role, receipt in pending.items()
                    if (receipt.get('output_background') or {}).get('sha256') == ancestor.get('sha256')]
        if len(matching) != 1:
            raise ValueError('Background observation ancestry is not exact')
        ancestor = pending.pop(matching[0])['input_background']
    if (sr.get('schema_version') != 'background_superresolution.v1'
            or sr.get('assets', {}).get('background', {}).get('sha256') != ancestor.get('sha256')):
        raise ValueError('Current background is not descended from the bound compact SR observation')
    result = {key: copy.deepcopy(sr['source_assets'][key]) for key in ('background', 'validity')}
    for value in result.values():
        bound(value)
    if explicit and any(result[key]['sha256'] != assets[role]['sha256'] for key,role in
                        (('background','compact_observation'),('validity','compact_validity'))):
        raise ValueError('Explicit compact observation does not match the original SR input')
    if explicit:
        result={key:copy.deepcopy(assets[role]) for key,role in
                (('background','compact_observation'),('validity','compact_validity'))}
    result.update(origin='explicit_SR_ancestry' if explicit else 'exact_generation_identity_and_SR_ancestry')
    result['sr_provenance']=copy.deepcopy(sr)
    clean_binding=inherited.get('clean_stage_binding')
    if not clean_binding:
        stage=(cached.get('provenance') or {}).get('background_stage_cache') or {}
        clean_binding=file_record(stage['binding'])
    bound(clean_binding); result['clean_stage_binding']=copy.deepcopy(clean_binding)
    clean_input=inherited.get('clean_input_asset')
    if not clean_input and inherited_receipt.get('audit'):
        audited=read(bound(inherited_receipt['audit']))
        clean_input=(audited.get('clean_preservation') or {}).get('input')
    if not clean_input:
        clean_record=read(bound(clean_binding)); clean_manifest=read(bound(clean_record['result_manifest']))
        path=Path(clean_manifest['input']); path=path if path.is_absolute() else bound(clean_binding).parent/path
        clean_input=file_record(path)
    bound(clean_input); result['clean_input_asset']=copy.deepcopy(clean_input)
    # The exact cache establishes this receipt once. Freeze only the observed
    # P/V and SR/clean ancestry, not every unrelated old generation dependency.
    if cache_path is not None: result['cache_plan_sha256']=sha(cache_path)
    return result


def audit_clean_preservation(root, run, plan, observation):
    """Reopen exact candidate-erasure binding and check every unmasked pixel."""
    from raw_support import open_rgb
    binding_path=bound(observation['clean_stage_binding']); binding=read(binding_path)
    if (binding.get('schema_version')!='design_background_stage_cache.v1'
            or binding.get('successful_worker_returncode')!=0
            or binding.get('request_identity',{}).get('output_contract') !=
                'preserve_candidate_RGB_exactly_outside_supplied_mask'):
        raise ValueError('Candidate clean stage lacks an exact preservation contract')
    manifest=read(bound(binding['result_manifest'])); request=binding['request_identity']
    input_path=bound(observation['clean_input_asset'])
    if sha(input_path)!=request['input_sha256']:
        raise ValueError('Candidate clean input bytes changed')
    clean_path=bound(binding['outputs']['clean_png']); mask_path=bound(binding['outputs']['removal_mask'])
    bound(binding['outputs']['clean_npz'])
    if (sha(mask_path)!=request['removal_mask_sha256']
            or sha(mask_path)!=plan['mask_roles']['candidate_removal']['sha256']):
        raise ValueError('Candidate clean removal owner changed')
    original=np.asarray(Image.open(input_path).convert('RGB'))
    actual=np.asarray(Image.open(clean_path).convert('RGB'))
    mask=np.asarray(Image.open(mask_path).convert('L'))>0
    reference=Path(run['reference']['path']); reference=reference if reference.is_absolute() else Path(root)/reference
    decoded=open_rgb(reference); decoded.thumbnail((1536,1536),Image.Resampling.LANCZOS)
    if not np.array_equal(original,np.asarray(decoded)):
        raise ValueError('Candidate clean input is not the exact selected reference proxy')
    if actual.shape!=original.shape or mask.shape!=original.shape[:2]:
        raise ValueError('Candidate clean grids are not paired')
    error=np.any(actual!=original,-1)&~mask
    if error.any():
        raise ValueError('Candidate clean modified pixels outside its removal mask')
    return {'binding':file_record(binding_path),'input':file_record(input_path),
        'clean':copy.deepcopy(binding['outputs']['clean_png']),
        'removal':copy.deepcopy(binding['outputs']['removal_mask']),
        'observed_unmasked_pixels':int((~mask).sum()),'outside_removal_exact':True,
        'semantic_effect_acceptance':False}


def eligibility(plan, observation, *, arrays=None):
    """All observed background, with no hand-written effect box or smoke claim."""
    assets = _background(plan)['assets']; size = tuple(observation['background']['size'])
    if (observation['validity']['size'] != list(size)
            or plan.get('canvas', {}).get('candidate_mapping', {}).get('scene_registration_supported') is not True):
        raise ValueError('Supported scene registration and paired compact grids required')
    current, validity = arrays if arrays is not None else (array(assets['background']), array(assets['validity']))
    native = tuple(plan['canvas']['source_oriented_size'])
    if tuple(current.shape[1::-1]) != native or validity.shape != current.shape[:2]:
        raise ValueError('Current background P/V must be native paired assets')
    alpha = array(plan['mask_roles']['foreground_alpha'])
    if alpha.shape != validity.shape or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError('Source coverage is invalid')
    a = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    v = cv2.resize(validity, size, interpolation=cv2.INTER_AREA)
    compact_v = array(observation['validity'])
    if np.any((validity < 0) | (validity > 1)) or np.any((compact_v < 0) | (compact_v > 1)):
        raise ValueError('Sampling validity is invalid')
    removal_asset = plan['mask_roles']['candidate_removal']
    removed = np.asarray(Image.open(bound(removal_asset)).convert('L')) > 0
    mapping = plan['canvas']['candidate_mapping']
    h = mapping.get('candidate_processing_proxy_to_final_proxy')
    if (tuple(size)!=tuple(plan['canvas']['source_proxy_size']) or h is None
            or removal_asset.get('coordinate_space') not in ('candidate_proxy','candidate_processing_proxy')):
        raise ValueError('Removal support needs its declared processing-proxy registration')
    matrix = np.asarray(h, np.float64)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-9:
        raise ValueError('Scene registration matrix is invalid')
    removed = cv2.warpPerspective(removed.astype(np.uint8), matrix, size,
                                flags=cv2.INTER_NEAREST, borderValue=1) > 0
    receiver = np.zeros(a.shape, bool)
    receiver_asset = assets.get('native_receiver_preservation')
    if receiver_asset:
        receipt = read(bound(receiver_asset)); support = receipt.get('support_asset') or {}
        if receipt.get('state') == 'prepared':
            weights = array(support); box = receipt.get('native_box') or support.get('native_box')
            if not box or len(box) != 4:
                raise ValueError('Prepared receiver has no explicit native support box')
            x0, y0 = np.floor(np.asarray(box[:2])*np.asarray(size)/np.asarray(native)).astype(int)
            x1, y1 = np.ceil(np.asarray(box[2:])*np.asarray(size)/np.asarray(native)).astype(int)
            receiver[y0:y1, x0:x1] = cv2.resize(weights, (x1-x0, y1-y0), interpolation=cv2.INTER_AREA) > 0
    eligible = (a < 1e-6) & (v > .99999) & (compact_v > .99999) & ~removed & ~receiver
    # Feather inward only. No pixels outside the qualified component acquire
    # response, and distance excludes both unknown coverage and receiver edges.
    weight = np.clip((distance_transform_edt(eligible)-2)/6, 0, 1).astype(np.float32)
    return weight, {'origin': 'registered_observed_nonremoval_only',
        'eligible_proxy_pixels': int(eligible.sum()), 'full_dose_proxy_pixels': int((weight > .99).sum()),
        'removed_proxy_pixels': int(removed.sum()), 'receiver_excluded_proxy_pixels': int(receiver.sum()),
        'person_alpha_unchanged': True, 'hand_ROI_used': False}


def metrics(expected, actual, weight):
    selected = weight > .99
    if not selected.any():
        return {'pixels': 0, 'luminance_RMSE': 0., 'RGB_abs_error_q50_q90_q99_max': [0.]*4}
    delta = expected-actual; lum = delta @ np.array([.2126, .7152, .0722], np.float32)
    return {'pixels': int(selected.sum()), 'luminance_RMSE': float(np.sqrt(np.mean(lum[selected]**2))),
            'RGB_abs_error_q50_q90_q99_max': np.quantile(np.max(np.abs(delta), -1)[selected], [.5,.9,.99,1]).tolist()}


def audit_scene_stages(root, run, output, *, diagnosis=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    plan = (run.get('options', {}).get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
    report = {'schema': SCHEMA, 'working_domain': DOMAIN, 'status': 'unsupported',
        'source_sha256': run.get('source', {}).get('sha256'),
        'reference_sha256': run.get('reference', {}).get('sha256'), 'run_id': run.get('run_id'),
        'producer_sha256': sha(__file__), 'hand_ROI_used': False, 'API_requests': 0,
        'pixel_assets_modified': False, 'visual_acceptance': False}
    try:
        if plan.get('working_domain') != DOMAIN:
            raise ValueError('Scene audit requires the current paired appearance domain')
        bg = _background(plan)
        if not bg.get('enabled') or float(bg.get('strength', 1)) != 1:
            raise ValueError('Scene repair requires a complete enabled background endpoint')
        observation = resolve_observation(root, run, plan)
        clean_preservation=audit_clean_preservation(root,run,plan,observation)
        current, validity = array(bg['assets']['background']), array(bg['assets']['validity'])
        weight, support = eligibility(plan, observation, arrays=(current, validity))
        compact = array(observation['background']); size = tuple(observation['background']['size'])
        down = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
        numeric = metrics(compact, down, weight)
        support_path = output/'observed-background-support.npz'
        np.savez_compressed(support_path, values=weight)
        report.update(status='repair_eligible' if numeric['luminance_RMSE'] > 2e-4 else 'audited_no_repair',
            reason='measured_SR_observation_drift' if numeric['luminance_RMSE'] > 2e-4 else 'registered_background_observation_retained',
            input_background=copy.deepcopy(bg['assets']['background']), validity=copy.deepcopy(bg['assets']['validity']),
            alpha=copy.deepcopy(plan['mask_roles']['foreground_alpha']), observation=observation,
            clean_preservation=clean_preservation,
            support_asset=file_record(support_path, size=list(size), coordinate_space='source_proxy'),
            eligibility=support, consistency_before=numeric,
            qualification_scope='SR_data_consistency_only; no_smoke_or_physical_light_quality_claim')
        final_path = Path(run.get('artifacts', {}).get('render_scene_master', ''))
        if final_path.is_file():
            final = cv2.imread(str(final_path), cv2.IMREAD_UNCHANGED)
            if final is None or final.dtype != np.float32 or final.shape != current.shape:
                raise ValueError('Final scene evidence is not the actual native float32 master')
            final = cv2.resize(final[..., ::-1], size, interpolation=cv2.INTER_AREA)
            report['final_background_consumption'] = metrics(down, final, weight)
            report['final_scene'] = file_record(final_path)
        if diagnosis is not None:
            report['diagnosis'] = copy.deepcopy(diagnosis)
        # Bind the small receipt and all source observations, not latest folders.
        report['binding_identity_sha256'] = hashlib.sha256(json.dumps({key: report[key] for key in
            ('source_sha256','reference_sha256','input_background','validity','alpha','observation','support_asset')},
            sort_keys=True).encode()).hexdigest()
    except (KeyError, ValueError, OSError, TypeError) as exc:
        report.update(status='unsupported', reason=str(exc))
    path = output/'scene-stage-audit.json'; write(path, report)
    return {**report, 'receipt_asset': file_record(path)}
