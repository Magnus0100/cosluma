"""Reserved CPU integration of a reviewed generated increment, never a new image model.

The model chooses a finite relative focus scale. The immutable working lineage
and actual coverage own the editable pixels; this tool cannot select a new ROI,
blur the underlying portrait, grant approval, or claim native generated detail.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from asset_identity import sha256_file
from current_model_registry import atomic_json
from photo_editing import identity, validate_scope
import workflow_timing

SCHEMA = 'cosluma_generated_focus_integration.v1'
TOOL = 'integrate_generated_focus'
PARAMETERS = {'focus_sigma': {'type': 'number', 'minimum': 0., 'maximum': .02,
                            'unit': 'native_short_edge'}}
OPERATOR = 'coverage_normalized_gaussian_of_display_linear_working_minus_best'


def _same_asset(left,right):
    return (isinstance(left,dict) and isinstance(right,dict)
        and all(left.get(key)==right.get(key) for key in ('asset_id','receipt_sha256')))


def _asset(value):
    path = Path(value).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file(path)}


def _checked_asset(value,state=None):
    from photo_session_evidence import resolve_file
    return Path(resolve_file(state or {},value)['path'])


def _cancel(config):
    if config.get('cancel') and config['cancel']():
        raise InterruptedError('Focus integration cancelled; original complete candidates retained')


def _sigma(parameters):
    if not isinstance(parameters, dict) or set(parameters) != {'focus_sigma'}:
        raise ValueError('Focus integration requires exactly the typed focus_sigma parameter')
    value = parameters['focus_sigma']
    if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= .02:
        raise ValueError('Focus sigma must be finite and between zero and .02 of the native short edge')
    return float(value)


def _working_assets(source,best,working,provenance,*,appearance_assets,scope,design_id,evidence_state=None):
    from photo_appearance import load_asset,validate_asset
    from photo_session_evidence import resolve_asset
    state=evidence_state or {}
    appearance_assets={key:resolve_asset(state,value) for key,value in appearance_assets.items()}
    for record in (source,best,working):_checked_asset(record,state)
    if working['sha256']==best['sha256']:raise ValueError('Focus requires an actual different working candidate')
    if not isinstance(appearance_assets,dict) or set(appearance_assets)!={'best','working'}:
        raise ValueError('Focus requires actual typed best and working appearance assets')
    binding={'source_sha256':source['sha256'],'scope':scope,'design_id':design_id}
    base_receipt=validate_asset(appearance_assets['best'],**binding)
    edited_receipt=validate_asset(appearance_assets['working'],**binding)
    expected={'asset_id':base_receipt['asset_id'],'master_sha256':base_receipt['files']['master']['sha256'],
        'preview_sha256':base_receipt['files']['preview']['sha256']}
    if (edited_receipt['kind']!='mixed_regional' or edited_receipt['parent']!=expected
            or edited_receipt['generation']['core_box']==[0,0,*edited_receipt['canvas_size']]
            or base_receipt['files']['preview']['sha256']!=best['sha256']
            or edited_receipt['files']['preview']['sha256']!=working['sha256']):
        raise ValueError('Working float appearance must be a genuine local edit of this exact best master')
    composition_path=_checked_asset(provenance.get('composition_receipt'),state)
    composition=json.loads(composition_path.read_text(encoding='utf8'))
    attempt_path=_checked_asset(provenance.get('tool_attempt'),state)
    attempt=json.loads(attempt_path.read_text(encoding='utf8'))
    current=composition;current_sha=working['sha256'];visited=set()
    inputs=[source,best,working,provenance['composition_receipt'],provenance['tool_attempt']]
    while True:
        old=current.get('binding') or {}
        if (current.get('schema') not in {'cosluma_photo_edit_tool.v1',SCHEMA}
                or old.get('source',{}).get('sha256')!=source['sha256']
                or old.get('current_best',{}).get('sha256')!=best['sha256']
                or not _same_asset(current.get('parent_appearance'),appearance_assets['best'])
                or current.get('core_box')!=edited_receipt['generation']['core_box']
                or current.get('context_box')!=edited_receipt['generation']['context_box']
                or current.get('complete_candidate',{}).get('sha256')!=current_sha):
            raise ValueError('Working generation/core/parent evidence differs from the typed float asset')
        if current.get('schema')!=SCHEMA:break
        operator=current.get('focus_operator') or {}
        historical_sigma=_sigma({'focus_sigma':operator.get('focus_sigma')})
        if (current.get('image_generation_submissions')!=0 or operator.get('name')!=OPERATOR
                or operator.get('base_blurred') is not False or operator.get('quality_improvement_claimed') is not False
                or not math.isclose(operator.get('sigma_pixels',-1),historical_sigma*min(edited_receipt['canvas_size']),abs_tol=1e-9)):
            raise ValueError('CPU focus cannot claim a different operator or new generation')
        prior=current.get('input_composition_receipt');path=_checked_asset(prior,state)
        key=(str(path),prior['sha256'])
        if key in visited:raise ValueError('Focus provenance contains a cycle')
        visited.add(key);inputs.append(prior)
        original_working=current['binding'].get('working');_checked_asset(original_working,state);inputs.append(original_working)
        ancestor=json.loads(path.read_text(encoding='utf8'))
        if ancestor.get('complete_candidate',{}).get('sha256')!=original_working['sha256']:
            raise ValueError('CPU focus ancestry lost its actual input candidate')
        current=ancestor;current_sha=original_working['sha256']
    if (composition.get('complete_candidate',{}).get('sha256')!=working['sha256']
            or not _same_asset(composition.get('appearance_asset'),appearance_assets['working'])
            or attempt.get('state')!='complete' or attempt.get('binding')!=composition.get('binding')
            or attempt.get('complete_candidate',{}).get('sha256')!=working['sha256']
            or (attempt.get('composition_receipt') or {}).get('sha256')!=provenance['composition_receipt']['sha256']):
        raise ValueError('Working completion receipt differs from its real typed candidate')
    generated=edited_receipt['generation'];directory=Path(appearance_assets['working']['path'])
    if ((provenance.get('provider_receipt') or {}).get('sha256')!=generated['provider_receipt']['sha256']
            or (provenance.get('generated_image') or {}).get('sha256')!=generated['image']['sha256']):
        raise ValueError('Working provenance must retain the actual original provider image/receipt identity')
    original_image={'path':str(directory/generated['image']['file']),'sha256':generated['image']['sha256']}
    provider={'path':str(directory/generated['provider_receipt']['file']),'sha256':generated['provider_receipt']['sha256']}
    for record in (original_image,provider):_checked_asset(record,state);inputs.append(record)
    base=edited=None
    try:
        base=load_asset(appearance_assets['best'],**binding);edited=load_asset(appearance_assets['working'],**binding)
        coverage=edited['edit_coverage'];core=edited_receipt['generation']['core_box']
        if not isinstance(coverage,np.ndarray) or coverage.shape!=base['master'].shape[:2]:
            raise ValueError('Focus requires actual local edit coverage')
        for top in range(0,len(coverage),128):
            outside=coverage[top:top+128]==0
            if not np.array_equal(base['master'][top:top+128][outside],edited['master'][top:top+128][outside]):
                raise ValueError('Working float pixels outside generated coverage differ from best')
    except BaseException:
        from photo_edit_tools import _close_loaded
        _close_loaded(base,edited)
        raise
    return base,edited,coverage,core,composition,original_image,provider,inputs


def validate_focus_working(source,best,working,provenance,*,appearance_assets,scope,design_id,evidence_state=None):
    """Read-only admission of a genuine local float increment, never a PNG fallback."""
    scope=validate_scope(scope)
    base,edited,coverage,core,_,original,_,_=_working_assets(source,best,working,provenance,
        appearance_assets=appearance_assets,scope=scope,design_id=design_id,evidence_state=evidence_state)
    try:
        return {'schema':SCHEMA,'source_sha256':source['sha256'],'best_sha256':best['sha256'],
            'working_sha256':working['sha256'],'provenance_sha256':identity(provenance),
            'best_asset_id':appearance_assets['best']['asset_id'],'working_asset_id':appearance_assets['working']['asset_id'],
            'canvas_size':base['receipt']['canvas_size'],'core_box':core,
            'context_box':edited['receipt']['generation']['context_box'],'generated_image':original,
            'effective_coverage_fraction':float(np.count_nonzero(coverage)/coverage.size),
            'image_generation_submissions':0,'camera_detail_claimed':False}
    finally:
        from photo_edit_tools import _close_loaded
        _close_loaded(base,edited)


def _reservation(context,config):
    import photo_edit_session as session
    from photo_session_evidence import resolve_file,resolve_asset
    from photo_edit_tools import _typed_inputs
    state=session._read(Path(config['session_directory']));scope=validate_scope(config['scope'])
    if context.get('scope')!=scope:raise ValueError('Focus requires its checked editing scope')
    images=context.get('images')
    if not isinstance(images,list) or [row.get('role') for row in images]!=['source','best','working']:
        raise ValueError('Focus needs source, best and a real reviewed working candidate')
    source,_=_typed_inputs(context,state,scope,context['design']);_,best,working=images
    best_row=next((r for r in state['candidates'] if r['candidate_id']==session.current_canvas_id(state)),None)
    row=next((r for r in state['candidates'] if r['candidate_id']==state.get('working_candidate_id')),None)
    attempt=next((r for r in state['attempts'] if r['attempt_id']==context.get('attempt_id')),None)
    decision=context.get('decision') or {};descriptor=(state.get('runtime') or {}).get('tools',{}).get(TOOL) or {}
    costs=dict.fromkeys(session.COUNTERS,0);costs.update(tool_calls=1,improvement_steps=session.tool_improvement_cost(state))
    if (state['session_id']!=context.get('session_id') or state['status']!='active' or state['phase']!='tool'
            or context.get('operation_stage')!='improvement' or state['pending_decision']!=decision
            or state['design']!=context.get('design') or state['user_intent']!=context.get('user_intent')
            or not best_row or not row or best_row==row or not row.get('review') or not best_row.get('review')
            or resolve_file(state,best_row['image'])!={k:best[k] for k in ('path','sha256')}
            or resolve_file(state,row['image'])!={k:working[k] for k in ('path','sha256')}
            or resolve_asset(state,best_row.get('appearance_asset'))!=context['appearance_assets']['best']
            or resolve_asset(state,row.get('appearance_asset'))!=context['appearance_assets']['working']
            or not attempt or attempt['kind']!='tool' or attempt['status']!='reserved'
            or attempt['context']!=context or attempt['context_sha256']!=identity(context) or attempt['costs']!=costs
            or descriptor.get('producer_identity')!=config['producer_identity']
            or descriptor.get('requires_working') is not True or descriptor.get('requires_region') is not False
            or descriptor.get('parameter_schema')!=PARAMETERS
            or decision.get('action')!='tool' or decision.get('tool')!=TOOL or decision.get('operations')!=['detail_repair']):
        raise ValueError('Focus must match the actual reserved typed working/session facts')
    for candidate in (best_row,row):
        board=(candidate['review'].get('provenance') or {}).get('review_board')
        if not board or candidate['review']['provenance'].get('source')!=board:raise ValueError('Focus requires its actual independent observation')
        _checked_asset(board,state)
    return source,best,working,row,_sigma(decision.get('parameters')),state


def integrate_increment(best,working,coverage,core,sigma_pixels):
    """Normalize blur of the associated float generated difference; base stays sharp."""
    from scipy.ndimage import gaussian_filter
    if (best.dtype!=np.float32 or working.dtype!=np.float32 or best.shape!=working.shape
            or coverage.dtype!=np.float32 or coverage.shape!=best.shape[:2]
            or not math.isfinite(sigma_pixels) or sigma_pixels<0):
        raise ValueError('Focus requires matched native float masters, coverage and finite sigma')
    if sigma_pixels==0:return working.copy()
    x0,y0,x1,y1=core;mask=coverage[y0:y1,x0:x1]
    delta=working[y0:y1,x0:x1]-best[y0:y1,x0:x1]
    denominator=gaussian_filter(mask,sigma_pixels,mode='constant',cval=0.,output=np.float32)
    numerator=gaussian_filter(delta,(sigma_pixels,sigma_pixels,0.),mode='constant',cval=0.,output=np.float32)
    valid=(mask>0)&(denominator>np.finfo(np.float32).tiny);increment=np.zeros_like(delta)
    np.divide(numerator,denominator[...,None],out=increment,where=valid[...,None]);increment*=mask[...,None]
    if not np.isfinite(increment).all():raise ValueError('Focus produced a nonfinite float increment')
    result=best.copy();patch=np.clip(best[y0:y1,x0:x1]+increment,0,1)
    result[y0:y1,x0:x1][valid]=patch[valid]
    return result


@workflow_timing.timed('photographic_focus_integration',model_request=False,http_request=False)
def integrate_generated_focus(context,config):
    from photo_appearance import freeze_candidate,load_asset,validate_asset
    _cancel(config);source,best,working,row,sigma,state=_reservation(context,config)
    assets={key:context['appearance_assets'][key] for key in ('best','working')}
    base,edited,coverage,core,_,original,provider,inputs=_working_assets(source,best,working,row['provenance'],
        appearance_assets=assets,scope=config['scope'],design_id=context['design']['design_id'],evidence_state=state)
    sigma_pixels=sigma*min(base['master'].shape[:2]);output=Path(config['output']).resolve()/context['attempt_id']
    output.mkdir(parents=True,exist_ok=True);attempt_path=output/'focus-attempt.json'
    binding={'session_id':context['session_id'],'attempt_id':context['attempt_id'],'context_sha256':identity(context),
        'source':source,'current_best':best,'working':working,'current_appearance':assets['best'],
        'working_appearance':assets['working'],'scope_id':config['scope']['scope_id'],
        'design_id':context['design']['design_id'],'decision':context['decision'],'image_generation_submissions':0}
    record={'schema':SCHEMA,'state':'reserved','binding':binding,'automatic_retry':False,
        'image_generation_submissions':0,'human_approved':False}
    with attempt_path.open('x',encoding='utf8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    started=time.perf_counter();asset_binding={'source_sha256':source['sha256'],'scope':config['scope'],
        'design_id':context['design']['design_id']}
    loaded=None
    try:
        _cancel(config);result=integrate_increment(base['master'],edited['master'],coverage,core,sigma_pixels)
        for value in inputs:_checked_asset(value,state)
        if sigma==0:asset=copy.deepcopy(assets['working'])
        else:
            asset=freeze_candidate(output/'appearance',kind='mixed_regional',master=result,parent=assets['best'],
                coverage=coverage,generation={'image':original,'provider_receipt':provider,
                    'core_box':core,'context_box':edited['receipt']['generation']['context_box']},
                cancel=config.get('cancel'),**asset_binding)
        loaded=load_asset(asset,**asset_binding);candidate=_asset(loaded['preview_path'])
        operator={'name':OPERATOR,'focus_sigma':sigma,'sigma_unit':'native_short_edge','sigma_pixels':sigma_pixels,
            'boundary':'constant_zero_associated_increment_and_coverage','gaussian_truncate':4.,
            'working_domain':'display_linear_sRGB','base_blurred':False,'output_gamut':'clip_sdr',
            'quality_improvement_claimed':False,'zero_sigma_noop':sigma==0}
        composition={'schema':SCHEMA,'binding':binding,'working_domain':'display_linear_sRGB',
            'core_box':core,'context_box':edited['receipt']['generation']['context_box'],
            'coverage':loaded['receipt']['edit_coverage'],'coverage_semantics':'current_canvas_edit_coverage_not_source_subject_alpha',
            'appearance_asset':asset,'parent_appearance':assets['best'],'focus_operator':operator,
            'outside_core_float_identical_to_current_best':True,'outside_core_pixels_identical_to_current_best':True,
            'source_subject_RGB_restored':False,'complete_candidate':candidate,'original_generated_image':original,
            'input_working_provenance_sha256':identity(row['provenance']),
            'input_composition_receipt':copy.deepcopy(row['provenance']['composition_receipt']),
            'image_generation_submissions':0,'texture_origin':'CPU_processed_model_generated_increment',
            'native_generated_detail':False,'camera_detail_claimed':False,'identity_verified':False,
            'physical_optics_verified':False,'needs_visual_review':True,'human_approved':False,
            'producer':{'sha256':sha256_file(__file__),'identity':config['producer_identity']}}
        path=output/'composition.json';atomic_json(path,composition)
        for value in inputs:_checked_asset(value,state)
        validate_asset(asset,**asset_binding);_cancel(config)
        record.update(state='complete',complete_candidate=candidate,appearance_asset=asset,
            composition_receipt=_asset(path),elapsed_seconds=time.perf_counter()-started);atomic_json(attempt_path,record)
        return {'image':str(loaded['preview_path']),'appearance_asset':asset,'provenance':{'schema':SCHEMA,'tool':TOOL,
            'source':source,'current_best':best,'working':working,'parent_appearance':assets['best'],'appearance_asset':asset,
            'full_candidate':{**candidate,'size':base['receipt']['canvas_size']},'focus_operator':operator,
            'generated_image':{**original,'size':edited['receipt']['generated_resolution']},
            'provider_receipt':provider,'composition_receipt':_asset(path),'tool_attempt':_asset(attempt_path),
            'image_generation_submissions':0,'automatic_retry':False,'needs_visual_review':True,'human_approved':False},
            'resources':{'image_submissions':0,'local_repairs':0,'wall_seconds':time.perf_counter()-started}}
    except BaseException as error:
        record.update(state='interrupted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'needs_attention',
            error={'type':type(error).__name__,'message':str(error)[:1200]});atomic_json(attempt_path,record)
        raise
    finally:
        from photo_edit_tools import _close_loaded
        _close_loaded(base,edited,loaded)


def make_focus_tool(root,output,*,scope,session_directory,cancel=None):
    from photo_edit_session import Tool,current_canvas_id
    scope=validate_scope(scope)
    producer=identity({'code':{name:sha256_file(Path(__file__).parent/name) for name in (
        'photo_focus_integration.py','photo_edit_tools.py','photo_edit_session.py','photo_editing.py',
        'photo_appearance.py','photo_session_evidence.py','float_appearance.py','raw_support.py')},'scope_id':scope['scope_id'],'parameters':PARAMETERS})
    config={'root':str(Path(root).resolve()),'output':str(Path(output).resolve()),'scope':scope,
        'session_directory':str(Path(session_directory).resolve()),'cancel':cancel,'producer_identity':producer}
    def input_check(state):
        from photo_session_evidence import resolve_file,resolve_asset
        if state.get('scope')!=scope:return False
        best=next((r for r in state.get('candidates',[]) if r['candidate_id']==current_canvas_id(state)),None)
        working=next((r for r in state.get('candidates',[]) if r['candidate_id']==state.get('working_candidate_id')),None)
        if not best or not working or not best.get('appearance_asset') or not working.get('appearance_asset'):return False
        validate_focus_working(resolve_file(state,state['binding']['source']),resolve_file(state,best['image']),
            resolve_file(state,working['image']),working['provenance'],
            appearance_assets={'best':resolve_asset(state,best['appearance_asset']),
                'working':resolve_asset(state,working['appearance_asset'])},scope=scope,
            design_id=state['binding']['design_id'],evidence_state=state)
        return True
    return Tool(TOOL,('detail_repair',),{},producer,lambda context:integrate_generated_focus(context,config),
        requires_working=True,parameter_schema=copy.deepcopy(PARAMETERS),input_check=input_check)
