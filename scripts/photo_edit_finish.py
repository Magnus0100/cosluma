"""Shared bounded session driver and reviewed-pixel save for the coordinator.

The existing production entry point owns task checks, origin budgets, initial
generation and result publication. This module neither plans nor runs a second
photo pipeline. Manual improvement and default completion share its loop.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from asset_identity import sha256_file
from photo_editing import identity


def _path(root,value):
    path=Path(value)
    return (path if path.is_absolute() else Path(root)/path).resolve()


def _cancelled(cancel):
    return bool(cancel and cancel())


def _can_improve(state,tools,*,zero_image_only=False):
    from photo_edit_session import tool_inputs_available,tool_improvement_cost
    remaining={key:state['binding']['budget'][key]-state['used'][key] for key in state['used']}
    for tool in tools:
        if zero_image_only and tool.costs.get('image_submissions',0)!=0:continue
        if not tool_inputs_available(state,tool):continue
        costs={**dict.fromkeys(remaining,0),**tool.costs}
        costs['vision_requests']+=1+int(state['binding'].get('visual_review',True))
        costs['tool_calls']=1;costs['improvement_steps']=tool_improvement_cost(state)
        if all(count<=remaining[key] for key,count in costs.items()):return True
    return False


def drive_session(directory,*,tools,decide,evaluate,model_identity,cancel=None,hooks=None,
                  before_step=None,after_step=None):
    """Advance the single persisted session until pass, bound, failure or stop."""
    import photo_edit_session as session
    directory=Path(directory).resolve();state=session.snapshot(directory);failure=None
    while state['status']=='active':
        zero_image_only=False
        if state['phase']=='decide' and not _cancelled(cancel):
            if hooks is not None:
                remaining=hooks.remaining();initial=session.source_native(state) and session.operation_stage(state)=='initial'
                if remaining['initial_operations' if initial else 'improvement_rounds']==0:
                    state=session.stop(directory,'origin_improvement_budget_exhausted');break
                zero_image_only=not initial and remaining.get('zero_image_only',False)
            if not tools or not _can_improve(state,tools,zero_image_only=zero_image_only):
                state=session.stop(directory,'tool_capabilities_exhausted' if not tools else 'checked_budget_exhausted');break
        if before_step is not None:before_step(state)
        try:
            state=session.advance(directory,tools=tools,decide=decide,evaluate=evaluate,
                model_identity=model_identity,cancel=cancel,zero_image_only=zero_image_only)
        except Exception as error:
            failure={'type':type(error).__name__,'message':str(error)[:1600]}
            state=session.snapshot(directory)
            break
        finally:
            if after_step is not None:after_step(session.snapshot(directory))
    selected=session.selected_output(directory)['candidate']
    return {'state':state,'selected':selected,'failure':failure,
        'cancelled':state['status']=='cancelled' or _cancelled(cancel)}


def session_summary(directory,state,selected):
    from photo_edit_session import creative_delivery_eligibility
    result={'status':state['status'],'phase':state['phase'],'path':str(Path(directory)/'session.json'),
        'session_id':state['session_id'],'reserved_budget_usage':copy.deepcopy(state['used']),
        'automated_target_met':state['automated_target_met'],'human_approved':False,
        'creative_delivery_eligibility':creative_delivery_eligibility(state,selected),
        'independent_evaluation':copy.deepcopy((selected or {}).get('review')),
        'review_objective':(selected or {}).get('review_objective','current'),
        'objective_reviewed':bool((selected or {}).get('review') and
            (selected or {}).get('review_objective','current')=='current'),
        'stop_reason':state['stop_reason'],'native_detail_acceptance':False,'automatic_image_retry':False}
    if state['binding'].get('editing_initialization') or state['binding'].get('appearance_required'):
        result.update(initialization=copy.deepcopy(state['binding'].get('editing_initialization')),
            variant_id=state['binding'].get('variant_id'),initial_operation=copy.deepcopy(state.get('initial_operation')),
            source_sha256=state['binding']['source']['sha256'],source_appearance=copy.deepcopy(state.get('source_appearance')),
            selected_is_source_baseline=bool(selected and (selected['candidate_id']==state.get('baseline_candidate_id')
                or selected.get('appearance_kind')=='source_baseline')))
        if state.get('source_appearance'):
            from appearance_assets import validate_asset
            from photo_session_evidence import resolve_asset
            result['source_preview_sha256']=validate_asset(resolve_asset(state,state['source_appearance']),source_sha256=result['source_sha256'],
                scope=state['scope'],design_id=state['design']['design_id'])['files']['preview']['sha256']
    return result


def reviewed_candidate_record(root,selected,options,*,evidence_state=None):
    """Keep generation and verified CPU processing distinct in saved provenance."""
    from single_source_workflow import candidate_record
    provenance=selected['provenance'];image=Path(selected['image']['path'])
    if selected.get('appearance_asset'):
        from appearance_assets import validate_asset
        asset=selected['appearance_asset']
        scope=(options.get('task_plan') or {}).get('editing_scope')
        receipt=validate_asset(asset,source_sha256=selected['source_sha256'],scope=scope,design_id=selected['design_id'])
        if receipt['kind']=='source_baseline':
            record=candidate_record(image,Path(root),provider='source_native_baseline',model=None,prompt='',generation_id=None)
            record.update(photo_appearance=copy.deepcopy(asset),processing={'image_generation_submissions':0,'source_native_baseline':True})
            return record
        if receipt['kind']=='geometric_transform':
            if (provenance.get('image_generation_submissions')!=0
                    or provenance.get('tool')!='face_contour_deform'
                    or (provenance.get('geometric_asset') or {}).get('asset_id')!=asset['asset_id']):
                raise ValueError('Geometric result lacks its actual zero-generation typed operation')
            record=candidate_record(image,Path(root),provider='local_geometric_deformation',
                model='face_contour_deform',prompt='',generation_id=None)
            record.update(photo_appearance=copy.deepcopy(asset),provenance=str(Path(asset['path'])/'receipt.json'),
                provenance_status='present',processing={'image_generation_submissions':0,
                    'geometric_asset':copy.deepcopy(asset),'operation':copy.deepcopy(receipt['operation'])})
            return record
        if receipt['kind']=='photographic_adjustment':
            from photo_edit_session import PHOTOGRAPHIC_PARAMETERS,validate_parameters
            operation=receipt.get('photographic_operation') or {}
            archived=operation.get('operation_receipt') or {}
            path=Path(asset['path'])/archived['file']
            if (sha256_file(path)!=archived.get('sha256')
                    or (provenance.get('operation_receipt') or {}).get('sha256')!=archived['sha256']
                    or provenance.get('image_generation_submissions')!=0):
                raise ValueError('Photographic development lacks its actual owned operation receipt')
            actual=json.loads(path.read_text(encoding='utf8'))
            output=actual.get('output') or {};operator=actual.get('operator') or {}
            if (actual.get('schema')!='cosluma_photographic_development.v1'
                    or actual.get('image_generation_submissions')!=0
                    or actual.get('input_parent')!=receipt['parent']
                    or output.get('master_sha256')!=receipt['files']['master']['sha256']
                    or output.get('preview',{}).get('sha256')!=selected['image']['sha256']
                    or operator.get('working_domain')!=receipt['working_domain']
                    or operator.get('physical_relighting') is not False
                    or not isinstance(actual.get('stats'),dict)):
                raise ValueError('Photographic operation does not bind the actual float result')
            validate_parameters(operator.get('parameters'),PHOTOGRAPHIC_PARAMETERS)
            record=candidate_record(image,Path(root),provider='local_photographic_development',
                model='develop_photograph',prompt='',generation_id=None)
            record.update(photo_appearance=copy.deepcopy(asset),provenance=str(path),provenance_status='present',
                processing={'image_generation_submissions':0,
                    'operation_receipt':{'path':str(path),'sha256':archived['sha256']},
                    'operator':copy.deepcopy(operator),'stats':copy.deepcopy(actual['stats'])})
            return record
        if provenance.get('image_generation_submissions')!=0:
            actual=json.loads((Path(asset['path'])/receipt['generation']['provider_receipt']['file']).read_text(encoding='utf8'))
            record=candidate_record(image,Path(root),provider=actual['provider'],model=actual['model'],
                prompt=actual.get('prompt',''),generation_id=actual.get('generation_id'))
            record.update(photo_appearance=copy.deepcopy(asset),provenance=str(Path(asset['path'])/receipt['generation']['provider_receipt']['file']))
            return record
    record=candidate_record(image,Path(root),provider=options.get('provider'),model=options.get('model'),
        prompt=str(provenance.get('prompt') or 'model-selected improvement'),
        generation_id=provenance.get('generation_id'))
    if selected.get('appearance_asset'):
        record['photo_appearance']=copy.deepcopy(selected['appearance_asset'])
        if provenance.get('provider'):record['provider']=provenance['provider']
        if provenance.get('model'):record['model']=provenance['model']
    if provenance.get('image_generation_submissions')==0:
        receipt=provenance['composition_receipt']
        if evidence_state is not None:
            from photo_session_evidence import resolve_file
            path=Path(resolve_file(evidence_state,receipt)['path'])
        else:path=Path(receipt['path'])
        if sha256_file(path)!=receipt['sha256']:raise ValueError('Photographic processing receipt changed')
        actual=json.loads(path.read_text(encoding='utf-8'))
        if (actual.get('image_generation_submissions')!=0
                or (actual.get('complete_candidate') or {}).get('sha256')!=selected['image']['sha256']):
            raise ValueError('Processing receipt does not bind this complete candidate')
        record.update(provider='local_photographic_processing',model=provenance['tool'],
            prompt='',generation_id=None,provenance=str(path),provenance_status='present',
            processing={'image_generation_submissions':0,'operator':copy.deepcopy(actual['focus_operator']),
                'composition_receipt':copy.deepcopy(receipt),
                'original_generated_image':copy.deepcopy(actual['original_generated_image'])})
    return record


def save_selected_candidate(root,run,args,*,selected,session_directory,cancel=None):
    """Save actual reviewed RGB/ICC once; swap the in-memory run only on success."""
    import single_source_workflow as workflow
    from PIL import Image,ImageChops
    root=Path(root).resolve()
    if selected is None:raise ValueError('A complete selected candidate is required')
    reviewed=_path(root,selected['image']['path'])
    if sha256_file(reviewed)!=selected['image']['sha256']:raise ValueError('Selected reviewed pixels changed')
    existing=(run.get('artifacts') or {}).get('auto_draft')
    current_asset=(run.get('options') or {}).get('photo_appearance') or {}
    selected_asset=selected.get('appearance_asset') or {}
    same_master=(current_asset.get('asset_id')==selected_asset.get('asset_id')
        and current_asset.get('receipt_sha256')==selected_asset.get('receipt_sha256'))
    if (same_master and existing and _path(root,existing).is_file()
            and sha256_file(_path(root,existing))==selected['image']['sha256']):
        return {'changed':False,'run':run,'rendered':None}
    if not selected.get('review'):
        import photo_edit_session as session
        state=session.snapshot(session_directory)
        allowed=session.unreviewed_initial_draft_allowed(state,selected)
        if not allowed:raise ValueError('Only a checked review-disabled initial draft may be saved without QA')
    if _cancelled(cancel):raise InterruptedError('Cancelled before saving selected pixels')
    if not selected.get('appearance_asset'):
        raise ValueError('New saved edits require their verified float appearance; historical PNG-only results are read-only')
    from generated_appearance_render import prepare_selected_appearance
    staged=prepare_selected_appearance(root,run,selected,session_directory=session_directory)
    review_path=root/'outputs/runs'/run['run_id']/'visual_review.json'
    previous_review=review_path.read_bytes() if review_path.is_file() else None
    produced_review_sha=None
    try:
        rendered=workflow.render_candidate(root,staged,args)
        if review_path.is_file():produced_review_sha=sha256_file(review_path)
        preview=Path(rendered['outputs']['auto_draft'])
        with Image.open(reviewed) as actual,Image.open(preview) as encoded:
            if (actual.mode!='RGB' or actual.size!=encoded.size or not actual.info.get('icc_profile')
                    or actual.info.get('icc_profile')!=encoded.info.get('icc_profile')
                    or ImageChops.difference(actual,encoded).getbbox() is not None):
                raise ValueError('Master-derived output differs from the actual reviewed candidate')
        temporary=preview.with_name(preview.name+'.reviewed-'+uuid4().hex+'.tmp')
        try:
            shutil.copyfile(reviewed,temporary)
            if sha256_file(temporary)!=selected['image']['sha256']:raise ValueError('Reviewed candidate changed during save')
            if _cancelled(cancel):raise InterruptedError('Cancelled after candidate render; prior complete draft retained')
            os.replace(temporary,preview)
        finally:temporary.unlink(missing_ok=True)
        rendered['metadata']['reviewed_preview']={'sha256':selected['image']['sha256'],
            'saved_bytes_identical':True,'master_RGB8_and_ICC_equal':True}
        workflow.write_json(Path(rendered['outputs']['metadata']),rendered['metadata'])
        if _cancelled(cancel):raise InterruptedError('Cancelled before selecting the newly saved candidate')
        staged.update(auto_draft=rendered,render=rendered['metadata'],review=rendered['review'],
            artifacts=rendered['outputs'],delivery_status='auto_draft')
        staged['artifacts']['visual_review']=rendered['review_path']
        run.clear();run.update(staged)
        return {'changed':True,'run':run,'rendered':rendered}
    except BaseException:
        # The renderer writes this run's review file. Restore only our own
        # unapproved write; a concurrent independent edit/approval is preserved.
        if produced_review_sha and review_path.is_file() and sha256_file(review_path)==produced_review_sha:
            if previous_review is None:review_path.unlink()
            else:
                temporary=review_path.with_name(review_path.name+'.restore-'+uuid4().hex)
                temporary.write_bytes(previous_review);os.replace(temporary,review_path)
        raise


def _native_plan(run):
    from photo_edit_session import INITIALIZATION
    plan=run.get('task_plan') or {}
    if (plan.get('mode')!='create_reference' or plan.get('editing_initialization')!=INITIALIZATION
            or not plan.get('editing_scope')):
        raise ValueError('Current source-native checked initialization is required; historical whole-image creation is read-only')
    return plan


def _connection(run,args):
    options=run.get('options') or {}
    return {key:getattr(args,key,None) or options.get(key) for key in
        ('provider','vision_model','api_base','api_url','api_key_env')}


def _callbacks(root,run,args,directory,hooks):
    from photo_edit_tools import make_tools
    from photo_session_decision import decide_images
    from photo_session_review import evaluate_images
    from runtime_resources import code_path
    options=run.get('options') or {};connection=_connection(run,args)
    marker=getattr(args,'cancel_request_path',None)
    cancelled=lambda:bool(marker and Path(marker).is_file())
    tools=make_tools(root,directory.parent/'editing-tools',scope=run['task_plan']['editing_scope'],
        session_directory=directory,provider_options={key:getattr(args,key,None) or options.get(key)
        for key in ('provider','model','api_base','api_url','api_key_env','size')},cancel=cancelled)
    producer={name:sha256_file(code_path(name)) for name in ('photo_edit_finish.py','photo_edit_session.py',
        'photo_edit_tools.py','photo_session_decision.py','photo_session_review.py','photo_request_budget.py','photo_appearance.py')}
    return (hooks.wrap_tools(tools),
        hooks.wrap_decide(lambda context:decide_images(context,directory.parent/'editing-decisions'/context['attempt_id'],**connection)),
        hooks.wrap_evaluate(lambda context:evaluate_images(context,directory.parent/'editing-reviews'/context['attempt_id'],**connection)),
        identity({'connection':connection,'producer':producer}),cancelled,producer)


def _usage(root,run,hooks):
    import single_source_workflow as workflow
    report=hooks.report()
    run.setdefault('generation_budget',{}).update(total_image_submissions=report['reserved_image_requests'],
        image_submissions=report['reserved_image_requests'],
        initial_operation_image_submissions=report['initial_operation_image_submissions'],
        editing_refinement_image_submissions=report['improvement_image_submissions'],
        confirmed_image_submissions=report['confirmed_image_submissions'],
        image_submission_count_uncertain=bool(report['unknown_image_reservations']),
        total_text_vision_requests=report['reserved_vision_requests'])
    workflow.write_json(root/'outputs/runs'/run['run_id']/'manifest.json',run)
    return report


def _progress(root,run,directory,current):
    import photo_edit_session as session
    import single_source_workflow as workflow
    stage=session.operation_stage(current,review=current['phase']=='evaluate')
    selected=session.selected_output(directory)['candidate']
    run['photo_edit_session']=session_summary(directory,current,selected)
    workflow.append_event(run,'preparing' if stage=='baseline' else 'finishing',
        detail='source_native:'+stage+':'+current['phase'],
        log_path=root/'outputs/runs'/run['run_id']/'logs/run.log.jsonl')
    workflow.write_json(root/'outputs/runs'/run['run_id']/'manifest.json',run)


def initial_candidates(root,run,args):
    """Create bounded variants through the same session, starting from source float.

    Called only by the existing checked coordinator after actual design observation.
    There is no mandatory full-image generation before model tool selection.
    """
    import photo_edit_session as session
    from photo_appearance import create_source_baseline,validate_asset
    from photo_request_budget import make_hooks
    root=Path(root).resolve();plan=_native_plan(run);scope=plan['editing_scope']
    design=run['generation_design']['observed_design'];run_dir=root/'outputs/runs'/run['run_id']
    source=run['source'];asset=run.get('initial_source_appearance')
    if asset is None:
        destination=run_dir/'appearance/source'
        if destination.is_dir():
            receipt=json.loads((destination/'receipt.json').read_text(encoding='utf8'))
            asset={'path':str(destination),'asset_id':receipt['asset_id'],'receipt_sha256':sha256_file(destination/'receipt.json')}
        else:
            marker=getattr(args,'cancel_request_path',None)
            asset=create_source_baseline(destination,source_path=source['path'],source_sha256=source['sha256'],
                scope=scope,design_id=design['design_id'],cancel=lambda:bool(marker and Path(marker).is_file()))
        validate_asset(asset,source_sha256=source['sha256'],scope=scope,design_id=design['design_id'])
        run['initial_source_appearance']=copy.deepcopy(asset)
    hooks=make_hooks(root,run,_connection(run,args));records=[]
    user=copy.deepcopy(run['generation_design'].get('art_direction',{}).get('user_intent') or {})
    user={key:str(user.get(key) or '') for key in ('brief','character_name','character_notes')}
    old={row['variant_id']:row for row in run.get('variants',[])}
    count=plan['image_budget']['initial_reference']
    for index in range(1,count+1):
        variant_id='v'+str(index);directory=run_dir/'variants'/variant_id/'editing-session'
        tools,decide,evaluate,model,cancelled,producer=_callbacks(root,run,args,directory,hooks)
        state=session.open_session(directory,source=source['path'],scope=scope,design=design,
            execution_identity=plan['plan_id'],producer_identity=identity(producer),budget=hooks.session_budget(),
            initial_usage=hooks.initial_usage(),user_intent=user,editing_initialization=plan['editing_initialization'],
            variant_id=variant_id,visual_review=run['task_intent']['services']['visual_review'],source_appearance=asset)
        if not state['candidates']:state=session.seed_source_baseline(directory,asset)
        if state['phase']=='evaluate' and state['pending_candidate_id']==state['baseline_candidate_id'] and not state['attempts']:
            cached=hooks.cached_review(session._context(state,review=True))
            if cached is not None:state=session.adopt_source_review(directory,cached)
        driven=drive_session(directory,tools=tools,decide=decide,evaluate=evaluate,model_identity=model,
            cancel=cancelled,hooks=hooks,before_step=lambda current:_progress(root,run,directory,current),
            after_step=lambda _: _usage(root,run,hooks))
        state=driven['state'];selected=driven['selected'];summary=session_summary(directory,state,selected)
        unchanged=bool(selected and selected['candidate_id']==state['baseline_candidate_id'])
        known_draft=session.unreviewed_initial_draft_allowed(state,selected)
        status=('cancelled' if driven['cancelled'] else 'needs_attention' if known_draft and (driven['failure'] or state['status']=='needs_attention')
            else 'failed' if driven['failure'] or state['status']=='needs_attention'
            else 'no_edited_candidate' if unchanged else 'complete')
        record={**copy.deepcopy(old.get(variant_id,{})),'variant_id':variant_id,'status':status,
            'pixel_status':'complete' if selected and status!='failed' else 'not_completed',
            'candidate_path':selected['image']['path'] if selected else None,
            'candidate_sha256':selected['image']['sha256'] if selected else None,
            'photo_appearance':copy.deepcopy((selected or {}).get('appearance_asset')),
            'photo_edit_session':summary,'generation_design':copy.deepcopy(run['generation_design']),
            'independent_evaluation':copy.deepcopy((selected or {}).get('review')),'failure':driven['failure']}
        run['photo_edit_session']=copy.deepcopy(summary)
        if selected:
            record['reference']=reviewed_candidate_record(root,selected,{**run.get('options',{}),'task_plan':plan},evidence_state=state)
            record.update(prompt=record['reference'].get('prompt',''),model=record['reference'].get('model'),
                style='main',style_label='候选'+str(index))
        records.append(record);run['variants']=copy.deepcopy(records);_usage(root,run,hooks)
        if status in {'failed','cancelled','needs_attention'}:break
    failed=[row['variant_id'] for row in records if row['status'] in {'failed','cancelled','needs_attention'}]
    if failed:run['partial_warning']={'failed_variants':failed,'automatic_image_retry':False}
    if not any(row['status'] in {'complete','no_edited_candidate','needs_attention'} for row in records):
        raise RuntimeError('No source-native variant completed its checked initial operation; inspect the saved request evidence')
    return records


def _selected_directory(root,run,variant):
    """Clone state into a child revision; parent candidate/master assets remain read-only."""
    import photo_edit_session as session
    from current_model_registry import atomic_json
    original=Path(variant['photo_edit_session']['path']).resolve().parent
    owned=root/'outputs/runs'/run['run_id']/'variants'/variant['variant_id']/'editing-session'
    if original!=owned:
        state=session.snapshot(original)
        if not (owned/'session.json').exists():
            from photo_session_evidence import freeze_session_evidence
            owned.mkdir(parents=True,exist_ok=True)
            state['evidence']=freeze_session_evidence(original,owned/'evidence',state=state)
            atomic_json(owned/'session.json',state)
        elif session.snapshot(owned)['session_id']!=state['session_id']:
            raise ValueError('Selected child session belongs to another source/variant')
        variant['photo_edit_session']['path']=str(owned/'session.json')
    return owned


def finish_saved_run(root,run,args):
    """Continue only the selected source-native session and save its true float asset."""
    import photo_edit_session as session
    import single_source_workflow as workflow
    from photo_request_budget import make_hooks
    root=Path(root).resolve();plan=_native_plan(run)
    variant=next((row for row in run.get('variants',[]) if row['variant_id']==run.get('selected_variant_id')),None)
    if not variant or not variant.get('photo_edit_session'):raise ValueError('Select an actual source-native variant first')
    directory=_selected_directory(root,run,variant);state=session.snapshot(directory)
    before_usage=copy.deepcopy(state['used'])
    if (not session.source_native(state) or state['binding']['variant_id']!=variant['variant_id']
            or state['binding']['execution_identity']!=plan['plan_id']):
        raise ValueError('Selected session does not share its checked source initialization')
    hooks=make_hooks(root,run,_connection(run,args))
    tools,decide,evaluate,model,cancelled,_=_callbacks(root,run,args,directory,hooks)
    state=session.resume_selected(directory)
    driven=drive_session(directory,tools=tools,decide=decide,evaluate=evaluate,model_identity=model,
        cancel=cancelled,hooks=hooks,before_step=lambda current:_progress(root,run,directory,current),
        after_step=lambda _: _usage(root,run,hooks))
    state=driven['state'];selected=driven['selected'];failure=driven['failure']
    if state['status']=='needs_attention' and not failure:
        failure=copy.deepcopy(variant.get('failure') or {'type':'UnresolvedReview','message':'Inspect existing review evidence without resubmitting'})
    if not driven['cancelled'] and selected and selected['candidate_id']!=state.get('baseline_candidate_id'):
        try:save_selected_candidate(root,run,args,selected=selected,session_directory=directory,cancel=cancelled)
        except Exception as error:failure={'type':type(error).__name__,'message':str(error)[:1600]}
    visible=selected
    if driven['cancelled'] or failure:
        active=(run.get('artifacts') or {}).get('auto_draft')
        active_path=_path(root,active) if active else None
        active_sha=sha256_file(active_path) if active_path and active_path.is_file() else None
        active_asset=(run.get('options') or {}).get('photo_appearance') or {}
        visible=next((row for row in state['candidates'] if row['image']['sha256']==active_sha
            and (row.get('appearance_asset') or {}).get('asset_id')==active_asset.get('asset_id')),None)
    summary=session_summary(directory,state,visible);origin=hooks.report()
    summary.update(origin_request_budget=origin,historical_design_requests=origin['historical_design_requests'],
        historical_image_submissions=origin['historical_initial_image_requests'])
    if driven['cancelled']:summary.update(status='cancelled',automated_target_met=False,objective_reviewed=False)
    if failure:
        actual=visible.get('review') if visible else None
        summary.update(status='needs_attention',failure=failure,session_best_target_met=state['automated_target_met'],
            automated_target_met=bool(actual and visible.get('appearance_kind')!='source_baseline'
                and visible.get('review_objective')=='current' and all(actual[axis]['status']=='pass' for axis in session.AXES)))
    run['photo_edit_session']=summary
    run['editing_autofinish_result']={'status':summary['status'],'phase':summary['phase'],'failure':failure,
        'new_requests':{key:state['used'][key]-before_usage[key] for key in state['used']},
        'new_requests_basis':'reserved_this_continuation; actual origin requests are reported by origin_budget',
        'origin_budget':origin,'human_approved':False,
        'selected_image':copy.deepcopy((visible or {}).get('image')),'reviewed_best_image':copy.deepcopy((selected or {}).get('image')),
        'selected_best_applied':not bool(failure or driven['cancelled']),
        'no_edited_candidate':summary['selected_is_source_baseline']}
    if driven['cancelled']:run['status']='cancelled'
    elif failure:run['status']='needs_attention'
    elif summary['selected_is_source_baseline']:run['status']='no_edited_candidate'
    else:run['status']='needs_visual_review'
    run.setdefault('artifacts',{})['photo_edit_session']=str(directory/'session.json')
    _usage(root,run,hooks)
    return run
