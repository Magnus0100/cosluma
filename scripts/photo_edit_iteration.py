"""Checked, separately budgeted improvement of an actually reviewed draft."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from asset_identity import sha256_file
from photo_editing import identity, validate_scope
from photo_session_evidence import resolve_file,resolve_asset,read_json

SCHEMA='cosluma_editing_refinement_authorization.v1'


def iteration_request(value):
    if not isinstance(value,dict) or set(value)-{'maximum_improvements','reason','user_feedback','maximum_image_submissions'}:
        raise ValueError('Editing iteration requires a bounded improvement request')
    count=value.get('maximum_improvements',1)
    reason=value.get('reason','')
    if type(count) is not int or not 1<=count<=8:
        raise ValueError('Maximum improvements must be an integer from 1 to 8')
    if not isinstance(reason,str) or len(reason)>8000:
        raise ValueError('Editing feedback must be text')
    feedback=value.get('user_feedback','')
    if not isinstance(feedback,str) or len(feedback)>8000:
        raise ValueError('User feedback must be text of at most 8000 characters')
    result={'maximum_improvements':count,'reason':reason.strip()}
    if 'maximum_image_submissions' in value:
        images=value['maximum_image_submissions']
        if type(images) is not int or not 0<=images<=count:
            raise ValueError('Image submissions must be an integer from zero to maximum improvements')
        result['maximum_image_submissions']=images
    if feedback.strip():result['user_feedback']=feedback.strip()
    return result


def _path(root,value):
    path=Path(value)
    return (path if path.is_absolute() else root/path).resolve()


def _working_evidence(root,state,*,scope,design,source_sha256,base_sha256):
    """Find the last actually reviewed local edit without adopting it as best."""
    rows=state.get('candidates') or []
    if not rows:return None
    candidate=rows[-1]
    if candidate.get('candidate_id')==state.get('best_candidate_id') or not candidate.get('review'):
        return None
    review=candidate['review']
    if ((candidate.get('image') or {}).get('sha256')==base_sha256
            or (review.get('creative_benefit') or {}).get('status')!='pass'
            or (review.get('credibility') or {}).get('status') not in {'fail','unknown'}):
        return None
    from photo_appearance import validate_asset
    asset=candidate.get('appearance_asset')
    if not asset:return None
    receipt=validate_asset(resolve_asset(state,asset),source_sha256=source_sha256,scope=scope,design_id=design['design_id'])
    if receipt['kind']!='mixed_regional':return None
    baseline=next((row for row in rows if row.get('candidate_id')==state.get('best_candidate_id')),None)
    if not baseline or not baseline.get('appearance_asset'):return None
    prior=validate_asset(resolve_asset(state,baseline['appearance_asset']),source_sha256=source_sha256,scope=scope,design_id=design['design_id'])
    if receipt['parent']!={'asset_id':prior['asset_id'],'master_sha256':prior['files']['master']['sha256'],
            'preview_sha256':prior['files']['preview']['sha256']} or prior['files']['preview']['sha256']!=base_sha256:
        return None
    image=candidate['image'];image_path=Path(resolve_file(state,image)['path'])
    if sha256_file(image_path)!=image['sha256'] or image['sha256']!=receipt['files']['preview']['sha256']:
        raise ValueError('Reviewed working preview differs from its actual float master')
    from photo_focus_integration import make_focus_tool
    focus=make_focus_tool(root,root/'outputs/focus-eligibility',scope=scope,session_directory=root)
    if focus.input_check(state) is not True:return None
    attempt=next((row for row in state.get('attempts',[]) if row.get('response')==review
        and row.get('status')=='completed' and row.get('kind') in {'evaluate','adopted_evaluation'}),None)
    if not attempt:raise ValueError('Working candidate has no completed independent observation')
    context=attempt.get('context') or {}
    from photo_session_review import _verify_images,_validate_payload
    if not isinstance(context.get('images'),list):
        raise ValueError('Working review requires its actual observed image roles')
    _verify_images([{**row,**resolve_file(state,row)} for row in context['images']])
    images=context['images']
    if (context.get('scope')!=scope or context.get('design')!=design
            or (context.get('source_original') or images[0])['sha256']!=source_sha256 or images[1]['sha256']!=image['sha256']
            or review.get('observed_images')!=[{key:row[key] for key in ('role','sha256')} for row in images]):
        raise ValueError('Working review belongs to different inputs or editing permissions')
    _validate_payload({key:review.get(key) for key in ('credibility','creative_benefit','comparison','primary_issue')},
        len(images)==3)
    provenance=review.get('provenance') or {};board=provenance.get('review_board') or {}
    if (board!=provenance.get('source') or not board.get('path')
            or resolve_file(state,board)['sha256']!=board.get('sha256')
            or provenance.get('network_used') is not True or provenance.get('generation_used') is not False
            or not provenance.get('model')):
        raise ValueError('Working candidate requires unchanged actual independent review evidence')
    proof={'candidate_id':candidate['candidate_id'],'image_sha256':image['sha256'],
        'review_sha256':identity(review),'review_context_sha256':identity(context),
        'appearance_asset':copy.deepcopy(asset),'appearance_receipt_sha256':asset['receipt_sha256']}
    return {'candidate':candidate,'review_context':context,'proof':proof}


def authorize(root, *, parent_run_id, source, reference, scope, iteration,provider_target=None):
    """Read-only preflight: bind a saved parent and its real review evidence."""
    root=Path(root).resolve();scope=validate_scope(scope);iteration=iteration_request(iteration)
    if (not isinstance(parent_run_id,str) or not parent_run_id.strip()
            or parent_run_id in {'.','..'} or '/' in parent_run_id or '\\' in parent_run_id):
        raise ValueError('Editing improvement requires its saved parent version')
    manifest=(root/'outputs/runs'/parent_run_id/'manifest.json').resolve()
    if not manifest.is_relative_to(root/'outputs/runs'):
        raise ValueError('Parent version lies outside this workspace')
    manifest_sha=sha256_file(manifest)
    parent=json.loads(manifest.read_text(encoding='utf-8'))
    if parent.get('run_id')!=parent_run_id:
        raise ValueError('Saved parent identity differs from the requested version')
    if (parent.get('result_publication') or {}).get('state')!='committed':
        raise ValueError('Editing improvement requires a completely saved parent')
    if (parent.get('options',{}).get('shaping_state') or {}).get('enabled') is True:
        raise ValueError('Active shaping supports strength changes and exact restore; ordinary feedback editing of this shaping result is not supported')
    if ((parent.get('photo_edit_session') or {}).get('status')=='needs_attention'
            or (parent.get('editing_iteration_result') or {}).get('failure')):
        raise ValueError('Inspect the existing unresolved editing attempt before another image request')
    parent_scope=(parent.get('task_plan') or {}).get('editing_scope')
    if parent_scope!=scope:
        raise ValueError('Improvement cannot change the parent editing permissions')
    draft=_path(root,(parent.get('artifacts') or {})['auto_draft'])
    source_path=_path(root,parent['source']['path'])
    source_sha=sha256_file(source_path);draft_sha=sha256_file(draft)
    if (source['sha256']!=source_sha or reference['sha256']!=draft_sha
            or source_sha!=parent['source']['sha256']
            or _path(root,source['path'])!=source_path or _path(root,reference['path'])!=draft):
        raise ValueError('Improvement inputs must be the saved original and current complete draft')
    generation=parent.get('generation_design') or {}
    design=generation.get('observed_design') or {}
    from generated_appearance_render import validate_run
    validate_run(root,parent)
    review=(parent.get('photo_edit_session') or {}).get('independent_evaluation') or {}
    from photo_appearance import validate_asset
    base_asset=parent.get('options',{}).get('photo_appearance')
    source_asset=parent.get('initial_source_appearance')
    if not base_asset or not source_asset:
        raise ValueError('Historical results are read-only; import an explicitly current typed appearance before improvement')
    base_receipt=validate_asset(base_asset,source_sha256=source_sha,scope=scope,design_id=design['design_id'])
    source_receipt=validate_asset(source_asset,source_sha256=source_sha,scope=scope,design_id=design['design_id'])
    if base_receipt['files']['preview']['sha256']!=draft_sha:raise ValueError('Saved preview differs from its float appearance')
    expected=[{'role':'source','sha256':source_receipt['files']['preview']['sha256']},{'role':'candidate','sha256':draft_sha}]
    provenance=review.get('provenance') or {}
    from photo_session_review import _validate_payload
    _validate_payload({key:review.get(key) for key in ('credibility','creative_benefit','comparison','primary_issue')},
        len(review.get('observed_images') or [])==3)
    board=provenance.get('review_board') or {}
    if (board!=provenance.get('source') or not board.get('path')):
        raise ValueError('Parent independent review observation changed')
    session_path=_path(root,parent['photo_edit_session']['path'])
    session_sha=sha256_file(session_path)
    import photo_edit_session as session
    saved_session=session.read_saved(session_path.parent)
    resolve_file(saved_session,board)
    best=next((row for row in saved_session.get('candidates',[])
        if row.get('candidate_id')==saved_session.get('best_candidate_id')),None)
    if (saved_session.get('session_id')!=parent['photo_edit_session'].get('session_id')
            or saved_session.get('session_id')!=identity(saved_session.get('binding'))
            or not best or best.get('review')!=review
            or best.get('image',{}).get('sha256')!=draft_sha):
        raise ValueError('Parent review differs from its persisted independent observation')
    observed=review.get('observed_images') or []
    if (len(observed) not in (2,3) or observed[:2]!=expected
            or (len(observed)==3 and observed[2].get('role')!='previous_best')
            or provenance.get('network_used') is not True
            or provenance.get('generation_used') is not False or not provenance.get('model')
            or not design.get('design_id') or generation.get('scope_id')!=scope['scope_id']):
        raise ValueError('Improvement requires independent evidence for these exact saved pixels')
    if (not iteration.get('user_feedback') and session.creative_delivery_eligibility(saved_session,best)['eligible'] and
            all((review.get(axis) or {}).get('status')=='pass' for axis in ('credibility','creative_benefit'))):
        raise ValueError('Both reviewed targets already pass; explicit user feedback is required for another improvement')
    working=_working_evidence(root,saved_session,scope=scope,design=design,
        source_sha256=source_sha,base_sha256=draft_sha)
    image_free_tools=[]
    if iteration.get('maximum_image_submissions')==0:
        from photo_edit_tools import make_tools
        tools=make_tools(root,root/'outputs/readonly-capability-check',scope=scope,
            session_directory=session_path.parent)
        image_free_tools=[tool.name for tool in tools if tool.costs.get('image_submissions',0)==0
            and session.tool_inputs_available(saved_session,tool)]
        if not image_free_tools:
            raise ValueError('An image-free improvement requires a genuinely eligible photographic capability')
    from photo_request_budget import manual_target_key,check_manual_target
    manual_target=manual_target_key(source_sha256=source_sha,base_sha256=draft_sha,
        scope_id=scope['scope_id'],design_id=design['design_id'],
        user_feedback=iteration.get('user_feedback',''),provider_target=provider_target)
    check_manual_target(root,manual_target)
    if sha256_file(manifest)!=manifest_sha or sha256_file(session_path)!=session_sha:
        raise ValueError('Parent changed while checking improvement')
    return {'schema':SCHEMA,'parent_run_id':parent_run_id,'parent_manifest_sha256':manifest_sha,
        'parent_session_sha256':session_sha,
        'source_sha256':source_sha,'base_image_sha256':draft_sha,
        'parent_candidate_sha256':parent['reference']['sha256'],
        'parent_appearance_id':base_asset['asset_id'],'parent_appearance':copy.deepcopy(base_asset),
        'source_appearance':copy.deepcopy(source_asset),'parent_master_sha256':base_receipt['files']['master']['sha256'],
        'scope_id':scope['scope_id'],'design_id':design['design_id'],
        'review_sha256':identity(review),'iteration':iteration,
        'working_candidate':working['proof'] if working else None,
        'image_free_tools':image_free_tools,
        'manual_provider_target':copy.deepcopy(provider_target),'manual_target':manual_target}


def revalidate(root,plan):
    inputs=plan['inputs'];authorization=plan['editing_refinement_authorization']
    current=authorize(root,parent_run_id=authorization['parent_run_id'],source=inputs['source'],
        reference=inputs['reference'],scope=plan['editing_scope'],iteration=plan['editing_iteration'],
        provider_target=authorization.get('manual_provider_target'))
    if current!=authorization:
        raise ValueError('Saved draft or review changed after checking improvement')
    return current


def execute(root,args):
    """Claim this exact target before entering its one checked workflow."""
    from photo_request_budget import manual_provider_target,manual_target_guard
    root=Path(root).resolve();plan=args.task_plan;authorization=revalidate(root,plan)
    target=manual_provider_target(vars(args),image_generation=plan['image_budget']['editing_refinement']>0)
    if target!=authorization['manual_provider_target']:
        raise ValueError('Manual provider target changed after planning; check a new plan before execution')
    with manual_target_guard(root,authorization['manual_target'],run_id=args.run_id):
        return _execute_authorized(root,args,authorization)


def _execute_authorized(root,args,authorization):
    """Run bounded model-selected tools inside the existing coordinator."""
    import single_source_workflow as workflow
    import photo_edit_session as session
    from photo_edit_tools import make_tools
    from photo_session_decision import decide_images
    from photo_session_review import evaluate_images
    from photo_edit_finish import drive_session,save_selected_candidate,session_summary,reviewed_candidate_record
    from raw_support import dimensions
    from runtime_resources import code_path
    root=Path(root).resolve();plan=copy.deepcopy(args.task_plan)
    from runtime_preflight import preflight_for_run
    runtime=preflight_for_run(root,registry_path=getattr(args,'model_registry',None),
        require_local_models=False,mode='existing_reference',editing_scope=plan['editing_scope'])
    _,parent=workflow.load_run(root,authorization['parent_run_id'])
    path,run=workflow._source_revision(root,parent['run_id'],snapshot=False,
        child_run_id=args.run_id)
    run['task_plan']=copy.deepcopy(plan);run['task_intent']=copy.deepcopy(args.task_intent)
    options=run.setdefault('options',{})
    # _source_revision has materialized both immutable assets in this child.
    # Retain those owned descriptors; historical review/context bytes stay as
    # observed and are never relabelled as a fresh observation.
    from photo_appearance import validate_asset
    asset_binding={'source_sha256':run['source']['sha256'],
        'scope':plan['editing_scope'],'design_id':run['generation_design']['observed_design']['design_id']}
    source_receipt=validate_asset(run['initial_source_appearance'],**asset_binding)
    base_receipt=validate_asset(options['photo_appearance'],**asset_binding)
    if (source_receipt['kind']!='source_baseline'
            or base_receipt['files']['preview']['sha256']!=authorization['base_image_sha256']
            or base_receipt['asset_id']!=authorization['parent_appearance_id']):
        raise ValueError('Owned child appearances differ from the checked parent pixels')
    options['runtime_preflight']=runtime
    options.update(task_plan=copy.deepcopy(plan),task_intent=copy.deepcopy(args.task_intent),
        editing_request=copy.deepcopy(plan['editing_request']),editing_iteration=copy.deepcopy(plan['editing_iteration']))
    for key in ('provider','model','vision_model','api_base','api_url','api_key_env','model_registry','preserve_project_selection'):
        if getattr(args,key,None) is not None:
            value=getattr(args,key)
            options[key]=str(value) if isinstance(value,Path) else value
    workflow._begin_result_publication(run,operation='editing_improvement',checked=True)
    budget={'image_submissions':plan['image_budget']['editing_refinement'],
        'vision_requests':plan['services']['text_vision_requests_upper_bound'],
        'local_repairs':plan['editing_iteration']['maximum_improvements'],
        'tool_calls':plan['editing_iteration']['maximum_improvements'],
        'improvement_steps':plan['editing_iteration']['maximum_improvements']}
    run['generation_budget']={'origin_run_id':run['run_id'],'max_image_submissions':budget['image_submissions'],
        'image_submissions':0,'total_image_submissions':0,'automatic_image_retry':False,
        'historical_generation_parent_run_id':parent['run_id'],'planning_requests_this_attempt':0}
    generation=run['generation_design'];scope=plan['editing_scope']
    original_intent=copy.deepcopy((generation.get('art_direction') or {}).get('user_intent') or {})
    user_intent={key:str(original_intent.get(key) or '') for key in ('brief','character_name','character_notes')}
    # reason describes this execution; only explicit user_feedback changes the
    # review objective. Do not silently append it to the original user brief.
    feedback=plan['editing_iteration'].get('user_feedback')
    if feedback:user_intent['user_feedback']=feedback
    directory=path.parent/'editing-session'
    log_path=path.parent/'logs/run.log.jsonl'
    producer={name:sha256_file(code_path(name)) for name in (
        'photo_edit_iteration.py','photo_edit_finish.py','photo_edit_tools.py','photo_edit_session.py',
        'photo_session_decision.py','photo_session_review.py')}
    state=session.open_session(directory,source=run['source']['path'],scope=scope,
        design=generation['observed_design'],execution_identity=plan['plan_id'],
        producer_identity=identity(producer),budget=budget,user_intent=user_intent,
        appearance_required=True,source_appearance=run['initial_source_appearance'])
    parent_session_directory=_path(root,parent['photo_edit_session']['path']).parent
    previous=session.snapshot(parent_session_directory)
    state=session.own_evidence(directory,source_directory=parent_session_directory,source_state=previous)
    base=Path(options['photo_appearance']['path'])/base_receipt['files']['preview']['file']
    parent_review=parent['photo_edit_session']['independent_evaluation']
    baseline_provenance={'parent_run_id':parent['run_id'],'authorization':authorization}
    if feedback or len(parent_review.get('observed_images') or [])==3:
        observed_attempt=next(row for row in previous['attempts'] if row.get('response')==parent_review
            and row.get('status')=='completed' and row.get('kind') in {'evaluate','adopted_evaluation'})
        if feedback:baseline_provenance['review_context']=copy.deepcopy(observed_attempt['context'])
        baseline_provenance['review_images']=copy.deepcopy(observed_attempt['context']['images'])
    state=session.adopt_reviewed_candidate(directory,base,review=parent_review,provenance=baseline_provenance,
        baseline_only=bool(feedback),appearance_asset=options['photo_appearance'])
    if authorization.get('working_candidate'):
        working=_working_evidence(root,previous,scope=scope,design=generation['observed_design'],
            source_sha256=run['source']['sha256'],base_sha256=authorization['base_image_sha256'])
        if not working or working['proof']!=authorization['working_candidate']:
            raise ValueError('Checked working candidate changed before adoption')
        observed=working['candidate']
        state=session.adopt_reviewed_working_candidate(directory,resolve_file(state,observed['image'])['path'],
            review=observed['review'],provenance=observed['provenance'],review_context=working['review_context'],
            appearance_asset=resolve_asset(state,observed['appearance_asset']))
    connection={key:getattr(args,key,None) or options.get(key) for key in
        ('provider','vision_model','api_base','api_url','api_key_env')}
    provider_options={key:getattr(args,key,None) or options.get(key) for key in
        ('provider','model','api_base','api_url','api_key_env','size')}
    def cancelled():
        marker=getattr(args,'cancel_request_path',None)
        return bool(marker and Path(marker).is_file())
    tools=make_tools(root,path.parent/'editing-tools',scope=scope,session_directory=directory,
        provider_options=provider_options,cancel=cancelled)
    def decide(context):return decide_images(context,path.parent/'editing-decisions'/context['attempt_id'],**connection)
    def evaluate(context):return evaluate_images(context,path.parent/'editing-reviews'/context['attempt_id'],**connection)
    def progress(current):
        chosen=(current.get('pending_decision') or {}).get('tool')
        generating=current['phase']=='tool' and any(tool.name==chosen and
            tool.costs.get('image_submissions',0)>0 for tool in tools)
        workflow.append_event(run,'generating' if generating else 'finishing',
            detail='model_selected_improvement:'+current['phase'],log_path=log_path)
        workflow.write_json(path,run)
    def record_usage(current):
        run['generation_budget'].update(image_submissions=current['used']['image_submissions'],
            total_image_submissions=current['used']['image_submissions'],
            text_vision_submissions=current['used']['vision_requests'])
        workflow.write_json(path,run)
    driven=drive_session(directory,tools=tools,decide=decide,evaluate=evaluate,
        model_identity=identity({'connection':connection,'producer':producer}),cancel=cancelled,
        before_step=progress,after_step=record_usage)
    state=driven['state'];selected=driven['selected'];failure=driven['failure']
    if driven['cancelled']:
        workflow.append_event(run,'cancelled',detail='improvement_cancelled_parent_preserved',log_path=log_path)
        workflow.write_json(path,run)
        return run
    if selected is None:raise RuntimeError('No complete baseline survived editing improvement')
    improved=selected['attempt_id']!='enclosing_checked_task' and selected['image']['sha256']!=authorization['base_image_sha256']
    # The initial adopted pixels remain the fallback; only a reviewed better
    # complete candidate can replace the parent appearance.
    if improved:
        run['reference']=reviewed_candidate_record(root,selected,options,evidence_state=state)
    run['editing_iteration_result']={'authorization':authorization,'session':str(directory/'session.json'),
        'session_sha256':sha256_file(directory/'session.json'),'selected_candidate_id':selected['candidate_id'],
        'selected_image':copy.deepcopy(selected['image']),'improved':improved,'failure':failure,
        'new_requests':copy.deepcopy(state['used']),'new_requests_basis':'reserved_session_usage',
        'human_approved':False}
    binding={'schema':'cosluma_derived_edit_candidate.v1','scope_id':scope['scope_id'],
        'design_id':generation['observed_design']['design_id'],
        'source_sha256':run['source']['sha256'],'base_image_sha256':authorization['base_image_sha256'],
        'parent_candidate_sha256':authorization['parent_candidate_sha256'],
        'candidate_sha256':run['reference']['sha256'],'improved':improved,
        'selected_image_sha256':selected['image']['sha256'],
        'selected_provenance_sha256':identity(selected['provenance']),
        'selected_review_sha256':identity(selected['review']),
        'session_id':state['session_id'],'producer':producer}
    binding['binding_id']=identity(binding)
    run['derived_edit_candidate']=binding
    workflow.append_event(run,'rendering',detail='save_best_reviewed_complete_version',log_path=log_path)
    save_selected_candidate(root,run,args,selected=selected,session_directory=directory,cancel=cancelled)
    run['artifacts']['photo_edit_session']=str(directory/'session.json')
    run['photo_edit_session']=session_summary(directory,state,selected)
    if not improved:run['warning']='Improvement did not produce a reviewed better version; the complete baseline is retained.'
    workflow.append_event(run,'needs_visual_review',detail='reviewed_improvement' if improved else 'baseline_retained',log_path=log_path)
    workflow.write_json(path,run)
    workflow.update_project_state(root,run)
    return run
