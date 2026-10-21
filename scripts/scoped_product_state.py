"""Read-only product facts for generated creation and reviewed improvement.

Execution, publication, automated findings and human approval stay separate.
No model, pixel-array decoding or parent timing reuse occurs here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

AXES={'credibility':'自然可信度','creative_benefit':'创作收益'}
STATUS={'complete':'已完成','pending':'待处理','running':'处理中','needs_attention':'需核查',
        'failed':'执行失败','cancelled':'已取消','not_requested':'本轮未调用','partial':'部分完成'}
STAGE_MODULES={
    'photo_source':('source_import',), 'observe_design':('visual_art_direction',),
    'generate_edit':('generation','provider_transport'), 'reviewed_inputs':('reopen_manifest','manifest_read'),
    'decide_edit':('creative_decision_board','creative_tool_decision'),
    'selected_edit_tool':('provider_transport','photographic_focus_integration','photographic_development'),
    'independent_review':('independent_review_board','independent_image_review'),
    'independent_recheck':('independent_review_board','independent_image_review'),
    'freeze_appearance':('photo_appearance_freeze','photo_appearance_validation'),
    'save_appearance':('photo_appearance_render','render_export'),
    'source_native_baseline':('source_native_baseline','photo_source_baseline'),
    'baseline_review':(),
    'select_best':(), 'export':('export',)}


def selected(run):
    from scoped_workflow import selected as scoped
    plan=run.get('task_plan') or {}
    intent=run.get('task_intent') or (run.get('options') or {}).get('task_intent') or {}
    return scoped(intent.get('mode'),plan.get('editing_scope'),plan.get('editing_iteration'))


def _json(path,limit=4*1024*1024):
    try:
        if not path.is_file() or path.stat().st_size>limit:return {}
        value=json.loads(path.read_text(encoding='utf-8-sig'))
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):return {}


def observe(root,run,manifest_path=None):
    from scoped_workflow import build_scoped_graph
    from artifact_receipts import observe_receipts
    root=Path(root).resolve();plan=run.get('task_plan') or {}
    intent=run.get('task_intent') or (run.get('options') or {}).get('task_intent') or {}
    def path(value):
        item=Path(value);return item if item.is_absolute() else root/item
    graph=build_scoped_graph({**run,'root':str(root)},plan['editing_scope'],
        bool((intent.get('services') or {}).get('visual_review')),editing_iteration=plan.get('editing_iteration'),
        editing_autofinish=plan.get('editing_autofinish'))
    ledger_binding=run.get('workflow_timing') or {};timing={}
    if ledger_binding.get('path'):
        candidate=_json(path(ledger_binding['path']))
        if (candidate.get('schema')=='cosplay_workflow_timing.v1'
                and candidate.get('request_id')==ledger_binding.get('request_id')
                and run.get('run_id') in candidate.get('run_ids',[])):
            timing=candidate
    review={}
    review_path=(run.get('artifacts') or {}).get('visual_review')
    if review_path:review=_json(path(review_path))
    evaluation=(run.get('photo_edit_session') or {}).get('independent_evaluation') or {}
    session_path=(run.get('photo_edit_session') or {}).get('path')
    state=_json(path(session_path)) if session_path else {}
    best=next((row for row in state.get('candidates',[]) if row.get('candidate_id')==state.get('best_candidate_id')),None)
    binding_good=bool(best and best.get('review')==evaluation
        and any(row.get('role')=='candidate' and row.get('sha256')==(best.get('image') or {}).get('sha256')
            for row in evaluation.get('observed_images',[])))
    from scoped_workflow import _native_asset_fact
    native=bool((run.get('options') or {}).get('photo_appearance') or plan.get('editing_initialization'))
    source_sha=(run.get('source') or {}).get('sha256')
    if native:
        receipt=_native_asset_fact(root,run.get('initial_source_appearance') or state.get('source_appearance'),run,plan['editing_scope'])
        source_sha=((receipt or {}).get('files',{}).get('preview') or {}).get('sha256')
        binding_good=bool(binding_good and source_sha
            and (state.get('binding') or {}).get('source',{}).get('sha256')==(run.get('source') or {}).get('sha256'))
    manifest=manifest_path or root/'outputs/runs'/str(run.get('run_id') or '')/'manifest.json'
    return {'scoped_graph':graph,'workflow_timing':timing,'review':review,'independent_review_binding':binding_good,
        'independent_review_source_sha256':source_sha,
        'independent_review_objective':best.get('review_objective','current') if binding_good else None,
        'assets':{'manifest':Path(manifest).is_file()},'artifact_receipts':observe_receipts(root,run)}


def timing_summary(run,ledger):
    """Use this request's measured exclusive intervals, never add nested spans."""
    from workflow_timing import MODULE_LABELS
    valid=(ledger.get('schema')=='cosplay_workflow_timing.v1'
        and ledger.get('request_id')==(run.get('workflow_timing') or {}).get('request_id')
        and run.get('run_id') in ledger.get('run_ids',[]))
    if not valid:return {'available':False,'basis':'no_current_request_ledger','modules':[]}
    totals={}
    for span in ledger.get('spans') or []:
        if span.get('parent_id') is None:continue
        name=span.get('name');value=span.get('exclusive_seconds')
        if not isinstance(name,str) or type(value) not in {int,float} or not math.isfinite(value) or value<0:continue
        row=totals.setdefault(name,{'name':name,'label':MODULE_LABELS.get(name,name),'calls':0,'seconds':0.,'status':'complete'})
        if name=='provider_transport':row['label']='执行选定图像编辑' if (run.get('task_plan') or {}).get('editing_iteration') or (run.get('task_plan') or {}).get('editing_initialization') else '生成完整编辑图'
        if name=='provider_transport' and not (run.get('task_plan') or {}).get('editing_initialization') and (run.get('task_plan') or {}).get('editing_autofinish',{}).get('maximum_improvements'):
            row['label']='图像模型生成／条件编辑'
        if name=='generation':row['label']='生成调度与保存'
        if name=='create_run':row['label']='任务编排自身开销'
        if name=='render_candidate':row['label']='成片保存调度'
        row['calls']+=1;row['seconds']+=value
        if span.get('status') in {'running','failed','cancelled'}:row['status']=span['status']
    # No-op compatibility wrappers are not evidence that Beauty ran. Keep their
    # measured overhead in the full ledger rather than listing a beauty module.
    totals.pop('portrait_beauty',None)
    modules=sorted(totals.values(),key=lambda row:(-row['seconds'],row['name']))
    return {'available':True,'request_id':ledger['request_id'],'status':ledger.get('status'),
        'whole_seconds':ledger.get('whole_wall_seconds'),'modules':modules,
        'basis':'current_request_measured_exclusive_wall_seconds','snapshot_only':ledger.get('status')=='running',
        'parent_timing_included':False,'nested_durations_additive':False}


def build(run,observations):
    from pipeline_stage_report import _export_receipt_state, _task_summary, VISUAL_CHECK_KEYS
    graph=observations.get('scoped_graph') or {};nodes=graph.get('nodes') or []
    states={row['id']:row['status'] for row in nodes}
    plan=run.get('task_plan') or {};iteration=plan.get('editing_iteration') is not None
    publication=run.get('result_publication') or {};committed=publication.get('state')=='committed'
    native=bool(plan.get('editing_initialization'))
    auto=bool(not native and (plan.get('editing_autofinish') or {}).get('maximum_improvements'))
    rendered=states.get('save_best' if auto else 'save_appearance')=='complete'
    saved=bool(rendered and committed)
    session=run.get('photo_edit_session') or {};evaluation=session.get('independent_evaluation') or {}
    eligibility=session.get('creative_delivery_eligibility') or {}
    photographic_only=eligibility.get('eligible') is False and graph.get('selected_appearance_kind')=='photographic_adjustment'
    origin=session.get('origin_request_budget') or (run.get('editing_autofinish_result') or {}).get('origin_budget') or {}
    usage=None
    if origin.get('origin_plan_id')==plan.get('plan_id') and origin.get('origin_run_id'):
        usage={'basis':'saved_origin_budget_snapshot_including_design',
            'origin_run_id':origin['origin_run_id'],
            'image_submissions':{key:origin.get(field) for key,field in
                (('confirmed','confirmed_image_submissions'),('reserved','reserved_image_requests'),('limit','checked_image_limit'))},
            'vision_requests':{key:origin.get(field) for key,field in
                (('confirmed','confirmed_vision_requests'),('reserved','reserved_vision_requests'),('limit','checked_vision_limit'))},
            'local_session_reservations':session.get('reserved_budget_usage') or {},
            'local_session_reservations_are_actual_http_totals':False}
    provenance=evaluation.get('provenance') or {}
    review_valid=bool(observations.get('independent_review_binding') and evaluation and provenance.get('network_used') is True
        and provenance.get('generation_used') is False and provenance.get('model')
        and any(row.get('role')=='source' and row.get('sha256')==observations.get('independent_review_source_sha256',(run.get('source') or {}).get('sha256'))
                for row in evaluation.get('observed_images',[])))
    feedback=(plan.get('editing_iteration') or {}).get('user_feedback') or ''
    objective_reviewed=bool(review_valid and (not feedback
        or observations.get('independent_review_objective')=='current'))
    checks=[{'id':key,'label':label,'status':(evaluation.get(key) or {}).get('status','unknown'),
        'reason':'；'.join(str(item) for item in (evaluation.get(key) or {}).get('findings',[])),
        'action':evaluation.get('primary_issue') or '对照原片与成片复核。'} for key,label in AXES.items()] if objective_reviewed else []
    timing=timing_summary(run,observations.get('workflow_timing') or {})
    terminal_failure=run.get('status') in {'failed','blocked','render_failed','local_reconstruction_failed'}
    failure=((run.get('editing_iteration_result') or {}).get('failure')
        or (run.get('editing_autofinish_result') or {}).get('failure'))
    cancelled=run.get('status')=='cancelled'
    executing=(not committed and not cancelled and not terminal_failure and not failure
        and timing.get('status') not in {'failed','cancelled'})
    human=observations.get('review') or {}
    approved=bool(saved and run.get('status') in {'accepted','delivered'}
        and human.get('run_id')==run.get('run_id') and human.get('revision')==run.get('revision')
        and human.get('status') in {'accepted','delivered'}
        and all((human.get('checks') or {}).get(key)=='pass' for key in VISUAL_CHECK_KEYS))
    if cancelled:
        execution,state,label='cancelled','cancelled','本次制作已取消，原保存版本保留'
    elif saved and not terminal_failure and (failure or session.get('status')=='needs_attention' or run.get('status')=='needs_attention'):
        execution,state,label='complete','needs_request_attention','完整画面已保存；审片或后续请求结果需要核查'
    elif terminal_failure or failure or timing.get('status')=='failed':
        execution,state,label='failed','needs_execution_recovery','本次执行需要核查；已有完整画面保留'
    elif not committed and rendered:
        execution,state,label='pending_publication','pending_publication','成片已生成，保存提交尚未完成'
    elif executing:
        execution,state,label='running','in_progress','正在按独立审片改进' if iteration else '正在制作角色成片'
    elif approved:
        execution,state,label='complete','accepted','已人工批准'
    elif saved and native and not graph.get('edited_output'):
        execution,state,label='complete','needs_editing','原图基线已保留，尚无获准的实际修改'
    elif saved and photographic_only:
        execution,state,label='complete','needs_creative_edit','摄影调整稿已保存，场景创作目标尚未完成'
    elif saved and feedback and not objective_reviewed:
        execution,state,label='complete','needs_evidence','完整版本已保存，本轮用户反馈尚未验证'
    elif saved and any(row['status']=='fail' for row in checks):
        execution,state,label='complete','needs_repair','制作已完成，成片有待改善项'
    elif saved and (not checks or any(row['status']=='unknown' for row in checks)):
        execution,state,label='complete','needs_evidence','制作已完成，画质证据待补'
    elif saved:
        execution,state,label='complete','needs_visual_review','制作已完成，等待人工审片批准'
    else:
        execution,state,label='pending_publication','pending_publication','等待完整成片保存提交'
    if session.get('status')=='needs_attention' and execution=='complete':
        state,label='needs_request_attention','完整画面已保存；审片或后续请求结果需要核查'
    next_action=('先核查已有请求与失败记录；保留当前完整画面，不自动重发模型请求。'
        if execution=='failed' or state=='needs_request_attention' else
        '先核查完整保存提交，现有图像文件与原版本保留。' if execution=='pending_publication' else
        '原保存版本保留，可在任务中心查看取消结果。' if cancelled else
        '曝光与色偏调整已保存；仍需符合制作范围的场景或效果收益，自动审片通过不替代此目标。' if photographic_only else
        '本轮反馈尚无当前目标的审片证据；父版本历史通过不代表已满足反馈。' if feedback and not objective_reviewed else
        evaluation.get('primary_issue') if checks and any(row['status']!='pass' for row in checks) else '核对保存画面、审片与导出。')
    stages=[]
    auto_seconds={}
    if auto and timing.get('available'):
        shared={'provider_transport':('generate_edit','selected_edit_tool'),
            'generated_appearance_freeze':('freeze_appearance','freeze_best'),
            'generated_appearance_render':('save_appearance','save_best'),'render_export':('save_appearance','save_best'),
            'independent_review_board':('independent_review','independent_recheck'),
            'independent_image_review':('independent_review','independent_recheck')}
        seen={}
        for span in (observations.get('workflow_timing') or {}).get('spans',[]):
            name=span.get('name')
            if name in {'photographic_focus_integration','photographic_development'}:
                auto_seconds['selected_edit_tool']=auto_seconds.get('selected_edit_tool',0.)+float(span.get('exclusive_seconds') or 0)
                continue
            if name not in shared:continue
            index=seen.get(name,0);seen[name]=index+1
            stage=shared[name][int(index>0)]
            auto_seconds[stage]=auto_seconds.get(stage,0.)+float(span.get('exclusive_seconds') or 0)
        for stage,names in (('generate_edit',('generation',)),('save_best',('production_contract_save',))):
            auto_seconds[stage]=auto_seconds.get(stage,0.)+sum(row['seconds'] for row in timing['modules'] if row['name'] in names)
    for node in nodes:
        status=node['status']
        detail=node['detail']
        if node['id']=='save_appearance' and rendered and not committed:
            status='partial';detail+=' 成片文件存在，但本次保存提交尚未完成。'
        measured=bool(timing.get('available') and any(row['name'] in STAGE_MODULES.get(node['id'],()) for row in timing['modules']))
        seconds=sum(row['seconds'] for row in timing.get('modules',[]) if row['name'] in STAGE_MODULES.get(node['id'],()))
        if auto and node['id'] in {'generate_edit','selected_edit_tool','freeze_appearance','freeze_best','save_appearance','save_best','independent_review','independent_recheck'}:
            measured=node['id'] in auto_seconds;seconds=auto_seconds.get(node['id'],0.)
        if native and node['id'] in {'baseline_review','decide_edit','selected_edit_tool','independent_recheck'}:
            measured=node.get('measured_seconds') is not None;seconds=node.get('measured_seconds')
        stages.append({'id':node['id'],'label':node['label'],'status':status,'status_label':STATUS.get(status,status),
            'detail':detail,'next_action':'查看当前保存事实与任务记录。',
            'execution':node.get('execution'),
            'measured_seconds':seconds if measured else None,
            'timing_basis':node.get('timing_basis') if native and measured else 'current_request_exclusive_intervals','evidence':{'prepared':node['prepared']}})
    exports=_export_receipt_state(run,observations)
    actions=[{'action_id':'inspect_native_evidence','label':'查看已保存原尺寸','supported':rendered,'reason':'查看实际母版像素，不认证相机原生细节。'},
        {'action_id':'inspect_scene_producer','label':'查看原片与成片','supported':rendered,'reason':'对照当前完整画面。'},
        {'action_id':'review_and_accept','label':'人工审片与批准','supported':saved,'reason':'自动观察不能替代人工批准。'}]
    resolution=((run.get('render') or {}).get('render_precision') or {}).get('generated_resolution') or {}
    lineage=((run.get('render') or {}).get('render_precision') or {}).get('generated_appearance_lineage')
    return {'schema_version':'cosplay_pipeline_stage_report.v1','run_id':run.get('run_id'),'revision':run.get('revision'),
        'workflow_family':'scoped_appearance','scope':'saved_revision','basis':'saved_scoped_execution_and_asset_presence',
        'creative_task':_task_summary(run),'native_route':False,'draft_state':state,'headline':label,
        'next_action':next_action,
        'request_usage':usage,
        'visual_approved':approved,'reconstruction_required':False,
        'whole_work':{'result_state':state,'result_label':label,'execution_state':execution,
            'saved_current_revision':saved,'saved_artifacts_present':rendered,'publication_state':publication.get('state'),
            'current_revision_visual_approved':approved,'output_strategy':'source_native_typed_appearance' if native else 'permitted_generated_appearance',
            'edited_output':graph.get('edited_output'),'selected_appearance_kind':graph.get('selected_appearance_kind'),
            'creative_delivery_eligibility':eligibility or None,
            'automated_target_met':bool(objective_reviewed and session.get('automated_target_met')),
            'complete_automatic_pipeline_claim':False,'open_findings':[{'criterion':row['id'],**row} for row in checks if row['status']!='pass'],
            'recovery_actions':actions,'native_detail_acceptance':False,'generated_resolution':resolution,
            'generated_appearance_lineage':lineage,
            'user_feedback':{'text':feedback,'objective_reviewed':objective_reviewed,
                'automated_target_met':bool(objective_reviewed and session.get('automated_target_met'))} if feedback else None,
            'save_reopen_export_journey_complete':bool(saved and exports['complete'])},
        'cosplay_post_diagnosis':{'status':'reviewed' if objective_reviewed else 'historical_baseline' if review_valid else 'not_requested' if session.get('status')=='not_requested' else 'pending',
            'checks':checks,'historical_baseline_review':evaluation if feedback and review_valid and not objective_reviewed else None,
            'binding_state':'saved_independent_observation' if review_valid else 'unavailable',
            'producer_run_id':run.get('run_id'),'current_repair_authorized':False,'current_render_binding_matches':review_valid,
            'native_detail_acceptance':False},'timing':timing,'stages':stages,
        'assets_rehashed':False,'quality_inferred_from_worker_success':False}
