"""Read-only dependencies and observed facts for scoped appearance production.

This module describes the existing coordinator path. It submits no work and
does not convert graph signatures, file presence or review into approval.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import copy

from cos_node_registry import Node, Port, SCHEMA, validate_connections
from photo_editing import identity, validate_scope, validate_observed_design

SOURCE_NATIVE_INITIALIZATION = {'schema':'cosluma_source_native_initialization.v1','kind':'source_native'}

SOURCE = Port('PHOTO_SOURCE', 'original_file', 'oriented_source_canvas')
DESIGN = Port('OBSERVED_DESIGN', 'structured_photographic_intent', 'none')
CANDIDATE = Port('PHOTO_APPEARANCE', 'display_linear_sRGB', 'source_canvas')
FROZEN = Port('PHOTO_APPEARANCE', 'display_linear_sRGB', 'source_canvas')
MASTER = Port('SAVED_APPEARANCE', 'display_linear_sRGB', 'source_canvas')
REVIEW = Port('INDEPENDENT_REVIEW', 'observed_findings', 'source_canvas')

NODES = {
    'photo_source':Node('原片与创作范围','原始身份、妆造、场景与许可','source_import',(0,0),'查看原片',
        {},{'source':SOURCE},'single_source_workflow.source_facts','not_applicable'),
    'observe_design':Node('观察与具体设计','观察原片，确定可见编辑及统一用光','visual_art_direction',(1,0),'查看设计',
        {'source':SOURCE},{'design':DESIGN},'photo_art_direction.prepare','not_applicable'),
    'export':Node('导出成片与工程','消费已保存的PNG与TIFF母版','save_and_export',(6,0),'导出',
        {'image':MASTER},{},'export_artifact.export_source_run','not_applicable'),
}
DEPENDENCIES = {
    'photo_source':('single_source_workflow.py','raw_support.py','asset_identity.py'),
    'observe_design':('photo_art_direction.py','photo_editing.py','vision_transport.py'),
    'export':('export_artifact.py','portable_project.py','raw_support.py'),
}

BEST = Port('REVIEWED_BEST','sRGB','current_best_canvas')
DECISION = Port('EDIT_DECISION','bounded_photographic_operation','current_best_canvas')

ITERATION_NODES = {
    'reviewed_inputs':Node('保存版本与父审片','沿用原图、当前完整图、原设计和父审片','editing_iteration_preflight',(0,0),'查看保存输入',
        {},{'source':SOURCE,'design':DESIGN,'best':BEST,'review':REVIEW},'photo_edit_iteration.authorize','not_applicable'),
    'decide_edit':Node('模型选择下一操作','观察原图、当前最佳和审片，选工具或停止','editing_decision',(1,0),'查看模型决定',
        {'source':SOURCE,'design':DESIGN,'best':BEST,'review':REVIEW},{'decision':DECISION},
        'photo_session_decision.decide_images','not_applicable'),
    'selected_edit_tool':Node('执行选定编辑工具','按决定做摄影显影、生成编辑或有资格的摄影融合','editing_selected_tool',(2,0),'查看实际工具',
        {'source':SOURCE,'design':DESIGN,'best':BEST,'decision':DECISION},{'candidate':CANDIDATE},
        'photo_edit_tools.make_tools','not_applicable'),
    'independent_recheck':Node('独立复审新候选','比较原图、候选和此前最佳；分别评价两项目标','independent_image_review',(3,0),'查看本轮复审',
        {'source':SOURCE,'design':DESIGN,'best':BEST,'candidate':CANDIDATE},{'review':REVIEW},
        'photo_session_review.evaluate_images','not_applicable'),
    'select_best':Node('保留最佳完整版本','只采用受审片支持的改进，否则保留完整基线','editing_best_selection',(4,0),'查看版本选择',
        {'best':BEST,'parent_review':REVIEW,'candidate':CANDIDATE,'review':REVIEW},{'candidate':CANDIDATE},
        'photo_edit_session.selected_output','not_applicable'),
    'freeze_appearance':Node('核验最佳外观责任','绑定实际选择的原生与生成来源、坐标与母版','photo_appearance_validation',(5,0),'查看冻结外观',
        {'source':SOURCE,'design':DESIGN,'candidate':CANDIDATE},{'appearance':FROZEN},
        'photo_appearance.validate_asset','not_applicable'),
    'save_appearance':Node('保存最佳成片与母版','完整保存与原版本分离；取消不发布基线副本','render_export',(6,0),'查看保存成片',
        {'source':SOURCE,'appearance':FROZEN},{'image':MASTER},'photo_edit_finish.save_selected_candidate','not_applicable'),
    'export':Node('导出保存版本','消费已保存像素；导出不重做模型编辑','save_and_export',(7,0),'导出',
        {'image':MASTER},{},'export_artifact.export_source_run','not_applicable'),
}
ITERATION_DEPENDENCIES = {
    'reviewed_inputs':('photo_edit_iteration.py','photo_edit_finish.py','photo_edit_session.py','photo_session_evidence.py','photo_editing.py','asset_identity.py'),
    'decide_edit':('photo_session_decision.py','photo_edit_session.py','photo_session_evidence.py','photo_session_review.py','vision_transport.py'),
    'selected_edit_tool':('photo_edit_tools.py','photo_development.py','photo_focus_integration.py','photo_edit_session.py','photo_session_evidence.py','photo_editing.py','photo_appearance.py','reference_provider.py','qwen_grid_diagnostic.py','raw_support.py','float_appearance.py'),
    'independent_recheck':('photo_session_review.py','photo_edit_session.py','photo_session_evidence.py','vision_transport.py'),
    'select_best':('photo_edit_iteration.py','photo_edit_finish.py','photo_edit_session.py','photo_session_evidence.py'),
    'freeze_appearance':('photo_appearance.py','photo_edit_iteration.py','photo_editing.py','asset_identity.py'),
    'save_appearance':('photo_edit_finish.py','photo_session_evidence.py','generated_appearance_render.py','photo_appearance.py','photo_edit_iteration.py','raw_support.py','display_transform.py'),
    'export':DEPENDENCIES['export'],
}


def _iteration_tool_capabilities(scope):
    """Read the real tool factory; discard executors and path-only configuration."""
    from photo_edit_tools import make_tools
    from runtime_resources import code_path
    placeholder=code_path('scoped_workflow.py').parent
    tools=make_tools(placeholder,placeholder,scope=scope,session_directory=placeholder)
    return {tool.name:{'operations':list(tool.operations),'costs':dict(tool.costs),
        'requires_region':tool.requires_region,'requires_working':tool.requires_working,
        'parameter_schema':copy.deepcopy(tool.parameter_schema or {}),
        'runtime_input_check_required':callable(tool.input_check),'factory_producer':tool.producer_identity}
        for tool in tools}


def _compile_iteration_workflow(mode,scope,visual_review,iteration):
    from asset_identity import sha256_file
    from runtime_resources import code_path
    from cos_workflow_cache import attach_cache_contract
    from photo_edit_iteration import iteration_request
    iteration=iteration_request(iteration)
    if not selected(mode,scope,iteration) or visual_review is not True:
        raise ValueError('Scoped editing improvement requires existing-reference inputs and independent review')
    links=[('reviewed_inputs','source','decide_edit','source'),('reviewed_inputs','design','decide_edit','design'),
        ('reviewed_inputs','best','decide_edit','best'),('reviewed_inputs','review','decide_edit','review'),
        ('reviewed_inputs','source','selected_edit_tool','source'),('reviewed_inputs','design','selected_edit_tool','design'),
        ('reviewed_inputs','best','selected_edit_tool','best'),('decide_edit','decision','selected_edit_tool','decision'),
        ('reviewed_inputs','source','independent_recheck','source'),('reviewed_inputs','design','independent_recheck','design'),
        ('reviewed_inputs','best','independent_recheck','best'),('selected_edit_tool','candidate','independent_recheck','candidate'),
        ('reviewed_inputs','best','select_best','best'),('reviewed_inputs','review','select_best','parent_review'),
        ('selected_edit_tool','candidate','select_best','candidate'),('independent_recheck','review','select_best','review'),
        ('reviewed_inputs','source','freeze_appearance','source'),('reviewed_inputs','design','freeze_appearance','design'),
        ('select_best','candidate','freeze_appearance','candidate'),('reviewed_inputs','source','save_appearance','source'),
        ('freeze_appearance','appearance','save_appearance','appearance'),('save_appearance','image','export','image')]
    edges=[{'from':a,'output':b,'to':c,'input':d} for a,b,c,d in links]
    order,incoming=validate_connections(list(ITERATION_NODES),edges,registry=ITERATION_NODES)
    capabilities=_iteration_tool_capabilities(scope)
    compiled={'schema':SCHEMA,'mode':mode,'scope':'scoped_editing_iteration_dependencies',
        'editing_scope':scope,'editing_iteration':iteration,'visual_review':True,'readonly':True,
        'executor':'existing_production_coordinator_and_photo_edit_iteration','editable_connections':False,
        'producer':{name:sha256_file(code_path(name)) for name in
            ('scoped_workflow.py','cos_node_registry.py','photo_editing.py','photo_edit_iteration.py')},
        'bounded_dynamic_loop':{'entry':'decide_edit','body':['selected_edit_tool','independent_recheck'],
            'maximum_improvements':iteration['maximum_improvements'],'stop_allowed':True,
            'selection_exit':'select_best','unknown_requests_require_reconciliation':True,
            'graph_edges_are_dependencies_not_a_mandatory_call_list':True},
        'order':order,'connections':edges,'nodes':[]}
    for key in order:
        controls={'scope_id':scope['scope_id'],'visual_review':True,'editing_iteration':iteration,
            'producer':{name:sha256_file(code_path(name)) for name in ITERATION_DEPENDENCIES[key]}}
        if key=='selected_edit_tool':
            controls['available_tools']=capabilities
            controls['tool_catalog_basis']='potential_factory_tools_checked_against_actual_session_inputs'
        if key in {'selected_edit_tool','independent_recheck'}:controls['conditional_on_tool_decision']=True
        compiled['nodes'].append({'id':key,'implementation':ITERATION_NODES[key].implementation,'inputs':incoming[key],
            'outputs':{name:asdict(port) for name,port in ITERATION_NODES[key].outputs.items()},
            'alpha_role':'not_applicable','controls':controls})
    compiled=attach_cache_contract(compiled)
    compiled['graph_id']=identity(compiled)
    return compiled


def selected(mode, scope, editing_iteration=None, *, shaping_request=None):
    if scope is None:return False
    scope=validate_scope(scope)
    if shaping_request is not None:
        from photo_shaping import normalize_request
        normalize_request(shaping_request)
        return mode=='existing_reference' and editing_iteration is None
    if editing_iteration is not None:
        from photo_edit_iteration import iteration_request
        iteration_request(editing_iteration)
        return mode=='existing_reference'
    return mode=='create_reference'


def _native_registry(review):
    from dataclasses import replace
    appearance=Port('PHOTO_APPEARANCE','display_linear_sRGB','source_canvas')
    registry={
        'photo_source':NODES['photo_source'], 'observe_design':NODES['observe_design'],
        'source_native_baseline':Node('原图原生精度基线','冻结原始几何、解码与原生像素；不是完成编辑','source_native_baseline',(2,0),'查看原图基线',
            {'source':SOURCE,'design':DESIGN},{'appearance':appearance},'photo_appearance.create_source_baseline','not_applicable'),
        'baseline_review':Node('评价原图基线','实际观察原图的自然度与创作缺口','independent_image_review',(3,0),'查看基线评价',
            {'source':SOURCE,'design':DESIGN,'appearance':appearance},{'review':REVIEW},'photo_session_review.evaluate_images','not_applicable'),
        'decide_edit':Node('模型选择编辑操作','观察当前像素与目标，选择有资格的工具或停止','editing_decision',(4,0),'查看模型决定',
            {'source':SOURCE,'design':DESIGN,'appearance':appearance,**({'review':REVIEW} if review else {})},
            {'decision':DECISION},'photo_session_decision.decide_images','not_applicable'),
        'selected_edit_tool':Node('执行模型选定编辑','摄影显影不新增生图；局部生成保留区外像素，整体生成会重写全图纹理','editing_selected_tool',(5,0),'查看实际工具',
            {'source':SOURCE,'design':DESIGN,'appearance':appearance,'decision':DECISION},{'appearance':appearance},
            'photo_edit_tools.make_tools','not_applicable'),
        'independent_recheck':Node('独立审片编辑候选','观察真实候选与此前最佳，不自动批准','independent_image_review',(6,0),'查看候选评价',
            {'source':SOURCE,'design':DESIGN,'baseline':appearance,'candidate':appearance},{'review':REVIEW},
            'photo_session_review.evaluate_images','not_applicable'),
        'select_best':Node('选择最佳完整版本','原图基线保留不冒称已完成修改','editing_best_selection',(7,0),'查看实际选择',
            {'baseline':appearance,'candidate':appearance,**({'review':REVIEW} if review else {})},{'appearance':appearance},
            'photo_edit_session.selected_output','not_applicable'),
        'freeze_appearance':Node('核验冻结外观责任','绑定原生/局部/整体像素来源与实际预览母版','photo_appearance_validation',(8,0),'查看像素来源',
            {'appearance':appearance},{'appearance':appearance},'photo_appearance.validate_asset','not_applicable'),
        'save_appearance':Node('原尺寸保存与母版','完整原子保存选中外观，不隐式重做模型','render_export',(9,0),'查看保存成片',
            {'source':SOURCE,'appearance':appearance},{'image':MASTER},'photo_edit_finish.save_selected_candidate','not_applicable'),
        'export':replace(NODES['export'],position=(10,0)),
    }
    if not review:
        for key in ('baseline_review','independent_recheck'):registry.pop(key)
    return registry


def _shaping_registry(request):
    from dataclasses import replace
    from photo_shaping import execution_method
    appearance=Port('PHOTO_APPEARANCE','display_linear_sRGB','source_canvas')
    common={
        'shaping_inputs':Node('调整前版本与独立授权','核验原片、原设计和本任务拥有的调整前完整版本 U','shaping_preflight',(0,0),'查看调整前版本',
            {},{'source':SOURCE,'design':DESIGN,'appearance':appearance},'photo_shaping.validate_authorization','not_applicable'),
        'save_appearance':Node('保存独立完整子版本','原子保存当前像素；不继承人工批准','render_export',(7,0),'查看保存成片',
            {'source':SOURCE,'appearance':appearance},{'image':MASTER},'generated_appearance_render.render_candidate','not_applicable'),
        'export':replace(NODES['export'],position=(8,0)),
    }
    if request['action']=='restore':
        common['restore_baseline']=Node('返回形体调整前版本','原样选择 U 母版、预览和覆盖；不调用模型或逆向塑形','shaping_restore',(1,0),'查看恢复版本',
            {'appearance':appearance},{'appearance':appearance},'photo_shaping.execute','not_applicable')
        return common
    native=_native_registry(True)
    for key in ('baseline_review','decide_edit','selected_edit_tool','independent_recheck','select_best','freeze_appearance'):
        common[key]=replace(native[key],position=(native[key].position[0]-2,0))
    common['baseline_review']=replace(common['baseline_review'],label='重新观察调整前版本',
        description='本次实际观察 U；历史审片不冒充当前观察')
    common['selected_edit_tool']=replace(common['selected_edit_tool'],label='执行获准的局部形体调整',
        description='仅本次明确勾选部位及幅度，输入始终为同一个 U')
    common['select_best']=replace(common['select_best'],description='形体效果须经实际复审；未完成或未知不替换当前保存版本')
    if execution_method(request)=='face_contour_deform':
        common['selected_edit_tool']=replace(common['selected_edit_tool'],label='从母版计算脸部轮廓形变',
            description='固定单调映射和同坐标RGB/覆盖采样；零次生图',implementation='photo_shaping_tool.make_tool')
        common['freeze_appearance']=replace(common['freeze_appearance'],implementation='appearance_assets.validate_asset')
    if 'face' in request['parts']:
        support=Port('SHAPING_GEOMETRY_SUPPORT','canonical_face_contour','source_canvas')
        for key,node in list(common.items()):
            if key!='shaping_inputs':common[key]=replace(node,position=(node.position[0]+1,node.position[1]))
        common['face_geometry']=Node('准备实际 U 的脸部几何支持',
            'CPU 定位实际调整前版本，冻结轮廓带和中央五官保护；几何准备不证明自然度',
            'shaping_face_geometry',(1,0),'查看几何准备',{'appearance':appearance},{'support':support},
            'photo_shaping_geometry.prepare_geometry','not_applicable')
        for key in ('baseline_review','selected_edit_tool'):
            common[key]=replace(common[key],inputs={**common[key].inputs,'geometry':support})
    return common


def _compile_shaping_workflow(mode,scope,visual_review,request):
    from photo_shaping import normalize_request,execution_method,image_submissions
    from asset_identity import sha256_file
    from runtime_resources import code_path
    from cos_workflow_cache import attach_cache_contract
    request=normalize_request(request);scope=validate_scope(scope)
    if not selected(mode,scope,shaping_request=request) or type(visual_review) is not bool:
        raise ValueError('Shaping requires current checked existing-reference inputs')
    apply=request['action']=='apply'
    method=execution_method(request)
    if apply and visual_review is not True:
        raise ValueError('Shaping requires baseline observation and candidate review')
    registry=_shaping_registry(request)
    links=[('shaping_inputs','source','save_appearance','source'),('save_appearance','image','export','image')]
    if not apply:
        links += [('shaping_inputs','appearance','restore_baseline','appearance'),
            ('restore_baseline','appearance','save_appearance','appearance')]
    else:
        for target in ('baseline_review','decide_edit','selected_edit_tool','independent_recheck'):
            links += [('shaping_inputs','source',target,'source'),('shaping_inputs','design',target,'design')]
        links += [('shaping_inputs','appearance','baseline_review','appearance'),
            ('shaping_inputs','appearance','decide_edit','appearance'),('baseline_review','review','decide_edit','review'),
            ('shaping_inputs','appearance','selected_edit_tool','appearance'),('decide_edit','decision','selected_edit_tool','decision'),
            ('shaping_inputs','appearance','independent_recheck','baseline'),('selected_edit_tool','appearance','independent_recheck','candidate'),
            ('shaping_inputs','appearance','select_best','baseline'),('selected_edit_tool','appearance','select_best','candidate'),
            ('independent_recheck','review','select_best','review'),('select_best','appearance','freeze_appearance','appearance'),
            ('freeze_appearance','appearance','save_appearance','appearance')]
        if 'face' in request['parts']:
            links += [('shaping_inputs','appearance','face_geometry','appearance'),
                ('face_geometry','support','baseline_review','geometry'),
                ('face_geometry','support','selected_edit_tool','geometry')]
    edges=[{'from':a,'output':b,'to':c,'input':d} for a,b,c,d in links]
    order,incoming=validate_connections(list(registry),edges,registry=registry)
    producer_names=['scoped_workflow.py','cos_node_registry.py','photo_shaping.py',
        'photo_appearance.py','generated_appearance_render.py']
    if apply:
        producer_names += ['photo_edit_session.py','photo_session_evidence.py','photo_session_decision.py',
            'photo_session_review.py','photo_edit_finish.py','task_model_preflight.py']
        if method=='edit_region':
            producer_names += ['photo_edit_tools.py','reference_provider.py','qwen_grid_diagnostic.py']
        if 'face' in request['parts']:
            producer_names += ['photo_shaping_geometry.py','portrait_regions_worker.py','runtime_resources.py',
                'current_model_registry.py','asset_identity.py']
        if method=='face_contour_deform':
            producer_names += ['face_contour_deform.py','geometric_appearance.py','appearance_assets.py','photo_shaping_tool.py']
    compiled={'schema':SCHEMA,'mode':mode,'scope':'checked_shaping_dependencies','editing_scope':scope,
        'shaping_request':request,'visual_review':visual_review,'readonly':True,'editable_connections':False,
        'executor':'existing_production_coordinator_and_photo_shaping',
        'request_limits':{'image_submissions':image_submissions(request),'vision_requests':3 if apply else 0},
        'producer':{name:sha256_file(code_path(name)) for name in sorted(producer_names)},
        'preparation_stages':['face_geometry'] if apply and 'face' in request['parts'] else [],
        'bounded_dynamic_loop':{'entry':'baseline_review' if apply else 'restore_baseline',
            'body':['decide_edit','selected_edit_tool','independent_recheck'] if apply else [],
            'maximum_improvements':1 if apply else 0,'stop_allowed':True,'selection_exit':'select_best' if apply else 'save_appearance',
            'unknown_requests_require_reconciliation':True,'graph_edges_are_dependencies_not_a_mandatory_call_list':True},
        'order':order,'connections':edges,'nodes':[]}
    for key in order:
        controls={'scope_id':scope['scope_id'],'shaping_request':request,
            'producer':compiled['producer'],'model_requests_allowed':apply and key in
                {'face_geometry','baseline_review','decide_edit','selected_edit_tool','independent_recheck'}}
        if key=='face_geometry':
            controls['execution_resources']={'cpu_face_localization':1,'image_submissions':0,'vision_requests':0}
            controls['geometry_is_visual_acceptance']=False
        if key=='selected_edit_tool':
            controls['model_requests_allowed']=method=='edit_region'
            controls['available_tools']={method:{'operations':['shape_'+part for part in request['parts']],
                'costs':{'image_submissions':image_submissions(request),'tool_calls':1},'requires_region':method=='edit_region'}}
        compiled['nodes'].append({'id':key,'implementation':registry[key].implementation,'inputs':incoming[key],
            'outputs':{name:asdict(port) for name,port in registry[key].outputs.items()},'alpha_role':'not_applicable','controls':controls})
    compiled=attach_cache_contract(compiled);compiled['graph_id']=identity(compiled)
    return compiled


def _shaping_geometry_fact(root,run,scope,state,invalid):
    """Observe a frozen preparation proof without plane derivation/runtime IO."""
    from asset_identity import sha256_file
    from runtime_resources import code_path
    from photo_shaping_geometry import PARAMETERS, SCHEMA as GEOMETRY_SCHEMA
    from photo_session_evidence import _index, _owned
    if invalid:raise ValueError('Current session identity is unavailable')
    record=run.get('shaping_geometry')
    if not isinstance(record,dict) or set(record)!={'path','sha256'}:
        raise ValueError('No actual owned face geometry receipt')
    owned,index=_index(state) if state and state.get('evidence') else (None,None)
    own_render=(run.get('render') or {}).get('run_id')==run.get('run_id')
    if (own_render or (run.get('result_publication') or {}).get('state')=='committed') and index is None:
        raise ValueError('Saved geometry is missing its owned current session evidence')
    run_directory=(root/'outputs/runs'/str(run.get('run_id') or '')).resolve()
    def locate(item,*,small=False):
        if not isinstance(item,dict) or set(item)!={'path','sha256'}:
            raise ValueError('Incomplete saved geometry evidence locator')
        if index is not None:
            entry=index.get('files',{}).get(item['sha256'])
            if not isinstance(entry,dict) or entry.get('sha256')!=item['sha256']:
                raise ValueError('Geometry file is absent from owned session evidence')
            path=_owned(owned,entry['file'])
        else:
            path=Path(item['path']);path=(path if path.is_absolute() else root/path).resolve()
            if not path.is_relative_to(run_directory):raise ValueError('Geometry preparation is not owned by this run')
        if not path.is_file():raise ValueError('Saved geometry evidence is missing')
        if small:
            if path.stat().st_size>512*1024:raise ValueError('Geometry proof exceeds the metadata read limit')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Saved geometry proof changed')
            return json.loads(raw)
        if sha256_file(path)!=item['sha256']:raise ValueError('Saved geometry mask bytes changed')
        return path
    support=locate(record,small=True);binding=support.get('binding') or {}
    if (support.get('schema')!=GEOMETRY_SCHEMA
            or support.get('support_id')!=identity({key:value for key,value in support.items() if key!='support_id'})
            or binding.get('parameters')!=PARAMETERS):
        raise ValueError('Saved face geometry proof identity changed')
    producer={name:sha256_file(code_path(name)) for name in
        ('photo_shaping_geometry.py','portrait_regions_worker.py','runtime_resources.py','asset_identity.py')}
    if binding.get('producer')!=producer:raise ValueError('Saved face geometry producer is stale')
    base=((run.get('options') or {}).get('shaping_state') or {}).get('base_appearance')
    if base is None:base=(run.get('options') or {}).get('photo_appearance')
    receipt=_native_asset_fact(root,base,run,scope)
    if not receipt:raise ValueError('Geometry has no matching typed adjustment-before version U')
    expected={'asset_id':base['asset_id'],'receipt_sha256':base['receipt_sha256'],
        'source_sha256':run['source']['sha256'],'scope_id':scope['scope_id'],
        'design_id':run['generation_design']['observed_design']['design_id'],
        'master_sha256':receipt['files']['master']['sha256'],'preview_sha256':receipt['files']['preview']['sha256'],
        'native_size':receipt['canvas_size']}
    auth=(run.get('task_plan') or {}).get('shaping_authorization') or {}
    if (binding.get('base')!=expected or any((auth.get('base_appearance') or {}).get(key)!=base[key]
            for key in ('asset_id','receipt_sha256'))):
        raise ValueError('Face geometry belongs to different checked U')
    detections=locate(support.get('detections'),small=True);locate(support.get('masks'))
    runtime=binding.get('runtime') or {}
    if (detections.get('source_sha256')!=expected['preview_sha256']
            or detections.get('source_native_size')!=expected['native_size']
            or detections.get('model_sha256')!=runtime.get('model_sha256')
            or detections.get('mediapipe_version')!=runtime.get('mediapipe_version')
            or detections.get('method')!='mediapipe_offline_multi_face_landmarks'
            or not isinstance(detections.get('faces'),list) or len(detections['faces'])!=1):
        raise ValueError('Saved localization facts differ from actual U/runtime evidence')
    return {'support':copy.deepcopy(record),'support_id':support['support_id'],
        'base_asset_id':base['asset_id'],'owned_session_evidence':index is not None,
        'verification_scope':'saved_geometry_metadata_and_owned_file_sha256',
        'planes_rederived':False,'runtime_reinspected':False,'naturalness_verified':False}


def _build_shaping_graph(run,scope,visual_review,request):
    compiled=_compile_shaping_workflow('existing_reference',scope,visual_review,request)
    registry=_shaping_registry(compiled['shaping_request']);root=Path(run.get('root') or '.').resolve()
    plan=run.get('task_plan') or {};saved=plan.get('layer_workflow') or {}
    options=run.get('options') or {};current=options.get('photo_appearance');shape=options.get('shaping_state')
    facts={key:['pending',False] for key in compiled['order']};details={};valid=False
    if shape is not None:
        try:
            # The canvas observes small bound metadata. Full plane validation
            # remains at execution/render/export, rather than every UI refresh.
            from photo_shaping import _contract, STATE_SCHEMA
            auth=_contract(shape['authorization']);base=shape['base_appearance']
            original=_native_asset_fact(root,base,run,scope);selected=_native_asset_fact(root,current,run,scope)
            valid=(shape['schema']==STATE_SCHEMA and type(shape['enabled']) is bool
                and shape['enabled']==(compiled['shaping_request']['action']=='apply')
                and shape['selected_asset_id']==current['asset_id']
                and all(base[key]==auth['base_appearance'][key] for key in ('asset_id','receipt_sha256'))
                and auth['source_sha256']==run['source']['sha256'] and auth['scope_id']==scope['scope_id']
                and auth['design_id']==run['generation_design']['observed_design']['design_id']
                and auth==plan.get('shaping_authorization') and shape['request']==auth['request']==compiled['shaping_request']
                and bool(original and selected) and original['source_origin']==selected['source_origin']
                and original['canvas_size']==selected['canvas_size']
                and original['files']['preview']['sha256']==auth['base_image_sha256']
                and original['files']['master']['sha256']==auth['base_master_sha256']
                and ((current['asset_id']!=base['asset_id']) if shape['enabled'] else current['asset_id']==base['asset_id']))
        except (OSError,ValueError,KeyError,TypeError):pass
        facts['shaping_inputs']=['complete' if valid else 'needs_attention',valid]
    state,invalid=_iteration_session(root,run,scope) if compiled['shaping_request']['action']=='apply' else (None,False)
    geometry_fact=None
    if 'face_geometry' in facts:
        try:
            geometry_fact=_shaping_geometry_fact(root,run,scope,state,invalid)
            facts['face_geometry']=['complete',True]
            details['face_geometry']='已读取绑定实际 U 的冻结几何凭据；仅检查保存证据和资产拥有，未重新定位、读权重或推导大 mask，不证明自然度或审片通过。'
        except (OSError,ValueError,KeyError,TypeError) as error:
            facts['face_geometry']=['needs_attention',False]
            details['face_geometry']='实际几何准备凭据缺失、身份不符或过期：'+str(error)
    tools=None
    if state:
        attempts,_=_native_tool_attempts(root,run,scope,[state]);tools=_tool_execution_facts({'attempts':attempts})
        for key,kind in (('baseline_review','evaluate'),('decide_edit','decide'),('selected_edit_tool','tool'),('independent_recheck','evaluate')):
            rows=[row for row in attempts if row.get('kind')==kind and (kind!='evaluate' or
                ((row.get('operation_stage') or (row.get('context') or {}).get('operation_stage'))=='baseline')==(key=='baseline_review'))]
            complete=bool(rows) and all(row.get('status')=='completed' for row in rows)
            if any(row.get('status') in {'unknown','failed','interrupted','needs_attention'} for row in rows):status='needs_attention'
            elif any(row.get('status')=='reserved' for row in rows):status='running'
            elif complete:status='complete'
            elif state.get('status')!='active':status='not_requested'
            else:status='pending'
            facts[key]=[status,complete]
    if invalid:
        for key in ('baseline_review','decide_edit','selected_edit_tool','independent_recheck'):
            facts[key]=['needs_attention',False]
    final=_native_asset_fact(root,current,run,scope)
    render=run.get('render') or {};outputs=run.get('artifacts') or {}
    own_render=(render.get('run_id')==run.get('run_id') and (render.get('photo_appearance') or {}).get('asset_id')==(current or {}).get('asset_id'))
    if valid and final:
        key='select_best' if compiled['shaping_request']['action']=='apply' else 'restore_baseline'
        if own_render:facts[key]=['complete',True]
        if 'freeze_appearance' in facts:facts['freeze_appearance']=['complete',True]
    if own_render:
        good=valid and bool(final) and all(_exists(root,outputs.get(key)) for key in ('auto_draft','render_master','render_scene_master'))
        facts['save_appearance']=['complete' if good else 'needs_attention',bool(good)]
    if run.get('run_id'):
        from artifact_receipts import observe_receipts
        if observe_receipts(root,run)['exports']:facts['export']=['complete',True]
    nodes=[]
    for key in compiled['order']:
        node=registry[key];status,prepared=facts[key]
        nodes.append({'id':key,'label':node.label,'description':node.description,'stage':node.stage,
            'position':[node.position[0]*270,node.position[1]*220],'parameters':{'enabled':True,'strength':1.},
            'editable':False,'prepared':prepared,'status':status,'detail':node.description+'。'+details.get(key,'')+'状态依据当前资产和本次会话，不证明人工接受。',
            'action':node.action,'contract':asdict(node),'cache_key':compiled['cache']['node_keys'][key],
            'execution':tools if key=='selected_edit_tool' else geometry_fact if key=='face_geometry' else None,'asset':None})
    return {'schema':'cosplay_workflow_graph.v1','run_id':run.get('run_id'),'readonly':True,'nodes':nodes,
        'connections':[{**edge,'type':registry[edge['from']].outputs[edge['output']].type} for edge in compiled['connections']],
        'layer_workflow':compiled,'canonical_processing_order':True,'editing_scope':scope,'shaping_request':compiled['shaping_request'],
        'bounded_dynamic_loop':compiled['bounded_dynamic_loop'],'execution_contract_graph_id':saved.get('graph_id'),
        'historical_execution_graph_id':saved.get('graph_id') if saved.get('graph_id')!=compiled['graph_id'] else None,
        'execution_contract_mutated':False,'display_basis':'owned_shaping_baseline_and_current_session_facts',
        'cache_hits_inferred':False,'artistic_acceptance_inferred':False}


def compile_scoped_workflow(mode, scope, visual_review=False, *, editing_iteration=None, editing_autofinish=None,
                            editing_initialization=None, initial_variant_count=1, shaping_request=None):
    if shaping_request is not None:
        if editing_iteration is not None or editing_autofinish is not None or editing_initialization is not None:
            raise ValueError('Shaping must use its independent current checked workflow')
        return _compile_shaping_workflow(mode,scope,visual_review,shaping_request)
    if editing_iteration is not None:
        return _compile_iteration_workflow(mode,scope,visual_review,editing_iteration)
    from asset_identity import sha256_file
    from runtime_resources import code_path
    from cos_workflow_cache import attach_cache_contract
    scope=validate_scope(scope)
    if (not selected(mode,scope) or type(visual_review) is not bool
            or editing_initialization!=SOURCE_NATIVE_INITIALIZATION
            or type(initial_variant_count) is not int or not 1<=initial_variant_count<=8):
        raise ValueError('Current source-native creation initialization and candidate count required')
    policy={**(editing_autofinish or {'maximum_improvements':0})}
    policy.setdefault('maximum_zero_image_finishes',0)
    if (set(policy)!={'maximum_improvements','maximum_zero_image_finishes'} or type(policy['maximum_improvements']) is not int
            or not 0<=policy['maximum_improvements']<=8 or (policy['maximum_improvements'] and not visual_review)
            or type(policy['maximum_zero_image_finishes']) is not int or not 0<=policy['maximum_zero_image_finishes']<=1
            or (policy['maximum_zero_image_finishes'] and (not policy['maximum_improvements'] or not visual_review))):
        raise ValueError('Current source-native improvement policy required')
    registry=_native_registry(visual_review)
    links=[('photo_source','source','observe_design','source'),
        ('photo_source','source','source_native_baseline','source'),('observe_design','design','source_native_baseline','design'),
        ('photo_source','source','decide_edit','source'),('observe_design','design','decide_edit','design'),
        ('source_native_baseline','appearance','decide_edit','appearance'),
        ('photo_source','source','selected_edit_tool','source'),('observe_design','design','selected_edit_tool','design'),
        ('source_native_baseline','appearance','selected_edit_tool','appearance'),('decide_edit','decision','selected_edit_tool','decision'),
        ('source_native_baseline','appearance','select_best','baseline'),('selected_edit_tool','appearance','select_best','candidate'),
        ('select_best','appearance','freeze_appearance','appearance'),('photo_source','source','save_appearance','source'),
        ('freeze_appearance','appearance','save_appearance','appearance'),('save_appearance','image','export','image')]
    if visual_review:
        links += [('photo_source','source','baseline_review','source'),('observe_design','design','baseline_review','design'),
            ('source_native_baseline','appearance','baseline_review','appearance'),('baseline_review','review','decide_edit','review'),
            ('photo_source','source','independent_recheck','source'),('observe_design','design','independent_recheck','design'),
            ('source_native_baseline','appearance','independent_recheck','baseline'),('selected_edit_tool','appearance','independent_recheck','candidate'),
            ('independent_recheck','review','select_best','review')]
    edges=[{'from':a,'output':b,'to':c,'input':d} for a,b,c,d in links]
    order,incoming=validate_connections(list(registry),edges,registry=registry)
    dependencies={
        'photo_source':DEPENDENCIES['photo_source'],'observe_design':DEPENDENCIES['observe_design'],
        'source_native_baseline':('photo_appearance.py','raw_support.py','display_transform.py','asset_identity.py'),
        'baseline_review':ITERATION_DEPENDENCIES['independent_recheck'],
        'decide_edit':ITERATION_DEPENDENCIES['decide_edit'],
        'selected_edit_tool':(*ITERATION_DEPENDENCIES['selected_edit_tool'],'photo_appearance.py'),
        'independent_recheck':ITERATION_DEPENDENCIES['independent_recheck'],
        'select_best':('photo_edit_session.py','photo_session_evidence.py','photo_edit_finish.py','photo_appearance.py'),
        'freeze_appearance':('photo_appearance.py','photo_editing.py','asset_identity.py'),
        'save_appearance':('photo_edit_finish.py','photo_session_evidence.py','generated_appearance_render.py','photo_appearance.py','raw_support.py'),
        'export':DEPENDENCIES['export']}
    count=initial_variant_count;k=policy['maximum_improvements'];f=policy['maximum_zero_image_finishes']
    compiled={'schema':SCHEMA,'mode':mode,'scope':'source_native_editing_dependencies','editing_scope':scope,
        'editing_initialization':copy.deepcopy(editing_initialization),'editing_autofinish':copy.deepcopy(policy),
        'visual_review':visual_review,'readonly':True,'editable_connections':False,
        'executor':'existing_production_coordinator_and_photo_edit_finish',
        'initial_candidates':{'count':count,'serial':True,'source_baseline_review_shared':visual_review},
        'request_limits':{'image_submissions':count+k,'vision_requests':2+2*count+2*(k+f) if visual_review else 1+count},
        'bounded_dynamic_loop':{'entry':'decide_edit','body':['selected_edit_tool',*(['independent_recheck'] if visual_review else [])],
            'initial_operations_per_candidate':1,'maximum_improvements':k,'maximum_zero_image_finishes':f,
            'improvements_shared_across_variants':True,
            'operation_stages':['initial','improvement'],'stop_allowed':True,'selection_exit':'select_best',
            'unknown_requests_require_reconciliation':True,'graph_edges_are_dependencies_not_a_mandatory_call_list':True},
        'producer':{name:sha256_file(code_path(name)) for name in ('scoped_workflow.py','cos_node_registry.py','photo_editing.py','photo_appearance.py')},
        'order':order,'connections':edges,'nodes':[]}
    for key in order:
        controls={'scope_id':scope['scope_id'],'editing_initialization':editing_initialization,'visual_review':visual_review,
            'producer':{name:sha256_file(code_path(name)) for name in dependencies[key]}}
        if key=='selected_edit_tool':
            controls.update(available_tools=_iteration_tool_capabilities(scope),
                tool_catalog_basis='potential_factory_tools_checked_against_actual_session_inputs',conditional_on_tool_decision=True)
        compiled['nodes'].append({'id':key,'implementation':registry[key].implementation,'inputs':incoming[key],
            'outputs':{name:asdict(port) for name,port in registry[key].outputs.items()},'alpha_role':'not_applicable','controls':controls})
    compiled=attach_cache_contract(compiled);compiled['graph_id']=identity(compiled)
    return compiled


def _exists(root,value):
    if not isinstance(value,str) or not value:return False
    path=Path(value)
    return (path if path.is_absolute() else root/path).is_file()


def _observation(run,scope):
    generation=run.get('generation_design') or (run.get('options') or {}).get('generation_design') or {}
    observed=generation.get('observed_design') or {}
    try:
        content={key:observed.get(key) for key in ('summary','observation','changes','lighting','keep','uncertain')}
        if validate_observed_design(content,scope)!=observed:return False
    except (ValueError,TypeError,KeyError):return False
    direction=generation.get('art_direction') or {}
    provenance=direction.get('provenance') or {}
    return (generation.get('status')=='ready' and generation.get('source_sha256')==(run.get('source') or {}).get('sha256')
        and direction.get('status')=='ready' and direction.get('design')==observed
        and provenance.get('network_used') is True and provenance.get('generation_used') is False
        and bool(provenance.get('model')) and (provenance.get('source') or {}).get('sha256')==generation['source_sha256'])


def _iteration_session(root,run,scope):
    plan=run.get('task_plan') or {}
    value=((run.get('photo_edit_session') or {}).get('path')
        or (run.get('editing_iteration_result') or {}).get('session'))
    path=Path(value) if value else root/'outputs/runs'/str(run.get('run_id') or '')/'editing-session/session.json'
    path=path if path.is_absolute() else root/path
    if not path.is_file():return None, bool(value)
    try:
        if path.stat().st_size>2*1024*1024:return None,True
        state=json.loads(path.read_bytes())
        binding=state.get('binding') or {}
        if (state.get('session_id')!=identity(binding) or state.get('scope')!=scope
                or binding.get('scope_id')!=scope['scope_id']
                or binding.get('execution_identity')!=plan.get('plan_id')
                or (binding.get('source') or {}).get('sha256')!=(run.get('source') or {}).get('sha256')):
            return None,True
        return state,False
    except (OSError,ValueError,TypeError,KeyError):return None,True


def _tool_execution_facts(state):
    """Observed tool calls are independent of diffusion request counts."""
    calls=[]
    for row in state.get('attempts',[]):
        if row.get('kind')!='tool':continue
        costs=row.get('costs') or {};response=row.get('response') or {}
        actual=(response.get('resources') or {}).get('image_submissions')
        actual=actual if row.get('status')=='completed' and type(actual) is int and actual>=0 else None
        seconds=row.get('wall_seconds')
        seconds=seconds if type(seconds) in {int,float} and math.isfinite(seconds) and seconds>=0 else None
        def count(key):
            value=costs.get(key)
            return value if type(value) is int and value>=0 else None
        calls.append({'tool':(row.get('context') or {}).get('decision',{}).get('tool'),
            'status':row.get('status'),'tool_calls_reserved':count('tool_calls'),
            'image_submissions_reserved':count('image_submissions'),
            'actual_image_submissions':actual,'wall_seconds':seconds})
    return {'tool_invocations':calls,'completed_tool_calls':sum(row['status']=='completed' for row in calls),
        'tool_calls_reserved':sum(row['tool_calls_reserved'] for row in calls)
            if all(row['tool_calls_reserved'] is not None for row in calls) else None,
        'image_submissions_reserved':sum(row['image_submissions_reserved'] for row in calls)
            if all(row['image_submissions_reserved'] is not None for row in calls) else None,
        'actual_image_submissions':sum(row['actual_image_submissions'] for row in calls)
            if calls and all(row['actual_image_submissions'] is not None for row in calls) else None,
        'zero_image_tool_calls':sum(row['status']=='completed' and row['actual_image_submissions']==0 for row in calls),
        'tool_wall_seconds':sum(row['wall_seconds'] for row in calls if row['wall_seconds'] is not None),
        'basis':'persisted_tool_attempts_and_completed_tool_resources_not_image_budget'}


def _build_iteration_graph(run,scope,visual_review,iteration):
    if not (run.get('options') or {}).get('photo_appearance'):
        raise ValueError('Unsupported retired creation; a current typed appearance is required')
    compiled=compile_scoped_workflow('existing_reference',scope,visual_review,editing_iteration=iteration)
    root=Path(run.get('root') or '.').resolve()
    plan=run.get('task_plan') or {}; saved=plan.get('layer_workflow') or {}
    authorization=plan.get('editing_refinement_authorization') or {}
    source=run.get('source') or {}; reference=run.get('reference') or {}
    facts={key:['pending',False] for key in compiled['order']}
    details={}
    if authorization:
        inputs=plan.get('inputs') or {}
        good=(authorization.get('scope_id')==scope['scope_id']
            and authorization.get('source_sha256')==source.get('sha256')
            and authorization.get('base_image_sha256')==(inputs.get('reference') or {}).get('sha256')
            and bool(authorization.get('review_sha256'))
            and _exists(root,source.get('path')) and _exists(root,(inputs.get('reference') or {}).get('path')))
        facts['reviewed_inputs']=['complete' if good else 'needs_attention',bool(good)]
    state,invalid=_iteration_session(root,run,scope)
    if invalid:
        for key in ('decide_edit','selected_edit_tool','independent_recheck','select_best'):
            facts[key]=['needs_attention',False]
            details[key]='本任务的小型持久会话缺失或身份不符；旧会话不能证明本轮调用。'
    tool_execution=None
    if state:
        attempts,_verified=_native_tool_attempts(root,run,scope,[state])
        tool_execution=_tool_execution_facts({'attempts':attempts})
        for key,kind in (('decide_edit','decide'),('selected_edit_tool','tool'),('independent_recheck','evaluate')):
            rows=[row for row in attempts if row.get('kind')==kind]
            completed=any(row.get('status')=='completed' for row in rows)
            unresolved=[row for row in rows if row.get('status') in {'reserved','unknown','interrupted','failed','needs_attention'}]
            if unresolved:
                status='running' if state.get('status')=='active' and all(row.get('status')=='reserved' for row in unresolved) else 'needs_attention'
            elif completed:status='complete'
            elif state.get('status')!='active':status='not_requested'
            else:status='pending'
            facts[key]=[status,completed]
        tool_names=sorted({str((row.get('context') or {}).get('decision',{}).get('tool'))
            for row in attempts if row.get('kind')=='tool' and (row.get('context') or {}).get('decision',{}).get('tool')})
        if tool_names:details['selected_edit_tool']='本轮已记录的工具：'+', '.join(tool_names)+'。'
        if tool_execution and tool_execution['tool_invocations']:
            images=tool_execution['actual_image_submissions']
            details['selected_edit_tool']=details.get('selected_edit_tool','')+f" 已完成 {tool_execution['completed_tool_calls']} 次工具；"+(
                f'实际新增图像请求 {images} 次。' if images is not None else '实际图像请求数尚待核实。')
            if tool_execution['zero_image_tool_calls']:
                details['selected_edit_tool']+=f" 其中 {tool_execution['zero_image_tool_calls']} 次工具没有新增生图。"
        decisions=[row for row in attempts if row.get('kind')=='decide' and row.get('status')=='completed']
        if decisions and (decisions[-1].get('response') or {}).get('action')=='stop':
            details['decide_edit']='模型已明确选择停止；未因此启动未选择的工具。'
        best=next((row for row in state.get('candidates',[]) if row.get('candidate_id')==state.get('best_candidate_id')),None)
        if state.get('status')=='cancelled':
            facts['select_best']=['cancelled',False]
        elif state.get('status')!='active' and best:
            try:
                from photo_session_evidence import resolve_file
                good=bool(best.get('review') and resolve_file(state,best['image']))
            except (OSError,ValueError,KeyError,TypeError):good=False
            facts['select_best']=['complete' if good else 'needs_attention',good]
            if (run.get('editing_iteration_result') or {}).get('improved') is False:
                details['select_best']='没有受复审支持的更佳候选；保留此前完整画面，不代表本轮取得画质改进。'
    result=run.get('editing_iteration_result') or {}
    frozen=(run.get('options') or {}).get('photo_appearance')
    render=run.get('render') or {}; outputs=run.get('artifacts') or {}
    own_render=render.get('kind')=='scoped_photo_appearance_render' and render.get('run_id')==run.get('run_id')
    if frozen and (result or own_render):
        try:
            good=bool(_native_asset_fact(root,frozen,run,scope))
            facts['freeze_appearance']=['complete' if good else 'needs_attention',bool(good)]
        except (OSError,ValueError,TypeError,KeyError):facts['freeze_appearance']=['needs_attention',False]
    if own_render:
        good=(facts['freeze_appearance'][1]
            and (render.get('photo_appearance') or {}).get('asset_id')==(frozen or {}).get('asset_id')
            and render.get('editing_scope_id')==scope['scope_id']
            and render.get('render_precision',{}).get('working_domain')=='display_linear_sRGB'
            and all(_exists(root,outputs.get(key)) for key in ('auto_draft','render_master','render_scene_master')))
        facts['save_appearance']=['complete' if good else 'needs_attention',bool(good)]
    if state and state.get('status')=='cancelled':
        for key in ('freeze_appearance','save_appearance'):facts[key]=['not_requested',False]
    if run.get('run_id'):
        from artifact_receipts import observe_receipts
        if observe_receipts(root,run)['exports']:facts['export']=['complete',True]
    nodes=[]
    for key in compiled['order']:
        node=ITERATION_NODES[key];status,prepared=facts[key]
        detail=node.description+'。'+details.get(key,'')
        if key in {'selected_edit_tool','independent_recheck'}:
            detail+='这是有界动态步骤，模型可以停止；潜在工具须通过运行时输入资格检查，不是固定全部执行。'
        detail+='状态只读取本轮会话及保存资产，不隐式调用模型，不证明人工批准或原生细节。'
        nodes.append({'id':key,'label':node.label,'description':node.description,'stage':node.stage,
            'position':[node.position[0]*270,node.position[1]*220], 'parameters':{'enabled':True,'strength':1.},
            'editable':False,'prepared':prepared,'status':status,'detail':detail,'action':node.action,
            'execution':tool_execution if key=='selected_edit_tool' else None,
            'contract':asdict(node),'cache_key':compiled['cache']['node_keys'][key],'asset':None})
    return {'schema':'cosplay_workflow_graph.v1','nodes':nodes,
        'connections':[{**edge,'type':ITERATION_NODES[edge['from']].outputs[edge['output']].type} for edge in compiled['connections']],
        'layer_workflow':compiled,'canonical_processing_order':True,'run_id':run.get('run_id'),'readonly':True,
        'editing_scope':scope,'editing_iteration':compiled['editing_iteration'],
        'bounded_dynamic_loop':compiled['bounded_dynamic_loop'],
        'execution_contract_graph_id':saved.get('graph_id'),
        'historical_execution_graph_id':saved.get('graph_id') if saved.get('graph_id')!=compiled['graph_id'] else None,
        'execution_contract_mutated':False,'display_basis':'scope_and_saved_iteration_session_facts',
        'cache_hits_inferred':False,'artistic_acceptance_inferred':False}


def _native_asset_fact(root, descriptor, run, scope):
    """Read small bound metadata only; production owns full pixel validation."""
    if not isinstance(descriptor,dict) or set(descriptor)!={'path','asset_id','receipt_sha256'}:return None
    try:
        directory=Path(descriptor['path']);directory=directory if directory.is_absolute() else root/directory
        path=directory/'receipt.json'
        if not path.is_file() or path.stat().st_size>256*1024:return None
        raw=path.read_bytes();receipt=json.loads(raw)
        design=(run.get('generation_design') or {}).get('observed_design') or {}
        if (hashlib.sha256(raw).hexdigest()!=descriptor['receipt_sha256']
                or receipt.get('schema') not in {'cosluma_photo_appearance.v1','cosluma_geometric_appearance.v1'}
                or receipt.get('asset_id')!=descriptor['asset_id']
                or identity({key:value for key,value in receipt.items() if key!='asset_id'})!=receipt['asset_id']
                or receipt.get('source_sha256')!=(run.get('source') or {}).get('sha256')
                or receipt.get('scope_id')!=scope['scope_id'] or receipt.get('design_id')!=design.get('design_id')):
            return None
        if receipt.get('schema')=='cosluma_geometric_appearance.v1':
            from appearance_assets import declared_files_from_receipt
            declared_files_from_receipt(receipt,descriptor['receipt_sha256'])
        for key in ('master','preview'):
            record=receipt.get('files',{}).get(key) or {};relative=record.get('file')
            if (not isinstance(relative,str) or Path(relative).name!=relative or not record.get('sha256')
                    or not (directory/relative).is_file()):return None
        return receipt
    except (OSError,ValueError,KeyError,TypeError):return None


def _native_sessions(root,run,scope):
    states=[];invalid=False;seen=set()
    records=[run.get('photo_edit_session') or {},*[(row.get('photo_edit_session') or {}) for row in run.get('variants',[])]]
    for record in records:
        value=record.get('path')
        if not value or value in seen:continue
        seen.add(value)
        state,bad=_iteration_session(root,{**run,'photo_edit_session':record},scope)
        if state is not None:
            binding=state.get('binding') or {}
            if (binding.get('editing_initialization')!=SOURCE_NATIVE_INITIALIZATION
                    or not isinstance(binding.get('variant_id'),str)):
                invalid=True;continue
            states.append(state)
        else:invalid|=bad
    return states,invalid


def _native_tool_attempts(root,run,scope,states):
    verified_tool_assets=set()
    observed_attempts=[]
    for state in states:
        for row in state.get('attempts',[]):
            item=row
            if row.get('kind')=='tool' and row.get('status')=='completed':
                candidate=next((value for value in state.get('candidates',[]) if value.get('attempt_id')==row.get('attempt_id')),None)
                descriptor=(candidate or {}).get('appearance_asset')
                try:
                    from photo_session_evidence import resolve_asset
                    actual=resolve_asset(state,descriptor)
                except (OSError,ValueError,KeyError,TypeError):actual=None
                receipt=_native_asset_fact(root,actual,run,scope)
                good=bool(receipt and (row.get('response') or {}).get('appearance_asset')==descriptor
                    and (candidate.get('image') or {}).get('sha256')==receipt['files']['preview']['sha256'])
                if good:verified_tool_assets.add(receipt['asset_id'])
                else:item={**row,'status':'needs_attention'}
            observed_attempts.append(item)
    return observed_attempts,verified_tool_assets


def build_scoped_graph(run,scope,visual_review, *, editing_iteration=None,editing_autofinish=None,
                       editing_initialization=None,shaping_request=None):
    shaping_request=(run.get('task_plan') or {}).get('shaping_request') if shaping_request is None else shaping_request
    if shaping_request is not None:
        return _build_shaping_graph(run,scope,visual_review,shaping_request)
    if editing_iteration is not None:
        return _build_iteration_graph(run,scope,visual_review,editing_iteration)
    plan=run.get('task_plan') or {}
    initialization=plan.get('editing_initialization') if editing_initialization is None else editing_initialization
    if initialization!=SOURCE_NATIVE_INITIALIZATION:raise ValueError('Unsupported retired creation; start a current source-native task')
    count=(plan.get('image_budget') or {}).get('initial_reference') or (plan.get('candidate_selection') or {}).get('default_variant_count') or 1
    compiled=compile_scoped_workflow('create_reference',scope,visual_review,editing_autofinish=editing_autofinish,
        editing_initialization=initialization,initial_variant_count=count)
    root=Path(run.get('root') or '.').resolve();registry=_native_registry(visual_review)
    states,invalid=_native_sessions(root,run,scope)
    attempts=[row for state in states for row in state.get('attempts',[])]
    attempts,verified_tool_assets=_native_tool_attempts(root,run,scope,states)
    facts={key:['pending',False] for key in compiled['order']};details={};elapsed={}
    source=run.get('source') or {}
    if source.get('sha256'):facts['photo_source']=['complete' if _exists(root,source.get('path')) else 'needs_attention',True]
    if _observation(run,scope):facts['observe_design']=['complete',True]
    baseline=_native_asset_fact(root,run.get('initial_source_appearance'),run,scope)
    if baseline and baseline.get('kind')=='source_baseline':facts['source_native_baseline']=['complete',True]
    elif run.get('initial_source_appearance'):facts['source_native_baseline']=['needs_attention',False]
    finished=bool(states and all(state.get('status') not in {'active'} or state.get('phase')=='waiting_selection' for state in states))
    def stage(row):return row.get('operation_stage') or (row.get('context') or {}).get('operation_stage')
    mapping={'decide_edit':lambda row:row.get('kind')=='decide',
        'selected_edit_tool':lambda row:row.get('kind')=='tool',
        'baseline_review':lambda row:row.get('kind')=='evaluate' and stage(row)=='baseline',
        'independent_recheck':lambda row:row.get('kind')=='evaluate' and stage(row) in {'initial','improvement'}}
    for key,predicate in mapping.items():
        if key not in facts:continue
        rows=[row for row in attempts if predicate(row)]
        if any(row.get('status') in {'failed','unknown','needs_attention','interrupted'} for row in rows):
            facts[key]=['needs_attention',False]
        elif any(row.get('status') in {'reserved','running'} for row in rows):facts[key]=['running',False]
        elif rows and all(row.get('status')=='completed' for row in rows):facts[key]=['complete',True]
        elif finished:facts[key]=['not_requested',False]
        if rows:
            details[key]=f"实际 {len(rows)} 项记录；首次操作 {sum(stage(row)=='initial' for row in rows)}，后续改进 {sum(stage(row)=='improvement' for row in rows)}。"
            if key=='baseline_review':details[key]+='包含同源同目标评价的缓存复用记录；记录项数不是实际HTTP次数。'
            values=[row.get('wall_seconds') for row in rows]
            if all(type(value) in {int,float} and math.isfinite(value) and value>=0 for value in values):elapsed[key]=sum(values)
        if invalid:facts[key]=['needs_attention',False]
    tools=_tool_execution_facts({'attempts':attempts})
    if tools['tool_invocations']:
        images=tools['actual_image_submissions']
        details['selected_edit_tool']=details.get('selected_edit_tool','')+(
            f'实际图像请求 {images} 次；零生图工具 {tools["zero_image_tool_calls"]} 次。' if images is not None else '实际图像请求尚待核实。')
    summary=run.get('photo_edit_session') or {}
    final_descriptor=(run.get('options') or {}).get('photo_appearance')
    final=_native_asset_fact(root,final_descriptor,run,scope)
    edited=bool(final and final.get('kind')!='source_baseline' and final['asset_id'] in verified_tool_assets
        and not summary.get('selected_is_source_baseline',False))
    if states and finished:
        facts['select_best']=['complete' if edited else 'partial',edited]
        if not edited:details['select_best']='保留原图基线；没有获准的已完成修改，不冒称新生成成片。'
        elif (summary.get('creative_delivery_eligibility') or {}).get('eligible') is False:
            details['select_best']='摄影调整已保存为完整编辑稿；制作范围的场景或效果目标尚未完成，不因审片两项通过而宣称达成。'
    if final_descriptor:facts['freeze_appearance']=['complete' if final else 'needs_attention',bool(final)]
    render=run.get('render') or {};outputs=run.get('artifacts') or {}
    own_render=(render.get('run_id')==run.get('run_id')
        and (render.get('photo_appearance') or {}).get('asset_id')==(final_descriptor or {}).get('asset_id'))
    if own_render:
        good=bool(final and all(_exists(root,outputs.get(key)) for key in ('auto_draft','render_master','render_scene_master')))
        facts['save_appearance']=['complete' if good else 'needs_attention',good]
        if not edited:details['save_appearance']='保存的是原图基线，不代表已完成创作编辑。'
    if run.get('run_id'):
        from artifact_receipts import observe_receipts
        if observe_receipts(root,run)['exports']:facts['export']=['complete',True]
    saved=plan.get('layer_workflow') or {};nodes=[]
    for key in compiled['order']:
        node=registry[key];status,prepared=facts[key]
        detail=node.description+'。'+details.get(key,'')+'状态读取本次真实资产与会话；不隐式推理，不认证相机原生细节或人工批准。'
        nodes.append({'id':key,'label':node.label,'description':node.description,'stage':node.stage,
            'position':[node.position[0]*270,node.position[1]*220],'parameters':{'enabled':True,'strength':1.},
            'editable':False,'prepared':prepared,'status':status,'detail':detail,'action':node.action,
            'contract':asdict(node),'cache_key':compiled['cache']['node_keys'][key],
            'measured_seconds':elapsed.get(key),'timing_basis':'persisted_current_session_attempt_wall_seconds',
            'execution':tools if key=='selected_edit_tool' else None,'asset':None})
    return {'schema':'cosplay_workflow_graph.v1','run_id':run.get('run_id'),'readonly':True,'nodes':nodes,
        'connections':[{**edge,'type':registry[edge['from']].outputs[edge['output']].type} for edge in compiled['connections']],
        'layer_workflow':compiled,'canonical_processing_order':True,'editing_scope':scope,
        'editing_initialization':initialization,'bounded_dynamic_loop':compiled['bounded_dynamic_loop'],
        'execution_contract_graph_id':saved.get('graph_id'),
        'historical_execution_graph_id':saved.get('graph_id') if saved.get('graph_id')!=compiled['graph_id'] else None,
        'execution_contract_mutated':False,'display_basis':'typed_source_native_assets_and_actual_session_attempts',
        'edited_output':edited,'selected_appearance_kind':final.get('kind') if final else None,
        'creative_delivery_eligibility':copy.deepcopy(summary.get('creative_delivery_eligibility')),
        'cache_hits_inferred':False,'artistic_acceptance_inferred':False}
