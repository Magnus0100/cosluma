"""Cos-specific node controls over the production renderer's frozen layers.

The canonical connections express the rendering order. Nodes edit real layer
parameters; templates contain intent and layout, never another photograph's
generated pixels, masks, paths or credentials.
"""
from __future__ import annotations

import copy
import math
from cos_node_registry import NODES, EDITABLE, LAYER_KINDS, catalogue, compile_workflow

SCHEMA = 'cosplay_workflow_graph.v1'
TEMPLATE = 'cosplay_workflow_template.v1'
CATALOG = tuple((key, node.label, node.description, node.stage, *node.position, node.action)
                for key, node in NODES.items())
PRESETS = {
    'portrait':{'name':'仅人像美化','nodes':{'beauty':{'enabled':True,'strength':1.},
        'colour':{'enabled':False,'strength':1.},'background':{'enabled':False,'strength':1.},
        'boundary':{'enabled':False,'strength':1.},'effects':{'enabled':False,'strength':1.}}},
    'cos_scene':{'name':'完整 Cos 场景','nodes':{key:{'enabled':True,'strength':1.} for key in EDITABLE}},
    'scene_without_beauty':{'name':'保留原片人物外观','nodes':{
        key:{'enabled':key!='beauty','strength':1.} for key in EDITABLE}},
}


def _parameters(value):
    if not isinstance(value,dict) or set(value)-{'enabled','strength'}:
        raise ValueError('Workflow node controls require enabled and strength')
    if type(value.get('enabled')) is not bool:
        raise ValueError('Workflow node enabled must be a boolean')
    raw=value.get('strength',1.)
    if isinstance(raw,bool) or not isinstance(raw,(int,float)):
        raise ValueError('Workflow node strength must be numeric')
    strength=float(raw)
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Workflow node strength must lie in [0,1]')
    return {'enabled':value['enabled'],'strength':strength}


def validate_template(value):
    if not isinstance(value,dict) or value.get('schema') != TEMPLATE or set(value)-{'schema','name','nodes','layout'}:
        raise ValueError('A current Cos workflow template is required')
    nodes=value.get('nodes') or {}
    if not isinstance(nodes,dict) or set(nodes)-EDITABLE:
        raise ValueError('Unknown or noneditable workflow node')
    clean={'schema':TEMPLATE,'name':str(value.get('name') or 'Cos 工作流')[:100],
        'nodes':{key:_parameters(params) for key,params in nodes.items()},'layout':{}}
    layout=value.get('layout') or {}
    if not isinstance(layout,dict) or set(layout)-{row[0] for row in CATALOG}:
        raise ValueError('Unknown workflow layout node')
    for key,position in layout.items():
        if (not isinstance(position,(list,tuple)) or len(position)!=2
                or any(isinstance(x,bool) or not isinstance(x,(int,float))
                       or not math.isfinite(x) or abs(x)>100000 for x in position)):
            raise ValueError('Workflow node positions must be finite coordinates')
        clean['layout'][key]=list(position)
    return clean


def preset(name):
    return validate_template({'schema':TEMPLATE,**copy.deepcopy(PRESETS[name]),'layout':{}})


def compile_template(template, mode):
    controls = validate_template(template)['nodes'] if template else {}
    return compile_workflow(mode, controls)


def route_for_run(run):
    intent = (run.get('options') or {}).get('task_intent') or run.get('task_intent') or {}
    strategy = (run.get('execution_strategy')
                or (run.get('task_plan') or {}).get('execution_strategy')
                or (run.get('options') or {}).get('execution_strategy'))
    if not run.get('run_id'):
        return 'existing_reference'
    if strategy != 'layered_reference_transfer' or intent.get('mode') not in {
            'local_portrait', 'existing_reference', 'create_reference'}:
        raise ValueError('A current layered creative task is required for the workflow canvas')
    return intent['mode']


def build_graph(run, report=None, *, template=None):
    mode = route_for_run(run)
    from scoped_workflow import selected, build_scoped_graph, SOURCE_NATIVE_INITIALIZATION
    scope=(run.get('task_plan') or {}).get('editing_scope')
    iteration=(run.get('task_plan') or {}).get('editing_iteration')
    if mode=='create_reference' and (run.get('task_plan') or {}).get('editing_initialization')!=SOURCE_NATIVE_INITIALIZATION:
        raise ValueError('Unsupported retired creation; start a current source-native task')
    shaping=(run.get('task_plan') or {}).get('shaping_request')
    if selected(mode,scope,iteration,shaping_request=shaping):
        intent=run.get('task_intent') or (run.get('options') or {}).get('task_intent') or {}
        review=bool((intent.get('services') or {}).get('visual_review'))
        return build_scoped_graph(run,scope,review,editing_iteration=iteration,
            editing_autofinish=(run.get('task_plan') or {}).get('editing_autofinish'),shaping_request=shaping)
    revision=((run.get('options') or {}).get('appearance_revision') or {})
    stored=template or revision.get('workflow_graph')
    intention=validate_template(stored) if stored else {'nodes':{},'layout':{}}
    compiled=compile_workflow(mode,intention['nodes'])
    definitions=catalogue();active=set(compiled['order'])
    stages={row['id']:row for row in (report or {}).get('stages') or []}
    plan=revision.get('design_reconstruction_plan') or {}
    layers=plan.get('layers') or []
    nodes=[]
    for key,label,description,stage,column,row,action in CATALOG:
        selected=[item for item in layers if item.get('kind') in LAYER_KINDS.get(key,set())]
        beauty=revision.get('portrait_beauty') or {}
        if key=='beauty':
            strength=float(beauty.get('strength',1.)); enabled=bool(beauty.get('result') and strength)
            prepared=bool(beauty.get('result'))
        elif key in LAYER_KINDS:
            enabled=any(item.get('enabled') for item in selected)
            strength=max((float(item.get('strength',1.)) for item in selected),default=1.)
            prepared=bool(selected)
        else:
            enabled=True;strength=1.;prepared=False
        params=intention['nodes'].get(key,{'enabled':enabled,'strength':strength})
        stage_record=stages.get(stage,{})
        state=stage_record.get('status','pending')
        detail=stage_record.get('detail') or description
        if key=='beauty':
            state='complete' if prepared else 'pending'
            detail='已冻结生成脸部图层；可调整、保存和离线回放。' if prepared else '自动定位人脸，使用本机 BeautyGRPO 准备可编辑图层。'
        if key not in active:
            state='not_requested';detail='本地人像路线不使用此模块。'
        nodes.append({'id':key,'label':label,'description':description,'stage':stage,
            'position':intention['layout'].get(key,[column*270,row*220]),
            'parameters':params,'editable':key in EDITABLE,'prepared':prepared,
            'status':state,'detail':detail,'action':action,
            'contract':definitions[key],
            'cache_key':compiled['cache']['node_keys'].get(key),
            'asset':beauty.get('result') if key=='beauty' else None})
    return {'schema':SCHEMA,'nodes':nodes,
        'connections':[{**edge,'type':NODES[edge['from']].outputs[edge['output']].type}
                       for edge in compiled['connections']],
        'layer_workflow':compiled,
        'canonical_processing_order':True,'run_id':run.get('run_id')}


def template_from_graph(graph, name='Cos 工作流'):
    if graph.get('readonly'):
        return validate_template({'schema':TEMPLATE,'name':'只读任务流程（无可编辑模板）','nodes':{},'layout':{}})
    return validate_template({'schema':TEMPLATE,'name':name,
        'nodes':{node['id']:node['parameters'] for node in graph['nodes'] if node['editable']},
        'layout':{node['id']:node['position'] for node in graph['nodes']}})


def revision_for_template(run, template):
    """Compile editable nodes to production components, preserving frozen data."""
    from scoped_workflow import selected
    if route_for_run(run)=='create_reference' and (run.get('task_plan') or {}).get('editing_scope') is None:
        raise ValueError('Unsupported retired creation; start a current source-native task')
    if selected(route_for_run(run),(run.get('task_plan') or {}).get('editing_scope'),
                (run.get('task_plan') or {}).get('editing_iteration'),shaping_request=(run.get('task_plan') or {}).get('shaping_request')):
        raise ValueError('Scoped appearance workflow is read-only; source-space layer controls are unavailable')
    template=validate_template(template)
    compiled=compile_template(template,route_for_run(run))
    revision=((run.get('options') or {}).get('appearance_revision') or {})
    patch={'workflow_graph':template}
    if 'beauty' in template['nodes'] and revision.get('portrait_beauty'):
        params=template['nodes']['beauty'];beauty=copy.deepcopy(revision['portrait_beauty'])
        beauty['strength']=params['strength'] if params['enabled'] else 0.
        patch['portrait_beauty']=beauty
    if revision.get('design_reconstruction_plan'):
        plan=copy.deepcopy(revision['design_reconstruction_plan'])
        for node in compiled['order']:
            if node not in template['nodes']:continue
            params=template['nodes'][node]
            for layer in plan.get('layers') or []:
                if layer.get('kind') in LAYER_KINDS.get(node,set()):
                    layer['enabled']=params['enabled'];layer['strength']=params['strength']
        patch['design_reconstruction_plan']=plan
    return patch
