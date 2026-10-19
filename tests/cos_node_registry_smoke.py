"""Production contracts, invalid graph rejection and shared HTTP/plan identity."""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import urllib.request

from backend_server import Backend, LocalHTTPServer, make_handler
from cos_node_registry import NODES, Port, catalogue, compile_workflow, validate_connections
from cos_workflow_cache import compare
from cos_workflow_graph import build_graph, compile_template, preset, revision_for_template, route_for_run
from current_contract import validate_current_request
import creative_task


def rejected(fn, message):
    try:
        fn()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError('Invalid graph was accepted')


def main():
    root = Path(__file__).resolve().parents[1]
    # Contracts name real production entry points, without importing GPU stacks.
    for node in NODES.values():
        module, function = node.implementation.split('.')
        tree = ast.parse((root/'scripts'/(module+'.py')).read_text(encoding='utf8'))
        assert function in {row.name for row in tree.body if isinstance(row, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert node.alpha_role != 'produce' or node is NODES['matte']
    scene = compile_workflow('existing_reference')
    ids = scene['order']; edges = scene['connections']
    assert scene == compile_workflow('existing_reference')
    assert scene['cache']['strategy'].startswith('producer_code_controls')
    assert scene['cache']['signature_reuse_is_not_a_verified_pixel_cache_hit']
    assert scene['cache']['affected_by']['beauty'][-1] == 'export'
    order, inputs = validate_connections(ids, edges)
    assert inputs['matte']['image'] == ['source', 'image']
    assert inputs['beauty']['image'] == ['source', 'image']
    assert inputs['colour']['image'] == ['beauty', 'image']
    assert inputs['colour']['subject'] == ['matte', 'subject']
    assert order.index('beauty') < order.index('colour') < order.index('boundary') < order.index('composite')
    rejected(lambda: validate_connections(ids, edges[1:]), 'Disconnected')
    rejected(lambda: validate_connections(ids, edges + [edges[0]]), 'Multiple writers')
    broken = copy.deepcopy(edges);broken[0]['output'] = 'unknown'
    rejected(lambda: validate_connections(ids, broken), 'socket')
    broken = copy.deepcopy(edges);broken[0]['from'] = 'missing'
    rejected(lambda: validate_connections(ids, broken), 'missing node')
    # Replace the alpha pair with appearance RGB: never permit implicit alpha.
    broken = copy.deepcopy(edges)
    row = next(row for row in broken if row['to']=='colour' and row['input']=='subject')
    row.update({'from':'beauty', 'output':'image'})
    rejected(lambda: validate_connections(ids, broken), 'type/domain mismatch')
    broken = copy.deepcopy(edges)
    next(row for row in broken if row['to']=='beauty').update({'from':'beauty'})
    rejected(lambda: validate_connections(ids, broken), 'cycle')
    registry = {**NODES, 'source':replace(NODES['source'], outputs={
        'image':Port('DISPLAY_IMAGE','scene_linear_sRGB','source_canvas')})}
    rejected(lambda: validate_connections(ids, edges, registry=registry), 'type/domain mismatch')
    local = compile_workflow('local_portrait')
    assert set(local['order']) == {'source','matte','beauty','composite','export'}
    assert len(local['connections']) == 6
    rejected(lambda: build_graph({'run_id':'unsupported-output','reference':{'sha256':'reference'}}),
             'current layered creative task')
    template = preset('cos_scene')
    before = compile_template(template,'existing_reference')
    template['layout']['beauty'] = [900, -200]
    assert before == compile_template(template,'existing_reference'), 'Layout invalidated a computation'
    template['nodes']['beauty']['strength'] = .4
    changed_workflow = compile_template(template,'existing_reference')
    assert before['graph_id'] != changed_workflow['graph_id']
    delta = compare(before, changed_workflow)
    assert delta['changed'] == ['beauty']
    assert delta['invalidated'] == ['beauty','colour','boundary','composite','export']
    assert delta['reused'] == ['reference','source','matte','background','effects']
    baseline = {'foreground_alpha': {'sha256':'original-alpha'}, 'foreground_rgb':{'sha256':'original-rgb'}}
    run = {'execution_strategy':'layered_reference_transfer','task_intent':{'mode':'existing_reference'},'run_id':'scene','reference':{'sha256':'reference'},'options':{'appearance_revision':{
        'portrait_beauty':{'result':'frozen-face.json','strength':1.},
        'design_reconstruction_plan':{'mask_roles':baseline,'layers':[
            {'kind':'reference_appearance_stack','enabled':True,'strength':1.},
            {'kind':'source_boundary_light','enabled':True,'strength':1.}]}}}}
    changed = revision_for_template(run,template)
    assert changed['portrait_beauty']['strength'] == .4
    assert changed['design_reconstruction_plan']['mask_roles'] == baseline
    graph = build_graph(run,template=template)
    assert graph['layer_workflow'] == compile_template(template,'existing_reference')
    rejected(lambda: validate_current_request({'appearance_revision':{'workflow_graph':{
        **template,'connections':[]}}}), 'current Cos workflow template')
    rejected(lambda: validate_current_request({'workflow_graph':{
        **template,'connections':[]}}), 'current Cos workflow template')
    intent = creative_task.normalize_task_intent({'schema_version':creative_task.INTENT_SCHEMA,'mode':'local_portrait'})
    plan = {'schema_version':creative_task.PLAN_SCHEMA, 'intent':intent,'mode':intent['mode'],
        'execution_strategy':'layered_reference_transfer', 'ready':True,
        'image_budget':creative_task.image_budget(intent),'plan_id':'a'*64,'layer_workflow':local}
    assert creative_task.validate_task_plan(plan,intent) == plan
    plan['layer_workflow'] = copy.deepcopy(local)
    plan['layer_workflow']['connections'][0]['from'] = 'beauty'
    rejected(lambda: creative_task.validate_task_plan(plan,intent), 'node contracts')
    with tempfile.TemporaryDirectory() as directory:
        backend = Backend(Path(directory))
        server = LocalHTTPServer(('127.0.0.1',0),make_handler(backend))
        worker = threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/api/workflow-nodes',timeout=5) as response:
                payload = json.load(response)
            assert payload['ok'] and payload['nodes'] == json.loads(json.dumps(catalogue()))
            assert payload['editable_connections'] is False
            assert payload['cache_contract']['schema'] == 'cosplay_workflow_cache.v1'
            assert payload['cache_contract']['layout_ignored'] is True
            assert payload['cache_contract']['node_keys']['beauty']
        finally:
            server.shutdown();server.server_close();worker.join(timeout=5)
    print('cos_node_registry_smoke: passed')


if __name__ == '__main__':
    main()
