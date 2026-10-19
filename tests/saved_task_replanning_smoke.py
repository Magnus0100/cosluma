"""HTTP retry accepts current saved tasks and rejects obsolete producer graphs."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import urllib.error
import urllib.request
from unittest.mock import patch

from PIL import Image

from backend_server import Backend, LocalHTTPServer, make_handler
from current_stack_fixture import prepare_fixture
import creative_task as task
from production_coordinator import route_contract
import reference_pipeline as pipeline
import single_source_workflow as workflow
from task_model_preflight_smoke import build_registry_fixture, mocked_task_capability


def reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Saved graph was allowed to execute as a current checked plan')


def main():
    with tempfile.TemporaryDirectory(prefix='saved-task-replan-') as directory, ExitStack() as stack:
        root = Path(directory)
        parent = prepare_fixture(root/'assets')
        registry = build_registry_fixture(root/'models')
        intent = task.normalize_task_intent({'schema_version': task.INTENT_SCHEMA, 'mode': 'existing_reference',
            'brief': '保留原片身份，重新协调光色', 'services': {'text_vision': False, 'visual_review': False}})
        stack.enter_context(patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability))
        checked = task.plan_task(root, {'source': parent['source']['path'], 'reference': parent['reference']['path'],
            'model_registry': str(registry), 'creative_task': intent})
        old = copy.deepcopy(checked['plan'])
        graph = old['layer_workflow']
        # This is the actual pre-producer-key cache strategy and structure.
        # Its node keys and graph id describe the older producer closure.
        graph['cache']['strategy'] = 'implementation_controls_and_ordered_upstream_input_signature'
        graph['cache'].pop('signature_reuse_is_not_a_verified_pixel_cache_hit', None)
        graph['cache']['node_keys'] = {name: task.digest({'old_producer': name}) for name in graph['order']}
        graph['graph_id'] = task.digest({key: value for key, value in graph.items() if key != 'graph_id'})
        old.pop('preparation_policy', None)
        old.pop('final_repair_only', None)
        old['plan_id'] = task.digest({key: value for key, value in old.items() if key != 'plan_id'})
        parent.update(run_id='run-ab', sample='saved-offline', status='needs_visual_review',
            task_intent=intent, task_plan=checked['plan'],
            generation_design={'status': 'not_requested', 'reason': 'Task explicitly disabled text/vision planning.'})
        preview = root/'old-saved.png'; Image.new('RGB', (24, 20), '#657585').save(preview)
        parent.update(auto_draft={'outputs': {'auto_draft': str(preview)}}, artifacts={'auto_draft': str(preview)})
        parent['options'].update(model_registry=str(registry), algorithm_profile='dinov3_vitl16',
            task_intent=intent, task_plan=copy.deepcopy(checked['plan']), generation_design=copy.deepcopy(parent['generation_design']))
        path = root/'outputs/runs/run-ab/manifest.json'
        workflow.write_json(path, parent)
        original = path.read_bytes()
        reject(lambda: task.validate_task_plan(old, intent))
        reject(lambda: route_contract({'task_intent': intent, 'task_plan': old}))
        reject(lambda: task.execution_request(root, {**checked['request'], 'plan_id': old['plan_id']}))
        reject(lambda: task.saved_task_intent_for_replanning(old, intent,
            source_sha256=parent['source']['sha256'], reference_sha256=parent['reference']['sha256']))

        # Real HTTP, durable job, CLI/coordinator and child-save boundaries;
        # expensive numeric producers are bounded CPU fixtures.
        stages = []
        fresh_plan = copy.deepcopy(parent['options']['appearance_revision']['design_reconstruction_plan'])
        fresh_plan['layers'][0]['params']['producer_fixture'] = 'current producer'
        def prepare(_root, run, **kwargs):
            stages.append((run['run_id'], 'prepare_current'))
            return {'cache_hit': False, 'design_reconstruction_plan': copy.deepcopy(fresh_plan)}
        def boundary(_root, run, _output, **kwargs):
            stages.append((run['run_id'], 'boundary'))
            return {'ok': True, 'no_change': True, 'receipt': {},
                'design_reconstruction_plan': run['options']['appearance_revision']['design_reconstruction_plan']}
        def render(_root, run, _args):
            stages.append((run['run_id'], 'render'))
            assert run['_native_plan_prepared_for_render']
            assert run['task_plan']['layer_workflow'] == checked['plan']['layer_workflow']
            output = root/'outputs/runs'/run['run_id']/'render.png'
            Image.new('RGB', (24, 20), '#566779').save(output)
            return {'outputs': {'auto_draft': str(output)},
                'metadata': {'run_id': run['run_id'], 'revision': run['revision']},
                'review': {'checks': {key: 'not_checked' for key in workflow.VISUAL_CHECK_KEYS}}}
        patches = (
            patch('design_reconstruction.materialize_plan', side_effect=lambda _root, _run, value: copy.deepcopy(value)),
            patch('scene_design.bind_plan_receiver_intent', side_effect=lambda _run, value: value),
            patch('design_scene_assets.prepare_native_receivers', side_effect=lambda _root, _run, value: value),
            patch('design_reconstruction_pipeline.ready', return_value=True),
            patch('runtime_memory.native_analysis_cpu_bytes', return_value=0),
            patch('beautygrpo_prepare.release_idle_for_stage', return_value={}),
            patch('start_qwen_image.release_for_run', return_value={'state': 'not_managed_qwen'}),
            patch('runtime_preflight.preflight_for_run', return_value={'status': 'ready'}),
            patch.object(pipeline, 'prepare', side_effect=prepare),
            patch.object(workflow, '_prepare_task_portrait_beauty'),
            patch('boundary_light_coordination_worker.prepare_for_workbench', side_effect=boundary),
            patch.object(workflow, 'render_candidate', side_effect=render),
            patch.object(workflow, 'diagnose_cosplay_post'),
        )
        for active_patch in patches:
            stack.enter_context(active_patch)
        backend = Backend(root, embedded=True)
        server = LocalHTTPServer(('127.0.0.1', 0), make_handler(backend))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def post(suffix, body=None, expected=200):
            request = urllib.request.Request(base+suffix, data=json.dumps(body or {}).encode('utf8'),
                headers={'Content-Type': 'application/json'})
            try:
                response = opener.open(request, timeout=5)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                result = json.load(response)
                assert response.status == expected, (response.status, result)
                return result
        try:
            for endpoint in ('complete-post', 'retry'):
                started = post('/api/source-runs/run-ab/' + endpoint)
                assert started['task_plan']['preparation_policy'] == 'prepare_current'
                assert started['task_plan']['image_budget'] == dict.fromkeys(old['image_budget'], 0)
                assert started['task_plan']['layer_workflow'] == checked['plan']['layer_workflow']
                assert started['task_plan']['plan_id'] != old['plan_id']
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    job = backend.jobs.get(started['job_id'])
                    if job['status'] in {'succeeded', 'failed', 'blocked', 'cancelled'}:
                        break
                    time.sleep(.02)
                assert job['status'] == 'succeeded', job
                child = workflow.load_run(root, job['run_id'])[1]
                assert child['parent_run_id'] == 'run-ab' and child['revision'] == parent['revision'] + 1
                assert child['task_intent'] == intent
                assert child['generation_design']['status'] == 'not_requested'
                assert child['generation_budget']['image_submissions'] == 0
                assert child['task_plan']['layer_workflow'] == checked['plan']['layer_workflow']
                assert [stage for run_id, stage in stages if run_id == child['run_id']] == ['prepare_current', 'boundary', 'render']
                assert path.read_bytes() == original
            count = len(backend.jobs.list())
            # Neither historical binding checks nor current execution gates
            # are relaxed by the replan-only reader.
            for field in ('source', 'reference', 'budget'):
                changed = copy.deepcopy(parent)
                if field == 'budget':
                    changed['task_plan']['image_budget']['independent_effect'] = 1
                else:
                    changed['task_plan']['inputs'][field]['sha256'] = '0'*64
                workflow.write_json(path, changed)
                for endpoint in ('complete-post', 'retry'):
                    post('/api/source-runs/run-ab/' + endpoint, expected=400)
            path.write_bytes(original)
            for role in ('source', 'reference'):
                image = Path(parent[role]['path']); content = image.read_bytes()
                try:
                    image.write_bytes(content + b'changed input')
                    for endpoint in ('complete-post', 'retry'):
                        post('/api/source-runs/run-ab/' + endpoint, expected=400)
                finally:
                    image.write_bytes(content)
            for endpoint in ('complete-post', 'retry'):
                post('/api/source-runs/run-ab/' + endpoint, {'generate': True}, expected=400)
            post('/api/tasks/start', {**checked['request'], 'plan_id': old['plan_id']}, expected=400)
            assert len(backend.jobs.list()) == count and path.read_bytes() == original
        finally:
            server.shutdown(); server.server_close(); thread.join(5)
        print(json.dumps({'ok': True, 'offline_unrequested_design_http_redo_and_retry': True,
            'obsolete_producer_graph_rejected': True, 'new_checked_graph_executes': True,
            'wrong_source_reference_budget_refused': True, 'zero_generation_parent_unchanged': True,
            'model_inferences': 0, 'image_generation_requests': 0}))


if __name__ == '__main__':
    main()
