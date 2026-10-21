"""Real files, HTTP/CLI plans and durable batch retry regressions; no inference."""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from unittest.mock import patch

from PIL import Image

import creative_task as task
import runtime_preflight
import task_model_preflight
from backend_server import Backend, LocalHTTPServer, make_handler
from current_model_registry import (DIRECTORY_KEYS, PATH_KEYS, PROFILES, SCHEMA,
    verify_registered_models)
from task_model_preflight_smoke import mocked_task_capability


def rejected(fn):
    try:
        fn()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('Expected a rejected task before execution')


def snapshot(root):
    return {str(p.relative_to(root)): task.sha(p) for p in root.rglob('*') if p.is_file()}


def _varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7f
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _delimited(number, payload):
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _varint_field(number, value):
    return _varint(number << 3) + _varint(value)


def _tensor_type(elem_type, dims):
    shape = b''.join(_delimited(1, _varint_field(1, dim)) for dim in dims)
    return _delimited(1, _varint_field(1, elem_type) + _delimited(2, shape))


def _value_info(name, dims):
    return _delimited(1, name.encode('utf8')) + _delimited(2, _tensor_type(1, dims))


def onnx_identity_model():
    """A tiny real ONNX model: one float Identity node over a one-element input.

    The registry gate and any loader accept it as a model, so this offline
    fixture no longer hands a text placeholder to a real loading path.  It is
    never used for quality or for a silhouette: the graph only echoes its input,
    and suites that exercise the silhouette adapter inject their own provider.
    """
    node = (_delimited(1, b'x') + _delimited(2, b'y') + _delimited(3, b'fixture-node')
            + _delimited(4, b'Identity'))
    graph = (_delimited(1, node) + _delimited(2, b'cosluma-fixture-graph')
             + _delimited(11, _value_info('x', (1,))) + _delimited(12, _value_info('y', (1,))))
    return (_varint_field(1, 7) + _delimited(2, b'cosluma-fixture')
            + _delimited(7, graph) + _delimited(8, _varint_field(2, 13)))


def model_registry(root, *, model_bytes=None, pin=None, name='models.json', complete=False):
    """One registry whose silhouette model file is a real (tiny) ONNX model.

    complete=True also fills the source-geometry keys a layered route requires,
    with text placeholders that this suite never loads.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model = root / 'birefnet.onnx'
    model.write_bytes(onnx_identity_model() if model_bytes is None else model_bytes)
    onnx_root = root / 'onnx'
    (onnx_root / 'onnxruntime').mkdir(parents=True, exist_ok=True)
    values = {'schema': SCHEMA, 'ready': True, **PROFILES,
              'birefnet_model_path': str(model), 'onnx_runtime_root': str(onnx_root)}
    if complete:
        runtime = root / 'runtime' / 'python.exe'
        checkpoint = root / 'checkpoints' / 'parser.safetensors'
        for target in (runtime, checkpoint):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'metadata fixture, never loaded')
        for directory in (root / 'repo', root / 'vitmatte'):
            directory.mkdir(parents=True, exist_ok=True)
        values.update({'python': str(runtime), 'semantics_checkpoint': str(checkpoint),
                       'semantics_repo_dir': str(root / 'repo'),
                       'vitmatte_model_root': str(root / 'vitmatte')})
    if pin is not None:
        values['birefnet_model_sha256'] = pin
    path = root / name
    path.write_text(json.dumps(values), encoding='utf8')
    return path, model


def silhouette_preflight(root, registry):
    return runtime_preflight.preflight_for_run(
        root, registry_path=str(registry), require_local_models=True, mode='local_portrait',
        execution_strategy='layered_reference_transfer')


def preflight_error(root, registry):
    try:
        silhouette_preflight(root, registry)
    except runtime_preflight.RuntimePreflightError as exc:
        return exc
    raise AssertionError('An unusable model file passed the silhouette preflight: ' + str(registry))


def model_identity_gate():
    """A registered model file that is not a model fails during the preflight.

    t18 traced a real round that died inside ONNX Runtime with InvalidProtobuf
    because the registry pointed at a 30-byte text placeholder while the
    preflight reported ready.  Configuration, model loading and execution have
    to stay three separate user-visible failures.
    """
    with tempfile.TemporaryDirectory(prefix='cosplay-model-identity-') as folder:
        root = Path(folder).resolve()
        valid, model = model_registry(root / 'valid')
        identity = verify_registered_models({'birefnet_model_path': str(model)},
                                            keys=('birefnet_model_path',))['birefnet_model_path']
        assert identity['ir_version'] == 7 and identity['size'] == model.stat().st_size, identity
        accepted = preflight_error(root, valid)
        assert accepted.kind == 'configuration_onnx_runtime_unavailable', accepted.kind

        placeholder, placeholder_model = model_registry(
            root / 'placeholder', model_bytes=b'metadata fixture, never loaded')
        error = preflight_error(root, placeholder)
        assert error.kind == 'configuration_invalid_model', error.kind
        assert error.reason == 'text_placeholder', error.reason
        for expected in (str(placeholder_model), 'layer: configuration', 'size: 30 bytes',
                         'starts with ASCII text', placeholder_model.read_bytes()[:16].hex(),
                         'ir_version: unknown', 'not pinned by the registry'):
            assert expected in str(error), (expected, str(error))

        tiny, _tiny_model = model_registry(root / 'tiny', model_bytes=b'\x08\x07\x00\x00')
        error = preflight_error(root, tiny)
        assert (error.kind, error.reason) == ('configuration_invalid_model', 'too_small'), error.reason
        assert 'size: 4 bytes' in str(error), str(error)

        truncated, truncated_model = model_registry(root / 'truncated')
        truncated_model.write_bytes(onnx_identity_model()[:-24])
        error = preflight_error(root, truncated)
        assert (error.kind, error.reason) == ('configuration_invalid_model', 'truncated'), error.reason
        assert str(truncated_model) in str(error) and 'truncated' in str(error)

        text, _text_model = model_registry(root / 'text', model_bytes=b'x' * 4096)
        error = preflight_error(root, text)
        assert error.reason == 'text_placeholder', error.reason
        assert 'starts with ASCII text' in str(error)

        digest = hashlib.sha256(onnx_identity_model()).hexdigest()
        pinned, pinned_model = model_registry(root / 'pinned', pin=digest)
        verified = verify_registered_models(
            {'birefnet_model_path': str(pinned_model), 'birefnet_model_sha256': digest},
            keys=('birefnet_model_path',))['birefnet_model_path']
        assert verified['sha256'] == digest, verified
        wrong, _wrong_model = model_registry(root / 'wrong-digest', pin='0' * 64)
        error = preflight_error(root, wrong)
        assert error.kind == 'configuration_model_digest_mismatch', error.kind
        assert digest in str(error) and '0' * 64 in str(error), str(error)
        bad_pin, _bad_model = model_registry(root / 'bad-pin', pin='not-a-digest')
        error = preflight_error(root, bad_pin)
        assert (error.kind, error.reason) == ('configuration_invalid_model', 'invalid_pin'), error.reason

        # A model *loading* failure is a different user-visible failure than a
        # configuration one: the file was accepted, the runtime could not open
        # what it was given.
        layered, _layered_model = model_registry(root / 'layered', complete=True)
        with patch.object(runtime_preflight, '_probe_runtime',
                          return_value='No module named fixture_torch'):
            try:
                runtime_preflight.preflight_for_run(root, registry_path=str(layered),
                    require_local_models=True, execution_strategy='layered_reference_transfer')
            except runtime_preflight.RuntimePreflightError as exc:
                assert exc.kind == 'model_loading_runtime_probe_failed', exc.kind
                assert 'fixture_torch' in str(exc), str(exc)
            else:
                raise AssertionError('A failing model runtime probe was reported as ready')
    return {'invalid_model_rejected_before_rendering': True}


def fixture(root):
    values = {'schema': SCHEMA, 'ready': True, **PROFILES}
    for key in PATH_KEYS:
        target = root / 'models' / key
        target.parent.mkdir(parents=True, exist_ok=True)
        if key in DIRECTORY_KEYS:
            target.mkdir()
        elif key == 'birefnet_model_path':
            # A real (tiny) ONNX model, because the silhouette route consumes
            # this key: the registry gate and any loader now open it.  The other
            # keys stay text placeholders that this suite never loads.
            target.write_bytes(onnx_identity_model())
        else:
            target.write_bytes(b'metadata fixture, never loaded')
        values[key] = str(target)
    registry = root / 'models.json'
    registry.write_text(json.dumps(values), encoding='utf8')
    sources = []
    for i in range(3):
        path = root / f'source-{i}.png'
        Image.new('RGB', (25, 20), (40+i*20, 60, 80)).save(path)
        sources.append(str(path))
    reference = root / 'reference.png'
    Image.new('RGB', (23, 19), 'orange').save(reference)
    return registry, sources, str(reference)


def intent(mode):
    return {'schema_version': task.INTENT_SCHEMA, 'mode': mode, 'brief': '统一暗调、保持妆造',
            'constraints': {'preserve_background': mode == 'local_portrait', 'no_effects': mode == 'local_portrait'},
            'services': {'text_vision': mode == 'create_reference', 'visual_review': False}}


def main():
    gate = model_identity_gate()
    with tempfile.TemporaryDirectory(prefix='cosplay-creative-task-') as folder:
        root = Path(folder).resolve()
        registry, sources, reference = fixture(root)
        from project_cli import _checked_one_click_args
        rejected(lambda: _checked_one_click_args(argparse.Namespace(source=sources[0]), root))
        env = {'COSPLAY_PROVIDER_SETTINGS': str(root / 'provider-settings.json'),
               'COSPLAY_MODEL_SETTINGS': str(root / 'model-selection.json'),
               'COSPLAY_MODEL_REGISTRY': str(registry), 'TASK_TEST_KEY': 'local-test-only',
               'NO_PROXY': '127.0.0.1,localhost', 'PYTHONDONTWRITEBYTECODE': '1'}
        common = {'source': sources[0], 'model_registry': str(registry), 'provider': 'openai-compatible',
                  'api_base': 'https://example.invalid/v1', 'api_key_env': 'TASK_TEST_KEY',
                  'algorithm_profile': 'dinov3_vitl16',
                  'material_finish_profile': 'owner_adaptive_multiscale_skin_v2'}
        with patch.dict(os.environ, env):
            capability_patch = patch.object(task_model_preflight, 'inspect_task_models',
                                            side_effect=mocked_task_capability)
            capability_patch.start()
            before = snapshot(root)
            with patch.object(urllib.request, 'urlopen', side_effect=AssertionError('Preflight requested network')):
                local = task.plan_task(root, {**common, 'creative_task': intent('local_portrait')})
                imported = task.plan_task(root, {**common, 'reference': reference,
                                               'creative_task': intent('existing_reference')})
                created = task.plan_task(root, {**common, 'variant_count': 3,
                                              'creative_task': intent('create_reference')})
            assert snapshot(root) == before, 'Planning wrote project files'
            assert all(item['plan']['ready'] for item in (local, imported, created))
            from cos_node_registry import compile_workflow
            for item in (local, imported, created):
                assert item['plan']['layer_workflow'] == compile_workflow(item['plan']['mode'],
                    editing_scope=item['plan'].get('editing_scope'),
                    visual_review=item['plan']['intent']['services']['visual_review'],
                    editing_autofinish=item['plan'].get('editing_autofinish'),
                    editing_initialization=item['plan'].get('editing_initialization'),
                    initial_variant_count=item['plan']['image_budget'].get('initial_reference',1))
                task.validate_task_plan(item['plan'],item['plan']['intent'])
            assert local['plan']['services']['offline'] and not local['request']['complete_post']
            assert imported['plan']['image_budget'] == dict.fromkeys(task.image_budget(intent('existing_reference')), 0)
            assert imported['request']['complete_post'] and not imported['request']['generate']
            assert created['plan']['generation_submissions_upper_bound'] == 3
            omitted = intent('create_reference'); omitted['services'].pop('visual_review')
            omitted['services'].pop('text_vision')
            assert task.normalize_task_intent(omitted)['services']['visual_review'] is True
            assert task.normalize_task_intent(omitted)['services']['text_vision'] is True
            assert created['plan']['services']['image_generation'] is True
            assert created['plan']['services']['offline'] is False
            for prepared in (local, imported, created):
                assert prepared['plan']['intent']['services']['visual_review'] is False
                assert prepared['plan']['services']['visual_review'] is False
                assert not any('复审' in row['label'] for row in prepared['plan']['stages'])
                assert '成片与匹配原尺寸审片窗口' not in prepared['plan']['services']['sent_content']
            opted_in = copy.deepcopy(imported['request'])
            opted_in['creative_task']['services']['text_vision'] = True
            opted_in['creative_task']['services']['visual_review'] = True
            reviewed_plan = task.plan_task(root, opted_in)
            assert reviewed_plan['plan']['ready'] and reviewed_plan['request']['complete_post']
            assert reviewed_plan['plan']['plan_id'] != imported['plan']['plan_id']
            assert '成片与匹配原尺寸审片窗口' in reviewed_plan['plan']['services']['sent_content']
            assert any('复审' in row['label'] for row in reviewed_plan['plan']['stages'])
            assert task.execution_request(root, {**reviewed_plan['request'],
                'plan_id': reviewed_plan['plan']['plan_id']})['task_intent']['services']['visual_review'] is True
            for mode, enabled in (('local_portrait', False), ('existing_reference', False)):
                invalid = intent(mode)
                invalid['services'].update(text_vision=enabled, visual_review=True)
                rejected(lambda: task.normalize_task_intent(invalid))
            invalid = intent('create_reference'); invalid['services']['visual_review'] = 1
            rejected(lambda: task.normalize_task_intent(invalid))
            for prepared in (local, imported, created):
                repeated = task.plan_task(root, prepared['request'])
                assert repeated['plan']['plan_id'] == prepared['plan']['plan_id'], 'Safe request changed plan identity'
                execution = task.execution_request(root, {**prepared['request'], 'plan_id': prepared['plan']['plan_id']})
                assert execution['task_plan']['plan_id'] == prepared['plan']['plan_id']
                assert 'creative_task' not in execution and execution['task_intent'] == prepared['plan']['intent']
            pending = {**imported['request'], 'plan_id': imported['plan']['plan_id']}
            changed = copy.deepcopy(pending)
            changed['creative_task']['brief'] += ' 变化'
            rejected(lambda: task.execution_request(root, changed))
            changed = copy.deepcopy(pending)
            changed['creative_task']['constraints']['preserve_background'] = True
            rejected(lambda: task.execution_request(root, changed))
            source_data = Path(sources[0]).read_bytes()
            Image.new('RGB', (25, 20), 'red').save(sources[0])
            rejected(lambda: task.execution_request(root, pending))
            Path(sources[0]).write_bytes(source_data)
            ref_data = Path(reference).read_bytes()
            Image.new('RGB', (23, 19), 'blue').save(reference)
            rejected(lambda: task.execution_request(root, pending))
            Path(reference).write_bytes(ref_data)
            registry_data = registry.read_bytes()
            registry.write_bytes(registry_data + b'\n')
            rejected(lambda: task.execution_request(root, pending))
            registry.write_bytes(registry_data)
            for extra in ({'algorithm_profile': 'dinov2'}, {'material_finish_profile': 'legacy'},
                          {'floor_repair': {}}, {'token': 'forbidden'}, {'generate': True}):
                rejected(lambda: task.plan_task(root, {**pending, **extra}))
            offline = copy.deepcopy(pending)
            offline['creative_task']['services']['text_vision'] = False
            unplanned = task.plan_task(root, offline)
            assert unplanned['plan']['ready'] and unplanned['request']['complete_post']
            assert unplanned['plan']['services']['text_vision_requests_upper_bound'] == 0
            assert not task.execution_request(root, {**unplanned['request'],
                'plan_id': unplanned['plan']['plan_id']})['task_intent']['services']['text_vision']
            deep = root / ('deep-' + 'a'*90)
            deep_before = snapshot(root)
            long_plan = task.plan_task(deep, imported['request'])
            assert not long_plan['plan']['ready'] and not long_plan['plan']['capabilities']['storage_preflight']['ready']
            assert not deep.exists() and snapshot(root) == deep_before, 'Path preflight wrote a directory'
            # Local editing does not require full-reference assets to exist.
            dino = Path(json.loads(registry.read_text())['dinov3_model_root'])
            dino.rmdir()
            assert task.plan_task(root, local['request'])['plan']['ready']
            assert not task.plan_task(root, imported['request'])['plan']['ready']
            dino.mkdir()

            backend = Backend(root)
            server = LocalHTTPServer(('127.0.0.1', 0), make_handler(backend))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

            def post(path, body):
                request = urllib.request.Request(base+path, data=json.dumps(body).encode('utf8'),
                    headers={'Content-Type': 'application/json'})
                with opener.open(request, timeout=5) as response:
                    return json.load(response)

            try:
                capability_patch.stop()
                http = post('/api/tasks/plan', imported['request'])
                assert not http['plan']['ready'], 'Toy model files must not pass real installation checks'
                request_file = root / 'request.json'
                request_file.write_text(json.dumps(imported['request']), encoding='utf8')
                proc = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).resolve().parents[1]/'scripts/project_cli.py'),
                    '--root', str(root), 'task-plan', '--request-file', str(request_file)],
                    capture_output=True, text=True, encoding='utf8', timeout=20)
                assert proc.returncode == 0, proc.stderr
                assert json.loads(proc.stdout)['plan']['plan_id'] == http['plan']['plan_id']
                capability_patch.start()
                calls = []

                def submit(body):
                    calls.append(copy.deepcopy(body))
                    job = backend.jobs.create(backend._one_click_payload(body, body['source']))
                    backend._job_workers[job['job_id']] = threading.current_thread()
                    return {'ok': True, 'async': True, 'job_id': job['job_id'], 'job': job}

                with patch.object(backend, 'one_click', side_effect=submit):
                    started = post('/api/tasks/start', pending)
                    assert started['job']['task_plan']['plan_id'] == imported['plan']['plan_id']
                    assert calls[-1]['cosplay_correction_budget'] == 0
                    body = {k: v for k, v in common.items() if k in task.SHARED_FIELDS}
                    body.update(creative_task=intent('existing_reference'),
                        items=[{'source': source, 'reference': reference} for source in sources[1:]])
                    batch = task.plan_batch(root, body)['batch_plan']
                    assert batch['ready'] and batch['image_budget_total'] == imported['plan']['image_budget']
                    assert len({row['plan']['plan_id'] for row in batch['items']}) == 2
                    rejected(lambda: task.plan_batch(root, {**body, 'generation_design': {'wrong': 'shared'}}))
                    rejected(lambda: task.plan_batch(root, {**body, 'items': [body['items'][0]]*2}))
                    count = len(calls)
                    altered = copy.deepcopy(body)
                    altered['creative_task']['brief'] += ' stale'
                    rejected(lambda: backend.start_task_batch({**altered, 'plan_id': batch['plan_id']}))
                    assert len(calls) == count, 'Stale batch partially submitted'
                    result = post('/api/tasks/batch', {**body, 'plan_id': batch['plan_id']})['batch']
                    assert len(result['items']) == 2
                    for item in result['items']:
                        assert item['request']['preserve_project_selection']
                        assert item['request']['task_plan']['inputs']['source']['path'] == item['source']
                    row = result['items'][0]
                    backend.jobs.update(row['job_id'], status='failed', stage='test_failure')
                    outcomes = []

                    def retry():
                        try:
                            outcomes.append(backend.retry_task_batch_item(result['batch_id'], {'index': 0}))
                        except ValueError:
                            outcomes.append('rejected')

                    runners = [threading.Thread(target=retry) for _ in range(2)]
                    for runner in runners:
                        runner.start()
                    for runner in runners:
                        runner.join(5)
                        assert not runner.is_alive()
                    assert sum(isinstance(item, dict) for item in outcomes) == 1, outcomes
                    assert len(calls) == count+3, 'Duplicate retry submitted'
                    saved = backend.batch_status(result['batch_id'])['batch']
                    assert len(saved['items'][0]['attempts']) == 2 and len(saved['items'][1]['attempts']) == 1
                    assert calls[-1]['task_intent']['mode'] == 'existing_reference'
                    assert not calls[-1]['generate'] and calls[-1]['cosplay_correction_budget'] == 0
                    create_body = {**body, 'creative_task': intent('create_reference'),
                                   'items': [{'source': sources[0]}]}
                    cp = task.plan_batch(root, create_body)['batch_plan']
                    cr = backend.start_task_batch({**create_body, 'plan_id': cp['plan_id']})['batch']
                    backend.jobs.update(cr['items'][0]['job_id'], status='failed', stage='unknown_generation')
                    count = len(calls)
                    rejected(lambda: backend.retry_task_batch_item(cr['batch_id'], {'index': 0}))
                    assert len(calls) == count, 'Unknown image submission silently retried'
                    from current_stack_fixture import prepare_fixture
                    from generation_design import DESIGN_VERSION
                    import single_source_workflow as workflow
                    parent = prepare_fixture(root / 'saved-current')
                    parent['sample'] = 'saved'
                    parent['auto_draft'] = {'path': str(root / 'saved.png')}
                    parent['generation_design'] = {'schema_version': DESIGN_VERSION, 'status': 'ready'}
                    saved_intent = intent('existing_reference')
                    saved_intent['constraints'] = {'preserve_background': True, 'no_effects': True}
                    parent['task_intent'] = saved_intent
                    parent['options'].update({**common, 'task_intent': copy.deepcopy(saved_intent)})
                    parent['options'].pop('source')
                    parent['task_plan'] = task.plan_task(root, {**common, 'source': parent['source']['path'],
                        'reference': parent['reference']['path'], 'creative_task': saved_intent})['plan']
                    initial_parent = task.digest(parent)
                    with patch.object(workflow, 'load_run', return_value=(root / 'saved-manifest.json', parent)):
                        backend.complete_source_run(parent['run_id'], {})
                        assert calls[-1]['task_intent']['constraints'] == saved_intent['constraints']
                        assert sum(calls[-1]['task_plan']['image_budget'].values()) == 0
                        assert calls[-1]['cosplay_correction_budget'] == 0 and not calls[-1]['generate']
                        retried = backend.retry_source_run(parent['run_id'], {})
                        assert calls[-1]['complete_post'] and not calls[-1].get('preserve_effective_prompt')
                        assert calls[-1]['task_intent'] == saved_intent and not calls[-1]['generate']
                        backend.jobs.update(retried['job_id'],status='succeeded',stage='fixture_complete')
                        rejected(lambda: backend.retry_source_run(parent['run_id'], {'generate': True}))
                    assert task.digest(parent) == initial_parent, 'Continuation changed its saved parent'
                    assert not hasattr(backend,'prepare_generation')
                    rejected(lambda: backend.recognize_character({'source':sources[0],'creative_task':intent('local_portrait')}))
                    allowed={**reviewed_plan['request'],'task_intent':reviewed_plan['plan']['intent'],
                        'task_plan':reviewed_plan['plan']}
                    with patch('character_profile.analyze_character',return_value={'identification':'CPU_checked_fixture'}) as recognize:
                        result=backend.recognize_character(allowed)
                        assert result['profile']['identification']=='CPU_checked_fixture' and recognize.call_count==1
                        rejected(lambda: backend.recognize_character({**allowed,'source':sources[1]}))
                        assert recognize.call_count==1,'Stale recognition scope reached the model'
                orphan = backend.jobs.create({'kind': 'one_click', 'source': sources[0]})
                recovered = Backend(root).jobs_list()['jobs']
                state = next(item for item in recovered if item['job_id'] == orphan['job_id'])
                assert state['status'] == 'failed' and state['stage'] == 'interrupted'
                for retired_route in ('/api/run','/api/generation/prepare',
                    '/api/source-runs/retired/plan-light','/api/source-runs/retired/plan-color'):
                    try:post(retired_route,{})
                    except urllib.error.HTTPError as error:assert error.code==404
                    else:raise AssertionError('Retired route still exists: '+retired_route)
            finally:
                capability_patch.stop()
                server.shutdown()
                server.server_close()
                thread.join(5)
    print(json.dumps({'ok': True, 'pure_preflight': True, 'content_bound_plan': True,
        'real_http_cli_parity': True, 'durable_independent_batches': True,
        'serialized_failed_retry': True, 'unknown_generation_retry_blocked': True,
        'execution_model_capability_mocked': True, 'http_cli_model_preflight_real': True, **gate}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
