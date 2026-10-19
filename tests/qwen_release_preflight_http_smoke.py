"""Local HTTP/process checks for Qwen release discovery; no models are loaded."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import runtime_memory as memory
import start_qwen_image as qwen


def serve(folder: Path) -> None:
    state = {'generation':None, 'unloaded':False}
    expected_main = str((folder/'ComfyUI/main.py').resolve())

    class Handler(BaseHTTPRequestHandler):
        def send(self, body: dict, code: int = 200) -> None:
            data = json.dumps(body).encode('utf8')
            self.send_response(code)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def configure(self, payload=None) -> dict:
            config = json.loads((folder/'server-config.json').read_text('utf8'))
            if config['generation'] != state['generation']:
                state.update(generation=config['generation'], unloaded=False)
            with (folder/'requests.jsonl').open('a', encoding='utf8') as log:
                log.write(json.dumps({'method':self.command, 'path':self.path,
                                      'payload':payload})+'\n')
            return config

        def do_GET(self):
            config = self.configure()
            if self.path == '/system_stats':
                time.sleep(config.get('status_delay', 0))
                if config.get('status_error'):
                    self.send({'error':'synthetic unavailable service'}, 503)
                    return
                if config.get('replace_owner'):
                    replacement = {**ready, 'started_at':ready['started_at']-1000}
                    (folder/'server-process.json').write_text(json.dumps(replacement), 'utf8')
                self.send({'system':{'argv':[expected_main+'/foreign' if config.get('foreign_argv')
                                             else expected_main]},
                           'devices':[{'type':'cuda', 'torch_vram_total':8*memory.GIB,
                                       'torch_vram_free':8*memory.GIB if state['unloaded'] else 0}]})
            elif self.path == '/queue':
                self.send({'queue_running':[1] if config.get('busy') else [], 'queue_pending':[]})
            else:
                self.send({'error':'unexpected path'}, 404)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.configure(payload)
            if self.path == '/free':
                state['unloaded'] = bool(payload['unload_models'])
                self.send({})
            else:
                self.send({'error':'unexpected path'}, 404)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    ready = {'pid':os.getpid(), 'endpoint':f'http://127.0.0.1:{server.server_port}',
             'python':str(qwen.process_executable(os.getpid())), 'main':expected_main,
             'started_at':qwen.process_created_at(os.getpid())}
    (folder/'ready.json').write_text(json.dumps(ready), 'utf8')
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--negative-delay', type=float, default=2)
    args = parser.parse_args()
    if args.serve:
        serve(args.serve)
        return
    healthy = {'available_bytes':48*memory.GIB, 'available_commit_bytes':48*memory.GIB,
               'total_bytes':64*memory.GIB}
    demand = {'next_stage_cpu_bytes':memory.BEAUTY_COLD_PHYSICAL_BYTES,
              'next_stage_commit_bytes':memory.BEAUTY_COLD_COMMIT_BYTES, 'next_stage':'beautygrpo'}
    run = {'options':{'provider':'qwen_image_2_1_uncensored_fast'},
           'reference':{'provider':'qwen_image_2_1_uncensored_fast'}, 'task_intent':'existing_reference'}
    results = {}
    with tempfile.TemporaryDirectory(prefix='cosplay-qwen-release-http-') as temporary:
        folder = Path(temporary)
        main_script=folder/'ComfyUI/main.py'
        main_script.parent.mkdir(parents=True)
        main_script.write_text('import runpy\nrunpy.run_path('+repr(str(Path(__file__).resolve()))
                               +', run_name="__main__")\n','utf8')
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        process = subprocess.Popen([sys.executable, str(main_script), '--serve', str(folder)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags)
        try:
            deadline = time.monotonic()+10
            while not (folder/'ready.json').exists():
                if process.poll() is not None:
                    raise RuntimeError(process.stderr.read().decode('utf8', errors='replace'))
                if time.monotonic() >= deadline:
                    raise RuntimeError('Synthetic local HTTP service did not become ready')
                time.sleep(.01)
            owner = json.loads((folder/'ready.json').read_text('utf8'))
            endpoint = owner['endpoint']
            assert qwen.process_alive(owner['pid'])
            assert qwen.process_executable(owner['pid']) == Path(owner['python']).resolve()
            assert qwen.process_created_at(owner['pid']) == owner['started_at']
            finished = subprocess.Popen([sys.executable, '-c', 'pass'], creationflags=flags,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finished.wait(timeout=5)
            assert not qwen.process_alive(finished.pid)

            def exercise(name, *, module=qwen, marker=owner, config=None, snapshot=healthy,
                         selected_run=run, configured=True):
                (folder/'server-config.json').write_text(json.dumps({'generation':name, **(config or {})}), 'utf8')
                (folder/'requests.jsonl').write_text('', 'utf8')
                marker_path = folder/'server-process.json'
                if marker is None:
                    marker_path.unlink(missing_ok=True)
                else:
                    marker_path.write_text(marker if isinstance(marker, str) else json.dumps(marker), 'utf8')
                status = module.service_status
                with ExitStack() as stack:
                    stack.enter_context(patch.object(module, 'ENDPOINT', endpoint))
                    # Keep the real HTTP implementation and its original 3s timeout.
                    stack.enter_context(patch.object(module, 'service_status', side_effect=lambda:status(endpoint)))
                    stack.enter_context(patch.object(module, 'deployment_root', return_value=folder if configured else None))
                    stack.enter_context(patch.object(memory, 'memory_status', return_value=snapshot))
                    try:
                        receipt = module.release_for_run(selected_run, **demand)
                    except RuntimeError as error:
                        receipt = {'state':'error', 'error':str(error),
                                   'handoff_receipt':getattr(error, 'handoff_receipt', None)}
                requests = [json.loads(row) for row in (folder/'requests.jsonl').read_text('utf8').splitlines()]
                counts = dict(Counter(request['path'] for request in requests))
                result = {'receipt':receipt, 'requests':requests, 'probe_counts':counts}
                results[name] = result
                return result

            for name, marker, expected in (
                ('marker_absent', None, 'not_running'),
                ('pid_absent', {key:value for key,value in owner.items() if key != 'pid'}, 'not_running'),
                ('pid_dead', {**owner, 'pid':finished.pid}, 'not_running'),
                ('pid_reused', {**owner, 'started_at':owner['started_at']-1000}, 'unowned_service_untouched'),
                ('executable_mismatch', {**owner, 'python':str(folder/'foreign/python.exe')}, 'unowned_service_untouched'),
                ('creation_stamp_absent', {key:value for key,value in owner.items() if key != 'started_at'}, 'unowned_service_untouched'),
                ('marker_endpoint_mismatch', {**owner, 'endpoint':'http://127.0.0.1:1'}, 'unowned_service_untouched'),
                ('marker_main_mismatch', {**owner, 'main':str(folder/'foreign/main.py')}, 'unowned_service_untouched'),
                ('marker_malformed', '{', 'unowned_service_untouched'),
                ('marker_non_object', [], 'unowned_service_untouched'),
            ):
                result = exercise(name, marker=marker)
                assert result['receipt']['state'] == expected, result
                assert result['receipt']['memory_state_verified'] and not result['requests'], result
                assert not result['receipt']['local_service_touched'] and result['receipt']['free_requests'] == 0
            for name, config in (('http_argv_mismatch', {'foreign_argv':True}),
                                 ('owner_changed_after_http', {'replace_owner':True})):
                result = exercise(name, config=config)
                assert result['receipt']['state'] == 'unowned_service_untouched', result
                assert result['probe_counts'] == {'/system_stats':1}, result
            busy = exercise('busy_queue', config={'busy':True})
            assert busy['receipt']['state'] == 'error' and 'still generating' in busy['receipt']['error'], busy
            assert busy['probe_counts'] == {'/system_stats':1, '/queue':1}, busy
            live = exercise('live_owned')
            assert live['receipt']['state'] == 'idle_models_released', live
            assert live['receipt']['memory_state_verified'] and live['receipt']['gpu_unload_verified'], live
            assert live['probe_counts'] == {'/system_stats':2, '/queue':1, '/free':1}, live
            assert live['requests'][2]['payload'] == {'unload_models':True, 'free_memory':False}, live
            blocked = exercise('absent_pid_budget_blocked', marker={**owner, 'pid':finished.pid},
                               snapshot={**healthy, 'available_commit_bytes':0})
            assert blocked['receipt']['state'] == 'error' and not blocked['requests'], blocked
            assert blocked['receipt']['handoff_receipt']['state'] == 'not_running', blocked
            assert not blocked['receipt']['handoff_receipt']['memory_state_verified'], blocked
            for name, selected, expected in (
                ('external_endpoint', {'options':{'provider':'qwen_image_2_1_fast',
                                                  'api_base':'https://external.invalid'}}, 'external_service_untouched'),
                ('other_provider', {'options':{'provider':'manual_import'}}, 'not_managed_qwen')):
                result = exercise(name, selected_run=selected)
                assert result['receipt']['state'] == expected and not result['requests'], result
            result = exercise('unconfigured_deployment', configured=False)
            assert result['receipt']['state'] == 'not_running' and not result['requests'], result
            # Discovery remains usable for callers that intentionally discover a service.
            (folder/'server-process.json').unlink(missing_ok=True)
            assert qwen.service_status(endpoint)['system']['argv'] == [owner['main']]

            comparison = None
            if args.baseline:
                spec = importlib.util.spec_from_file_location('qwen_release_baseline', args.baseline)
                baseline = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(baseline)
                baseline_live = exercise('baseline_live_owned', module=baseline)
                assert baseline_live['probe_counts'] == live['probe_counts']
                assert baseline_live['receipt']['requested_flags'] == live['receipt']['requested_flags']
                delayed = {'status_error':True, 'status_delay':args.negative_delay}
                before = [exercise(f'baseline_negative_hook_{index}', module=baseline,
                                   marker={**owner, 'pid':finished.pid}, config=delayed) for index in range(2)]
                after = [exercise(f'candidate_negative_hook_{index}', marker={**owner, 'pid':finished.pid},
                                  config=delayed) for index in range(2)]
                assert all(item['receipt']['state'] == 'not_running' for item in before+after)
                assert all(item['probe_counts'] == {'/system_stats':1} for item in before)
                assert all(not item['requests'] for item in after)
                comparison = {'hooks':2, 'synthetic_negative_http_delay_seconds':args.negative_delay,
                              'before_seconds':sum(item['receipt']['seconds'] for item in before),
                              'after_seconds':sum(item['receipt']['seconds'] for item in after),
                              'before_system_stats_probes':2, 'after_system_stats_probes':0,
                              'before_free_requests':0, 'after_free_requests':0}
            report = {'ok':True, 'real_local_http_and_process_wrappers':True, 'models_loaded':False,
                      'cases':results, 'negative_hook_pair_comparison':comparison}
            if args.receipt:
                args.receipt.parent.mkdir(parents=True, exist_ok=True)
                args.receipt.write_text(json.dumps(report, indent=2)+'\n', 'utf8')
            print(json.dumps({'ok':True, 'cases':len(results), 'real_local_http_and_process_wrappers':True,
                              'negative_hook_pair_comparison':comparison, 'receipt':str(args.receipt) if args.receipt else None}))
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            process.stderr.close()


if __name__ == '__main__':
    main()
