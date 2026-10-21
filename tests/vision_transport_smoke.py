"""Offline checks for protocol fallback and recognition provenance."""
from __future__ import annotations

import tempfile
import copy
import io
import json
import urllib.error
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import vision_transport as vision


class MemoryResponse(io.BytesIO):
    def __init__(self, value, sse=False):
        super().__init__(value if isinstance(value, bytes) else json.dumps(value).encode())
        self.headers = Message()
        self.headers['Content-Type'] = 'text/event-stream' if sse else 'application/json'


def timing_cases(root, source, config):
    """Distinguish real I/O boundaries with known delays, without network or sleeps."""
    passed = []
    original_dumps, original_loads = json.dumps, json.loads
    original_input, original_base64, original_profile = vision._input_image, vision.base64.b64encode, vision._profile_json
    for name, response_value, sse, expected, error_type in [
        ('timed-json', {'status': 'completed', 'output_text': '{"ok":true}'}, False, 62, None),
        ('timed-sse', b'data: {"type":"response.output_text.delta","delta":"ignored"}\n'
                      b'data: {"type":"response.completed","response":{"status":"completed","output_text":"{\\"ok\\":true}"}}\n', True, 71, None),
        ('timed-invalid-profile', {'status': 'completed', 'output_text': '{"broken":'}, False, 62, vision.VisionResponseError),
        ('timed-http', 'http', False, 41, vision.VisionHTTPError),
        ('timed-timeout', 'timeout', False, 32, TimeoutError),
        ('timed-read-failure', 'read-failure', False, 39, OSError),
        ('timed-invalid-envelope', b'{', False, 41, json.JSONDecodeError),
    ]:
        clock = [0.0]
        def advance(seconds):
            clock[0] += seconds
        class TimedResponse(MemoryResponse):
            def read(self, *args):
                advance(7)
                if response_value == 'read-failure':
                    raise OSError('synthetic read failure')
                return super().read(*args)
            def __next__(self):
                advance(7)
                return super().__next__()
        response = TimedResponse(response_value if response_value not in ('http', 'timeout', 'read-failure') else b'error', sse)
        calls = []
        def opened(request, *, timeout):
            advance(5)
            calls.append(request)
            assert timeout == 180 and request.full_url.endswith('/responses')
            body = original_loads(request.data)
            assert body['model'] == 'gpt-6.1-sol' and body['stream'] is True and body['store'] is False
            assert request.get_header('Authorization') == 'Bearer local-secret'
            if response_value == 'http':
                raise urllib.error.HTTPError(request.full_url, 502, 'synthetic gateway failure', {}, response)
            if response_value == 'timeout':
                raise TimeoutError('synthetic timed out')
            return response
        def encode(value, *args, **kwargs):
            if isinstance(value, dict) and ('input' in value or 'messages' in value):
                advance(3)
            return original_dumps(value, *args, **kwargs)
        def parse(*args, **kwargs):
            advance(2)
            return original_loads(*args, **kwargs)
        def prepare(*args, **kwargs):
            advance(11)
            return original_input(*args, **kwargs)
        def base64_encode(*args, **kwargs):
            advance(13)
            return original_base64(*args, **kwargs)
        def profile(*args, **kwargs):
            advance(19)
            return original_profile(*args, **kwargs)
        directory = root/name
        error = None
        with patch.object(vision, 'perf_counter', side_effect=lambda: clock[0]), \
             patch.object(vision, '_connection', return_value=(config, 'http://localhost/v1', 'local-secret', None)), \
             patch.object(vision, '_input_image', side_effect=prepare), \
             patch.object(vision.base64, 'b64encode', side_effect=base64_encode), \
             patch.object(vision, '_profile_json', side_effect=profile), \
             patch.object(vision.json, 'dumps', side_effect=encode), \
             patch.object(vision.json, 'loads', side_effect=parse), \
             patch.object(vision.urllib.request, 'urlopen', side_effect=opened):
            try:
                payload, provenance = vision.request_vision_json(source, directory, 'Timing board prompt', vision_model='gpt-6.1-sol')
            except Exception as exc:
                assert error_type and isinstance(exc, error_type), (name, exc)
                error = {'type': type(exc).__name__, 'message': str(exc), **vision.error_metadata(exc)}
                timings = error['timings']
            else:
                assert error_type is None and payload == {'ok': True}, name
                timings = provenance['timings']
        assert len(calls) == 1 and timings['total_seconds'] == expected, (name, timings)
        assert timings['image_prepare_seconds'] == 11 and timings['image_base64_seconds'] == 13
        request_timings, = timings['http_requests']
        assert request_timings['request_encode_seconds'] == 3 and request_timings['wait_headers_seconds'] == 5
        assert request_timings['request_bytes'] == len(calls[0].data)
        assert request_timings['response_read_seconds'] == (14 if sse else 0 if response_value == 'timeout' else 7)
        assert request_timings['response_parse_seconds'] == (4 if sse else 0 if response_value in ('http', 'timeout', 'read-failure') else 2)
        assert timings['profile_parse_seconds'] == (21 if error_type in (None, vision.VisionResponseError) else 0)
        if error:
            expected_policy = 'one_retry_after_completed_response_invalid_json' if error_type is vision.VisionResponseError else \
                'one_retry_after_explicit_http_429_or_5xx' if error_type is vision.VisionHTTPError else None
            assert vision.retry_policy_for_error(error) == expected_policy
            if error_type is vision.VisionResponseError:
                receipt = json.loads((directory/'response-failure.json').read_text())
                assert receipt['timings'] == timings and timings['failed_stage'] == 'profile_parse_seconds'
            else:
                assert timings['failed_stage'] == 'http_request'
        else:
            assert 'failed_stage' not in timings and 'failed_stage' not in request_timings
        assert 'local-secret' not in str(timings) and 'data:image/' not in str(timings)
        passed.append(name+'_phase_receipt_and_retry_preserved')
    return passed


def failure_cases(root, source, config):
    passed = []
    def execute(name, response, *, sse=False, chat=False):
        directory = root/name
        calls = []
        def opened(*args, **kwargs):
            calls.append(args[0])
            if isinstance(response, Exception):
                raise response
            return MemoryResponse(response, sse)
        try:
            with patch.object(vision, '_connection', return_value=(config, 'http://localhost/v1', 'local-secret', '/chat/completions' if chat else None)), \
                 patch.object(vision.urllib.request, 'urlopen', side_effect=opened):
                result = vision.request_vision_json(source, directory, 'Bound board prompt', vision_model='gpt-6.1-sol')
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc), **vision.error_metadata(exc, None)}
            return error, exc, calls, directory
        return result, None, calls, directory

    qualified = []
    for name, payload, sse, chat, policy in [
            ('sse-failed-http2', b'data: {"type":"response.created","response":{"id":"early-failed-id","status":"in_progress"}}\n'
                b'data: '+json.dumps({'type': 'response.failed', 'response': {'status': 'failed', 'error': {
                'type': 'server_error', 'code': 'upstream_http2_stream_error', 'message': 'local-secret data:image/jpeg;base64,AAAA'}}}).encode()+b'\n', True, False, 'one_retry_after_explicit_provider_upstream_error'),
            ('sse-error-upstream', b'data: '+json.dumps({'type': 'error', 'error': {'type': 'server_error', 'code': 'upstream_error', 'message': 'failed'}}).encode()+b'\n', True, False, 'one_retry_after_explicit_provider_upstream_error'),
            ('http200-error', {'error': {'type': 'server_error', 'code': 'upstream_error', 'message': 'local-secret', 'authorization': 'Bearer local-secret', 'echo': 'data:image/png;base64,BBBB', 'local-secret': 'echoed key name', 'b64_json': 'AAAA-secret-image'}}, False, False, 'one_retry_after_explicit_provider_upstream_error'),
            ('responses-completed-bad-json', {'status': 'completed', 'output_text': '{"broken":'}, False, False, 'one_retry_after_completed_response_invalid_json'),
            ('chat-stop-bad-json', {'choices': [{'finish_reason': 'stop', 'message': {'content': '```json\n{"broken":\n```'}}]}, False, True, 'one_retry_after_completed_response_invalid_json')]:
        error, exc, calls, directory = execute(name, payload, sse=sse, chat=chat)
        assert isinstance(exc, vision.VisionResponseError), (name, error)
        assert len(calls) == 1, 'Transport must not hide a retry'
        http_attempt = directory/'http-attempts/attempt-0001'
        http_binding = json.loads((http_attempt/'request.json').read_text(encoding='utf-8'))
        observed = vision.read_terminal_response(http_attempt, source=source, prompt='Bound board prompt',
            expected_request_binding_id=http_binding['request_binding_id'])
        assert observed['response']['outcome'] == ('completed' if 'bad-json' in name else 'failed')
        assert name != 'sse-failed-http2' or observed['response']['response_id'] == 'early-failed-id'
        assert vision.retry_policy_for_error(error) == policy, (name, error)
        proof = vision.validate_saved_failure(error, source, 'Bound board prompt')
        assert len(proof) == 4 and all(Path(row['path']).is_relative_to(directory) for row in proof)
        for file in directory.rglob('*.json'):
            text = file.read_text()
            assert 'local-secret' not in text and 'data:image/' not in text and 'base64,' not in text and 'AAAA-secret-image' not in text
        assert 'local-secret' not in str(error) and 'data:image/' not in str(error)
        qualified.append((error, directory))
        passed.append(name+'_saved_bound_single_request')
        try:
            vision.validate_saved_failure(error, source, 'Different prompt')
        except ValueError:
            passed.append(name+'_changed_prompt_rejected')
        else:
            raise AssertionError('Wrong prompt qualified')
        files_before = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        with patch.object(vision, '_connection', side_effect=AssertionError('No connection on existing evidence')):
            try:
                vision.request_vision_json(source, directory, 'Bound board prompt', vision_model='gpt-6.1-sol')
            except FileExistsError:
                pass
            else:
                raise AssertionError('Attempt evidence overwritten')
        assert all(Path(p).read_bytes() == data for p, data in files_before.items())
    alias = root/'same-board-different-owner.png'
    alias.write_bytes(source.read_bytes())
    assert vision.validate_saved_failure(qualified[0][0], alias, 'Bound board prompt')
    passed.append('same_byte_rebind_uses_original_request_ownership')

    for name, response, sse, chat in [
            ('incomplete', b'data: {"type":"response.incomplete","response":{"status":"incomplete"}}\n', True, False),
            ('contradictory-incomplete', b'data: {"type":"response.incomplete","response":{"id":"contradictory-incomplete",'
                b'"status":"completed","output_text":"{\\"ok\\":true}"}}\n', True, False),
            ('stream-ended', b'data: {"type":"response.output_text.delta","delta":"partial"}\n', True, False),
            ('timeout', TimeoutError('timed out'), False, False),
            ('disconnect', urllib.error.URLError('disconnected'), False, False),
            ('unknown-provider-error', {'error': {'type': 'server_error', 'code': 'unknown_error'}}, False, False),
            ('wrong-provider-type', {'error': {'type': 'authentication_error', 'code': 'upstream_error'}}, False, False),
            ('failed-string-only', {'status': 'failed', 'error': "{'type':'server_error','code':'upstream_error'}"}, False, False),
            ('completed-contradiction', {'status': 'completed', 'error': {'type': 'server_error', 'code': 'upstream_error'}}, False, False),
            ('unfinished-bad-json', {'status': 'incomplete', 'output_text': '{"broken":'}, False, False),
            ('chat-length-bad-json', {'choices': [{'finish_reason': 'length', 'message': {'content': '{"broken":'}}]}, False, True),
            ('completed-nonobject', {'status': 'completed', 'output_text': '[1,2]'}, False, False),
            ('completed-no-text', {'status': 'completed', 'output': []}, False, False)]:
        error, exc, calls, directory = execute(name, response, sse=sse, chat=chat)
        assert exc is not None and len(calls) == 1, (name, error)
        assert vision.retry_policy_for_error(error) is None, (name, error)
        if name == 'contradictory-incomplete':
            assert isinstance(exc, RuntimeError) and str(exc) == 'vision response incomplete'
            receipt = json.loads((directory/'http-attempts/attempt-0001/response.json').read_text(encoding='utf-8'))
            assert receipt['event_type'] == 'response.incomplete' and receipt['outcome'] == 'unknown'
            saved = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
            with patch.object(vision, '_connection', side_effect=AssertionError('No replay discovery')), \
                 patch.object(vision.urllib.request, 'urlopen', side_effect=AssertionError('No replay HTTP')):
                try:
                    vision.replay_vision_json(source, directory, 'Bound board prompt',
                        expected_connection_identity=vision._selected_connection_identity(None, 'gpt-6.1-sol', None, None, None),
                        vision_model='gpt-6.1-sol')
                except ValueError:
                    pass
                else:
                    raise AssertionError('Contradictory incomplete frame replayed as completed')
            assert all(Path(p).read_bytes() == data for p, data in saved.items())
        passed.append(name+'_no_retry')
    valid, exc, calls, _ = execute('valid-unexpected-schema', {'status': 'completed', 'output_text': '{"unexpected":1}'})
    assert exc is None and valid[0] == {'unexpected': 1} and len(calls) == 1
    passed.append('valid_object_schema_remains_consumer_owned')
    for status in (429, 500, 503):
        exc = vision.VisionHTTPError(status, 'observed failure')
        error = {'type': type(exc).__name__, 'message': str(exc), **vision.error_metadata(exc, None)}
        assert vision.retry_policy_for_error(error) == 'one_retry_after_explicit_http_429_or_5xx'
        assert vision.validate_saved_failure(error, source, 'Bound board prompt') == []
    passed.append('HTTP_retry_ABI_unchanged')
    legacy = {'type': 'RuntimeError', 'message': "vision response failed: {'type':'server_error','code':'upstream_error'}", 'http_status': None}
    assert vision.retry_policy_for_error(legacy) is None
    try:
        vision.validate_saved_failure(legacy, source, 'Bound board prompt')
    except ValueError:
        passed.append('historical_RuntimeError_summary_never_grants_retry')
    else:
        raise AssertionError('Historical summary acquired new retry authority')

    error, directory = qualified[0]
    for label, file in [('payload', directory/'response-failure-payload.json'),
                        ('JPEG', directory/'inputs/vision-input.jpg'),
                        ('binding', directory/'request-binding.json')]:
        old = file.read_bytes()
        file.write_bytes(old+b'changed')
        try:
            vision.validate_saved_failure(error, source, 'Bound board prompt')
        except ValueError:
            passed.append(label+'_tamper_rejected')
        else:
            raise AssertionError(label+' tamper accepted')
        finally:
            file.write_bytes(old)
    forged = copy.deepcopy(error)
    forged['failure_kind'] = 'completed_invalid_json'
    assert vision.retry_policy_for_error(forged) is None
    passed.append('classification_string_cannot_relabel_terminal_payload')
    receipt_path = Path(error['failure_evidence']['path'])
    payload_path = directory/'response-failure-payload.json'
    old_receipt, old_payload = receipt_path.read_bytes(), payload_path.read_bytes()
    changed_payload = json.loads(old_payload)
    changed_payload['response']['error']['code'] = 'unqualified_error'
    payload_path.write_text(json.dumps(changed_payload))
    changed_receipt = json.loads(old_receipt)
    changed_receipt['response_payload'] = vision._record(payload_path)
    receipt_path.write_text(json.dumps(changed_receipt))
    forged = {**error, 'failure_evidence': vision._record(receipt_path)}
    assert vision.retry_policy_for_error(forged) is None
    passed.append('rehashed_unknown_server_code_still_rejected')
    payload_path.write_bytes(old_payload); receipt_path.write_bytes(old_receipt)
    changed_payload = json.loads(old_payload)
    changed_payload['type'] = 'response.incomplete'
    payload_path.write_text(json.dumps(changed_payload))
    changed_receipt = json.loads(old_receipt)
    changed_receipt['response_payload'] = vision._record(payload_path)
    receipt_path.write_text(json.dumps(changed_receipt))
    forged = {**error, 'failure_evidence': vision._record(receipt_path)}
    assert vision.retry_policy_for_error(forged) is None
    passed.append('rehashed_terminal_event_disagrees_with_actual_payload_rejected')
    payload_path.write_bytes(old_payload); receipt_path.write_bytes(old_receipt)
    for original_error, kind in ((qualified[0][0], 'failed'), (qualified[3][0], 'completed-invalid')):
        path = Path(original_error['failure_evidence']['path'])
        receipt_bytes = path.read_bytes()
        receipt = json.loads(receipt_bytes)
        response_path = Path(receipt['response_payload']['path'])
        payload_bytes = response_path.read_bytes()
        payload = json.loads(payload_bytes)
        if kind == 'failed':
            payload['response']['status'] = 'incomplete'
        else:
            payload['output_text'] = '{"valid_object":true}'
        response_path.write_text(json.dumps(payload))
        receipt['response_payload'] = vision._record(response_path)
        path.write_text(json.dumps(receipt))
        current = {**original_error, 'failure_evidence': vision._record(path)}
        assert vision.retry_policy_for_error(current) is None
        passed.append(kind+'_rehashed_payload_cannot_claim_retry')
        response_path.write_bytes(payload_bytes); path.write_bytes(receipt_bytes)

    path = Path(error['failure_evidence']['path'])
    receipt_bytes = path.read_bytes(); receipt = json.loads(receipt_bytes)
    binding_path = Path(receipt['request_binding']['path'])
    binding_bytes = binding_path.read_bytes(); binding = json.loads(binding_bytes)
    image_path = Path(binding['api_input']['path']); image_bytes = image_path.read_bytes()
    Image.new('RGB', (64, 96), (150, 70, 100)).save(image_path, 'JPEG')
    binding['api_input'].update(vision._record(image_path))
    binding_path.write_text(json.dumps(binding)); receipt['request_binding'] = vision._record(binding_path)
    path.write_text(json.dumps(receipt)); current = {**error, 'failure_evidence': vision._record(path)}
    assert vision.retry_policy_for_error(current) == 'one_retry_after_explicit_provider_upstream_error'
    try:
        vision.validate_saved_failure(current, source, 'Bound board prompt')
    except ValueError as exc:
        assert 'does not depict' in str(exc), str(exc)
        passed.append('rehashed_other_JPEG_requires_actual_board_derivation')
    else:
        raise AssertionError('Rehashed unrelated API pixels accepted')
    image_path.write_bytes(image_bytes); binding_path.write_bytes(binding_bytes); path.write_bytes(receipt_bytes)
    before = source.read_bytes()
    source.write_bytes(before+b'changed')
    try:
        vision.validate_saved_failure(error, source, 'Bound board prompt')
    except ValueError:
        passed.append('actual_source_changed_rejected')
    else:
        raise AssertionError('Changed source accepted')
    finally:
        source.write_bytes(before)
    return passed


def observation_cases(root, source, config):
    """Real loopback headers/SSE, with the client killed before any finally block."""
    passed, requests = [], []
    release = threading.Event()
    profile_marker = root/'profile-entered'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            requests.append((self.path, body))
            mode = self.path.split('/')[1]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json' if mode == 'json-invalid' else 'text/event-stream')
            self.send_header('x-request-id', 'req-local')
            self.send_header('retry-after', '2')
            self.send_header('Set-Cookie', 'local-secret')
            self.send_header('Authorization', 'Bearer local-secret')
            self.end_headers(); self.wfile.flush()
            if mode == 'headers':
                release.wait(15)
                return
            if mode == 'progress':
                for kind in ('response.created', 'response.in_progress'):
                    payload = {'type': kind, 'response': {'id': 'resp-local', 'status': 'in_progress'}}
                    self.wfile.write(b'data: '+json.dumps(payload).encode()+b'\n\n'); self.wfile.flush()
                release.wait(15)
                return
            payload = {'id': 'resp-local', 'model': 'gpt-6.1-sol', 'status': 'completed',
                'output_text': '{"broken":' if mode == 'json-invalid' else '{"ok":true}',
                'authorization': 'Bearer local-secret', 'echo': 'data:image/png;base64,AAAA'}
            if mode == 'json-invalid':
                self.wfile.write(json.dumps(payload).encode())
            else:
                self.wfile.write(b'data: '+json.dumps({'type': 'response.completed', 'response': payload}).encode()+b'\n\n')
            self.wfile.flush()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True); serving.start()
    endpoint = 'http://127.0.0.1:'+str(server.server_port)
    client = '''import json,sys,threading
from pathlib import Path
from unittest.mock import patch
import vision_transport as vision
source,directory,endpoint,marker,mode=sys.argv[1:]
if mode=='completed':
    original=vision._profile_json
    def blocked(text):
        Path(marker).write_text('profile called after transport completed')
        threading.Event().wait(30)
        return original(text)
    vision._profile_json=blocked
with patch.object(vision,'_connection',return_value=({'registry_id':'loopback'},endpoint,'local-secret',None)):
    vision.request_vision_json(Path(source),Path(directory),'Bound observation prompt',vision_model='gpt-6.1-sol')
'''
    completed_directory = None
    try:
        for mode in ('headers', 'progress', 'completed', 'json-invalid'):
            release.clear()
            directory = root/('local-'+mode)
            process = subprocess.Popen([sys.executable, '-B', '-c', client, str(source), str(directory),
                endpoint+'/'+mode, str(profile_marker), mode], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            attempt = directory/'http-attempts/attempt-0001'
            if mode == 'json-invalid':
                stdout, stderr = process.communicate(timeout=15)
                assert process.returncode != 0 and b'VisionResponseError' in stderr, (stdout, stderr)
            else:
                try:
                    deadline = time.monotonic()+15
                    while time.monotonic() < deadline:
                        if mode == 'completed' and profile_marker.exists():
                            break
                        observed = attempt/'observation.json'
                        if observed.is_file():
                            try:
                                current = json.loads(observed.read_text(encoding='utf-8'))
                            except (FileNotFoundError, PermissionError):
                                # Windows may briefly deny reads during atomic replacement.
                                current = None
                            if current is not None and (
                                    mode == 'headers' and current['stage'] == 'headers_received'
                                    or mode == 'progress' and current.get('event_type') == 'response.in_progress'):
                                break
                        assert process.poll() is None, process.communicate()
                        threading.Event().wait(.01)
                    else:
                        raise AssertionError('Client did not persist observation before blocking: '+mode)
                finally:
                    process.kill(); process.communicate(timeout=5)
                    release.set()
            request = json.loads((attempt/'request.json').read_text(encoding='utf-8'))
            observed = json.loads((attempt/'observation.json').read_text(encoding='utf-8'))
            assert observed['headers'] == {'x-request-id': 'req-local', 'retry-after': '2',
                'content-type': 'application/json' if mode == 'json-invalid' else 'text/event-stream'}
            assert request['request_body_sha256'] == vision.hashlib.sha256(requests[-1][1]).hexdigest()
            assert request['model'] == 'gpt-6.1-sol' and request['profile'] == 'json_object_v1'
            assert request['api_input']['sha256'] == vision.reference_provider.sha256(Path(request['api_input']['path']))
            if mode in {'headers', 'progress'}:
                assert observed['outcome'] == 'unknown' and not (attempt/'response.json').exists()
                assert mode != 'progress' or observed['response_id'] == 'resp-local'
                try:
                    vision.read_terminal_response(attempt, source=source, prompt='Bound observation prompt',
                        expected_request_binding_id=request['request_binding_id'])
                except FileNotFoundError:
                    pass
                else:
                    raise AssertionError('An interrupted stream became terminal')
                saved = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
                with patch.object(vision, '_connection', side_effect=AssertionError('No connection on interrupted invocation')):
                    try:
                        vision.request_vision_json(source, directory, 'Bound observation prompt', vision_model='gpt-6.1-sol')
                    except FileExistsError:
                        pass
                    else:
                        raise AssertionError('Interrupted invocation was overwritten')
                assert all(Path(p).read_bytes() == data for p, data in saved.items())
            else:
                with patch.object(vision.urllib.request, 'urlopen', side_effect=AssertionError('Reader must not request')):
                    before = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
                    verified = vision.read_terminal_response(attempt, source=source, prompt='Bound observation prompt',
                        expected_request_binding_id=request['request_binding_id'])
                assert verified['response']['outcome'] == 'completed' and not verified['retry_authorized']
                assert not verified['human_approved'] and all(Path(p).read_bytes() == data for p, data in before.items())
                assert mode != 'json-invalid' or (directory/'response-failure.json').is_file()
                completed_directory = attempt
            before = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
            with patch.object(vision, '_connection', side_effect=AssertionError('Replay must not resolve credentials')), \
                 patch.object(vision.urllib.request, 'urlopen', side_effect=AssertionError('Replay must not request')):
                try:
                    replayed, provenance = vision.replay_vision_json(source, directory, 'Bound observation prompt',
                        expected_connection_identity=vision._selected_connection_identity(None, 'gpt-6.1-sol', None, None, None),
                        vision_model='gpt-6.1-sol')
                except (ValueError, FileNotFoundError, RuntimeError):
                    assert mode != 'completed', mode
                else:
                    assert mode == 'completed' and replayed == {'ok': True}
                    assert provenance['response_id'] == 'resp-local' and provenance['network_used'] is True
                    assert provenance['response_replay']['new_http_requests'] == 0
                    assert provenance['response_replay']['new_model_requests'] == 0
            assert all(Path(p).read_bytes() == data for p, data in before.items())
            for p in directory.rglob('*.json'):
                content = p.read_text(encoding='utf-8')
                assert 'local-secret' not in content and 'data:image/' not in content and 'base64,' not in content
            passed.append('loopback_'+mode+'_persisted_before_hard_exit_or_profile_failure')

        request_path = completed_directory/'request.json'
        response_path = completed_directory/'response.json'
        original_request = request_path.read_bytes(); original_response = response_path.read_bytes()
        binding = json.loads(original_request)['request_binding_id']
        def reader(prompt='Bound observation prompt'):
            return vision.read_terminal_response(completed_directory, source=source, prompt=prompt,
                expected_request_binding_id=binding)
        for field in ('model', 'profile', 'producer', 'api_input', 'prompt_sha256', 'endpoint_sha256'):
            changed = json.loads(original_request); changed[field] = 'tampered'
            changed['request_binding_id'] = vision._observation_identity(
                {k: v for k, v in changed.items() if k != 'request_binding_id'})
            request_path.write_text(json.dumps(changed), encoding='utf-8')
            try:
                reader()
            except ValueError:
                pass
            else:
                raise AssertionError('Changed request binding accepted: '+field)
            request_path.write_bytes(original_request)
        changed = json.loads(original_response); changed['payload']['output_text'] = '{"different":true}'
        response_path.write_text(json.dumps(changed), encoding='utf-8')
        try:
            reader()
        except ValueError:
            pass
        else:
            raise AssertionError('Changed terminal payload accepted')
        response_path.write_bytes(original_response)
        try:
            reader('Changed prompt')
        except ValueError:
            pass
        else:
            raise AssertionError('Changed expected prompt accepted')
        passed.append('reader_checks_independent_binding_payload_and_expected_prompt')
        def discovered(request, *, timeout):
            if request.full_url.endswith('/models'):
                assert request.data is None
                return MemoryResponse({'data': [{'id': 'gpt-6.1-sol'}]})
            return MemoryResponse({'id': 'after-discovery', 'model': 'gpt-6.1-sol',
                'status': 'completed', 'output_text': '{"normal":true}'})
        directory = root/'discovery'
        with patch.object(vision, '_connection', return_value=(config, 'http://localhost/v1', 'local-secret', None)), \
             patch.dict(vision.os.environ, {}, clear=True), \
             patch.object(vision.urllib.request, 'urlopen', side_effect=discovered):
            payload, provenance = vision.request_vision_json(source, directory, 'Bound observation prompt')
        assert payload == {'normal': True} and provenance['response_id'] == 'after-discovery'
        discovery, actual = sorted((directory/'http-attempts').iterdir())
        first = json.loads((discovery/'request.json').read_text(encoding='utf-8'))
        second = json.loads((actual/'request.json').read_text(encoding='utf-8'))
        assert first['protocol'] == 'models' and first['api_input'] is None and first['model'] is None
        assert second['protocol'] == 'responses' and second['model'] == 'gpt-6.1-sol'
        assert second['api_input']['source'] == {'path': str(source), 'sha256': first['source']['sha256']}
        assert second['request_binding_id'] != first['request_binding_id']
        verified = vision.read_terminal_response(actual, source=source, prompt='Bound observation prompt',
            expected_request_binding_id=second['request_binding_id'])
        assert verified['response']['payload']['output_text'] == '{"normal":true}'
        with patch.dict(vision.os.environ, {}, clear=True), \
             patch.object(vision, '_connection', side_effect=AssertionError('No discovery on replay')), \
             patch.object(vision.urllib.request, 'urlopen', side_effect=AssertionError('No HTTP on replay')):
            replayed, provenance = vision.replay_vision_json(source, directory, 'Bound observation prompt',
                expected_connection_identity=vision._selected_connection_identity(None, None, None, None, None))
        assert replayed == {'normal': True} and provenance['discovered_models'] == ['gpt-6.1-sol']
        assert provenance['response_replay']['new_model_requests'] == 0
        passed.append('discovery_and_inference_attempts_separately_bound_normal_payload_unchanged')
        return passed, len(requests)
    finally:
        release.set(); server.shutdown(); server.server_close(); serving.join(timeout=2)


def replay_cases(root, source, config):
    """RAW-shaped source metadata and independently supplied replay expectations."""
    directory = root/'raw-metadata'
    def raw_boundary(source, output, *, input_profile='standard'):
        assert input_profile == 'standard'
        decoded = vision.open_rgb(source)
        decoded.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        from raw_support import deterministic_srgb_profile
        path = output/'inputs/provider-input.jpg'; path.parent.mkdir(parents=True)
        decoded.save(path, 'JPEG', quality=95, subsampling=0, icc_profile=deterministic_srgb_profile())
        return path, {**vision._record(path), 'source': vision._record(source), 'format': 'JPEG',
            'size': list(decoded.size), 'derived': True, 'decode': {'decode_version': vision.RGB_DECODE_VERSION}}
    response = {'id': 'raw-profile', 'model': 'gpt-6.1-sol', 'status': 'completed', 'output_text': '{"normal":true}'}
    with patch.object(vision, '_connection', return_value=(config, 'http://localhost/v1', 'local-secret', None)), \
         patch.object(vision, '_input_image', side_effect=raw_boundary), \
         patch.object(vision.urllib.request, 'urlopen', return_value=MemoryResponse(response)):
        original, provenance = vision.request_vision_json(source, directory, 'RAW metadata prompt', vision_model='gpt-6.1-sol')
    expected = vision._selected_connection_identity(None, 'gpt-6.1-sol', None, None, None)
    def replay(connection=expected, prompt='RAW metadata prompt'):
        with patch.object(vision, 'is_raw', return_value=True), \
             patch.object(vision, '_replay_rgb', side_effect=vision.open_rgb), \
             patch.object(vision, '_connection', side_effect=AssertionError('No credential resolution')), \
             patch.object(vision.urllib.request, 'urlopen', side_effect=AssertionError('No HTTP')):
            return vision.replay_vision_json(source, directory, prompt,
                expected_connection_identity=connection, vision_model='gpt-6.1-sol')
    before = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    actual, replay_provenance = replay()
    assert actual == original == {'normal': True} and replay_provenance['response_id'] == provenance['response_id']
    assert all(Path(p).read_bytes() == data for p, data in before.items())
    for connection, prompt in [('0'*64, 'RAW metadata prompt'), (expected, 'Different requested profile')]:
        try:
            replay(connection, prompt)
        except ValueError:
            pass
        else:
            raise AssertionError('Journal identity replaced the caller expectation')
    with patch.object(vision, '_observation_producer', return_value={'vision_transport.py': '0'*64}):
        try:
            replay()
        except ValueError:
            pass
        else:
            raise AssertionError('Historical producer was silently re-signed')
    binding_path = directory/'request-binding.json'; original_binding = binding_path.read_bytes()
    image_path = Path(provenance['api_input']['path']); original_image = image_path.read_bytes()
    Image.new('RGB', (64, 96), 'red').save(image_path, 'JPEG')
    changed = json.loads(original_binding); changed['api_input'].update(vision._read_observation_record(image_path))
    binding_path.write_text(json.dumps(changed), encoding='utf-8')
    try:
        replay()
    except ValueError:
        pass
    else:
        raise AssertionError('A rehashed unrelated JPEG was treated as the current source')
    image_path.write_bytes(original_image); binding_path.write_bytes(original_binding)
    source_bytes = source.read_bytes(); source.write_bytes(source_bytes+b'changed identity')
    try:
        replay()
    except ValueError:
        pass
    else:
        raise AssertionError('Changed source replayed as the original')
    finally:
        source.write_bytes(source_bytes)
    from raw_support import deterministic_srgb_profile
    oriented = root/'profiled-orientation.png'
    raster = Image.new('RGB', (47, 61), (35, 78, 120)); exif = raster.getexif(); exif[274] = 6
    raster.save(oriented, icc_profile=deterministic_srgb_profile(), exif=exif)
    directory = root/'profiled-replay'
    with patch.object(vision, '_connection', return_value=(config, 'http://localhost/v1', 'local-secret', None)), \
         patch.object(vision.urllib.request, 'urlopen', return_value=MemoryResponse(response)):
        actual, provenance = vision.request_vision_json(oriented, directory, 'ICC orientation prompt', vision_model='gpt-6.1-sol')
    trace = root/'do-not-append-profile'; trace.write_text('frozen trace', encoding='utf-8')
    with patch.dict(vision.os.environ, {'COSPLAY_STAGE_PROFILE_PATH': str(trace)}), \
         patch.object(vision, '_connection', side_effect=AssertionError('No connection on replay')):
        replayed, _ = vision.replay_vision_json(oriented, directory, 'ICC orientation prompt',
            expected_connection_identity=expected, vision_model='gpt-6.1-sol')
    assert replayed == actual and trace.read_text(encoding='utf-8') == 'frozen trace'
    return ['raw_source_record_with_bytes_replays_without_writes',
        'ICC_EXIF_replay_matches_actual_API_JPEG_without_opt_in_profile_writes',
        'replay_rejects_changed_caller_connection_prompt_producer_source_and_rehashed_unrelated_JPEG']


def input_profile_cases(root, small, config):
    """Caller-owned board resolution changes pixels, never protocol or retry count."""
    import numpy as np
    import raw_support
    from types import SimpleNamespace
    from raw_support import deterministic_srgb_profile
    passed = []
    for invalid in (None, 'STANDARD', 2048, [], {}):
        destination = root/'invalid-profile'
        with patch.object(vision, '_connection', side_effect=AssertionError('Invalid profile resolved credentials')), \
             patch.object(vision.urllib.request, 'urlopen', side_effect=AssertionError('Invalid profile requested HTTP')):
            for action in (
                    lambda: vision.request_vision_json(small, destination, 'Profile prompt', input_profile=invalid),
                    lambda: vision.replay_vision_json(small, destination, 'Profile prompt',
                        expected_connection_identity='0'*64, input_profile=invalid),
                    lambda: vision._input_image(small, destination, input_profile=invalid),
                    lambda: vision.read_terminal_response(destination, source=small, prompt='Profile prompt',
                        expected_request_binding_id='0'*64, input_profile=invalid),
                    lambda: vision.validate_saved_failure({}, small, 'Profile prompt', input_profile=invalid)):
                try:
                    action()
                except ValueError as exc:
                    assert 'input profile' in str(exc)
                else:
                    raise AssertionError('Invalid profile accepted')
        assert not destination.exists()
    passed.append('invalid_input_profiles_rejected_before_credentials_HTTP_or_writes')

    source = root/'large-profiled.png'
    values = np.empty((1536, 2304, 3), np.uint8)
    values[:, :, 0] = np.arange(2304, dtype=np.uint16)[None, :] % 256
    values[:, :, 1] = np.arange(1536, dtype=np.uint16)[:, None] % 256
    values[:, :, 2] = 107
    raster = Image.fromarray(values); exif = raster.getexif(); exif[274] = 6
    raster.save(source, icc_profile=deterministic_srgb_profile(), exif=exif)
    default, _ = vision._input_image(source, root/'default-profile')
    standard, metadata = vision._input_image(source, root/'explicit-standard', input_profile='standard')
    expected_image = vision.open_rgb(source)
    expected_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    with io.BytesIO() as encoded:
        expected_image.save(encoded, 'JPEG', quality=93, subsampling=0, icc_profile=deterministic_srgb_profile())
        assert default.read_bytes() == standard.read_bytes() == encoded.getvalue()
    assert metadata['decode']['oriented_size'] == [1536, 2304]
    passed.append('default_standard_JPEG_exactly_preserves_1024_recipe_and_ICC_EXIF')

    expected_connection = vision._selected_connection_identity(None, 'gpt-6.1-sol', None, None, None)
    response = {'id': 'profile-response', 'model': 'gpt-6.1-sol', 'status': 'completed', 'output_text': '{"ok":true}'}
    for image, label in ((source, 'oriented-raster'), (small, 'small-same-pixels')):
        JPEGs = []
        for input_profile, edge in (('standard', 1024), ('detailed_board', 2048)):
            directory = root/(label+'-'+input_profile); calls = []
            def opened(request, **_kwargs):
                calls.append(request)
                return MemoryResponse(response)
            with patch.object(vision, '_connection', return_value=(config, 'http://localhost/v1', 'local-secret', None)), \
                 patch.object(vision.urllib.request, 'urlopen', side_effect=opened):
                payload, provenance = vision.request_vision_json(image, directory, 'Profile prompt',
                    vision_model='gpt-6.1-sol', input_profile=input_profile)
            assert len(calls) == 1 and len(provenance['timings']['http_requests']) == 1
            bound = json.loads((directory/'request-binding.json').read_text('utf-8'))
            http_path = directory/'http-attempts/attempt-0001'
            http = json.loads((http_path/'request.json').read_text('utf-8'))
            assert bound['input_profile'] == http['input_profile'] == provenance['input_profile'] == input_profile
            assert http['profile'] == 'json_object_v1' and http['request_body_sha256'] == vision.hashlib.sha256(calls[0].data).hexdigest()
            body = json.loads(calls[0].data)
            assert 'input_profile' not in body and set(body) == {'model', 'store', 'stream', 'input'}
            JPEG = Path(provenance['api_input']['path']).read_bytes(); JPEGs.append(JPEG)
            assert vision.base64.b64decode(body['input'][0]['content'][1]['image_url'].split(',', 1)[1]) == JPEG
            assert provenance['api_input']['decode']['long_edge_limit'] == edge
            assert max(provenance['api_input']['size']) == (edge if image == source else 96)
            before = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
            trace = root/'profile-replay-trace'; trace.write_text('unchanged', encoding='utf-8')
            with patch.dict(vision.os.environ, {'COSPLAY_STAGE_PROFILE_PATH': str(trace)}), \
                 patch.object(vision, '_connection', side_effect=AssertionError('Replay resolved credentials')), \
                 patch.object(vision.urllib.request, 'urlopen', side_effect=AssertionError('Replay requested HTTP')), \
                 patch.object(vision, '_exclusive_json', side_effect=AssertionError('Replay wrote JSON')), \
                 patch.object(vision, '_observation_json', side_effect=AssertionError('Replay wrote observation')):
                replayed, replay_provenance = vision.replay_vision_json(image, directory, 'Profile prompt',
                    expected_connection_identity=expected_connection, vision_model='gpt-6.1-sol', input_profile=input_profile)
                assert replayed == payload and replay_provenance['input_profile'] == input_profile
                assert replay_provenance['response_replay']['new_http_requests'] == 0
                wrong = 'standard' if input_profile == 'detailed_board' else 'detailed_board'
                for action in (
                        lambda: vision.replay_vision_json(image, directory, 'Profile prompt',
                            expected_connection_identity=expected_connection, vision_model='gpt-6.1-sol', input_profile=wrong),
                        lambda: vision.read_terminal_response(http_path, source=image, prompt='Profile prompt',
                            expected_request_binding_id=http['request_binding_id'], input_profile=wrong)):
                    try:
                        action()
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('Caller profile changed without invalidating journal')
            assert trace.read_text('utf-8') == 'unchanged'
            assert all(Path(p).read_bytes() == content for p, content in before.items())
            passed.append(label+'_'+input_profile+'_actual_body_replay_profile_binding_and_zero_writes')
        assert (JPEGs[0] == JPEGs[1]) == (image == small)

    raw_source = root/'profile-fixture.ARW'; raw_source.write_bytes(b'offline RAW decoder fixture')
    raw_values = np.asarray(values, np.uint16)*np.uint16(257)
    class Raw:
        sizes = SimpleNamespace(width=2304, height=1536, flip=0)
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def postprocess(self, **options):
            assert options == {'use_camera_wb': True, 'no_auto_bright': True, 'output_bps': 16,
                'gamma': (1, 1), 'output_color': 'sRGB', 'half_size': False}
            return raw_values
    fake_rawpy = SimpleNamespace(imread=lambda _path: Raw(), ColorSpace=SimpleNamespace(sRGB='sRGB'))
    for input_profile, edge in (('standard', 1024), ('detailed_board', 2048)):
        directory = root/('actual-raw-'+input_profile)
        with patch.object(vision.reference_provider, '_require_rawpy', return_value=fake_rawpy), \
             patch.object(raw_support, '_require_rawpy', return_value=fake_rawpy), \
             patch.object(vision, '_connection', return_value=(config, 'http://localhost/v1', 'local-secret', None)), \
             patch.object(vision.urllib.request, 'urlopen', return_value=MemoryResponse(response)):
            payload, provenance = vision.request_vision_json(raw_source, directory, 'RAW profile prompt',
                vision_model='gpt-6.1-sol', input_profile=input_profile)
        assert max(provenance['api_input']['size']) == edge and provenance['api_input']['decode']['long_edge_limit'] == edge
        before = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        with patch.object(vision.reference_provider, '_require_rawpy', return_value=fake_rawpy), \
             patch.object(vision, '_connection', side_effect=AssertionError('RAW replay resolved credentials')), \
             patch.object(vision.urllib.request, 'urlopen', side_effect=AssertionError('RAW replay requested HTTP')):
            replayed, _ = vision.replay_vision_json(raw_source, directory, 'RAW profile prompt',
                expected_connection_identity=expected_connection, vision_model='gpt-6.1-sol', input_profile=input_profile)
        assert replayed == payload and all(Path(p).read_bytes() == content for p, content in before.items())
        passed.append('RAW16_'+input_profile+'_real_prepare_and_readonly_recipe_match')

    failed = root/'detailed-profile-failure'
    with patch.object(vision, '_connection', return_value=(config, 'http://localhost/v1', 'local-secret', None)), \
         patch.object(vision.urllib.request, 'urlopen', return_value=MemoryResponse({'status': 'completed', 'output_text': '{"broken":'})):
        try:
            vision.request_vision_json(source, failed, 'Profile failure', vision_model='gpt-6.1-sol', input_profile='detailed_board')
        except vision.VisionResponseError as exc:
            error = {'type': type(exc).__name__, 'message': str(exc), **vision.error_metadata(exc)}
        else:
            raise AssertionError('Invalid profile JSON accepted')
    assert len(vision.validate_saved_failure(error, source, 'Profile failure', input_profile='detailed_board')) == 4
    try:
        vision.validate_saved_failure(error, source, 'Profile failure')
    except ValueError:
        pass
    else:
        raise AssertionError('Detailed failure accepted by standard caller')
    passed.append('detailed_failure_exact_profile_validation_preserves_retry_policy')
    return passed


def main():
    with tempfile.TemporaryDirectory(prefix="cosplay_vision_") as temporary:
        root = Path(temporary)
        source = root / "source.png"
        Image.new("RGB", (64, 96), (50, 70, 100)).save(source)
        requests = []
        def local_response(request, *, timeout):
            requests.append(request.full_url)
            assert request.get_header('Authorization') == 'Bearer local-secret'
            body = json.loads(request.data)
            if request.full_url.endswith("/responses"):
                assert body["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
                raise urllib.error.HTTPError(request.full_url, 404, 'unknown endpoint', {}, io.BytesIO(b'unknown endpoint'))
            return MemoryResponse({"id": "offline-response", "model": "gpt-5.4", "choices": [{"message": {
                "content": '```json\n{"character_name":"unknown","series":"unknown"}\n```'
            }, 'finish_reason': 'stop'}]})
        config = {"registry_id": "offline", "kind": "openai-compatible"}
        with patch.object(vision, "_connection", return_value=(config, "http://localhost/v1", "local-secret", None)), \
             patch.object(vision.urllib.request, "urlopen", side_effect=local_response):
            payload, provenance = vision.request_vision_json(source, root / "probe", "Identify supported character", vision_model="gpt-5.4")
            assert payload["character_name"] == "unknown"
            assert provenance["protocol"] == "chat/completions" and provenance["fallback"]["http_status"] == 404
            assert len(requests) == 2
            attempts = sorted((root/'probe/http-attempts').iterdir())
            assert len(attempts) == 2
            bindings = [json.loads((p/'request.json').read_text(encoding='utf-8')) for p in attempts]
            assert [v['protocol'] for v in bindings] == ['responses', 'chat/completions']
            assert bindings[0]['request_binding_id'] != bindings[1]['request_binding_id']
            first, second = [json.loads((p/'response.json').read_text(encoding='utf-8')) for p in attempts]
            assert first['outcome'] == 'http_error' and first['http_status'] == 404
            assert second['outcome'] == 'completed' and second['response_id'] == 'offline-response'
            with patch.object(vision, '_connection', side_effect=AssertionError('Replay must not connect')):
                replayed, replay_provenance = vision.replay_vision_json(source, root/'probe', 'Identify supported character',
                    expected_connection_identity=vision._selected_connection_identity(None, 'gpt-5.4', None, None, None),
                    vision_model='gpt-5.4')
            assert replayed == payload and replay_provenance['fallback'] == provenance['fallback']
            assert provenance["source"]["sha256"] == vision.reference_provider.sha256(source)
            assert provenance["input_sha256"] == vision.reference_provider.sha256(Path(provenance["api_input"]["path"]))
            assert "local-secret" not in str(provenance) and "data:image/" not in str(provenance) and "base64," not in str(provenance)
            try:
                vision.request_vision_json(source, root / "probe", "Identify", vision_model="gpt-image-2.5-sunburst")
            except ValueError:
                pass
            else:
                raise AssertionError("Image generation model must be rejected before a request")
            assert len(requests) == 2
        assert vision._vision_models({"data": [{"id": "gpt-image-1"}, {"id": "gpt-6.1-sol"}]}) == ["gpt-6.1-sol"]
        passed = failure_cases(root, source, config)
        passed.extend(timing_cases(root, source, config))
        observations, loopback_count = observation_cases(root, source, config)
        passed.extend(observations)
        passed.extend(replay_cases(root, source, config))
        passed.extend(input_profile_cases(root, source, config))
    print(json.dumps({'schema': 'vision_transport_smoke.v2', 'status': 'passed',
        'cases_passed': len(passed)+1, 'cases': ['original_protocol_model_and_provenance_contract', *passed],
        'external_network_requests': 0, 'loopback_http_requests': loopback_count,
        'gpu_requests': 0, 'hidden_transport_retries': 0}))


if __name__ == "__main__":
    main()
