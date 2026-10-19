"""Offline checks for protocol fallback and recognition provenance."""
from __future__ import annotations

import tempfile
import copy
import io
import json
import urllib.error
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
        ('timed-http', 'http', False, 39, vision.VisionHTTPError),
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
            if isinstance(value, dict) and 'model' in value:
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
            ('sse-failed-http2', b'data: '+json.dumps({'type': 'response.failed', 'response': {'status': 'failed', 'error': {
                'type': 'server_error', 'code': 'upstream_http2_stream_error', 'message': 'local-secret data:image/jpeg;base64,AAAA'}}}).encode()+b'\n', True, False, 'one_retry_after_explicit_provider_upstream_error'),
            ('sse-error-upstream', b'data: '+json.dumps({'type': 'error', 'error': {'type': 'server_error', 'code': 'upstream_error', 'message': 'failed'}}).encode()+b'\n', True, False, 'one_retry_after_explicit_provider_upstream_error'),
            ('http200-error', {'error': {'type': 'server_error', 'code': 'upstream_error', 'message': 'local-secret', 'authorization': 'Bearer local-secret', 'echo': 'data:image/png;base64,BBBB', 'local-secret': 'echoed key name', 'b64_json': 'AAAA-secret-image'}}, False, False, 'one_retry_after_explicit_provider_upstream_error'),
            ('responses-completed-bad-json', {'status': 'completed', 'output_text': '{"broken":'}, False, False, 'one_retry_after_completed_response_invalid_json'),
            ('chat-stop-bad-json', {'choices': [{'finish_reason': 'stop', 'message': {'content': '```json\n{"broken":\n```'}}]}, False, True, 'one_retry_after_completed_response_invalid_json')]:
        error, exc, calls, directory = execute(name, payload, sse=sse, chat=chat)
        assert isinstance(exc, vision.VisionResponseError), (name, error)
        assert len(calls) == 1, 'Transport must not hide a retry'
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


def main():
    with tempfile.TemporaryDirectory(prefix="cosplay_vision_") as temporary:
        root = Path(temporary)
        source = root / "source.png"
        Image.new("RGB", (64, 96), (50, 70, 100)).save(source)
        requests = []
        def local_response(endpoint, key, body=None, *, timings=None):
            requests.append(endpoint)
            assert key == "local-secret"
            if endpoint.endswith("/responses"):
                assert body["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
                raise vision.VisionHTTPError(404, "unknown endpoint")
            return {"id": "offline-response", "model": "gpt-5.4", "choices": [{"message": {
                "content": '```json\n{"character_name":"unknown","series":"unknown"}\n```'
            }}]}
        config = {"registry_id": "offline", "kind": "openai-compatible"}
        with patch.object(vision, "_connection", return_value=(config, "http://localhost/v1", "local-secret", None)), \
             patch.object(vision, "_request", side_effect=local_response):
            payload, provenance = vision.request_vision_json(source, root / "probe", "Identify supported character", vision_model="gpt-5.4")
            assert payload["character_name"] == "unknown"
            assert provenance["protocol"] == "chat/completions" and provenance["fallback"]["http_status"] == 404
            assert len(requests) == 2
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
    print(json.dumps({'schema': 'vision_transport_smoke.v2', 'status': 'passed',
        'cases_passed': len(passed)+1, 'cases': ['original_protocol_model_and_provenance_contract', *passed],
        'network_requests': 0, 'gpu_requests': 0, 'hidden_transport_retries': 0}))


if __name__ == "__main__":
    main()
