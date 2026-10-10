"""Bound terminal retry evidence, single-board isolation and strict saved replay."""
from __future__ import annotations

from collections import Counter
import copy
from email.message import Message
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image

import cosplay_native_review as native
import cosplay_post_review as visual
from cosplay_native_review_smoke import add_frozen_feature_fixture, make_native_fixture
from cosplay_post_review_smoke import multi_fake_transport


class _TerminalStream:
    """Feed an actual transport parser an offline terminal SSE event."""
    def __init__(self, event):
        self.headers = Message()
        self.headers['Content-Type'] = 'text/event-stream'
        self.event = event

    def __enter__(self): return self
    def __exit__(self, *_): return False
    def __iter__(self):
        yield b'data: '+json.dumps(self.event).encode('utf-8')+b'\n'


def main():
    cases = []
    with tempfile.TemporaryDirectory(prefix='cosplay-board-http-retry-') as temporary:
        root = Path(temporary)
        run = make_native_fixture(root/'inputs', native_size=(660, 840))
        add_frozen_feature_fixture(run, root/'features')
        with patch.object(native, '_current_source_feature_observation', side_effect=AssertionError('No model allowed')):
            prepared = native.prepare_cosplay_native_review(run, root/'sampling')
        design = {'design_brief': {'summary': 'Synthetic board retry fixture', 'primary_effect_plan': {'mode': 'none'}}}
        base_transport = multi_fake_transport()
        actual_transport = visual.vision_transport.request_vision_json

        def produce(name, error_factory=None, *, always_fail=False, malformed_retry=False,
                    real_nested_input=False, terminal_event=None):
            calls = Counter()
            destinations = []
            def request(source, output, prompt, **kwargs):
                board = Path(source).stem
                calls[board] += 1
                destinations.append((board, str(output)))
                if (error_factory is not None or terminal_event is not None) and board == 'native-01' and (calls[board] == 1 or always_fail):
                    if terminal_event is not None:
                        config = {'registry_id': 'offline', 'kind': 'openai-compatible'}
                        with patch.object(visual.vision_transport, '_connection',
                                return_value=(config, 'https://offline.invalid/v1', 'synthetic-secret', None)), \
                             patch.object(visual.vision_transport.urllib.request, 'urlopen',
                                return_value=_TerminalStream(terminal_event)):
                            return actual_transport(source, output, prompt, vision_model='gpt-5.4')
                    visual.vision_transport._input_image(source, output)
                    raise error_factory()
                payload, provenance = base_transport(source, output, prompt, **kwargs)
                if real_nested_input:
                    sent, image_meta = visual.vision_transport._input_image(source, output)
                    assert sent == output.resolve()/'inputs/vision-input.jpg'
                    provenance.update(api_input=image_meta, input_sha256=image_meta['sha256'])
                if malformed_retry and board == 'native-01': payload = {'schema': 'wrong'}
                return payload, provenance
            with patch.object(visual.vision_transport, 'request_vision_json', side_effect=request):
                result = visual.review_cosplay_post(prepared['source_path'], Path(run['reference']['path']),
                    root/name, design, final=prepared['final_path'], native_crops=prepared['native_crops'],
                    native_selection=prepared['receipt'], max_workers=1)
            return result, calls, destinations

        current, calls, destinations = produce('valid', lambda: visual.vision_transport.VisionHTTPError(502, 'synthetic gateway failure'), real_nested_input=True)
        assert current['status'] == 'reviewed' and current['actual_api_evidence_valid']
        assert calls['native-01'] == 2 and all(count == 1 for board, count in calls.items() if board != 'native-01')
        board = next(row for row in current['reviews'] if row['id'] == 'native-01')
        assert board['text_vision_requests_attempted'] == 2 and board['first_error']['http_status'] == 502
        assert len(board['vision_attempts']) == 2 and current['text_vision_requests_attempted'] == len(current['reviews']) + 1
        assert destinations.count(('native-01', str(root/'valid/native-01/vision-attempt-2'))) == 1
        with patch.object(visual.vision_transport, 'request_vision_json', side_effect=AssertionError('No replay network allowed')):
            checked, proof = visual.validate_saved_comprehensive_review(prepared['source_path'],
                run['reference']['path'], root/'valid/review.json', design)
        assert checked == current and all(any(row['path'] == item['path'] for row in proof) for item in board['vision_attempts'])
        cases.append('HTTP502_one_failed_board_retry_successful_boards_not_resent_saved_replay')
        ordinary, calls, _ = produce('production-input', None, real_nested_input=True)
        assert ordinary['status'] == 'reviewed' and all(count == 1 for count in calls.values())
        assert all(row['text_vision_requests_attempted'] == 1 and 'vision_attempts' not in row for row in ordinary['reviews'])
        assert all(Path(row['api_input']['path']).parent.name == 'inputs' for row in ordinary['reviews'])
        with patch.object(visual.vision_transport, 'request_vision_json', side_effect=AssertionError('No replay network allowed')):
            checked, proof = visual.validate_saved_comprehensive_review(prepared['source_path'],
                run['reference']['path'], root/'production-input/review.json', design)
        assert checked == ordinary and proof
        cases.append('production_nested_JPEG_single_attempt_strict_replay')

        baseline = {path: path.read_bytes() for path in (root/'valid').rglob('*') if path.is_file()}
        def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        def synchronize(change):
            receipt = json.loads((root/'valid/review.json').read_text('utf-8'))
            target = next(row for row in receipt['reviews'] if row['id'] == 'native-01')
            change(target)
            write(root/'valid/native-01/review.json', target); write(root/'valid/review.json', receipt)
        def reject(name, mutation):
            mutation()
            with patch.object(visual.vision_transport, 'request_vision_json', side_effect=AssertionError('No replay network allowed')):
                try: visual.validate_saved_comprehensive_review(prepared['source_path'], run['reference']['path'], root/'valid/review.json', design)
                except ValueError: pass
                else: raise AssertionError(name+' accepted altered retry evidence')
            for path, content in baseline.items(): path.write_bytes(content)
            cases.append(name)
        first_marker = Path(board['vision_attempts'][0]['path'])
        first_receipt = json.loads(first_marker.read_text('utf-8'))
        first_input = Path(first_receipt['files'][0]['path'])
        reject('first_failed_HTTP_input_bytes_tamper_rejected', lambda: first_input.write_bytes(b'changed'))
        reject('first_HTTP_error_receipt_bytes_tamper_rejected', lambda: first_marker.write_text('{}', encoding='utf-8'))
        reject('attempt_count_forgery_rejected', lambda: synchronize(lambda row: row.update(text_vision_requests_attempted=1)))
        reject('omitted_attempt_ledger_rejected', lambda: synchronize(lambda row: [row.pop(key) for key in ('vision_attempts', 'first_error', 'retry_policy')]))
        def forged_http():
            item = copy.deepcopy(first_receipt)
            item['error']['http_status'] = 400; item['error']['message'] = 'vision provider HTTP 400: forged'
            write(first_marker, item)
            synchronize(lambda row: row.update(first_error=item['error'], vision_attempts=[
                {'path': str(first_marker), 'sha256': visual._sha(first_marker), 'bytes': first_marker.stat().st_size}, row['vision_attempts'][1]]))
        reject('rehashed_nonretryable_HTTP_forgery_rejected', forged_http)
        terminal_response = root/'valid/native-01/vision-attempt-2/model-response.json'
        reject('terminal_attempt_response_copy_tamper_rejected', lambda: terminal_response.write_text('{}', encoding='utf-8'))

        limited, calls, _ = produce('429', lambda: visual.vision_transport.VisionHTTPError(429, 'synthetic rate limit'))
        assert limited['status'] == 'reviewed' and calls['native-01'] == 2
        cases.append('explicit_HTTP429_retried_once')
        repeated, calls, _ = produce('twice-503', lambda: visual.vision_transport.VisionHTTPError(503, 'synthetic unavailable'), always_fail=True)
        failed = next(row for row in repeated['reviews'] if row['id'] == 'native-01')
        assert repeated['status'] == 'partial' and calls['native-01'] == 2 and failed['error']['http_status'] == 503
        assert len(failed['vision_attempts']) == 2 and failed['first_error']['http_status'] == 503
        cases.append('second_HTTP_failure_retained_without_third_attempt')
        for name, factory in (
                ('HTTP400', lambda: visual.vision_transport.VisionHTTPError(400, 'synthetic bad request')),
                ('timeout', lambda: TimeoutError('synthetic timeout outcome unknown')),
                ('disconnect', lambda: RuntimeError('synthetic connection outcome unknown')),
                ('schema', lambda: ValueError('synthetic schema error'))):
            result, calls, _ = produce(name, factory)
            assert result['status'] == 'partial' and calls['native-01'] == 1
            failed = next(row for row in result['reviews'] if row['id'] == 'native-01')
            assert 'vision_attempts' not in failed and failed['text_vision_requests_attempted'] == 1
            cases.append(name+'_not_blindly_retried')
        invalid, calls, _ = produce('schema-after-retry', lambda: visual.vision_transport.VisionHTTPError(502, 'synthetic gateway'), malformed_retry=True)
        assert invalid['status'] == 'partial' and calls['native-01'] == 2
        cases.append('invalid_retry_response_retained_without_third_attempt')
        invalid, calls, _ = produce('invalid-first-response', None, malformed_retry=True)
        assert invalid['status'] == 'partial' and calls['native-01'] == 1
        cases.append('invalid_completed_response_not_blindly_retried')

        provider_event = {'type': 'response.failed', 'response': {'id': 'offline-failed',
            'status': 'failed', 'error': {'type': 'server_error', 'code': 'upstream_http2_stream_error',
                'message': 'Synthetic terminal upstream stream failure'}}}
        json_event = {'type': 'response.completed', 'response': {'id': 'offline-completed',
            'status': 'completed', 'model': 'gpt-5.4', 'output_text': '{"broken":'}}
        typed_results = {}
        for name, event, kind, policy in (
                ('terminal-stream', provider_event, 'provider_failure',
                    'one_retry_after_explicit_provider_upstream_error'),
                ('terminal-upstream', {'type': 'error', 'type_detail': 'server_error',
                    'error': {'type': 'server_error', 'code': 'upstream_error',
                              'message': 'Synthetic terminal provider error'}}, 'provider_failure',
                    'one_retry_after_explicit_provider_upstream_error'),
                ('completed-bad-json', json_event, 'completed_invalid_json',
                    'one_retry_after_completed_response_invalid_json')):
            result, counts, _ = produce(name, terminal_event=event, real_nested_input=True)
            assert result['status'] == 'reviewed' and result['actual_api_evidence_valid']
            assert counts['native-01'] == 2 and all(count == 1 for key, count in counts.items() if key != 'native-01')
            reviewed = next(row for row in result['reviews'] if row['id'] == 'native-01')
            assert reviewed['first_error']['failure_kind'] == kind and reviewed['retry_policy'] == policy
            first = json.loads(Path(reviewed['vision_attempts'][0]['path']).read_text('utf-8'))
            assert first['status'] == kind
            assert Path(reviewed['api_input']['path']).parent.name == 'inputs'
            with patch.object(visual.vision_transport, 'request_vision_json', side_effect=AssertionError('No replay network allowed')):
                replayed, bound = visual.validate_saved_comprehensive_review(prepared['source_path'],
                    run['reference']['path'], root/name/'review.json', design)
            assert replayed == result
            assert any(item == reviewed['first_error']['failure_evidence'] for item in bound)
            typed_results[name] = (result, event)
            cases.append(name+'_one_retry_nested_JPEG_strict_replay_success')

            failed, counts, _ = produce(name+'-twice', terminal_event=event, always_fail=True, real_nested_input=True)
            failure = next(row for row in failed['reviews'] if row['id'] == 'native-01')
            assert failed['status'] == 'partial' and not failed['actual_api_evidence_valid']
            assert counts['native-01'] == 2 and len(failure['vision_attempts']) == 2
            assert failure['error']['failure_kind'] == kind
            second = json.loads(Path(failure['vision_attempts'][1]['path']).read_text('utf-8'))
            assert second['status'] == kind and second['error'] == failure['error']
            cases.append(name+'_second_failure_retained_without_third_attempt')

        def typed_forgery_cases(name, kind):
            folder = root/name
            baseline = {file: file.read_bytes() for file in folder.rglob('*') if file.is_file()}
            result, _ = typed_results[name]
            reviewed = next(row for row in result['reviews'] if row['id'] == 'native-01')
            marker = Path(reviewed['vision_attempts'][0]['path'])
            failure_path = Path(reviewed['first_error']['failure_evidence']['path'])
            failure = json.loads(failure_path.read_text('utf-8'))
            payload_path = Path(failure['response_payload']['path'])
            binding_path = Path(failure['request_binding']['path'])
            binding = json.loads(binding_path.read_text('utf-8'))
            input_path = Path(binding['api_input']['path'])

            def edit(path, mutation):
                value = json.loads(path.read_text('utf-8'))
                mutation(value)
                write(path, value)

            def refresh():
                # Rehash every enclosing receipt deliberately. Rejection must
                # come from reconstructed terminal/request conditions.
                saved_failure = json.loads(failure_path.read_text('utf-8'))
                saved_failure['request_binding'] = visual.vision_transport._record(binding_path)
                saved_failure['response_payload'] = visual.vision_transport._record(payload_path)
                write(failure_path, saved_failure)
                attempt = json.loads(marker.read_text('utf-8'))
                attempt['error']['failure_evidence'] = visual.vision_transport._record(failure_path)
                attempt['files'] = [visual.vision_transport._record(file) for file in sorted(marker.parent.rglob('*'))
                                    if file.is_file() and file != marker]
                write(marker, attempt)
                receipt = json.loads((folder/'review.json').read_text('utf-8'))
                board = next(row for row in receipt['reviews'] if row['id'] == 'native-01')
                board['first_error'] = attempt['error']
                board['vision_attempts'][0] = visual.vision_transport._record(marker)
                write(folder/'native-01/review.json', board)
                write(folder/'review.json', receipt)

            def reject_forgery(case, mutation):
                mutation(); refresh()
                with patch.object(visual.vision_transport, 'request_vision_json', side_effect=AssertionError('No forged replay network allowed')):
                    try:
                        visual.validate_saved_comprehensive_review(prepared['source_path'], run['reference']['path'], folder/'review.json', design)
                    except ValueError: pass
                    else: raise AssertionError(case+' accepted rehashed terminal evidence')
                for file, data in baseline.items(): file.write_bytes(data)
                cases.append(name+'_'+case+'_rehashed_rejected')

            reject_forgery('error_type', lambda: edit(marker, lambda row: row['error'].update(type='RuntimeError')))
            reject_forgery('status', lambda: edit(marker, lambda row: row.update(status='http_error')))
            reject_forgery('prompt', lambda: edit(binding_path, lambda row: row.update(prompt_sha256=visual._digest(b'other prompt'))))
            reject_forgery('source', lambda: edit(binding_path, lambda row: row['source'].update(sha256='0'*64)))
            def wrong_input():
                with Image.open(input_path) as current_input:
                    Image.new('RGB', current_input.size, '#cc3355').save(input_path, 'JPEG', quality=93, subsampling=0)
                edit(binding_path, lambda row: row['api_input'].update(sha256=visual._sha(input_path), bytes=input_path.stat().st_size))
            reject_forgery('JPEG_pixels', wrong_input)
            # refresh() normally rebuilds the full file ledger. This explicit
            # ledger omission is applied after rebuilding all enclosing hashes.
            def ledger_omission():
                refresh()
                edit(marker, lambda row: row['files'].pop())
                receipt = json.loads((folder/'review.json').read_text('utf-8'))
                board = next(row for row in receipt['reviews'] if row['id'] == 'native-01')
                board['vision_attempts'][0] = visual.vision_transport._record(marker)
                write(folder/'native-01/review.json', board); write(folder/'review.json', receipt)
            ledger_omission()
            with patch.object(visual.vision_transport, 'request_vision_json', side_effect=AssertionError('No ledger replay network allowed')):
                try: visual.validate_saved_comprehensive_review(prepared['source_path'], run['reference']['path'], folder/'review.json', design)
                except ValueError: pass
                else: raise AssertionError('Rehashed omitted failure ledger file was accepted')
            for file, data in baseline.items(): file.write_bytes(data)
            cases.append(name+'_ledger_omission_rehashed_rejected')

            if kind == 'provider_failure':
                reject_forgery('provider_code', lambda: edit(payload_path,
                    lambda row: row['response']['error'].update(code='invalid_api_key')))
                reject_forgery('provider_type', lambda: edit(payload_path,
                    lambda row: row['response']['error'].update(type='invalid_request_error')))
                reject_forgery('terminal_event_type', lambda: edit(payload_path,
                    lambda row: row.update(type='response.incomplete')))
                reject_forgery('terminal_completion', lambda: edit(payload_path,
                    lambda row: row['response'].update(status='completed')))
                def forged_kind():
                    edit(failure_path, lambda row: row.update(failure_kind='completed_invalid_json'))
                    edit(marker, lambda row: row['error'].update(failure_kind='completed_invalid_json'))
                reject_forgery('classification', forged_kind)
            else:
                reject_forgery('terminal_completion', lambda: edit(payload_path, lambda row: row.update(status='in_progress')))
                reject_forgery('valid_JSON_wrong_schema', lambda: edit(payload_path, lambda row: row.update(output_text='{"schema":"wrong"}')))
                reject_forgery('empty_response', lambda: edit(payload_path, lambda row: row.update(output_text='')))

        typed_forgery_cases('terminal-stream', 'provider_failure')
        typed_forgery_cases('completed-bad-json', 'completed_invalid_json')

        unknown_code = copy.deepcopy(provider_event)
        unknown_code['response']['error']['code'] = 'other_provider_error'
        missing_type = copy.deepcopy(provider_event)
        missing_type['response']['error'].pop('type')
        incomplete_json = copy.deepcopy(json_event)
        incomplete_json['response']['status'] = 'incomplete'
        no_completion = copy.deepcopy(json_event)
        no_completion['response'].pop('status')
        valid_json = copy.deepcopy(json_event)
        valid_json['response']['output_text'] = '{"schema":"wrong"}'
        nonobject_json = copy.deepcopy(json_event)
        nonobject_json['response']['output_text'] = '[]'
        for name, event in (
                ('unknown-provider-code', unknown_code), ('missing-server-error-type', missing_type),
                ('incomplete-json', incomplete_json), ('unproved-completion-json', no_completion),
                ('valid-json-schema-error', valid_json), ('valid-json-nonobject', nonobject_json),
                ('incomplete-event', {'type': 'response.incomplete', 'response': {'status': 'incomplete'}}),
                ('string-only-provider-failure', {'type': 'error', 'message': 'server_error upstream_error'})):
            result, counts, _ = produce(name, terminal_event=event, real_nested_input=True)
            failure = next(row for row in result['reviews'] if row['id'] == 'native-01')
            assert result['status'] == 'partial' and not result['actual_api_evidence_valid']
            assert counts['native-01'] == 1 and failure['text_vision_requests_attempted'] == 1
            assert 'vision_attempts' not in failure
            cases.append(name+'_not_blindly_retried')

        print(json.dumps({'ok': True, 'cases': cases, 'synthetic_assets': True,
            'vision_transport_mocked': True, 'network_requests': 0, 'model_inferences': 0,
            'image_generation_requests': 0, 'inference_verified': False}))


if __name__ == '__main__': main()
