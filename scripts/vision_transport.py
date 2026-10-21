"""GPT vision JSON transport, separate from the image-generation adapter."""
from __future__ import annotations

import base64
import hashlib
import json
import io
import os
import re
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
from urllib.parse import urlsplit
from uuid import uuid4

from PIL import Image, ImageCms, ImageOps

import provider_settings
import reference_provider
from raw_support import RGB_DECODE_VERSION, is_raw, open_rgb


TIMEOUT = 180
VISION_GUIDE = "https://developers.openai.com/api/docs/guides/images-vision"


class VisionHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"vision provider HTTP {status}: {detail}")


class VisionResponseError(RuntimeError):
    """An observed response failure; retry authority requires its saved evidence."""
    def __init__(self, failure_kind, message, *, response=None, event_type=None):
        self.failure_kind = failure_kind
        self.failure_evidence = None
        self._response = response
        self._event_type = event_type
        super().__init__(message)


def _sanitized(value, key):
    if isinstance(value, dict):
        return {_sanitized(name, key): '<redacted>' if str(name).lower() in
                {'authorization', 'api_key', 'api-key', 'x-api-key', 'access_token',
                 'b64_json', 'image_base64', 'base64_image'}
                else _sanitized(item, key) for name, item in value.items()}
    if isinstance(value, list):
        return [_sanitized(item, key) for item in value]
    if isinstance(value, str):
        value = value.replace(key, '<redacted>') if key else value
        return re.sub(r'data:image/[^\s"\']+', '<image-data-redacted>', value)
    return value


def _record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': reference_provider.sha256(path), 'bytes': path.stat().st_size}


def _exclusive_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _observation_json(path, value):
    """Publish a complete observation before consuming the next transport event."""
    path = Path(path)
    temporary = path.with_name('.'+path.name+'.'+uuid4().hex+'.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        # Windows readers may briefly deny delete-sharing on the old summary.
        # Retry only this local rename; a persistence error never resends HTTP.
        for retry in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if retry == 7:
                    raise
                sleep(.01)
    finally:
        temporary.unlink(missing_ok=True)


def _observation_identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _read_observation_record(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def _http_observation_locator(directory, binding_id):
    return {'path': str(directory), 'request_binding_id': binding_id,
        'request': _read_observation_record(directory/'request.json'),
        'response': _read_observation_record(directory/'response.json')}


def _observation_producer(*, read_only=False):
    from runtime_resources import code_path
    digest = (lambda path: _read_observation_record(path)['sha256']) if read_only else reference_provider.sha256
    return {name: digest(code_path(name)) for name in
            ('vision_transport.py', 'raw_support.py', 'reference_provider.py')}


def _selected_connection_identity(provider, vision_model, api_base, api_url, api_key_env):
    return _observation_identity({'selected': {'provider': provider, 'vision_model': vision_model,
        'api_base': api_base, 'api_url': api_url, 'api_key_env': api_key_env},
        'environment': {name: os.environ.get(name) for name in (
            'COSPLAY_VISION_PROVIDER', 'COSPLAY_VISION_MODEL', 'COSPLAY_VISION_API_BASE', 'COSPLAY_VISION_API_URL')}})


def _observed_outcome(payload, protocol, event_type=None, http_status=None):
    if http_status is not None and http_status >= 400:
        return 'http_error'
    if isinstance(payload, dict):
        frame_status = {'response.completed': 'completed', 'response.failed': 'failed',
                        'response.incomplete': 'incomplete'}.get(event_type)
        if (frame_status is not None and payload.get('status') not in (None, frame_status)
                or payload.get('status') == 'completed' and payload.get('error')):
            return 'unknown'
    if _terminal_provider_failure(payload, event_type):
        return 'failed'
    if _completed(payload, protocol):
        return 'completed'
    if event_type == 'response.incomplete' or (isinstance(payload, dict)
            and payload.get('status') == 'incomplete'):
        return 'incomplete'
    return 'unknown'


def _input_edge(input_profile):
    if not isinstance(input_profile, str) or input_profile not in {'standard', 'detailed_board'}:
        raise ValueError('Unsupported vision input profile')
    return 1024 if input_profile == 'standard' else 2048


class _HTTPObservation:
    """One invocation owns distinct HTTP attempts, including discovery/fallback.

    These are observations, not retry authority or proof of model acceptance.
    A submitting record deliberately remains unknown until a response is seen.
    """
    def __init__(self, directory, *, source, prompt, provider, key, connection_identity, input_profile):
        self.directory = Path(directory)/'http-attempts'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.source = _record(source)
        self.prompt_sha256 = hashlib.sha256(_final_prompt(prompt).encode('utf-8')).hexdigest()
        self.producer = _observation_producer()
        self.provider, self.key = provider, key
        self.connection_identity = connection_identity
        self.input_profile = input_profile
        self.api_input = None
        self.image_binding = None
        self.attempts = []

    def attempt(self, endpoint, protocol, model):
        directory = self.directory/('attempt-'+str(len(self.attempts)+1).zfill(4))
        directory.mkdir()
        parsed = urlsplit(endpoint)
        public_endpoint = parsed._replace(netloc=parsed.netloc.rsplit('@', 1)[-1], query='', fragment='').geturl()
        binding = {'schema': 'vision_http_request.v1', 'source': self.source,
            'api_input': self.api_input, 'prompt_sha256': self.prompt_sha256,
            'image_request_binding': self.image_binding,
            'provider': self.provider, 'model': model, 'protocol': protocol,
            'connection_identity': self.connection_identity,
            'endpoint': public_endpoint, 'endpoint_sha256': hashlib.sha256(endpoint.encode()).hexdigest(),
            'profile': 'json_object_v1' if protocol != 'models' else 'model_discovery',
            'input_profile': self.input_profile,
            'producer': self.producer}
        attempt = _HTTPAttempt(directory, _sanitized(binding, self.key), self.key)
        self.attempts.append(attempt)
        return attempt


class _HTTPAttempt:
    def __init__(self, directory, binding, key):
        self.directory, self.binding, self.key = directory, binding, key
        self.observed = {'schema': 'vision_http_observation.v1', 'outcome': 'unknown'}

    def submitting(self, encoded):
        self.binding.update(request_bytes=len(encoded) if encoded is not None else 0,
            request_body_sha256=hashlib.sha256(encoded or b'').hexdigest())
        self.binding_id = _observation_identity(self.binding)
        # Exclusive creation also rejects accidental reuse within this invocation.
        with (self.directory/'request.json').open('x', encoding='utf-8') as stream:
            json.dump({**self.binding, 'request_binding_id': self.binding_id}, stream,
                ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        self.observed.update(request_binding_id=self.binding_id,
            stage='submitting', semantics='request_may_be_sent_not_provider_acceptance')
        self.save()

    def save(self):
        self.observed['observed_at'] = datetime.now(timezone.utc).isoformat()
        _observation_json(self.directory/'observation.json', _sanitized(self.observed, self.key))

    def headers(self, response):
        headers = response.headers
        self.observed.update(stage='headers_received', http_status=getattr(response, 'status', None)
            or getattr(response, 'code', None), headers={name: headers.get(name) for name in
            ('x-request-id', 'request-id', 'openai-request-id', 'retry-after', 'content-type')
            if headers.get(name) is not None})
        self.save()

    def payload(self, payload, *, event_type=None, complete=False):
        response = payload.get('response', payload) if isinstance(payload, dict) else {}
        if isinstance(response, dict) and response.get('id'):
            previous = self.observed.get('response_id')
            if previous is not None and previous != response['id']:
                self.observed.update(response_identity_conflict=True,
                    conflicting_response_ids=[previous, response['id']])
            self.observed['response_id'] = response['id']
        if complete:
            outcome = _observed_outcome(response, self.binding['protocol'], event_type,
                self.observed.get('http_status'))
            terminal = {**self.observed, 'schema': 'vision_http_response.v1',
                'stage': 'response_received', 'event_type': event_type, 'outcome': outcome,
                'payload': payload, 'observed_at': datetime.now(timezone.utc).isoformat()}
            if isinstance(response, dict):
                terminal['profile_text_redacted'] = (_response_text(response, self.binding['protocol']) !=
                    _response_text(_sanitized(response, self.key), self.binding['protocol']))
            terminal = _sanitized(terminal, self.key)
            terminal['receipt_id'] = _observation_identity(terminal)
            # One atomic file contains both the complete payload and its binding.
            # A kill before the later summary save still leaves readable evidence.
            _observation_json(self.directory/'response.json', terminal)
            self.observed.update(stage='response_received', outcome=outcome, event_type=event_type)
        else:
            self.observed.update(stage='stream_in_progress', event_type=event_type)
        self.save()


def read_terminal_response(directory, *, source, prompt, expected_request_binding_id,
                           input_profile='standard'):
    """Read verified new-format evidence only; never retrieve, replay or resubmit.

    The caller supplies the independently retained request binding identity.
    A response receipt establishes no budget, retry or acceptance authority.
    """
    _input_edge(input_profile)
    directory = Path(directory).resolve()
    request = json.loads((directory/'request.json').read_text(encoding='utf-8'))
    binding = {key: value for key, value in request.items() if key != 'request_binding_id'}
    if (request.get('schema') != 'vision_http_request.v1'
            or not re.fullmatch('[a-f0-9]{64}', str(expected_request_binding_id))
            or request.get('request_binding_id') != expected_request_binding_id
            or _observation_identity(binding) != expected_request_binding_id
            or request.get('profile') != 'json_object_v1'
            or request.get('input_profile') != input_profile
            or request.get('protocol') not in {'responses', 'chat/completions'}
            or request.get('producer') != _observation_producer(read_only=True)
            or request.get('source') != _read_observation_record(source)
            or request.get('prompt_sha256') != hashlib.sha256(_final_prompt(prompt).encode('utf-8')).hexdigest()):
        raise ValueError('Saved HTTP request differs from the expected inputs, prompt, profile or producer')
    image = request.get('api_input') or {}
    actual_image = _read_observation_record(image['path'])
    image_source = image.get('source') or {}
    if (not {'path', 'sha256'} <= set(image_source) <= {'path', 'sha256', 'bytes'}
            or image_source != {key: request['source'][key] for key in image_source}
            or any(actual_image[key] != image.get(key) for key in ('path', 'sha256', 'bytes'))):
        raise ValueError('Saved HTTP request image changed')
    response = json.loads((directory/'response.json').read_text(encoding='utf-8'))
    payload = response.get('payload')
    envelope = payload.get('response', payload) if isinstance(payload, dict) else payload
    outcome = _observed_outcome(envelope, request['protocol'], response.get('event_type'), response.get('http_status'))
    if (response.get('schema') != 'vision_http_response.v1'
            or response.get('receipt_id') != _observation_identity(
                {key: value for key, value in response.items() if key != 'receipt_id'})
            or response.get('request_binding_id') != expected_request_binding_id
            or response.get('response_identity_conflict') is True
            or response.get('event_type') is not None and (
                not isinstance(payload, dict) or payload.get('type') != response['event_type'])
            or response.get('outcome') != outcome or outcome not in {'completed', 'failed', 'incomplete', 'http_error'}
            or isinstance(envelope, dict) and envelope.get('id') is not None
                and response.get('response_id') != envelope['id']):
        raise ValueError('Saved response does not establish a matching observed terminal outcome')
    return {'request': request, 'response': response, 'receipt': _read_observation_record(directory/'response.json'),
        'retry_authorized': False, 'human_approved': False}


def _replay_rgb(source):
    """Reconstruct the bound decode without the decoder's opt-in profile writes.

    A different decoder/recipe must fail the subsequent exact JPEG check.
    """
    from raw_support import deterministic_srgb_profile, linear_preview
    if is_raw(source):
        import numpy as np
        rawpy = reference_provider._require_rawpy()
        with rawpy.imread(str(source)) as raw:
            values = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                gamma=(1, 1), output_color=rawpy.ColorSpace.sRGB, half_size=False)
        return linear_preview(np.asarray(values, np.float32)/np.float32(65535))
    with Image.open(source) as opened:
        oriented = ImageOps.exif_transpose(opened)
        if opened.info.get('icc_profile'):
            input_profile = ImageCms.ImageCmsProfile(io.BytesIO(opened.info['icc_profile']))
            output_profile = ImageCms.ImageCmsProfile(io.BytesIO(deterministic_srgb_profile()))
            cms_input = oriented if oriented.mode in {'RGB', 'RGBA', 'CMYK', 'LAB', 'L'} else oriented.convert('RGB')
            return ImageCms.profileToProfile(cms_input, input_profile, output_profile, outputMode='RGB')
        return oriented.convert('RGB')


def replay_vision_json(source, out_dir, prompt, *, expected_connection_identity, vision_model=None,
                       input_profile='standard'):
    """Reparse a completed observation against caller-owned current intent.

    No connection/credential discovery, HTTP, writes, retry or budget mutation.
    The caller obtains expected_connection_identity from its current checked
    binding, never from these journals. Old formats/producers are not adopted.
    """
    long_edge = _input_edge(input_profile)
    if not re.fullmatch('[a-f0-9]{64}', str(expected_connection_identity)):
        raise ValueError('A caller-owned connection identity is required for vision replay')
    source, out_dir = Path(source).resolve(), Path(out_dir).resolve()
    original = _read_observation_record(source)
    final_prompt = _final_prompt(prompt)
    producer = _observation_producer(read_only=True)
    model = vision_model or os.environ.get('COSPLAY_VISION_MODEL')
    attempts = sorted((out_dir/'http-attempts').iterdir())
    if not attempts or [p.name for p in attempts] != ['attempt-'+str(i).zfill(4) for i in range(1, len(attempts)+1)]:
        raise ValueError('Vision replay requires the original ordered HTTP attempts')
    image_binding = json.loads((out_dir/'request-binding.json').read_text(encoding='utf-8'))
    image = image_binding.get('api_input') or {}
    # Reconstruct the current JPEG in memory, leaving the historical input alone.
    from raw_support import deterministic_srgb_profile
    decoded = _replay_rgb(source)
    decoded.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    with io.BytesIO() as encoded_image:
        decoded.save(encoded_image, 'JPEG', quality=95 if is_raw(source) else 93,
            subsampling=0, icc_profile=deterministic_srgb_profile())
        if hashlib.sha256(encoded_image.getvalue()).hexdigest() != image.get('sha256'):
            raise ValueError('Saved API image does not depict the current source/decoder')
    if (image_binding.get('schema') != 'vision_request_binding.v1'
            or image_binding.get('input_profile') != input_profile
            or image_binding.get('source') != original
            or image_binding.get('prompt_sha256') != hashlib.sha256(final_prompt.encode()).hexdigest()
            or image_binding.get('allowed_protocols') not in (['responses', 'chat/completions'], ['chat/completions'])):
        raise ValueError('Saved vision image binding differs from current source/prompt/protocol')
    image_uri = 'data:image/jpeg;base64,'+base64.b64encode(Path(image['path']).read_bytes()).decode('ascii')
    discovered, fallback, observed = [], None, []
    for index, directory in enumerate(attempts):
        saved = json.loads((directory/'request.json').read_text(encoding='utf-8'))
        protocol = saved.get('protocol')
        expected = {key: value for key, value in saved.items() if key != 'request_binding_id'}
        expected.update(source=original, producer=producer, connection_identity=expected_connection_identity,
            prompt_sha256=hashlib.sha256(final_prompt.encode()).hexdigest(), input_profile=input_profile)
        if protocol == 'models':
            if index != 0 or model is not None:
                raise ValueError('Unexpected discovery request in vision replay')
            expected.update(api_input=None, image_request_binding=None, model=None, profile='model_discovery',
                request_bytes=0, request_body_sha256=hashlib.sha256(b'').hexdigest())
            expected_id = _observation_identity(expected)
            response = json.loads((directory/'response.json').read_text(encoding='utf-8'))
            if (saved.get('schema') != 'vision_http_request.v1' or saved.get('request_binding_id') != expected_id
                    or response.get('schema') != 'vision_http_response.v1'
                    or response.get('receipt_id') != _observation_identity(
                        {k: v for k, v in response.items() if k != 'receipt_id'})
                    or response.get('request_binding_id') != expected_id
                    or response.get('outcome') != 'unknown' or response.get('event_type') is not None):
                raise ValueError('Saved model discovery differs from the caller binding')
            discovered = _vision_models(response['payload'])
            if not discovered:
                raise ValueError('Saved discovery has no GPT vision text model')
            model = max(discovered, key=_model_rank)
            observed.append(_http_observation_locator(directory, expected_id))
            continue
        if protocol not in image_binding['allowed_protocols'] or not model or 'gpt-image' in str(model).lower():
            raise ValueError('Saved protocol/model differs from the current vision request')
        if protocol == 'responses':
            body = {'model': model, 'store': False, 'stream': True, 'input': [{'role': 'user', 'content': [
                {'type': 'input_text', 'text': final_prompt}, {'type': 'input_image', 'image_url': image_uri}]}]}
        else:
            if image_binding['allowed_protocols'][0] == 'responses' and fallback is None:
                raise ValueError('Chat replay requires the original observed protocol fallback')
            body = {'model': model, 'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': final_prompt}, {'type': 'image_url', 'image_url': {'url': image_uri}}]}]}
        encoded = json.dumps(body).encode('utf-8')
        expected.update(api_input=image, image_request_binding=_read_observation_record(out_dir/'request-binding.json'),
            model=model, profile='json_object_v1', request_bytes=len(encoded),
            request_body_sha256=hashlib.sha256(encoded).hexdigest())
        expected_id = _observation_identity(expected)
        verified = read_terminal_response(directory, source=source, prompt=prompt,
            expected_request_binding_id=expected_id, input_profile=input_profile)
        response = verified['response']
        observed.append(_http_observation_locator(directory, expected_id))
        if response['outcome'] == 'http_error' and protocol == 'responses' and index+1 < len(attempts):
            if fallback is not None or response.get('http_status') not in {400, 404, 405, 422, 501}:
                raise ValueError('Saved HTTP outcome cannot establish the original protocol fallback')
            fallback = {'from': 'responses', 'http_status': response['http_status'], 'reason': 'endpoint_unsupported'}
            continue
        if (index != len(attempts)-1 or response['outcome'] != 'completed'
                or response.get('profile_text_redacted') is not False):
            raise ValueError('Vision replay requires one completed, unchanged profile response')
        envelope = response['payload']
        if response.get('event_type') is not None:
            envelope = envelope['response']
        text = _response_text(envelope, protocol)
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Saved completed response has no profile text')
        payload = _profile_json(text)
        if _read_observation_record(source) != original:
            raise ValueError('Source changed during read-only vision replay')
        provenance = {'schema_version': 'vision_transport.v1', 'provider': saved['provider'],
            'model': model, 'response_model': envelope.get('model'), 'response_id': envelope.get('id'),
            'endpoint': saved['endpoint'], 'protocol': protocol, 'created_at': response['observed_at'],
            'timeout_seconds': TIMEOUT, 'source': {'path': original['path'], 'sha256': original['sha256']},
            'api_input': image, 'input_sha256': image['sha256'], 'prompt_sha256': expected['prompt_sha256'],
            'input_profile': input_profile,
            'usage': envelope.get('usage'), 'discovered_models': discovered, 'fallback': fallback,
            'network_used': True, 'generation_used': False, 'docs_source': VISION_GUIDE,
            'http_observations': observed, 'response_replay': {'receipt': verified['receipt'],
                'new_http_requests': 0, 'new_model_requests': 0, 'human_approved': False}}
        return payload, provenance
    raise ValueError('No completed vision profile is available for replay')


def _final_prompt(prompt):
    return prompt + "\nReturn only one JSON object, without markdown. Follow the requested JSON schema and its uncertainty conventions. Do not guess unsupported identities."


def _terminal_provider_failure(payload, event_type=None):
    if not isinstance(payload, dict):
        return False
    return bool(event_type in {'response.failed', 'error'}
                or payload.get('status') == 'failed' or payload.get('error'))


def _provider_retryable(payload, event_type):
    if not _terminal_provider_failure(payload, event_type):
        return False
    if event_type is not None and payload.get('type') != event_type:
        return False
    if payload.get('type') in {'response.incomplete', 'response.completed'}:
        return False
    if payload.get('status') in {'completed', 'incomplete', 'in_progress', 'queued'}:
        return False
    terminal = payload.get('response') if event_type == 'response.failed' else payload
    if not isinstance(terminal, dict) or terminal.get('status') in {'completed', 'incomplete', 'in_progress', 'queued'}:
        return False
    error = terminal.get('error')
    return bool(isinstance(error, dict) and error.get('type') == 'server_error'
        and error.get('code') in {'upstream_http2_stream_error', 'upstream_error'})


def _response_text(response, protocol):
    if protocol == 'chat/completions':
        choices = response.get('choices')
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        message = choices[0].get('message') or {}
        return message.get('content') if isinstance(message, dict) else None
    return response.get('output_text') or ''.join(
        part.get('text', '') for item in response.get('output', []) if isinstance(item, dict)
        and item.get('type') == 'message' for part in item.get('content', [])
        if isinstance(part, dict) and part.get('type') == 'output_text')


def _completed(response, protocol):
    if response.get('type') in {'error', 'response.failed', 'response.incomplete'}:
        return False
    if protocol == 'responses':
        return response.get('status') == 'completed'
    choices = response.get('choices')
    return bool(isinstance(choices, list) and choices and isinstance(choices[0], dict)
                and choices[0].get('finish_reason') == 'stop')


def _syntax_invalid(text):
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        _profile_json(text)
    except RuntimeError as exc:
        return isinstance(exc.__cause__, json.JSONDecodeError)
    return False


def error_metadata(exc, api_key_env=None):
    """Additional structured metadata; consumers retain type/message and HTTP ABI."""
    metadata = {}
    if isinstance(exc, VisionHTTPError):
        metadata['http_status'] = exc.status
    if isinstance(exc, VisionResponseError) and exc.failure_evidence:
        metadata.update(failure_kind=exc.failure_kind, failure_evidence=dict(exc.failure_evidence))
    if getattr(exc, 'vision_timings', None) is not None:
        metadata['timings'] = exc.vision_timings
    return metadata


def retry_policy_for_error(error):
    if not isinstance(error, dict):
        return None
    status = error.get('http_status')
    if (error.get('type') == 'VisionHTTPError' and type(status) is int
            and (status == 429 or 500 <= status <= 599)
            and str(error.get('message', '')).startswith('vision provider HTTP '+str(status)+': ')):
        return 'one_retry_after_explicit_http_429_or_5xx'
    if error.get('type') != 'VisionResponseError' or not isinstance(error.get('failure_evidence'), dict):
        return None
    try:
        asset = error['failure_evidence']
        if _record(asset['path']) != asset:
            return None
        receipt = json.loads(Path(asset['path']).read_text(encoding='utf-8'))
        payload = receipt['response_payload']
        if _record(payload['path']) != payload or receipt.get('failure_kind') != error.get('failure_kind'):
            return None
        response = json.loads(Path(payload['path']).read_text(encoding='utf-8'))
        if error['failure_kind'] == 'provider_failure' and _provider_retryable(response, receipt.get('event_type')):
            return 'one_retry_after_explicit_provider_upstream_error'
        if (error['failure_kind'] == 'completed_invalid_json' and _completed(response, receipt['protocol'])
                and not _terminal_provider_failure(response)
                and _syntax_invalid(_response_text(response, receipt['protocol']))):
            return 'one_retry_after_completed_response_invalid_json'
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    return None


def validate_saved_failure(error, comparison, prompt, *, input_profile='standard'):
    """Reconstruct qualification and exact request pixels without sending a request.

    Returned closure is inside the historical attempt directory. The original
    comparison is independently checked against the current same-byte board.
    """
    _input_edge(input_profile)
    policy = retry_policy_for_error(error)
    if policy is None:
        raise ValueError('Saved vision failure does not establish retry qualification')
    if error.get('type') == 'VisionHTTPError':
        return []
    asset = error['failure_evidence']
    receipt = json.loads(Path(asset['path']).read_text(encoding='utf-8'))
    directory = Path(asset['path']).resolve().parent
    if (Path(asset['path']).name != 'response-failure.json'
            or receipt.get('schema') != 'vision_response_failure.v1'):
        raise ValueError('Unknown saved response failure receipt')
    binding = receipt['request_binding']
    if _record(binding['path']) != binding or Path(binding['path']).resolve() != directory/'request-binding.json':
        raise ValueError('Saved failure request binding changed or belongs to another attempt')
    request = json.loads(Path(binding['path']).read_text(encoding='utf-8'))
    expected_prompt_hash = hashlib.sha256(_final_prompt(prompt).encode('utf-8')).hexdigest()
    if (request.get('schema') != 'vision_request_binding.v1'
            or request.get('input_profile') != input_profile
            or receipt['protocol'] not in {'responses', 'chat/completions'}
            or receipt['protocol'] not in request['allowed_protocols']
            or request['prompt_sha256'] != expected_prompt_hash):
        raise ValueError('Saved failure protocol or prompt differs from the actual board request')
    original = request['source']
    if (_record(original['path']) != original or reference_provider.sha256(Path(comparison)) != original['sha256']):
        raise ValueError('Saved failure source differs from the actual comparison pixels')
    image = request['api_input']
    image_path = Path(image['path']).resolve()
    if (image_path != directory/'inputs/vision-input.jpg' or image.get('source') !=
            {'path': original['path'], 'sha256': original['sha256']}):
        raise ValueError('Saved failure JPEG belongs to another source or attempt')
    image_record = _record(image_path)
    if any(image_record[key] != image.get(key) for key in ('path', 'sha256', 'bytes')):
        raise ValueError('Saved failure API JPEG bytes changed')
    with tempfile.TemporaryDirectory(prefix='vision-failure-replay-') as tmp:
        replay, metadata = _input_image(Path(original['path']), Path(tmp), input_profile=input_profile)
        if reference_provider.sha256(replay) != image['sha256'] or any(
                metadata[k] != image.get(k) for k in ('format', 'size', 'derived', 'decode')):
            raise ValueError('Saved failure JPEG does not depict the bound comparison transport')
    payload = receipt['response_payload']
    if Path(payload['path']).resolve() != directory/'response-failure-payload.json' or _record(payload['path']) != payload:
        raise ValueError('Saved terminal response payload changed or belongs to another attempt')
    proof = [asset, binding, image_record, payload]
    if any(_record(row['path']) != row for row in proof):
        raise ValueError('Saved failure evidence changed during validation')
    return proof


def _save_response_failure(exc, out_dir, request_binding, protocol, key):
    response = _sanitized(exc._response, key)
    payload_path = out_dir/'response-failure-payload.json'
    _exclusive_json(payload_path, response)
    receipt = {'schema': 'vision_response_failure.v1', 'failure_kind': exc.failure_kind,
        'event_type': exc._event_type, 'protocol': protocol,
        'request_binding': request_binding, 'response_payload': _record(payload_path)}
    if getattr(exc, 'vision_timings', None) is not None:
        receipt['timings'] = exc.vision_timings
    path = out_dir/'response-failure.json'
    _exclusive_json(path, receipt)
    exc.failure_evidence = _record(path)
    exc.args = (_safe_detail(str(exc), key),)
    exc._response = response


def _safe_detail(value: str, key: str) -> str:
    value = value.replace(key, "<redacted>") if key else value
    value = re.sub(r"data:image/[^\s\"']+", "<image-data-redacted>", value)
    return value[:500]


@contextmanager
def _timed(timings, field):
    started = perf_counter()
    try:
        yield
    except Exception:
        timings['failed_stage'] = field
        raise
    finally:
        timings[field] = timings.get(field, 0.0) + perf_counter() - started


def _request(endpoint: str, key: str, body: dict | None = None, *, timings=None, observation=None) -> dict:
    """One HTTP request; header wait includes connection and sending, not just inference."""
    timings = timings if timings is not None else {}
    started = perf_counter()
    timings.update(request_encode_seconds=0.0, wait_headers_seconds=0.0,
                   response_read_seconds=0.0, response_parse_seconds=0.0)
    try:
        with _timed(timings, 'request_encode_seconds'):
            encoded = json.dumps(body).encode("utf-8") if body is not None else None
            request = urllib.request.Request(endpoint, data=encoded,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                         "Accept": "application/json, text/event-stream",
                         "User-Agent": "cosplay-vision/1.0"})
            timings['request_bytes'] = len(encoded) if encoded is not None else 0
        if observation is not None:
            observation.submitting(encoded)
        try:
            with _timed(timings, 'wait_headers_seconds'):
                response = urllib.request.urlopen(request, timeout=TIMEOUT)
            if observation is not None:
                observation.headers(response)
            with response:
                if response.headers.get_content_type() == "text/event-stream":
                    timings['response_format'] = 'sse'
                    lines = iter(response)
                    while True:
                        with _timed(timings, 'response_read_seconds'):
                            line = next(lines, None)
                        if line is None:
                            break
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == b"[DONE]":
                            continue
                        with _timed(timings, 'response_parse_seconds'):
                            event = json.loads(data)
                        if observation is not None and event.get('type') in {
                                'response.created', 'response.in_progress', 'response.completed',
                                'response.failed', 'response.incomplete', 'error'}:
                            observation.payload(event, event_type=event.get('type'),
                                complete=event.get('type') not in {'response.created', 'response.in_progress'})
                        if event.get("type") == "response.completed":
                            return event["response"]
                        if event.get("type") in {"error", "response.failed"}:
                            message = event.get("message") or event.get("response", {}).get("error") or event.get("type")
                            raise VisionResponseError('provider_failure', "vision response failed: " + _safe_detail(str(message), key),
                                response=event, event_type=event.get('type'))
                        if event.get('type') == 'response.incomplete':
                            raise RuntimeError('vision response incomplete')
                    raise RuntimeError("vision response stream ended without a completed response")
                timings['response_format'] = 'json'
                with _timed(timings, 'response_read_seconds'):
                    raw = response.read()
                with _timed(timings, 'response_parse_seconds'):
                    payload = json.loads(raw)
                if observation is not None:
                    observation.payload(payload, complete=True)
        except urllib.error.HTTPError as exc:
            timings['http_status'] = exc.code
            if observation is not None:
                observation.headers(exc)
            with _timed(timings, 'response_read_seconds'):
                raw = exc.read(4096)
            with _timed(timings, 'response_parse_seconds'):
                detail = _safe_detail(raw.decode("utf-8", errors="replace"), key)
            if observation is not None:
                try:
                    error_payload = json.loads(raw)
                except (ValueError, UnicodeError):
                    error_payload = {'unparsed_body_sha256': hashlib.sha256(raw).hexdigest()}
                observation.payload(error_payload, complete=True)
            raise VisionHTTPError(exc.code, detail) from None
        except urllib.error.URLError as exc:
            raise RuntimeError("vision provider connection failed: " + _safe_detail(str(exc.reason), key)) from None
        if not isinstance(payload, dict):
            raise RuntimeError("vision provider returned a non-object JSON response")
        if _terminal_provider_failure(payload):
            raise VisionResponseError('provider_failure', 'vision response failed: '+_safe_detail(str(payload.get('error') or 'failed'), key), response=payload)
        return payload
    except Exception:
        timings['failed_stage'] = timings.get('failed_stage', 'response_validation')
        raise
    finally:
        timings['total_seconds'] = perf_counter() - started


def _connection(*, provider=None, api_base=None, api_url=None, api_key_env=None):
    config = reference_provider.provider_config(provider=provider, api_base=api_base,
                                                api_url=api_url, api_key_env=api_key_env)
    entry = reference_provider._provider_entry(config["registry_id"], reference_provider.load_provider_registry())
    if config["kind"] != "openai-compatible" and entry.get("vision_provider") == "auto":
        # Image generation and text/vision planning have independent transports.
        # Never send GPT credentials to a local image server's endpoint.
        chosen = os.environ.get("COSPLAY_VISION_PROVIDER")
        if not chosen:
            default = provider_settings.load_settings().get("default_provider")
            if default and reference_provider._provider_entry(default, reference_provider.load_provider_registry()).get("kind") == "openai-compatible":
                chosen = default
            else:
                chosen = "ccswitch_sub2api" if provider_settings.discover_ccswitch()["available"] else "openai-compatible"
        config = reference_provider.provider_config(provider=chosen,
            api_base=os.environ.get("COSPLAY_VISION_API_BASE"),
            api_url=os.environ.get("COSPLAY_VISION_API_URL"),
            api_key_env=os.environ.get("COSPLAY_VISION_API_KEY_ENV"))
    if config["kind"] != "openai-compatible":
        raise ValueError("GPT vision requires an OpenAI-compatible provider")
    endpoint = str(config["endpoint"] or "").rstrip("/")
    selected_endpoint = None
    for suffix in ("/images/edits", "/responses", "/chat/completions", "/models"):
        if endpoint.endswith(suffix):
            if suffix in {"/responses", "/chat/completions"}:
                selected_endpoint = suffix
            endpoint = endpoint[:-len(suffix)]
            break
    key = provider_settings.resolve_credential(config["registry_id"], config["api_key_env"],
                                               config["credential_source"], endpoint)
    if not key:
        raise RuntimeError("vision provider credentials are unavailable")
    return config, endpoint, key, selected_endpoint


def _vision_models(payload: dict) -> list[str]:
    names = [str(item["id"]) for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]
    return [name for name in names if re.match(r"^gpt-(?:[456](?:[.\-]|$)|4o)", name, re.I)
            and not any(term in name.lower() for term in ("image", "audio", "realtime", "transcribe", "search", "spark"))]




def _model_rank(name: str):
    match = re.match(r"gpt-(\d+)(?:\.(\d+))?", name)
    major, minor = (int(match[1]), int(match[2] or 0)) if match else (4, 0)
    return major, minor, -int(any(term in name for term in ("mini", "luna", "nano"))), name


def _input_image(source: Path, out_dir: Path, *, input_profile='standard') -> tuple[Path, dict]:
    long_edge = _input_edge(input_profile)
    if is_raw(source):
        return reference_provider.prepare_api_input(source, out_dir, long_edge=long_edge)
    decoded = open_rgb(source)
    oriented_size = list(decoded.size)
    decoded.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    target = out_dir.resolve() / "inputs" / "vision-input.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    from raw_support import deterministic_srgb_profile
    profile = deterministic_srgb_profile()
    with target.open('xb') as stream:
        decoded.save(stream, "JPEG", quality=93, subsampling=0, icc_profile=profile)
    return target, {"path": str(target), "sha256": reference_provider.sha256(target),
        "bytes": target.stat().st_size, "format": "JPEG", "size": list(decoded.size),
        "derived": True, "source": {"path": str(source.resolve()), "sha256": reference_provider.sha256(source)},
        "decode": {"decode_version": RGB_DECODE_VERSION, "oriented_size": oriented_size,
                   "output_transfer": "IEC_sRGB", "long_edge_limit": long_edge}}


def _profile_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("vision model returned invalid profile JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("vision model must return one profile JSON object")
    return payload


def request_vision_json(source: Path, out_dir: Path, prompt: str, *, provider=None,
                        vision_model=None, api_base=None, api_url=None,
                        api_key_env=None, input_profile='standard') -> tuple[dict, dict]:
    """Send one visual recognition request; errors remain explicit and usable."""
    _input_edge(input_profile)
    started = perf_counter()
    timings = {'schema': 'vision_transport_timings.v1', 'clock': 'perf_counter',
        'image_prepare_seconds': 0.0, 'image_base64_seconds': 0.0,
        'request_binding_seconds': 0.0, 'profile_parse_seconds': 0.0, 'http_requests': []}
    binding = None
    try:
        source, out_dir = Path(source).resolve(), Path(out_dir).resolve()
        model = vision_model or os.environ.get("COSPLAY_VISION_MODEL")
        if "gpt-image" in str(model).lower():
            raise ValueError("gpt-image models cannot perform text vision recognition")
        if out_dir.exists() and any(path.is_file() for path in out_dir.rglob('*')):
            raise FileExistsError('Vision attempt evidence already exists; use a new independent directory')
        config, endpoint, key, selected_endpoint = _connection(provider=provider,
            api_base=api_base, api_url=api_url, api_key_env=api_key_env)
        observation = _HTTPObservation(out_dir, source=source, prompt=prompt,
            provider=config['registry_id'], key=key, connection_identity=_selected_connection_identity(
                provider, vision_model, api_base, api_url, api_key_env), input_profile=input_profile)
        def send(suffix, body=None):
            request_timings = {'protocol': suffix.lstrip('/')}
            timings['http_requests'].append(request_timings)
            try:
                attempt = observation.attempt(endpoint + suffix, suffix.lstrip('/'),
                    body.get('model') if body else None)
                result = _request(endpoint + suffix, key, body, timings=request_timings, observation=attempt)
                # Also inspect injected/non-streamed HTTP200 terminal envelopes.
                if _terminal_provider_failure(result):
                    raise VisionResponseError('provider_failure', 'vision response failed: '+_safe_detail(str(result.get('error') or 'failed'), key), response=result)
            except Exception:
                timings['failed_stage'] = 'http_request'
                raise
            return result
        discovered_models = []
        if not model:
            discovered_models = _vision_models(send('/models'))
            if not discovered_models:
                raise RuntimeError("provider /models returned no GPT vision text model; set COSPLAY_VISION_MODEL")
            model = max(discovered_models, key=_model_rank)
        if "gpt-image" in str(model).lower():
            raise ValueError("gpt-image models cannot perform text vision recognition")
        with _timed(timings, 'image_prepare_seconds'):
            source_before = reference_provider.sha256(source)
            image, image_meta = _input_image(source, out_dir, input_profile=input_profile)
            if image_meta['source']['sha256'] != source_before:
                raise ValueError('Vision source changed during input preparation')
        with _timed(timings, 'image_base64_seconds'):
            image_uri = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode("ascii")
        observation.api_input = image_meta
        final_prompt = _final_prompt(prompt)
        protocol = "responses" if selected_endpoint != "/chat/completions" else "chat/completions"
        fallback = None
        with _timed(timings, 'request_binding_seconds'):
            binding_path = out_dir/'request-binding.json'
            _exclusive_json(binding_path, {'schema': 'vision_request_binding.v1', 'source': _record(source),
                'input_profile': input_profile,
                'api_input': image_meta, 'prompt_sha256': hashlib.sha256(final_prompt.encode('utf-8')).hexdigest(),
                'allowed_protocols': ['responses', 'chat/completions'] if protocol == 'responses' else ['chat/completions']})
            binding = _record(binding_path)
        observation.image_binding = binding
        if protocol == "responses":
            body = {"model": model, "store": False, "stream": True,
                    "input": [{"role": "user", "content": [
                        {"type": "input_text", "text": final_prompt},
                        {"type": "input_image", "image_url": image_uri}]}]}
            try:
                response = send("/responses", body)
            except VisionHTTPError as exc:
                unsupported = exc.status in {405, 501} or (exc.status == 404
                    and "model" not in exc.detail.lower()) or (exc.status in {400, 422}
                    and "responses" in exc.detail.lower()
                    and any(term in exc.detail.lower() for term in ("unsupported", "not supported", "unknown endpoint")))
                if not unsupported:
                    raise
                fallback = {"from": "responses", "http_status": exc.status, "reason": "endpoint_unsupported"}
                protocol = "chat/completions"
                timings.pop('failed_stage', None)
        if protocol == "chat/completions":
            body = {"model": model, "messages": [{"role": "user", "content": [
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": {"url": image_uri}}]}]}
            response = send("/chat/completions", body)
        with _timed(timings, 'profile_parse_seconds'):
            text = _response_text(response, protocol)
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("vision model returned no profile text")
            try:
                payload = _profile_json(text)
            except RuntimeError as exc:
                if isinstance(exc.__cause__, json.JSONDecodeError) and _completed(response, protocol):
                    raise VisionResponseError('completed_invalid_json', 'completed vision model returned invalid profile JSON', response=response) from None
                raise
        provenance = {"schema_version": "vision_transport.v1", "provider": config["registry_id"],
            "model": model, "response_model": response.get("model"), "response_id": response.get("id"),
            "endpoint": endpoint + "/" + protocol, "protocol": protocol,
            "created_at": datetime.now(timezone.utc).isoformat(), "timeout_seconds": TIMEOUT,
            "source": {"path": str(source), "sha256": reference_provider.sha256(source)},
            "api_input": image_meta, "input_sha256": image_meta["sha256"],
            "input_profile": input_profile,
            "prompt_sha256": hashlib.sha256(final_prompt.encode("utf-8")).hexdigest(),
            "usage": response.get("usage"), "discovered_models": discovered_models,
            "fallback": fallback, "network_used": True, "generation_used": False,
            "http_observations": [_http_observation_locator(item.directory, item.binding_id)
                for item in observation.attempts],
            "docs_source": VISION_GUIDE}
    except Exception as exc:
        timings['total_seconds'] = perf_counter() - started
        exc.vision_timings = timings
        if isinstance(exc, VisionResponseError) and binding is not None:
            _save_response_failure(exc, out_dir, binding, protocol, key)
        raise
    timings['total_seconds'] = perf_counter() - started
    provenance['timings'] = timings
    return payload, provenance
