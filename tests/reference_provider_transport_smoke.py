"""Current image transport counts submissions and keeps download credentials local."""
from __future__ import annotations

import base64
from email.message import Message
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image

import reference_provider as provider
import workflow_timing


class Response(io.BytesIO):
    def __init__(self, body: bytes, content_type: str):
        super().__init__(body)
        self.status = 200
        self.headers = Message()
        self.headers['Content-Type'] = content_type


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay-provider-transport-') as directory:
        source = Path(directory)/'source.png'
        Image.new('RGB', (96, 64), 'steelblue').save(source)
        image_bytes = source.read_bytes()
        for mode in ('url', 'b64_json'):
            annotations, calls = {}, []
            data = ({'url': 'https://signed.example/returned.png'} if mode == 'url'
                    else {'b64_json': base64.b64encode(image_bytes).decode('ascii')})

            def fake_open(request, **kwargs):
                calls.append(request)
                if request.get_method() == 'POST':
                    assert request.get_header('Authorization') == 'Bearer test-key'
                    return Response(json.dumps({'data': [data]}).encode(), 'application/json')
                assert request.get_method() == 'GET'
                assert request.get_header('Authorization') is None
                return Response(image_bytes, 'image/png')

            with (patch.object(provider.urllib.request, 'urlopen', side_effect=fake_open),
                  patch.object(workflow_timing, 'annotate',
                               side_effect=lambda **fields: annotations.update(fields))):
                _, actual, response_mode = provider.request_edit(
                    'https://provider.example/v1/images/edits', 'test-key', source,
                    'configured-current-model', 'exact brief', None, 1)
            assert actual == image_bytes and response_mode == mode
            assert annotations['model_submissions'] == 1
            assert annotations['http_submissions'] == len(calls) == (2 if mode == 'url' else 1)
    print(json.dumps({'ok': True, 'presigned_get_counted': True,
                      'download_without_provider_credentials': True, 'real_model_requests': 0}))


if __name__ == '__main__':
    main()
