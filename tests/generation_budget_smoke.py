"""Offline checks for paid submission limits; never contacts a provider."""
from pathlib import Path
from unittest.mock import patch
import io
import json
import tempfile
import urllib.error

from PIL import Image
import reference_provider
import single_source_workflow as workflow


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay_budget_') as folder:
        root = Path(folder)
        source = root / 'source.png'
        Image.new('RGB', (24, 32), '#806050').save(source)
        for failure in (TimeoutError('response lost'), urllib.error.HTTPError(
                'https://example.invalid', 429, 'limit', {}, io.BytesIO(b'limited'))):
            events = []
            with patch('reference_provider.urllib.request.urlopen', side_effect=failure) as transport:
                try:
                    reference_provider.request_edit('https://example.invalid', 'fixture-key', source,
                        'fixture-model', 'test', None, 1,
                        retry={'max_attempts': 8, 'backoff_seconds': 1, 'retry_statuses': [429]}, events=events)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('failure should be reported')
                assert transport.call_count == 1
                attempts = [event for event in events if event.get('event') == 'provider_attempt']
                assert len(attempts) == 1 and attempts[0]['retrying'] is False

        # New creation requires a checked source-native task. No old partial
        # raw-generation resume may be reached by a direct worker call.
        args=workflow.parse_args(['--root',str(root),'--source',str(source),'--generate'])
        with patch('photo_edit_finish.initial_candidates',side_effect=AssertionError('Unchecked request reached model')):
            try:workflow.create_run(args)
            except ValueError as error:assert 'checked' in str(error)
            else:raise AssertionError('Unchecked creation bypassed task plan')
    print(json.dumps({'ok':True,'paid_calls':0,'timeout_and_429_not_retried':True,'unchecked_creation_rejected':True}))


if __name__=='__main__':main()
