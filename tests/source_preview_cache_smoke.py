"""Verify strict source-preview execution caching and invalidation."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from PIL import Image


def main():
    import source_preview
    from source_preview import SourcePreviewCache

    with tempfile.TemporaryDirectory(prefix='cosplay-source-preview-cache-') as folder:
        root = Path(folder)
        source = root / 'source.png'
        Image.new('RGB', (32, 24), (210, 180, 130)).save(source)
        run = {
            'kind': 'single_input_run', 'input_mode': 'single_source',
            'run_id': 'preview-cache', 'sample': 'preview-cache', 'revision': 1,
            'root': str(root),
            'source': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                       'oriented_size': [32, 24]},
            'reference': None,
            'options': {'render_mode': 'native_source_design'},
        }
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps(run), encoding='utf8')
        cache = SourcePreviewCache(root, max_edge=16)
        calls = []
        original = source_preview.render_source_only_float

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        source_preview.render_source_only_float = counted
        try:
            first = cache.preview('preview-cache', run, manifest,
                                  {'request_id': 'first', 'appearance_revision': {'params': {'exposure': .1}}})
            second = cache.preview('preview-cache', run, manifest,
                                   {'request_id': 'second', 'appearance_revision': {'params': {'exposure': .1}}})
            changed = cache.preview('preview-cache', run, manifest,
                                    {'request_id': 'changed', 'appearance_revision': {'params': {'exposure': .2}}})
            assert len(calls) == 2, f'unexpected renderer calls after parameter edit: {len(calls)}'
            assert first['cache_hit'] is False and first['preview_cache_hit'] is False
            assert second['cache_hit'] is True and second['preview_cache_hit'] is True
            assert second['session_cache_hit'] is True
            assert changed['cache_hit'] is False and changed['preview_cache_hit'] is False
            assert changed['session_cache_hit'] is True
            assert first['preview_path'] != changed['preview_path']

            # Replacing the source invalidates both the execution artifact and
            # the in-memory proxy session, even when the manifest is untouched.
            Image.new('RGB', (32, 24), (40, 60, 90)).save(source)
            mutated = dict(run)
            mutated['source'] = dict(run['source'])
            mutated['source']['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            source_preview.render_source_only_float = counted
            replaced = cache.preview('preview-cache', mutated, manifest,
                                     {'request_id': 'replaced', 'appearance_revision': {'params': {'exposure': .1}}})
            assert len(calls) == 3
            assert replaced['cache_hit'] is False and replaced['preview_cache_hit'] is False
            assert replaced['session_cache_hit'] is False

            # A renderer identity change must never consume a previous JPEG.
            old_identity = source_preview._renderer_identity
            source_preview._renderer_identity = lambda: {'engine': 'test-changed', 'code': {}}
            try:
                renderer_changed = cache.preview('preview-cache', mutated, manifest,
                                                 {'request_id': 'renderer', 'appearance_revision': {'params': {'exposure': .1}}})
            finally:
                source_preview._renderer_identity = old_identity
            assert len(calls) == 4
            assert renderer_changed['cache_hit'] is False
        finally:
            source_preview.render_source_only_float = original
    print('OK: exact preview reuse; parameter invalidation; source replacement invalidation; renderer identity invalidation')


if __name__ == '__main__':
    main()
