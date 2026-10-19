"""Verify RAW metadata avoids repeated LibRaw dimension probes."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import raw_support


def main() -> int:
    with tempfile.TemporaryDirectory(prefix='cosplay-raw-metadata-') as folder:
        root = Path(folder)
        source = root / 'source.arw'
        source.write_bytes(b'fixture-raw-container')
        cache = root / 'metadata-cache'
        previous = os.environ.get('COSPLAY_SOURCE_METADATA_CACHE')
        os.environ['COSPLAY_SOURCE_METADATA_CACHE'] = str(cache)
        try:
            raw_support._STORAGE_MEMO.clear()
            calls = []

            def probe(path):
                calls.append(Path(path))
                return (6000, 4000)

            with patch.object(raw_support, 'dimensions', side_effect=probe):
                first = raw_support.storage_info(source)
                raw_support._STORAGE_MEMO.clear()  # simulate a new worker
                second = raw_support.storage_info(source)
                assert first == second and len(calls) == 1
                source.write_bytes(b'changed-raw-container')
                raw_support._STORAGE_MEMO.clear()
                raw_support.storage_info(source)
                assert len(calls) == 2
        finally:
            raw_support._STORAGE_MEMO.clear()
            if previous is None:
                os.environ.pop('COSPLAY_SOURCE_METADATA_CACHE', None)
            else:
                os.environ['COSPLAY_SOURCE_METADATA_CACHE'] = previous
    print('{"ok": true, "raw_metadata_sidecar_reused": true, "stat_change_reprobed": true}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
