"""CPU proof of Sapiens pinned identity reuse, explicit audits and refusal."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import asset_identity
import sapiens_model_profiles as profiles


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay-sapiens-identity-') as temporary:
        root = Path(temporary)
        checkpoint = root/'checkpoint.safetensors'
        original = b'pinned-checkpoint' * 512
        checkpoint.write_bytes(original)
        expected = hashlib.sha256(original).hexdigest()
        name = 'fixture_pinned_segmentation'
        fixture = {**profiles.profile('sapiens2_seg_0.4b', 'segmentation'),
                   'checkpoint_sha256': expected}
        with patch.dict(os.environ, {'COSPLAY_ASSET_IDENTITY_CACHE': str(root/'identities'),
                                    'COSPLAY_DISABLE_PERSISTENT_ASSET_IDENTITY': '0'}), \
             patch.dict(profiles.PROFILES, {name: fixture}):
            asset_identity.clear_cache()
            item, observed = profiles.verified_checkpoint(checkpoint, name, 'segmentation')
            assert item == fixture and observed == expected
            assert asset_identity.cache_stats()['misses'] == 1

            # Restart-like reuse must not open the checkpoint bytes again.
            asset_identity.clear_cache()
            original_open = Path.open
            def refuse_byte_read(path, mode='r', *args, **kwargs):
                if path == checkpoint and mode == 'rb':
                    raise AssertionError('Unexpected checkpoint byte read')
                return original_open(path, mode, *args, **kwargs)
            with patch.object(Path, 'open', refuse_byte_read):
                assert profiles.verified_checkpoint(checkpoint, name, 'segmentation')[1] == expected
            assert asset_identity.cache_stats() == {'hits': 1, 'misses': 0, 'entries': 1}

            # An explicit audit must read the bytes even with a valid sidecar.
            assert profiles.verified_checkpoint(checkpoint, name, 'segmentation', force=True)[1] == expected
            assert asset_identity.cache_stats()['misses'] == 1

            # A same-size content edit invalidates the identity and must be
            # refused against the original pin, rather than silently accepted.
            old_stat = checkpoint.stat()
            checkpoint.write_bytes(b'X' + original[1:])
            os.utime(checkpoint, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000_000))
            try:
                profiles.verified_checkpoint(checkpoint, name, 'segmentation')
            except ValueError as error:
                assert 'bytes do not match pinned profile' in str(error)
            else:
                raise AssertionError('Changed checkpoint accepted against the original pin')
            assert asset_identity.cache_stats()['misses'] == 2
            try:
                profiles.verified_checkpoint(checkpoint, name, 'matting')
            except ValueError as error:
                assert 'does not support task' in str(error)
            else:
                raise AssertionError('Segmentation checkpoint accepted for another task')
            asset_identity.clear_cache()
    print(json.dumps({'ok': True, 'unchanged_checkpoint_read': False,
        'explicit_force_audit': True, 'same_size_edit_refused': True,
        'wrong_task_refused': True, 'gpu_used': False}))


if __name__ == '__main__':
    main()
