"""Reused native archives bind actual bytes across link/copy publication races."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np

from asset_identity import sha256_file
from native_array_storage import save_native_array
from native_source_baseline import _verify_native_rgb_archive
import reference_appearance_prepare as preparation


def reject(function):
    try:
        function()
    except (ValueError, OSError):
        return
    raise AssertionError('A changed reused native publication was accepted')


def main():
    values = np.arange(17*31*3, dtype=np.float32).reshape(17, 31, 3)/2048
    cases = []
    with tempfile.TemporaryDirectory(prefix='native-materialization-') as temporary:
        root = Path(temporary)
        for mode in ('link', 'copy'):
            for typed in (False, True):
                for race in ('none', 'source_replace', 'target_replace', 'target_replace_after_SHA'):
                    directory = root/(mode+'-'+str(typed)+'-'+race)
                    directory.mkdir()
                    source, target = directory/'source.npz', directory/'target.npz'
                    save_native_array(source, values)
                    reused = {'path': str(source), 'sha256': sha256_file(source, force=True),
                        'size': [31, 17], 'representation': 'straight_RGB'}
                    before = copy.deepcopy(reused)
                    proof = _verify_native_rgb_archive(reused, values.shape) if typed else None
                    # A previous target identity must not override the new
                    # full-byte publication check, including on a retry.
                    target.write_bytes(source.read_bytes())
                    sha256_file(target, force=True)
                    replacement = directory/'replacement.npz'
                    save_native_array(replacement, values*.5)
                    source_stat = source.stat()
                    os.utime(replacement, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
                    actual_materialize = os.link if mode == 'link' else shutil.copy2
                    def materialize(source_path, target_path):
                        if race == 'source_replace':
                            replacement.replace(source_path)
                        result = actual_materialize(source_path, target_path)
                        if race == 'target_replace':
                            replacement.replace(target_path)
                        return result
                    hashes = []
                    def actual_hash(path, **kwargs):
                        assert kwargs.get('force') is True
                        digest = sha256_file(path, **kwargs)
                        hashes.append((str(path), digest))
                        if race == 'target_replace_after_SHA':
                            replacement.replace(path)
                        return digest
                    from contextlib import ExitStack
                    with ExitStack() as patches:
                        if mode == 'link':
                            patches.enter_context(patch.object(os, 'link', side_effect=materialize))
                        else:
                            patches.enter_context(patch.object(os, 'link', side_effect=OSError('copy fallback')))
                            patches.enter_context(patch.object(shutil, 'copy2', side_effect=materialize))
                        patches.enter_context(patch.object(preparation, 'sha256_file', side_effect=actual_hash))
                        publish = lambda: preparation._materialize_reused_array_asset(target, reused, verified=proof)
                        if race == 'none':
                            result = publish()
                            assert result == {**reused, 'path': str(target)}
                            assert sha256_file(target, force=True) == reused['sha256']
                            assert len(hashes) == 1
                        else:
                            reject(publish)
                    assert reused == before
                    cases.append({'mode': mode, 'typed_descriptor': typed, 'race': race,
                        'passed': True, 'actual_target_full_SHA_calls': len(hashes)})
    print(json.dumps({'schema': 'native_asset_materialization_smoke.v1', 'passed': True,
        'cases': cases, 'models_or_GPU_used': False}))


if __name__ == '__main__':
    main()
