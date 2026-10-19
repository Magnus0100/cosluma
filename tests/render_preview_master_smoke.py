"""Check shared display encoding, precision, profiles and independent reopen."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import tifffile

import display_transform
import float_appearance
from raw_support import (RGB_DECODE_VERSION, deterministic_srgb_profile, linear_preview,
                         save_render_master, save_render_preview_and_master)


REOPEN = """
import hashlib, json, sys
from PIL import Image
import tifffile
with tifffile.TiffFile(sys.argv[1]) as resource:
    page = resource.pages[0]
    values = page.asarray()
    master = {'shape': list(values.shape), 'dtype': str(values.dtype),
              'sha256': hashlib.sha256(values.tobytes()).hexdigest(),
              'icc_sha256': hashlib.sha256(page.tags[34675].value).hexdigest(),
              'rows_per_strip': page.rowsperstrip}
with Image.open(sys.argv[2]) as image:
    preview = {'mode': image.mode, 'size': list(image.size),
               'sha256': hashlib.sha256(image.tobytes()).hexdigest(),
               'icc_sha256': hashlib.sha256(image.info['icc_profile']).hexdigest()}
print(json.dumps({'master': master, 'preview': preview}))
"""


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def main():
    rng = np.random.default_rng(20261014)
    # Cross strip boundaries and include transfer thresholds and gamut clipping.
    values = rng.uniform(-.02, 1.2, (513, 37, 3)).astype(np.float32)
    values[0, :8, :] = np.asarray([-.01, 0., 1e-6, .0031308, .04045, .18, 1., 1.2], np.float32)[:, None]
    before = values.copy()
    profile = deterministic_srgb_profile()
    cases = []
    transforms = [('untagged_transform', None), ('display_clip', {'method': 'clip_sdr'}),
                  ('scene_shoulder', dict(display_transform.DEFAULT))]
    with tempfile.TemporaryDirectory(prefix='cosplay-shared-encode-') as directory:
        root = Path(directory)
        for name, transform in transforms:
            old_preview = linear_preview(values, display_transform=transform)
            old_path, new_path, png = root/(name+'-old.tif'), root/(name+'-new.tif'), root/(name+'.png')
            assert save_render_master(values, old_path, display_transform=transform) is None
            observed = []
            encoder = float_appearance.linear_to_srgb if transform is None else display_transform.encode_display

            def observe(strip, *args):
                observed.append(strip.shape)
                return encoder(strip, *args)

            owner, attribute = (float_appearance, 'linear_to_srgb') if transform is None else (display_transform, 'encode_display')
            with patch.object(owner, attribute, observe):
                new_preview = save_render_preview_and_master(values, new_path, display_transform=transform)
            assert observed == [(256, 37, 3), (256, 37, 3), (1, 37, 3)], observed
            assert new_preview.mode == 'RGB' and new_preview.size == (37, 513)
            assert new_preview.tobytes() == old_preview.tobytes()
            assert new_preview.info == old_preview.info
            assert new_preview.info['decode_version'] == RGB_DECODE_VERSION
            old_pixels = tifffile.imread(old_path)
            with tifffile.TiffFile(new_path) as resource:
                page = resource.pages[0]
                new_pixels = page.asarray()
                assert new_pixels.dtype == np.uint16 and np.array_equal(new_pixels, old_pixels)
                assert page.tags[34675].value == profile and page.rowsperstrip == 256
            assert new_path.read_bytes() == old_path.read_bytes()
            new_preview.save(png, icc_profile=profile, compress_level=1)
            env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
            reopened = json.loads(subprocess.run([sys.executable, '-B', '-c', REOPEN, str(new_path), str(png)],
                check=True, capture_output=True, text=True, env=env).stdout)
            assert reopened['master']['sha256'] == digest(old_pixels)
            assert reopened['master']['dtype'] == 'uint16'
            assert reopened['master']['shape'] == [513, 37, 3]
            assert reopened['master']['rows_per_strip'] == 256
            assert reopened['preview']['sha256'] == hashlib.sha256(old_preview.tobytes()).hexdigest()
            expected_icc = hashlib.sha256(profile).hexdigest()
            assert reopened['master']['icc_sha256'] == reopened['preview']['icc_sha256'] == expected_icc
            cases.append({'transform': name, 'encodes': len(observed), 'preview_exact': True,
                          'uint16_exact': True, 'tiff_bytes_exact': True, 'independent_reopen': True})
    assert np.array_equal(values.view(np.uint32), before.view(np.uint32))
    print(json.dumps({'schema': 'render_preview_master_smoke.v1', 'passed': True, 'cases': cases,
                      'input_unchanged': True, 'models_or_network_used': False}))


if __name__ == '__main__':
    main()
