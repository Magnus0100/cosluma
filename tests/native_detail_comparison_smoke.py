"""Saved detail compares identical native coordinates without a RAW decode."""
from pathlib import Path
import json
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

from current_stack_fixture import prepare_fixture
from native_detail import saved_native_detail
from raw_support import linear_preview, save_scene_master


def main():
    with tempfile.TemporaryDirectory(prefix='native-detail-comparison-') as temporary:
        root = Path(temporary)
        run = prepare_fixture(root/'assets', native=(256, 384))
        run.update(revision=3, status='needs_visual_review')
        yy, xx = np.mgrid[:384, :256]
        final = np.stack((xx/512, yy/768, (xx+yy)/1024), -1).astype(np.float32)
        master = root/'render_scene.tif'
        save_scene_master(final, master, working_domain='display_linear_sRGB')
        run['artifacts'] = {'render_scene_master': str(master)}
        run['render'] = {'operator': {'working_domain': 'display_linear_sRGB'}}
        path = root/'outputs/runs'/run['run_id']/'manifest.json'
        path.parent.mkdir(parents=True); path.write_text(json.dumps(run), 'utf8')
        before = path.read_bytes()
        body = {'center': [.37, .61], 'edge': 128, 'include_source': True}
        with patch('raw_support.open_linear', side_effect=AssertionError('Full RAW decode is not a native crop')):
            value = saved_native_detail(root, run['run_id'], body)
        l,t,r,b = value['crop_box']
        expected_final = linear_preview(final[t:b,l:r], display_transform={'method': 'clip_sdr'})
        assert Image.open(value['preview_path']).tobytes() == expected_final.tobytes()
        source = Image.open(run['source']['path']).convert('RGB').crop((l,t,r,b))
        assert Image.open(value['source_preview_path']).tobytes() == source.tobytes()
        assert value['source_comparison']['crop_box'] == value['crop_box']
        assert not value['source_comparison']['resampled'] and not value['raw_decoded']
        assert value['revision'] == 3 and path.read_bytes() == before
        old_api = saved_native_detail(root, run['run_id'], {'edge':128})
        assert 'source_preview_path' not in old_api
        asset = run['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']['assets']['old_environment']
        Path(asset['path']).write_bytes(b'tampered')
        try:
            saved_native_detail(root, run['run_id'], body)
        except ValueError as error:
            assert 'baseline changed' in str(error)
        else:
            raise AssertionError('Changed source comparison asset was displayed')
    print(json.dumps({'ok':True,'same_native_coordinates':True,
        'source_and_saved_final_pixels_checked':True,'no_RAW_decode':True,
        'tampered_baseline_rejected':True,'saved_manifest_unchanged':True}))


if __name__ == '__main__':
    main()
