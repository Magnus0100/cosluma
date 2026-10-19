"""The default native compositor consumes candidate light evidence before over."""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import design_reconstruction as design  # noqa: E402
from native_subject_detail_runtime_smoke import fixture  # noqa: E402


def _asset(path, size):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size': list(size), 'coordinate_space': 'candidate_proxy', 'representation': 'segmentation'}


def main():
    with tempfile.TemporaryDirectory(prefix='design-candidate-light-') as temporary:
        root = Path(temporary)
        run, target, _, alpha, background = fixture(root/'origin')
        plan = design.plan_for_run(run)
        plan['execution']['source_environment_light'] = True
        occupancy_path = root/'candidate-occupancy.png'
        Image.fromarray(np.uint8(alpha*255+.5), 'L').save(occupancy_path)
        plan['mask_roles']['candidate_occupancy'] = _asset(occupancy_path, [48, 32])
        plan['canvas']['candidate_mapping'] = {'candidate_processing_proxy_to_final_proxy': np.eye(3).tolist()}
        geometry = Image.fromarray(np.uint8(alpha*255))
        with patch('reference_appearance_stack.validate_layer', return_value={}):
            result, summary = design.apply_plan(np.zeros_like(target), np.zeros_like(target),
                                                plan, geometry, run=run, include_optics=False)
        evidence = summary['candidate_subject_light']
        assert evidence['schema'] == 'candidate_subject_light.v1'
        assert evidence['candidate_texture_imported'] is False
        assert evidence['candidate_geometry_imported'] is False
        assert evidence['alpha_modified'] is False
        assert evidence['applied'] is True
        assert np.isfinite(result).all()
        assert np.all(result >= 0)
        print({'ok': True, 'schema': evidence['schema'],
               'observed_overlap_pixels': evidence['observed_overlap_pixels'],
               'supported_pixels': evidence['supported_pixels'],
               'candidate_texture_imported': evidence['candidate_texture_imported'],
               'alpha_modified': evidence['alpha_modified']})


if __name__ == '__main__':
    main()
