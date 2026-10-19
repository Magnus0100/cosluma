"""Candidate relighting evidence never becomes candidate foreground texture."""
from __future__ import annotations

import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import hashlib

from candidate_subject_light import apply, estimate, estimate_from_reference


def main():
    h, w = 96, 112
    source = np.full((h, w, 3), .18, np.float32)
    source[18:78, 24:88] = [.24, .20, .18]
    alpha = np.zeros((h, w), np.float32)
    alpha[12:84, 18:94] = 1
    candidate = source.copy()
    # A broad warm key on the left and a cool fill on the right.  The field
    # should retain the low-frequency evidence while ignoring checker detail.
    candidate[20:76, 20:55] = [.52, .26, .12]
    candidate[20:76, 55:90] = [.12, .18, .38]
    candidate_mask = alpha.copy()
    gain, report = estimate(candidate, source, candidate_mask, alpha)
    assert report['schema'] == 'candidate_subject_light.v1'
    assert report['candidate_texture_imported'] is False
    assert report['candidate_geometry_imported'] is False
    assert gain.shape == source.shape and np.isfinite(gain).all()
    result = apply(source*alpha[..., None], gain, alpha)
    assert np.all(result[alpha == 0] == 0)
    assert np.all(result >= 0) and np.all(result <= alpha[..., None]+1e-6)
    assert np.any(np.abs(result[alpha == 1] - source[alpha == 1]) > 1e-5)
    assert report['supported_pixels'] > 0
    with tempfile.TemporaryDirectory(prefix='candidate-subject-light-') as temporary:
        root = Path(temporary)
        reference_path = root/'reference.png'
        Image.fromarray(np.uint8(np.clip(candidate, 0, 1)*255+.5), 'RGB').save(reference_path)
        occupancy_path = root/'occupancy.png'
        Image.fromarray(np.uint8(candidate_mask*255+.5), 'L').save(occupancy_path)
        reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        occupancy_sha = hashlib.sha256(occupancy_path.read_bytes()).hexdigest()
        native_gain, native_report = estimate_from_reference(
            reference_path, {'path': str(occupancy_path), 'sha256': occupancy_sha},
            source*alpha[..., None], alpha, source_proxy_size=(w, h),
            expected_reference_sha256=reference_sha)
        assert native_gain.shape == source.shape
        assert native_report['reference_sha256'] == reference_sha
        assert native_report['candidate_texture_imported'] is False
    print({'ok': True, 'schema': report['schema'],
           'supported_pixels': report['supported_pixels'],
           'candidate_texture_imported': report['candidate_texture_imported'],
           'alpha_modified': report['alpha_modified']})


if __name__ == '__main__':
    main()
