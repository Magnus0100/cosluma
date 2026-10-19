"""The final background plate supplies bounded local light to source RGB."""
from __future__ import annotations

import numpy as np

from source_environment_light import apply, estimate


def main():
    h, w = 64, 80
    background = np.full((h, w, 3), .12, np.float32)
    background[20:44, 42:68] = [.65, .38, .20]
    validity = np.ones((h, w), np.float32)
    alpha = np.zeros((h, w), np.float32)
    alpha[12:54, 24:58] = 1
    gain, report = estimate(background, validity, alpha)
    assert report['schema'] == 'source_environment_light.v1'
    assert report['candidate_subject_RGB_used'] is False
    assert gain.shape == background.shape and np.isfinite(gain).all()
    assert gain.min() >= 1 and gain.max() <= 1.6
    source = np.full_like(background, .18)*alpha[..., None]
    result = apply(source, gain, alpha)
    assert result.shape == source.shape and np.isfinite(result).all()
    assert np.all(result[alpha == 0] == 0), 'transparent foreground must remain zero'
    assert np.all(result >= 0) and np.all(result <= alpha[..., None]+1e-6)
    assert np.any(np.abs(result[alpha == 1] - source[alpha == 1]) > 1e-5), 'local background light did not reach subject'
    print({'ok': True, 'schema': report['schema'], 'changed_pixels': int(np.count_nonzero(result != source)),
           'candidate_subject_RGB_used': report['candidate_subject_RGB_used'], 'alpha_modified': report['alpha_modified']})


if __name__ == '__main__':
    main()
