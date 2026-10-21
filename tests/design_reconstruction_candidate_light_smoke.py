"""The default native compositor consumes candidate light evidence before over,
and its joint light arm keeps its photographed-environment anchor honest.

The delivery path re-samples the plan's own immutable paired-source plate as the
anchor of the cross-boundary solve. These checks prove that wire for real: the
anchor handed to the solve is compared with the plate itself, and an anchor this
run cannot support must be recorded and must not move a single delivered pixel
(the fallback render is compared byte-for-byte with the same plan solved without
any anchor). Injection happens at the solver the delivery path imports, so the
sampling and the argument passing still execute exactly as production does.
"""
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
import source_environment_light as environment_light  # noqa: E402
from native_subject_detail_runtime_smoke import fixture  # noqa: E402

JOINT = 'joint_photographed_environment_change_two_way'
LOCAL = 'local_environment_field_two_way'


def _asset(path, size):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size': list(size), 'coordinate_space': 'candidate_proxy', 'representation': 'segmentation'}


def _digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, np.float32).tobytes()).hexdigest()


def _render(plan, geometry, run, total, arm):
    """Render the delivery path once; record what each solve call really used.

    ``arm`` selects how the solver behaves: ``healthy`` solves with whatever
    anchor it is handed, ``rejecting`` refuses an anchor exactly where the
    delivery path would use it (the run cannot support this anchor), and
    ``no_anchor`` is the pre-t9 arm that asks the solve without an anchor.
    """
    real_estimate = environment_light.estimate
    used = []

    def solver(*args, **kwargs):
        anchor = kwargs.get('source_environment')
        if arm == 'no_anchor':
            used.append(None)
            kwargs.pop('source_environment', None)
            return real_estimate(*args, **kwargs)
        if anchor is not None and arm == 'rejecting':
            used.append(np.array(anchor, np.float32, copy=True))
            raise ValueError('synthetic anchor outside the supported environment window')
        used.append(None if anchor is None else np.array(anchor, np.float32, copy=True))
        return real_estimate(*args, **kwargs)

    with patch('reference_appearance_stack.validate_layer', return_value={}), \
            patch('source_environment_light.estimate', side_effect=solver):
        values, summary = design.apply_plan(np.zeros_like(total), np.zeros_like(total),
                                            plan, geometry, run=run, include_optics=False)
    return values, summary, used


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
        result, summary, delivered_anchors = _render(plan, geometry, run, target, 'healthy')
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

        # (1) With an anchor the delivered solve is the joint photographed-
        # environment change, anchored by the plan's own paired-source plate.
        height, width = target.shape[:2]
        plate = np.asarray(design.sample_asset(
            plan['source_baseline']['assets']['old_environment'], 'values',
            (0, 0, width, height), (width, height)), np.float32)
        delivered = summary['source_environment_light']
        assert len(delivered_anchors) == 1, delivered_anchors
        assert delivered_anchors[0] is not None
        assert np.array_equal(delivered_anchors[0], plate), 'the solve did not receive the paired source plate'
        assert delivered['source_environment_supplied'] is True, delivered
        assert delivered['solve'] == JOINT, delivered
        assert delivered['source_environment_bypass_reason'] is None, delivered
        assert delivered['source_environment_asset'] == \
            plan['source_baseline']['assets']['old_environment']['sha256'], delivered
        # The joint branch really solved the photographed change.  In this
        # synthetic fixture the photographed plate and the designed plate are the
        # same values, so the measured change is zero and the joint gain happens
        # to coincide with the local one; the branch is proven by its own field
        # and numbers, never by an arbitrary pixel delta.
        assert delivered['field_source'] == 'photographed_to_final_environment_change', delivered
        assert np.asarray(delivered['environment_change_ev_global'], np.float64).shape == (3,)
        assert np.isfinite(np.asarray(delivered['environment_change_ev_global'], np.float64)).all()
        assert np.isfinite(float(delivered['joint_dose'])), delivered

        # (2) An unusable anchor is recorded, the solve is re-run without it, and
        # the delivered pixels stay exactly where the no-anchor arm leaves them.
        fallback, fallback_summary, fallback_anchors = _render(plan, geometry, run, target, 'rejecting')
        recorded = fallback_summary['source_environment_light']
        assert len(fallback_anchors) == 2, fallback_anchors
        assert fallback_anchors[0] is not None and fallback_anchors[1] is None, \
            'the fallback must re-solve without the anchor instead of silently keeping it'
        assert recorded['source_environment_supplied'] is False, recorded
        assert str(recorded['source_environment_bypass_reason']).startswith(
            'source_environment_rejected:'), recorded
        assert recorded['source_environment_asset'] is None, recorded
        assert recorded['solve'] == LOCAL, recorded
        assert recorded['field_source'] == 'validity_weighted_final_background_low_frequency', recorded
        assert 'environment_change_ev_global' not in recorded, \
            'the fallback must be the local solve, not a joint solve with a hidden anchor'

        control, control_summary, control_anchors = _render(plan, geometry, run, target, 'no_anchor')
        assert control_anchors == [None], control_anchors
        assert control_summary['source_environment_light']['solve'] == LOCAL
        assert _digest(fallback) == _digest(control), \
            'a rejected anchor changed the delivered pixels'
        print({'ok': True, 'anchor_supplied': delivered['source_environment_supplied'],
               'solve': delivered['solve'], 'gate_match': True,
               'rejected_bypass': str(recorded['source_environment_bypass_reason'])[:60],
               'rejected_equals_no_anchor': True,
               'delivered_sha256': _digest(result)[:16],
               'fallback_sha256': _digest(fallback)[:16],
               'control_sha256': _digest(control)[:16]})


if __name__ == '__main__':
    main()
