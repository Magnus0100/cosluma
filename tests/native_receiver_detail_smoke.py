"""CPU counterexamples for reviewed receiver support, overlap and source detail."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter
from native_receiver_detail import (prepare_receiver_detail, qualified_review_windows,
    validate_diagnosed_plate, reviewed_floor_continuation)
from native_receiver_preservation import LUMA, sha


def main():
    floor = np.zeros((64,80),bool);floor[48:]=True
    footwear=np.zeros_like(floor,dtype=np.float32);footwear[10:30,20:60]=1
    alpha=footwear.copy();alpha[30:48,20:60]=.1
    labels=np.zeros_like(floor,dtype=np.int32);labels[10:30,20:60]=9
    extra,_=reviewed_floor_continuation(floor,footwear,alpha,alpha,labels,pitch=2)
    assert extra[30:48,20:60].all() and not np.any(extra & (alpha>=.5))
    for kind in ('no_floor','far_floor','single_floor_pixel','foreground_barrier','unknown_shoe'):
        f=floor.copy();a=alpha.copy();sem=labels.copy()
        if kind=='no_floor':f[:]=False
        if kind=='far_floor':f[:]=False;f[60:]=True
        if kind=='single_floor_pixel':f[:]=False;f[48,40]=True
        if kind=='foreground_barrier':a[35,:]=.8
        if kind=='unknown_shoe':sem[:]=0
        assert not reviewed_floor_continuation(f,footwear,a,a,sem,pitch=2)[0].any(),kind
    with tempfile.TemporaryDirectory(prefix='cosplay_receiver_detail_') as folder:
        root = Path(folder)
        native = [128, 96]
        yy, xx = np.mgrid[:96, :128]
        texture = .02*np.sin(xx*.7)*np.cos(yy*.6)
        curve = .06*np.exp(-((yy-(55+.006*(xx-64)**2))/1.8)**2)
        source = np.repeat((.12+texture+curve)[..., None], 3, axis=-1).astype(np.float32)
        candidate = np.repeat(gaussian_filter(source[..., 0], 3)[..., None], 3, axis=-1)
        candidate *= np.array([1.2, 1., .8], np.float32)
        alpha = np.zeros((96, 128), np.float32); alpha[20:46, 45:82] = 1
        alpha[46, 45:82] = .3
        # A learned floor remains usable across nonzero matte tails. A thin
        # tail across the reflection must not cut a hole in its native detail.
        alpha[58:61, 30:105] = 1e-4
        alpha[70:72, 90:95] = .7
        observed = alpha.copy(); observed[62:65, 55:58] = .2
        floor = np.ones_like(alpha); floor[:30] = 0
        joint = floor.copy(); joint[:, :15] = 0
        validity = np.ones_like(alpha); validity[75:80, 65:72] = .8

        def asset(name, values, space='source_canvas', role=None, representation=None):
            path = root/(name+'.npz'); np.savez_compressed(path, values=values)
            item = {'path': str(path), 'sha256': sha(path), 'size': native.copy(),
                    'coordinate_space': space, 'original_canvas_size': native.copy()}
            if role: item['role'] = role
            if representation: item['representation'] = representation
            return item

        old = asset('source', source)
        a = asset('alpha', alpha); oa = asset('observed-alpha', observed)
        bg = asset('background', candidate*validity[..., None], representation='sampling_validity_associated_RGB')
        v = asset('validity', validity)
        j = asset('joint', joint, 'final_source_proxy', 'joint_source_candidate_receiver_support')
        f = asset('floor', floor, 'final_source_proxy', 'learned_source_receiver_support')
        plan = {'working_domain': 'display_linear_sRGB', 'source_sha256': 's', 'candidate_sha256': 'r',
            'canvas': {'source_oriented_size': native, 'source_proxy_size': native,
                'candidate_mapping': {'scene_registration_supported': True,
                                      'candidate_original_to_final_original': np.eye(3).tolist()}},
            'source_baseline': {'working_domain': 'display_linear_sRGB', 'assets': {
                'old_environment': old, 'foreground_alpha': oa}}, 'mask_roles': {'foreground_alpha': a},
            'layers': [{'kind': 'background_design', 'assets': {'background': bg, 'validity': v}}]}
        run = {'source': {'sha256': 's'}, 'reference': {'sha256': 'r'},
               'options': {'appearance_revision': {'design_reconstruction_plan': plan}}}
        rows = []
        for index, box in enumerate(([30, 38, 88, 84], [60, 40, 105, 82])):
            rows.append({'pair_id': ['left_shoe_contact', 'right_shoe_contact'][index],
                'region': ['left_shoe_contact', 'right_shoe_contact'][index],
                'scope': 'matched_source_final_native_pair_only', 'native_box': box,
                'source_panel': 's'+str(index), 'final_panel': 'f'+str(index),
                'checks': {'identity_makeup': {'status': 'pass'}, 'skin_materials': {'status': 'pass'},
                    'native_detail': {'status': 'fail', 'cause': 'retouch_missing', 'correction_stage': 'native_retouch',
                        'evidence': [{'basis': 'visible_pixels', 'panels': ['s'+str(index), 'f'+str(index)]}]}}})
        run['cosplay_post_review'] = {'status': 'reviewed', 'actual_api_evidence_valid': True, 'local_observations': rows}
        plan['source_baseline']['assets']['foreground_premultiplied'] = old
        run.update(run_id='fixture', revision=1)
        run['cosplay_post_review']['native_selection'] = {'source_baseline_assets': copy.deepcopy(plan['source_baseline']['assets'])}
        metadata = {'run_id': 'fixture', 'revision': 1, 'operator': {'design_reconstruction_plan': {
            'working_domain': 'display_linear_sRGB', 'foreground_alpha_sha256': a['sha256'],
            'layers': [{'id':'background', 'kind':'background_design', 'enabled':True, 'strength':1.,
                        'assets': copy.deepcopy(plan['layers'][0]['assets'])}]}}}
        plan['layers'][0].update(id='background', enabled=True, strength=1.)
        metadata_path = root/'metadata.json'; metadata_path.write_text(json.dumps(metadata), encoding='utf8')
        run.update(render=metadata, artifacts={'metadata':str(metadata_path)})
        manifest_path = root/'outputs/runs/fixture/manifest.json'; manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps(run), encoding='utf8')
        assert validate_diagnosed_plate(root, run)['plate_matches_saved_render']
        for changed_role in ('background', 'old_environment', 'mapping', 'alpha'):
            changed = copy.deepcopy(run); p = changed['options']['appearance_revision']['design_reconstruction_plan']
            if changed_role == 'background': p['layers'][0]['assets']['background'] = old
            if changed_role == 'old_environment': p['source_baseline']['assets']['old_environment'] = bg
            if changed_role == 'mapping': p['canvas']['candidate_mapping']['candidate_original_to_final_original'][0][2] = 3
            if changed_role == 'alpha': p['mask_roles']['foreground_alpha'] = oa
            try: validate_diagnosed_plate(root, changed)
            except ValueError: pass
            else: raise AssertionError('Different but self-consistent donor/plate cannot consume this diagnosis')
        windows = qualified_review_windows(run)
        assert len(windows) == 2
        for mutation in ('unknown_material', 'uncited', 'wrong_region', 'wrong_owner'):
            changed = copy.deepcopy(run)
            for row in changed['cosplay_post_review']['local_observations']:
                if mutation == 'unknown_material': row['checks']['skin_materials']['status'] = 'unknown'
                if mutation == 'uncited': row['checks']['native_detail']['evidence'] = []
                if mutation == 'wrong_region': row['region'] = 'hair'
                if mutation == 'wrong_owner': row['checks']['native_detail']['correction_stage'] = 'reference_generation'
            assert not qualified_review_windows(changed)
        preserved = {str(p): sha(p) for p in root.glob('*.npz')}
        contacts = [{'assets': {'receiver_support': j, 'source_support_surface': f}}]
        with patch('native_receiver_detail.validate_diagnosis', return_value=(windows, {'sha256': 'qa'})), \
             patch('reference_appearance_prepare.preserve_contact_layers', return_value=(contacts, {})):
            prepared = prepare_receiver_detail(root, run, root/'prepared')
            assert prepared['changed']
            receipt = prepared['receipt']
            assert sha(root/'prepared/receiver-detail.json') == prepared['design_reconstruction_plan']['layers'][0]['assets']['native_receiver_detail']['sha256']
            with np.load(receipt['output_background']['path']) as data: after = data['values']
            reviewed = np.zeros_like(alpha, bool)
            for row in windows:
                x0, y0, x1, y1 = row['native_box']; reviewed[y0:y1, x0:x1] = True
            allowed = reviewed & (alpha < .5) & (observed < .5) & (floor > .5) & (joint > .5) & (validity >= .999)
            before = candidate*validity[..., None]
            assert np.array_equal(after[~allowed], before[~allowed])
            assert receipt['checks']['permitted_pixels'] == int(allowed.sum())
            changed = np.any(after != before, axis=-1)
            assert changed[58:61, 40:85].all(), 'Nonzero matte tails must not break learned-floor detail'
            assert not np.any(changed & ((alpha >= .5) | (observed >= .5)))
            # Directly recover lost curve/texture instead of creating sharpening noise.
            selected = allowed & (yy > 49) & (yy < 70) & (xx > 37) & (xx < 96)
            def bands(value):
                y = value @ LUMA
                return (y-gaussian_filter(y, 4))[selected]
            target, prior, restored = bands(source), bands(before), bands(after)
            correlations = [float(np.corrcoef(target, value)[0, 1]) for value in (prior, restored)]
            assert correlations[1] > correlations[0]+.1 and correlations[1] > .9, correlations
            repeated = copy.deepcopy(run)
            repeated['options']['appearance_revision']['design_reconstruction_plan'] = prepared['design_reconstruction_plan']
            kept = prepare_receiver_detail(root, repeated, root/'repeat')
            assert not kept['changed'] and not (root/'repeat/receiver-detail-background.npz').exists()
            unregistered = copy.deepcopy(run)
            unregistered['options']['appearance_revision']['design_reconstruction_plan']['canvas']['candidate_mapping']['scene_registration_supported'] = False
            assert not prepare_receiver_detail(root, unregistered, root/'unregistered')['changed']
        assert all(sha(Path(p)) == expected for p, expected in preserved.items())
        print(json.dumps({'ok': True, 'overlap_unioned_once': True, 'unsupported_exact': True,
            'source_detail_correlations_before_after': correlations, 'immutable_parent_assets': True,
            'offline_fixture_is_visual_acceptance': False}))


if __name__ == '__main__':
    main()
