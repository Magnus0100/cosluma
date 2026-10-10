"""CPU contract and real full/ROI pixels for preserving native contact at freeze."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from reference_appearance_prepare import preserve_contact_layers, sha


def main():
    from design_scene_assets import prepare_contact_receiver_assets
    from design_reconstruction import _apply_native_plan, sample_asset
    from native_roi_render import render_native_roi
    from native_sole_contact import sample_native_contact
    from reference_appearance_stack import PREPARED_STAGES

    native, proxy = (72, 80), (36, 40)
    alpha = np.zeros(native[::-1], np.float32)
    alpha[30:50, 24:40] = 1
    alpha[50:64, 24:40] = .25
    fp = alpha[..., None] * np.array([.22, .31, .40], np.float32)
    delta = alpha[..., None] * np.array([.01, -.015, .02], np.float32)
    old = np.full((*native[::-1], 3), .3, np.float32)
    target = np.full_like(old, [.6, .7, .8])
    proposal = np.zeros(proxy[::-1], np.float32)
    proposal[25:32, 12:20] = 1
    footwear = np.zeros_like(proposal); footwear[15:25, 12:20] = 1
    source_surface = np.zeros_like(proposal); source_surface[25:] = .8
    candidate_surface = np.zeros_like(proposal); candidate_surface[25:] = .6
    semantics = np.zeros(proxy[::-1], np.uint8); semantics[15:25, 12:20] = 9

    with tempfile.TemporaryDirectory(prefix='cosplay_contact_freeze_') as folder:
        root = Path(folder)

        def asset(name, values, *, key='values', coordinate_space='source_canvas', **extra):
            path = root / (name + '.npz')
            np.savez_compressed(path, **{key: values})
            return dict(path=str(path), sha256=sha(path), size=list(values.shape[:2][::-1]),
                coordinate_space=coordinate_space, original_canvas_size=list(native),
                working_space='display_linear_sRGB', sampling='bilinear', **extra)

        a = asset('alpha', alpha)
        scene = {'source': {'oriented_size': list(native), 'sha256': 'a' * 64},
            'candidate': {'oriented_size': list(native), 'sha256': 'b' * 64},
            'background_mapping': {'supported': True, 'candidate_original_to_source_original': np.eye(3).tolist()},
            'contact': {'source': {'supported': True,
                'contact_support': asset('proposal', proposal),
                'footwear': asset('footwear', footwear),
                'support_surface': asset('source_surface', source_surface)},
                'candidate': {'support_surface': asset('candidate_surface', candidate_surface)}}}
        contact, _ = prepare_contact_receiver_assets(scene, proxy, root / 'contact',
            source_foreground_alpha=a,
            source_semantics=asset('semantics', semantics, key='labels', coordinate_space='source_proxy'))
        assert contact['params']['receiver_qualification']['qualified_for_automatic_contact']
        contact['strength'] = .7
        contact['params']['opacity'] = .4
        fpa, olda = asset('fp', fp), asset('old', old)
        bg = {'id': 'background_design', 'kind': 'background_design', 'enabled': True, 'strength': 1.,
            'params': {}, 'assets': {'foreground_premultiplied': fpa, 'old_environment': olda,
                'background': asset('target', target,representation='sampling_validity_associated_RGB',validity_associated=True), 'validity': asset('validity', np.ones_like(alpha))}}
        stack = {'id': 'reference_appearance_stack', 'kind': 'reference_appearance_stack',
            'enabled': True, 'strength': 1., 'params': {'schema': 'reference_appearance_stack.v2',
                'working_domain': 'display_linear_sRGB', 'source_sha256': 'a' * 64,
                'reference_sha256': 'b' * 64, 'alpha_sha256': a['sha256'], 'source_fp_sha256': fpa['sha256'],
                'canvas_size': list(native), 'prepared_stages': list(PREPARED_STAGES),
                'freeze_mode': 'native_premultiplied_delta', 'provenance_identity_sha256': 'c' * 64},
            'assets': {'automatic_delta': asset('delta', delta, representation='premultiplied_delta')}}
        plan = {'schema_version':'design_reconstruction_plan.v1','enabled': True, 'working_domain': 'display_linear_sRGB',
            'source_sha256': 'a' * 64, 'candidate_sha256': 'b' * 64,
            'canvas': {'source_oriented_size': list(native), 'source_proxy_size': list(proxy),
                'candidate_mapping': {'scene_registration_supported': True}},
            'mask_roles': {'foreground_alpha': a}, 'layers': [stack, bg, contact],
            'source_baseline': {'schema':'appearance_source_baseline.v1','working_domain':'display_linear_sRGB','assets': {'foreground_premultiplied': fpa, 'foreground_alpha': a,
                'old_environment': olda}},
            'execution': {'source_reference': 'paired_native_baseline', 'receiver_context': 'current_environment'}}
        run = {'source': {'sha256': 'a' * 64}, 'reference': {'sha256': 'b' * 64},
            'options': {'render_mode':'native_source_design','appearance_revision': {'design_reconstruction_plan': plan}}}
        from reference_appearance_controls import prepare_owned_controls
        from reference_support import build_reference_support,BINDING_KEYS
        plan=prepare_owned_controls(plan,root/'controls',run=run)
        stack,bg,contact=plan['layers']
        binding={key:stack['params'][key] for key in BINDING_KEYS if key!='automatic_delta_sha256'}
        binding['automatic_delta_sha256']=stack['assets']['automatic_delta']['sha256']
        support=build_reference_support({'support_policy':'reference_support.v1','operators':{},'observations':{}},{'support_policy':'reference_support.v1','regions':{}},{'support_policy':'reference_support.v1','regions':{}},binding=binding)
        support_path=root/'support.json';support_path.write_text(json.dumps(support),'utf8')
        stack['params']['reference_support']=support
        stack['assets']['reference_support']={'path':str(support_path),'sha256':sha(support_path)}
        run['options']['appearance_revision']['design_reconstruction_plan']=plan
        layers, receipt = preserve_contact_layers(plan, run)
        assert layers == [contact] and layers[0] is not contact and receipt['preserved_count'] == 1
        assert layers[0]['params'] is not contact['params']
        assert not receipt['new_contact_inferred'] and not receipt['visual_acceptance']
        plan['layers'] = [stack, bg, *layers]

        source = fp + (1 - alpha[..., None]) * old
        geometry = Image.fromarray(np.uint8(alpha * 255))
        actual, execution = _apply_native_plan(source, plan, geometry, run, False)
        support = sample_native_contact(contact['assets']['attenuation_support'], (0, 0, *native), native)
        receiver = sample_asset(contact['assets']['receiver_support'], 'values', (0, 0, *native), native)
        attenuation = support * receiver * np.float32(.4 * .7)
        expected = fp + delta + (1 - alpha[..., None]) * target * (1 - attenuation[..., None])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(actual[alpha == 1], (fp + delta)[alpha == 1])
        partial = (support > .05) & (alpha == .25)
        assert partial.any()
        # Q=.48 and partial coverage make accidental Q^2 or alpha preweight
        # observable; this assertion uses the production renderer result.
        no_contact = fp + delta + (1 - alpha[..., None]) * target
        decrease = no_contact - actual
        np.testing.assert_allclose(decrease[partial],
            ((1 - alpha[..., None]) * target * attenuation[..., None])[partial], rtol=0, atol=1e-7)
        assert decrease[partial].max() > .005
        record = next(x for x in execution['layers'] if x['kind'] == 'contact_shadow')
        assert record['receiver_support_consumed'] and record['final_alpha_consumed_by_composition']
        assert not record['foreground_coverage_preweighted']
        box = (20, 45, 44, 68)
        roi, _ = render_native_roi(run, box)
        np.testing.assert_allclose(roi, actual[box[1]:box[3], box[0]:box[2]], rtol=0, atol=1e-7)

        disabled = copy.deepcopy(plan)
        disabled['layers'][-1]['enabled'] = False
        kept, _ = preserve_contact_layers(disabled, run)
        assert len(kept) == 1 and kept[0]['enabled'] is False
        off, _ = _apply_native_plan(source, disabled, geometry, run, False)
        np.testing.assert_allclose(off, no_contact, rtol=0, atol=1e-7)
        absent = copy.deepcopy(plan); absent['layers'] = [stack, bg,
            {'kind': 'particles', 'enabled': True}, {'kind': 'rim_light', 'enabled': True}]
        assert preserve_contact_layers(absent, run)[0] == []
        historical = copy.deepcopy(plan)
        historical['layers'][-1]['provenance']['contract_schema'] = 'contact_receiver_contract.v3'
        assert preserve_contact_layers(historical, run)[0] == []
        overlap = contact['params']['receiver_qualification']['native_box']
        assert preserve_contact_layers(plan, run, changed_receiver_box=overlap)[0] == []
        assert len(preserve_contact_layers(plan, run, changed_receiver_box=(0, 70, 5, 75))[0]) == 1

        def rejects(p=plan, r=run):
            try:
                preserve_contact_layers(p, r)
            except ValueError:
                return
            raise AssertionError('Changed contact binding must not freeze')

        for role in ('source', 'reference'):
            changed = copy.deepcopy(run); changed[role]['sha256'] = 'd' * 64
            rejects(r=changed)
        changed = copy.deepcopy(plan); changed['mask_roles']['foreground_alpha']['sha256'] = 'e' * 64
        rejects(p=changed)
        changed = copy.deepcopy(plan); changed['canvas']['source_oriented_size'][0] += 1
        rejects(p=changed)
        changed = copy.deepcopy(plan); changed['canvas']['candidate_mapping']['scene_registration_supported'] = False
        rejects(p=changed)
        changed = copy.deepcopy(plan)
        for location in (changed['layers'][-1]['params'], changed['layers'][-1]['provenance']):
            location['receiver_qualification']['receiver_support_baked_into_attenuation'] = True
        rejects(p=changed)
        changed = copy.deepcopy(plan)
        changed['layers'][-1]['assets']['receiver_support']['path'] = asset('corrupt', np.ones_like(receiver))['path']
        rejects(p=changed)
        changed = copy.deepcopy(plan)
        invalid = asset('invalid', np.full(proxy[::-1], 1.1, np.float32))
        changed['layers'][-1]['assets']['receiver_support'].update(path=invalid['path'], sha256=invalid['sha256'])
        changed['layers'][-1]['provenance']['asset_bindings']['receiver_support_sha256'] = invalid['sha256']
        rejects(p=changed)
    print('OK: real qualified native contact retained; disabled state retained; absent/historical/overlapping-repair contacts omitted; source/reference/alpha/canvas/mapping/bytes/range bindings; actual full and ROI pixels; Q once and final alpha once; no FX resurrection')


if __name__ == '__main__':
    main()
