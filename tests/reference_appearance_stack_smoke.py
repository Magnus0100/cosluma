"""Check frozen appearance replay across native, preview and saved consumers."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from reference_appearance_stack import (KIND, SCHEMA, WORKING_DOMAIN, PREPARED_STAGES,
                                        apply_stack, validate_layer)


def main():
    from design_reconstruction import materialize_plan, sample_asset
    from display_transform import encode_display, plan_for_run as display_for_run
    from float_appearance import render_source_float
    from native_roi_render import render_native_roi
    from native_source_baseline import source_for_plan

    rng = np.random.default_rng(31)
    native = (67, 789)
    h, w = native[1], native[0]
    alpha = rng.uniform(0, 1, (h, w)).astype(np.float32)
    alpha[:3] = 0
    alpha[3:6] = .05
    fp = rng.uniform(.02, .5, (h, w, 3)).astype(np.float32) * alpha[..., None]
    old = rng.uniform(.1, .45, (h, w, 3)).astype(np.float32)
    target = rng.uniform(.1, .45, (h, w, 3)).astype(np.float32)
    delta = rng.uniform(-.01, .04, (h, w, 3)).astype(np.float32) * alpha[..., None]
    validity = np.ones((h, w), np.float32)
    placeholder = Image.new('L', (1, 1), 0)
    with tempfile.TemporaryDirectory(prefix='cosplay_appearance_stack_') as folder:
        root = Path(folder)
        inputs = root / 'inputs'; inputs.mkdir()
        def asset(name, values, **extra):
            path = inputs / (name + '.npz')
            np.savez_compressed(path, values=np.asarray(values, np.float32))
            return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'size': list(native), 'coordinate_space': 'source_canvas',
                    'working_space': WORKING_DOMAIN, 'sampling': 'bilinear', **extra}
        fp_asset = asset('fp', fp)
        alpha_asset = asset('alpha', alpha)
        old_asset = asset('old', old)
        background = {'id': 'background_design', 'kind': 'background_design', 'enabled': True,
                      'strength': 1., 'params': {}, 'assets': {
                          'foreground_premultiplied': fp_asset, 'old_environment': old_asset,
                          'background': asset('target', target, representation='sampling_validity_associated_RGB', validity_associated=True), 'validity': asset('validity', validity)}}
        params = {'schema': SCHEMA, 'working_domain': WORKING_DOMAIN,
                  'source_sha256': 'a' * 64, 'reference_sha256': 'b' * 64,
                  'source_fp_sha256': fp_asset['sha256'], 'alpha_sha256': alpha_asset['sha256'],
                  'canvas_size': list(native), 'prepared_stages': list(PREPARED_STAGES),
                  'freeze_mode': 'native_premultiplied_delta', 'provenance_identity_sha256': 'c' * 64}
        layer = {'id': KIND, 'kind': KIND, 'enabled': True, 'strength': 1., 'params': params,
                 'assets': {'automatic_delta': asset('delta', delta, representation='premultiplied_delta')}}
        plan = {'schema_version':'design_reconstruction_plan.v1','enabled': True, 'layers': [layer, background], 'working_domain': WORKING_DOMAIN,
                'canvas': {'source_oriented_size': list(native)},
                'source_sha256': params['source_sha256'], 'candidate_sha256': params['reference_sha256'],
                'mask_roles': {'foreground_alpha': alpha_asset},
                'source_baseline': {'schema':'appearance_source_baseline.v1','working_domain':'display_linear_sRGB',
                                    'source_display_applied_once':True, 'assets': {'foreground_premultiplied': fp_asset,
                                             'foreground_alpha': alpha_asset, 'old_environment': old_asset},
                                    'source_precision': 'raw16_linear', 'decode_version': 'fixture'},
                'execution': {'source_reference': 'paired_native_baseline'}}
        source_path = inputs / 'source.png'; Image.new('RGB', native).save(source_path)
        reference_path = inputs / 'reference.png'; Image.new('RGB', native).save(reference_path)
        run = {'kind': 'single_input_run', 'input_mode': 'single_source', 'run_id': 'fixture',
               'sample': 'fixture', 'revision': 1, 'source': {'path': str(source_path), 'sha256': params['source_sha256']},
               'reference': {'path': str(reference_path), 'sha256': params['reference_sha256']},
               'options': {'render_mode': 'native_source_design',
                           'appearance_revision': {'design_reconstruction_plan': plan}}}
        from reference_appearance_controls import prepare_owned_controls
        plan=prepare_owned_controls(plan,root/'controls',run=run)
        run['options']['appearance_revision']['design_reconstruction_plan']=plan
        layer=plan['layers'][0]
        from reference_support import build_reference_support
        binding={key:params[key] for key in ('source_sha256','reference_sha256','alpha_sha256','source_fp_sha256','canvas_size')}
        binding['automatic_delta_sha256']=layer['assets']['automatic_delta']['sha256']
        support=build_reference_support({'support_policy':'reference_support.v1','operators':{},'observations':{}},
            {'support_policy':'reference_support.v1','regions':{}},{'support_policy':'reference_support.v1','regions':{}},binding=binding)
        support_path=root/'support.json';support_path.write_text(json.dumps(support),encoding='utf8')
        layer['params']['reference_support']=support
        layer['assets']['reference_support']={'path':str(support_path),'sha256':hashlib.sha256(support_path.read_bytes()).hexdigest()}
        source = source_for_plan(plan, native)
        before = fp.copy()
        layer['strength'] = 0
        assert np.array_equal(apply_stack(fp, layer, plan, run)[0], fp)
        base, operator = render_source_float(source, run)
        expected = fp + (1 - alpha[..., None]) * target
        np.testing.assert_allclose(base, expected, rtol=0, atol=1e-7)
        assert not next(x for x in operator['design_reconstruction_plan']['layers'] if x['kind'] == KIND)['applied']
        layer['strength'] = 1
        matched = apply_stack(fp, layer, plan, run)[0]
        np.testing.assert_array_equal(matched, fp + delta)
        np.testing.assert_array_equal(fp, before)
        np.testing.assert_array_equal(matched[:3], np.zeros_like(matched[:3]))
        # .05 alpha receives the associated delta once, never alpha squared.
        np.testing.assert_array_equal(matched[3:6] - fp[3:6], (fp + delta)[3:6] - fp[3:6])
        full, operator = render_source_float(source, run)
        np.testing.assert_allclose(full, matched + (1 - alpha[..., None]) * target, rtol=0, atol=1e-7)
        receipt = next(x for x in operator['design_reconstruction_plan']['layers'] if x['kind'] == KIND)
        assert receipt['prepared_stages'] == list(PREPARED_STAGES) and receipt['consumed_asset_sha256'] == layer['assets']['automatic_delta']['sha256']
        assert receipt['execution_stage'] == 'native_frozen_reference_appearance_before_composition'
        # Manual controls follow the frozen automatic delta in the same domain.
        run['options']['appearance_revision']['params'] = {'subject': {'exposure': .12}, 'background': {'temperature': .1}, 'global': {'exposure': -.04}}
        edited, _ = render_source_float(source, run)
        assert np.max(np.abs(edited - full)) > .001
        box = (8, 241, 51, 282)
        roi, roi_receipt = render_native_roi(run, box)
        np.testing.assert_allclose(roi, edited[box[1]:box[3], box[0]:box[2]], rtol=0, atol=1e-7)
        assert roi_receipt['working_domain'] == WORKING_DOMAIN
        display = display_for_run(run)
        assert display['method'] == 'clip_sdr'
        np.testing.assert_allclose(encode_display(np.array([[[.9, .9, .9]]], np.float32), display),
                                   np.full((1, 1, 3), .9546871, np.float32), rtol=0, atol=1e-6)
        # Explicit inherited soft-shoulder settings cannot cause a second D.
        run['options']['appearance_revision']['display_transform'] = {'method': 'luminance_soft_shoulder', 'shoulder_start': .5}
        assert display_for_run(run)['method'] == 'clip_sdr'
        try:display_for_run({'options':{}})
        except ValueError:pass
        else:raise AssertionError('Missing current display plan was accepted')
        off = copy.deepcopy(run)
        off['options']['appearance_revision'].pop('params')
        off['options']['appearance_revision']['design_reconstruction_plan']['enabled'] = False
        recovered, _ = render_source_float(source, off)
        np.testing.assert_array_equal(recovered, source)
        # Retained coverage is associated at native before reduction.
        fraction = ((np.indices((h, w)).sum(0) % 2) == 0).astype(np.float32)
        plan['mask_roles']['edited_edge_light_applicability'] = asset('retained', fraction)
        retained = apply_stack(fp, layer, plan, run)[0]
        np.testing.assert_array_equal(retained, fp + delta * fraction[..., None])
        small = (19, 337)
        proxy_fp = sample_asset(fp_asset, 'values', (0, 0, *small), small)
        proxy = apply_stack(proxy_fp, layer, plan, run)[0]
        expected_delta = np.stack([np.asarray(Image.fromarray(delta[..., c] * fraction).resize(small, Image.Resampling.BILINEAR), np.float32) for c in range(3)], -1)
        np.testing.assert_allclose(proxy, proxy_fp + expected_delta, rtol=0, atol=2e-7)
        edited_alpha = alpha * .95
        plan['mask_roles']['foreground_alpha'] = asset('edited_alpha', edited_alpha)
        validate_layer(layer, plan, run)
        unsupported = copy.deepcopy(plan)
        unsupported['mask_roles'].pop('edited_edge_light_applicability')
        try:
            validate_layer(layer, unsupported, run)
            raise AssertionError('Changed alpha must invalidate unsupported automatic replay')
        except ValueError:
            pass
        # Paths are locators. Same-byte materialization and portability keep identity.
        owned = materialize_plan(root, run, plan)
        owned_layer = owned['layers'][0]
        assert validate_layer(owned_layer, owned, run)['consumed_asset_sha256'] == receipt['consumed_asset_sha256']
        run['options']['appearance_revision']['design_reconstruction_plan'] = owned
        run_dir = root / 'outputs/runs/fixture'; run_dir.mkdir(parents=True, exist_ok=True)
        manifest = run_dir / 'manifest.json'; manifest.write_text(json.dumps(run), encoding='utf-8')
        from portable_project import export_project, import_project
        package = root / 'fixture.cosplay'; export_project(root, 'fixture', package, include_raw=False, include_design=False)
        imported = import_project(root / 'relocated', package, run_id='relocated-fixture')['run']
        imported_plan = imported['options']['appearance_revision']['design_reconstruction_plan']
        validate_layer(imported_plan['layers'][0], imported_plan, imported)
        imported_source = source_for_plan(imported_plan, native)
        reopened, _ = render_source_float(imported_source, imported)
        saved, _ = render_source_float(source, run)
        np.testing.assert_array_equal(reopened, saved)
        # Saved strips and export retain working-linear domain metadata.
        from raw_support import save_scene_master, linear_preview
        from native_detail import read_scene_crop, saved_native_detail
        working = run_dir / 'render_scene.tif'
        save_scene_master(saved, working, working_domain=WORKING_DOMAIN)
        decoded, _ = read_scene_crop(working, box)
        np.testing.assert_array_equal(decoded, saved[box[1]:box[3], box[0]:box[2]])
        image = run_dir / 'auto_draft.png'; linear_preview(saved, display_transform=display).save(image)
        from raw_support import save_render_master
        master=run_dir/'render_master.tif';save_render_master(saved,master,display_transform=display)
        run['artifacts'] = {'render_scene_master': str(working), 'auto_draft': str(image), 'render_master':str(master)}
        run['render'] = {'operator': {'working_domain': WORKING_DOMAIN, 'display_transform': display},
          'render_precision':{'working_domain':WORKING_DOMAIN,'working_precision':'float32_linear','master_encoding':'uint16_srgb'}}
        manifest.write_text(json.dumps(run), encoding='utf-8')
        from source_preview import SourcePreviewCache
        preview = SourcePreviewCache(root, max_edge=96).preview('fixture', run, manifest, {})
        assert preview['working_domain'] == WORKING_DOMAIN and preview['display_transform']['method'] == 'clip_sdr'
        from live_native_detail import live_native_detail
        live = live_native_detail(root, 'fixture', {'edge':128})
        assert live['working_domain'] == WORKING_DOMAIN and live['display_transform']['method'] == 'clip_sdr'
        assert live['bounded_roi_renderer'] and not live['raw_decoded']
        detail = saved_native_detail(root, 'fixture', {'edge': 128})
        assert detail['working_domain'] == WORKING_DOMAIN and detail['display_transform']['method'] == 'clip_sdr'
        from export_artifact import export_source_run
        exported = export_source_run(root, 'fixture', directory=str(root / 'exports'), format='TIFF32_LINEAR')
        assert exported['metadata']['working_linear_master'] and not exported['metadata']['scene_linear_master']
        assert exported['metadata']['working_domain'] == WORKING_DOMAIN
        assert Path(exported['path']).read_bytes() == working.read_bytes()
        # Editing source assets cannot keep claiming the old content identity.
        mutated = copy.deepcopy(owned)
        broken_path = inputs / 'broken_delta.npz'; np.savez_compressed(broken_path, values=delta + .001)
        mutated['layers'][0]['assets']['automatic_delta']['path'] = str(broken_path)
        try:
            validate_layer(mutated['layers'][0], mutated, run)
            raise AssertionError('Changed bytes must invalidate frozen replay')
        except ValueError:
            pass
    print('OK: aggregate 0/1; no alpha squared; manual edits; exact native ROI; retained-native sampling; no double display transform; materialization; portable reopen; source preview; live/saved detail; working-linear TIFF32 export; content mutation invalidation')


if __name__ == '__main__':
    main()
