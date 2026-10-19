"""Check native independent controls, exact prepared endpoints and replay closure."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

from reference_appearance_controls import (CONTROL_RECIPE, blend_delta, corner_targets,
    coupling_measurements, from_lab, to_lab, prepare_owned_controls, _corner_bands)
from reference_appearance_stack import (KIND, PREPARED_STAGES, SCHEMA,
    WORKING_DOMAIN, apply_stack, validate_controls, validate_layer)


def main():
    from design_reconstruction import materialize_plan, sample_asset
    from float_appearance import render_source_float
    from native_roi_render import render_native_roi
    from native_source_baseline import source_for_plan
    from portable_project import export_project, import_project
    from display_transform import source_transform_for_precision
    from raw_support import RGB_DECODE_VERSION

    rng = np.random.default_rng(841)
    rgb = rng.uniform(.08, .6, (4096, 3))
    tiny = np.power(10., rng.uniform(-18, 0, (4096, 1)))
    np.testing.assert_allclose(to_lab(rgb*tiny), np.cbrt(tiny)*to_lab(rgb), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(from_lab(to_lab(rgb*tiny)), rgb*tiny, rtol=1e-11, atol=1e-14)
    # A legal pair can produce illegal mixed Lab corners; reduce only the
    # requested channel dose, not source coverage or untouched Lab channels.
    saturated_source = np.eye(3, dtype=np.float32)*.9 + .02
    saturated_target = saturated_source[[1, 2, 0]]
    corners, doses, _ = corner_targets(saturated_source, saturated_target, np.ones(3, np.float32))
    assert any(np.any(values < .999) for values in doses.values())
    for role, values in corners.items():
        assert np.min(values) >= 0 and np.max(values) <= 1
        current = to_lab(values)
        if role == 'colour_delta':
            np.testing.assert_allclose(current[:, 0], to_lab(saturated_source)[:, 0], rtol=0, atol=2e-8)
        else:
            np.testing.assert_allclose(current[:, 1:], to_lab(saturated_source)[:, 1:], rtol=0, atol=2e-8)

    # Zero coverage and unsupported endpoints select the original RGB bits
    # without any nonlinear conversion. Supported equal boundary endpoints
    # still retain their existing round-trip gamut dose in the QA receipt.
    empty_alpha = np.zeros((17, 31), np.float32)
    empty_source = np.full((*empty_alpha.shape, 3), np.float32(-0.0))
    empty_source[4, 8] = [.2, .1, .3]
    with patch('reference_appearance_controls.to_lab', side_effect=AssertionError('unsupported Lab conversion')):
        empty_corners, empty_doses, empty_support = corner_targets(empty_source, empty_source, empty_alpha)
    assert not empty_support.any()
    for role in empty_corners:
        np.testing.assert_array_equal(empty_corners[role].view(np.uint32), empty_source.view(np.uint32))
        assert not empty_doses[role].any()
    sparse_source = np.zeros((17, 31, 3), np.float32)
    sparse_target, sparse_alpha = sparse_source.copy(), empty_alpha.copy()
    sparse_source[8, 10:13] = saturated_source
    sparse_target[8, 10:13] = saturated_target
    sparse_alpha[8, 10:14] = 1
    sparse_source[8, 13] = sparse_target[8, 13] = 1
    conversions = []
    def observed_lab(values):
        conversions.append(values.shape)
        return to_lab(values)
    with patch('reference_appearance_controls.to_lab', side_effect=observed_lab):
        sparse_corners, sparse_doses, sparse_support = corner_targets(sparse_source, sparse_target, sparse_alpha)
    assert conversions == [(4, 3), (4, 3)]
    assert sparse_support.sum() == 4
    for role in corners:
        np.testing.assert_array_equal(sparse_corners[role][8, 10:13].view(np.uint32), corners[role].view(np.uint32))
        np.testing.assert_array_equal(sparse_doses[role][8, 10:13].view(np.uint64), doses[role].view(np.uint64))
        np.testing.assert_array_equal(sparse_corners[role][8, 13], sparse_source[8, 13])
        assert sparse_doses[role][8, 13] == 0  # exact equal white round-trip is above its boundary

    native = (83, 269); width, height = native
    alpha = rng.uniform(.05, 1, (height, width)).astype(np.float32)
    alpha[:24] = 1.; alpha[24:28] = 0.; alpha[28:32] = 1e-12
    fp = rng.uniform(.12, .42, (height, width, 3)).astype(np.float32)*alpha[..., None]
    target = rng.uniform(.18, .50, (height, width, 3)).astype(np.float32)*alpha[..., None]
    fp[33, 10] = np.array([.8, .01, .01])*alpha[33, 10]
    target[33, 10] = np.array([.01, .8, .01])*alpha[33, 10]
    # An invalid target is reported without changing the prepared full-dose endpoint.
    target[40, 30] = np.array([1.6, .2, .2])*alpha[40, 30]
    delta = target-fp
    original_target = fp+delta
    # Parallel preparation retains the same 128-row arithmetic, tail shape,
    # ordered publication and dose bits as the serial producer.
    serial_bands = list(_corner_bands(fp, alpha, delta, workers=1))
    parallel_bands = list(_corner_bands(fp, alpha, delta, workers=2))
    assert [band[0] for band in parallel_bands] == [0, 128, 256]
    for serial, parallel in zip(serial_bands, parallel_bands):
        for index in (1, 2, 3, 6):
            np.testing.assert_array_equal(serial[index].tobytes(), parallel[index].tobytes())
        for index in (4, 5):
            for role in serial[index]:
                assert serial[index][role].tobytes() == parallel[index][role].tobytes()
    old = rng.uniform(.10, .25, (height, width, 3)).astype(np.float32)
    background = rng.uniform(.15, .3, (height, width, 3)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix='cosplay_reference_controls_') as folder:
        root = Path(folder); inputs = root/'input'; inputs.mkdir()
        def asset(name, values, representation=None, **extra):
            path = inputs/(name+'.npz'); np.savez_compressed(path, values=np.asarray(values, np.float32))
            return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'size':list(native), 'coordinate_space':'source_canvas',
                    'working_space':WORKING_DOMAIN, 'sampling':'bilinear',
                    **({'representation':representation,**extra} if representation else {})}
        fa, aa, oa = asset('fp', fp), asset('alpha', alpha), asset('old', old)
        params = {'schema':SCHEMA,'working_domain':WORKING_DOMAIN,'source_sha256':'a'*64,
            'reference_sha256':'b'*64, 'source_fp_sha256':fa['sha256'], 'alpha_sha256':aa['sha256'],
            'canvas_size':list(native), 'prepared_stages':list(PREPARED_STAGES),
            'freeze_mode':'native_premultiplied_delta'}
        layer = {'id':KIND, 'kind':KIND, 'enabled':True, 'strength':1., 'params':params,
                 'assets':{'automatic_delta':asset('full',delta,'premultiplied_delta')}}
        bg = {'id':'background_design','kind':'background_design','enabled':True,'strength':1.,'params':{},
            'assets':{'foreground_premultiplied':fa,'old_environment':oa,
                'background':asset('background',background,representation='sampling_validity_associated_RGB',validity_associated=True),'validity':asset('validity',np.ones_like(alpha))}}
        plan = {'schema_version':'design_reconstruction_plan.v1','enabled':True, 'layers':[layer,bg], 'working_domain':WORKING_DOMAIN,
            'canvas':{'source_oriented_size':list(native)}, 'source_sha256':params['source_sha256'],
            'candidate_sha256':params['reference_sha256'], 'mask_roles':{'foreground_alpha':aa},
            'source_baseline':{'schema':'appearance_source_baseline.v1','working_domain':'display_linear_sRGB','assets':{'foreground_premultiplied':fa,'foreground_alpha':aa,'old_environment':oa},
                               'source_sha256':params['source_sha256'], 'source_precision':'raster8_srgb',
                               'decode_version':RGB_DECODE_VERSION, 'source_display':source_transform_for_precision('raster8_srgb'),
                               'source_display_applied_once':True},
            'execution':{'source_reference':'paired_native_baseline'}}
        source_path, reference_path = inputs/'source.png',inputs/'reference.png'
        Image.new('RGB', native).save(source_path); Image.new('RGB',native).save(reference_path)
        run = {'kind':'single_input_run','input_mode':'single_source','run_id':'fixture','sample':'fixture',
            'revision':1,'source':{'path':str(source_path),'sha256':params['source_sha256']},
            'reference':{'path':str(reference_path),'sha256':params['reference_sha256']},
            'options':{'render_mode':'native_source_design','appearance_revision':{'design_reconstruction_plan':plan}}}
        from reference_support import build_reference_support
        binding={key:params[key] for key in ('source_sha256','reference_sha256','alpha_sha256','source_fp_sha256','canvas_size')}
        binding['automatic_delta_sha256']=layer['assets']['automatic_delta']['sha256']
        support=build_reference_support({'support_policy':'reference_support.v1','operators':{},'observations':{}},
            {'support_policy':'reference_support.v1','regions':{}},{'support_policy':'reference_support.v1','regions':{}},binding=binding)
        support_path=root/'support.json';support_path.write_text(json.dumps(support),encoding='utf8')
        layer['params']['reference_support']=support
        layer['assets']['reference_support']={'path':str(support_path),'sha256':hashlib.sha256(support_path.read_bytes()).hexdigest()}
        before = json.dumps(plan, sort_keys=True)
        input_hashes = {item['path']:hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() for item in (fa,aa,oa,layer['assets']['automatic_delta'])}
        upgraded = prepare_owned_controls(plan, root/'controls', run=run)
        # A retry after a worker interruption must attach the atomically
        # published control bank without rereading native arrays or replaying
        # Beauty.  This is the production cache path for the expensive corner
        # derivation; fixture assets remain tiny so the test is deterministic.
        with (patch('reference_appearance_controls._read_native', side_effect=AssertionError('cached control reread')),
              patch('portrait_beauty_layer.apply_for_run', side_effect=AssertionError('cached Beauty replay'))):
            cached_plan = prepare_owned_controls(plan, root/'controls', run=run)
        assert cached_plan['provenance']['appearance_controls']['cache_hit'] is True
        for role in ('colour_delta', 'lightness_delta', 'control_receipt'):
            assert cached_plan['layers'][0]['assets'][role] == upgraded['layers'][0]['assets'][role]
        # A fresh freeze supplies these exact in-memory endpoints. Neither
        # loading them again nor invoking another Beauty replay is necessary.
        native_inputs = {'working_foreground': fp, 'alpha': alpha, 'automatic_delta': delta,
                         'beauty_execution': {'enabled': False, 'applied': False}}
        memory_before = [values.tobytes() for values in (fp, alpha, delta)]
        from reference_appearance_stack import _validate_delta
        with (patch('reference_appearance_controls._read_native', side_effect=AssertionError('duplicate native read')),
              patch('portrait_beauty_layer.apply_for_run', side_effect=AssertionError('duplicate Beauty replay')),
              patch('reference_appearance_stack._validate_delta', side_effect=AssertionError('duplicate saved delta unpack')),
              patch('reference_appearance_stack._validate_mask', side_effect=AssertionError('duplicate saved alpha unpack'))):
            memory_plan = prepare_owned_controls(plan, root/'memory-controls', run=run, native_inputs=native_inputs)
        assert memory_before == [values.tobytes() for values in (fp, alpha, delta)]
        native_controls = {}
        for role in ('colour_delta', 'lightness_delta'):
            with (np.load(upgraded['layers'][0]['assets'][role]['path']) as disk,
                  np.load(memory_plan['layers'][0]['assets'][role]['path']) as memory):
                np.testing.assert_array_equal(disk['values'].view(np.uint32), memory['values'].view(np.uint32))
                native_controls[role] = memory['values']
        # Default reopen still reads the actual saved bank. A producer handoff
        # cannot bypass file binding or the native shape/float/finite contract.
        _validate_delta.cache_clear()
        with patch('reference_appearance_stack._validate_delta', wraps=_validate_delta) as reopened:
            validate_controls(memory_plan['layers'][0], memory_plan, run)
        assert all(any(role.replace('_', '-') in Path(call.args[0]).name for call in reopened.call_args_list)
                   for role in native_controls)
        nonfinite = native_controls['colour_delta'].copy(); nonfinite[0, 0, 0] = np.nan
        for invalid in ({'colour_delta': native_controls['colour_delta']},
                        {**native_controls, 'colour_delta': native_controls['colour_delta'][:-1]},
                        {**native_controls, 'colour_delta': native_controls['colour_delta'].astype(np.float64)},
                        {**native_controls, 'colour_delta': nonfinite}):
            try:validate_controls(memory_plan['layers'][0], memory_plan, run, native_controls=invalid)
            except ValueError:pass
            else:raise AssertionError('Invalid live control bank accepted')
        broken_live = copy.deepcopy(memory_plan)
        broken_live['layers'][0]['assets']['colour_delta']['sha256'] = '0'*64
        try:validate_controls(broken_live['layers'][0], broken_live, run, native_controls=native_controls)
        except ValueError:pass
        else:raise AssertionError('Changed saved bank accepted through live arrays')
        for bad_inputs in ({**native_inputs, 'automatic_delta': fp},
                           {**native_inputs, 'beauty_execution': {'enabled': True, 'applied': True}},
                           {**native_inputs, 'alpha': alpha.astype(np.float64)},
                           {**native_inputs, 'alpha': np.full_like(alpha, np.nan)},
                           {**native_inputs, 'alpha': np.full_like(alpha, 1.1)},
                           {**native_inputs, 'automatic_delta': np.full_like(delta, np.nan)}):
            try:prepare_owned_controls(plan, root/'bad-memory-controls', run=run, native_inputs=bad_inputs)
            except ValueError:pass
            else:raise AssertionError('Aliased or wrong-Beauty native control input accepted')
        assert json.dumps(plan,sort_keys=True) == before
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path,digest in input_hashes.items())
        v2 = upgraded['layers'][0]
        assert v2['params']['schema'] == SCHEMA and v2['params']['control_recipe'] == CONTROL_RECIPE
        assert v2['assets']['automatic_delta'] == layer['assets']['automatic_delta']
        receipt = json.loads(Path(v2['assets']['control_receipt']['path']).read_text(encoding='utf8'))
        assert receipt['original_endpoint_exceptions']['target_above_alpha_pixels'] == 1
        assert receipt['endpoint_contract']['TBS_refitted'] is False
        assert receipt['intermediate_coupling']['interpretation'].endswith('not_strict_Lab_channel_identity')
        assert any(row['premultiplied_L_error_max'] > 1e-5 for row in receipt['intermediate_coupling']['dose_grid'])
        early=copy.deepcopy(upgraded)
        early_layer=early['layers'][0]
        early_layer['params'].pop('control_receipt_identity_sha256')
        early_layer['params']['control_assets_sha256']['control_receipt']=early_layer['assets']['control_receipt']['sha256']
        try:prepare_owned_controls(early,root/'old-metadata',run=run)
        except ValueError:pass
        else:raise AssertionError('Obsolete control identity was repaired automatically')
        np.testing.assert_array_equal(apply_stack(fp,v2,upgraded,run)[0],original_target)
        v2.update(colour_strength=0.,lightness_strength=0.)
        np.testing.assert_array_equal(apply_stack(fp,v2,upgraded,run)[0],fp)
        for controls in ({'colour_strength':1.,'lightness_strength':0.},
                         {'colour_strength':0.,'lightness_strength':1.},
                         {'colour_strength':.4,'lightness_strength':.7}):
            v2.update(**controls)
            rendered = apply_stack(fp,v2,upgraded,run)[0]
            assert np.isfinite(rendered).all()
            np.testing.assert_array_equal(rendered[24:28],fp[24:28])
            # Invalid original target corners retain source, not a hidden clamp.
            if controls['colour_strength']*controls['lightness_strength'] == 0:
                np.testing.assert_array_equal(rendered[40,30],fp[40,30])
            valid = np.ones(alpha.shape,bool); valid[40,30]=False
            assert np.max((rendered-alpha[...,None])[valid]) < 2e-7
        v2.update(colour_strength=.4,lightness_strength=.7,strength=.8)
        full = apply_stack(fp,v2,upgraded,run)[0]
        # Native bounded ROI and full composition share all controls and order.
        run['options']['appearance_revision']['design_reconstruction_plan']=upgraded
        run['options']['appearance_revision']['params']={'subject':{'exposure':.03}}
        source = source_for_plan(upgraded,native); placeholder=Image.new('L',(1,1),0)
        image, operator = render_source_float(source, run)
        box=(3,15,72,160); roi, _=render_native_roi(run,box)
        np.testing.assert_allclose(roi,image[box[1]:box[3],box[0]:box[2]],rtol=0,atol=1e-7)
        executed=next(x for x in operator['design_reconstruction_plan']['layers'] if x['kind']==KIND)
        assert executed['colour_strength']==.4 and executed['lightness_strength']==.7
        assert executed['strict_Lab_channel_identity'] is False
        assert set(executed['consumed_control_assets_sha256'])=={'colour_delta','lightness_delta','control_receipt'}
        # Linear sampling commutes with the native corner bank. No nonlinear
        # conversion is performed after shrinking RGB, including tiny alpha.
        small=(23,61); small_fp=sample_asset(fa,'values',(0,0,*small),small)
        preview=apply_stack(small_fp,v2,upgraded,run)[0]
        expected=np.stack([np.asarray(Image.fromarray(full[...,c]).resize(small,Image.Resampling.BILINEAR),np.float32) for c in range(3)],-1)
        np.testing.assert_allclose(preview,expected,rtol=0,atol=2e-7)
        retained=((np.indices(alpha.shape).sum(0)%2)==0).astype(np.float32)
        upgraded['mask_roles']['edited_edge_light_applicability']=asset('retained',retained)
        kept=apply_stack(fp,v2,upgraded,run)[0]
        np.testing.assert_allclose(kept,fp+(full-fp)*retained[...,None],rtol=0,atol=1e-7)
        p=apply_stack(small_fp,v2,upgraded,run)[0]
        kd=(full-fp)*retained[...,None]
        kd=np.stack([np.asarray(Image.fromarray(kd[...,c]).resize(small,Image.Resampling.BILINEAR),np.float32) for c in range(3)],-1)
        np.testing.assert_allclose(p,small_fp+kd,rtol=0,atol=2e-7)
        # Saved asset closure and path-independent receipt survive relocation.
        owned=materialize_plan(root,run,upgraded)
        run['options']['appearance_revision']['design_reconstruction_plan']=owned
        run_dir=root/'outputs/runs/fixture';run_dir.mkdir(parents=True,exist_ok=True)
        (run_dir/'manifest.json').write_text(json.dumps(run),encoding='utf8')
        package=root/'fixture.cosplay';export_project(root,'fixture',package,include_raw=False,include_design=False)
        imported=import_project(root/'relocated',package,run_id='relocated-fixture')['run']
        relocated=imported['options']['appearance_revision']['design_reconstruction_plan']
        validate_layer(relocated['layers'][0],relocated,imported)
        np.testing.assert_array_equal(apply_stack(fp,relocated['layers'][0],relocated,imported)[0],apply_stack(fp,owned['layers'][0],owned,run)[0])
        for invalid in (float('nan'),-1.,1.1):
            broken=copy.deepcopy(owned);broken['layers'][0]['colour_strength']=invalid
            try:validate_layer(broken['layers'][0],broken,run)
            except ValueError:pass
            else:raise AssertionError('Invalid independent dose accepted')
        broken=copy.deepcopy(owned)
        broken['layers'][0]['params']['control_source_delta_sha256']='0'*64
        try:validate_layer(broken['layers'][0],broken,run)
        except ValueError:pass
        else:raise AssertionError('Stale corner bank accepted')
        # Updating a locator's byte SHA cannot authorize changed receipt facts.
        broken=copy.deepcopy(upgraded)
        changed=copy.deepcopy(receipt);changed['endpoint_contract']['TBS_refitted']=True
        changed_path=inputs/'changed-control-receipt.json'
        changed_path.write_text(json.dumps(changed,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        broken['layers'][0]['assets']['control_receipt'].update(path=str(changed_path),
            sha256=hashlib.sha256(changed_path.read_bytes()).hexdigest())
        try:validate_layer(broken['layers'][0],broken,run)
        except ValueError:pass
        else:raise AssertionError('Mutated control provenance accepted')
        try:prepare_owned_controls(plan,root/'cancelled',run=run,cancel=lambda:True)
        except InterruptedError:pass
        else:raise AssertionError('Cancelled control preparation proceeded')
        cancel_calls = 0
        def cancel_after_first_block():
            nonlocal cancel_calls
            cancel_calls += 1
            return cancel_calls >= 3
        try:prepare_owned_controls(plan,root/'cancelled-block',run=run,cancel=cancel_after_first_block)
        except InterruptedError:pass
        else:raise AssertionError('Control row-stream ignored cancellation after its first block')
        assert not list((root/'cancelled-block').rglob('*.npz')), 'Cancelled controls published a partial bank'
        assert not list(root.rglob('*.tmp.npy')), 'Control preparation left scratch arrays behind'
        assert not list(root.rglob('*.tmp.npz')), 'Control preparation left partial row archives behind'
    print('OK: premultiplied Lab homogeneity/tiny alpha; unsupported native pixels skip Lab with exact RGB bits; sparse/dense corners and doses bitwise equal; equal boundary dose preserved; native gamut reduction; current exact 00/11; live native inputs bitwise equal without reads/replay/mutation; freshly saved controls not unpacked again; default reopen checks saved bank; live shape/dtype/nonfinite/file-binding rejection; alias/wrong-Beauty rejection; no refit/parent mutation; measured interpolation coupling; native full/ROI; reduced native bank; retained coverage; portable v2 closure; dose/stale/cancel rejection')


if __name__=='__main__':
    main()
