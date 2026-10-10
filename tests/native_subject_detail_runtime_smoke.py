"""Finite synthetic full/ROI/preview and portable frozen subject-detail proof."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

import design_reconstruction as design
from native_subject_detail_runtime import KIND, SCHEMA, RECIPE, DOMAIN, array_sha, apply_detail
from native_roi_render import render_native_roi
from portable_project import export_project, import_project


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
    return record(path)


def fixture(root):
    from raw_support import RGB_DECODE_VERSION
    from display_transform import source_transform_for_precision
    root.mkdir(parents=True); size = [48, 32]; h, w = size[1], size[0]
    alpha = np.zeros((h,w), np.float32); alpha[4:28,8:44] = .999
    fp = np.ones((h,w,3), np.float32)*np.float32(.14)*alpha[...,None]
    environment = np.ones_like(fp)*np.float32(.21); automatic = np.ones_like(fp)*np.float32(.015)*alpha[...,None]
    assets = {}
    for name, values in [('foreground_premultiplied', fp), ('foreground_alpha', alpha),
            ('old_environment', environment), ('validity', np.ones((h,w), np.float32)),
            ('automatic_delta', automatic), ('colour_delta', automatic*.8), ('lightness_delta', automatic*.7)]:
        path = root/(name+'.npz'); np.savez_compressed(path, values=values)
        assets[name] = {**record(path), 'size': size, 'working_space': DOMAIN,
            'representation': 'premultiplied_delta', 'coordinate_space': 'source_canvas'}
    labels = root/'labels.npz'; np.savez_compressed(labels, labels=np.full((h,w),6,np.uint8))
    semantic = {**record(labels), 'size': size, 'native_size': size}
    source = root/'source.png'; reference = root/'reference.png'
    Image.new('RGB',size,(45,80,90)).save(source); Image.new('RGB',size,(40,90,100)).save(reference)
    run = {'kind':'single_input_run','input_mode':'single_source','run_id':'fixture','sample':'fixture',
        'revision':1,'root':str(root),'source':record(source),'reference':record(reference),'artifacts':{},
        'options':{'render_mode':'native_source_design'}}
    semantic['source_sha256']=run['source']['sha256']
    baseline = {role:assets[role] for role in ('foreground_premultiplied','foreground_alpha','old_environment')}
    stack = {'id':'appearance','kind':'reference_appearance_stack','enabled':True,'strength':1.,
        'params':{'schema':'reference_appearance_stack.v2'},
        'assets':{role:assets[role] for role in ('automatic_delta','colour_delta','lightness_delta')}}
    plan = {'schema_version':'design_reconstruction_plan.v1','enabled':True,'working_domain':DOMAIN,
        'source_sha256':run['source']['sha256'],'candidate_sha256':run['reference']['sha256'],
        'canvas':{'source_oriented_size':size,'source_proxy_size':size},
        'execution':{'source_reference':'paired_native_baseline'},
        'source_baseline':{'schema':'appearance_source_baseline.v1','working_domain':DOMAIN,
            'source_sha256':run['source']['sha256'],'decode_version':RGB_DECODE_VERSION,
            'source_precision':'raster8_srgb','source_display':source_transform_for_precision('raster8_srgb'),
            'source_display_applied_once':True,'assets':baseline},
        'mask_roles':{'foreground_alpha':assets['foreground_alpha'],'source_semantics':semantic},
        'layers':[{'id':'background','kind':'background_design','enabled':True,'strength':1.,
            'assets':{**baseline,'validity':assets['validity'], 'background':{**assets['old_environment'],
                'representation':'sampling_validity_associated_RGB','validity_associated':True}}}, stack]}
    weights = design.reference_stack_execution_weights(stack)['bank_weights']
    binding = {'source_sha256':plan['source_sha256'],'reference_sha256':plan['candidate_sha256'],
        'source_fp_sha256':assets['foreground_premultiplied']['sha256'],
        'source_alpha_sha256':assets['foreground_alpha']['sha256'],'alpha_sha256':assets['foreground_alpha']['sha256'],
        'semantic_sha256':semantic['sha256'],'appearance_banks':{'automatic_delta':assets['automatic_delta']['sha256']},
        'appearance_weights':weights,'native_size':size,'working_domain':DOMAIN}
    target = fp+automatic; arrays = {}; patches = []; full_delta = np.zeros_like(fp)
    for index, box in enumerate(([16,8,32,24],[36,20,42,26])):
        l,t,r,b = box; values = np.full((b-t,r-l,3),.004,np.float32)*alpha[t:b,l:r,None]
        key = f'delta_{index}'; arrays[key] = values; full_delta[t:b,l:r] = values
        patches.append({'native_box':box,'array_key':key,'delta_array_sha256':array_sha(values),
            'target_patch_sha256':array_sha(target[t:b,l:r]),'changed_pixels':int(np.any(values!=0,axis=-1).sum())})
    delta_path = root/'sparse-detail.npz'; np.savez_compressed(delta_path,**arrays)
    frozen = {'schema':SCHEMA,'recipe':RECIPE,'binding':binding,'delta':record(delta_path),
        'patches':patches,'donor_assets':baseline,'owner_asset':semantic,'approved_retouch_binding':None}
    receipt = {'schema':SCHEMA,'recipe':RECIPE,'state':'prepared','source_sha256':plan['source_sha256'],
        'reference_sha256':plan['candidate_sha256'],'qualification':{'qualified':True,'actual_paired_QA_verified':True},
        'checks':{'changed_pixels':sum(row['changed_pixels'] for row in patches),'alpha_changed':False,
            'source_geometry_changed':False,'approved_source_retouch_reapplied':False}}
    receipt['reviewed_windows']=[{'region':'left_hand','owner_class_ids':[6],
        'native_box':row['native_box']} for row in patches]
    layer = {'id':KIND,'kind':KIND,'enabled':True,'strength':1.,'params':{'schema':SCHEMA,'recipe':RECIPE,'binding':binding},
        'assets':{'frozen_delta':write(root/'frozen-detail.json',frozen),
            'qualification_receipt':write(root/'qualification.json',receipt)}}
    plan['layers'].append(layer); run['options']['appearance_revision'] = {'design_reconstruction_plan':plan}
    return run, target, full_delta, alpha, environment


def closure(value):
    from verify_saved_native_delivery import json_records
    pending = [value]; seen = set(); count = 0
    while pending:
        for item in json_records(pending.pop()):
            path = Path(item['path']); assert path.is_file(), str(path)
            assert sha(path) == item['sha256'], str(path)
            count += 1
            if path.suffix == '.json' and path not in seen:
                seen.add(path); pending.append(json.loads(path.read_text('utf8')))
    return count


def run_smoke():
    checks = 0
    def require(value):
        nonlocal checks
        assert value; checks += 1
    # Keep the fixture outside the source checkout; deletion affects only this
    # freshly created and resolved temporary directory.
    with tempfile.TemporaryDirectory(prefix='cos-subject-replay-') as temporary:
        base = Path(temporary); origin = base/'origin'
        run,target,delta,alpha,background = fixture(origin)
        plan = design.plan_for_run(run); layer = plan['layers'][-1]
        # Only the independently covered appearance-stack validator is mocked.
        # Its production pixel consumer, native renderer and new consumer run.
        with patch('reference_appearance_stack.validate_layer',return_value={}):
            values,receipt = apply_detail(target,layer,plan,run)
            require(np.array_equal(values,target+delta)); require(receipt['source_retouch_reapplied'] is False)
            require(np.array_equal(values[alpha==0],target[alpha==0]))
            roi = (21,13,29,22)
            clipped,_ = apply_detail(target[13:22,21:29],layer,plan,run,box=roi,canvas_size=(48,32))
            require(np.array_equal(clipped,values[13:22,21:29]))
            outside,_ = apply_detail(target[0:4,0:6],layer,plan,run,box=(0,0,6,4),canvas_size=(48,32))
            require(np.array_equal(outside,target[0:4,0:6]))
            preview_target = design._sample_channels(tuple(Image.fromarray(target[...,c]) for c in range(3)),(0,0,24,16),(24,16))
            reduced,_ = apply_detail(preview_target,layer,plan,run)
            expected_delta = np.stack([np.asarray(Image.fromarray(delta[...,c]).resize((24,16),Image.Resampling.BILINEAR)) for c in range(3)],-1)
            require(np.allclose(reduced,preview_target+expected_delta,rtol=0,atol=1e-8))
            require(np.array_equal(reduced[:2,:2],preview_target[:2,:2]))
            geometry = Image.fromarray(np.uint8(alpha*255))
            whole,execution = design.apply_plan(np.zeros_like(target),np.zeros_like(target),plan,geometry,run=run,include_optics=False)
            expected = target+delta+(1-alpha[...,None])*background
            require(np.array_equal(whole,expected))
            require(next(row for row in execution['layers'] if row['kind']==KIND)['applied'])
            seen_manual=[]
            def observe_manual(values,*args,**kwargs):
                if kwargs.get('branch')=='foreground':seen_manual.append(values.copy())
                return values
            with patch('float_appearance.apply_manual_layers_float',side_effect=observe_manual):
                design.apply_plan(np.zeros_like(target),np.zeros_like(target),plan,geometry,run=run,include_optics=False)
            require(len(seen_manual)==1 and np.array_equal(seen_manual[0],target+delta))
            native_roi,roi_execution = render_native_roi(run,roi)
            require(np.array_equal(native_roi,whole[13:22,21:29]))
            require(next(row for row in roi_execution['layers'] if row['kind']==KIND)['applied'])
            preview,_ = design.apply_plan(np.zeros((16,24,3),np.float32),np.zeros((16,24,3),np.float32),plan,
                geometry.resize((24,16)),run=run,include_optics=False)
            disabled = copy.deepcopy(run); disabled_plan = design.plan_for_run(disabled); disabled_plan['layers'][-1]['enabled']=False
            without,_ = design.apply_plan(np.zeros((16,24,3),np.float32),np.zeros((16,24,3),np.float32),disabled_plan,
                geometry.resize((24,16)),run=disabled,include_optics=False)
            require(np.allclose(preview-without,expected_delta,rtol=0,atol=2e-8))
            edited = copy.deepcopy(run); edited_plan=design.plan_for_run(edited); edited_plan['layers'][1]['strength']=.5
            unchanged,state = apply_detail(target,edited_plan['layers'][-1],edited_plan,edited)
            require(np.array_equal(unchanged,target) and 'doses_changed' in state['bypass_reason'])
            alpha_path = origin/'edited-alpha.npz'; np.savez_compressed(alpha_path,values=alpha*.9)
            edited = copy.deepcopy(run); edited_plan=design.plan_for_run(edited)
            edited_plan['mask_roles']['foreground_alpha']={**record(alpha_path),'size':[48,32]}
            unchanged,state=apply_detail(target,edited_plan['layers'][-1],edited_plan,edited)
            require(np.array_equal(unchanged,target) and 'alpha_changed' in state['bypass_reason'])
            edited=copy.deepcopy(run);edited['options']['denoise_strength']=.2
            unchanged,state=apply_detail(target,design.plan_for_run(edited)['layers'][-1],design.plan_for_run(edited),edited)
            require(np.array_equal(unchanged,target) and 'denoise_changed' in state['bypass_reason'])
            half=copy.deepcopy(layer);half['strength']=.5
            values,_=apply_detail(target,half,plan,run);require(np.array_equal(values,target+delta*.5))
            for damage in ('nonfinite','float64','wrong_shape','extra_array','missing_array','empty_patch'):
                bad=copy.deepcopy(run);bad_plan=design.plan_for_run(bad);bad_layer=bad_plan['layers'][-1]
                frozen=json.loads(Path(layer['assets']['frozen_delta']['path']).read_text('utf8'))
                with np.load(frozen['delta']['path'],allow_pickle=False) as saved_arrays:
                    arrays={key:saved_arrays[key].copy() for key in saved_arrays.files}
                key=frozen['patches'][0]['array_key']
                if damage=='nonfinite':arrays[key][0,0,0]=np.nan
                elif damage=='float64':arrays[key]=arrays[key].astype(np.float64)
                elif damage=='wrong_shape':arrays[key]=arrays[key][:-1]
                elif damage=='extra_array':arrays['undeclared']=arrays[key]
                elif damage=='missing_array':arrays.pop(key)
                elif damage=='empty_patch':arrays[key][:]=0
                if key in arrays:frozen['patches'][0]['delta_array_sha256']=array_sha(arrays[key])
                archive_path=origin/('bad-array-'+damage+'.npz');np.savez_compressed(archive_path,**arrays)
                frozen['delta']=record(archive_path)
                bad_layer['assets']['frozen_delta']=write(origin/('bad-array-'+damage+'.json'),frozen)
                try:apply_detail(target,bad_layer,bad_plan,bad)
                except ValueError:checks+=1
                else:raise AssertionError('Malformed sparse archive accepted: '+damage)
            for damage in ('delta_array_sha256','target_patch_sha256','overlap','out_of_bounds','count','source','reference','baseline','semantic'):
                bad=copy.deepcopy(run);bad_plan=design.plan_for_run(bad);bad_layer=bad_plan['layers'][-1]
                frozen=json.loads(Path(layer['assets']['frozen_delta']['path']).read_text('utf8'))
                if damage in ('delta_array_sha256','target_patch_sha256'):frozen['patches'][0][damage]='0'*64
                elif damage=='overlap':frozen['patches'][1]['native_box']=frozen['patches'][0]['native_box']
                elif damage=='out_of_bounds':frozen['patches'][0]['native_box']=[-1,8,32,24]
                elif damage=='count':frozen['patches'][0]['changed_pixels']+=1
                elif damage in ('source','reference'):bad[damage]['sha256']='0'*64
                elif damage=='baseline':bad_plan['source_baseline']['assets']['foreground_premultiplied']['sha256']='0'*64
                elif damage=='semantic':bad_plan['mask_roles']['source_semantics']['sha256']='0'*64
                bad_layer['assets']['frozen_delta']=write(origin/('bad-'+damage+'.json'),frozen)
                try:apply_detail(target,bad_layer,bad_plan,bad)
                except ValueError:checks+=1
                else:raise AssertionError('Frozen corruption accepted: '+damage)
            for damage in ('missing_windows','unreviewed_window','unsupported_owner','wrong_semantic_owner',
                    'identity_feature','nonopaque_current','nonopaque_donor'):
                bad=copy.deepcopy(run);bad_plan=design.plan_for_run(bad);bad_layer=bad_plan['layers'][-1]
                frozen=json.loads(Path(layer['assets']['frozen_delta']['path']).read_text('utf8'))
                qa=json.loads(Path(layer['assets']['qualification_receipt']['path']).read_text('utf8'))
                if damage=='missing_windows':qa.pop('reviewed_windows')
                elif damage=='unreviewed_window':qa['reviewed_windows'][0]['native_box']=[17,8,32,24]
                elif damage=='unsupported_owner':qa['reviewed_windows'][0]['owner_class_ids']=[3]
                elif damage=='wrong_semantic_owner':
                    path=origin/'wrong-owner-labels.npz';np.savez_compressed(path,labels=np.full((32,48),1,np.uint8))
                    semantic={**bad_plan['mask_roles']['source_semantics'],**record(path)}
                    bad_plan['mask_roles']['source_semantics']=semantic;frozen['owner_asset']=semantic
                    frozen['binding']['semantic_sha256']=semantic['sha256']
                elif damage=='identity_feature':
                    protect=np.zeros((32,48),np.float32);protect[8,16]=1
                    path=origin/'identity-protection.npz';np.savez_compressed(path,protection=protect)
                    bad_plan['mask_roles']['identity_protection']={**record(path),'size':[48,32]}
                elif damage=='nonopaque_current':
                    path=origin/'nonopaque-current.npz';np.savez_compressed(path,values=alpha*.9)
                    asset={**bad_plan['mask_roles']['foreground_alpha'],**record(path)}
                    bad_plan['mask_roles']['foreground_alpha']=asset;frozen['binding']['alpha_sha256']=asset['sha256']
                elif damage=='nonopaque_donor':
                    path=origin/'nonopaque-donor.npz';np.savez_compressed(path,values=alpha*.9)
                    asset={**bad_plan['source_baseline']['assets']['foreground_alpha'],**record(path)}
                    bad_plan['source_baseline']['assets']['foreground_alpha']=asset;frozen['donor_assets']['foreground_alpha']=asset
                    frozen['binding']['source_alpha_sha256']=asset['sha256']
                bad_layer['params']['binding']=copy.deepcopy(frozen['binding'])
                bad_layer['assets']['frozen_delta']=write(origin/('bad-owner-'+damage+'.json'),frozen)
                bad_layer['assets']['qualification_receipt']=write(origin/('bad-owner-'+damage+'-qa.json'),qa)
                try:apply_detail(target,bad_layer,bad_plan,bad)
                except ValueError:checks+=1
                else:raise AssertionError('Unqualified subject ownership accepted: '+damage)
            # Explicit alpha-zero leakage, including a bypassed user version.
            bad=copy.deepcopy(run);bad_plan=design.plan_for_run(bad);bad_layer=bad_plan['layers'][-1]
            frozen=json.loads(Path(layer['assets']['frozen_delta']['path']).read_text('utf8'))
            leak=np.ones((2,2,3),np.float32)*.002; leak_path=origin/'leak.npz';np.savez_compressed(leak_path,delta_0=leak)
            frozen['delta']=record(leak_path);frozen['patches']=[{'native_box':[0,0,2,2],'array_key':'delta_0',
                'delta_array_sha256':array_sha(leak),'changed_pixels':4,'target_patch_sha256':array_sha(target[:2,:2])}]
            bad_layer['assets']['frozen_delta']=write(origin/'leak.json',frozen)
            qa=json.loads(Path(layer['assets']['qualification_receipt']['path']).read_text('utf8'));qa['checks']['changed_pixels']=4
            bad_layer['assets']['qualification_receipt']=write(origin/'leak-qa.json',qa)
            for dose in (1.,.5):
                bad_plan['layers'][1]['strength']=dose
                try:apply_detail(target,bad_layer,bad_plan,bad)
                except ValueError:checks+=1
                else:raise AssertionError('Alpha-zero leak accepted')
            # A frozen approved source-stage record is owned and compared,
            # never executed again on the already prepared Fp donor.
            retouch_delta=origin/'retouch-delta.npz';np.savez_compressed(retouch_delta,delta_0=np.zeros((2,2,3),np.float32))
            retouch_layer={'schema_version':'native_retouch_delta.v1',
                'recipe':'source-native-compact-outlier-two-crop-approval-v1',
                'enabled':False,'source':copy.deepcopy(run['source']),'native_size':[48,32],
                'delta':record(retouch_delta),'patches':[],'approvals':[],
                'clone_retouch':{'enabled':False,'strokes':[]},'source_rgb_donor_only':True,
                'reference_rgb_used':False,'generation_used':False}
            retouch_binding={'schema':'automatic_native_retouch_binding.v1',
                'source_sha256':run['source']['sha256'],'native_size':[48,32],
                'working_domain':DOMAIN,'approved_patches':0,'owner_run_id':'original_owner',
                'layer':write(origin/'approved-retouch.json',retouch_layer)}
            plan['source_baseline']['native_source_retouch']=copy.deepcopy(retouch_binding)
            frozen=json.loads(Path(layer['assets']['frozen_delta']['path']).read_text('utf8'))
            frozen['approved_retouch_binding']=copy.deepcopy(retouch_binding)
            layer['assets']['frozen_delta']=write(origin/'frozen-detail-retouch.json',frozen)
            values,_=apply_detail(target,layer,plan,run);require(np.array_equal(values,target+delta))
            bad=copy.deepcopy(run);bad_plan=design.plan_for_run(bad)
            bad_plan['source_baseline']['native_source_retouch']['approved_patches']=1
            try:apply_detail(target,bad_plan['layers'][-1],bad_plan,bad)
            except ValueError:checks+=1
            else:raise AssertionError('Changed approved source retouch binding accepted')
            owned_run=copy.deepcopy(run);saved=base/'saved'
            owned_plan=design.materialize_plan(saved,owned_run,plan)
            owned_run['root']=str(saved);owned_run['options']['appearance_revision']['design_reconstruction_plan']=owned_plan
            owned_records=closure(owned_plan);require(owned_records>12)
            write(saved/'outputs/runs/fixture/manifest.json',owned_run)
            archive=base/'fixture.cosplay';export_project(saved,'fixture',archive)
            moved=import_project(base/'portable',archive,run_id='relocated')['run'];moved_plan=design.plan_for_run(moved)
            require(closure(moved_plan)>=owned_records)
            require(origin.resolve().is_relative_to(base.resolve()) and origin.name=='origin')
            shutil.rmtree(origin)
            require(not origin.exists())
            for relocated,relocated_plan in ((owned_run,owned_plan),(moved,moved_plan)):
                with patch('native_retouch_producer.apply_frozen_retouch',side_effect=AssertionError('Retouch applied twice')):
                    replay,replay_exec=design.apply_plan(np.zeros_like(target),np.zeros_like(target),relocated_plan,
                        geometry,run=relocated,include_optics=False)
                require(np.array_equal(replay,whole));require(next(row for row in replay_exec['layers'] if row['kind']==KIND)['applied'])
                roi_values,_=render_native_roi(relocated,roi);require(np.array_equal(roi_values,whole[13:22,21:29]))
                preview_values,_=design.apply_plan(np.zeros((16,24,3),np.float32),np.zeros((16,24,3),np.float32),relocated_plan,
                    geometry.resize((24,16)),run=relocated,include_optics=False)
                require(np.array_equal(preview_values,preview))
    return {'ok':True,'checks':checks,'full_roi_preview_shared_consumer':True,
        'native_support_before_preview_reduction':True,'zero_alpha_no_leak':True,
        'source_and_current_opaque_owner_reviewed_window_and_identity_exclusion':True,
        'approved_donor_retouch_identity_compared_across_relocation':True,
        'coverage_or_dose_change_explicit_bypass':True,'approved_source_retouch_reapplied':False,
        'materialized_and_portable_replay_after_original_root_deleted':True,
        'source_or_generation_models_required':False,'appearance_validator_mocked':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    report=run_smoke()
    if args.output:write(args.output,report)
    print(json.dumps(report,ensure_ascii=False))
