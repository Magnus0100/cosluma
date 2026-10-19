"""CPU material permission, hand pigment and frozen coordinate regressions."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import registered_reference_style as style
import skin_illumination as skin
from registered_reference_style_smoke import frozen


def fixture(directory, *, domains=('arms', 'hands'), source_sha='a'*64,
            reference_sha='b'*64, goal=None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    h,w=288,224
    labels=np.zeros((h,w),np.uint8)
    labels[18:105,25:110]=3
    labels[125:204,142:207]=6
    labels[204:276,142:207]=7
    labels[120:280,20:120]=23
    lab=np.full((h,w,3),(.50,.005,.005),np.float32)
    lab[labels==3]=(.66,.020,.014)
    lab[labels==6]=(.50,.006,.030)
    lab[labels==7]=(.50,.012,.014)
    lab[218:267,165:175,1]=.065
    # Native texture remains a source signal. Bright/chromatic nail pigment
    # differs from the majority hand skin and must not become a skin swatch.
    yy,xx=np.indices((h,w))
    lab[...,0]+=(np.where((xx+yy)%2,.001,-.001)).astype(np.float32)
    nail=np.zeros((h,w),bool);nail[137:154,150:165]=True
    lab[nail,0]=.76; lab[nail,1]=.08;lab[nail,2]=.015
    values=style._from_lab(lab)
    alpha=np.ones((h,w),np.float32)
    alpha[242:247,189:194]=0
    protection=np.zeros((h,w),np.float32)
    protection[42:54,45:90]=1
    goal=goal or {'schema':'skin_cast_contract.v1','permission_source':'explicit_task_brief',
        'domains':{d:[{'instruction':'清除红绿杂光并统一肤色','operations':['remove_declared_unwanted_casts','harmonize_skin_chroma'],
            'cast_targets':['red','green']}] for d in domains},'vetoed_domains':[],
        'anchor':'reliable_face_skin_chroma_in_requested_lighting','target_temperature':'follow_user_intent',
        'rgb_changes_allowed':True,'reference_rgb_sampling_allowed':False,
        'activation_status':'not_evaluated'}
    assets={}
    for name,array,member in [('source_semantics',labels,'labels'),('identity_protection',protection,'protection'),('foreground_alpha',alpha,'values')]:
        path=directory/(name+'.npz');np.savez_compressed(path,**{member:array})
        assets[name]=style._asset(path,size=[w,h],native_size=[w,h],
            coordinate_space='source_canvas' if name=='foreground_alpha' else 'source_proxy')
    assets['source_semantics'].update(schema_version='source_semantic_binding.v1',source_sha256=source_sha,sampling='nearest')
    assets['identity_protection']['representation']='eyes_lips_protection'
    binding={'goal_sha256':skin._digest(goal),'source_sha256':source_sha,'native_size':[w,h],
        'source_assets':{k:v['sha256'] for k,v in assets.items()},'profile':skin.PROFILE,
        'hand_gain_fit_policy':skin.HAND_GAIN_FIT_POLICY,
        'hand_spatial_fit_policy':skin.HAND_SPATIAL_POLICY,
        'reference_sha256':reference_sha,'baseline_master_sha256':'c'*64,'producer_code':style.producer_code()}
    context={'schema':skin.SCHEMA,'goal':goal,'source_sha256':source_sha,'native_size':[w,h],
        'source_assets':assets,'binding':binding}
    output,evidence=skin.prepare(values,context,[w,h],directory/'owned-skin')
    fields=skin.validate(output,evidence,binding)
    return values,labels,alpha,nail,context,output,evidence,fields


def v2_layer(directory, values, context, assets, evidence):
    native=[values.shape[1],values.shape[0]]
    layer=frozen(Path(directory)/'base',np.zeros((7,9,12),np.float32),native)
    layer['id']=style.KIND
    layer['params']['schema']=style.SCHEMA_V2
    path=Path(layer['assets']['contract']['path'])
    receipt=json.loads(path.read_text('utf8'))
    receipt['schema']=style.SCHEMA_V2;receipt['profile']=style.profile_for(True)
    receipt['identity']['source_sha256']=context['binding']['source_sha256']
    receipt['identity']['reference_sha256']=context['binding']['reference_sha256']
    receipt['identity']['material_context']=context['binding']
    receipt['identity']['reference_contract']={'source_sha256':receipt['identity']['source_sha256'],
        'reference_sha256':receipt['identity']['reference_sha256']}
    receipt['identity']['producer_code']=context['binding']['producer_code']
    receipt['identity']['profile']=style.profile_for(True)
    receipt['material_illumination']={'schema':skin.SCHEMA,'receipt_asset':copy.deepcopy(assets['skin_receipt'])}
    style._write(path,receipt)
    layer['params']['identity_sha256']=style._digest(receipt['identity'])
    layer['assets']['contract']=style._asset(path)
    layer['assets'].update(assets)
    return layer


class SkinTests(unittest.TestCase):
    def test_spatial_cast_field_improves_opposite_red_green_without_smoothing_source(self):
        with tempfile.TemporaryDirectory() as t:
            source,labels,alpha,_,context,_,_,_=fixture(t)
            lab=style._to_lab(source)
            y,x=np.indices(labels.shape)
            hand=labels==6
            lab[...,1][hand]=(.04-.065*(x[hand]-142)/65).astype(np.float32)
            source=style._from_lab(lab)
            fields,evidence=skin.fit(source,labels,np.zeros(labels.shape,np.float32),alpha,context['goal'])
            fields['native_size']=np.array([224,288],np.int32)
            fields['opaque_packed']=np.packbits((alpha>.999).ravel(),bitorder='little')
            current={k:v for k,v in fields.items() if k not in ('hand_log_gain','hand_field_owner','hand_component_owner')}
            old,_=skin.apply(source,current,(0,0,224,288),(224,288))
            new,_=skin.apply(source,fields,(0,0,224,288),(224,288))
            central=hand&(y>145)&(y<174)&(x>151)&(x<198)
            anchor=np.array(evidence['bounded_anchor_ab'])
            old_error=np.linalg.norm(style._to_lab(old)[central,1:]-anchor,axis=-1)
            new_error=np.linalg.norm(style._to_lab(new)[central,1:]-anchor,axis=-1)
            self.assertLess(float(np.median(new_error)),float(np.median(old_error))*.8)
            self.assertTrue(np.array_equal(new[~np.isin(labels,[6,7])],old[~np.isin(labels,[6,7])]))
            self.assertLessEqual(float(np.abs(fields['hand_log_gain']).max()),.5+1e-7)
            self.assertGreater(int(np.count_nonzero(fields['hand_component_owner'])),0)
            spatial=evidence['hands'][0]['spatial_illumination']
            self.assertTrue(any(r['state']=='source_skin_illumination_estimated' for r in spatial))
            for r in spatial:
                if 'bounded_sample_L_shift' in r:
                    self.assertGreaterEqual(r['bounded_sample_L_shift'][0],-.015-2e-6)
                    self.assertLessEqual(r['bounded_sample_L_shift'][1],.035+2e-6)

    def test_unsupported_hand_island_has_constant_gain_not_borrowed_spatial_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            source,labels,alpha,_,context,_,_,_=fixture(t)
            labels=labels.copy();labels[110:113,130:135]=6
            fields,evidence=skin.fit(source,labels,np.zeros(labels.shape,np.float32),alpha,context['goal'])
            self.assertTrue(np.all(fields['hand_component_owner'][110:113,130:135]==0))
            sampled=skin._sample_hand_gain(fields,0,(130,110,135,113),(224,288))
            self.assertTrue(np.array_equal(sampled,np.broadcast_to(fields['gains'][0],sampled.shape)))
            self.assertTrue(any(row.get('reason')=='insufficient_reliable_skin_samples'
                for row in evidence['hands'][0]['spatial_illumination']))

    def test_associated_field_thin_ROI_gap_and_component_do_not_mix(self):
        fields={'gains':np.ones((2,3),np.float32),
            'hand_log_gain':np.zeros((2,8,8,3),np.float32),
            'hand_field_owner':np.zeros((2,8,8),np.int32),
            'hand_component_owner':np.zeros((16,16),np.int32)}
        fields['hand_field_owner'][0,:,:4]=1;fields['hand_field_owner'][0,:,4:]=2
        fields['hand_component_owner'][:,:8]=1;fields['hand_component_owner'][:,9:]=2
        fields['hand_log_gain'][0,:,:4]=(.5,-.5,0)
        fields['hand_log_gain'][0,:,4:]=(-.5,.5,0)
        full=skin._sample_hand_gain(fields,0,(0,0,64,64),(64,64))
        for box in ((29,0,30,64),(32,0,36,64),(35,16,42,17)):
            x0,y0,x1,y1=box
            self.assertTrue(np.array_equal(skin._sample_hand_gain(fields,0,box,(64,64)),full[y0:y1,x0:x1]))
        self.assertTrue(np.array_equal(full[:,32:36],np.ones((64,4,3),np.float32)))
        np.testing.assert_allclose(full[:,36:,0],2**(-.5),atol=1e-7,rtol=0)
        np.testing.assert_allclose(full[:,:32,0],2**.5,atol=1e-7,rtol=0)

    def test_empty_permission_block_returns_old_pixels_before_colour_work(self):
        with tempfile.TemporaryDirectory() as t:
            source,_,_,_,_,_,_,fields=fixture(t)
            box=(0,0,20,12);values=source[:12,:20]
            with patch.object(style,'_to_lab',side_effect=AssertionError('Empty block colour work')), \
                 patch.object(skin,'_sample_hand_gain',side_effect=AssertionError('Empty block field work')):
                actual,report=skin.apply(values,fields,box,(224,288))
            self.assertTrue(np.array_equal(values,actual))
            self.assertEqual(report['active_pixels'],0)

    def test_green_hand_separates_exposure_before_channel_box_projection(self):
        rgb=np.array([.03585,.04063,.02710],np.float32)
        target=style._to_lab(rgb)
        target[0]+=.03325;target[1:]=(.01365,.00551)
        desired=style._from_lab(target)
        gain,ev,report=skin._fit_hand_gain(rgb,desired)
        raw=np.log2(desired/rgb)
        old=rgb*np.exp2(np.clip(raw,-.5,.5))
        new=rgb*gain
        anchor=target[1:]
        self.assertLess(np.linalg.norm(style._to_lab(new)[1:]-anchor),
                        np.linalg.norm(style._to_lab(old)[1:]-anchor)*.3)
        np.testing.assert_allclose(ev-ev[1],raw-raw[1],atol=2e-7,rtol=0)
        self.assertLess(float(ev[1]),0)
        self.assertLessEqual(float(abs(ev).max()),.5+1e-7)
        self.assertEqual(report['colour_dose'],1.)
        self.assertFalse(report['colour_target_limited'])
        self.assertLessEqual(report['sample_RGB_lightness_shift'],.035+2e-7)

    def test_gain_noop_and_infeasible_colour_report_without_relaxing_bounds(self):
        neutral=np.array([.17,.15,.12],np.float32)
        gain,ev,report=skin._fit_hand_gain(neutral,neutral)
        self.assertTrue(np.array_equal(gain,np.ones(3,np.float32)))
        self.assertTrue(np.array_equal(ev,np.zeros(3)))
        self.assertEqual(report['sample_RGB_lightness_shift'],0)
        desired=neutral*np.array([4.,.25,4.])
        gain,ev,report=skin._fit_hand_gain(neutral,desired)
        self.assertLessEqual(float(abs(ev).max()),.5+1e-7)
        self.assertLess(report['colour_dose'],1)
        self.assertTrue(report['colour_target_limited'])
        self.assertLessEqual(report['sample_RGB_lightness_shift'],.035+2e-7)
        self.assertTrue(np.isfinite(gain).all())

    def test_extreme_illuminants_preserve_both_robust_sample_lightness_bounds(self):
        rgb=np.array([.15,.21,.18],np.float32)
        for raw in ((-4,0,-3),(-4,-4,-4),(4,0,3),(0,-4,0),(4,4,4)):
            gain,ev,report=skin._fit_hand_gain(rgb,rgb*np.exp2(np.array(raw)))
            self.assertLessEqual(float(abs(ev).max()),.5+1e-7)
            shift=report['sample_RGB_lightness_shift']
            self.assertGreaterEqual(shift,skin.PROFILE['minimum_L_shift']-2e-7)
            self.assertLessEqual(shift,skin.PROFILE['maximum_L_shift']+2e-7)
            self.assertTrue(np.isfinite(gain).all())

    def test_preserved_hand_red_lighting_veto_survives_new_gain_fitter(self):
        from creative_direction import skin_cast_contract_for
        goal=skin_cast_contract_for({'mode':'existing_reference',
            'brief':'清除手臂杂光，保留手部红色舞台灯，不要去除手部偏色',
            'constraints':{'reference_role':'source_conditioned_design','background_mode':'enhance_source'}})
        self.assertNotIn('hands',goal['domains'])
        with tempfile.TemporaryDirectory() as t:
            source,labels,_,nail,_,_,evidence,fields=fixture(t,goal=goal)
            result,_=skin.apply(source,fields,(0,0,224,288),(224,288))
            self.assertTrue(np.array_equal(source[labels==6],result[labels==6]))
            self.assertTrue(np.array_equal(source[nail],result[nail]))
            self.assertEqual(evidence['hands'],[])

    def test_shared_RGB_gamut_dose_keeps_bright_hand_specular_structure(self):
        with tempfile.TemporaryDirectory() as t:
            source,_,_,_,_,_,_,fields=fixture(t)
            source=source.copy()
            source[140:145,150:155]=(.96,.94,.92)
            source[146:151,150:155]=(.91,.89,.87)
            fields={k:v.copy() for k,v in fields.items()}
            fields['gains'][0]=np.array([2**.5,1.1,1.2],np.float32)
            fields['hand_log_gain'][0]=np.log2(fields['gains'][0])
            result,report=skin.apply(source,fields,(0,0,224,288),(224,288))
            self.assertLess(report['gamut_dose_minimum'],1)
            self.assertGreater(float(result[142,152].sum()),float(result[148,152].sum()))
            sampled_gain=skin._sample_hand_gain(fields,0,(0,0,224,288),(224,288))
            delta=source[142,152]*(sampled_gain[142,152]-1)
            ratios=(result[142,152]-source[142,152])/delta
            np.testing.assert_allclose(ratios,np.full(3,ratios[0]),atol=2e-6,rtol=0)
            self.assertLessEqual(float(result.max()),1)

    def test_colour_fit_policy_bound_but_legacy_receipt_uses_frozen_gain(self):
        with tempfile.TemporaryDirectory() as t:
            source,_,_,_,context,assets,evidence,fields=fixture(t)
            self.assertEqual(evidence['hand_gain_fit_policy'],skin.HAND_GAIN_FIT_POLICY)
            self.assertEqual(context['binding']['hand_gain_fit_policy'],skin.HAND_GAIN_FIT_POLICY)
            self.assertEqual(evidence['hands'][0]['gain_fit']['policy'],skin.HAND_GAIN_FIT_POLICY)
            old_evidence=copy.deepcopy(evidence)
            old_evidence.pop('hand_gain_fit_policy')
            old_evidence['binding'].pop('hand_gain_fit_policy')
            old_evidence['hands'][0].pop('gain_fit')
            old_evidence.pop('hand_spatial_fit_policy')
            old_evidence['binding'].pop('hand_spatial_fit_policy')
            legacy_fields={k:v for k,v in fields.items() if k not in
                ('hand_log_gain','hand_field_owner','hand_component_owner')}
            field=Path(t)/'old-field.npz';domain=Path(t)/'old-domain.npz'
            np.savez_compressed(field,**{k:legacy_fields[k] for k in ('arm_offset','gains')})
            np.savez_compressed(domain,**{k:v for k,v in legacy_fields.items() if k not in ('arm_offset','gains')})
            old_assets={**assets,'skin_field':skin._asset(field),'skin_domain':skin._asset(domain)}
            old_evidence['assets_sha256']={k:v['sha256'] for k,v in old_assets.items() if k!='skin_receipt'}
            expected=skin.apply(source,legacy_fields,(0,0,224,288),(224,288))[0]
            # Missing fit metadata is a frozen legacy result, never an excuse
            # to rerun the latest fitter or reinterpret its saved RGB gains.
            with patch.object(skin,'_fit_hand_gain',side_effect=AssertionError('Frozen replay refit')):
                legacy=skin.validate(old_assets,old_evidence,old_evidence['binding'])
                actual=skin.apply(source,legacy,(0,0,224,288),(224,288))[0]
                self.assertTrue(np.array_equal(expected,actual))
            bad=copy.deepcopy(evidence);bad['hand_gain_fit_policy']='unknown_policy'
            with self.assertRaisesRegex(ValueError,'policy'):
                skin.validate(assets,bad,context['binding'])

    def test_red_stripe_improves_with_exact_non_skin_and_native_alpha_hole(self):
        with tempfile.TemporaryDirectory() as t:
            source,labels,alpha,_,_,_,_,fields=fixture(t)
            out,report=skin.apply(source,fields,(0,0,224,288),(224,288))
            before,after=style._to_lab(source),style._to_lab(out)
            stripe=np.zeros(labels.shape,bool);stripe[220:265,166:174]=True
            self.assertLess(float(np.mean(after[...,1][stripe])),float(np.mean(before[...,1][stripe]))*.6)
            outside=~np.isin(labels,[6,7,11,15,16,20])
            self.assertTrue(np.array_equal(source[outside],out[outside]))
            self.assertTrue(np.array_equal(source[alpha==0],out[alpha==0]))
            self.assertTrue(report['outside_permission_exact'])
            self.assertGreaterEqual(float(out.min()),0);self.assertLessEqual(float(out.max()),1)

    def test_hand_and_nail_share_true_RGB_gain_without_chroma_flattening(self):
        with tempfile.TemporaryDirectory() as t:
            source,labels,_,nail,_,_,evidence,fields=fixture(t)
            out,_=skin.apply(source,fields,(0,0,224,288),(224,288))
            gain=skin._sample_hand_gain(fields,0,(0,0,224,288),(224,288))
            self.assertLessEqual(float(np.max(abs(np.log2(gain)))),.5+1e-6)
            # This nail is remote from the anatomical wrist transition. Its
            # pigment and texture receive the exact same illumination as skin.
            interior=nail.copy();interior[137:139]=False;interior[152:154]=False
            expected=source[interior]*gain[interior]
            np.testing.assert_allclose(out[interior],expected,rtol=0,atol=1e-7)
            self.assertFalse(np.array_equal(source[interior],out[interior]))
            self.assertGreater(float(np.mean(style._to_lab(out)[nail,1])),float(np.mean(style._to_lab(out)[labels==6,1]))+.02)
            self.assertLess(evidence['hands'][0]['sample_pixels'],int((labels==6).sum()))
            self.assertFalse(evidence['hands'][0]['nail_RGB_identity_claimed'])

    def test_arm_permission_does_not_silently_authorize_hands_or_face(self):
        with tempfile.TemporaryDirectory() as t:
            source,labels,_,_,_,_,_,fields=fixture(t,domains=('arms',))
            out,_=skin.apply(source,fields,(0,0,224,288),(224,288))
            self.assertTrue(np.array_equal(source[labels==6],out[labels==6]))
            self.assertTrue(np.array_equal(source[labels==3],out[labels==3]))
            self.assertFalse(np.array_equal(source[labels==7],out[labels==7]))
            self.assertTrue(np.array_equal(skin.apply(source,fields,(0,0,224,288),(224,288),0)[0],source))

    def test_explicit_red_only_preserves_non_red_delta_permissions(self):
        current=np.array([[[-.03,.04],[.05,-.02]]],np.float32)
        policy={'enabled':True,'general':False,'colours':{'red'}}
        delta=skin._permitted_ab_delta(current,np.array([.02,.01],np.float32),policy)
        self.assertEqual(float(delta[0,0,0]),0)
        self.assertLess(float(delta[0,1,0]),0)
        self.assertTrue(np.array_equal(delta[...,1],np.zeros((1,2),np.float32)))

    def test_geodesic_proximal_distance_never_shortcuts_through_instrument(self):
        # A U shaped hand has a distal finger geometrically near the arm but
        # connected only by a long path around a non-skin hole.
        hand=np.zeros((40,40),bool)
        hand[5:35,10:14]=True;hand[31:35,10:28]=True;hand[5:35,24:28]=True
        arm=np.zeros_like(hand);arm[1:5,10:14]=True
        weight=skin._proximal_weight(hand,arm,12)
        self.assertLess(float(weight[5,11]),.1)
        self.assertEqual(float(weight[5,25]),1.)
        self.assertEqual(float(weight[33,25]),1.)

    def test_v2_full_thin_ROI_preview_and_owned_portable_replay_agree(self):
        with tempfile.TemporaryDirectory() as t:
            source,_,_,_,context,assets,evidence,_=fixture(t)
            layer=v2_layer(t,source,context,assets,evidence)
            full=style.apply(source,layer,(0,0,224,288),(224,288))
            for box in ((141,130,208,260),(0,127,224,130),(160,125,161,288)):
                x0,y0,x1,y1=box
                actual=style.apply(source[y0:y1,x0:x1],layer,box,(224,288))
                self.assertTrue(np.array_equal(actual,full[y0:y1,x0:x1]),box)
            preview=source[::2,::2].copy()
            p=style.apply_with_report(preview,layer,(0,0,112,144),(112,144),preview=True)[0]
            p_roi=style.apply_with_report(preview[50:130,65:109],layer,(65,50,109,130),(112,144),preview=True)[0]
            self.assertTrue(np.array_equal(p_roi,p[50:130,65:109]))
            self.assertTrue(np.array_equal(style.apply(source,layer,(0,0,224,288),(224,288),strength=0),source))
            copied=Path(t)/'portable';copied.mkdir()
            for asset in layer['assets'].values():
                destination=copied/Path(asset['path']).name
                shutil.copyfile(asset['path'],destination);asset['path']=str(destination)
            self.assertTrue(np.array_equal(style.apply(source,layer,(0,0,224,288),(224,288)),full))
            Path(layer['assets']['skin_receipt']['path']).write_text('{}',encoding='utf8')
            with self.assertRaises(ValueError):style.validate_layer(layer)

    def test_v1_remains_bitexact_and_v2_tampered_field_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            source,_,_,_,context,assets,evidence,_=fixture(t)
            old=frozen(Path(t)/'v1',np.zeros((7,9,12),np.float32),(224,288))
            self.assertTrue(np.array_equal(style.apply(source,old,(0,0,224,288),(224,288)),source))
            layer=v2_layer(t,source,context,assets,evidence)
            Path(layer['assets']['skin_field']['path']).write_bytes(b'not a valid frozen field')
            with self.assertRaises(ValueError):style.apply(source,layer,(0,0,224,288),(224,288))

    def test_real_materialize_consumer_portable_rebind_JSON_bytes_without_weakening_identity(self):
        """Actual leaf producer -> materialize_plan -> renderer -> portable.

        The old path-copy-only fixture missed JSON reserialization. On Windows
        the producer/move/import paths may use no newline, CRLF and UTF8 LF.
        Nested byte identities must follow the actual owned receipt each time.
        """
        from current_stack_fixture import prepare_fixture
        from creative_direction import skin_cast_contract_for
        from design_reconstruction import materialize_plan,plan_for_run
        from float_appearance import render_source_float
        from native_source_baseline import source_for_plan
        from native_roi_render import render_native_roi
        from native_current_only_smoke import save
        from portable_project import export_project,import_project
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'workspace'
            run=prepare_fixture(root,native=(224,288))
            intent={'mode':'existing_reference','brief':'清除手臂和手的红绿杂光，统一手臂与手的肤色',
                'constraints':{'reference_role':'source_conditioned_design','background_mode':'enhance_source'}}
            goal=skin_cast_contract_for(intent)
            self.assertIsNotNone(goal)
            values,_,_,_,context,assets,evidence,_=fixture(root/'producer',
                source_sha=run['source']['sha256'],reference_sha=run['reference']['sha256'],goal=goal)
            layer=v2_layer(root/'producer',values,context,assets,evidence)
            original_receipt_sha=assets['skin_receipt']['sha256']
            run.update(task_intent=intent,task_plan={'ready':True,'intent':intent,
                'creative_direction':{'material_refinement':goal}})
            plan=plan_for_run(run);source=source_for_plan(plan)
            baseline,_=render_source_float(source,run)
            expected=style.apply(baseline,layer,(0,0,224,288),(224,288))
            plan['layers'].append(layer)
            owned=materialize_plan(root,run,plan)
            run['options']['appearance_revision']['design_reconstruction_plan']=owned
            owned_layer=next(row for row in owned['layers'] if row['kind']==style.KIND)
            receipt=style.validate_layer(owned_layer)
            nested=receipt['material_illumination']['receipt_asset']
            self.assertEqual(nested['sha256'],owned_layer['assets']['skin_receipt']['sha256'])
            self.assertEqual(style.sha256_file(nested['path']),nested['sha256'])
            # This must genuinely exercise rewritten bytes, not just a path copy.
            self.assertNotEqual(original_receipt_sha,owned_layer['assets']['skin_receipt']['sha256'])
            whole,operator=render_source_float(source,run)
            self.assertTrue(np.array_equal(whole,expected))
            roi,_=render_native_roi(run,(140,128,209,277))
            self.assertTrue(np.array_equal(roi,whole[128:277,140:209]))
            save(root,run,whole,operator)
            archive=Path(t)/'actual-v2.cosplayproj'
            export_project(root,run['run_id'],archive)
            imported=import_project(Path(t)/'fresh-empty',archive,run_id='imported-v2')['run']
            imported_layer=next(row for row in plan_for_run(imported)['layers'] if row['kind']==style.KIND)
            reopened=style.validate_layer(imported_layer)
            imported_record=reopened['material_illumination']['receipt_asset']
            self.assertTrue(Path(imported_record['path']).resolve().is_relative_to(Path(t)/'fresh-empty'))
            self.assertEqual(imported_record['sha256'],imported_layer['assets']['skin_receipt']['sha256'])
            imported_roi,_=render_native_roi(imported,(140,128,209,277))
            self.assertTrue(np.array_equal(imported_roi,roi))
            # Every link is still strict. A main contract cannot silently point
            # to another leaf just because the outer asset has a valid hash.
            bad=copy.deepcopy(reopened)
            bad['material_illumination']['receipt_asset']['sha256']='0'*64
            path=Path(imported_layer['assets']['contract']['path'])
            style._write(path,bad);imported_layer['assets']['contract']=style._asset(path)
            with self.assertRaises(ValueError):style.validate_layer(imported_layer)

    def test_no_anchor_or_unknown_permission_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            values,labels,_,_,context,_,_,_=fixture(t)
            labels=labels.copy();labels[labels==3]=0
            with self.assertRaisesRegex(ValueError,'anchor'):
                skin.fit(values,labels,np.zeros(labels.shape,np.float32),np.ones(labels.shape,np.float32),context['goal'])
            bad=copy.deepcopy(context['goal']);bad['vetoed_domains']=['arms']
            with self.assertRaisesRegex(ValueError,'permission'):
                skin.fit(values,labels,np.zeros(labels.shape,np.float32),np.ones(labels.shape,np.float32),bad)

    def test_checked_context_no_retrospective_permission_and_source_mask_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            _,_,_,_,context,_,_,_=fixture(t)
            intent={'constraints':{'reference_role':'source_conditioned_design'}}
            source_plan={'source_sha256':'a'*64,'canvas':{'source_oriented_size':[224,288]},
                'mask_roles':context['source_assets'],'source_baseline':{'schema':'appearance_source_baseline.v1',
                    'source_sha256':'a'*64,'assets':{'foreground_alpha':context['source_assets']['foreground_alpha']}}}
            run={'source':{'sha256':'a'*64},'task_intent':intent,
                'task_plan':{'ready':True,'intent':intent,'creative_direction':{}},
                'options':{'appearance_revision':{'design_reconstruction_plan':source_plan}}}
            self.assertIsNone(skin.checked_context(run,[224,288]))
            run['task_plan']['creative_direction']['material_refinement']=context['goal']
            with patch('creative_direction.skin_cast_contract_for',return_value=context['goal']):
                actual=skin.checked_context(run,[224,288])['binding']
                self.assertEqual(actual['source_assets'],context['binding']['source_assets'])
                self.assertEqual(actual['goal_sha256'],context['binding']['goal_sha256'])
                source_plan['source_sha256']='f'*64
                with self.assertRaises(ValueError):skin.checked_context(run,[224,288])
                source_plan['source_sha256']='a'*64
                Path(context['source_assets']['source_semantics']['path']).write_bytes(b'changed')
                with self.assertRaises(ValueError):skin.checked_context(run,[224,288])


if __name__=='__main__':unittest.main()
