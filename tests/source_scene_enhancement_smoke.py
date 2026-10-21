"""CPU native-source scenery, once-only domains, provenance and portable closure."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
from PIL import Image

import source_scene_enhancement as scene
from native_array_storage import save_native_array
from display_transform import source_transform_for_precision,display_linear


def fixture(root,*,size=(192,160),regional=False):
    import creative_task as task
    from creative_direction import compile_direction
    from cos_node_registry import compile_workflow
    root=Path(root);root.mkdir(parents=True,exist_ok=True);w,h=size
    yy,xx=np.indices((h,w),dtype=np.float32)
    source=np.stack((.08+.08*xx/w,.12+.03*yy/h,.15+.01*((xx+yy)%2)),axis=-1).astype(np.float32)
    alpha=np.zeros((h,w),np.float32);alpha[h//2:,w*3//4:]=1
    def arr(name,values,**meta):
        p=root/(name+'.npz');save_native_array(p,values)
        return scene.asset(p,size=[w,h],native_size=[w,h],coordinate_space='source_canvas',sampling='bilinear',**meta)
    raw=root/'source.png';Image.fromarray(np.uint8(source*255)).save(raw)
    ref=root/'reference.png';Image.fromarray(np.uint8(np.clip(source*1.12,0,1)*255)).save(ref)
    source_asset=scene.asset(raw);ref_asset=scene.asset(ref)
    alpha_asset=arr('alpha',alpha);old=arr('old',np.where(alpha[...,None]==1,0,source),representation='straight_RGB')
    fp=arr('fp',source*alpha[...,None],representation='premultiplied_RGB')
    background=arr('candidate',source*1.12,representation='sampling_validity_associated_RGB',validity_associated=True)
    validity=arr('validity',np.ones((h,w),np.float32),representation='sampling_coverage')
    # The reference already exists. An opted-in native regional repair belongs
    # to this explicit import task, never the retired creation pipeline.
    intent={'schema_version':task.INTENT_SCHEMA,'mode':'existing_reference',
        'brief':'保留原场景；主效：两只旧纸蝴蝶。','constraints':{'preserve_background':False,'no_effects':False,'background_mode':'enhance_source'},
        'services':{'text_vision':False,'visual_review':False},
        'regional_refinement':{'maximum_submissions':int(regional),'coordinate_space':'source_canvas',
            'regions':[{'id':'bg-1','core_box':[128,128,640,640],'instruction':'唯一局部生成两只小旧纸蝴蝶，保留原场景。'}] if regional else []}}
    intent=task.normalize_task_intent(intent);direction=compile_direction(intent,source_sha256=source_asset['sha256'])
    checked={'schema_version':task.PLAN_SCHEMA,'mode':intent['mode'],'intent':intent,'ready':True,
        'image_budget':task.image_budget(intent),'creative_direction':direction,
        'inputs':{'source':source_asset,'reference':ref_asset},
        'plan_id':'a'*64,'execution_strategy':'layered_reference_transfer','preparation_policy':'prepare_current',
        'layer_workflow':compile_workflow(intent['mode'])}
    plan={'schema_version':'design_reconstruction_plan.v1','enabled':True,'source_sha256':source_asset['sha256'],
        'candidate_sha256':ref_asset['sha256'],'working_domain':'display_linear_sRGB',
        'canvas':{'source_oriented_size':[w,h],'source_proxy_size':[min(w,256),min(h,256)],
            'candidate_mapping':{'scene_registration_supported':True,
                'candidate_processing_proxy_to_final_proxy':np.eye(3).tolist(),
                'candidate_original_to_final_original':np.eye(3).tolist()}},
        'source_baseline':{'schema':'appearance_source_baseline.v1','source_sha256':source_asset['sha256'],
            'working_domain':'display_linear_sRGB','source_display_applied_once':True,
            'source_precision':'raster8_srgb','source_display':source_transform_for_precision('raster8_srgb'),
            'assets':{'old_environment':old,'foreground_premultiplied':fp,'foreground_alpha':alpha_asset}},
        'mask_roles':{'foreground_alpha':alpha_asset},'layers':[{'kind':'background_design','id':'background_design',
            'enabled':True,'strength':1.,'assets':{'background':background,'validity':validity,
                'old_environment':old,'foreground_premultiplied':fp}}]}
    run={'run_id':'run-fixture','source':source_asset,'reference':ref_asset,'task_intent':intent,'task_plan':checked,
        'creative_direction':direction,'options':{'provider':'qwen_image_2_1_uncensored_fast','appearance_revision':{}}}
    return run,plan,source,alpha


class SourceTests(unittest.TestCase):
    def test_real_freeze_keeps_scene_execution_provenance_and_background_alpha(self):
        import reference_appearance_prepare as freeze
        from reference_colour_fit import REGIONS
        from runtime_resources import code_path
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run,plan,source,alpha=fixture(root/'in')
            plan=scene.prepare(root,run,plan,root/'scene')
            expected=copy.deepcopy(plan['provenance']['source_scene_enhancement'])
            run['options']['appearance_revision']['design_reconstruction_plan']=plan
            matching=root/'matching';matching.mkdir()
            weights=matching/'source-weights.npz';np.savez(weights,values=np.zeros((*alpha.shape,len(REGIONS)),np.float32))
            parameters={'binding':{'source':run['source']['sha256'],'reference':run['reference']['sha256'],
                'alpha':plan['mask_roles']['foreground_alpha']['sha256'],
                'algorithm_sha256':freeze.sha(code_path('reference_colour_fit.py')),
                'source_weights_sha256':freeze.sha(weights)}}
            scene.write(matching/'parameters.json',parameters)
            scene.write(matching/'composite-receipt.json',{'parameters_sha256':freeze.sha(matching/'parameters.json'),
                'renderer_sha256':freeze.sha(code_path('render_reference_colour_composite.py')),'output_sha256':'c'*64})
            bg=scene.background(plan)['assets'];native=tuple(plan['canvas']['source_oriented_size'])
            with np.load(plan['source_baseline']['assets']['old_environment']['path']) as z:old=z['values'].copy()
            with np.load(plan['source_baseline']['assets']['foreground_premultiplied']['path']) as z:fp=z['values'].copy()
            def matching_math(working,delta,*args,**kwargs):
                delta.fill(0);return True,True,{},{}
            def controls(current,out,**kwargs):
                out.mkdir(parents=True,exist_ok=True);p=out/'receipt.json'
                scene.write(p,{'endpoint_contract':{},'gamut_dose':{},'original_endpoint_exceptions':{}})
                current['layers'][0]['assets']['control_receipt']=scene.asset(p)
                current['layers'][0]['params']['control_recipe']='CPU-fixture-control'
                return current
            patches=[patch('render_reference_colour_composite.validate_material_inputs'),
                patch('reference_colour_blend.validate_layer',return_value=(np.zeros((*alpha.shape,6),np.float32),{})),
                patch('reference_scene_balance.validate_scene_layer',return_value=(np.zeros((*alpha.shape,4),np.float32),{},source)),
                patch('render_reference_colour_composite.field_owners',return_value=None),
                patch('portrait_beauty_layer.hair_correction_guard_for_run',return_value=None),
                patch('native_source_baseline.source_for_reference_prepare',return_value=(source.copy(),{'source_precision':'raster8_srgb','decode_version':'CPU-fixture'},plan['source_baseline']['source_display'])),
                patch.object(freeze,'_matching_donor_source',return_value=(source.copy(),{'source_endpoint_cache_hit':True,'beauty_replay_performed':False})),
                patch('render_reference_colour_composite.sample_reference_field',return_value=np.zeros((*alpha.shape,4),np.float32)),
                patch('render_reference_colour_composite.donor_fields',return_value=np.zeros((*alpha.shape,3),np.float32)),
                patch('reference_support.donor_support_report',return_value={}),
                patch('native_source_baseline.display_pair_identity',return_value={}),
                patch('native_source_baseline.prepare_display_pair_for_freeze',return_value=(old,fp,np.zeros_like(alpha),0,0.,None)),
                patch('native_source_baseline.publish_display_pair_assets',return_value={}),
                patch('portrait_beauty_layer.apply_for_run',return_value=(fp,{'applied':False})),
                patch('reference_pre_scene_cache.open_reader',return_value=None),
                patch.object(freeze,'_freeze_matching_native',side_effect=matching_math),
                patch.object(freeze,'preserve_contact_layers',return_value=([],{})),
                patch('native_source_baseline.retouch_baseline_record',return_value=None),
                patch('native_receiver_preservation.prepare_native_receiver_preservation',side_effect=lambda run,plan,out:{'design_reconstruction_plan':plan,'receiver_preservation':{'state':'unknown','qualification':{}}}),
                patch('reference_appearance_controls.prepare_owned_controls',side_effect=controls)]
            with ExitStack() as stack:
                for item in patches:stack.enter_context(item)
                frozen=freeze.freeze(run,matching,root/'frozen')['design_reconstruction_plan']
            self.assertEqual(frozen['provenance']['source_scene_enhancement'],expected)
            self.assertEqual(scene.background(frozen)['assets']['background'],bg['background'])
            self.assertEqual(scene.background(frozen)['assets']['validity'],bg['validity'])
            self.assertEqual(frozen['mask_roles']['foreground_alpha'],plan['mask_roles']['foreground_alpha'])
            self.assertEqual(scene.validate_observation(frozen,run)['schema'],scene.SCHEMA)
            run['options']['appearance_revision']['design_reconstruction_plan']=frozen
            # Admit a snapshot actually written by the freeze producer. A
            # coordinator-only fixture concealed divergent recipe constants.
            import reference_pipeline as coordinator
            from reference_recipe import RECIPE
            layer=frozen['layers'][0]
            snapshot_asset=layer['assets']['preparation_snapshot']
            snapshot_path=Path(snapshot_asset['path'])
            snapshot=json.loads(snapshot_path.read_text(encoding='utf8'))
            self.assertEqual(snapshot['recipe'],RECIPE)
            self.assertEqual(frozen['recipe'],RECIPE)
            identity={'recipe':RECIPE,'code':coordinator._code_identity(root)}
            result={'design_reconstruction_plan':frozen}
            coordinator._stamp_inputs(root,result,identity)
            with patch('reference_appearance_stack.validate_layer'):
                self.assertTrue(coordinator._prepared_matches(root,run,result,input_identity=identity))
                before=snapshot_path.read_bytes()
                original_sha=snapshot_asset['sha256']
                stale=json.loads(before);stale['recipe']='historical-incompatible-recipe'
                snapshot_path.write_text(json.dumps(stale),encoding='utf8')
                snapshot_asset['sha256']=freeze.sha(snapshot_path)
                self.assertFalse(coordinator._prepared_matches(root,run,result,input_identity=identity))
                snapshot_path.write_bytes(before);snapshot_asset['sha256']=original_sha
            from single_source_workflow import _record_regional_execution
            _record_regional_execution(run)
            self.assertEqual(run['generation_budget']['regional_refinement_image_submissions'],0)
            frozen['provenance'].pop('source_scene_enhancement')
            _record_regional_execution(run)
            self.assertTrue(frozen['provenance']['source_scene_enhancement']['metadata_recovered_from_verified_frozen_receipt'])
            self.assertEqual(run['generation_budget']['total_image_submissions'],0)
            frozen['provenance'].pop('source_scene_enhancement')
            run['task_plan']['image_budget']['regional_refinement']=1
            with self.assertRaisesRegex(ValueError,'no source enhancement execution receipt'):
                _record_regional_execution(run)

    def test_actual_prepare_scene_enhance_branch_skips_SR_and_cache_counts(self):
        import design_reconstruction_pipeline as pipeline
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run,template,source,alpha=fixture(root/'in')
            run['source']['oriented_size']=list(source.shape[1::-1])
            folder=root/'analysis';folder.mkdir();Image.open(run['source']['path']).save(folder/'source-srgb.png')
            labels=np.zeros(alpha.shape,np.uint8);labels[alpha>0]=23
            labels_path=folder/'labels.npz';np.savez(labels_path,labels=labels)
            analysis={'analysis_directory':str(folder),'semantic_result':{'labels_path':str(labels_path)},'cache_hit':True}
            foreground={'alpha_path':template['mask_roles']['foreground_alpha']['path'],
                'old_background_path':template['source_baseline']['assets']['old_environment']['path'],
                'premultiplied_foreground_path':template['source_baseline']['assets']['foreground_premultiplied']['path'],
                'unknown_edge_support_path':template['mask_roles']['foreground_alpha']['path'],
                'native_source_size':list(source.shape[1::-1]),'identity':{'synthetic':True},
                'provenance':{'source_precision':{'source_precision':'raster8_srgb'}}}
            scene_assets={'background':copy.deepcopy(scene.background(template)['assets']['background']),
                'background_validity':copy.deepcopy(scene.background(template)['assets']['validity']),
                'mapping':template['canvas']['candidate_mapping'],'layers':[],'provenance':{}}
            from raw_support import RGB_DECODE_VERSION
            ready_identity={'source':run['source']['sha256'],'candidate':run['reference']['sha256'],'synthetic_CPU':True}
            face=root/'face.npz';save_native_array(face,np.zeros_like(alpha))
            patches=[patch.object(pipeline,'local_runtime_paths',return_value={'python':sys.executable}),
                patch.object(pipeline,'_model_registry_receipt',return_value={'sha256':'fixture'}),
                patch.object(pipeline,'ready',return_value=True),patch.object(pipeline,'identity_for_run',return_value=ready_identity),
                patch.object(pipeline,'prepare_local_appearance_pair',return_value=(analysis,analysis)),
                patch('local_scene_analysis.prepare_local_scene',return_value={'contact':{}}),
                patch.object(pipeline,'source_silhouette',return_value=(Image.fromarray(np.uint8(alpha*255)),{})),
                patch('source_foreground_cache.prepare_source_foreground',return_value=foreground),
                patch('native_source_baseline.retouch_baseline_record',return_value=None),
                patch('design_scene_assets.removal_inputs',return_value={'extra_removal_mask':np.zeros(alpha.shape,bool)}),
                patch('local_floor_pollution.prepare_candidate_floor_pollution',return_value={'analysis_status':'unsupported'}),
                patch('local_floor_pollution.protect_verified_reflections',side_effect=lambda removal,*a,**k:(removal,{})),
                patch('design_scene_assets.prepare_design_scene_assets',return_value=scene_assets),
                patch('scene_design.build_scene_design',return_value={}),
                patch('scene_design.bind_scene_environment',side_effect=lambda design,*a:design),
                patch('local_face_protection.prepare_face_protection_pair',return_value=(
                    {'field_path':str(face),'size':list(alpha.shape[::-1])},
                    {'field_path':str(face),'size':list(alpha.shape[::-1])})),
                patch('source_semantic_binding.bind_source_semantics',return_value={'path':str(labels_path),'sha256':'fixture'}),
                patch('design_scene_assets.prepare_native_receivers',side_effect=lambda root,run,plan:plan),
                patch.object(pipeline,'prepare_source_responsibility',return_value={}),
                patch('foreground_effects.prepare_effects',side_effect=lambda root,run,plan:{'plan':plan,'requests':[]}),
                patch.object(pipeline,'_prepare_background_stage',side_effect=AssertionError('enhance source cleaned a candidate')),
                patch.object(pipeline,'_prepare_background_sr',side_effect=AssertionError('enhance source ran SR'))]
            with ExitStack() as stack:
                for p in patches:stack.enter_context(p)
                result=pipeline.prepare_scene(root,run)
                stats=result['design_reconstruction_plan']['provenance']['source_scene_enhancement']
                self.assertEqual(stats['new_model_submissions'],0)
                with patch.object(pipeline,'_current_plan_cache',return_value=True):
                    cached=pipeline.prepare_scene(root,run)
                self.assertTrue(cached['cache_hit']);self.assertEqual(cached['design_reconstruction_plan']['provenance']['source_scene_enhancement']['provider_ids'],[])

    def test_native_source_and_cache_domains(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run,plan,source,alpha=fixture(root/'in')
            with patch('regional_scene_worker.subprocess.Popen',side_effect=AssertionError('zero-budget launched a model')):
                result=scene.prepare(root,run,plan,root/'out')
                again=scene.prepare(root,run,plan,root/'out')
            receipt=scene.validate_observation(result,run)
            self.assertEqual(receipt['output_background']['sha256'],scene.background(again)['assets']['background']['sha256'])
            self.assertEqual(result['source_baseline'],plan['source_baseline']);self.assertEqual(result['mask_roles'],plan['mask_roles'])
            self.assertEqual(result['provenance']['source_scene_enhancement']['new_model_submissions'],0)
            self.assertEqual(result['provenance']['source_scene_enhancement']['provider_ids'],[])
            with np.load(receipt['output_background']['path']) as z:actual=z['values']
            self.assertTrue(np.all(actual[alpha==1]==0));self.assertGreater(float(actual[alpha==0].mean()),0)
            self.assertFalse(receipt['source_display_consumed_here'])
            raw=copy.deepcopy(plan);raw['working_domain']='scene_linear_sRGB'
            raw['source_baseline'].update(schema='source_preparation_baseline.v1',working_domain='scene_linear_sRGB',
                source_precision='raw16_linear',source_display_applied_once=False,
                source_display=source_transform_for_precision('raw16_linear'))
            fresh=scene.prepare(root,run,raw,root/'raw')
            self.assertTrue(scene.validate_observation(fresh,run)['source_display_consumed_here'])
            self.assertIsNone(scene.transform_for(plan));self.assertIsNotNone(scene.transform_for(raw))

    def test_real_audit_source_ancestry_and_portable_json_dependencies(self):
        from scene_stage_audit import resolve_observation,audit_scene_stages
        from design_reconstruction import materialize_plan
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run,plan,_,_=fixture(root/'in');result=scene.prepare(root,run,plan,root/'cache')
            self.assertEqual(resolve_observation(root,run,result)['origin'],'source_native_scene')
            run['options']['appearance_revision']['design_reconstruction_plan']=result
            report=audit_scene_stages(root,run,root/'audit')
            self.assertEqual(report['status'],'audited_no_repair');self.assertNotIn('sr_provenance',report['observation'])
            owned=materialize_plan(root,run,result)
            shutil.rmtree(root/'cache')
            receipt=scene.validate_observation(owned,run)
            self.assertTrue(Path(receipt['base_background']['path']).is_file())

    def test_low_frequency_gain_and_exact_zero_permission(self):
        rng=np.random.default_rng(17);source=rng.uniform(.02,.6,(32,64,3)).astype(np.float32)
        target=source*1.2;a=np.zeros((32,64),np.float32);v=np.ones_like(a)
        gain,report=scene.fit_gain(source,target,a,v,registered=True)
        self.assertLessEqual(abs(report['exposure_EV']),.5)
        self.assertTrue(np.allclose(scene.adjust(source,gain,np.zeros_like(a)),source))
        self.assertTrue(np.array_equal(scene.fit_gain(source,target,a,v,registered=False)[0],np.ones(3)))
        adjusted=scene.adjust(source,gain,v);self.assertTrue(np.all((adjusted>=0)&(adjusted<=1)))
        self.assertTrue(np.allclose(adjusted/source,gain,atol=1e-6))

    def test_edited_alpha_cannot_certify_unobserved_source_background(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run,plan,_,alpha=fixture(root/'in')
            path=root/'edited.npz';save_native_array(path,np.zeros_like(alpha))
            plan['mask_roles']['foreground_alpha']=scene.asset(path,size=list(alpha.shape[::-1]))
            with self.assertRaisesRegex(ValueError,'original subject coverage'):
                scene.prepare(root,run,plan,root/'out')


if __name__=='__main__':unittest.main()
