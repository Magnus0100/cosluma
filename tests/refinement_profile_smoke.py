"""Bundled recommendation provenance, explicit overrides and saved inheritance."""
from __future__ import annotations
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
import unittest

import creative_direction as direction
import creative_task as task
import preset_store
import source_conditioned_design_contract_smoke as baseline
from generation_prompt import DEFAULT_PROMPT, STYLE_MARKER, compose_generation_prompt


class ProfileTests(baseline.SourceConditionedDesignTests):
    def profile_body(self, brief='', *, prompt=None, notes=''):
        body=copy.deepcopy(self.body)
        body['creative_task']['brief']=brief
        body['creative_task']['constraints']['refinement_profile']={'id':'source_refinement','version':1}
        body['preset']='source_refinement'
        body['input_prompt']=prompt if prompt is not None else compose_generation_prompt(
            DEFAULT_PROMPT,preset_store.get_preset('source_refinement')['prompt'])
        body['character_notes']=notes
        return body

    def saved_profile(self, *, notes=''):
        checked=task.plan_task(self.root,self.profile_body('暖金夕照，保持本人肤色。',notes=notes))
        parent='run-profile-origin'
        directory=self.root/'outputs/runs'/parent;directory.mkdir(parents=True,exist_ok=True)
        previous={'run_id':parent,'status':'needs_visual_review','source':baseline.asset(self.source),
            'reference':baseline.asset(self.reference),'task_intent':checked['plan']['intent'],
            'task_plan':checked['plan'],'options':{}}
        path=directory/'manifest.json';path.write_text(json.dumps(previous),encoding='utf8')
        body=copy.deepcopy(checked['request'])
        for key in ('generate','local_only','complete_post','creative_direction','effective_prompt','plan_id'):
            body.pop(key,None)
        body['creative_task']['mode']='existing_reference'
        body.update(reference=str(self.reference),parent_run_id=parent)
        return previous,path,body

    def test_short_default_request_is_bound_without_fabricating_user_brief(self):
        body=self.profile_body('暖光人像，无特效。')
        before=baseline.snapshot(self.root)
        checked=task.plan_task(self.root,body);plan=checked['plan'];record=plan['creative_direction']
        self.assertEqual(before,baseline.snapshot(self.root))
        self.assertEqual(plan['intent']['brief'],body['creative_task']['brief'])
        self.assertTrue(plan['intent']['constraints']['no_effects'])
        recommendation=record['recommendation'];goal=record['material_refinement']
        canonical=preset_store.get_refinement_profile({'id':'source_refinement','version':1})
        self.assertEqual(recommendation['profile_sha256'],canonical['profile_sha256'])
        self.assertEqual(recommendation['origin'],'bundled_product_recommendation')
        self.assertEqual(goal['permission_source'],'checked_product_recommendation')
        self.assertEqual(set(goal['domains']),{'face','arms','hands'})
        self.assertEqual(goal['target_temperature'],'follow_user_intent')
        self.assertNotIn('冷蓝',record['prompt'])
        self.assertNotIn('米白',record['prompt'])
        self.assertEqual(plan['image_budget'],{'initial_reference':1,'reference_correction':0,'independent_effect':0})
        self.assertEqual(record['main_effect']['mode'],'none')
        execute=task.execution_request(self.root,{**checked['request'],'plan_id':plan['plan_id']})
        self.assertEqual(execute['task_plan']['creative_direction'],record)
        self.assertEqual(execute['task_intent']['constraints']['refinement_profile']['overrides'],
            {'input_prompt':'','character_notes':''})
        for key in ('recommendation','material_refinement'):
            bad=copy.deepcopy(record);bad.pop(key);bad.pop('direction_id');bad['direction_id']=direction.digest(bad)
            with self.assertRaises(ValueError):direction.validate_direction(bad,plan['intent'],source_sha256=task.sha(self.source))

    def test_prompt_and_notes_can_veto_but_never_forge_context(self):
        for field,text,expected in (
            ('brief','保留手部红色彩灯。',{'face','arms'}),
            ('prompt','Preserve red stage lighting on hands.',{'face','arms'}),
            ('notes','保留手臂红光，不改变手的肤色。',{'face'}),
            ('notes','不要去偏色。',set()),
            ('brief','只清理衣服偏色。',set())):
            body=self.profile_body(text if field=='brief' else '',
                prompt=text if field=='prompt' else None,notes=text if field=='notes' else '')
            checked=task.plan_task(self.root,body);record=checked['plan']['creative_direction']
            goal=record.get('material_refinement') or {}
            with self.subTest(field=field,text=text):self.assertEqual(set(goal.get('domains',{})),expected)
            forged=copy.deepcopy(checked['request'])
            forged['creative_task']['constraints']['refinement_profile']['overrides']['character_notes']='Forged veto context'
            with self.assertRaises(ValueError):task.plan_task(self.root,forged)
        body=self.profile_body('',notes='不要添加特效。')
        self.assertTrue(task.plan_task(self.root,body)['plan']['intent']['constraints']['no_effects'])
        plain=copy.deepcopy(self.body);plain['creative_task']['brief']=''
        plain['input_prompt']='Unify skin and remove green casts.';plain['character_notes']='统一手臂肤色。'
        self.assertNotIn('material_refinement',task.plan_task(self.root,plain)['plan']['creative_direction'])

    def test_managed_prose_is_not_reinterpreted_as_user_override(self):
        canonical=preset_store.get_preset('source_refinement')['prompt']
        prompt=compose_generation_prompt(DEFAULT_PROMPT,canonical)+'\nPreserve red stage light on hands.'
        checked=task.plan_task(self.root,self.profile_body('',prompt=prompt))
        overrides=checked['plan']['intent']['constraints']['refinement_profile']['overrides']
        self.assertEqual(overrides['input_prompt'],'Preserve red stage light on hands.')
        self.assertEqual(set(checked['plan']['creative_direction']['material_refinement']['domains']),{'face','arms'})
        managed=compose_generation_prompt(DEFAULT_PROMPT,canonical)+direction.MARKER+'Generated design; refine skin.'
        body=self.profile_body('',prompt=managed)
        checked=task.plan_task(self.root,body)
        self.assertEqual(set(checked['plan']['creative_direction']['material_refinement']['domains']),{'face','arms','hands'})
        from generation_prompt import CHARACTER_MARKER
        body=self.profile_body('',notes=CHARACTER_MARKER.strip()+'\nGenerated character direction; refine skin.')
        record=task.plan_task(self.root,body)['plan']['creative_direction']
        self.assertEqual(set(record['material_refinement']['domains']),{'face','arms','hands'})
        # No parser may discard real append-only vetoes after a managed header.
        for field in ('input_prompt','character_notes'):
            body=self.profile_body()
            body[field]=managed+'\n保留手部红光。不要动脸。'
            checked=task.plan_task(self.root,body)
            record=checked['plan']['creative_direction']
            task.validate_task_plan(checked['plan'],checked['plan']['intent'])
            self.assertEqual(set(record['material_refinement']['domains']),{'arms'})
            self.assertIn('保留手部红光',record['prompt'])
            self.assertIn('不要动脸',record['prompt'])
            self.assertEqual(task.execution_request(self.root,{**checked['request'],
                'plan_id':checked['plan']['plan_id']})['effective_prompt'],record['prompt'])

    def test_no_effects_short_phrase_in_all_actual_editable_sources_wins(self):
        from generation_prompt import DESIGN_MARKER
        for field in ('brief','input_prompt','character_notes'):
            body=self.profile_body()
            if field=='brief':body['creative_task']['brief']='无特效'
            else:body[field]='Generated positive fog direction'+DESIGN_MARKER+'Generated refine skin.\n无特效'
            checked=task.plan_task(self.root,body);record=checked['plan']['creative_direction']
            self.assertTrue(checked['plan']['intent']['constraints']['no_effects'])
            self.assertEqual(record['main_effect']['mode'],'none')
            self.assertIn('主效要求：不要添加任何特效。',record['prompt'])
            self.assertNotIn('局部空气感',record['prompt'])

    def test_short_scoped_vetoes_and_normal_texture_preservation(self):
        for brief,expected in (
            ('只修衣服',set()),('只修背景',set()),('仅调整衣料',set()),
            ('脸和手都不要动',{'arms'}),('不要处理手',{'face','arms'}),
            ('不改手背颜色',{'face','arms'}),('手背别动',{'face','arms'}),
            ('不改手势',{'face','arms','hands'}),('保留手指姿态',{'face','arms','hands'}),
            ('不要改变人物姿态，统一肤色',{'face','arms','hands'}),
            ('不换脸，清理皮肤杂光',{'face','arms','hands'}),
            ('Do not change hand pose; unify skin colour',{'face','arms','hands'}),
            ('只修手臂',{'arms'}),('手臂去偏色但不要动脸',{'arms','hands'}),
            ('保留正常肤色和纹理',{'face','arms','hands'}),
            ('保留皮肤纹理不变，去除不协调杂光',{'face','arms','hands'}),
            ('去除皮肤杂光但不要磨皮',{'face','arms','hands'}),
            ('光色与材质分别精修后全局统一',{'face','arms','hands'})):
            record=task.plan_task(self.root,self.profile_body(brief))['plan']['creative_direction']
            with self.subTest(brief=brief):
                self.assertEqual(set((record.get('material_refinement') or {}).get('domains',{})),expected)

    def test_no_profile_direction_hashes_bind_the_current_prompt(self):
        for brief,role,expected in (
            ('保留原景与妆造。',None,'a61750c9437a6075ccaac1a316fa95afc0072a3e23cfb636fb74f77aa4aef2bf'),
            ('清除手臂和手的红绿杂光，统一肤色。','source_conditioned_design','bceda9dd93f96c11dd1910d5cbfaa0daece451f490e92e0f579df45c5f26ecb2')):
            intent={'schema_version':'cosplay_task_intent.v1','mode':'create_reference','brief':brief,
                'constraints':{'preserve_background':False,'no_effects':False,'background_mode':'enhance_source'},
                'services':{'text_vision':False,'visual_review':False}}
            if role:intent['constraints']['reference_role']=role
            record=direction.compile_direction(intent,source_sha256='a'*64,character_name='用户角色',
                character_notes='保留手部红光。',input_prompt='Original editable prompt')
            self.assertEqual(record['direction_id'],expected)

    def test_metadata_is_not_arbitrary_permission_or_generation_budget(self):
        for selector in ({'id':'other','version':1},{'id':'source_refinement','version':True},
                         {'id':'source_refinement','version':2},{'id':'source_refinement','version':1,'goals':{'all':'allowed'}}):
            with self.subTest(selector=selector),self.assertRaises(ValueError):preset_store.get_refinement_profile(selector)
        body=self.profile_body();body['preset']='stage_light'
        with self.assertRaises(ValueError):task.plan_task(self.root,body)
        body=self.profile_body();body['preset_snapshot']=copy.deepcopy(preset_store.get_preset('source_refinement'))
        body['preset_snapshot']['refinement_profile']['goals']['skin']='white_skin'
        with self.assertRaises(ValueError):task.plan_task(self.root,body)
        for change in ({'variant_count':2},{'defer_reference_selection':True},{'variant_styles':['night_scene']}):
            with self.subTest(change=change),self.assertRaises(ValueError):task.plan_task(self.root,{**self.profile_body(),**change})
        for mode in ('local_portrait','existing_reference'):
            body=self.profile_body();body['creative_task']['mode']=mode
            if mode=='existing_reference':body['reference']=str(self.reference)
            with self.subTest(mode=mode),self.assertRaises(ValueError):task.plan_task(self.root,body)

    def test_existing_reference_requires_verified_saved_profile_inheritance(self):
        previous,path,body=self.saved_profile();before=path.read_bytes()
        checked=task.plan_task(self.root,body);plan=checked['plan']
        self.assertEqual(plan['image_budget'],{'initial_reference':0,'reference_correction':0,'independent_effect':0})
        self.assertEqual(plan['refinement_inheritance']['previous_plan_id'],previous['task_plan']['plan_id'])
        self.assertEqual(plan['refinement_inheritance']['previous_manifest_sha256'],task.sha(path))
        self.assertEqual(plan['creative_direction']['material_refinement']['permission_source'],'checked_product_recommendation')
        task.execution_request(self.root,{**checked['request'],'plan_id':plan['plan_id']})
        self.assertEqual(path.read_bytes(),before)
        for mutate in ('source','reference','profile','active'):
            changed=copy.deepcopy(previous)
            if mutate in {'source','reference'}:changed[mutate]['sha256']='f'*64
            elif mutate=='profile':changed['task_intent']['constraints'].pop('refinement_profile')
            else:changed['status']='running'
            path.write_text(json.dumps(changed),encoding='utf8')
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):task.plan_task(self.root,body)
            path.write_bytes(before)
        tampered=copy.deepcopy(plan);tampered['refinement_inheritance']['profile_sha256']='f'*64
        with self.assertRaises(ValueError):task.validate_task_plan(tampered,tampered['intent'])

    def test_backend_continuation_preserves_profile_and_legacy_remains_unchanged(self):
        from backend_server import Backend
        previous,path,body=self.saved_profile();backend=Backend(self.root,embedded=True)
        with patch.object(backend,'one_click',side_effect=lambda checked:{'captured':checked}):
            result=backend._continue_checked_task(previous,body)
        execute=result['captured']
        self.assertEqual(execute['task_plan']['image_budget']['initial_reference'],0)
        self.assertEqual(execute['task_plan']['refinement_inheritance']['parent_run_id'],previous['run_id'])
        self.assertEqual(execute['task_intent']['constraints']['refinement_profile'],previous['task_intent']['constraints']['refinement_profile'])
        legacy=task.plan_task(self.root,self.body)
        self.assertNotIn('refinement_profile',legacy['plan']['intent']['constraints'])
        self.assertNotIn('recommendation',legacy['plan']['creative_direction'])
        self.assertNotIn('refinement_inheritance',legacy['plan'])
        # Minimal continuation payload still retains the real frozen role
        # notes and editable prompt. Missing fields are not new empty overrides.
        previous,path,body=self.saved_profile(notes='保留手部红光。')
        with patch.object(backend,'one_click',side_effect=lambda checked:{'captured':checked}):
            result=backend._continue_checked_task(previous,{'source':str(self.source),'reference':str(self.reference)})
        self.assertEqual(result['captured']['task_intent']['constraints']['refinement_profile'],
            previous['task_intent']['constraints']['refinement_profile'])
        self.assertNotIn('hands',result['captured']['creative_direction']['material_refinement']['domains'])

    def test_existing_skin_consumer_accepts_exact_recommendation_without_numerical_changes(self):
        import skin_illumination as skin
        from skin_illumination_smoke import fixture
        checked=task.plan_task(self.root,self.profile_body('暖光人像。'))
        goal=checked['plan']['creative_direction']['material_refinement']
        values,labels,alpha,nail,context,assets,evidence,fields=fixture(self.root/'consumer',goal=goal)
        self.assertEqual(evidence['goal'],goal)
        self.assertEqual(evidence['processed_domains'],['arms','hands'])
        self.assertEqual(evidence['model_submissions'],0)
        self.assertFalse(evidence['reference_RGB_sampled'])
        output,receipt=skin.apply(values,fields,(0,0,values.shape[1],values.shape[0]),[values.shape[1],values.shape[0]])
        self.assertGreater(receipt['active_pixels'],0)
        self.assertEqual(skin.PROFILE,evidence['profile'])
        # Exercise the actual strict checked-context gate with the new source
        # metadata. The consumer need not special-case a product permission.
        native=[values.shape[1],values.shape[0]]
        source_sha=checked['plan']['inputs']['source']['sha256']
        roles=copy.deepcopy(context['source_assets'])
        roles['source_semantics']['source_sha256']=source_sha
        run={'source':{'sha256':source_sha},'task_plan':checked['plan'],'task_intent':checked['plan']['intent'],
            'options':{'appearance_revision':{'design_reconstruction_plan':{
                'source_sha256':source_sha,'canvas':{'source_oriented_size':native},'mask_roles':roles,
                'source_baseline':{'schema':'appearance_source_baseline.v1','source_sha256':source_sha,
                    'assets':{'foreground_alpha':roles['foreground_alpha']}}}}}}
        actual=skin.checked_context(run,native)
        self.assertEqual(actual['goal'],goal)
        self.assertEqual(actual['binding']['goal_sha256'],skin._digest(goal))
        material,evidence=skin.prepare(values,actual,native,self.root/'actual-consumer')
        fields=skin.validate(material,evidence,actual['binding'])
        import numpy as np
        # Different skin brightness stays source-owned; no absolute white
        # swatch or reference RGB is introduced by the recommended contract.
        darker=values*.25
        dark_assets,dark_evidence=skin.prepare(darker,actual,native,self.root/'dark-consumer')
        dark_fields=skin.validate(dark_assets,dark_evidence,actual['binding'])
        dark_output,_=skin.apply(darker,dark_fields,(0,0,*native),native)
        self.assertTrue(np.array_equal(dark_output[labels==3],darker[labels==3]))
        import registered_reference_style as style
        dark_lab=style._to_lab(dark_output)
        self.assertLess(float(np.median(dark_lab[labels==7,0])),float(np.median(dark_lab[labels==3,0]))+.02)
        self.assertFalse(np.array_equal(dark_output[labels==7],darker[labels==7]))

    def test_copied_installed_catalog_without_frozen_or_source_fallback(self):
        root=self.root/'standalone';internal=root/'_internal';(internal/'presets').mkdir(parents=True)
        shutil.copyfile(preset_store.DEFAULT_PATH,internal/'presets/presets.json')
        shutil.copyfile(Path(preset_store.__file__),internal/'preset_store.py')
        spec=importlib.util.spec_from_file_location('copied_preset_store',internal/'preset_store.py')
        copied=importlib.util.module_from_spec(spec)
        with patch.dict(os.environ,{},clear=True):spec.loader.exec_module(copied)
        selector={'id':'source_refinement','version':1}
        self.assertEqual(copied.get_refinement_profile(selector),preset_store.get_refinement_profile(selector))
        self.assertEqual(copied.DEFAULT_PATH,internal/'presets/presets.json')
        spec=importlib.util.spec_from_file_location('frozen_preset_store',internal/'preset_store.py')
        frozen=importlib.util.module_from_spec(spec)
        with patch.object(sys,'frozen',True,create=True),patch.object(sys,'_MEIPASS',str(internal),create=True):
            spec.loader.exec_module(frozen)
            self.assertEqual(frozen.get_refinement_profile(selector)['profile_sha256'],copied.get_refinement_profile(selector)['profile_sha256'])
        (internal/'presets/presets.json').unlink()
        spec=importlib.util.spec_from_file_location('missing_preset_store',internal/'preset_store.py')
        with self.assertRaises(FileNotFoundError):spec.loader.exec_module(importlib.util.module_from_spec(spec))
        # A catalog may describe a recommendation, never silently smuggle
        # authored regional controls into its otherwise fixed permission.
        payload=json.loads(preset_store.DEFAULT_PATH.read_text(encoding='utf8'))
        row=next(item for item in payload['presets'] if item['id']=='source_refinement')
        row['appearance_regions']={'face':{'exposure':2}}
        (internal/'presets/presets.json').write_text(json.dumps(payload),encoding='utf8')
        with self.assertRaises(ValueError):copied.get_refinement_profile(selector)

    def test_neutral_soft_key_covers_hands_and_props_without_whitening(self):
        body=self.profile_body('原景冷蓝电影感，人物柔和中性主光，保留旧漆和金属材质。')
        checked=task.plan_task(self.root,body)
        prompt=checked['plan']['creative_direction']['prompt']
        self.assertIn('以柔主光塑造人物',prompt)
        self.assertIn('hands and existing props as well as the face',prompt)
        self.assertIn('aged paint, wood grain and metallic reflections',prompt)
        self.assertIn('Do not paint all objects white',prompt)
        # The conditional recommendation never changes the actual veto or
        # silently turns a user's coloured stage light into neutral light.
        body=self.profile_body('保留手部红绿光，不要柔和中性主光，无特效。')
        plan=task.plan_task(self.root,body)['plan']
        self.assertNotIn('以柔主光塑造人物',plan['creative_direction']['prompt'])
        self.assertNotIn('hands',plan['creative_direction']['material_refinement']['domains'])
        self.assertTrue(plan['intent']['constraints']['no_effects'])

    def test_recommended_materialize_native_consumer_and_portable_roundtrip(self):
        import numpy as np
        import registered_reference_style as style
        from skin_illumination_smoke import fixture,v2_layer
        from current_stack_fixture import prepare_fixture
        from design_reconstruction import materialize_plan,plan_for_run
        from float_appearance import render_source_float
        from native_source_baseline import source_for_plan
        from native_roi_render import render_native_roi
        from native_current_only_smoke import save
        from portable_project import export_project,import_project
        root=self.root/'portable-workspace';run=prepare_fixture(root,native=(224,288))
        body=self.profile_body('暖光人像，保留肤色。');body['source']=run['source']['path']
        checked=task.plan_task(self.root,body)
        goal=checked['plan']['creative_direction']['material_refinement']
        values,_,_,_,context,assets,evidence,_=fixture(root/'producer',
            source_sha=run['source']['sha256'],reference_sha=run['reference']['sha256'],goal=goal)
        layer=v2_layer(root/'producer',values,context,assets,evidence)
        run.update(task_intent=checked['plan']['intent'],task_plan=checked['plan'],
            creative_direction=checked['plan']['creative_direction'])
        plan=plan_for_run(run);source=source_for_plan(plan)
        baseline_values,_=render_source_float(source,run)
        expected=style.apply(baseline_values,layer,(0,0,224,288),(224,288))
        plan['layers'].append(layer)
        owned=materialize_plan(root,run,plan)
        run['options']['appearance_revision']['design_reconstruction_plan']=owned
        whole,operator=render_source_float(source,run)
        self.assertTrue(np.array_equal(whole,expected))
        roi,_=render_native_roi(run,(140,128,209,277))
        self.assertTrue(np.array_equal(roi,whole[128:277,140:209]))
        save(root,run,whole,operator)
        archive=self.root/'recommended.cosplayproj';export_project(root,run['run_id'],archive)
        imported=import_project(self.root/'empty-import',archive,run_id='imported-recommendation')['run']
        self.assertEqual(imported['task_plan']['creative_direction']['recommendation'],checked['plan']['creative_direction']['recommendation'])
        self.assertEqual(imported['task_plan']['creative_direction']['material_refinement'],goal)
        imported_roi,_=render_native_roi(imported,(140,128,209,277))
        self.assertTrue(np.array_equal(imported_roi,roi))


if __name__=='__main__':unittest.main()
