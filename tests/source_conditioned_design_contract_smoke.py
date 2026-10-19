"""Source-only generated art direction is distinct from a user's finish image."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import creative_direction as direction
import creative_task as task
from creative_direction_smoke import enhanced
from creative_task_smoke import fixture, snapshot
from task_model_preflight_smoke import mocked_task_capability


def asset(path):
    return {'path':str(path), 'sha256':task.sha(path)}


class SourceConditionedDesignTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='scd-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);registry,sources,reference=fixture(self.root)
        self.source=Path(sources[0]);self.reference=Path(reference)
        self.common={'source':str(self.source),'model_registry':str(registry),
            'provider':'qwen_image_2_1_uncensored_fast','character_name':'用户提供角色',
            'input_prompt':'Create coherent cold cinematic light from this original photograph.'}
        self.intent=enhanced()
        self.intent['brief']='冷蓝环境、干净脸肤、发丝冷rim，薄雾和虚化红叶；保留人物和原景结构。'
        self.intent['constraints']['reference_role']='source_conditioned_design'
        self.body={**self.common,'creative_task':self.intent}
        self.enterContext(patch('task_model_preflight.inspect_task_models',side_effect=mocked_task_capability))
        self.health=self.enterContext(patch('reference_provider.provider_health',return_value={'ready':True}))
        self.enterContext(patch('urllib.request.urlopen',side_effect=AssertionError('No network allowed')))
        graph={'4':{'class_type':'LoadImage','inputs':{'image':'source-upload.png'}},
            '5':{'class_type':'CosplayTextEncodeQwenImage21','inputs':{'images.image_1':['4',0],'prompt':'Source-only design'}}}
        self.provenance={'status':'reference_candidate','provider':'qwen_image_2_1_uncensored_fast',
            'generation_id':'actual-prompt-id','prompt':'Source-only design','source':asset(self.source),
            'asset':asset(self.reference),'provider_contract':{
                'request':{'prompt':'Source-only design','input_sha256':task.sha(self.source),
                    'api_input':{**asset(self.source),'derived':False}},
                'response':{'id':'actual-prompt-id'},
                'transport':{'events':[{'event':'comfyui_submitted','prompt_id':'actual-prompt-id','request':{'prompt':graph}},
                    {'event':'comfyui_completed','prompt_id':'actual-prompt-id'}]},
                'comfyui':{'prompt_id':'actual-prompt-id','execution_status':{'completed':True,'status_str':'success'},
                    'condition_inputs':[{'role':'source',**asset(self.source),'upload':{'name':'source-upload.png'}}],
                    'request':{'prompt':graph}}}}
        self.provenance_path=self.reference.with_name(self.reference.stem+'.provenance.json')
        self.write_provenance(self.provenance)

    def write_provenance(self,value):
        self.provenance_path.write_text(json.dumps(value),encoding='utf8')

    def existing_body(self):
        intent=copy.deepcopy(self.intent);intent['mode']='existing_reference'
        return {**self.common,'reference':str(self.reference),'creative_task':intent}

    def test_create_freezes_no_target_then_binds_real_returned_reference(self):
        before=snapshot(self.root);checked=task.plan_task(self.root,self.body)
        self.assertEqual(before,snapshot(self.root))
        self.assertTrue(checked['plan']['ready'])
        self.assertNotIn('reference',checked['plan']['inputs'])
        self.assertNotIn('reference',checked['request'])
        self.assertEqual(checked['plan']['image_budget'],{'initial_reference':1,'reference_correction':0,'independent_effect':0})
        descriptor=checked['plan']['reference_contract']
        self.assertEqual(descriptor['schema'],'source_conditioned_design_reference.v1')
        self.assertEqual(descriptor['reference_binding'],'after_generation')
        self.assertIsNone(descriptor['reference_sha256'])
        self.assertIsNone(descriptor['reference_provenance_sha256'])
        self.assertIsNone(descriptor['provider_prompt_id'])
        self.assertEqual(descriptor['registration_status'],'not_checked')
        self.assertFalse(descriptor['reference_rgb_compositing_allowed'])
        self.assertTrue(checked['plan']['services']['local_image_inference'])
        from qt_workbench import creative_task_plan_text
        self.assertIn('不需要人工精修图作为输入',creative_task_plan_text(checked['plan']))
        frozen=copy.deepcopy(checked['plan'])
        bound=direction.bind_source_conditioned_reference(self.root,checked['plan'],asset(self.reference))
        self.assertEqual(bound['reference_binding'],'generated_source_verified')
        self.assertEqual(bound['reference_sha256'],task.sha(self.reference))
        self.assertEqual(bound['reference_provenance_sha256'],task.sha(self.provenance_path))
        self.assertEqual(bound['provider_prompt_id'],'actual-prompt-id')
        self.assertEqual(bound['registration_status'],'not_checked')
        self.assertEqual(checked['plan'],frozen)
        self.assertEqual(checked['plan']['plan_id'],task.plan_task(self.root,checked['request'])['plan']['plan_id'])
        task.execution_request(self.root,{**checked['request'],'plan_id':checked['plan']['plan_id']})

    def test_existing_requires_actual_same_source_provenance_and_new_budget_zero(self):
        body=self.existing_body();before=snapshot(self.root)
        checked=task.plan_task(self.root,body)
        self.assertEqual(before,snapshot(self.root))
        self.assertEqual(sum(checked['plan']['image_budget'].values()),0)
        self.assertFalse(checked['request']['generate'])
        self.assertFalse(checked['plan']['services']['image_generation'])
        descriptor=checked['plan']['reference_contract']
        self.assertEqual(descriptor['reference_binding'],'generated_source_verified')
        self.assertEqual(descriptor,direction.bind_source_conditioned_reference(self.root,checked['plan'],asset(self.reference)))
        task.validate_task_plan(checked['plan'],checked['plan']['intent'])
        self.assertEqual(checked['plan']['plan_id'],task.plan_task(self.root,checked['request'])['plan']['plan_id'])
        original=self.provenance_path.read_bytes();self.provenance_path.unlink()
        with self.assertRaises(ValueError):task.plan_task(self.root,body)
        self.provenance_path.write_bytes(original)
        self.provenance_path.write_bytes(original+b'\n')
        with self.assertRaises(ValueError):
            task.execution_request(self.root,{**checked['request'],'plan_id':checked['plan']['plan_id']})
        with self.assertRaises(ValueError):
            direction.bind_source_conditioned_reference(self.root,checked['plan'],asset(self.reference))

    def test_other_source_finish_images_and_multi_condition_evidence_rejected(self):
        mutations=[lambda p:p.update(provider='human_editor'),
            lambda p:p['source'].update(sha256='f'*64),
            lambda p:p['asset'].update(sha256='f'*64),
            lambda p:p.update(generation_id='unmatched'),
            lambda p:p['provider_contract']['comfyui']['condition_inputs'].append(copy.deepcopy(p['provider_contract']['comfyui']['condition_inputs'][0])),
            lambda p:p['provider_contract']['comfyui']['condition_inputs'][0].update(role='scene_reference'),
            lambda p:p['provider_contract']['comfyui']['request']['prompt']['5']['inputs'].update({'images.image_2':['4',0]}),
            lambda p:p['provider_contract']['comfyui']['execution_status'].update(completed=False),
            lambda p:p['provider_contract']['request']['api_input'].update(sha256='e'*64)]
        for mutate in mutations:
            value=copy.deepcopy(self.provenance);mutate(value);self.write_provenance(value)
            with self.assertRaises(ValueError):task.plan_task(self.root,self.existing_body())
        self.write_provenance(self.provenance)

    def test_conflicts_and_descriptor_tampering_rejected(self):
        for extra in ({'reference':str(self.reference)}, {'provider':'openai-compatible'},
                      {'variant_count':2},{'variant_styles':['night_scene']},{'defer_reference_selection':True},
                      {'regional_resume_authorization':{}},{'regional_candidate_reuse_authorization':{}}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):task.plan_task(self.root,{**self.body,**extra})
        for regional in ({},{'maximum_submissions':0,'regions':[]}):
            body=copy.deepcopy(self.body);body['creative_task']['regional_refinement']=regional
            with self.assertRaises(ValueError):task.plan_task(self.root,body)
        local=copy.deepcopy(self.body);local['creative_task']['mode']='local_portrait'
        with self.assertRaises(ValueError):task.plan_task(self.root,local)
        checked=task.plan_task(self.root,self.body)
        for field,value in (('reference_sha256','f'*64),('registration_status','registered'),
                            ('reference_binding','generated_source_verified'),('texture_owner','reference'),
                            ('reference_rgb_compositing_allowed',0)):
            bad=copy.deepcopy(checked['plan']);bad['reference_contract'][field]=value
            with self.assertRaises(ValueError):task.validate_task_plan(bad,bad['intent'])

    def test_same_photo_and_legacy_contracts_remain_distinct(self):
        same=self.existing_body();same['creative_task']['constraints']['reference_role']='same_photo_finish'
        self.provenance_path.unlink()  # User finish calibration does not require generation provenance.
        checked=task.plan_task(self.root,same)
        self.assertEqual(checked['plan']['reference_contract']['schema'],'same_photo_finish_reference.v1')
        self.assertEqual(len(checked['plan']['reference_contract']),9)
        self.assertEqual(sum(checked['plan']['image_budget'].values()),0)
        with self.assertRaises(ValueError):direction.bind_source_conditioned_reference(self.root,checked['plan'],asset(self.reference))
        legacy=copy.deepcopy(same);legacy['creative_task']['constraints'].pop('reference_role')
        old=task.plan_task(self.root,legacy)
        self.assertNotIn('reference_contract',old['plan'])
        self.assertNotIn('reference_role',old['plan']['intent']['constraints'])
        self.assertEqual(old['plan']['layer_workflow'],checked['plan']['layer_workflow'])

    def test_model_prompt_uses_declared_photographic_goals_not_render_mechanics(self):
        body=copy.deepcopy(self.body)
        body['creative_task']['brief']=('冷调夜景，消除脸、手臂和浅色道具的红绿杂光；肤色干净但保留妆面与纹理，'
            '柔和主光；发丝细窄方向冷rim，保留黑发丝纹；原景空间深度、薄雾与虚焦前景叶片。')
        checked=task.plan_task(self.root,body)
        direction_record=checked['plan']['creative_direction'];prompt=direction_record['prompt']
        managed=prompt.split(direction.MARKER,1)[1]
        for photographic in ('连续处理整只手','柔主光','细窄、克制','各自可信的底色和材质','建立纵深','局部空气感'):
            self.assertIn(photographic,managed)
        for implementation in ('低频','系数','配准','RGB','ownership','source_native','冻结合同'):
            self.assertNotIn(implementation,managed)
        execute=task.execution_request(self.root,{**checked['request'],'plan_id':checked['plan']['plan_id']})
        self.assertEqual(execute['prompt'],prompt)
        self.assertEqual(execute['effective_prompt'],prompt)
        self.assertEqual(execute['task_plan']['creative_direction']['prompt'],prompt)
        self.assertEqual(direction.validate_direction(direction_record,checked['plan']['intent'],
            source_sha256=checked['plan']['inputs']['source']['sha256']),direction_record)
        tampered=copy.deepcopy(direction_record);tampered['prompt']+='\nChanged art direction'
        with self.assertRaises(ValueError):
            direction.validate_direction(tampered,checked['plan']['intent'],
                source_sha256=checked['plan']['inputs']['source']['sha256'])
        self.assertEqual(checked['plan']['reference_contract']['transfer_scope'],
            'tone_and_source_guided_local_colour_coefficients')
        self.assertEqual(checked['plan']['image_budget'],{'initial_reference':1,'reference_correction':0,'independent_effect':0})

    def test_warm_minimal_and_intentional_colour_are_not_overridden(self):
        body=copy.deepcopy(self.body);body['input_prompt']='Warm sunset portrait.'
        body['creative_task']['brief']='暖金夕照，保留脸颊橙红彩灯和手臂红色灯带；细窄暖色发丝轮廓光与原景空间深度。'
        prompt=task.plan_task(self.root,body)['request']['prompt'].split(direction.MARKER,1)[1]
        self.assertIn('暖金夕照',prompt)
        self.assertIn('保留脸颊橙红彩灯',prompt)
        for invented in ('冷蓝','冷白','钴蓝','琵琶','蝴蝶','连续处理整只手','仅在用户明确要求清理杂光'):
            self.assertNotIn(invented,prompt)
        body['input_prompt']='Keep the original subject.'
        body['creative_task']['brief']='保留本人、衣服和原有场景，不添加新物件。'
        prompt=task.plan_task(self.root,body)['request']['prompt'].split(direction.MARKER,1)[1]
        for invented in ('柔主光','轮廓光','薄雾','前景物','连续处理整只手'):
            self.assertNotIn(invented,prompt)
        preserve='不要去除手臂原有红绿杂光，保留舞台颜色。'
        self.assertEqual(direction._source_photographic_goals(preserve,no_effects=False),[])
        hard_light=direction._source_photographic_goals('不要柔光；需要硬主光，不加前景叶片。',no_effects=False)
        self.assertFalse(any('以柔主光塑造' in line or '前景物' in line for line in hard_light))
        nofx='主效：薄雾；不要添加特效；保留自然肤质。'
        goals=direction._source_photographic_goals(nofx,no_effects=True)
        self.assertFalse(any('薄雾' in line for line in goals))
        body['creative_task']['brief']=nofx
        record=task.plan_task(self.root,body)['plan']['creative_direction']
        self.assertEqual(record['main_effect']['mode'],'none')
        self.assertIn('主效要求：不要添加任何特效。',record['prompt'])

    def skin_contract(self, brief):
        intent=copy.deepcopy(self.intent);intent['brief']=brief
        return direction.skin_cast_contract_for(intent)

    def test_explicit_skin_contract_is_frozen_and_rechecked(self):
        body=copy.deepcopy(self.body)
        body['creative_task']['brief']='暖金环境；统一脸、手指到前臂的肤色；清除皮肤红绿杂光；保留眼妆、唇色和指甲相对颜色。'
        checked=task.plan_task(self.root,body);record=checked['plan']['creative_direction']
        contract=record['material_refinement']
        self.assertEqual(contract['schema'],'skin_cast_contract.v1')
        self.assertEqual(set(contract['domains']),{'face','arms','hands'})
        self.assertEqual(contract['target_temperature'],'follow_user_intent')
        self.assertEqual(contract['anchor'],'reliable_face_skin_chroma_in_requested_lighting')
        self.assertEqual(contract['activation_status'],'not_evaluated')
        self.assertTrue(contract['rgb_changes_allowed'])
        self.assertFalse(contract['reference_rgb_sampling_allowed'])
        self.assertIn('relative_intrinsic_nail_colour',contract['preserve'])
        request=task.execution_request(self.root,{**checked['request'],'plan_id':checked['plan']['plan_id']})
        self.assertEqual(request['task_plan']['creative_direction']['material_refinement'],contract)
        self.assertEqual(task.validate_task_plan(checked['plan'],checked['plan']['intent']),checked['plan'])
        for field,value in (('activation_status','applied'),('reference_rgb_sampling_allowed',True),
                            ('target_temperature','cold_white'),('domains',{'hands':[]})):
            bad=copy.deepcopy(record);bad['material_refinement'][field]=value
            bad.pop('direction_id');bad['direction_id']=direction.digest(bad)
            with self.subTest(field=field),self.assertRaises(ValueError):
                direction.validate_direction(bad,checked['plan']['intent'],source_sha256=task.sha(self.source))

    def test_skin_permissions_are_scoped_and_vetoes_win(self):
        contract=self.skin_contract('清除手臂红色杂光；保留手部绿色彩灯。')
        self.assertEqual(set(contract['domains']),{'arms'})
        self.assertEqual(contract['vetoed_domains'],['hands'])
        self.assertEqual(contract['domains']['arms'][0]['cast_targets'],['red'])
        contract=self.skin_contract('去除皮肤红绿杂光；保留手臂红色灯带。')
        self.assertEqual(set(contract['domains']),{'face','hands'})
        for brief in ('清除皮肤红绿杂光；不要去偏色。',
                      '统一肤色；不统一肤色。',
                      '去除手臂杂光；不要去除手臂杂光。',
                      '去除手臂杂光；不用去除手臂杂光。',
                      '统一皮肤；别统一肤色。',
                      '清除皮肤红绿杂光；别去偏色。',
                      '清除皮肤红绿杂光；请别去偏色。',
                      '统一皮肤；保留皮肤红色舞台光。',
                      '统一皮肤；不要改变皮肤彩灯。',
                      '统一皮肤；保留皮肤彩色照明。'):
            with self.subTest(brief=brief):self.assertIsNone(self.skin_contract(brief))

    def test_actual_production_brief_separately_refine_is_not_a_veto(self):
        # Complete production prose, without private source identity or target RGB.
        brief=('在原景内做精致电影感人像后期。环境为低调冷蓝银灰，人物受到柔和明净的中性主光，'
            '脸和旧白琴面明显亮于背景，有自然亮部层次，不要整体压暗或灰蓝肤色。'
            '脸、手指到整个前臂统一为清透自然的中性米白肤色，保留少量健康粉色和真实纹理；'
            '完整清除皮肤上的红绿舞台杂光，尤其前臂中央红色竖向长条，不能留下任何彩色条带和青绿色手。'
            '保留眼妆、玫瑰色嘴唇与唇纹。琴面中性象牙旧白，木与金属材质各自清楚，不泛绿。'
            '旗袍深钴蓝，织纹和褶皱有层次；黑发保持乌黑和发丝细节，左后方只要少量细窄柔和银蓝发丝轮廓光，'
            '不可出现宽蓝光带或饱和描边。保留原景建筑、原有虚焦暗红前景，在背景左侧克制的薄雾与虚实层次中建立深度，'
            '雾不盖脸眼手。保持摄影质感，光色与材质分别精修后全局统一。'
            '禁止全图蓝滤镜、塑料磨皮、蝴蝶、法阵、霓虹、文字水印、换脸、改姿态和完整换景。')
        contract=self.skin_contract(brief)
        self.assertEqual(set(contract['domains']),{'face','arms','hands'})
        self.assertEqual(contract['vetoed_domains'],[])
        self.assertIn('remove_declared_unwanted_casts',contract['domains']['arms'][-1]['operations'])
        body=copy.deepcopy(self.body);body['creative_task']['brief']=brief
        checked=task.plan_task(self.root,body)
        self.assertEqual(checked['plan']['creative_direction']['material_refinement'],contract)
        self.assertEqual(task.validate_task_plan(checked['plan'],checked['plan']['intent']),checked['plan'])

    def test_absent_permission_and_other_roles_never_enable_skin_operator(self):
        for brief in ('冷蓝电影感，皮肤干净。','去除衣服红绿杂光，保留自然皮肤。',
                      '保留皮肤舞台彩灯。','修饰手臂和手部，保持材质。'):
            with self.subTest(brief=brief):self.assertIsNone(self.skin_contract(brief))
        for role in (None,'same_photo_finish'):
            intent=copy.deepcopy(self.intent);intent['brief']='统一皮肤并清除红绿杂光。'
            if role is None:intent['constraints'].pop('reference_role')
            else:intent['constraints']['reference_role']=role
            self.assertIsNone(direction.skin_cast_contract_for(intent))
        intent=copy.deepcopy(self.intent);intent['brief']='清除皮肤杂光。'
        intent['constraints']['background_mode']='replace'
        self.assertIsNone(direction.skin_cast_contract_for(intent))

    def test_warm_lighting_and_no_effects_do_not_cancel_explicit_skin_correction(self):
        intent=copy.deepcopy(self.intent);intent['constraints']['no_effects']=True
        intent['brief']='暖色主光；去除手臂绿色杂光；保留背景红色彩灯；不要特效。'
        record=direction.compile_direction(intent,source_sha256='a'*64)
        self.assertEqual(set(record['material_refinement']['domains']),{'arms'})
        self.assertEqual(record['material_refinement']['target_temperature'],'follow_user_intent')
        self.assertEqual(record['main_effect']['mode'],'none')
        self.assertEqual(set(self.skin_contract('Remove green casts from hands; preserve red stage lights on arms.')['domains']),{'hands'})


if __name__=='__main__':
    unittest.main()
