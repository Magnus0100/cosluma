"""Actual imported-pair planning and shared pure prose contracts; no network."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from generation_design import (normalize_design_brief,
                               prepare_existing_reference_design, sha256, correction_prompt)


def plan_fixture(*, dramatic: bool = True) -> dict:
    """A complete transport fixture; its synthesis intentionally omits details."""
    return {
        'summary': '完成角色写真，空间和人物形成统一作品。',
        'scene_analysis': '人物服装清晰，现有背景层次较弱。',
        'character_basis': 'Use the visible costume colour motif; the fictional character remains unconfirmed.',
        'subject_retouch_plan': 'Remove transient skin distractions and retain pores, makeup contours and contact-lens patterns.',
        'scene_transformation_plan': {
            'mode': 'transform' if dramatic else 'refine',
            'description': ('Replace the distant clutter with a recessed stone arcade behind the fixed person.'
                            if dramatic else 'Clean distractions within the existing background and retain its objects.'),
            'visible_changes': (['Distant arches recede at the original camera perspective.',
                                 'A dim open foreground keeps the costume silhouette legible.']
                                if dramatic else ['Reduce distracting background contrast around the face.']),
        },
        'primary_effect_plan': {
            'mode': 'scene_effect' if dramatic else 'none',
            'description': ('A narrow amber mist ribbon crosses behind the left forearm.'
                            if dramatic else 'Use a clean photographic finish without added atmospheric effects.'),
            'reason': 'Establish a readable focal hierarchy without obscuring the face.',
        },
        'spatial_integration_plan': 'Place faint foreground blur before the person, the ribbon behind the forearm and arches farther away; retain foot contact and let the forearm occlude the ribbon.',
        'lighting_plan': 'The amber ribbon lights the nearby sleeve selectively; broad soft scene fill keeps facial detail readable.',
        'palette_material_plan': 'Keep black cloth dimensional, white fabric textured and metal reflective; give skin, sclera, iris and lip pigment distinct responses.',
        'tradeoffs': 'Let the face lead and omit secondary particle showers or extra props.',
        'user_constraints': {'no_effects_quote': '', 'preserve_background_quote': ''},
        'reference_application': '造型来自可见证据，新增空间是创意解释，不虚构角色设定。',
        'english_art_direction': 'Deliver a complete photographic cosplay portrait with a clear focal hierarchy.',
    }


def expect_rejected(plan: dict, *, style_id: str = 'fantasy_atmosphere', brief: str = ''):
    try:
        normalize_design_brief(plan, style_id=style_id, user_brief=brief)
    except ValueError:
        return
    raise AssertionError('Incomplete or contradictory design was accepted')


def main():
    material = {'criterion': 'skin_materials', 'action': 'Correct the reference skin cast.'}
    scene = {'criterion': 'effects_space', 'action': 'Restore the missing amber smoke behind the shoulder.'}
    material_prompt = correction_prompt([material])
    scene_prompt = correction_prompt([scene], ('identity_makeup', 'concept_hierarchy', 'effects_space', 'scene_integration'))
    mixed_prompt = correction_prompt([material, scene])
    assert 'DIAGNOSED MATERIAL CORRECTIONS:' in material_prompt
    assert 'Preserve existing scene objects, effect placement, depth and hierarchy.' in material_prompt
    assert 'DIAGNOSED SCENE AND EFFECT CORRECTIONS:' in scene_prompt and scene['action'] in scene_prompt
    assert 'placement and hierarchy may change' in scene_prompt
    assert 'Retain subject material colours' in scene_prompt
    assert 'retain the intended fog colour and placement' in scene_prompt
    assert 'actual hair, clothing, foliage and object occlusion intact' in scene_prompt
    assert 'diagnosed mist or haze defects' not in material_prompt
    assert 'material regions' not in scene_prompt and 'Make transitions continuous' not in scene_prompt
    assert 'DIAGNOSED MATERIAL AND SCENE CORRECTIONS:' in mixed_prompt
    assert 'Retain subject material colours' not in mixed_prompt
    for actions, criteria in [([{'criterion': 'identity_makeup', 'action': 'Redraw the face.'}], None),
                              ([material], ('effects_space',)), ([], None)]:
        try:
            correction_prompt(actions, criteria)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported or unselected generation instruction was accepted')
    import generation_design as module
    assert not hasattr(module,'prepare_generation_design')
    # Keep pure normalization/compiler contracts used by actual imported pairs.
    for style_id in ('stage_light','night_scene','fantasy_atmosphere','shrine_drama'):
        for dramatic in (False,True):
            normalized=normalize_design_brief(plan_fixture(dramatic=dramatic),style_id=style_id,user_brief='')
            compiled=module.compile_design_direction(normalized)
            for field in ('character_basis','subject_retouch_plan','spatial_integration_plan','lighting_plan','palette_material_plan','tradeoffs'):
                assert normalized[field] in compiled
    for missing in ('scene_transformation_plan','primary_effect_plan','spatial_integration_plan','character_basis','tradeoffs'):
        incomplete=plan_fixture();incomplete.pop(missing);expect_rejected(incomplete)
    empty=plan_fixture();empty['scene_transformation_plan']['visible_changes']=[];expect_rejected(empty)
    quoted=plan_fixture(dramatic=False)
    quoted['user_constraints']={'no_effects_quote':'Skip atmospheric additions.','preserve_background_quote':'Leave the location as photographed.'}
    normalize_design_brief(quoted,style_id='night_scene',user_brief='Skip atmospheric additions. Leave the location as photographed.')
    expect_rejected(quoted,brief='Make a finished portrait.')
    expect_rejected(plan_fixture(),brief='不要特效，保留原背景。')
    increment=plan_fixture();increment['foreground_effect_plan']={'enabled':True,'layers':[{
        'id':'front_wisp','slot':'front','operation':'over','description':'A distinct unbaked sparse front accent',
        'occlusion':'protect_face_hands','increment':True,'baked_in_reference':False,
        'effect_design':{'material':'scattering_smoke','colour_rgb8':[255,181,82],'extent_2d':[0,.6,.4,.9],'density':'sparse'}}]}
    typed=normalize_design_brief(increment,style_id='night_scene',user_brief='')
    assert 'Do not bake or draw these extra layers' in module.compile_design_direction(typed)
    for key,value in (('material','portrait'),):
        invalid=copy.deepcopy(increment);invalid['foreground_effect_plan']['layers'][0]['effect_design'][key]=value;expect_rejected(invalid)
    invalid=copy.deepcopy(increment);invalid['foreground_effect_plan']['layers'][0]['baked_in_reference']=True;expect_rejected(invalid)
    requests=[];response=plan_fixture(dramatic=False)
    response.update(reference_background_policy={'mode':'use_reference','reason':'Actual supplied pair'},
        reference_effect_observation={'present':False,'description':'No extra baked effects'})
    def planner(source,output,prompt,**kwargs):requests.append(prompt);return copy.deepcopy(response),{'model':'synthetic-pair-fixture'}
    with tempfile.TemporaryDirectory(prefix='paired-design-contract-') as temporary,patch('vision_transport.request_vision_json',side_effect=planner):
        root=Path(temporary);source=root/'source.png';reference=root/'reference.png'
        for path in (source,reference):Image.new('RGB',(32,48),'#607080').save(path)
        intent={'schema_version':'cosplay_task_intent.v1','mode':'existing_reference','brief':'A restrained coherent finish',
            'constraints':{'preserve_background':False,'no_effects':False},'services':{'text_vision':True,'visual_review':False}}
        result=prepare_existing_reference_design(source,reference,root/'pair',intent)
        assert result['status']=='ready',result
        assert result['image_submissions']==0 and result['text_vision_submissions']==1
        assert result['binding']['source_sha256']==sha256(source) and result['binding']['reference_sha256']==sha256(reference)
        assert prepare_existing_reference_design(source,reference,root/'pair',intent)==result and len(requests)==1
        response['source_render_plan']={'enabled':True}
        retired=prepare_existing_reference_design(source,reference,root/'retired-numeric',intent)
        assert retired['status']=='unavailable' and 'Retired numeric' in retired['error']
        count=len(requests);not_allowed={**intent,'mode':'create_reference'}
        unavailable=prepare_existing_reference_design(source,reference,root/'not-paired',not_allowed)
        assert unavailable['status']=='unavailable' and len(requests)==count
    print(json.dumps({'ok':True,'retired_generation_preparer_absent':True,'paired_import_binding_and_cache':True,
        'pure_prompt_compilation_retained':True,'constraints_and_numeric_rejection':True,'network_calls':0}))


if __name__=='__main__':main()
