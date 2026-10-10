"""Offline contract for complete planning, sent scene/FX and honest failure."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from generation_design import (DESIGN_VERSION, DRAMATIC_STYLES, normalize_design_brief,
                               prepare_generation_design, sha256, correction_prompt)
from generation_prompt import CHARACTER_MARKER, DEFAULT_PROMPT, DESIGN_MARKER, STYLE_MARKER
from preset_store import get_preset


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
    requests = []
    response = plan_fixture()

    def planner(source, destination, prompt, **connection):
        requests.append(prompt)
        return copy.deepcopy(response), {'request_id': 'offline-design-fixture'}

    with tempfile.TemporaryDirectory(prefix='cosplay_design_contract_') as temporary, \
         patch('vision_transport.request_vision_json', side_effect=planner), \
         patch('character_research.research_character', side_effect=AssertionError('No network research permitted')), \
         patch('source_color.diagnose_source_colors', return_value={'scope': 'offline fixture'}):
        root = Path(temporary)
        source = root / 'source.png'
        Image.new('RGB', (32, 48), '#607080').save(source)
        research = {'status': 'unavailable', 'summary': 'Use visible evidence only.'}
        custom = 'Make a finished cosplay portrait from the visible costume.'
        old = custom + STYLE_MARKER + get_preset('night_scene')['prompt'] + \
            CHARACTER_MARKER + 'OLD CHARACTER: no effects; preserve background.' + \
            DESIGN_MARKER + 'OLD SHOT: keep the old lighting.'
        result = prepare_generation_design(source, root / 'complete', {},
            style_id='fantasy_atmosphere', base_prompt=old, character_research=research)
        sent = result['effective_prompt']
        assert result['schema_version'] == DESIGN_VERSION and result['prompt_template_version'] == 'custom'
        # These instructions only exist in structured fields, not the synthesis.
        # They must reach the actual provider prompt, including every scene change.
        for field in ('character_basis', 'subject_retouch_plan', 'spatial_integration_plan',
                      'lighting_plan', 'palette_material_plan', 'tradeoffs'):
            assert response[field] in sent, f'Unconsumed plan: {field}'
        scene, effect = response['scene_transformation_plan'], response['primary_effect_plan']
        assert scene['description'] in sent and all(change in sent for change in scene['visible_changes'])
        assert effect['description'] in sent and effect['reason'] in sent
        # The existing scene consumer still reads descriptive compatibility
        # fields; those must contain this plan, not an old saved shot's prose.
        from scene_design import normalize_scene_intent
        scene_intent = normalize_scene_intent(result)['descriptions']
        assert effect['description'] in scene_intent['environment_effects_plan']
        assert response['spatial_integration_plan'] in scene_intent['environment_effects_plan']
        assert 'OLD CHARACTER' not in sent and 'OLD SHOT' not in sent
        assert sent.count(DESIGN_MARKER) == 1 and get_preset('night_scene')['prompt'] not in sent
        assert sent.count(get_preset('fantasy_atmosphere')['prompt']) == 1
        context = json.loads(requests[-1].split('\nREFERENCE DATA:\n', 1)[1])
        assert context['explicit_user_constraints'] == {'no_effects': False, 'preserve_background': False}
        assert context['user_editable_brief'].strip() == custom
        assert 'OLD SHOT' not in requests[-1] and 'source_render_plan' not in requests[-1]
        assert 'source_render_plan' not in result['design_brief']
        assert 'source_color_plan' not in result['design_brief']
        assert 'local_environment_intent' not in result['design_brief']
        assert json.loads((root / 'complete/generation_design.json').read_text(encoding='utf-8')) == result

        # Conditional instructions in the bundled template are not explicit
        # user restrictions. Preserve the template in its own planner context.
        default = prepare_generation_design(source, root / 'default', {},
            style_id='fantasy_atmosphere', character_research=research)
        default_context = json.loads(requests[-1].split('\nREFERENCE DATA:\n', 1)[1])
        assert not default_context['user_editable_brief'].strip()
        assert default_context['base_photographic_brief'] == DEFAULT_PROMPT
        assert default_context['explicit_user_constraints'] == {'no_effects': False, 'preserve_background': False}
        assert default['design_brief']['primary_effect_plan']['mode'] == 'scene_effect'

        for style_id in DRAMATIC_STYLES:
            normalize_design_brief(plan_fixture(), style_id=style_id, user_brief='')
            expect_rejected(plan_fixture(dramatic=False), style_id=style_id)

        response = plan_fixture(dramatic=False)
        response['spatial_integration_plan'] = 'Retain the existing scene depth, object occlusion and ground contact.'
        response['lighting_plan'] = 'Use soft directional light on skin and coherent fill across the existing background.'
        soft = prepare_generation_design(source, root / 'soft', {},
            style_id='soft_skin', base_prompt=custom, character_research=research)
        assert 'No added scene effects:' in soft['effective_prompt']
        assert soft['design_brief']['primary_effect_plan']['mode'] == 'none'
        restriction = 'No effects. Preserve the original background.'
        limited = prepare_generation_design(source, root / 'limited', {},
            style_id='fantasy_atmosphere', base_prompt=restriction, character_research=research)
        assert limited['design_brief']['user_constraints']['no_effects'] is True
        assert limited['design_brief']['user_constraints']['preserve_background'] is True
        assert 'this overrides the selected style' in limited['effective_prompt']
        expect_rejected(plan_fixture(), brief=restriction)
        expect_rejected(plan_fixture(), brief='不要特效，保留原背景。')
        appended = prepare_generation_design(source, root / 'appended-user-veto', {},
            style_id='fantasy_atmosphere', base_prompt=DEFAULT_PROMPT + '\n\n' + restriction,
            character_research=research)
        assert appended['design_brief']['user_constraints']['no_effects']

        # Less common phrasing can be recognized by the planner, with an exact
        # current-user quotation required rather than a made-up exception.
        quoted = copy.deepcopy(response)
        quoted['user_constraints'] = {
            'no_effects_quote': 'Skip atmospheric additions.',
            'preserve_background_quote': 'Leave the location as photographed.',
        }
        normalize_design_brief(quoted, style_id='night_scene',
            user_brief='Skip atmospheric additions. Leave the location as photographed.')
        expect_rejected(quoted, brief='Make a finished portrait.')

        # Retired numeric plans are rejected instead of being retained beside
        # the current creative design.
        response = plan_fixture()
        response['environment_effects_plan'] = 'OLD ENVIRONMENT PLAN'
        response['composition_plan'] = 'OLD COMPOSITION PLAN'
        response['source_render_plan'] = {'enabled': True, 'light_fields': [
            {'scope': 'subject', 'center': [.4, .5], 'radius': [.2, .3], 'exposure_ev': .25}]}
        try:
            prepare_generation_design(source, root/'retired', {}, style_id='night_scene', base_prompt=custom, character_research=research)
            raise AssertionError('Retired numeric generation plan accepted')
        except ValueError: pass

        for missing in ('scene_transformation_plan', 'primary_effect_plan',
                        'spatial_integration_plan', 'character_basis', 'tradeoffs'):
            incomplete = plan_fixture()
            del incomplete[missing]
            expect_rejected(incomplete)
        empty = plan_fixture()
        empty['scene_transformation_plan']['visible_changes'] = []
        expect_rejected(empty)
        increment = plan_fixture()
        increment['foreground_effect_plan'] = {'enabled': True, 'layers': [{
            'id': 'front_wisp', 'slot': 'front', 'operation': 'over',
            'description': 'A distinct sparse front accent, produced separately from the baked main effect.',
            'occlusion': 'protect_face_hands', 'increment': True, 'baked_in_reference': False,
            'effect_design': {'material': 'scattering_smoke', 'colour_rgb8': [255,181,82],
                              'extent_2d': [0,.6,.4,.9], 'density': 'sparse'}}]}
        typed = normalize_design_brief(increment, style_id='night_scene', user_brief='')
        from generation_design import compile_design_direction
        assert 'Do not bake or draw these extra layers' in compile_design_direction(typed)
        assert typed['foreground_effect_plan']['layers'][0]['effect_design']['extent_2d'] == [0,.6,.4,.9]
        invalid_increment = copy.deepcopy(increment)
        invalid_increment['foreground_effect_plan']['layers'][0].pop('effect_design')
        expect_rejected(invalid_increment)
        invalid_increment = copy.deepcopy(increment)
        invalid_increment['foreground_effect_plan']['layers'][0]['effect_design']['material'] = 'portrait'
        expect_rejected(invalid_increment)
        invalid_increment = copy.deepcopy(increment)
        invalid_increment['foreground_effect_plan']['layers'][0]['baked_in_reference'] = True
        expect_rejected(invalid_increment)

        # Exercise the real workflow failure boundary, not a mock of the
        # planner module. Partial output must never be labelled a ready design.
        from single_source_workflow import prepare_bound_design
        response = {'summary': '不完整', 'english_art_direction': 'Only a generic colour grade.'}
        failed = prepare_bound_design(source, root / 'failed', {}, source_hash=sha256(source),
            style_id='night_scene', input_prompt=old, character_research=research)
        assert failed['status'] == 'unavailable' and failed['design_brief']['status'] == 'unavailable'
        assert 'OLD SHOT' not in failed['effective_prompt'] and 'OLD CHARACTER' not in failed['effective_prompt']
        assert json.loads((root / 'failed/generation_design.json').read_text(encoding='utf-8'))['status'] == 'unavailable'

    print(json.dumps({'ok': True, 'complete_plan_compiled': True,
        'structured_scene_effect_space_sent': True, 'dramatic_visible_design_required': True,
        'soft_portrait_without_magic': True, 'user_restrictions_override_style': True,
        'incomplete_planning_unavailable': True, 'retired_numeric_plan_rejected': True,
        'old_shot_removed': True, 'selected_material_scene_prompts': True,
        'network_calls': 0, 'image_generation': False}))


if __name__ == '__main__':
    main()
