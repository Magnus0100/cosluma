"""Observe imported photo/reference pairs and compile their post-production intent."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from generation_prompt import DEFAULT_PROMPT, STYLE_MARKER, strip_generated_direction
from preset_store import get_preset
from reference_provider import sha256

DESIGN_VERSION = 'cosplay-researched-design-v7-character-portrait'
DESIGN_MARKER = '\n\nIMAGE-SPECIFIC ART DIRECTION\n'


def task_user_brief(task_intent: dict) -> str:
    """Compile explicit task constraints into the actual observed design brief."""
    from creative_task import task_brief
    return task_brief(task_intent)


def existing_reference_binding(source_hash: str, reference_hash: str, task_intent: dict,
                               profile: dict | None = None, style_id: str | None = None) -> dict:
    encoded = json.dumps(task_intent, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    profile_encoded = json.dumps(profile or {}, sort_keys=True, ensure_ascii=False)
    return {'source_sha256': source_hash, 'reference_sha256': reference_hash,
            'task_intent_sha256': hashlib.sha256(encoded.encode('utf8')).hexdigest(),
            'profile_sha256': hashlib.sha256(profile_encoded.encode('utf8')).hexdigest(),
            'style_id': style_id or '', 'input_prompt': task_user_brief(task_intent)}


def prepare_existing_reference_design(source: Path, reference: Path, out_dir: Path,
                                      task_intent: dict, profile: dict | None = None, *,
                                      style_id: str | None = None, provider=None,
                                      vision_model=None, api_base=None, api_url=None,
                                      api_key_env=None) -> dict:
    """Observe the actual source/reference pair once; never redraw the reference."""
    from PIL import Image, ImageDraw
    from raw_support import open_rgb
    from vision_transport import request_vision_json

    source, reference, out_dir = Path(source).resolve(), Path(reference).resolve(), Path(out_dir).resolve()
    binding = existing_reference_binding(sha256(source), sha256(reference), task_intent, profile, style_id)
    destination = out_dir/'generation_design.json'
    if destination.is_file():
        saved = json.loads(destination.read_text(encoding='utf8'))
        if saved.get('binding') == binding and saved.get('text_vision_submissions') == 1:
            return saved
    if (task_intent.get('mode') != 'existing_reference'
            or task_intent.get('services', {}).get('text_vision') is not True):
        return {'schema_version': DESIGN_VERSION, 'status': 'unavailable', 'binding': binding,
                'error': 'Paired design requires explicit text/vision permission; no request was sent.',
                'design_origin': 'existing_reference', 'image_submissions': 0, 'text_vision_submissions': 0}
    out_dir.mkdir(parents=True, exist_ok=True)
    board = Image.new('RGB', (1024, 576), '#202020')
    draw = ImageDraw.Draw(board)
    panels = {}
    for index, (role, path) in enumerate((('source', source), ('reference', reference))):
        image = open_rgb(path).convert('RGB')
        oriented_size = list(image.size)
        image.thumbnail((496, 536), Image.Resampling.LANCZOS)
        x = 8 + index*512 + (496-image.width)//2
        y = 32 + (536-image.height)//2
        board.paste(image, (x, y))
        draw.text((8 + index*512, 8), role.upper(), fill='white')
        panels[role] = {'path': str(path), 'sha256': binding[role+'_sha256'],
                        'oriented_size': oriented_size, 'panel_box': [x, y, x+image.width, y+image.height]}
    board_path = out_dir/'source-reference-observation.png'
    board.save(board_path)
    style = get_preset(style_id) if style_id else None
    style = style or {'id': 'existing_reference', 'name': 'Actual imported reference'}
    user_brief = task_user_brief(task_intent)
    prompt = _paired_design_prompt(profile or {},style,user_brief)
    prompt += """\nEXISTING REFERENCE TASK (overrides creative-generation instructions):
The supplied board has SOURCE on the left and the immutable REFERENCE on the right.
Observe both actual photographs separately. The source owns identity, geometry,
makeup detail and texture. The reference supplies only qualified colour/appearance
and background evidence. Design supported local post-production of this pair,
not a future generated reference. Image generation and reference correction
budgets are exactly zero. Do not propose new imagery to satisfy a missing effect.
Describe missing or incompatible scene/effect evidence in tradeoffs rather than
claiming it exists. Declare an independent increment only when the user explicitly
requires it and it is absent; it will remain unfulfilled without a supplied asset.
Do not infer the source pose, masks, pixels or coordinates from another shot.
No character research has been requested. User constraints override the reference:
preserve_background means keep source background; no_effects forbids additions.
Choose refine/transform and primary effect from this actual pair and user intent;
do not force a preset's dramatic mode or invent new lighting/physical reflection.
Add reference_background_policy with mode use_reference or preserve_source and
reason. preserve_source is allowed only for explicit preserve_background=true.
Add reference_effect_observation with present (boolean) and description of any
baked smoke, particles, magic, glow or added visual effects in the reference.
If no_effects=true and the reference contains such effects, report that conflict;
never pretend those pixels have been removed or switch backgrounds without the
explicit preserve_background constraint.
"""
    record = {'schema_version': DESIGN_VERSION, 'status': 'unavailable', 'binding': binding,
              'design_origin': 'existing_reference', 'source': panels['source'], 'reference': panels['reference'],
              'image_submissions': 0, 'text_vision_submissions': 1, 'character_research': {'status': 'not_requested'},
              'selected_style_id': style_id or '', 'input_prompt': user_brief,
              'source_sha256': binding['source_sha256'], 'reference_sha256': binding['reference_sha256'],
              'observation_board': {'path': str(board_path), 'sha256': sha256(board_path), 'panels': panels}}
    # Reserve before transport. An unknown response cannot grant a second plan request.
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    try:
        brief, provenance = request_vision_json(board_path, out_dir/'brief', prompt,
            provider=provider, vision_model=vision_model, api_base=api_base, api_url=api_url, api_key_env=api_key_env)
        if any(key in brief for key in ('source_render_plan', 'source_color_plan', 'local_environment_intent')):
            raise ValueError('Retired numeric generation plans are unsupported')
        brief = normalize_design_brief(brief, style_id='', user_brief=_user_brief(user_brief, profile or {}))
        policy = brief.get('reference_background_policy') or {}
        expected_mode = 'preserve_source' if task_intent['constraints']['preserve_background'] else 'use_reference'
        if policy.get('mode') != expected_mode:
            raise ValueError('Paired plan background policy contradicts the explicit task constraint')
        policy['reason'] = _required_text(policy.get('reason'), 'reference_background_policy.reason')
        observation = brief.get('reference_effect_observation') or {}
        if type(observation.get('present')) is not bool:
            raise ValueError('Paired plan must observe whether the reference contains baked effects')
        observation['description'] = _required_text(observation.get('description'), 'reference_effect_observation.description')
        if (task_intent['constraints']['no_effects'] and observation['present'] and expected_mode == 'use_reference'):
            raise ValueError('The imported reference contains baked effects incompatible with no_effects; no replacement or effect removal was performed')
        record.update(status='ready', design_brief=brief, provenance=provenance,
                      effective_prompt=user_brief + DESIGN_MARKER + compile_design_direction(brief))
    except Exception as exc:
        record['error'] = str(exc)
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return record


def _user_brief(base_prompt: str, profile: dict) -> str:
    # The untouched template, managed presets and previous generated plans
    # cannot invent a user veto. Text appended after the template is user text.
    editable = strip_generated_direction(base_prompt).split(STYLE_MARKER, 1)[0]
    if editable.startswith(DEFAULT_PROMPT):
        editable = editable[len(DEFAULT_PROMPT):].strip()
    return editable + '\n' + str(profile.get('user_notes') or '')


def _explicit_constraints(user_brief: str) -> dict:
    """Recognize common explicit vetoes; the planner can quote other wording."""
    no_effects = re.search(
        r'\b(?:no|without)\s+(?:(?:special|visual|magic|magical)\s+)?(?:effects?|fx|vfx)\b'
        r'|\b(?:do\s+not|don[\'’]t|never)\s+(?:add|use|create)\s+(?:any\s+)?'
        r'(?:(?:special|visual|magic|magical)\s+)?(?:effects?|fx|vfx)\b'
        r'|(?:不要|禁止|不添加|不加|不使用|不用)\s*(?:添加|使用)?\s*(?:任何|额外|魔法|视觉)?\s*特效',
        user_brief, re.IGNORECASE)
    preserve_background = re.search(
        r'\b(?:preserve|keep|retain)\s+(?:the\s+)?(?:(?:original|existing)\s+)?background\b'
        r'|\b(?:do\s+not|don[\'’]t|never)\s+(?:change|replace|modify|rebuild)\s+'
        r'(?:the\s+)?(?:(?:original|existing)\s+)?background\b'
        r'|(?:保留|保持)\s*(?:原始|原有|原)?\s*背景'
        r'|(?:不要|禁止|不)\s*(?:更换|替换|改变|修改|改动)\s*(?:原)?\s*背景',
        user_brief, re.IGNORECASE)
    return {'no_effects': bool(no_effects), 'preserve_background': bool(preserve_background)}


def _required_text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'The per-image planner returned no usable {field}')
    text = value.strip()
    if text.lower().rstrip('.') in {'none', 'n/a', 'not applicable', 'tbd', 'null'}:
        raise ValueError(f'The per-image planner returned a placeholder for {field}')
    return text


def normalize_design_brief(brief: dict, *, style_id: str, user_brief: str) -> dict:
    """Reject incomplete plans before they can be marked ready or sent."""
    if not isinstance(brief, dict):
        raise ValueError('The per-image planner must return a JSON object')
    result = dict(brief)
    for field in ('summary', 'scene_analysis', 'character_basis', 'subject_retouch_plan',
                  'spatial_integration_plan', 'lighting_plan', 'palette_material_plan',
                  'tradeoffs', 'reference_application', 'english_art_direction'):
        result[field] = _required_text(brief.get(field), field)

    constraints = _explicit_constraints(user_brief)
    supplied = brief.get('user_constraints') or {}
    if not isinstance(supplied, dict):
        raise ValueError('user_constraints must contain exact user quotations')
    for key in ('no_effects', 'preserve_background'):
        quote = supplied.get(key + '_quote', '')
        if not isinstance(quote, str):
            raise ValueError(f'user_constraints.{key}_quote must be text')
        quote = quote.strip()
        if quote and quote not in user_brief:
            raise ValueError(f'user_constraints.{key}_quote is not in the current user brief')
        constraints[key] = constraints[key] or bool(quote)
        constraints[key + '_quote'] = quote
    result['user_constraints'] = constraints

    scene = brief.get('scene_transformation_plan')
    if (not isinstance(scene, dict) or not isinstance(scene.get('mode'), str) or
            scene['mode'] not in {'refine', 'transform'}):
        raise ValueError('scene_transformation_plan must declare refine or transform')
    changes = scene.get('visible_changes')
    if not isinstance(changes, list) or not changes:
        raise ValueError('scene_transformation_plan requires visible scene decisions')
    result['scene_transformation_plan'] = {
        'mode': scene['mode'],
        'description': _required_text(scene.get('description'), 'scene_transformation_plan.description'),
        'visible_changes': [_required_text(value, 'scene_transformation_plan.visible_changes') for value in changes],
    }
    effect = brief.get('primary_effect_plan')
    if (not isinstance(effect, dict) or not isinstance(effect.get('mode'), str) or
            effect['mode'] not in {'none', 'scene_effect'}):
        raise ValueError('primary_effect_plan must declare none or scene_effect')
    result['primary_effect_plan'] = {
        'mode': effect['mode'],
        'description': _required_text(effect.get('description'), 'primary_effect_plan.description'),
        'reason': _required_text(effect.get('reason'), 'primary_effect_plan.reason'),
    }
    if constraints['no_effects'] and effect['mode'] != 'none':
        raise ValueError('The planned effects contradict the current user brief')
    if constraints['preserve_background'] and scene['mode'] != 'refine':
        raise ValueError('The scene replacement contradicts the current user brief')
    independent = brief.get('foreground_effect_plan')
    if independent is None:
        independent = {'enabled': False, 'layers': []}
    if (not isinstance(independent, dict) or set(independent) != {'enabled', 'layers'}
            or type(independent['enabled']) is not bool or not isinstance(independent['layers'], list)):
        raise ValueError('Independent effects require the explicit enabled/layers plan')
    if not independent['enabled'] and independent['layers']:
        raise ValueError('A disabled independent effect plan cannot contain layers')
    if independent['enabled']:
        if constraints['no_effects'] or effect['mode'] == 'none':
            raise ValueError('Independent effects contradict the current effect intent')
        from foreground_effects import declaration
        specs, _ = declaration({'generation_design': {'design_brief': {**result,
            'foreground_effect_plan': independent}}}, None)
        if len(specs) != len(independent['layers']):
            raise ValueError('Each independent layer must be an explicit unbaked increment')
        independent = {'enabled': True, 'layers': specs}
    result['foreground_effect_plan'] = independent
    # Existing saved-scene consumers read these two prose fields. Derive them
    # from the current structured plan; never retain a previous shot's aliases.
    result['composition_plan'] = result['english_art_direction'] + '\n' + result['tradeoffs']
    result['environment_effects_plan'] = '\n'.join(_environment_direction(result))
    return result


def _environment_direction(brief: dict) -> list[str]:
    scene, effect = brief['scene_transformation_plan'], brief['primary_effect_plan']
    parts = [
        ('Refine the existing background: ' if scene['mode'] == 'refine' else
         'Transform the scene around the fixed person and camera: ') + scene['description'],
        'Visible scene decisions: ' + '; '.join(scene['visible_changes']),
        ('No added scene effects: ' if effect['mode'] == 'none' else 'Primary scene effect: ') +
        effect['description'] + ' Purpose: ' + effect['reason'],
        'Depth, occlusion and contact: ' + brief['spatial_integration_plan'],
    ]
    if effect['mode'] == 'scene_effect':
        parts.append('Where the chosen effect contains illuminated mist or haze, keep its density and '
                     'brightness transitions gradual within the specified background region. Retain '
                     'real occlusion by hair, clothing, leaves and scene objects; any dark interruption '
                     'must follow an actual occluder or motivated light falloff.')
    return parts


def compile_design_direction(brief: dict) -> str:
    """Send structured scene/FX decisions even if the synthesis omits them."""
    constraints = brief['user_constraints']
    parts = [
        'Finished photograph concept: ' + brief['english_art_direction'],
        'Character and visible styling basis: ' + brief['character_basis'],
        'Subject retouch: ' + brief['subject_retouch_plan'],
        *_environment_direction(brief),
        'Overall and local illumination: ' + brief['lighting_plan'],
        'Palette and material finish: ' + brief['palette_material_plan'],
        'Priorities and restraint: ' + brief['tradeoffs'],
    ]
    if constraints['no_effects']:
        parts.append('The user forbids added effects; this overrides the selected style.')
    if constraints['preserve_background']:
        parts.append('The user requires the existing background; refine it without replacing its elements.')
    independent = brief.get('foreground_effect_plan') or {}
    if independent.get('enabled'):
        parts.append('Separate RGBA increments are produced after this reference. Do not bake or draw these extra layers into this reference: '+
                     '; '.join(row['id']+': '+row['description'] for row in independent['layers']))
    return '\n'.join(parts)


def correction_prompt(actions: list[dict], criteria=None) -> str:
    """Compile the actual selected failures for the three-image masked adapter."""
    allowed = {'skin_materials', 'concept_hierarchy', 'effects_space', 'scene_integration'}
    if not isinstance(actions, list) or not actions:
        raise ValueError('Correction requires nonempty current diagnosis actions')
    selected = set(criteria) if criteria is not None else {row.get('criterion') for row in actions}
    actual = {row.get('criterion') for row in actions}
    if (not selected or not selected.issubset(allowed | {'identity_makeup', 'native_detail'})
            or not actual.issubset(allowed)
            or any(row.get('criterion') not in selected or not isinstance(row.get('action'), str)
                   or not row['action'].strip() for row in actions)):
        raise ValueError('Only selected material, concept, effect or scene failures can generate corrections')
    material = 'skin_materials' in actual
    scene = bool(actual - {'skin_materials'})
    target = ('material colours and scene/effect relationships' if material and scene else
              'scene, effect, depth and focal hierarchy relationships' if scene else 'material colours')
    prompt = ('Correct the diagnosed ' + target + ' visibly and completely in <image1>, only in the '
        'white edit regions marked by <image3>. <image1> is the existing scene canvas and target; '
        '<image2> is the original photographed person, authoritative for identity, makeup contours, '
        'pose, anatomy, costume construction and native material texture. <image3> is a separate '
        'black/white edit mask in the exact <image1> canvas; white means edit and black means retain. '
        'Preserve untargeted content, the photographic medium, camera, perspective and framing. '
        'Never redraw the person or import a new face, body, makeup pattern or costume structure. ')
    if scene:
        prompt += ('Within the authorized scene area, add or correct the explicitly diagnosed missing '
            'effect, position, scale, depth, occlusion, contact or visual priority. Selected effect '
            'placement and hierarchy may change as required by the diagnosis; preserve untargeted '
            'scene objects and effects. Keep effects away from protected facial and body details. ')
        prompt += ('For diagnosed mist or haze defects, retain the intended fog colour and placement; '
            'repair its density and brightness transitions within the selected region while keeping '
            'actual hair, clothing, foliage and object occlusion intact. Keep existing haze strength '
            'and extent unless changing them is part of the selected diagnosis. ')
        if not material:
            prompt += 'Retain subject material colours; this correction edits the scene and effects only. '
    else:
        prompt += 'Preserve existing scene objects, effect placement, depth and hierarchy. '
    if material:
        prompt += ('Change only the diagnosed material colour relationships. Make transitions continuous '
            'without creating new structural contrast, sharpened edges or invented texture. ')
    label = ('MATERIAL AND SCENE CORRECTIONS' if material and scene else
             'SCENE AND EFFECT CORRECTIONS' if scene else 'MATERIAL CORRECTIONS')
    return (prompt + 'Return one photographic scene on <image1>\'s canvas.\n\nDIAGNOSED ' + label + ':\n' +
            '\n'.join(str(index+1)+'. '+row['action'].strip() for index, row in enumerate(actions))+'\n')


def _paired_design_prompt(profile: dict, style: dict, base_prompt: str) -> str:
    """One observed pair plan; no researched/future image-generation design."""
    character={key:profile.get(key) for key in ('character_name','series','costume_variant','identification','visual_cues','user_notes')}
    user_brief=_user_brief(base_prompt,profile)
    instructions="""Observe the supplied SOURCE and immutable REFERENCE photographs.
Describe supported post-production of this actual pair. The source owns the
person's identity, expression, pose, anatomy, makeup, costume and camera detail.
The reference is evidence, not permission to redraw or generate new content.
Keep source and reference observations separate. Do not claim missing material,
background or effects exist. No web research or image generation is requested.
Use the user's current brief and selected style; user restrictions override it.

Give concrete instructions for supported retouch, colour/material differences,
light relationships, foreground/background hierarchy and qualified reference
application. Existing reference effects must be observed honestly. Do not invent
new effects, new scenery, an unseen light source or physical measurements. Keep
photographic skin, makeup, eyes, matte fabric, leather and metal responses distinct.
Preserve actual shadows and occlusions. Avoid a uniform glowing contour, blanket
colour wash or invented detail. Put uncertain or unsupported changes in tradeoffs.

Return one JSON object with these fields:
summary, scene_analysis, character_basis, subject_retouch_plan,
scene_transformation_plan {mode:refine|transform,description,visible_changes:[text]},
primary_effect_plan {mode:none|scene_effect,description,reason},
foreground_effect_plan {enabled:false,layers:[]}, spatial_integration_plan,
lighting_plan, palette_material_plan, tradeoffs, reference_application,
english_art_direction, user_constraints {no_effects_quote,preserve_background_quote}.
Quote only exact current-user words for restrictions; otherwise use empty strings.
The plan describes qualified consumption of already supplied pixels, not future
image generation or numerical light/colour fields. Do not output coordinates,
RGB correction targets, source_render_plan, source_color_plan or local_environment_intent.
"""
    context={'character_context':character,'selected_photographic_style':style,
        'user_editable_brief':user_brief,'explicit_user_constraints':_explicit_constraints(user_brief)}
    return instructions+'\nREFERENCE DATA:\n'+json.dumps(context,ensure_ascii=False)
