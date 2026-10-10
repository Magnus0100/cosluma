"""Shared saved photographic intent and environment asset responsibility.

This module normalizes art constraints only. It does not infer physical lamps,
produce person pixels, generate effects, or replay captured reflection donors.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

VERSION = 'photographic-scene-design-v2-saved-intent'
SCHEMA = 'scene_design.v2'
INTENT_SCHEMA = 'saved_photographic_scene_intent.v1'
BRIEF_FIELDS = ('composition_plan', 'lighting_plan', 'palette_material_plan',
                'environment_effects_plan', 'subject_retouch_plan', 'summary')


def _read(value):
    return json.loads(Path(value).read_text(encoding='utf-8')) if isinstance(value, (str, Path)) else value


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def _text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest() if value else None


def _text_light_role(text, role):
    """Record only an explicit role/direction; absence stays unspecified.

    The bounded EN/ZH vocabulary is deterministic. It neither evaluates the
    text nor guesses a lamp distance, radiance, visibility or opposite fill.
    """
    terms = {'key': r'\b(?:moon key|key(?: light)?|main light|primary light)\b|主光',
             'fill': r'\bfill(?: light)?\b|补光|補光',
             'rim': r'\b(?:back/rim light|rim light|backlight|back light)\b|轮廓光|輪廓光|逆光',
             'ambient': r'\bambient(?: light)?\b|环境光|環境光'}
    for sentence in re.split(r'[.!?。！？\n]+', text):
        match = re.search(terms[role], sentence, re.I)
        if not match:
            continue
        if re.search(r'\b(?:no|without|avoid|never)\b[^,;]{0,24}$|不要[^，；]{0,12}$',
                     sentence[:match.start()], re.I):
            continue
        # An adjacent directional phrase belongs to this role, not another
        # light described later in the same sentence.
        local = sentence[max(0, match.start()-35):match.end()+110]
        after = re.search(r'\bfrom\s+([^,;]+)|(?:来自|來自|从|從)([^，；]+)',
                          sentence[match.end():match.end()+110], re.I)
        direction_text = (after.group(1) or after.group(2)) if after else local
        left = bool(re.search(r'\b(?:camera[- ]?)?left\b|左', direction_text, re.I))
        right = bool(re.search(r'\b(?:camera[- ]?)?right\b|右', direction_text, re.I))
        up = bool(re.search(r'\b(?:upper|above|top)\b|上方|左上|右上', direction_text, re.I))
        down = bool(re.search(r'\b(?:lower|below|bottom)\b|下方|左下|右下', direction_text, re.I))
        ambiguous = (left and right) or (up and down)
        direction = None if ambiguous or not any((left,right,up,down)) else [
            -1 if left else 1 if right else 0, 1 if up else -1 if down else 0]
        return {'status': 'specified', 'specification_text': sentence.strip(),
            'direction_to_light_xy': direction,
            'direction_confidence': .75 if direction else 0.,
            'interpretation_confidence': .75,
            'radiance_linear_rgb': None, 'angular_size_degrees': None,
            'executed': False,
            'execution_reason': 'art intent only; source geometry/material/visibility and light consumer are not established'}
    return {'status': 'unspecified', 'specification_text': None,
        'direction_to_light_xy': None, 'direction_confidence': 0.,
        'interpretation_confidence': 0., 'radiance_linear_rgb': None,
        'angular_size_degrees': None, 'executed': False,
        'execution_reason': 'no explicit supported role in saved text'}


def normalize_scene_intent(design_intent, reference=None):
    """Prefer a saved structured brief; otherwise interpret the saved prompt.

    Text is retained verbatim and hashed. Parsed directions are relative screen
    art constraints, not measurements or permission to enable an effect.
    """
    intent = _read(design_intent or {})
    brief = intent.get('design_brief') or intent
    reference = reference or {}
    prompt = reference.get('prompt') or ''
    if not isinstance(prompt, str):
        raise ValueError('Saved reference prompt must be text')
    descriptions = {name: copy.deepcopy(brief[name]) for name in BRIEF_FIELDS
                    if brief.get(name)}
    origins = {name: 'structured_brief' for name in descriptions}
    for name in ('lighting_plan', 'environment_effects_plan'):
        if name not in descriptions and prompt:
            descriptions[name] = prompt
            origins[name] = 'saved_reference_prompt_text_derived'
    lighting_text = descriptions.get('lighting_plan') or ''
    if not isinstance(lighting_text, str):
        raise ValueError('Saved lighting_plan must be descriptive text')
    environment = brief.get('local_environment_intent')
    if environment:
        from photographic_environment import normalize_environment_intent
        environment = normalize_environment_intent(environment)
    light = {'schema_version': 'photographic_light_intent.v1',
        'origin': 'structured_brief' if environment else origins.get('lighting_plan', 'unspecified'),
        'direction_frame': 'source_canvas_image_x_right_y_up_direction_toward_art_light',
        'roles': {role: _text_light_role(lighting_text, role)
                  for role in ('key', 'fill', 'rim', 'ambient')},
        'declared_environment_intent': environment,
        'description_sha256': _text_sha(lighting_text),
        'parser': 'bounded_explicit_photographic_vocabulary_en_zh_v1',
        'confidence_semantics': 'interpretation heuristic or explicit artistic declaration; not calibrated geometry/light confidence',
        'physical_lamps_recovered': False, 'physical_light_transport': False,
        'candidate_person_RGB_used': False, 'executable_light_fields_created': False}
    if environment:
        # Structured relative constraints outrank any text-derived direction.
        # Missing roles stay unspecified; do not invent opposite-side fill.
        light['direction_frame'] = environment['frame']
        light['roles'] = {role: _text_light_role('', role)
                          for role in ('key', 'fill', 'rim', 'ambient')}
        light['roles']['ambient'].update(status='declared_relative_art_constraint',
            radiance_linear_rgb=environment['ambient_linear_rgb'],
            interpretation_confidence=1., origin='structured_brief_environment')
        for item in environment['lights']:
            role_name=item['role'].lower()
            matches=[role for role,terms in (
                ('key',r'\b(?:key|main|primary)\b|主光'),
                ('fill',r'\bfill\b|补光|補光'),
                ('rim',r'\b(?:rim|backlight|back)\b|轮廓光|輪廓光|逆光'))
                if re.search(terms,role_name)]
            if len(matches)!=1 or light['roles'][matches[0]]['status']!='unspecified':
                continue  # All authored lights remain in declared_environment_intent.
            light['roles'][matches[0]].update(status='declared_relative_art_constraint',
                specification_text=item['role'], direction_to_light_xy=item['direction'][:2],
                direction_to_light_xyz=item['direction'], direction_confidence=1.,
                interpretation_confidence=1., radiance_linear_rgb=item['peak_linear_rgb'],
                angular_size_degrees=item['angular_fwhm_degrees'], origin='structured_brief_environment')
    return {'schema_version': INTENT_SCHEMA, 'descriptions': descriptions,
        'photographic_lighting': light,
        'provenance': {'normalization_recipe': VERSION,
            'description_origins': origins,
            'reference_sha256': reference.get('sha256'),
            'reference_prompt_sha256': _text_sha(prompt),
            'structured_brief_sha256': _json_sha(brief) if brief else None,
            'interpretation': 'saved art constraints; no new generation or recovered lamps'}}


def scene_intent_identity(design_intent, reference):
    """Content identity used before model preparation and cache lookup."""
    return _json_sha(normalize_scene_intent(design_intent, reference))


def _content_identity(value):
    # Saved assets may move from a cache into a run's owned closure. Moving
    # their paths cannot change artistic identity; their content hash can.
    if isinstance(value, dict):
        return {name: _content_identity(item) for name,item in value.items()
                if name not in {'path', 'identity'}}
    if isinstance(value, list):
        return [_content_identity(item) for item in value]
    return value


def bind_scene_environment(design, environment_assets):
    """Bind the actual plate/validity/removal used by this RenderPlan."""
    design = copy.deepcopy(design)
    required = {'background', 'validity', 'candidate_removal'}
    if not required.issubset(environment_assets):
        raise ValueError('Scene intent needs explicit background, validity and candidate-removal bindings')
    for name, asset in environment_assets.items():
        if not isinstance(asset, dict) or not asset.get('sha256'):
            raise ValueError('Scene environment asset has no content identity: '+name)
    design['light']['environment_asset_bindings'] = copy.deepcopy(environment_assets)
    design['light']['environment_binding_role'] = 'baked_design_observation_provenance_only; not calibrated light or clean-domain proof'
    design['identity'] = _json_sha(_content_identity(design))
    return design


def bind_plan_receiver_intent(run, plan):
    """Save authored landing choices against this plan's actual asset closure.

    Editing these choices never re-runs scene observations or promotes a saved
    scene schema. The declaration has its own version and no effect authority.
    """
    from scene_receiver_intent import bind_receiver_intent, normalize_receiver_intent
    plan = copy.deepcopy(plan)
    size = (plan.get('canvas') or {}).get('source_oriented_size') or run['source']['oriented_size']
    revision = (run.get('options') or {}).get('appearance_revision') or {}
    declaration = normalize_receiver_intent(revision.get('receiver_intent'), source_canvas_size=size)
    background = next((layer for layer in plan.get('layers', [])
                       if layer.get('kind') == 'background_design'), None)
    if background is None:
        return plan
    assets, masks = background['assets'], plan['mask_roles']
    environment = {'background': assets['background'], 'validity': assets['validity'],
                   'candidate_removal': masks['candidate_removal']}
    observations = {'foreground_alpha': masks['foreground_alpha']}
    contact = next((layer for layer in plan.get('layers', [])
                    if layer.get('kind') == 'contact_shadow'), None)
    if contact is not None:
        contact_assets = contact.get('assets') or {}
        if contact_assets.get('source_component_owner'):
            observations['foot_owner'] = contact_assets['source_component_owner']
        if contact_assets.get('receiver_support'):
            observations['receiver_observation'] = contact_assets['receiver_support']
    scene = plan['scene_design']
    scene['authored_receiver_intent'] = bind_receiver_intent(declaration,
        source=run['source'], reference=run['reference'], source_canvas_size=size,
        decode_version=plan['source_baseline']['decode_version'], environment_assets=environment,
        environment_mapping=plan['canvas'].get('candidate_mapping'), observation_assets=observations)
    scene['identity'] = _json_sha(_content_identity(scene))
    return plan


def build_scene_design(scene_analysis, design_intent, analysis_grid=None, *,
                       reference=None, environment_assets=None):
    """Resolve observations and saved art constraints without executing light."""
    scene, intent = _read(scene_analysis), _read(design_intent or {})
    brief = intent.get('design_brief') or intent
    normalized = normalize_scene_intent(intent, reference)
    descriptions = normalized['descriptions']
    receivers = [{'match_index': index,
        'source_mirror_index': match.get('source_mirror_index'),
        'candidate_mirror_index': match.get('candidate_mirror_index'),
        'geometry_supported': bool(match.get('supported')),
        'source_region': copy.deepcopy(match.get('source_reflection_region')),
        'appearance_policy': 'receiver_geometry_observation_only',
        'subject_reflection_status': 'unsupported_no_source_donor_or_recovered_reflection_transport',
        'physical_light_transport': False}
        for index,match in enumerate(scene.get('reflection_matches', []))]
    design = {'schema_version': SCHEMA, 'producer': VERSION,
        'analysis_grid': list(analysis_grid or scene['roles']['source']['size']),
        'coordinate_space': 'final_source_analysis_grid',
        'source_identity': copy.deepcopy(scene.get('source', {})),
        'candidate_identity': copy.deepcopy(scene.get('candidate', {})),
        'intent': descriptions, 'intent_provenance': normalized['provenance'],
        'layout': {'subject_geometry': 'immutable source observation',
            'environment_mapping': copy.deepcopy(scene.get('background_mapping', {})),
            'mapping_establishes_3d': False, 'new_layout_geometry': 'not_recovered'},
        'light': {'intent_description': descriptions.get('lighting_plan', ''),
            'photographic_intent': normalized['photographic_lighting'],
            'source_coordinate_fields': copy.deepcopy(brief.get('source_render_plan', {})),
            'radiometric_origin': 'RAW source observed radiance and generated display-linear environment',
            'source_Fp_is_albedo': False, 'intent_executes_effects': False},
        'receivers': receivers, 'contact': copy.deepcopy(scene.get('contact', {})),
        'effects': {'intent_description': descriptions.get('environment_effects_plan', ''),
            'candidate_baked_state': 'unknown unless separately observed',
            'increment_defaults': {'fog': False, 'rim': False, 'particles': False, 'bloom': False},
            'far_owner': 'environment_layer_stack', 'near_owner': 'post_composite_finish',
            'bloom_owner': 'visible_highlights_after_final_edits',
            'occlusion': 'explicit layer coverage required; no inferred 3D claim'},
        'detail': {'main_person': 'native source Fp observation',
            'person_donor_policy': 'source_observation_only; no_secondary_or_captured_donors',
            'environment': 'registered cleaned design P/V with completion origin; no source-background reinjection',
            'candidate_native_size': scene.get('candidate', {}).get('oriented_size'),
            'super_resolution_claim': False},
        'gpt_calls': 0, 'requires_generation': False}
    design['identity'] = _json_sha(_content_identity(design))
    return bind_scene_environment(design, environment_assets) if environment_assets is not None else design
