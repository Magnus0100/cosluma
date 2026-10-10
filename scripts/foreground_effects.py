"""Generate, freeze and composite independently owned artistic RGBA effects."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from effect_conditioning import METHOD, compile_condition, prompt_from_condition, validate_condition, validate_effect_design
from effect_code_identity import current_code_identity, require_current_code
from installed_resources import resource_path
from PIL import Image
from scipy.ndimage import binary_dilation

SCHEMA = 'independent_effect_layer.v1'
KIND = 'independent_effect'
OPERATION_ABI = 'linear-associated-RGBA-source-canvas-over-emission-add-v1'
RECIPE = 'typed-effect-empty-qwen21-canvas-linear-associated-rgba-artistic-slots-v4'
_REVIEW_PREVIEW_REQUEST = ContextVar('independent_effect_review_preview_request', default=None)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def digest(path):
    path = Path(path)
    stat = path.stat()
    return _content_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=64)
def _content_digest(path, size, modified):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def brief_for_run(run):
    design = run.get('generation_design') or {}
    if isinstance(design, str):
        design = read(design)
    return design.get('design_brief') or design


def declaration(run, plan):
    """Only an actual scene requirement creates new effects; baked/unknown back FX stay owned by plate."""
    brief = brief_for_run(run)
    primary = brief.get('primary_effect_plan') or {}
    constraints = brief.get('user_constraints') or {}
    if constraints.get('no_effects') or constraints.get('no_effects_quote') or primary.get('mode') == 'none':
        return [], 'current_brief_forbids_or_omits_effects'
    authored = brief.get('foreground_effect_plan')
    if authored is None:
        return [], 'no_declared_independent_increment; background_main_effect_requires_final_review'
    if not isinstance(authored, dict) or not authored.get('enabled'):
        return [], 'independent_effect_plan_disabled'
    layers = authored.get('layers')
    if not isinstance(layers, list) or not 1 <= len(layers) <= 4:
        raise ValueError('Independent effects require one to four declared layers')
    result = []
    names = set()
    for item in layers:
        if not isinstance(item, dict):
            raise ValueError('Effect declaration must be an object')
        name = item.get('id')
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,47}', name) or name in names:
            raise ValueError('Effects require unique safe layer ids')
        names.add(name)
        slot, operation = item.get('slot'), item.get('operation')
        if slot not in {'behind', 'front'} or operation not in {'over', 'emission_add'}:
            raise ValueError('Declare behind/front and over/emission_add independently')
        if not isinstance(item.get('description'), str) or not item['description'].strip():
            raise ValueError('Effect requires actual brief description')
        if item.get('increment') is not True or item.get('baked_in_reference') is not False:
            # No inferred smoke subtraction or doubling a baked plate.
            continue
        occlusion = item.get('occlusion', 'protect_face_hands' if slot == 'front' else 'source_person')
        if occlusion not in ({'protect_face_hands', 'in_front'} if slot == 'front' else {'source_person'}):
            raise ValueError('Effect occlusion must name its known ownership source')
        if item.get('receiver_response', 'none') != 'none':
            raise ValueError('Independent RGBA effects do not establish scene light spill/reflection/contact')
        declared = {'id': name, 'slot': slot, 'operation': operation,
                       'description': item['description'].strip(), 'occlusion': occlusion,
                       'increment': True, 'baked_in_reference': False,
                       'receiver_response': 'none', 'physical_depth_established': False}
        if item.get('effect_condition') is not None:
            declared['effect_condition'] = validate_condition(item['effect_condition'])
        else:
            declared['effect_design'] = validate_effect_design(item.get('effect_design'))
        condition = declared.get('effect_condition') or declared['effect_design']
        scattering = condition['material'] in {'scattering_smoke', 'atmospheric_haze', 'dust_particles'}
        if scattering != (operation == 'over'):
            raise ValueError('Declared effect material must match scattering-over or emitted-radiance operation')
        result.append(declared)
    return result, 'declared_artistic_increment' if result else 'all_effects_baked_unknown_or_not_incremental'


def binding(run, plan):
    from scene_design import scene_intent_identity
    if plan.get('source_sha256') != run['source']['sha256'] or plan.get('candidate_sha256') != run['reference']['sha256']:
        raise ValueError('Independent effect plan differs from current source/reference')
    background = next(row for row in plan['layers'] if row['kind'] == 'background_design')['assets']
    result = {'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
            'source_alpha_sha256': plan['mask_roles']['foreground_alpha']['sha256'],
            'background_sha256': background['background']['sha256'],
            'background_validity_sha256': background['validity']['sha256'],
            'canvas_size': list(plan['canvas']['source_oriented_size']),
            'candidate_mapping': copy.deepcopy(plan['canvas'].get('candidate_mapping') or {}),
            'scene_intent_sha256': scene_intent_identity(run.get('generation_design') or {}, run['reference']),
            'actual_brief_sha256': identity(brief_for_run(run))}
    return result


def asset(path, **metadata):
    return {'path': str(Path(path).resolve()), 'sha256': digest(path), **metadata}


def validate_text_only_graph(graph, size, *, positive_prompt=None):
    """Validate active conditioning, canvas and output ancestry, not unused node presence."""
    def one(kind):
        rows = [(str(key), row) for key, row in graph.items() if row['class_type'] == kind]
        if len(rows) != 1:
            raise ValueError('Effect workflow requires exactly one '+kind)
        return rows[0]
    encoder_id, encoder = one('TextEncodeQwenImage21')
    canvas_id, canvas = one('CosplayEmptyQwen21Canvas')
    sampler_id, sampler = one('KSampler')
    decoder_id, decoder = one('VAEDecode')
    _, save = one('SaveImage')
    if (any(key.startswith('images') for key in encoder['inputs'])
            or encoder['inputs'].get('negative_prompt') != ''
            or (positive_prompt is not None and encoder['inputs'].get('prompt') != positive_prompt)
            or [canvas['inputs'][name] for name in ('width', 'height')] != list(size)
            or canvas['inputs'].get('batch_size') != 1
            or sampler['inputs'].get('latent_image') != [canvas_id, 0]
            or sampler['inputs'].get('positive') != [encoder_id, 0]
            or sampler['inputs'].get('negative') != [encoder_id, 1]
            or decoder['inputs'].get('samples') != [sampler_id, 0]
            or save['inputs'].get('images') != [decoder_id, 0]):
        raise ValueError('Active effect graph must use only text, the empty canvas and unchanged native RGBA decode')


def generation_request(root, run, plan, spec, directory, *, workflow_path=None):
    """A single reviewable request, with an explicit transparent-output workflow."""
    from types import SimpleNamespace
    from single_source_workflow import task_image_budget
    if task_image_budget(run, SimpleNamespace(), 'independent_effect', default=1) < 1:
        raise ValueError('Task image budget forbids a new independent effect request')
    root, directory = Path(root).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    reference = Path(run['reference']['path'])
    reference = reference if reference.is_absolute() else root/reference
    if digest(reference) != run['reference']['sha256']:
        raise ValueError('Selected scene-reference bytes changed')
    workflow_path = (Path(workflow_path) if workflow_path is not None else
        resource_path('providers/workflows/qwen-image-2.1-effect-rgba-text-only.json'))
    if not workflow_path.is_file():
        raise RuntimeError('RGBA effect workflow is unavailable')
    from comfyui_adapter import load_workflow, prepare_workflow
    workflow = load_workflow(workflow_path)
    if workflow.get('asset_contract', {}).get('method') != 'text_only_empty_64_channel_canvas_v2':
        raise ValueError('Automatic independent effects require text-only empty-canvas generation; photo edit/extraction is refused')
    reference_size = Image.open(reference).size
    scale = (1024**2/(reference_size[0]*reference_size[1]))**.5
    generated_size = [max(32, round(n*scale/32)*32) for n in reference_size]
    encoders = [row for row in workflow['workflow'].values() if row['class_type'] == 'TextEncodeQwenImage21']
    if len(encoders) != 1 or any(key.startswith('images') for key in encoders[0]['inputs']):
        raise ValueError('Effect-only encoding cannot receive reference images or their RGB latents')
    canvases = [row for row in workflow['workflow'].values() if row['class_type'] == 'CosplayEmptyQwen21Canvas']
    if len(canvases) != 1:
        raise ValueError('Text-only effect generation requires one real 64-channel Qwen21 canvas')
    canvases[0]['inputs'].update(width=generated_size[0], height=generated_size[1], batch_size=1)
    node_source = resource_path('providers/comfyui/cosplay_effect_canvas.py')
    if not node_source.is_file():
        raise RuntimeError('Text-only effect canvas implementation is missing')
    node_target = directory/'canvas-node.py'
    node_target.write_bytes(node_source.read_bytes())
    # Freeze this exact dimensional workflow and implementation before any
    # service request; the installed global template may later be upgraded.
    effective = directory/'effective-workflow.json'
    write(effective, workflow)
    workflow_path = effective
    empty_path = directory/'empty-effect-canvas.png'
    Image.new('RGBA', tuple(generated_size), (0, 0, 0, 0)).save(empty_path)
    graph = prepare_workflow(workflow, 'pending-scene.png', 'pending-effect-prompt')
    validate_text_only_graph(graph, generated_size, positive_prompt='pending-effect-prompt')
    saves = [node for node in graph.values() if node['class_type'] == 'SaveImage']
    if len(saves) != 1:
        raise ValueError('Effect workflow requires one preserved RGBA SaveImage output')
    decoder = graph.get(str(saves[0]['inputs']['images'][0]), {})
    if decoder.get('class_type') != 'VAEDecode':
        raise ValueError('Effect workflow must preserve native VAE RGBA directly without alpha stripping')
    # Complete role/scene prose remains binding/QA context. Only a closed
    # effect condition reaches the positive model text, including no negative
    # human/wardrobe nouns that would themselves activate subject generation.
    context = {'canvas_size': generated_size, 'photo_input_to_model': False,
        'role_scene_prose_is_model_input': False, 'actual_effect_requirement': spec['description'],
        'lighting_intent_QA_only': brief_for_run(run).get('lighting_plan'),
        'palette_intent_QA_only': brief_for_run(run).get('palette_material_plan')}
    effect_condition = compile_condition(spec, plan)
    prompt = prompt_from_condition(effect_condition)
    selected = binding(run, plan)
    fx_code = current_code_identity()
    identity_inputs = {'binding': selected, 'specification': spec, 'workflow_sha256': digest(workflow_path),
                           'canvas_node_sha256': digest(node_target),
                           'effect_condition': effect_condition, 'fx_code_identity': fx_code,
                           'prompt': prompt, 'recipe': RECIPE}
    task = run.get('task_intent') or (run.get('options') or {}).get('task_intent')
    if task:
        identity_inputs['task_image_policy'] = {'mode': task['mode'],
            'independent_effect': task_image_budget(run, SimpleNamespace(), 'independent_effect', default=0)}
    request_id = identity(identity_inputs)
    sample = 'effect_'+spec['id']
    generated = directory/'generated'
    argv = ['--root', str(root), '--image', str(empty_path.resolve()), '--source', str(reference.resolve()),
            '--sample', sample, '--prompt', prompt, '--provider', 'qwen_image_2_1',
            '--model', str(workflow_path.resolve()), '--out-dir', str(generated),
            '--preserve-alpha', '--asset-role', 'effect_layer', '--timeout', '900']
    request = {'schema': 'independent_effect_generation_request.v1', 'identity': request_id,
               'binding': selected, 'specification': copy.deepcopy(spec), 'prompt': prompt,
               'workflow': asset(workflow_path), 'canvas_node': asset(node_target),
               'fx_code_identity': fx_code,
               'condition': asset(empty_path, role='empty_effect_canvas_no_photo_RGB'),
               'scene_context': asset(reference, role='scene_context_binding_only_not_model_input'),
               'generation_method': METHOD, 'effect_condition': effect_condition,
               'automatic_context': context, 'photo_RGB_conditioned': False,
               'canvas_latent': {'size': generated_size, 'channels': 64, 'initial_values': 'zero',
                                'source_photo_pixels': False},
               'generated_image_path': str(generated/(sample+'.png')),
               'generated_provenance_path': str(generated/(sample+'.provenance.json')),
               'runtime_receipt_path': str(directory/'generation-runtime.json'),
               'provider_argv': argv, 'automatic_retry': False, 'source_RGB_output_authority': False}
    if task:
        request['task_image_policy'] = copy.deepcopy(identity_inputs['task_image_policy'])
    write(directory/'generation-request.json', request)
    return request


def produce_request(request_path):
    """Execute one frozen request; failed submissions remain evidence, never automatic retries."""
    request = read(request_path)
    task = request.get('task_image_policy')
    if task and (task.get('mode') != 'create_reference' or task.get('independent_effect') != 1):
        raise ValueError('Frozen task policy forbids independent effect image submission')
    identity_inputs = {'binding': request['binding'], 'specification': request['specification'],
                         'workflow_sha256': request['workflow']['sha256'],
                         'canvas_node_sha256': request['canvas_node']['sha256'],
                         'effect_condition': request['effect_condition'], 'fx_code_identity': request['fx_code_identity'],
                         'prompt': request['prompt'], 'recipe': RECIPE}
    if task:
        identity_inputs['task_image_policy'] = task
    expected = identity(identity_inputs)
    if request.get('schema') != 'independent_effect_generation_request.v1' or request['identity'] != expected:
        raise ValueError('Frozen independent effect request identity changed')
    require_current_code(request['fx_code_identity'])
    for name in ('workflow', 'condition', 'canvas_node', 'scene_context'):
        item = request[name]
        if digest(item['path']) != item['sha256']:
            raise ValueError('Frozen effect request input changed: '+name)
    if (request.get('generation_method') != METHOD
            or request.get('photo_RGB_conditioned') is not False):
        raise ValueError('Photo-conditioned portrait extraction cannot run as an independent effect')
    if request['prompt'] != prompt_from_condition(request['effect_condition']):
        raise ValueError('Only the effect whitelist schema can produce model text; arbitrary role prose refused')
    from foreground_effects_runtime import ensure_effect_runtime
    # Service startup/reload performs no diffusion/model inference. The
    # coordinator has reserved its one request before entering this function.
    root = Path(request['provider_argv'][request['provider_argv'].index('--root')+1])
    if digest(resource_path('providers/comfyui/cosplay_effect_canvas.py')) != request['canvas_node']['sha256']:
        raise ValueError('Installed effect canvas implementation differs from frozen request')
    runtime = ensure_effect_runtime(root)
    if runtime.get('loaded_node_sha256') != request['canvas_node']['sha256']:
        raise ValueError('Actual loaded service implementation differs from frozen canvas node')
    write(request['runtime_receipt_path'], runtime)
    from reference_provider import main as provider_main
    return provider_main(request['provider_argv'])


def decode_rgba(path):
    with Image.open(path) as image:
        if image.mode != 'RGBA':
            raise ValueError('Effect output must be actual RGBA; RGB cannot be assigned invented alpha')
        if image.info.get('icc_profile'):
            from PIL import ImageCms
            import io
            profile = ImageCms.ImageCmsProfile(io.BytesIO(image.info['icc_profile']))
            converted = ImageCms.profileToProfile(image.convert('RGB'), profile, ImageCms.createProfile('sRGB'), outputMode='RGB')
            rgb = np.asarray(converted, np.float32)/255
        else:
            rgb = np.asarray(image, np.float32)[..., :3]/255
        alpha = np.asarray(image.getchannel('A'), np.float32)/255
    if not np.any(alpha <= .01) or not np.any(alpha > .01):
        raise ValueError('Actual effect alpha needs visible and transparent regions')
    linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4).astype(np.float32)
    # Linearize straight colour FIRST, then associate once BEFORE resampling.
    return linear*alpha[..., None], alpha


def _protection(plan, size):
    semantics = (plan.get('reconstruction_observations') or {}).get('source_semantics')
    if semantics is not None and not Path(semantics['path']).is_file():
        semantics = None
    if semantics is None:
        contact = next((x for x in plan['layers'] if x['kind'] == 'contact_shadow'), {})
        semantics = contact.get('assets', {}).get('source_shoe_semantics')
    if semantics is None:
        semantics = next((row['assets']['source_semantics'] for row in plan['layers']
                          if row.get('kind') == KIND and row.get('assets', {}).get('source_semantics')), None)
    if semantics is None or digest(semantics['path']) != semantics['sha256']:
        raise ValueError('Front effects protecting face/hands require frozen actual source semantics')
    with np.load(semantics['path']) as data:
        labels = np.asarray(Image.fromarray(data['labels']).resize(size, Image.Resampling.NEAREST))
    protect = binary_dilation(np.isin(labels, [3, 6, 15, 24, 25, 26, 27, 28]), iterations=max(1, round(min(size)*.002)))
    source = {'semantic_sha256': semantics['sha256'], 'classes': [3, 6, 15, 24, 25, 26, 27, 28],
              'semantic_asset': copy.deepcopy(semantics),
              'method': 'nearest_semantics_then_small_automatic_dilation; artistic_feature_protection_not_depth'}
    return (~protect).astype(np.float32), source


def bind_result(run, plan, request, image_path, provenance_path, output_dir):
    """Freeze the generated field against the current source/scene, pending visual acceptance."""
    if request['binding'] != binding(run, plan):
        raise ValueError('Effect generation belongs to a different source/scene/alpha/brief')
    spec = request['specification']
    specs, _ = declaration(run, plan)
    if spec not in specs:
        raise ValueError('Effect requirement changed or is no longer permitted')
    provenance = read(provenance_path)
    if provenance.get('asset_role') != 'effect_layer' or not provenance.get('alpha', {}).get('preserved'):
        raise ValueError('Registered effect must preserve real alpha')
    if provenance.get('asset', {}).get('sha256') != digest(image_path):
        raise ValueError('Registered effect image identity changed')
    if provenance.get('source', {}).get('sha256') != request['scene_context']['sha256']:
        raise ValueError('Effect provider recorded scene context differs from the frozen reference')
    if provenance.get('prompt') != request['prompt']:
        raise ValueError('Effect provider prompt differs from frozen request')
    premultiplied, alpha = decode_rgba(image_path)
    reference_size = list(Image.open(request['scene_context']['path']).size)
    generated_size = list(alpha.shape[::-1])
    if abs((generated_size[0]/generated_size[1])/(reference_size[0]/reference_size[1])-1) > .035:
        raise ValueError('Effect generation changed camera canvas aspect ratio')
    canvas = request['binding']['canvas_size']
    ratio = min(1., 2048/max(canvas))
    size = [max(1, round(x*ratio)) for x in canvas]
    from design_scene_assets import warp_values
    mapping = request['binding']['candidate_mapping']
    matrix = np.asarray(mapping.get('candidate_original_to_final_original'), np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Effect mapping requires the current explicit candidate-to-source canvas transform')
    scale = lambda a, b: np.diag([b[0]/a[0], b[1]/a[1], 1.])
    from effect_layout import PARAMETERS, SCHEMA as LAYOUT_SCHEMA, layout_transform, measure_before_luminance, bound_dose
    transform, extent_visibility, support_box = layout_transform(alpha, request['effect_condition']['extent_2d'], size)
    mapped, _ = warp_values(np.dstack([premultiplied, alpha]), transform, size)
    visibility = extent_visibility.copy()
    mapped *= extent_visibility[..., None]
    protection = {'method': 'none', 'source': 'declared_artistic_front_slot' if spec['slot'] == 'front' else 'source_alpha_at_final_person_composition'}
    if spec['occlusion'] == 'protect_face_hands':
        feature_visibility, protection = _protection(plan, tuple(size))
        visibility *= feature_visibility
        mapped *= feature_visibility[..., None]
    before_luma, measurement = measure_before_luminance(run, plan, size)
    mapped, dose = bound_dose(mapped, before_luma, extent_visibility, request['effect_condition']['material'])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    field_path = output_dir/'associated-effect-field.npz'
    measurement_path = output_dir/'before-luminance-measurement.npz'
    np.savez_compressed(measurement_path, values=measurement)
    np.savez_compressed(field_path, premultiplied_rgb=mapped[..., :3], alpha=np.clip(mapped[..., 3], 0, 1),
                        emission=mapped[..., :3], visibility=visibility)
    data = asset(field_path, size=size, canvas_size=canvas, coordinate_space='source_canvas',
                 color_space='linear_sRGB', representation='linear_associated_RGBA', sampling='bilinear')
    fx_code = current_code_identity()
    from effect_sampling import SAMPLING_ABI
    sampling_code = {'effect_sampling.py': fx_code['files']['effect_sampling.py']}
    consumer_identity = identity({'request_identity': request['identity'], 'binding': request['binding'],
        'specification': spec, 'fx_code_identity': fx_code, 'consumer_parameters': PARAMETERS,
        'sampling_abi': SAMPLING_ABI, 'sampling_code_identity': sampling_code})
    contract = {'schema': SCHEMA, 'recipe': RECIPE, 'binding': request['binding'],
                'request_identity': request['identity'], 'specification': spec, 'field_sha256': data['sha256'],
                'fx_code_identity': fx_code, 'consumer_identity': consumer_identity,
                'operation_abi': OPERATION_ABI,
                'sampling_abi': SAMPLING_ABI, 'sampling_code_identity': sampling_code,
                'provider_image': asset(image_path), 'provider_provenance': asset(provenance_path),
                'generation_request': copy.deepcopy(request),
                'generation_method': request['generation_method'], 'source_photo_RGB_conditioned': False,
                'alpha_conversion': 'IEC_sRGB_straight_to_linear_then_alpha_once_before_map_sample',
                'generated_size': generated_size, 'mapped_size': size, 'mapping': transform.tolist(),
                'consumer_layout': {'schema': LAYOUT_SCHEMA, 'parameters': copy.deepcopy(PARAMETERS),
                    'extent_2d': request['effect_condition']['extent_2d'], 'source_alpha_support_bbox': support_box,
                    'mapping': 'uniform_fit_alpha_support_to_frozen_source_canvas_extent; hard_extent_mask',
                    'preserves_source_person_geometry': True, 'dose': dose,
                    'before_luminance': asset(measurement_path)},
                'spatial_registration_supported': mapping.get('scene_registration_supported') is True,
                'physical_depth_established': False, 'occlusion': protection,
                'scene_object_depth': 'unknown; artistic placement and source-person/feature protection only; no recovered timber/paper depth',
                'receiver_response': 'none; no automatic surface_light_shadow_or_reflection',
                'baked_back_effects_consumed': False, 'source_person_RGB_used': False,
                'quality_status': 'pending_visual_review',
                'alpha_stats': {'transparent_pixels': int(np.count_nonzero(alpha <= .01)),
                                'visible_pixels': int(np.count_nonzero(alpha > .01)),
                                'fractional_pixels': int(np.count_nonzero((alpha > 0)&(alpha < 1)))}}
    contract_path = output_dir/'effect-contract.json'
    write(contract_path, contract)
    assets = {'effect': data, 'contract': asset(contract_path),
              'generated_rgba': asset(image_path), 'provider_provenance': asset(provenance_path),
              'before_luminance': asset(measurement_path)}
    if Path(request['runtime_receipt_path']).is_file():
        assets['generation_runtime'] = asset(request['runtime_receipt_path'])
    if protection.get('semantic_asset'):
        assets['source_semantics'] = copy.deepcopy(protection['semantic_asset'])
    return {'id': 'effect_'+spec['id'], 'kind': KIND, 'enabled': False, 'strength': 1.,
            'slot': spec['slot'], 'operation': spec['operation'], 'params': {'schema': SCHEMA},
            'assets': assets,
            'provenance': {'request_identity': request['identity'], 'quality_status': 'pending_visual_review',
                           'consumer_identity': consumer_identity,
                           'source_person_RGB_used': False, 'physical_depth_established': False}}


def validate_layer(layer, run, plan, *, require_accepted=True):
    if layer.get('kind') != KIND or layer.get('params', {}).get('schema') != SCHEMA:
        raise ValueError('Independent effect contract schema required')
    if (_REVIEW_PREVIEW_REQUEST.get() is not None
            and _REVIEW_PREVIEW_REQUEST.get() == (layer.get('provenance') or {}).get('request_identity')):
        require_accepted = False
    for name in ('effect', 'contract'):
        row = layer['assets'][name]
        if digest(row['path']) != row['sha256']:
            raise ValueError('Frozen independent effect asset changed: '+name)
    contract = read(layer['assets']['contract']['path'])
    require_current_code(contract.get('fx_code_identity'))
    if contract.get('operation_abi') != OPERATION_ABI:
        raise ValueError('Historical effect operator ABI is unavailable; retain its saved read-only master')
    from effect_sampling import SAMPLING_ABI
    if contract.get('sampling_abi') != SAMPLING_ABI:
        raise ValueError('Historical effect sampling ABI is unavailable; retain its saved read-only master')
    from effect_layout import SCHEMA as LAYOUT_SCHEMA, PARAMETERS
    if (contract.get('consumer_layout', {}).get('schema') != LAYOUT_SCHEMA
            or contract['consumer_layout'].get('parameters') != PARAMETERS
            or contract.get('sampling_code_identity') !=
                {'effect_sampling.py': contract['fx_code_identity']['files']['effect_sampling.py']}):
        raise ValueError('Only current independent effect layout/parameters/sampling code are supported')
    expected_consumer = identity({'request_identity': contract['request_identity'], 'binding': contract['binding'],
        'specification': contract['specification'], 'fx_code_identity': contract['fx_code_identity'],
        'consumer_parameters': contract.get('consumer_layout', {}).get('parameters'),
        'sampling_abi': contract['sampling_abi'], 'sampling_code_identity': contract['sampling_code_identity']})
    if (contract.get('consumer_identity') != expected_consumer
            or layer['provenance'].get('consumer_identity') != expected_consumer):
        raise ValueError('Frozen independent effect consumer implementation/parameters changed')
    if contract.get('schema') != SCHEMA or contract.get('binding') != binding(run, plan):
        raise ValueError('Effect source/scene/brief/alpha binding is stale')
    if contract['field_sha256'] != layer['assets']['effect']['sha256']:
        raise ValueError('Effect field differs from accepted contract')
    if layer['slot'] != contract['specification']['slot'] or layer['operation'] != contract['specification']['operation']:
        raise ValueError('Effect slot or operation changed after generation')
    if contract['specification'] not in declaration(run, plan)[0]:
        raise ValueError('Current main brief no longer requests this effect')
    if require_accepted and contract.get('quality_status') != 'accepted_visual_review':
        raise ValueError('Unreviewed independent effect cannot be consumed')
    if require_accepted:
        from automatic_foreground_effects_review import validate_effect_review
        validate_effect_review(contract.get('visual_review'), layer, run, plan, require_pass=True)
    return contract


def accept_layer(layer, run, plan, *, review):
    """Bind an actual visual verdict, not alpha range or transport success, before activation."""
    layer = copy.deepcopy(layer)
    contract = validate_layer(layer, run, plan, require_accepted=False)
    from automatic_foreground_effects_review import validate_effect_review
    validate_effect_review(review, layer, run, plan, require_pass=True)
    contract['quality_status'] = 'accepted_visual_review'
    contract['visual_review'] = copy.deepcopy(review)
    original = Path(layer['assets']['contract']['path'])
    target = original.with_name('effect-contract-accepted-'+identity(review)[:12]+'.json')
    write(target, contract)
    layer['assets']['contract'] = asset(target)
    layer['enabled'] = True
    layer['provenance']['quality_status'] = contract['quality_status']
    return layer


def apply_slot(values, plan, run, slot, box, canvas_size, *, sample=None):
    """Behind operates on background; front operates on the composite. Person coverage occurs once."""
    from effect_sampling import sample_frozen_effect
    # The frozen operator's bilinear global grid must be shared by full/ROI;
    # generic local-box samplers are not equivalent at noninteger enlargement.
    sample = lambda row, key: sample_frozen_effect(row, key, box, canvas_size)
    records = []
    for layer in plan.get('layers', []):
        if layer.get('kind') != KIND or layer.get('slot') != slot or not layer.get('enabled'):
            continue
        strength = float(layer.get('strength', 1))
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('Effect strength must be finite in [0,1]')
        if not strength:
            continue
        # Only the in-process diagnostic renderer may show the one pending
        # request under review. No run option, JSON status, or saved knob can
        # bypass the actual review requirement for a deliverable render.
        pending_preview = (_REVIEW_PREVIEW_REQUEST.get() is not None
                           and _REVIEW_PREVIEW_REQUEST.get() == layer['provenance']['request_identity'])
        contract = validate_layer(layer, run, plan, require_accepted=not pending_preview)
        field = layer['assets']['effect']
        if layer['operation'] == 'over':
            p, a = sample(field, 'premultiplied_rgb'), sample(field, 'alpha')
            values = p*strength+(1-a[..., None]*strength)*values
        else:
            values = values+sample(field, 'emission')*strength
        records.append({'id': layer['id'], 'kind': KIND, 'enabled': True, 'applied': True,
                        'strength': strength, 'slot': slot, 'operation': layer['operation'],
                        'execution_stage': 'background_before_person' if slot == 'behind' else 'front_after_person',
                        'RGBA_alpha_consumed_once': True, 'source_person_RGB_used': False,
                        'physical_depth_established': False, 'occlusion_source': contract['occlusion'],
                        'receiver_response': contract['receiver_response']})
    return values, records


def render_review_preview(run, plan, layer, *, maximum_edge=768):
    """Render before/after with the real compositor for diagnostic images only.

    The pending effect's request has scoped authority only during this function;
    its contract stays pending and no saved layer becomes accepted or enabled.
    """
    from design_reconstruction import apply_plan, sample_asset
    from native_source_baseline import source_for_plan
    from display_transform import plan_for_run as display_for_run
    from raw_support import linear_preview
    validate_layer(layer, run, plan, require_accepted=False)
    native = tuple(plan['canvas']['source_oriented_size'])
    ratio = min(1., maximum_edge/max(native))
    size = tuple(max(1, round(n*ratio)) for n in native)
    source = source_for_plan(plan, size)
    alpha = sample_asset(plan['mask_roles']['foreground_alpha'], 'values', (0, 0, *size), size)
    geometry = Image.fromarray(np.uint8(np.clip(alpha, 0, 1)*255))
    before_plan = copy.deepcopy(plan)
    before_plan['layers'] = [row for row in before_plan['layers'] if row.get('id') != layer['id']]
    before_run = copy.deepcopy(run)
    before_run.setdefault('options', {}).setdefault('appearance_revision', {})['design_reconstruction_plan'] = before_plan
    before, _ = apply_plan(source, source.copy(), before_plan, geometry, run=before_run)
    after_plan = copy.deepcopy(before_plan)
    pending = copy.deepcopy(layer)
    pending['enabled'] = True
    after_plan['layers'].append(pending)
    after_run = copy.deepcopy(before_run)
    after_run['options']['appearance_revision']['design_reconstruction_plan'] = after_plan
    token = _REVIEW_PREVIEW_REQUEST.set(layer['provenance']['request_identity'])
    try:
        after, _ = apply_plan(source, source.copy(), after_plan, geometry, run=after_run)
    finally:
        _REVIEW_PREVIEW_REQUEST.reset(token)
    display = display_for_run(run)
    return {'source': linear_preview(source, display_transform=display),
            'composite_before': linear_preview(before, display_transform=display),
            'composite_after': linear_preview(after, display_transform=display)}, {
        'source_size': list(native), 'display_size': list(size),
        'renderer': 'design_reconstruction.apply_plan; scoped_pending_request_preview_only',
        'native_detail_acceptance': False, 'effect_contract_accepted': False,
        'parent_plan_changed': False}


def prepare_effects(root, run, plan, *, generate=False, output_dir=None, workflow_path=None):
    """Automatic producer entry: ordinary portraits skip, required increments are cached by actual scene identity."""
    specs, reason = declaration(run, plan)
    result = copy.deepcopy(plan)
    if not specs:
        result['layers'] = [x for x in result.get('layers', []) if x.get('kind') != KIND]
        result.setdefault('provenance', {})['independent_effects'] = {'status': 'skipped', 'reason': reason}
        return {'ok': True, 'plan': result, 'status': 'skipped', 'requests': []}
    from types import SimpleNamespace
    from single_source_workflow import task_image_budget
    if task_image_budget(run, SimpleNamespace(), 'independent_effect', default=1) < 1:
        ready, missing = [], []
        owned_layers = {row['id']: row for row in result.get('layers', []) if row.get('kind') == KIND}
        for spec in specs:
            owned = owned_layers.get('effect_'+spec['id'])
            if owned and owned.get('provenance', {}).get('quality_status') == 'accepted_visual_review':
                validate_layer(owned, run, plan, require_accepted=True)
                ready.append(owned)
            else:
                missing.append(spec['id'])
        result['layers'] = [row for row in result.get('layers', []) if row.get('kind') != KIND]+ready
        status = 'blocked_by_image_budget' if missing else 'prepared_frozen'
        result.setdefault('provenance', {})['independent_effects'] = {'status': status,
            'unfulfilled_effect_ids': missing, 'image_submissions': 0,
            'reason': 'New effect images are forbidden by the bound task.'}
        return {'ok': True, 'plan': result, 'status': status, 'requests': [], 'unfulfilled_effect_ids': missing}
    directory = Path(output_dir or Path(root)/'outputs/cache/independent-effects'/identity(binding(run, plan))[:20])
    requests, layers = [], []
    existing = {x['id']: x for x in result['layers'] if x.get('kind') == KIND}
    for spec in specs:
        owned = existing.get('effect_'+spec['id'])
        if owned:
            # Only current producer/ABI layers may replay their frozen field.
            validate_layer(owned, run, plan, require_accepted=bool(owned.get('enabled')))
            layers.append(owned)
            continue
        if spec['occlusion'] == 'protect_face_hands':
            _protection(plan, (96, 144))  # validate observed semantic ownership before any GPU submission
        folder = directory/spec['id']
        request = generation_request(root, run, plan, spec, folder, workflow_path=workflow_path)
        request_path = folder/'generation-request.json'
        frozen = folder/'prepared-layer.json'
        owned = existing.get('effect_'+spec['id'])
        if owned and owned.get('provenance', {}).get('request_identity') == request['identity']:
            # The saved layer is an owned producer result even after its cache
            # has gone. Refreshing declarations must not discard it merely
            # because generation used a run-specific coordinator directory.
            validate_layer(owned, run, plan, require_accepted=bool(owned.get('enabled')))
            layers.append(owned)
            continue
        if frozen.is_file():
            layer = read(frozen)
            validate_layer(layer, run, plan, require_accepted=False)
            if layer['provenance']['request_identity'] != request['identity']:
                raise ValueError('Cached effect producer/workflow changed; use a new output directory')
        elif Path(request['generated_image_path']).is_file() and Path(request['generated_provenance_path']).is_file():
            layer = bind_result(run, plan, request, request['generated_image_path'], request['generated_provenance_path'], folder/'prepared')
            write(frozen, layer)
        elif generate:
            if produce_request(request_path) != 0:
                raise RuntimeError('Independent RGBA producer failed')
            layer = bind_result(run, plan, request, request['generated_image_path'], request['generated_provenance_path'], folder/'prepared')
            write(frozen, layer)
        else:
            requests.append({**request, 'request_path': str(request_path.resolve())})
            continue
        layers.append(layer)
    # Preserve controls when a prepared identity is reused; no redraw on toggle.
    result['layers'] = [x for x in result['layers'] if x.get('kind') != KIND]
    for layer in layers:
        if layer['id'] in existing and existing[layer['id']]['provenance']['request_identity'] == layer['provenance']['request_identity']:
            layer = existing[layer['id']]
            validate_layer(layer, run, result, require_accepted=bool(layer.get('enabled')))
        result['layers'].append(layer)
    attached = [x for x in result['layers'] if x.get('kind') == KIND]
    status = 'pending_generation' if requests else ('pending_visual_review'
        if any(x.get('provenance', {}).get('quality_status') != 'accepted_visual_review' for x in attached) else 'prepared')
    result.setdefault('provenance', {})['independent_effects'] = {'status': status, 'reason': reason,
        'request_identities': [x['identity'] for x in requests], 'source_person_RGB_used': False,
        'physical_depth_established': False, 'baked_back_effects_duplicated': False}
    return {'ok': True, 'plan': result, 'status': status, 'requests': requests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    produce = commands.add_parser('produce')
    produce.add_argument('--request', type=Path, required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    prepare.add_argument('--run', type=Path, required=True)
    prepare.add_argument('--output-dir', type=Path)
    prepare.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    if args.command == 'produce':
        return produce_request(args.request)
    from design_reconstruction import plan_for_run
    run = read(args.run)
    result = prepare_effects(args.root, run, plan_for_run(run), generate=args.generate, output_dir=args.output_dir)
    path = Path(args.output_dir or args.root/'outputs/inspection/independent-effects')/'effect-preparation.json'
    write(path, result)
    print(json.dumps({'ok': True, 'status': result['status'], 'requests': [x['request_path'] for x in result['requests']], 'path': str(path.resolve())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
