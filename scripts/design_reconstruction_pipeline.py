"""Prepare source-native foreground, separate design fields and scene assets."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter

from local_photo_analysis import (_worker, local_runtime_paths, prepare_local_appearance,
                                  analysis_identity, file_identity)
from raw_support import open_rgb, RGB_DECODE_VERSION
from source_alpha import source_silhouette
from runtime_resources import code_path
from source_appearance_responsibility import prepare as prepare_source_responsibility

RECIPE = 'native-source-design-v32-independent-source-semantics'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def _current_plan_cache(plan, *, expected_source_sha256=None, expected_candidate_sha256=None):
    """Accept only a complete current-schema plan for cache reuse.

    Historical images and provenance remain read-only evidence. Production
    preparation never infers missing fields or executes an older contract.
    """
    if not isinstance(plan, dict) or plan.get('recipe') != RECIPE:
        return False
    if not _current_layer_graph(plan):
        return False
    from source_semantic_binding import load_source_semantics
    try:
        load_source_semantics((plan.get('mask_roles') or {}).get('source_semantics'),
            source_sha256=plan['source_sha256'], native_size=plan['canvas']['source_oriented_size'],
            proxy_size=plan['canvas']['source_proxy_size'])
    except (OSError, ValueError, TypeError, KeyError):
        return False
    from scene_design import SCHEMA as SCENE_DESIGN_SCHEMA, VERSION as SCENE_DESIGN_VERSION
    scene_design = plan.get('scene_design') or {}
    if (scene_design.get('schema_version') != SCENE_DESIGN_SCHEMA
            or scene_design.get('intent_provenance', {}).get('normalization_recipe') != SCENE_DESIGN_VERSION
            or not scene_design.get('light', {}).get('environment_asset_bindings')):
        return False
    environment_bindings=scene_design['light']['environment_asset_bindings']
    background_layer=next((layer for layer in plan.get('layers', [])
                           if layer.get('kind')=='background_design'), {})
    background_assets=background_layer.get('assets', {})
    expected_environment={'background':background_assets.get('background'),
        'validity':background_assets.get('validity'),
        'candidate_removal':plan.get('mask_roles', {}).get('candidate_removal')}
    if any(not isinstance(asset,dict) or not asset.get('sha256')
           or environment_bindings.get(name,{}).get('sha256')!=asset['sha256']
           for name,asset in expected_environment.items()):
        return False
    from scene_receiver_intent import validate_bound_receiver_intent
    try:
        receiver = validate_bound_receiver_intent(scene_design.get('authored_receiver_intent'))
        binding = receiver['binding']
        if (binding['source_canvas_size'] != plan['canvas']['source_oriented_size']
                or binding['decode_version'] != plan['source_baseline']['decode_version']
                or binding['source_raw']['sha256'] != plan['source_sha256']
                or binding['reference']['sha256'] != plan['candidate_sha256']
                or binding['observation_assets']['foreground_alpha']['sha256'] != plan['mask_roles']['foreground_alpha']['sha256']
                or any(binding['environment_assets'].get(name, {}).get('sha256') != asset['sha256']
                       for name, asset in expected_environment.items())):
            return False
    except (ValueError, TypeError, KeyError):
        return False
    provenance = plan.get('provenance') if isinstance(plan.get('provenance'), dict) else {}
    contract = provenance.get('candidate_floor_contract')
    required_contract = {'analysis_status', 'clean_established',
                         'automatic_cleanup_applied', 'allowed_interpretation'}
    if not isinstance(contract, dict) or not required_contract.issubset(contract):
        return False
    return (not expected_source_sha256 or plan.get('source_sha256')==expected_source_sha256) and (
        not expected_candidate_sha256 or plan.get('candidate_sha256')==expected_candidate_sha256)


def _current_layer_graph(plan):
    from design_reconstruction import validate_native_plan_layers
    try:
        validate_native_plan_layers(plan)
        return True
    except ValueError:
        return False


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _background_stage_identity(root, input_path, removal_path, python, *, registry_path=None):
    """Content identity for the self-contained candidate-erasure worker."""
    from current_model_registry import load_registry
    checkpoint = Path(load_registry(root, registry_path=registry_path)['lama_checkpoint'])
    worker = code_path('local_background_worker.py')
    site_packages = Path(python).parent.parent / 'Lib/site-packages'
    runtime_packages = {
        name: {path.parent.name: file_identity(path)
               for path in sorted(site_packages.glob(name + '-*.dist-info/METADATA'))}
        for name in ('numpy', 'torch', 'pillow')
    }
    return {
        'schema_version': 'design_background_request.v1',
        'input_sha256': _hash(input_path), 'removal_mask_sha256': _hash(removal_path),
        'checkpoint_sha256': file_identity(checkpoint), 'worker_sha256': file_identity(worker),
        'worker_code_helpers': {},  # The worker imports no repository helpers.
        'runtime_python_sha256': file_identity(python),
        'runtime_package_metadata': runtime_packages,
        'parameters': {'max_edge': 1536, 'device': 'cuda:0', 'precision': 'float32'},
        'output_contract': 'preserve_candidate_RGB_exactly_outside_supplied_mask',
    }


def _background_stage_result(directory, identity):
    """Read only a complete content-bound background stage, never a folder guess."""
    binding_path = Path(directory) / 'binding.json'
    if not binding_path.is_file():
        return None
    try:
        binding = _read(binding_path)
        if (binding.get('schema_version') != 'design_background_stage_cache.v1'
                or binding.get('request_identity') != identity
                or binding.get('successful_worker_returncode') != 0):
            return None
        records = [binding.get('result_manifest', {})] + list((binding.get('outputs') or {}).values())
        if len(records) != 4 or any(not Path(item.get('path', '')).is_file()
                                   or _hash(item['path']) != item.get('sha256') for item in records):
            return None
        result = _read(binding['result_manifest']['path'])
        if result.get('provenance', {}).get('checkpoint_sha256') != identity['checkpoint_sha256']:
            return None
        if any(result.get(key) != binding['outputs'][key]['path'] for key in ('clean_png', 'clean_npz')):
            return None
        return {**result, 'stage_cache': {'cache_hit': True,
                'directory': str(Path(directory).resolve()), 'binding': str(binding_path.resolve())}}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _prepare_background_stage(root, input_path, removal_path, python, log_path, *, registry_path=None):
    identity = _background_stage_identity(root, input_path, removal_path, python, registry_path=registry_path)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(root) / 'outputs/cache/design-background-stage' / key
    directory.mkdir(parents=True, exist_ok=True)
    cached = _background_stage_result(directory, identity)
    if cached is not None:
        return cached
    _worker([python, '-X', 'utf8', code_path('local_background_worker.py'), '--input', input_path,
        '--removal-mask', removal_path, '--output-dir', directory,
        '--checkpoint', Path(local_runtime_paths(root, registry_path=registry_path)['model_registry']['lama_checkpoint'])], log_path)
    result_path = directory / 'result.json'
    result = _read(result_path)
    outputs = {key: {'path': str(Path(result[key]).resolve()), 'sha256': _hash(result[key])}
               for key in ('clean_png', 'clean_npz')}
    mask = directory / 'removal-mask.png'
    outputs['removal_mask'] = {'path': str(mask.resolve()), 'sha256': _hash(mask)}
    _write(directory / 'binding.json', {
        'schema_version': 'design_background_stage_cache.v1', 'request_identity': identity,
        'successful_worker_returncode': 0,
        'result_manifest': {'path': str(result_path.resolve()), 'sha256': _hash(result_path)},
        'outputs': outputs,
    })
    return {**result, 'stage_cache': {'cache_hit': False, 'directory': str(directory.resolve()),
                                     'binding': str((directory / 'binding.json').resolve())}}


def _asset(path, *, size, coordinate_space, color_space='none', representation='scalar',
           radiometric_origin='not_radiometric'):
    return {'path': str(Path(path).resolve()), 'sha256': _hash(path), 'size': list(size),
        'native_size': list(size), 'coordinate_space': coordinate_space,
        'dtype': 'float32' if Path(path).suffix == '.npz' else 'json' if Path(path).suffix=='.json' else 'uint8',
        'color_space': color_space, 'representation': representation,
        'radiometric_origin': radiometric_origin, 'sampling': 'bilinear'}


def ready(root, *, registry_path=None):
    from local_photo_analysis import local_models_ready
    from local_scene_analysis import local_scene_ready
    root = Path(root)
    runtime = local_runtime_paths(root, registry_path=registry_path)
    config = runtime['model_registry']
    required = [runtime['vitmatte_model_root']/name for name in
                ('model.safetensors', 'config.json', 'preprocessor_config.json')]
    required += [Path(config[key]) for key in
        ('birefnet_model_path', 'lama_checkpoint', 'background_sr_checkpoint', 'face_model_path', 'face_python')]
    return (local_models_ready(root, registry_path=registry_path)
            and local_scene_ready(root, registry_path=registry_path)
            and all(path.is_file() for path in required)
            and Path(config['onnx_runtime_root']).is_dir())


def production_code_files():
    return ('reference_pipeline.py',
     'reference_appearance_prepare.py',
     'reference_appearance_stack.py',
     'reference_appearance_controls.py',
     'reference_support.py',
     'reference_colour_fit.py',
     'colour_space.py',
     'evaluate_reference_colour.py',
     'reference_colour_blend.py',
     'reference_scene_balance.py',
     'render_reference_colour_composite.py',
     'background_superresolution_worker.py',
     'local_background_worker.py',
     'source_alpha.py',
     'source_foreground.py',
     'source_foreground_cache.py',
     'sapiens_model_profiles.py',
     'source_semantic_binding.py',
     'local_face_protection.py',
     'local_face_protection_worker.py',
     'local_scene_analysis.py',
     'local_semantics_worker.py',
     'local_photo_analysis.py',
     'design_scene_assets.py',
     'design_reconstruction_pipeline.py',
     'design_reconstruction.py',
     'foreground_effects.py',
     'float_appearance.py',
     'raw_support.py',
     'clone_retouch.py',
     'tone_curves.py',
     'hsl_adjust.py',
     'local_colour_brush.py',
     'native_source_baseline.py',
     'automatic_native_retouch.py',
     'native_retouch_producer.py',
     'automatic_subject_detail_completion.py',
     'native_subject_detail.py',
     'native_subject_detail_runtime.py',
     'automatic_material_completion.py',
     'automatic_scene_completion.py',
     'scene_completion_worker.py',
     'scene_stage_audit.py',
     'background_consistency.py',
     'automatic_foreground_effects.py',
     'automatic_foreground_effects_review.py',
     'foreground_effects_runtime.py',
     'effect_conditioning.py',
     'effect_layout.py',
     'effect_code_identity.py',
     'effect_sampling.py',
     'installed_resources.py',
     'system_post_completion.py',
     'system_model_choices.py',
     'cosplay_post_completion.py',
     'cosplay_post_review.py',
     'cosplay_native_review.py',
     'sam3_scene_worker.py',
     'sam31_model_profile.py',
     'sam31_static_model.py',
     'local_floor_pollution.py',
     'sam3_floor_worker.py',
     'scene_design.py',
     'scene_receiver_intent.py',
     'authored_receiver_shadow.py',
     'photographic_environment.py',
     'native_sole_contact.py',
     'native_receiver_preservation.py',
     'display_transform.py',
     'native_environment_completion.py',
     'native_matte.py',
     'native_matte_worker.py',
     'native_matte_dependencies.py',
     'runtime_resources.py',
     'source_appearance_responsibility.py',
     'image_helpers.py')


def production_code_report():
    from runtime_resources import resource_report
    return resource_report(production_code_files())


def _prepare_registered_environment(root, run, plan):
    """Own the design P/V and its origin without importing source scenery."""
    from native_environment_completion import (
        prepare_native_environment_completion, apply_environment_completion)
    background = next(row for row in plan['layers'] if row['kind'] == 'background_design')['assets']
    identity = {'source': run['source']['sha256'], 'reference': run['reference']['sha256'],
        'background': background['background']['sha256'], 'validity': background['validity']['sha256'],
        'alpha': plan['mask_roles']['foreground_alpha']['sha256'], 'canvas': plan['canvas'],
        'producer': _hash(code_path('native_environment_completion.py'))}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(root)/'outputs/cache/native-environment-completion'/key
    marker = directory/'completion.json'
    if marker.is_file():
        completion = _read(marker)
    else:
        completion = prepare_native_environment_completion(run, plan, directory)
        _write(marker, completion)
    result = apply_environment_completion(plan, completion)
    from scene_design import bind_scene_environment
    assets = next(row for row in result['layers'] if row['kind'] == 'background_design')['assets']
    result['scene_design'] = bind_scene_environment(result['scene_design'], {
        'background': assets['background'], 'validity': assets['validity'],
        'candidate_removal': result['mask_roles']['candidate_removal']})
    return result


def identity_for_run(root, run):
    import os
    from scene_design import scene_intent_identity
    from scene_receiver_intent import receiver_intent_identity
    from authored_receiver_shadow import normalize_receiver_shadow_plan
    root = Path(root)
    registry_path = (run.get('options') or {}).get('model_registry')
    runtime = local_runtime_paths(root, registry_path=registry_path)
    candidate = Path(run['reference']['path'])
    candidate = candidate if candidate.is_absolute() else root/candidate
    from local_scene_analysis import scene_model_identity
    config = runtime['model_registry']
    matting_model = Path(os.environ.get('COSPLAY_ALPHA_MODEL') or config['birefnet_model_path'])
    if not matting_model.is_absolute():
        matting_model = root/matting_model
    model_paths = {'saliency': matting_model, 'lama':Path(config['lama_checkpoint']),
        'face':Path(config['face_model_path']),
        'vitmatte':runtime['vitmatte_model_root']/'model.safetensors',
        'background_sr':Path(config['background_sr_checkpoint'])}
    return {'source': run['source']['sha256'], 'candidate': _hash(candidate), 'recipe': RECIPE,
        'native_source_retouch': __import__('native_source_baseline').retouch_identity(run,root=root),
        'local_model_registry': _model_registry_receipt(runtime),
        'scene_intent': scene_intent_identity(run.get('generation_design') or {}, run['reference']),
        'receiver_intent': receiver_intent_identity(
            ((run.get('options') or {}).get('appearance_revision') or {}).get('receiver_intent'),
            source_canvas_size=run['source']['oriented_size']),
        'receiver_shadow_plan': normalize_receiver_shadow_plan(
            ((run.get('options') or {}).get('appearance_revision') or {}).get('receiver_shadow_plan')),
        'decode': RGB_DECODE_VERSION, 'analysis': analysis_identity(root, registry_path=registry_path),
        'candidate_analysis': analysis_identity(root, role='candidate', registry_path=registry_path),
        'models': {name:file_identity(path) for name,path in model_paths.items()},
        'scene_models':scene_model_identity(config['scene_model_root'], profile=config['scene_profile']),
        'operators': {name:file_identity(code_path(name)) for name in production_code_files()},
        'parameters': {'candidate_mask_dilation_short_side': .012,
            'colour_match_strategy': 'original_reference_colour_match_then_composite',
            'mapping': 'background_feature_homography_with_validity_or_center_cover',
            'foreground': 'native_vitmatte_pymatting_paired',
            'extra_background_optics_default': 'disabled_baked_plate',
            'background_native_detail':'realesrgan_compact_residual_outside_removal_and_reflection',
            'floor_reflection':'preserve_qualified_paired_observations_without_source_RGB_donor'}}


def prepare(root, run, *, rebuild_subject=False):
    """Production preparation follows the same reference pipeline as the UI."""
    from reference_pipeline import prepare as prepare_reference_pipeline
    return prepare_reference_pipeline(root, run, rebuild=rebuild_subject)


def _model_registry_receipt(runtime):
    registry = runtime['registry_path'].resolve()
    return {'path': str(registry), 'sha256': file_identity(registry) if registry.is_file() else None,
            'semantics_profile': runtime['semantics_profile'], 'matting_profile': runtime['matting_profile'],
            'scene_profile':runtime['model_registry']['scene_profile'],
            'dinov3_profile':runtime['model_registry']['dinov3_profile']}


def _same_model_registry(previous, current):
    if not isinstance(previous, dict):
        return False
    return all(previous.get(key) == current[key]
               for key in ('sha256', 'semantics_profile', 'matting_profile', 'scene_profile', 'dinov3_profile'))


def prepare_scene(root, run, *, rebuild_subject=False):
    from native_source_baseline import retouch_identity, retouch_baseline_record
    root = Path(root)
    registry_path = (run.get('options') or {}).get('model_registry')
    registry_receipt = _model_registry_receipt(local_runtime_paths(root, registry_path=registry_path))
    started = time.perf_counter()
    previous = ((run.get('options') or {}).get('appearance_revision') or {}).get('design_reconstruction_plan')
    source, candidate = Path(run['source']['path']), Path(run['reference']['path'])
    source = source if source.is_absolute() else root/source
    candidate = candidate if candidate.is_absolute() else root/candidate
    if not ready(root, registry_path=registry_path):
        raise RuntimeError('Native design reconstruction requires installed local models; no GPT request was sent')
    generation_identity = identity_for_run(root, run)
    source_hash, candidate_hash = generation_identity['source'], generation_identity['candidate']
    identity = hashlib.sha256(json.dumps(generation_identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = root/'outputs/cache/design-reconstruction'/identity
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory/'plan.json'
    if plan_path.is_file():
        # Reuse only a complete current-schema plan.  Older cache entries are
        # deliberately rejected and rebuilt through the current producer;
        # renderer-side compatibility inference would hide stale contracts.
        cached = _read(plan_path)
        if _current_plan_cache(cached, expected_source_sha256=source_hash,
                               expected_candidate_sha256=candidate_hash):
            # Receiver authoring is a CPU edit over frozen native assets. A
            # reusable reconstruction must also refresh that edit: its small
            # disposable shadow cache may have been removed independently.
            from scene_design import bind_plan_receiver_intent
            from authored_receiver_shadow import prepare_receiver_shadow
            from design_scene_assets import prepare_native_receivers
            cached = prepare_native_receivers(root, run, cached)
            cached['source_appearance_responsibility'] = prepare_source_responsibility(run, cached)
            from foreground_effects import prepare_effects
            effects=prepare_effects(root,run,cached)
            cached=effects['plan']
            _write(plan_path, cached)
            return {'ok': True, 'design_reconstruction_plan': cached, 'cache_hit': True,
                    'independent_effect_requests':effects['requests'],
                    'elapsed_seconds': round(time.perf_counter()-started, 3)}
    source_analysis = prepare_local_appearance(root, source, source_hash=source_hash, registry_path=registry_path)
    candidate_analysis = prepare_local_appearance(root, candidate, source_hash=candidate_hash,
                                                 role='candidate', registry_path=registry_path)
    source_png = Path(source_analysis['analysis_directory'])/'source-srgb.png'
    candidate_png = Path(candidate_analysis['analysis_directory'])/'source-srgb.png'
    source_image = Image.open(source_png).convert('RGB')
    candidate_image = open_rgb(candidate)
    candidate_image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
    candidate_png = directory/'candidate-srgb.png'
    candidate_image.save(candidate_png)
    sa, source_saliency = source_silhouette(root, source, source_image, source_hash=source_hash, registry_path=registry_path)
    ca, candidate_saliency = source_silhouette(root, candidate, candidate_image, source_hash=candidate_hash, registry_path=registry_path)
    saliency_path, candidate_alpha_path = directory/'source-saliency-seed.png', directory/'candidate-occupancy.png'
    sa.save(saliency_path); ca.save(candidate_alpha_path)
    from source_foreground_cache import prepare_source_foreground
    foreground = prepare_source_foreground(root, source, source_hash=source_hash,
        saliency_path=saliency_path, semantics_path=source_analysis['semantic_result']['labels_path'],
        registry_path=registry_path,native_retouch_binding=(run.get('options') or {}).get('native_source_retouch'))
    source_precision = foreground['provenance']['source_precision']['source_precision']
    source_origin = ('LibRaw16_source' if source_precision == 'raw16_linear'
                     else 'inverse_sRGB_source_raster8')
    source_size = foreground['native_source_size']
    source_proxy_size = list(source_image.size)
    from local_scene_analysis import prepare_local_scene
    scene = prepare_local_scene(root, source, candidate, source_hash=source_hash, candidate_hash=candidate_hash,
        source_subject_mask=source_analysis['semantic_result']['labels_path'],
        candidate_subject_mask=candidate_analysis['semantic_result']['labels_path'], registry_path=registry_path)
    # The captured source reflection mixes old floor and lighting radiance.
    # It produced gray blocks even after material rebinding. New projects do
    # not prepare this rejected donor; historical images remain read-only evidence.
    from design_scene_assets import removal_inputs, prepare_design_scene_assets
    extras = removal_inputs(scene, candidate_image.size)
    from local_floor_pollution import prepare_candidate_floor_pollution, protect_verified_reflections
    floor_pollution=prepare_candidate_floor_pollution(root,scene,registry_path=registry_path)
    # A local cleanup result is deliberately conservative: unsupported and
    # empty searches, as well as partial fragment removal, never establish a
    # globally clean candidate floor.  Preserve the state in the immutable
    # plan so the UI cannot present an absent mask as a success signal.
    floor_pollution_state = str(floor_pollution.get('analysis_status') or
                                ('partial' if floor_pollution.get('mask') else 'unsupported'))
    h, w = candidate_image.height, candidate_image.width
    occupancy = np.asarray(ca) > 10
    removal = binary_dilation(occupancy, iterations=max(3, round(min(w, h)*.012)))
    removal |= extras['extra_removal_mask']
    if floor_pollution.get('mask'):
        floor_mask=Image.open(floor_pollution['mask']['path']).convert('L').resize(candidate_image.size,Image.Resampling.NEAREST)
        removal |= np.asarray(floor_mask)>0
    # Main saliency can include the reflected legs as well. Protection must
    # therefore follow every union, rather than just filtering the floor mask.
    removal,reflection_preservation=protect_verified_reflections(removal,floor_pollution,candidate_image.size,
        source_sha256=source_hash,candidate_sha256=candidate_hash)
    removal_path = directory/'candidate-removal-mask.png'
    Image.fromarray(np.uint8(removal)*255).save(removal_path)
    paths = local_runtime_paths(root, registry_path=registry_path)
    background_result = _prepare_background_stage(root, candidate_png, removal_path, paths['python'],
                                                 directory/'background-worker.log', registry_path=registry_path)
    from float_appearance import srgb_to_linear
    clean = srgb_to_linear(np.asarray(Image.open(background_result['clean_png']).convert('RGB'), np.float32)/255)
    scene_assets = prepare_design_scene_assets(scene, clean, source_image.size, directory/'scene-assets',
        source_foreground_alpha=foreground['alpha_path'],source_semantics={
            'path':source_analysis['semantic_result']['labels_path'],
            'sha256':_hash(source_analysis['semantic_result']['labels_path']),
            **{key:source_analysis['semantic_result'].get(key) for key in (
                'model', 'model_profile', 'model_revision', 'checkpoint_sha256', 'config_sha256', 'taxonomy_sha256')}})
    from scene_design import build_scene_design
    scene_design=build_scene_design(scene,run.get('generation_design') or {},analysis_grid=source_image.size,
        reference=run['reference'], environment_assets={
            'background':scene_assets['background'], 'validity':scene_assets['background_validity'],
            'candidate_removal':_asset(removal_path,size=candidate_image.size,
                coordinate_space='candidate_proxy',representation='pollution_removal')})
    scene_assets['background'].update(radiometric_origin='inverse_sRGB_generated_display',
        color_primaries='sRGB', transfer_function='linear')
    from local_face_protection import prepare_face_protection
    source_face = prepare_face_protection(root, source_png, source_analysis['semantic_result'], registry_path=registry_path)
    masks = {
        'foreground_alpha': _asset(foreground['alpha_path'], size=source_size, coordinate_space='source_canvas', representation='coverage'),
        'candidate_occupancy': _asset(candidate_alpha_path, size=candidate_image.size, coordinate_space='candidate_proxy', representation='segmentation'),
        'candidate_removal': _asset(removal_path, size=candidate_image.size, coordinate_space='candidate_proxy', representation='pollution_removal'),
        'edge_applicability': _asset(foreground['unknown_edge_support_path'], size=source_size, coordinate_space='source_canvas'),
        'identity_protection': _asset(source_face['field_path'], size=source_face['size'], coordinate_space='source_proxy', representation='eyes_lips_protection')}
    from source_semantic_binding import bind_source_semantics
    masks['source_semantics'] = bind_source_semantics(source_analysis['semantic_result'],
        source_sha256=source_hash, native_size=source_size, proxy_size=source_proxy_size)
    if floor_pollution.get('reflection_protection'):
        masks['candidate_reflection_protection']=dict(floor_pollution['reflection_protection'])
        source_observation=floor_pollution.get('source_observation') or {}
        if source_observation.get('detected_fragments_path'):
            masks['source_observed_floor_fragments']=_asset(source_observation['detected_fragments_path'],
                size=source_observation['size'],coordinate_space='source_scene_proxy',representation='observed_reflection_fragment')
    with np.load(source_analysis['semantic_result']['labels_path']) as data:
        labels = np.asarray(Image.fromarray(data['labels']).resize(source_image.size, Image.Resampling.NEAREST))
    source_coverage = np.asarray(sa, np.float32)/255
    role_arrays = {'hair': labels == 4, 'skin': np.isin(labels, [3,5,6,7,8,11,12,14,15,16,17,20,21,22]),
        'garment': np.isin(labels, [1,9,10,13,18,19,23])}
    for name, values in role_arrays.items():
        path = directory/f'source-{name}.png'
        Image.fromarray(np.uint8(values)*255).save(path)
        masks[name] = _asset(path, size=source_image.size, coordinate_space='source_proxy', representation='semantic_membership')
    # Optional 2D rim observation, never a claim of a recovered 3D lamp. Ground
    # and reflection receivers are excluded from the bright-region observation.
    y, x = np.mgrid[0:h, 0:w]
    luma = clean@np.asarray([.2126,.7152,.0722], np.float32)
    floor = (scene.get('contact',{}).get('candidate',{}).get('support_surface') or {}).get('path')
    available = ~removal
    if floor:
        available &= np.asarray(Image.open(floor).convert('L').resize(candidate_image.size, Image.Resampling.NEAREST)) == 0
    weights = np.maximum(luma-np.quantile(luma[available], .96), 0)*available if available.any() else np.zeros_like(luma)
    light_xy_candidate = np.array([(weights*x).sum()/max(weights.sum(),1e-8), (weights*y).sum()/max(weights.sum(),1e-8), 1.])
    matrix = np.asarray(scene_assets['mapping']['candidate_processing_proxy_to_final_proxy'])
    light_point = matrix@light_xy_candidate
    light_xy = (light_point[:2]/max(light_point[2],1e-8)/np.asarray(source_image.size)).tolist()
    color = (clean*weights[...,None]).sum((0,1))/max(weights.sum(),1e-8)
    color /= max(float(color.max()), .01)
    gy, gx = np.gradient(gaussian_filter(source_coverage, 2))
    sy, sx = np.mgrid[0:source_image.height, 0:source_image.width]
    dx, dy = light_xy[0]-sx/source_image.width, light_xy[1]-sy/source_image.height
    norm = np.maximum(np.sqrt(dx*dx+dy*dy)*np.sqrt(gx*gx+gy*gy), .0001)
    rim = np.maximum(source_coverage-gaussian_filter(source_coverage, max(2,min(source_image.size)*.005)),0)
    rim *= np.clip((-gx*dx-gy*dy)/norm,0,1)
    rim_path = directory/'optional-rim-emission.npz'
    np.savez_compressed(rim_path, values=(rim[...,None]*color*.35).astype(np.float32))
    fog_path = directory/'optional-atmosphere.npz'
    sh, sw = source_image.height, source_image.width
    density = np.broadcast_to(np.linspace(.22,0,sh,dtype=np.float32)[:,None],(sh,sw)).copy()
    fog_color = clean.mean((0,1)); fog_color /= max(float(fog_color.max()), .01)
    np.savez_compressed(fog_path, values=np.broadcast_to(fog_color,(sh,sw,3)).copy(),density=density)
    def layer(kind, label, assets, *, strength=1, enabled=True, params=None, blend='normal', stage=None):
        return {'id':kind, 'kind':kind, 'label':label, 'enabled':enabled, 'strength':strength,
            'blend_mode':blend, 'stage':stage, 'params':params or {}, 'assets':assets}
    layers = [
        layer('background_design','环境设计',{
            'background':scene_assets['background'], 'validity':scene_assets['background_validity'],
            'foreground_premultiplied':_asset(foreground['premultiplied_foreground_path'],size=source_size,
                coordinate_space='source_canvas',color_space='linear_sRGB',representation='premultiplied_RGB',radiometric_origin=source_origin),
            'old_environment':_asset(foreground['old_background_path'],size=source_size,
                coordinate_space='source_canvas',color_space='linear_sRGB',representation='unassociated_RGB',radiometric_origin='source_FB_estimate')},stage='environment')]
    layers.extend(scene_assets['layers'])
    layers.extend([
        layer('rim_light','轮廓光增量（二维近似）',{'emission':_asset(rim_path,size=source_image.size,
            coordinate_space='source_proxy',color_space='linear_sRGB',representation='surface_emission')},enabled=False,blend='add',
            params={'light_xy':light_xy,'method':'2D_bright_region','baked_effect_policy':'user_increment_only'},stage='foreground_surface'),
        layer('atmosphere','空间氛围增量',{'fog':_asset(fog_path,size=source_image.size,coordinate_space='source_proxy',color_space='linear_sRGB')},
            strength=.2,enabled=False,params={'near_fraction':.08,'depth_method':'vertical_proxy','baked_effect_policy':'user_increment_only'},stage='space'),
        layer('bloom','亮区柔光增量',{},strength=.12,enabled=False,blend='add',
            params={'source':'visible_composite','threshold_linear':.45,'radius_fraction':.015,'baked_effect_policy':'user_increment_only'},stage='optics')])
    from display_transform import source_transform_for_precision
    source_display=source_transform_for_precision(source_precision)
    plan = {'schema_version':'design_reconstruction_plan.v1','enabled':True,
        'working_domain':source_display['input_domain'],
        'summary':'先用参考图对原图人物仿色，再与环境溶图；本人、妆容与纹理来自原图。',
        'source_sha256':source_hash,'candidate_sha256':candidate_hash,'recipe':RECIPE,
        'canvas':{'source_oriented_size':source_size,'source_proxy_size':source_proxy_size,
            'foreground_coordinate_space':'source_canvas','candidate_mapping':scene_assets['mapping']},
        'source_baseline':{'schema':'source_preparation_baseline.v1',
            'working_domain':source_display['input_domain'],'source_display':source_display,
            'source_display_applied_once':False,'source_sha256':source_hash,'decode_version':RGB_DECODE_VERSION,
            'native_source_retouch':__import__('native_source_baseline').retouch_baseline_record(run),
            'source_precision':source_precision,'radiometric_origin':source_origin,
            'foreground_bundle_identity':foreground['identity'],
            'assets':{'foreground_premultiplied':dict(layers[0]['assets']['foreground_premultiplied']),
                'foreground_alpha':dict(masks['foreground_alpha']),
                'old_environment':dict(layers[0]['assets']['old_environment'])}},
        'generation_identity':generation_identity,'layers':layers,'mask_roles':masks,'scene_design':scene_design,
        'execution':{'stages':'fixed_foreground_environment_receiver_composite_optics',
            'background_coverage':'derived_one_minus_native_alpha','foreground_base':'paired_native_Fp',
            'automatic_subject_lighting':'original_reference_colour_match_then_composite',
            'skin_appearance_baseline':'source_detail_design_gain_pre_manual',
            'source_reference':'paired_native_baseline','receiver_context':'current_environment'},
        'effect_ownership':{'candidate_plate':'observed_baked_or_unknown',
            'person_colour_match':'reference_colour_statistics_applied_to_original_pixels',
            'extra_background_fog':'disabled','extra_bloom':'disabled','rim':'disabled_explicit_increment'},
        'provenance':{'local_model_registry':registry_receipt,
            'source_foreground':foreground,'saliency_seed':source_saliency,'candidate_occupancy':candidate_saliency,
            'scene':scene_assets['provenance'],'background':background_result['provenance'],
            'background_stage_cache':background_result['stage_cache'],
            'floor_pollution':floor_pollution,
            'candidate_floor_contract':{'analysis_status':floor_pollution_state,
                'clean_established':False,
                'automatic_cleanup_applied':floor_pollution_state=='partial',
                'allowed_interpretation':'qualified_paired_reflections_preserved; unmatched_local_fragment_cleanup_only; empty_or_unsupported_is_not_clean',
                'reflection_ownership':floor_pollution.get('reflection_ownership'),
                'removal_preservation':reflection_preservation},
            'contamination':{key:value for key,value in extras.items() if key != 'extra_removal_mask'},
            'gpt_image_calls':0,'candidate_reused':True},
        'limitations':foreground.get('limitations',[])+scene_assets.get('limitations',[])+[
            'Reference colour matching retains the original person geometry and texture; visual quality needs review.',
            f'Candidate floor cleanup state is {floor_pollution_state}; detected local fragments do not establish a fully clean floor.',
            'Background detail enhancement synthesizes detail absent from the design reference.',
            'Transparent edges and subject/environment integration require visual inspection.']}
    # Native environment detail is part of new-project preparation, not a
    # separate manual experiment the user must run after generating a design.
    background_run={**run,'options':{**run.get('options',{}),'appearance_revision':{
        **(run.get('options',{}).get('appearance_revision') or {}),'design_reconstruction_plan':plan}}}
    background_input=directory/'background-sr-input.json'
    _write(background_input,background_run)
    background_sr_dir=directory/'background-native-detail'
    _worker([paths['python'],'-X','utf8',code_path('background_superresolution_worker.py'),
        '--run',background_input,'--weights',Path(paths['model_registry']['background_sr_checkpoint']),
        '--output',background_sr_dir],directory/'background-sr.log')
    sr=_read(background_sr_dir/'manifest.json')
    background_layer=next(layer for layer in plan['layers'] if layer['kind']=='background_design')
    background_layer['assets'].update(sr['assets'])
    # Frozen compact observations authorize later CPU data-consistency repair;
    # they are neither native detail nor permission to paste candidate people.
    background_layer['assets']['compact_observation']=copy.deepcopy(sr['source_assets']['background'])
    background_layer['assets']['compact_validity']=copy.deepcopy(sr['source_assets']['validity'])
    observation_path=directory/'background-observation.json'
    observation_binding=plan['provenance']['background_stage_cache']['binding']
    _write(observation_path,{'schema':'background_observation.v1','source_sha256':source_hash,
        'reference_sha256':candidate_hash,'sr_provenance':copy.deepcopy(sr),
        'clean_stage_binding':{'path':str(Path(observation_binding).resolve()),'sha256':_hash(observation_binding)},
        'clean_input_asset':{'path':str(candidate_png.resolve()),'sha256':_hash(candidate_png)}})
    background_layer['assets']['background_observation']={
        'path':str(observation_path.resolve()),'sha256':_hash(observation_path),
        'representation':'frozen_background_observation_receipt'}
    plan['provenance']['background_native_detail']=sr
    # The completed design owns environment appearance. Source silhouette
    # topology cannot authorize importing a different source background.
    plan = _prepare_registered_environment(root, run, plan)
    # Bind the final owned plate after SR and environment registration, not only
    # the intermediate plate used during scene construction.
    from scene_design import bind_scene_environment, bind_plan_receiver_intent
    background_assets=next(layer for layer in plan['layers']
                           if layer.get('kind')=='background_design')['assets']
    environment_bindings={'background':background_assets['background'],
        'validity':background_assets['validity'],
        'candidate_removal':plan['mask_roles']['candidate_removal']}
    plan['scene_design']=bind_scene_environment(plan['scene_design'],environment_bindings)
    from design_scene_assets import prepare_native_receivers
    plan=prepare_native_receivers(root,run,plan)
    plan['source_appearance_responsibility'] = prepare_source_responsibility(run, plan)
    from foreground_effects import prepare_effects
    effects = prepare_effects(root, run, plan)
    plan = effects['plan']
    _write(plan_path,plan)
    return {'ok':True,'design_reconstruction_plan':plan,'cache_hit':False,
        'independent_effect_requests':effects['requests'],
        'elapsed_seconds':round(time.perf_counter()-started,3),'analysis_directory':str(directory.resolve())}

