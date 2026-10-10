"""One local preparation entry point for reference matching and composition.

Preparation may use the model runtime. Reopening, editing and exporting an
owned appearance stack only replay frozen native assets in the desktop runtime.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import sys
from material_solver_profiles import DEFAULT_PROFILE, solver_configuration

RECIPE = 'native-reference-appearance-freeze-v3-contact'
PRODUCERS = ('reference_appearance_prepare.py','reference_colour_fit.py',
             'reference_colour_blend.py','reference_scene_balance.py','render_reference_colour_composite.py')
STAGES = ('fit','colour_fusion','native_baseline','scene_balance','native_target','freeze')
INPUT_CONTRACT = 'reference_pipeline_inputs.v1'
EXTRA_INPUTS = frozenset(('object_masks', 'authored_colour_edit', 'material_targets'))
CODE_DEPENDENCIES = PRODUCERS + ('colour_space.py', 'evaluate_reference_colour.py', 'display_transform.py',
    'raw_support.py', 'float_appearance.py', 'source_foreground.py', 'local_photo_analysis.py',
    'local_semantics_worker.py', 'local_face_protection.py', 'local_face_protection_worker.py',
    'local_colour_brush.py', 'reference_support.py',
    'reference_appearance_controls.py', 'reference_local_correspondence.py',
    'reference_local_correspondence_worker.py', 'reference_adaptive_fusion.py',
    'reference_automatic_layers.py', 'reference_material_palette.py',
    'reference_material_targets.py', 'reference_material_targets_v2.py', 'material_solver_profiles.py',
    'material_anchor_review.py', 'qualified_material_finish.py', 'native_receiver_preservation.py')


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    tmp.replace(path)


def _python(root):
    from current_model_registry import load_registry
    candidate=Path(load_registry(root)['python'])
    if not candidate.is_file():
        raise RuntimeError('Local reference preparation runtime is unavailable; saved versions remain editable')
    return candidate


def _current_alpha(plan):
    return plan.get('mask_roles',{}).get('foreground_alpha',{})


def _resolve(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def _code_identity(root):
    # Numeric freeze changes require RECIPE changes, not packaging file hashes.
    from runtime_resources import code_path
    return {name: _sha(code_path(name)) for name in CODE_DEPENDENCIES
        if name != 'reference_appearance_prepare.py'}


def _code_file(filename):
    from runtime_resources import code_path
    return code_path(filename)




def _feature_identity(root, run):
    from reference_local_correspondence import local_correspondence_identity
    try:
        return {'available': True, 'identity': local_correspondence_identity(root,
            algorithm_profile=run.get('options', {}).get('algorithm_profile', 'dinov3_vitl16'),
            registry_path=run.get('options', {}).get('model_registry'))}
    except (OSError, RuntimeError) as error:
        # Identity records missing dependencies; it never substitutes a model
        # or converts unavailable inference into an automatic success.
        return {'available': False, 'reason': str(error)}


def _registry_identity(root, run):
    from local_photo_analysis import local_runtime_paths
    selected = run.get('options', {}).get('model_registry')
    runtime = local_runtime_paths(root, registry_path=_resolve(root, selected) if selected else None)
    registry = runtime['registry_path']
    if (selected or os.environ.get('COSPLAY_MODEL_REGISTRY')) and not registry.is_file():
        raise RuntimeError('Selected local model registry is missing: '+str(registry))
    return {'sha256': _sha(registry) if registry.is_file() else None,
            'semantics_profile': runtime['semantics_profile'],
            'matting_profile': runtime['matting_profile']}


def _asset_identity(root, asset):
    if _sha(_resolve(root, asset['path'])) != asset['sha256']:
        raise ValueError('Preparation input bytes differ from declared SHA: ' + str(asset['path']))
    return {key: copy.deepcopy(value) for key, value in asset.items()
        if key not in {'path', 'file', 'manifest', 'output'} and not key.endswith(('_path', '_file'))}


def _receipt_identity(root, value):
    """Bind JSON choices and its file closure, never live file locations."""
    if isinstance(value, list):
        return [_receipt_identity(root, item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in {'previews', 'validation', 'run_id', 'manifest', 'output'}:
            continue
        if isinstance(item, str) and (key == 'path' or key.endswith(('_path', '_file'))):
            actual = _sha(_resolve(root, item))
            expected = value.get('sha256') if key == 'path' else value.get(key.rsplit('_', 1)[0] + '_sha256')
            if expected and expected != actual:
                raise ValueError('Photo-specific dependency bytes changed: ' + item)
            result[key] = {'content_sha256': actual}
        else:
            result[key] = _receipt_identity(root, item)
    return result


def _extra_identity(root, run):
    extra = run.get('reference_pipeline_inputs') or {}
    if not isinstance(extra, dict) or set(extra) - EXTRA_INPUTS:
        raise ValueError('Unsupported reference preparation inputs')
    return {key: _receipt_identity(root, _read(_resolve(root, path))) for key, path in extra.items() if path}


def _environment_identity(root, plan):
    assets = _background(plan)
    return {key: _asset_identity(root, assets[key]) for key in ('background', 'validity')}


def _input_identity(root, run):
    """The effective T/B/S inputs, excluding later manual edits/strength."""
    from display_transform import source_transform_for_run
    from native_source_baseline import retouch_identity
    source_retouch = retouch_identity(run, root=root)
    revision = run['options']['appearance_revision']
    plan = revision['design_reconstruction_plan']
    display = source_transform_for_run(run)
    canvas = plan['canvas']
    registry = _registry_identity(root, run)
    environment = _environment_identity(root, plan)
    stack = next((row for row in plan.get('layers', []) if row.get('kind') == 'reference_appearance_stack'), None)
    if stack:
        saved_asset = stack.get('assets', {}).get('preparation_snapshot')
        if saved_asset:
            _asset_identity(root, saved_asset)
            saved = _read(_resolve(root, saved_asset['path']))
            # A repaired background is an output, not the original scene
            # input. Only an unchanged frozen output may retain that input.
            if saved.get('prepared_environment') == environment and saved.get('pipeline_inputs'):
                environment = saved['pipeline_inputs']['environment']
    return {'schema': INPUT_CONTRACT, 'recipe': RECIPE,
        'source': _asset_identity(root, {'path': run['source']['path'], 'sha256': run['source']['sha256']}),
        'reference': _asset_identity(root, {'path': run['reference']['path'], 'sha256': run['reference']['sha256']}),
        'alpha': _asset_identity(root, _current_alpha(plan)),
        'identity_protection': _asset_identity(root, plan['mask_roles']['identity_protection']),
        'canvas': {key: copy.deepcopy(canvas[key]) for key in ('source_oriented_size', 'source_proxy_size')},
        'mapping': copy.deepcopy(canvas['candidate_mapping']['candidate_original_to_final_original']),
        'source_display': display, 'environment': environment, 'extra_inputs': _extra_identity(root, run),
        'scene_options': {'warm_colour': True, 'automatic_local_fields': True},
        'algorithm_profile': run.get('options', {}).get('algorithm_profile', 'dinov3_vitl16'),
        'model_registry_sha256': registry['sha256'],
        'model_registry_profiles': {key: value for key, value in registry.items()
            if key != 'sha256'},
        'local_feature_model': _feature_identity(root, run), 'code': _code_identity(root),
        **({'native_source_retouch': source_retouch} if source_retouch else {})}


def _stamp_inputs(root, result, identity):
    """Stamp newly produced assets only; never rewrite a saved parent."""
    plan = result['design_reconstruction_plan']
    layer = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    asset = layer['assets']['preparation_snapshot']
    snapshot = _read(_resolve(root, asset['path']))
    snapshot['pipeline_inputs'] = copy.deepcopy(identity)
    snapshot['prepared_environment'] = _environment_identity(root, plan)
    _write(_resolve(root, asset['path']), snapshot)
    asset['sha256'] = _sha(_resolve(root, asset['path']))
    layer['params']['provenance_identity_sha256'] = asset['sha256']
    plan.setdefault('provenance', {}).setdefault('reference_pipeline', {})['preparation_snapshot_sha256'] = asset['sha256']
    result['preparation_input_identity_sha256'] = _digest(identity)


def attach_reference_support(root, plan, output):
    """Publish source-bound preparation evidence, from the current prepared pair."""
    from reference_support import build_reference_support
    layer = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    snapshot_asset = layer['assets']['preparation_snapshot']
    _asset_identity(root, snapshot_asset)
    snapshot = _read(_resolve(root, snapshot_asset['path']))
    binding = {key: copy.deepcopy(layer['params'][key]) for key in
        ('source_sha256', 'reference_sha256', 'alpha_sha256', 'source_fp_sha256', 'canvas_size')}
    binding['automatic_delta_sha256'] = layer['assets']['automatic_delta']['sha256']
    report = build_reference_support(snapshot['parameters'], snapshot['colour_layer'],
        snapshot['scene_layer'], binding=binding, donor=snapshot.get('donor_support'))
    path = Path(output) / 'reference-support.json'
    _write(path, report)
    layer['params']['reference_support'] = report
    layer['assets']['reference_support'] = {'path': str(path.resolve()), 'sha256': _sha(path),
        'representation': 'path_independent_reference_support'}
    return plan




def _attach_independent_effects(root, run, result):
    from foreground_effects import prepare_effects
    effects = prepare_effects(root, run, result['design_reconstruction_plan'])
    result = {**result, 'design_reconstruction_plan': effects['plan'],
              'independent_effect_requests': effects['requests']}
    return result


def _prepared_matches(root, run, result, *, check_producer=True, input_identity=None):
    try:
        from reference_appearance_stack import validate_layer
        plan=result['design_reconstruction_plan']
        layer=next(x for x in plan['layers'] if x['kind']=='reference_appearance_stack')
        current=run['options']['appearance_revision']['design_reconstruction_plan']
        if (plan['source_sha256'] != run['source']['sha256'] or plan['candidate_sha256'] != run['reference']['sha256']
                or layer['params']['alpha_sha256'] != _current_alpha(current)['sha256']):
            return False
        validate_layer(layer,plan,run)
        for asset in [*_background(plan).values(), *plan['source_baseline']['assets'].values(),
                      layer['assets']['preparation_snapshot']]:
            if _sha(_resolve(root, asset['path'])) != asset['sha256']:
                return False
        snapshot=_read(_resolve(root, layer['assets']['preparation_snapshot']['path']))
        if snapshot.get('recipe') != RECIPE:
            return False
        current_inputs = input_identity if input_identity is not None else _input_identity(root, run)
        saved_inputs = snapshot.get('pipeline_inputs')
        if not isinstance(saved_inputs, dict):
            return False  # Saved versions replay, but unbound caches do not.
        if not check_producer:
            current_inputs = {key: value for key, value in current_inputs.items() if key != 'code'}
            saved_inputs = {key: value for key, value in saved_inputs.items() if key != 'code'}
        if saved_inputs != current_inputs:
            return False
        # The freeze format has an explicit recipe version. Metadata-only
        # packaging fixes must not refit already validated native pixels.
        # A numeric freeze change requires a new RECIPE; T/B/S code changes
        # separately invalidate preparation through their actual hashes.
        if check_producer and saved_inputs.get('code') != _code_identity(root):
            return False
        return True
    except (OSError,ValueError,KeyError,StopIteration,TypeError):
        return False


def _background(plan):
    return next(x for x in plan['layers'] if x['kind']=='background_design')['assets']


def _diagnostic_ready(root, run, directory, *, input_identity=None):
    """Reuse this exact prepared pair; never inherit photo-specific strokes."""
    try:
        directory=Path(directory)
        current_inputs = input_identity if input_identity is not None else _input_identity(root, run)
        if _read(directory / 'pipeline-inputs.json') != current_inputs:
            return False
        p=_read(directory/'parameters.json')
        receipt=_read(directory/'composite-receipt.json')
        plan=run['options']['appearance_revision']['design_reconstruction_plan']
        if any(p['binding'][role] != run[role]['sha256'] for role in ('source','reference')):
            return False
        return (p['binding']['alpha'] == _current_alpha(plan)['sha256']
            and p['binding']['algorithm_sha256'] == _sha(_code_file('reference_colour_fit.py'))
            and p['binding']['evaluation_sha256'] == _sha(_code_file('evaluate_reference_colour.py'))
            and p['binding']['source_weights_sha256'] == _sha(directory/'source-weights.npz')
            and receipt['parameters_sha256'] == _sha(directory/'parameters.json')
            and receipt['renderer_sha256'] == _sha(_code_file('render_reference_colour_composite.py'))
            and bool(receipt.get('scene_balance'))
            and all((directory/name).is_file() for name in ('reference-colour-layer.json','scene-balance-layer.json')))
    except (OSError,ValueError,KeyError,TypeError):
        return False


def _process(root, command, directory, cancel=None):
    directory.mkdir(parents=True,exist_ok=True)
    log=directory/'preparation.log'
    with log.open('ab') as stream:
        process=subprocess.Popen([str(_python(root)),'-B','-X','utf8',*map(str,command)],
            cwd=str(root),stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            while process.poll() is None:
                if cancel and cancel():
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait(timeout=5)
                    raise InterruptedError('Reference preparation cancelled; saved versions and submitted edits are retained')
                time.sleep(.25)
        finally:
            if process.poll() is None:
                process.kill();process.wait(timeout=5)
        if process.returncode:
            tail=log.read_text(encoding='utf8',errors='replace')[-4000:]
            raise RuntimeError('Local reference preparation failed; retry with the same saved inputs.\n'+tail)


def prepare_material_finish(root, run, inputs_path, *, output=None, algorithm_profile=None,
                            cancel=None, progress=None):
    """Derive reviewed material colours from an owned complete saved endpoint.

    This records inheritance of previous T/B/S execution rather than assigning
    current producer identities to historical pixels. The caller saves a child.
    """
    from foreground_effects import prepare_effects
    root = Path(root).resolve()
    inputs_path = _resolve(root, inputs_path)
    profile = algorithm_profile or run.get('options', {}).get('material_finish_profile', DEFAULT_PROFILE)
    adapter = solver_configuration(profile)
    parent = run['options']['appearance_revision']['design_reconstruction_plan']
    layer = next(row for row in parent['layers'] if row['kind'] == 'reference_appearance_stack')
    identity = {'input_sha256': _sha(inputs_path),
        'parent_delta_sha256': layer['assets']['automatic_delta']['sha256'],
        'source_fp_sha256': layer['params']['source_fp_sha256'],
        'alpha_sha256': layer['params']['alpha_sha256'],
        'producer_sha256': _sha(_code_file('qualified_material_finish.py')),
        'algorithm_profile': profile,
        'application_dependencies_sha256': {name: _sha(_code_file(name)) for name in
            ('reference_colour_blend.py', 'render_reference_colour_composite.py')},
        'solver_dependencies_sha256': {name: _sha(_code_file(name)) for name in adapter['dependencies']}}
    if output is None:
        output = root/'outputs/cache/qualified-material-finish'/_digest(identity)[:20]
    output = Path(output).resolve()
    if Path(sys.executable).resolve() != _python(root).resolve():
        # Desktop Python deliberately does not carry cv2 or the model runtime.
        # Only the CPU preparation worker imports the numeric producer.
        output.mkdir(parents=True, exist_ok=True)
        parent_snapshot = output/'material-finish-parent.json'
        _write(parent_snapshot, run)
        if progress:
            progress('material_colour', '准备原生材质颜色修正。', 0)
        _process(root, [_code_file('qualified_material_finish.py'), '--root', root,
            '--run', parent_snapshot, '--inputs', inputs_path, '--output', output,
            '--profile', profile], output, cancel)
        result = _read(output/'prepared-plan.json')
    else:
        from qualified_material_finish import finish_owned_appearance
        result = finish_owned_appearance(root, run, inputs_path, output,
            algorithm_profile=profile, cancel=cancel, progress=progress)
    effects = prepare_effects(root, run, result['design_reconstruction_plan'])
    result['design_reconstruction_plan'] = effects['plan']
    result.update(independent_effect_requests=effects['requests'], derived_from_saved_target=True,
                  upstream_T_B_S_refitted=False, project_pointer_updated=False)
    return result


def prepare(root, run, *, rebuild=False, cancel=None, progress=None):
    """Prepare a separate, frozen plan; never save a revision or call GPT.

    progress(stage, message, fraction) describes real stages, not a time ETA.
    Retry reuses a validated final cache or repeats computation; it does not
    promise arbitrary mid-stage resume. The durable input and journal survive.
    """
    root=Path(root).resolve(); run=copy.deepcopy(run)
    def report(stage,message,fraction=None):
        if cancel and cancel():
            raise InterruptedError('Reference preparation cancelled')
        if progress: progress(stage,message,fraction)
    report('reference_inputs','检查原片、参考和当前人物边缘。',0)
    plan=(run.get('options',{}).get('appearance_revision') or {}).get('design_reconstruction_plan')
    extra_inputs = run.get('reference_pipeline_inputs') or {}
    if (set(extra_inputs) == {'material_targets'} and plan
            and any(row.get('kind') == 'reference_appearance_stack' for row in plan.get('layers', []))):
        return prepare_material_finish(root, run, extra_inputs['material_targets'],
            cancel=cancel, progress=progress)
    selected_registry = _registry_identity(root, run)
    previous_registry = (plan or {}).get('provenance', {}).get('local_model_registry', {})
    registry_changed = (previous_registry.get('sha256') != selected_registry['sha256'] or
        any(previous_registry.get(key, default) != selected_registry[key] for key, default in
            (('semantics_profile', 'sapiens2_seg_0.4b'),
             ('matting_profile', 'vitmatte_small_distinctions'))))
    if (not plan or not _current_alpha(plan)
            or registry_changed
            or plan.get('source_sha256') != run['source']['sha256']
            or plan.get('candidate_sha256') != (run.get('reference') or {}).get('sha256')
            or not any(x.get('kind')=='background_design' for x in plan.get('layers',[]))):
        from design_reconstruction_pipeline import prepare_scene
        plan=prepare_scene(root,run)['design_reconstruction_plan']
        run.setdefault('options',{}).setdefault('appearance_revision',{})['design_reconstruction_plan']=plan
    for role in ('source','reference'):
        path=Path(run[role]['path'])
        if not path.is_absolute():path=root/path
        run[role]['path']=str(path.resolve())
        if _sha(path) != run[role]['sha256']:
            raise ValueError('Original/reference contents changed; import the current file as a new input')
    alpha=_current_alpha(plan)
    if _sha(_resolve(root, alpha['path'])) != alpha['sha256']:
        raise ValueError('Current matte bytes do not match this edit snapshot')
    identity = _input_identity(root, run)
    stack=next((x for x in plan.get('layers',[]) if x.get('kind')=='reference_appearance_stack'),None)
    if stack and not rebuild and _prepared_matches(root,run,{'design_reconstruction_plan':plan},input_identity=identity):
        report('prepared','已复用本版本的人物与场景融合资产；准备不会改动已保存版本。',1)
        return _attach_independent_effects(root,run,{'ok':True,'design_reconstruction_plan':plan,'cache_hit':True,'new_rendered_revision':False,'project_pointer_updated':False})
    cache_root=root/'outputs/cache/reference-pipeline'
    if not rebuild:
        for path in sorted(cache_root.glob('**/prepared-plan.json')):
            try:
                result=_read(path)
            except (OSError, ValueError):
                continue
            if _prepared_matches(root,run,result,input_identity=identity):
                report('prepared','已复用匹配当前原片、参考及人物边缘的准备结果。',1)
                return _attach_independent_effects(root,run,{**result,'cache_hit':True,'new_rendered_revision':False,'project_pointer_updated':False})
    key=_digest(identity)[:20]
    work=cache_root/key;work.mkdir(parents=True,exist_ok=True)
    worker_run=copy.deepcopy(run)
    # Diagnostic fitting reads this alpha role; the original production source
    # baseline remains immutable. New matte preparation estimates a fresh pair.
    worker_plan=worker_run['options']['appearance_revision']['design_reconstruction_plan']
    worker_plan['source_baseline']['assets']['foreground_alpha']=copy.deepcopy(alpha)
    worker_plan['mask_roles']['foreground_alpha']=copy.deepcopy(alpha)
    # Re-fitting always decodes the original source and consumes its recorded
    # source transform. The prepared asset domain is never erased or relabelled.
    worker_path=work/'input-snapshot.json';_write(worker_path,worker_run)
    journal=work/'preparation-state.json'
    state={'schema':'reference_pipeline_job.v1','identity':identity,'input_sha256':_sha(worker_path),
           'status':'preparing','completed_stages':[],'saved_version_modified':False,'generation_requested':False}
    _write(journal,state)
    seed=root/'outputs/inspection/reference-match-v2'
    match=seed if not rebuild and _diagnostic_ready(root,worker_run,seed,input_identity=identity) else work/'matching'
    extra={key: str(_resolve(root, path)) for key, path in (run.get('reference_pipeline_inputs') or {}).items() if path}
    def stage(name,message,command):
        report(name,message,STAGES.index(name)/len(STAGES))
        state.update(stage=name,status='preparing');_write(journal,state)
        _process(root,command,work,cancel)
        state['completed_stages'].append(name);_write(journal,state)
    try:
        if match != seed:
            (match / 'pipeline-inputs.json').unlink(missing_ok=True)
            fit=[_code_file('evaluate_reference_colour.py'),'--root',root,'--run',worker_path,'--output',match]
            for key,flag in [('object_masks','--object-masks'),('authored_colour_edit','--authored-colour-edit')]:
                if extra.get(key):fit += [flag,extra[key]]
            stage('fit','按原片与参考准备人物仿色。',fit)
            stage('colour_fusion','准备同材质颜色融合和眼口衔接。',[
                _code_file('reference_colour_blend.py'),'--root',root,'--run',worker_path,'--match-dir',match])
            rendering=[_code_file('render_reference_colour_composite.py'),'--run',worker_path,'--match-dir',match,
                '--reference-blend',match/'reference-colour-layer.json']
            stage('native_baseline','建立真实原生仿色基线。',rendering)
            balance=[_code_file('reference_scene_balance.py'),'--root',root,'--run',worker_path,'--match-dir',match,'--warm-colour']
            stage('scene_balance','协调人物与环境的宽尺度明暗和光色。',balance)
            stage('native_target','生成完整原生人物融合目标。',rendering+['--scene-balance',match/'scene-balance-layer.json'])
            _write(match / 'pipeline-inputs.json', identity)
        else:
            report('reuse_native_target','复用已绑定当前原片与参考的原生候选，避免重复拟合。',.8)
        stage('freeze','保存可编辑的人物基底与融合差值。',[
            _code_file('reference_appearance_prepare.py'),'--run',worker_path,'--match-dir',match,'--output',work/'appearance'])
        result=_read(work/'appearance/prepared-plan.json')
        if extra.get('material_targets'):
            derived = copy.deepcopy(worker_run)
            derived['options']['appearance_revision']['design_reconstruction_plan'] = result['design_reconstruction_plan']
            result = prepare_material_finish(root, derived, extra['material_targets'],
                output=work/'material-finish', cancel=cancel, progress=progress)
        _stamp_inputs(root, result, identity)
        attach_reference_support(root, result['design_reconstruction_plan'], work / 'appearance')
        result = _attach_independent_effects(root, run, result)
        _write(work / 'appearance/prepared-plan.json', result)
        if not _prepared_matches(root,run,result):
            raise ValueError('Frozen appearance assets did not satisfy their current source/reference/matte contract')
        state.update(status='prepared',stage='prepared',prepared_plan=str(work/'appearance/prepared-plan.json'));_write(journal,state)
        report('prepared','人物与场景融合已准备；预览后保存高清新版本。',1)
        return result
    except BaseException as error:
        state.update(status='cancelled' if isinstance(error,InterruptedError) else 'failed',error=str(error));_write(journal,state)
        raise
