"""Scoped appearance adapter for the existing checked render entry point.

No generation, segmentation, beauty or source-detail restoration occurs here.
The coordinator owns execution authorization and atomic revision publication.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
from PIL import Image

from asset_identity import sha256_file
from current_model_registry import atomic_json
from appearance_assets import load_asset, validate_asset, DOMAIN as WORKING_DOMAIN
from photo_editing import compile_edit_prompt, validate_observed_design, validate_scope


def prepare_selected_appearance(root,run,selected,*,session_directory):
    """Stage the actual reviewed best for a new immutable appearance asset.

    The caller publishes this staged run only after rendering succeeds. The
    prior complete draft and receipt remain available on every failure path.
    """
    import photo_edit_session as session
    from photo_editing import identity
    root=Path(root).resolve();scope=validate_scope(run['task_plan']['editing_scope'])
    state=session.snapshot(session_directory)
    best=session.selected_output(session_directory)['candidate']
    unreviewed_initial=session.unreviewed_initial_draft_allowed(state,best)
    if (best!=selected or not best or (best.get('review') is None and not unreviewed_initial)
            or state['binding']['source']['sha256']!=run['source']['sha256']
            or state['binding']['scope_id']!=scope['scope_id']
            or state['binding']['design_id']!=run['generation_design']['observed_design']['design_id']
            or not best.get('appearance_asset')):
        raise ValueError('Only the actual reviewed complete edit can own new appearance')
    image=Path(best['image']['path']).resolve()
    if sha256_file(image)!=best['image']['sha256']:
        raise ValueError('Reviewed best image changed')
    staged=copy.deepcopy(run)
    asset=best['appearance_asset']
    receipt=validate_asset(asset,source_sha256=run['source']['sha256'],scope=scope,
        design_id=state['binding']['design_id'])
    if receipt['files']['preview']['sha256']!=best['image']['sha256']:
        raise ValueError('Selected appearance master differs from its actual reviewed preview')
    old=staged['options'].get('photo_appearance')
    if old:
        staged.setdefault('appearance_history',[]).append({'reference':copy.deepcopy(run.get('reference')),
            'photo_appearance':old,'render':copy.deepcopy(run.get('render')),
            'outputs':copy.deepcopy(run.get('artifacts') or {})})
    from photo_edit_finish import reviewed_candidate_record
    if (run.get('task_plan') or {}).get('shaping_request') is None:
        staged['reference']=reviewed_candidate_record(root,best,staged['options'],evidence_state=state)
    permission={'schema':'cosluma_reviewed_appearance_preparation.v1',
        'source_sha256':run['source']['sha256'],'candidate_sha256':best['image']['sha256'],
        'scope_id':scope['scope_id'],'design_id':state['binding']['design_id'],
        'session_id':state['session_id'],'candidate_id':best['candidate_id'],
        'selected_review_sha256':identity(best['review']),'asset_id':asset['asset_id']}
    permission['permission_id']=identity(permission)
    staged['options']['photo_appearance_selection']=permission
    staged['options']['photo_appearance']=copy.deepcopy(asset)
    return staged


def _path(root, value):
    path = Path(value)
    return (path if path.is_absolute() else root/path).resolve()


def _component(value):
    if not isinstance(value, str) or not value or value in {'.','..'} or '/' in value or '\\' in value:
        raise ValueError('Expected a run/sample path component')
    return value


def _display_encoding():
    return {'schema_version':'display_transform.v1','method':'clip_sdr','input_domain':WORKING_DOMAIN,
        'output_space':'sRGB','gamut_mapping':'encoding_clip_only'}


def saved_display_transform(root, run):
    """Verify a typed saved master and its one current display encoding.

    The source-space display selector deliberately remains owned by the native
    producer. Current geometric masters already occupy display-linear sRGB and
    must not be decoded, deformed or display-transformed a second time here.
    """
    scope,design,_,_=validate_run(root,run)
    asset=run['options']['photo_appearance']
    receipt=validate_asset(asset,source_sha256=run['source']['sha256'],scope=scope,
        design_id=design['observed_design']['design_id'])
    render=run.get('render') or {};operator=render.get('operator') or {}
    display=operator.get('display_transform')
    if (render.get('photo_appearance')!=asset or operator.get('asset_id')!=receipt['asset_id']
            or operator.get('master_sha256')!=receipt['files']['master']['sha256']
            or operator.get('working_domain')!=WORKING_DOMAIN or display!=_display_encoding()):
        raise ValueError('Saved typed appearance has no matching current display encoding')
    return copy.deepcopy(display)


def validate_run(root, run):
    """Validate the scoped design and input bindings without preparing pixels."""
    root = Path(root).resolve()
    plan = run.get('task_plan') or {}
    if plan.get('ready') is not True:
        raise ValueError('Generated appearance requires a ready checked task')
    scope = validate_scope(plan.get('editing_scope'))
    typed=(run.get('options') or {}).get('photo_appearance')
    if not typed:raise ValueError('Current scoped rendering is missing its typed native appearance asset')
    source = run.get('source') or {}
    reference = run.get('reference') or {}
    source_path = _path(root, source['path'])
    receipt=validate_asset(typed,source_sha256=source['sha256'],scope=scope,
        design_id=run['generation_design']['observed_design']['design_id'])
    candidate_path=Path(typed['path'])/receipt['files']['preview']['file']
    shaping=(run.get('options') or {}).get('shaping_state')
    shaping_request=plan.get('shaping_request')
    if shaping_request is not None:
        from photo_shaping import normalize_request, validate_authorization, validate_state
        request=normalize_request(shaping_request)
        state=validate_state(shaping,current=typed,source_sha256=source['sha256'],scope=scope,
            design_id=run['generation_design']['observed_design']['design_id'])
        if state['request']!=request or state['authorization']!=plan.get('shaping_authorization'):
            raise ValueError('Saved shaping differs from its checked execution contract')
        validate_authorization(state['authorization'],source_sha256=source['sha256'],scope=scope,
            design_id=run['generation_design']['observed_design']['design_id'],
            base_appearance=state['base_appearance'],source_appearance=run.get('initial_source_appearance'))
        base_receipt=validate_asset(state['base_appearance'],source_sha256=source['sha256'],scope=scope,
            design_id=run['generation_design']['observed_design']['design_id'])
        if (reference.get('sha256')!=base_receipt['files']['preview']['sha256']
                or ((plan.get('inputs') or {}).get('reference') or {}).get('sha256')!=reference.get('sha256')):
            raise ValueError('Shaping reference must remain the exact checked baseline U')
    elif shaping is not None:
        raise ValueError('Shaping state requires its checked shaping task')
    if shaping_request is None and reference and reference.get('sha256')!=receipt['files']['preview']['sha256']:
        raise ValueError('Current reference preview differs from its typed appearance master')
    if ((source_path.exists() and sha256_file(source_path)!=source.get('sha256'))
            or ((plan.get('inputs') or {}).get('source') or {}).get('sha256')!=source.get('sha256')):
        raise ValueError('Scoped rendering inputs differ from the checked source/candidate')
    checked_reference = (plan.get('inputs') or {}).get('reference')
    if shaping_request is None and checked_reference and checked_reference.get('sha256') != reference.get('sha256'):
        selection=(run.get('options') or {}).get('photo_appearance_selection') or {}
        from photo_editing import identity
        selected_body={key:value for key,value in selection.items() if key!='permission_id'}
        typed_selected=(typed is not None and selection.get('permission_id')==identity(selected_body)
            and selection.get('source_sha256')==source['sha256'] and selection.get('scope_id')==scope['scope_id']
            and selection.get('asset_id')==typed['asset_id']
            and selection.get('candidate_sha256')==reference.get('sha256'))
        authorization=plan.get('editing_refinement_authorization') or {}
        binding=run.get('derived_edit_candidate') or {}
        from photo_editing import identity
        content={key:value for key,value in binding.items() if key!='binding_id'}
        if not typed_selected and (binding.get('schema')!='cosluma_derived_edit_candidate.v1'
                or binding.get('binding_id')!=identity(content)
                or binding.get('scope_id')!=scope['scope_id']
                or binding.get('source_sha256')!=source['sha256']
                or binding.get('candidate_sha256')!=reference['sha256']
                or binding.get('base_image_sha256')!=authorization.get('base_image_sha256')
                or checked_reference.get('sha256')!=authorization.get('base_image_sha256')
                or (binding.get('improved') is True and binding.get('selected_image_sha256')!=reference['sha256'])
                or (binding.get('improved') is False and authorization.get('parent_candidate_sha256')!=reference['sha256'])):
            raise ValueError('Imported candidate differs from the checked task or selected editing result')
    design = run.get('generation_design') or (run.get('options') or {}).get('generation_design') or {}
    observed = design.get('observed_design') or {}
    if (design.get('schema_version') != 'cosluma_scoped_generation_design.v1'
            or design.get('status') != 'ready' or design.get('source_sha256') != source['sha256']
            or design.get('scope_id') != scope['scope_id']):
        raise ValueError('Scoped generation design lacks the current source/scope binding')
    content = {key:observed.get(key) for key in ('summary','observation','changes','lighting','keep','uncertain')}
    if validate_observed_design(content, scope) != observed:
        raise ValueError('Observed design has changed')
    compiled = design.get('compiled_prompt') or {}
    expected=compile_edit_prompt(observed,scope,compiled.get('family'))
    direction = design.get('art_direction') or {}
    bundle={family:compile_edit_prompt(observed,scope,family) for family in ('gpt_image','qwen_image_2_1')}
    if (compiled!=expected or design.get('effective_prompt')!=expected['prompt']
            or direction.get('compiled_prompts')!=bundle):
        raise ValueError('Current model prompt differs from the observed design/current compiler')
    binding = direction.get('binding') or {}
    provenance = direction.get('provenance') or {}
    if (direction.get('status') != 'ready' or direction.get('design') != observed
            or (binding.get('source') or {}).get('sha256') != source['sha256']
            or binding.get('scope_id') != scope['scope_id']
            or (provenance.get('source') or {}).get('sha256') != source['sha256']
            or provenance.get('network_used') is not True
            or provenance.get('generation_used') is not False or not provenance.get('model')):
        raise ValueError('Scoped design lacks matching image-observation evidence')
    options = run.get('options') or {}
    revision = options.get('appearance_revision') or {}
    if (options.get('manual_layers') or any(value for key,value in revision.items() if key != 'workflow_graph')
            or options.get('denoise_strength') or options.get('native_source_retouch')):
        raise ValueError('Source-space manual/beauty/lighting adjustments need explicit generated-canvas support')
    return scope, design, source_path, candidate_path


def _copy_same_bytes(source, target):
    shutil.copy2(source, target)


def render_candidate(root, run, args):
    """Save the checked native float asset; never reconstruct it from PNG/RAW."""
    import image_helpers
    from raw_support import save_render_preview_and_master, save_scene_master
    import single_source_workflow as workflow
    import workflow_timing
    root=Path(root).resolve();workflow.check_generation_cancel(args)
    with workflow_timing.span('photo_appearance_validation',model_request=False,http_request=False):
        scope,design,source_path,candidate_path=validate_run(root,run)
        plan=run['task_plan'];options=run.get('options') or {}
        auth=plan.get('editing_refinement_authorization') or {}
        manual=(plan.get('mode')=='existing_reference' and plan.get('editing_iteration') is not None
            and auth.get('schema')=='cosluma_editing_refinement_authorization.v1'
            and auth.get('source_sha256')==run['source']['sha256'] and auth.get('scope_id')==scope['scope_id']
            and auth.get('design_id')==design['observed_design']['design_id']
            and isinstance(auth.get('parent_appearance_id'),str))
        shaping=plan.get('mode')=='existing_reference' and plan.get('shaping_request') is not None
        if plan.get('editing_initialization')!={'schema':'cosluma_source_native_initialization.v1','kind':'source_native'} and not (manual or shaping):
            raise ValueError('Current scoped rendering requires the checked source-native initialization policy')
        asset=options.get('photo_appearance');baseline=run.get('initial_source_appearance')
        if not asset or not baseline:raise ValueError('Current rendering is missing its typed appearance/baseline asset')
        binding={'source_sha256':run['source']['sha256'],'scope':scope,
            'design_id':design['observed_design']['design_id']}
        receipt=validate_asset(asset,**binding);source_receipt=validate_asset(baseline,**binding)
        if source_receipt['kind']!='source_baseline' or source_receipt['source_origin']!=receipt['source_origin']:
            raise ValueError('Current appearance does not descend from the actual source baseline')
        if source_receipt['canvas_size']!=receipt['canvas_size']:
            raise ValueError('Current appearance changed its source canvas')
    run_dir=root/'outputs/runs'/_component(run['run_id']);sample=_component(run['sample'])
    parent=run_dir/'render'/sample;parent.mkdir(parents=True,exist_ok=True)
    destination=parent/('appearance-'+uuid4().hex[:12]);staging=destination.with_name('.'+destination.name+'.partial')
    staging.mkdir();review_path=run_dir/'visual_review.json'
    names={'source':source_path,'reference_candidate':candidate_path,
        'styled_source':destination/'styled_source.png','auto_draft':destination/'auto_draft.png',
        'alpha':destination/'generated-coverage.png','silhouette':destination/'generated-coverage-alias.png',
        'source_coverage':destination/'source-coverage.png',
        'render_master':destination/'render_master.tif','render_scene_master':destination/'render_scene.tif',
        'comparison':destination/'comparison.jpg','metadata':destination/'metadata.json','visual_review':review_path}
    paths={key:str(path) for key,path in names.items()}
    display=_display_encoding()
    operator={'engine':'photo-appearance-v1','working_domain':WORKING_DOMAIN,'asset_id':asset['asset_id'],
        'master_sha256':receipt['files']['master']['sha256'],
        'appearance_kind':receipt['kind'],'render_mode':'typed_native_appearance',
        'composition_policy':'frozen_current_float_master_with_explicit_generated_coverage',
        'source_subject_RGB_restored':False,'camera_detail_claimed':False,'display_transform':display,
        'visual_acceptance':False,'human_approved':False}
    precision={'source_precision':receipt['source_origin']['source_precision'],
        'decode_version':receipt['source_origin']['decode_version'],'working_precision':'float32_linear',
        'working_domain':WORKING_DOMAIN,'engine':operator['engine'],'master_encoding':'uint16_srgb',
        'scene_master_encoding':'float32_linear_sRGB','working_master_encoding':'float32_linear_sRGB',
        'scene_linear_master':False,'display_transform':display,'calibration_rgb8':False,
        'source_preparation':receipt['source_origin']['transform'],'source_origin':receipt['source_origin'],
        'appearance_origin':receipt['kind'],'generated_resolution':{'input_size':receipt['generated_resolution'],
            'output_canvas':receipt['canvas_size'],'native_generated_detail':False,
            'resampling':('frozen_geometric_RGB_and_G_same_inverse_mapping' if receipt['kind']=='geometric_transform' else
                'declared_generation_to_current_canvas_before_asset_freeze')},
        'generated_coverage':receipt['generated_coverage'],
        'source_coverage_semantics':receipt['source_coverage_semantics']}
    if receipt['kind']=='geometric_transform':
        operator['geometric_pixels_replayed_without_redeformation']=True
        precision['generated_resolution']['generation_this_operation']=False
        precision['generated_resolution']['original_generation_provenance']='owned_unshaped_parent'
    review={'schema_version':'single_source_visual_review.v1','mode':'single_source_visual',
        'status':'needs_review','disposition':'needs_review','supervision':'none',
        'run_id':run['run_id'],'revision':run.get('revision',1),'reviewer':None,'reviewed_at':None,
        'checks':{key:'not_checked' for key in workflow.VISUAL_CHECK_KEYS},
        'visual_layers_seen':['source','reference_candidate','auto_draft'],
        'visual_notes':'Typed appearance; identity, geometry and physical lighting require actual review.',
        'artifact_paths':paths}
    metadata={'schema_version':workflow.SCHEMA_VERSION,'software_version':workflow.SOFTWARE_VERSION,
        'kind':'scoped_photo_appearance_render','run_id':run['run_id'],'revision':run.get('revision',1),
        'sample':sample,'created_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'inputs':{'source':str(source_path),'reference_candidate':str(candidate_path),'manual_final':None,'gt':None},
        'outputs':paths,'operator':operator,'render_precision':precision,'photo_appearance':copy.deepcopy(asset),
        'initial_source_appearance':copy.deepcopy(baseline),
        'alpha':{'role':'cumulative_generated_coverage','backend':'typed_pixel_provenance',
            'path':paths['alpha'],'silhouette':paths['silhouette'],'source_coverage':paths['source_coverage'],
            'structure_safe':False,'is_source_foreground':False,'manual_layers_geometry':None},
        'composition_policy':operator['composition_policy'],'observed_design_id':binding['design_id'],
        'editing_scope_id':scope['scope_id'],'review':{'status':'needs_visual_review','gate':'visual_only',
            'path':str(review_path),'numeric_metrics_are_navigation_only':True},
        'manual_final':None,'manual_layers':{},'provenance':run.get('reference'),
        'reviewed_preview':{'sha256':receipt['files']['preview']['sha256'],
            'saved_bytes_identical':True,'master_RGB8_and_ICC_equal':True}}
    loaded=None
    try:
        loaded=load_asset(asset,**binding);rendered=loaded['master']
        with workflow_timing.span('photo_appearance_render',model_request=False,http_request=False):
            preview=save_render_preview_and_master(rendered,staging/'render_master.tif',display_transform=display)
            save_scene_master(rendered,staging/'render_scene.tif',working_domain=WORKING_DOMAIN)
            with Image.open(loaded['preview_path']) as checked:
                if (checked.info.get('icc_profile')!=preview.info.get('icc_profile')
                        or not np.array_equal(np.asarray(checked),np.asarray(preview))):
                    raise ValueError('Master-derived output differs from the actual candidate preview')
            _copy_same_bytes(loaded['preview_path'],staging/'auto_draft.png')
            _copy_same_bytes(staging/'auto_draft.png',staging/'styled_source.png')
            metadata['saved_artifact_sha256']={key:sha256_file(staging/Path(names[key]).name)
                for key in ('auto_draft','render_master','render_scene_master')}
            for key,value in (('generated-coverage',loaded['generated_coverage']),('source-coverage',loaded['source_coverage'])):
                canvas=(Image.new('L',tuple(receipt['canvas_size']),value*255) if type(value) is int else
                    Image.fromarray(np.uint8(value*255+.5)))
                image_helpers.save_png(canvas,staging/(key+'.png'))
            _copy_same_bytes(staging/'generated-coverage.png',staging/'generated-coverage-alias.png')
            with Image.open(Path(baseline['path'])/source_receipt['files']['preview']['file']) as source_preview,Image.open(loaded['preview_path']) as current:
                image_helpers.comparison_sheet((('source',source_preview),('current appearance',current),('draft',preview)),
                    staging/'comparison.jpg')
            atomic_json(staging/'metadata.json',metadata)
            validate_run(root,run);validate_asset(asset,**binding);validate_asset(baseline,**binding)
            workflow.check_generation_cancel(args)
            os.rename(staging,destination);atomic_json(review_path,review)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        shutil.rmtree(destination,ignore_errors=True)
        raise
    finally:
        closed=set()
        for item in (loaded,) if loaded is not None else ():
            for key in ('master','generated_coverage','edit_coverage'):
                mapping=getattr(item[key],'_mmap',None)
                if mapping is not None and id(mapping) not in closed:
                    mapping.close();closed.add(id(mapping))
    return {'outputs':paths,'metadata':metadata,'review':review,'review_path':str(review_path)}
