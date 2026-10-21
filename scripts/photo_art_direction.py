"""One reserved image-observation request produces scoped editing decisions.

This is an execution tool, never a preflight. Reopening an unresolved attempt
does not grant another model call. The existing vision transport owns routing
and credentials; the returned design grants neither pixels nor acceptance.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from asset_identity import sha256_file
from current_model_registry import atomic_json
from photo_editing import (compile_edit_prompt, identity, planner_prompt,
                           validate_observed_design, validate_scope)

SCHEMA='cosluma_photo_art_direction.v1'
GENERATION_SCHEMA='cosluma_scoped_generation_design.v1'


def _binding(source, scope, brief, character_name, character_notes, connection):
    from runtime_resources import code_path
    environment={key:os.environ.get(key) for key in (
        'COSPLAY_VISION_PROVIDER','COSPLAY_VISION_MODEL','COSPLAY_VISION_API_BASE','COSPLAY_VISION_API_URL')}
    return {'source':{'path':str(source),'sha256':sha256_file(source)},
            'scope_id':scope['scope_id'],
            'intent_sha256':identity({'brief':brief,'character_name':character_name,'character_notes':character_notes}),
            'connection_identity':identity({'selected':connection,'environment':environment}),
            'producer':{name:sha256_file(code_path(name)) for name in (
                'photo_art_direction.py','photo_editing.py','vision_transport.py')}}


def _ready(record, binding, scope):
    if (not isinstance(record,dict) or record.get('schema')!=SCHEMA or record.get('binding')!=binding):
        raise ValueError('Observed design belongs to different inputs, scope, code or model selection')
    if record.get('status')!='ready':
        raise RuntimeError('An art-direction attempt already exists; inspect its outcome before another explicit task')
    design=record['design']
    content={key:design[key] for key in ('summary','observation','changes','lighting','keep','uncertain')}
    if validate_observed_design(content,scope)!=design:
        raise ValueError('Saved observed design changed')
    expected={family:compile_edit_prompt(design,scope,family) for family in ('gpt_image','qwen_image_2_1')}
    if record.get('compiled_prompts')!=expected:
        raise ValueError('Saved model-specific edit prompts changed')
    provenance=record.get('provenance') or {}
    if (provenance.get('source')!=binding['source'] or provenance.get('network_used') is not True
            or provenance.get('generation_used') is not False or not provenance.get('model')):
        raise ValueError('Observed design has no matching vision execution evidence')
    return record


def _complete(record, payload, provenance, binding, scope, source, cancel):
    if cancel and cancel():raise InterruptedError('Art direction cancelled after model response')
    if sha256_file(source)!=binding['source']['sha256']:
        raise ValueError('Original source changed during visual planning')
    design=validate_observed_design(payload,scope)
    completed={**record,'status':'ready','proposed_design':payload,'provenance':provenance,
        'design':design,'compiled_prompts':{family:compile_edit_prompt(design,scope,family)
                                           for family in ('gpt_image','qwen_image_2_1')}}
    completed.pop('error',None)
    completed=_ready(completed,binding,scope)
    if cancel and cancel():raise InterruptedError('Art direction cancelled before design publication')
    return completed


def prepare(source, output, scope, *, brief='', character_name='', character_notes='',
            provider=None, vision_model=None, api_base=None, api_url=None, api_key_env=None,
            cancel=None):
    """Observe the actual photograph once, then compile both model dialects."""
    from vision_transport import request_vision_json, replay_vision_json
    import workflow_timing
    source=Path(source).resolve();output=Path(output).resolve();scope=validate_scope(scope)
    if not source.is_file():raise FileNotFoundError(source)
    prompt=planner_prompt(scope,brief=brief,character_name=character_name,character_notes=character_notes)
    connection={'provider':provider,'vision_model':vision_model,'api_base':api_base,
                'api_url':api_url,'api_key_env':api_key_env}
    binding=_binding(source,scope,brief,character_name,character_notes,connection)
    destination=output/'art-direction.json'
    if destination.is_file():
        record=json.loads(destination.read_text(encoding='utf-8'))
        if (not isinstance(record,dict) or record.get('schema')!=SCHEMA or record.get('binding')!=binding):
            raise ValueError('Observed design belongs to different inputs, scope, code or model selection')
        if record.get('status')=='ready':return {**_ready(record,binding,scope),'reused':True}
        if (record.get('status') not in {'reserved','interrupted'} or record.get('scope')!=scope
                or record.get('planning_prompt')!=prompt or record.get('model_requests_attempted')!=1
                or record.get('image_submissions')!=0 or record.get('automatic_resubmission') is not False):
            raise RuntimeError('An art-direction attempt already exists; inspect its outcome before another explicit task')
        if cancel and cancel():raise InterruptedError('Art direction cancelled before local response replay')
        # Only a durably saved complete response can finish interrupted local
        # preparation. Unknown/partial outcomes remain untouched; no HTTP or
        # quota is created, and old producer identities are never adopted.
        started=time.perf_counter()
        try:
            payload,provenance=replay_vision_json(source,output/'observation',prompt,
                expected_connection_identity=binding['connection_identity'],vision_model=vision_model)
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise RuntimeError('No verified completed art-direction response is available; no request was resubmitted') from error
        completed=_complete(record,payload,provenance,binding,scope,source,cancel)
        completed['response_replay_seconds']=time.perf_counter()-started
        atomic_json(destination,completed)
        return {**completed,'reused':True}
    if cancel and cancel():raise InterruptedError('Art direction cancelled before model submission')
    output.mkdir(parents=True,exist_ok=True)
    record={'schema':SCHEMA,'status':'reserved','binding':binding,'scope':scope,
            'user_intent':{'brief':brief,'character_name':character_name,'character_notes':character_notes},
            'planning_prompt':prompt,
            'model_requests_attempted':1,'image_submissions':0,'automatic_resubmission':False,
            'automatic_visual_acceptance':False}
    # Reserve exclusively before transport. The file also survives interruption
    # and prevents another process from consuming the same operation budget.
    with destination.open('x',encoding='utf-8') as handle:
        json.dump(record,handle,ensure_ascii=False,indent=2)
        handle.flush();os.fsync(handle.fileno())
    started=time.perf_counter()
    try:
        with workflow_timing.span('visual_art_direction',model_request=True,http_request=True):
            payload,provenance=request_vision_json(source,output/'observation',prompt,**connection)
        # Retain actual returned evidence even when its proposed operation is
        # invalid. A schema rejection is not permission to ask the model again.
        record.update(proposed_design=payload,provenance=provenance)
        record=_complete(record,payload,provenance,binding,scope,source,cancel)
        record['seconds']=time.perf_counter()-started
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'failed',
            error={'type':type(error).__name__,'message':str(error)[:1200]},seconds=time.perf_counter()-started)
        atomic_json(destination,record)
        raise
    atomic_json(destination,record)
    return {**record,'reused':False}


def generation_design(record, scope, *, family, source_sha256):
    """Bind an observed decision and its selected dialect to one generation."""
    scope=validate_scope(scope)
    _ready(record,record.get('binding'),scope)
    if record['binding']['source']['sha256']!=source_sha256 or record['scope']!=scope:
        raise ValueError('Observed generation design belongs to another source or scope')
    compiled=record['compiled_prompts'][family]
    return {'schema_version':GENERATION_SCHEMA,'status':'ready',
        'design_origin':'observed_scoped_edit','source_sha256':source_sha256,'scope_id':scope['scope_id'],
        'observed_design':record['design'],'art_direction':record,'compiled_prompt':compiled,
        'effective_prompt':compiled['prompt'],'text_vision_submissions':record['model_requests_attempted'],
        'automatic_visual_acceptance':False}


def validate_generation_design(design, scope, *, family, source_sha256):
    if not isinstance(design,dict) or design.get('schema_version')!=GENERATION_SCHEMA:
        raise ValueError('A complete observed editing design is required')
    expected=generation_design(design['art_direction'],scope,family=family,source_sha256=source_sha256)
    if design!=expected:
        raise ValueError('Frozen observed editing decisions or model prompt changed')
    return design
