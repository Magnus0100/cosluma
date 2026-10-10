"""Independent CPU counterexamples for completion pixels, budgets and cancellation.

Runs the real system coordinator and file hashing. Stage producers and text
answers are controlled offline fixtures; this is not actual API quality QA.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import cosplay_post_review as visual
from generation_design import DESIGN_VERSION
import single_source_workflow as workflow
import system_post_completion as system


def fixture(root,name):
    directory=root/workflow.RUNS_DIR/name;directory.mkdir(parents=True)
    source=root/'source.png';reference=root/'reference.png'
    if not source.exists():Image.new('RGB',(32,40),'#805040').save(source)
    if not reference.exists():Image.new('RGB',(32,40),'#506080').save(reference)
    artifacts={}
    for key in ('auto_draft','render_master','render_scene_master'):
        path=directory/(key+'.png');Image.new('RGB',(32,40),'#706060').save(path)
        artifacts[key]=str(path)
    run={'kind':'single_input_run','run_id':name,'revision':1,'sample':'fixture','status':'needs_visual_review',
        'source':{'path':str(source),'sha256':workflow.sha256(source)},
        'reference':{'path':str(reference),'sha256':workflow.sha256(reference)},
        'generation_design':{'schema_version':DESIGN_VERSION,'status':'ready'},
        'options':{},'artifacts':artifacts,'auto_draft':{'outputs':copy.deepcopy(artifacts)},
        'generation_budget':{'image_submissions':2,'total_image_submissions':2,'correction_image_submissions':0}}
    workflow.write_json(directory/'manifest.json',run)
    return run


def reviewed(root,current,args):
    output=root/workflow.RUNS_DIR/current['run_id']/'cosplay-post-review'
    # Coordinator qualification is controlled offline. This test exercises
    # real file/pixel identity and orchestration, not the review API or vision.
    final=Path(current['artifacts']['auto_draft'])
    result={'schema':visual.SCHEMA,'scope_contract':'cosplay_review_scope.v1',
        'status':'reviewed','actual_api_evidence_valid':True,
        'offline_controlled_diagnosis':True,'diagnostic_gate':'unknown',
        'binding':{'images':{'final':{'path':str(final),'sha256':workflow.sha256(final)}}},
        'checks':{name:{'reference':{'status':'pass'},'final':{'status':'unknown' if name=='native_detail' else 'pass'}}
            for name in visual.CRITERIA}}
    result['run_binding']={'run_id':current['run_id'],'revision':current['revision'],
        'source_sha256':current['source']['sha256'],'reference_sha256':current['reference']['sha256'],
        'final_sha256':workflow.sha256(Path(current['artifacts']['auto_draft'])),
        'render_artifacts':{key:workflow.sha256(Path(value))
            for key,value in current['artifacts'].items()}}
    current['cosplay_post_review']=result
    workflow.write_json(output/'review.json',result)
    return result


def args():
    return SimpleNamespace(generate=False,complete_post=True,cosplay_correction_budget=1,
        preserve_effective_prompt=False,cancel_request_path=None)


def own(root,current,name):
    child=copy.deepcopy(current);child['parent_run_id']=current['run_id'];child['run_id']=name;child['revision']+=1
    directory=root/workflow.RUNS_DIR/name;directory.mkdir(parents=True)
    for key,value in child['artifacts'].items():
        path=directory/Path(value).name;path.write_bytes(Path(value).read_bytes());child['artifacts'][key]=str(path)
    child['auto_draft']={'outputs':copy.deepcopy(child['artifacts'])};child['cosplay_post_review']={'status':'unavailable'}
    workflow.write_json(directory/'manifest.json',child)
    return child


def noop_effect(_root,current,_args,**kwargs):
    return {'handled':False,'run':current,'status':'skipped','reason':'offline fixture has no declared independent effect'}


def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='system-completion-independent-audit-') as folder:
        root=Path(folder);project=root/'project.json';project.write_text('{"selection":"saved-parent"}')
        project_sha=workflow.sha256(project)
        parent=root/'saved-parent.json';parent.write_text('{"saved":"unchanged"}');parent_sha=workflow.sha256(parent)
        # A matching run ID is insufficient when current final pixels changed.
        stale=fixture(root,'stale-final');reviewed(root,stale,args())
        assert system._review_current(stale),'valid offline pixel-bound fixture must qualify for reuse'
        for role in ('source','reference'):
            path=Path(stale[role]['path']);original=path.read_bytes();path.write_bytes(original+b'changed-frozen-input')
            try:
                assert not system._review_current(stale),'BUG: changed '+role+' bytes accepted by stored SHA metadata'
            finally:path.write_bytes(original)
        checks.append('changed_actual_source_and_reference_bytes_cannot_inherit_saved_review')
        Image.new('RGB',(32,40),'#204090').save(stale['artifacts']['auto_draft'])
        assert not system._review_current(stale),'BUG: changed final pixels accepted by same run/revision IDs'
        calls=[]
        def diagnose(_root,current,settings):
            calls.append(current['run_id']);return reviewed(_root,current,settings)
        with patch('cosplay_post_completion.continue_cosplay_post',side_effect=lambda _r,current,_a:current), \
             patch('automatic_foreground_effects.complete_effects_once',side_effect=noop_effect), \
             patch.object(workflow,'diagnose_cosplay_post',side_effect=diagnose):
            completed=system.complete_once(root,stale,args())
        assert calls==['stale-final'] and system._review_current(completed)
        checks.append('changed_final_pixels_force_a_new_bound_review_during_authorized_completion')
        # A completed task may reopen without a request, but a changed master
        # cannot inherit its previous completed verdict.
        Path(completed['artifacts']['render_master']).write_bytes(b'changed-current-master')
        with patch.object(workflow,'diagnose_cosplay_post',side_effect=AssertionError('saved reopen cannot call review')), \
             patch('cosplay_post_completion.continue_cosplay_post',side_effect=AssertionError('saved reopen cannot generate')), \
             patch('automatic_foreground_effects.complete_effects_once',side_effect=AssertionError('saved reopen cannot generate')):
            reopened=system.complete_once(root,stale,args())
        assert not system._review_current(reopened)
        assert (reopened.get('system_post_completion') or {}).get('status') not in ('completed','completed_with_findings'), \
            'BUG: changed completed-result master reopened with a completed verdict'
        checks.append('changed_completed_result_is_stale_without_implicit_review_or_image_requests')
        # A cancellation arriving during the final text call must win over its
        # successful answer; the draft is retained rather than marked complete.
        cancelled=fixture(root,'cancel-final-review');settings=args();signal=root/'cancel-final.txt'
        settings.cancel_request_path=signal
        def cancel_during_review(_root,current,settings):
            result=reviewed(_root,current,settings);signal.write_text('cancel');return result
        with patch('cosplay_post_completion.continue_cosplay_post',side_effect=lambda _r,current,_a:current), \
             patch('automatic_foreground_effects.complete_effects_once',side_effect=noop_effect), \
             patch.object(workflow,'diagnose_cosplay_post',side_effect=cancel_during_review):
            result=system.complete_once(root,cancelled,settings)
        assert result['system_post_completion']['status']=='cancelled_draft_retained', \
            'BUG: cancellation during final review was reported completed'
        assert result['system_post_completion']['all_stages_checked_once'] is False
        checks.append('cancel_during_final_review_retains_draft_and_cannot_report_completed')
        # Independent stage mocks control the pixels; the real coordinator
        # must consume them in order without resetting shared counts in children.
        ordered=fixture(root,'ordered');reviewed(root,ordered,args())
        ordered['cosplay_post_review']['checks']['skin_materials']['reference']['status']='fail'
        events=[]
        def material(_root,current,settings,**kwargs):
            events.append('material');child=own(root,current,'material')
            child['cosplay_post_completion']={'status':'qualified_material_child_saved','correction_submissions':0}
            assert child['generation_budget']['image_submissions']==2
            return {'handled':True,'run':child,'status':'qualified_material_child_saved'}
        def scene(_root,current,settings):
            events.append('scene');assert current['run_id']=='material' and system._review_current(current)
            assert settings._material_completion_checked and current['generation_budget']['correction_image_submissions']==0
            child=own(root,current,'scene');child['generation_budget'].update(correction_image_submissions=1,total_image_submissions=3)
            child['cosplay_post_completion']={'status':'corrected_child_saved','correction_submissions':1}
            return child
        def effect(_root,current,settings,**kwargs):
            events.append('effect');assert current['run_id']=='scene' and current['generation_budget']['total_image_submissions']==3
            child=own(root,current,'effect')
            child['automatic_foreground_effects']={'status':'independent_effect_child_saved','independent_effect_image_submissions':1,
                'actual_visual_gate':'pass'}
            return {'handled':True,'run':child,'status':'independent_effect_child_saved'}
        def qa(_root,current,settings):
            events.append('review:'+current['run_id']);return reviewed(_root,current,settings)
        with patch('automatic_material_completion.complete_materials_once',side_effect=material), \
             patch('cosplay_post_completion.continue_cosplay_post',side_effect=scene), \
             patch('automatic_foreground_effects.complete_effects_once',side_effect=effect), \
             patch.object(workflow,'diagnose_cosplay_post',side_effect=qa):
            result=system.complete_once(root,ordered,args());count=len(events)
            reopened=system.complete_once(root,ordered,args())
        assert events==['material','review:material','scene','effect','review:effect'] and len(events)==count
        assert result['generation_budget']['total_image_submissions_including_effects']==4
        assert result['generation_budget']['correction_image_submissions']==1
        assert result['generation_budget']['independent_effect_image_submissions']==1
        assert reopened['run_id']=='effect' and system._review_current(reopened)
        checks.append('material_scene_effect_child_order_shared_budget_and_valid_reopen_zero_requests')
        ordinary=args();ordinary.complete_post=False
        with patch('cosplay_post_completion.continue_cosplay_post',side_effect=AssertionError('ordinary reopen cannot generate')), \
             patch('automatic_foreground_effects.complete_effects_once',side_effect=AssertionError('ordinary reopen cannot generate')), \
             patch.object(workflow,'diagnose_cosplay_post',side_effect=AssertionError('ordinary reopen cannot review')):
            assert system.complete_once(root,result,ordinary)==result
        assert workflow.sha256(project)==project_sha and workflow.sha256(parent)==parent_sha
        checks.append('ordinary_reopen_no_api_and_parent_root_selection_preserved')
    print(json.dumps({'ok':True,'checks':len(checks),'names':checks,
        'scope':'real coordinator/file binding; controlled offline stage pixels and mock text answers',
        'external_api_requests':0,'gpu_used':False}))


if __name__=='__main__':main()
