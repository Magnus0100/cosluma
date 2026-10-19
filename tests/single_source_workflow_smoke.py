#!/usr/bin/env python3
"""Check source-only entry states and the explicit saved-master delivery gate.

The native source-only fixture uses the current source baseline without local models.
Reference colour preparation belongs to reference_pipeline_smoke.py; an imported
candidate in this temporary workspace must remain blocked and retained.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import design_reconstruction_pipeline as design_pipeline
import reference_pipeline
import single_source_workflow as workflow


def check_draft_during_diagnosis(root,common,candidate,local):
    """The normal saved-run reader can open a draft before slow QA returns."""
    from generation_design import DESIGN_VERSION
    observed=[]
    for final_status in ('reviewed','unavailable'):
        entered=threading.Event();release=threading.Event();results=[];errors=[]
        run_id='run-qa-'+final_status
        args=workflow.parse_args(common+['--run-id',run_id,'--reference',str(candidate)])
        args.cosplay_review_only=True
        args.generation_design={'schema_version':DESIGN_VERSION,'status':'ready','effective_prompt':'Offline fixture.'}
        def render(_root,run,_args):
            # The pending QA fixture explicitly authorizes saved-image review.
            from creative_task import normalize_task_intent
            run['task_intent']=normalize_task_intent({'schema_version':'cosplay_task_intent.v1',
                'mode':'existing_reference','services':{'text_vision':True,'visual_review':True}})
            run['generation_design']=_args.generation_design
            output=root/'outputs/runs'/run['run_id']/'render';output.mkdir(parents=True)
            artifacts={}
            for key in ('auto_draft','render_master','render_scene_master'):
                source=Path(local['artifacts'][key]);target=output/source.name
                shutil.copy2(source,target);artifacts[key]=str(target)
            return {'outputs':artifacts,'metadata':{'fixture':True},'review':{'fixture':True}}
        def diagnosis(_root,run,_args):
            entered.set()
            assert release.wait(5), 'Slow QA fixture was not released'
            review={'status':final_status,'actual_api_evidence_valid':False,'mocked':True,
                'visual_acceptance':False,'automatic_accepted':False,'human_review_required':True}
            if final_status=='unavailable':review['error']='Synthetic diagnosis failure; saved draft retained.'
            run['cosplay_post_review']=review
            return review
        def execute():
            try:results.append(workflow.create_run(args))
            except Exception as exc:errors.append(exc)
        with patch.object(workflow,'render_candidate',side_effect=render), \
             patch.object(workflow,'diagnose_cosplay_post',side_effect=diagnosis), \
             patch.object(workflow,'generate_candidate',side_effect=AssertionError('QA fixture generated an image')), \
             patch('runtime_preflight.preflight_for_run',return_value={'status':'ready'}):
            worker=threading.Thread(target=execute);worker.start()
            try:
                assert entered.wait(5), errors
                path,saved=workflow.load_run(root,run_id)
                assert path.is_file() and saved['status']=='needs_visual_review'
                assert saved['delivery_status']=='auto_draft' and not saved.get('acceptance')
                pending=saved['cosplay_post_review']
                assert pending['status']=='pending' and pending['actual_api_evidence_valid'] is False
                assert pending['automatic_accepted'] is False and pending['visual_acceptance'] is False
                assert all(Path(saved['artifacts'][key]).is_file()
                    for key in ('auto_draft','render_master','render_scene_master'))
                project=workflow.read_json(root/'project.json')
                assert project['state']['last_source_only_run']==saved['manifest']
                assert project['state']['status']=='needs_visual_review'
                observed.append(final_status)
            finally:release.set();worker.join(5)
            assert not worker.is_alive() and not errors, errors
        assert len(results)==1 and results[0]['cosplay_post_review']['status']==final_status
        _,saved=workflow.load_run(root,run_id)
        assert saved['cosplay_post_review']==results[0]['cosplay_post_review']
        assert saved['status']=='needs_visual_review' and not saved.get('acceptance')
        assert saved['cosplay_post_review']['automatic_accepted'] is False
        assert Path(saved['artifacts']['auto_draft']).read_bytes()==Path(local['artifacts']['auto_draft']).read_bytes()
    assert observed==['reviewed','unavailable']


def main() -> int:
    with tempfile.TemporaryDirectory(prefix='cosplay-source-only-') as value:
        root = Path(value)
        source, candidate = root/'source.jpg', root/'candidate.png'
        Image.new('RGB', (128, 96), (120, 80, 160)).save(source, quality=92)
        Image.new('RGB', (128, 96), (160, 120, 200)).save(candidate)
        common = ['--root',str(root),'--source',str(source),'--alpha-backend','birefnet']
        runtime_calls = []
        def runtime_gate(*args, **kwargs):
            runtime_calls.append(kwargs)
            return {'schema_version':'cosplay_runtime_preflight.v1','status':'ready',
                    'route':'local_portrait_source_silhouette',
                    'required_registry_keys':['birefnet_model_path','onnx_runtime_root']}
        with patch.object(workflow,'generate_candidate',side_effect=AssertionError('Local entry generated a candidate')), \
             patch.object(reference_pipeline,'prepare',side_effect=AssertionError('Unavailable reference preparation was invoked')), \
             patch.object(design_pipeline,'ready',return_value=False), \
             patch('runtime_preflight.preflight_for_run', side_effect=runtime_gate), \
             patch('source_alpha.source_silhouette',side_effect=lambda _root,_source,image,**kw: (Image.new('L',image.size,255),{'backend':'synthetic_test_matte'})):
            awaiting = workflow.create_run(workflow.parse_args(common+['--run-id','run-awaiting']))
            assert awaiting['status']=='awaiting_reference' and awaiting['reference'] is None
            assert not awaiting.get('auto_draft')
            blocked = workflow.create_run(workflow.parse_args(common+['--run-id','run-candidate','--reference',str(candidate)]))
            assert blocked['status']=='blocked' and blocked['error']['candidate_retained'] is True
            assert blocked['error']['automatic_image_retry'] is False
            assert Path(blocked['reference']['path']).read_bytes()==candidate.read_bytes()
            assert not blocked.get('auto_draft') and not blocked.get('fallback')
            local = workflow.create_run(workflow.parse_args(common+['--run-id','run-local','--local-only']))
            assert local['status']=='needs_visual_review' and local['reference'] is None
            assert local['options']['render_mode']=='native_source_design'
            assert local['delivery_status']=='local_source_draft'
        local_gate = next(call for call in runtime_calls if call.get('mode') == 'local_portrait')
        assert local_gate['require_local_models'] is True
        assert local_gate['execution_strategy'] == 'layered_reference_transfer'
        assert 'cosplay_post_review' not in local, 'Unrequested local QA became pending'
        check_draft_during_diagnosis(root,common,candidate,local)
        checks = {key:'pass' for key in workflow.VISUAL_CHECK_KEYS}
        for unavailable in ('run-awaiting','run-candidate'):
            try:
                workflow.accept_run(root,unavailable,reviewer='fixture',checks=checks)
                raise AssertionError('An unrendered entry passed the delivery gate')
            except RuntimeError as exc:
                assert 'not awaiting visual review' in str(exc), exc
        masters = {role:Path(local['artifacts'][role]) for role in ('render_master','render_scene_master')}
        assert all(path.is_file() for path in masters.values())
        assert local['render']['render_precision']['calibration_rgb8'] is False
        review_path = root/'outputs/runs/run-local/visual_review.json'
        review = json.loads(review_path.read_text(encoding='utf8'))
        assert all(value=='not_checked' for value in review['checks'].values())
        review['revision'] = 999
        review_path.write_text(json.dumps(review),encoding='utf8')
        try:
            workflow.accept_run(root,'run-local',reviewer='fixture',checks=checks)
            raise AssertionError('Delivery accepted review from another revision')
        except RuntimeError as exc:
            assert 'revision' in str(exc), exc
        review['revision'] = local['revision']
        review_path.write_text(json.dumps(review),encoding='utf8')
        try:
            workflow.accept_run(root,'run-local',reviewer='fixture')
            raise AssertionError('A successful render inherited visual acceptance')
        except RuntimeError as exc:
            assert 'visual checks' in str(exc), exc
        delivered = workflow.accept_run(root,'run-local',reviewer='fixture',checks=checks)
        assert delivered['status']=='delivered'
        delivery = root/'outputs/final_delivery/source_only/source/run-local'
        receipt = json.loads((delivery/'delivery.json').read_text(encoding='utf8'))
        assert receipt['run_id']=='run-local' and receipt['checks']==checks
        assert receipt['source_sha256']==workflow.sha256(source) and receipt['candidate_sha256'] is None
        assert receipt['software_version'] and receipt['fallback'] is None
        for path in masters.values():
            assert (delivery/path.name).read_bytes()==path.read_bytes()
            assert receipt['artifacts'][path.name]==workflow.sha256(path)
        assert receipt['render_master_sha256']==workflow.sha256(masters['render_master'])
        assert not (delivery/'candidate.png').exists()
        # The active workflow owns the existing per-run lock. Approval must
        # fail cleanly rather than racing its final manifest write.
        with workflow._run_id_lock(root, 'run-local'):
            try:
                workflow.accept_run(root, 'run-local', reviewer='fixture', checks=checks)
                raise AssertionError('approval bypassed the active run lock')
            except FileExistsError as exc:
                assert 'already in progress' in str(exc)
        bad = root/'bad.txt'; bad.write_text('not an image',encoding='utf8')
        try:
            workflow.create_run(workflow.parse_args(['--root',str(root),'--source',str(bad),'--run-id','run-bad']))
            raise AssertionError('Unsupported source extension was accepted')
        except ValueError: pass
        print(json.dumps({'ok':True,'awaiting_reference_without_render':True,'blocked_candidate_retained':True,
            'explicit_native_source_only_fixture':True,'unrendered_accept_rejected':True,
            'review_revision_and_eight_checks_required':True,'saved_master_delivery_bytes_and_hashes':True,
            'no_generation_or_reference_models':True,'unsupported_source_rejected':True,
            'draft_readable_during_pending_qa':True,'slow_and_failed_diagnosis_retains_draft':True,
            'visual_quality_acceptance_simulated':True}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
