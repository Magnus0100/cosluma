"""Counterexamples for final-only routing, evidence and one owned repair pass."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from generation_design import DESIGN_VERSION
from cosplay_repair_plan import plan_final_repairs
import single_source_workflow as workflow
import system_post_completion as system


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        project = root/'project.json'; project.write_text('{"selection":"saved"}')
        source, reference = root/'source.png', root/'reference.png'
        Image.new('RGB', (24, 32), '#705060').save(source)
        Image.new('RGB', (24, 32), '#906040').save(reference)
        directory = root/workflow.RUNS_DIR/'input'
        directory.mkdir(parents=True)
        final = directory/'draft.png'; Image.new('RGB', (24, 32), '#805040').save(final)
        run = {'kind': 'single_input_run', 'run_id': 'input', 'root': str(root), 'revision': 1, 'status': 'needs_visual_review',
               'source': {'path': str(source), 'sha256': workflow.sha256(source)},
               'reference': {'path': str(reference), 'sha256': workflow.sha256(reference)},
               'artifacts': {'auto_draft': str(final)}, 'options': {},
               'generation_design': {'schema_version': DESIGN_VERSION, 'status': 'ready'}}
        def review(current, failed=True):
            final_path = Path(current['artifacts']['auto_draft'])
            current['cosplay_post_review'] = {'status': 'reviewed', 'actual_api_evidence_valid': True,
                'run_binding': {'run_id': current['run_id'], 'revision': current['revision'],
                    'source_sha256': current['source']['sha256'], 'reference_sha256': current['reference']['sha256'],
                    'final_sha256': workflow.sha256(final_path),
                    'render_artifacts': {'auto_draft': workflow.sha256(final_path)}},
                'binding': {'images': {'final': {'path': str(final_path), 'sha256': workflow.sha256(final_path)}}},
                'diagnostic_gate': 'fail' if failed else 'unknown', 'checks': {
                    'effects_space': {'reference': {'status': 'pass'}, 'final': {
                        'status': 'fail' if failed else 'pass', 'cause': 'transfer_incomplete' if failed else 'none',
                        'correction_stage': 'scene_composition' if failed else None,
                        'action': 'Audit retained reference effect before selecting a repair.'}},
                    'identity_makeup': {'reference': {'status': 'unknown'}, 'final': {'status': 'unknown'}}}}
        review(run)
        plan = plan_final_repairs(run, current_review=True)
        assert plan['tasks'][0]['producer'] == 'background_consistency'
        assert plan['tasks'][0]['disposition'] == 'audit_before_local_repair'
        assert not plan['tasks'][1]['pixel_repair_qualified']
        assert not any(row['pixel_repair_qualified'] for row in plan_final_repairs(run, current_review=False)['tasks'])
        material = copy.deepcopy(run)
        material['cosplay_post_review']['checks']['skin_materials'] = {
            'reference': {'status': 'pass'}, 'final': {'status': 'fail', 'correction_stage': 'subject_appearance', 'cause': 'transfer_incomplete'}}
        assert any(row['producer'] == 'qualified_material_finish' for row in plan_final_repairs(material, current_review=True)['tasks'])
        preserved = {str(p): workflow.sha256(p) for p in (source, reference, final, project)}
        workflow.write_json(directory/'manifest.json', run)
        settings = SimpleNamespace(complete_post=True, generate=False, cosplay_correction_budget=1,
                                   preserve_effective_prompt=False, cancel_request_path=None)
        calls = []
        def scene(_root, current, args, **kwargs):
            calls.append('scene_audit_repair')
            assert system._review_current(current) and kwargs == {'bounded': True}
            child = copy.deepcopy(current)
            child.update(run_id='repaired', parent_run_id=current['run_id'], revision=2)
            target = root/workflow.RUNS_DIR/'repaired'/'draft.png'; target.parent.mkdir(parents=True)
            Image.new('RGB', (24, 32), '#906040').save(target)
            child['artifacts']['auto_draft'] = str(target)
            child['cosplay_post_review'] = {'status': 'unavailable'}
            workflow.write_json(target.parent/'manifest.json', child)
            return {'handled': True, 'run': child, 'status': 'local_scene_child_saved'}
        def diagnose(_root, current, args):
            calls.append('review:'+current['run_id']); review(current, failed=False)
        with patch('cosplay_post_completion.continue_cosplay_post', side_effect=lambda _r, current, _a: current), \
             patch('automatic_foreground_effects.complete_effects_once', side_effect=lambda _r, current, _a, **kwargs: {'run': current, 'status': 'skipped'}), \
             patch('automatic_scene_completion.complete_scene_once', side_effect=scene), \
             patch.object(workflow, 'diagnose_cosplay_post', side_effect=diagnose):
            result = system.complete_once(root, run, settings)
            assert calls == ['scene_audit_repair', 'review:repaired']
            assert result['run_id'] == 'repaired' and system._review_current(result)
            assert result['system_post_completion']['local_repair_rounds'] == 1
            assert result['system_post_completion']['visual_state'] == 'findings_open'
            assert result['system_post_completion']['unknown_checks'] == ['identity_makeup']
            assert system.complete_once(root, run, settings)['run_id'] == 'repaired'
            assert calls == ['scene_audit_repair', 'review:repaired']
        local = copy.deepcopy(result)
        local['run_id'] = 'local-only-task'; local['revision'] = 3
        local.pop('system_post_completion', None)
        review(local, failed=False)
        workflow.write_json(root/workflow.RUNS_DIR/local['run_id']/'manifest.json', local)
        no_images = copy.copy(settings); no_images.final_repair_only = True; no_images.cosplay_correction_budget = 0
        with patch('cosplay_post_completion.continue_cosplay_post', side_effect=AssertionError('Local repair cannot generate a reference')), \
             patch('automatic_foreground_effects.complete_effects_once', side_effect=AssertionError('Local repair cannot generate RGBA')):
            retained = system.complete_once(root, local, no_images)
        assert retained['system_post_completion']['reference_image_budget'] == 0
        assert retained['system_post_completion']['independent_effect_image_budget'] == 0
        assert retained['generation_budget']['independent_effect_image_budget'] == 0
        receiver = copy.deepcopy(local)
        receiver.update(run_id='receiver-only-task', revision=6)
        receiver.pop('system_post_completion', None)
        review(receiver, failed=False)
        missing_detail = {'status':'fail', 'cause':'retouch_missing', 'correction_stage':'native_retouch',
            'evidence':[{'basis':'visible_pixels','panels':['source-shoe','final-shoe']}]}
        receiver['cosplay_post_review']['checks']['native_detail'] = {'final': missing_detail}
        receiver['cosplay_post_review']['local_observations'] = [{'pair_id':'left_shoe_contact',
            'region':'left_shoe_contact', 'scope':'matched_source_final_native_pair_only', 'native_box':[0,0,12,12],
            'source_panel':'source-shoe', 'final_panel':'final-shoe', 'checks': {
                'native_detail':missing_detail, 'identity_makeup':{'status':'pass'}, 'skin_materials':{'status':'pass'}}}]
        tasks = plan_final_repairs(receiver,current_review=True)['tasks']
        assert any(row['producer']=='native_receiver_detail' for row in tasks)
        workflow.write_json(root/workflow.RUNS_DIR/receiver['run_id']/'manifest.json', receiver)
        calls.clear()
        def receiver_repair(_root,current,args,**kwargs):
            calls.append('receiver-local')
            assert system._review_current(current) and kwargs=={'bounded':True}
            child=copy.deepcopy(current); child.update(run_id='receiver-repaired',parent_run_id=current['run_id'],revision=7)
            target=root/workflow.RUNS_DIR/child['run_id']/'draft.png';target.parent.mkdir(parents=True)
            Image.new('RGB',(24,32),'#905040').save(target)
            child['artifacts']['auto_draft']=str(target);child['cosplay_post_review']={'status':'unavailable'}
            workflow.write_json(target.parent/'manifest.json',child)
            return {'run':child,'status':'receiver_detail_child_saved'}
        with patch('automatic_receiver_detail_completion.complete_receiver_detail_once',side_effect=receiver_repair), \
             patch.object(workflow,'diagnose_cosplay_post',side_effect=diagnose), \
             patch('cosplay_post_completion.continue_cosplay_post',side_effect=AssertionError('Receiver repair cannot generate')), \
             patch('automatic_foreground_effects.complete_effects_once',side_effect=AssertionError('Receiver repair cannot generate RGBA')):
            repaired=system.complete_once(root,receiver,no_images)
        assert calls==['receiver-local','review:receiver-repaired']
        assert repaired['system_post_completion']['local_repair_rounds']==1
        assert repaired['system_post_completion']['reference_image_budget']==0
        assert not any(row['producer']=='native_receiver_detail' for row in
                       repaired['system_post_completion']['remaining_repair_plan']['tasks'])
        partial = copy.deepcopy(local)
        partial.update(run_id='partial-board-task', revision=4)
        partial.pop('system_post_completion', None)
        review(partial, failed=False)
        partial['cosplay_post_review'].update(status='partial', actual_api_evidence_valid=False,
                                              text_vision_requests_attempted=8)
        workflow.write_json(root/workflow.RUNS_DIR/partial['run_id']/'manifest.json', partial)
        with patch.object(workflow, 'diagnose_cosplay_post', side_effect=AssertionError('Partial QA cannot resend eight boards')), \
             patch('cosplay_post_completion.continue_cosplay_post', side_effect=AssertionError('Partial QA cannot generate')), \
             patch('automatic_foreground_effects.complete_effects_once', side_effect=AssertionError('Partial QA cannot generate RGBA')):
            kept_partial = system.complete_once(root, partial, no_images)
        assert not system._review_current(kept_partial)
        assert kept_partial['system_post_completion']['visual_state']=='findings_open'
        assert kept_partial['cosplay_post_review']['text_vision_requests_attempted']==8
        missing = copy.deepcopy(local)
        missing.update(run_id='unavailable-evidence-task', revision=5)
        missing.pop('system_post_completion', None)
        missing['cosplay_post_review']={'status':'unavailable'}
        workflow.write_json(root/workflow.RUNS_DIR/missing['run_id']/'manifest.json', missing)
        attempts=[]
        def unavailable(_root, current, _args):
            attempts.append(current['run_id'])
            current['cosplay_post_review']={'status':'unavailable', 'actual_api_evidence_valid':False}
        with patch.object(workflow, 'diagnose_cosplay_post', side_effect=unavailable):
            kept_missing=system.complete_once(root, missing, no_images)
        assert attempts==['unavailable-evidence-task']
        assert kept_missing['system_post_completion']['status']=='completed_with_findings'
        assert all(workflow.sha256(Path(p)) == sha for p, sha in preserved.items())
    print(json.dumps({'ok': True, 'reference_pass_final_fail_consumed': True,
                      'unknown_and_stale_no_pixel_or_image_authority': True,
                      'owned_child_re_reviewed_once': True, 'ordinary_reentry_no_requests': True,
                      'unresolved_identity_preserved': True, 'offline_fixture_not_visual_acceptance': True}))


if __name__ == '__main__':
    main()
