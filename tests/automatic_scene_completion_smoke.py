"""Independent scene diagnosis identities and durable no-repeat coordinator."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import automatic_scene_completion as completion
from current_material_fixture import prepare_fixture
import cosplay_post_review as visual
from cosplay_post_review_smoke import response,verdict,multi_fake_transport


def main():
    cases=[]
    with tempfile.TemporaryDirectory(prefix='scene-diagnosis-current-') as folder:
        root=Path(folder);directory=root/'outputs/runs/native-offline-fixture'
        fixture=prepare_fixture(directory/'fixture',diagnosis_target='final');run=fixture['run']
        prepared=fixture['prepared_native'];payload=response(True)
        payload['checks']['effects_space']['final']=verdict('final',status='fail',cause='transfer_incomplete',
            stage='scene_composition',action='Restore the visibly missing reference environment effect.',
            panels=['source','reference','final'])
        with patch.object(visual.vision_transport,'request_vision_json',multi_fake_transport(payload)):
            review=visual.review_cosplay_post(prepared['source_path'],Path(run['reference']['path']),
                directory/'cosplay-post-review',run['generation_design'],final=prepared['final_path'],
                native_crops=prepared['native_crops'],native_selection=prepared['receipt'])
        assert review['status']=='reviewed',review.get('error')
        review['run_binding']={**prepared['receipt']['run_binding'],
            'final_sha256':review['binding']['images']['final']['sha256'],
            'render_artifacts':{name:completion.sha(path) for name,path in run['artifacts'].items()}}
        run['cosplay_post_review']=review;completion.write(directory/'cosplay-post-review/review.json',review)
        completion.write(directory/'manifest.json',run)
        assert completion._diagnosis(root,run)['failed_criteria']==['effects_space']
        cases.append('actual_final_scene_diagnosis')
        for name,operation in [('run',lambda r:r.update(run_id='other')),
            ('revision',lambda r:r.update(revision=42)),
            ('master',lambda r:r['artifacts'].update(render_scene_master=r['artifacts']['render_master']))]:
            changed=copy.deepcopy(run);operation(changed)
            try:completion._diagnosis(root,changed)
            except (ValueError,OSError):pass
            else:raise AssertionError('Stale scene endpoint accepted: '+name)
            cases.append(name+'_rejected')
        unsupported=copy.deepcopy(run)
        unsupported['cosplay_post_review']['checks']['effects_space']['final']['correction_stage']='unsupported_layering'
        assert completion._diagnosis(root,unsupported) is None;cases.append('unsupported_not_routed')
        unknown=copy.deepcopy(run);unknown['cosplay_post_review']['checks']['effects_space']['final']['status']='unknown'
        assert completion._diagnosis(root,unknown) is None;cases.append('unknown_not_routed')
        audit_path=root/'no-repair-audit.json';completion.write(audit_path,{'status':'audited_no_repair'})
        def audited(*args,**kwargs):
            return {'changed':False,'audit':{'status':'audited_no_repair','reason':'background_observation_retained',
                'receipt_asset':{'path':str(audit_path),'sha256':completion.sha(audit_path)}}}
        before=completion.sha(directory/'manifest.json')
        with patch.object(completion,'prepare_scene_consistency',audited):
            first=completion.complete_scene_once(root,run,SimpleNamespace())
            second=completion.complete_scene_once(root,run,SimpleNamespace())
        assert first['status']=='audited_no_repair' and second['status']=='already_attempted'
        assert completion.sha(directory/'manifest.json')==before
        cases.extend(['audited_no_repair_no_child','persistent_attempt_not_repeated','parent_manifest_exact'])
    print(json.dumps({'status':'passed','cases':cases,'API_requests':0,'model_inference_requests':0}))


if __name__=='__main__':main()
