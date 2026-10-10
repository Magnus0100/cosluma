"""Current masked adapter, generation budget and comprehensive QA regressions."""
import copy
import hashlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import cosplay_post_review as visual
from cosplay_post_review_smoke import multi_fake_transport,response,verdict
from cosplay_native_review import prepare_cosplay_native_review
from current_material_fixture import prepare_fixture
import reference_refinement as refinement
import reference_provider


def main():
    with tempfile.TemporaryDirectory(prefix='current-masked-refinement-') as folder:
        root=Path(folder);fixture=prepare_fixture(root/'inputs');run=fixture['run']
        source=Path(fixture['source']['binding']['image']['path']);reference=Path(run['reference']['path'])
        original=Path(run['source']['path']);qa=Path(fixture['diagnosis']['evidence']['path'])
        design=run['generation_design'];analysis=fixture['reference']['binding']['analysis']['path']
        workflow=root/'current-workflow.json'
        visual._write(workflow,{'model_name':'CPU current fixture','randomize_seed':False,'output_node':'3','workflow':{
          '1':{'class_type':'LoadImage','inputs':{'image':'{{source}}'}},
          '2':{'class_type':'TextEncodeQwenImage21','inputs':{'prompt':'{{prompt}}','images.image_1':['1',0]}},
          '3':{'class_type':'SaveImage','inputs':{'filename_prefix':'fixture'}}}})
        conf={'provider':'cpu-mock','registry_id':'cpu-mock','kind':'comfyui','model':str(workflow),
          'endpoint':'http://cpu-mock.invalid','configured':True,'generation_supported':True,'api_key_configured':True,
          'enabled':True,'workflow_configured':True,'workflow_path':str(workflow),'protocol':'comfyui-api',
          'api_key_env':None,'credential_source':'none','timeout':1,'retry':{'max_attempts':1}}
        wire=io.BytesIO();Image.new('RGB',Image.open(reference).size,'#728282').save(wire,'PNG')
        wire_bytes=wire.getvalue();multi_calls=[]
        def fake_multi(endpoint, image, model_path, prompt, timeout, events, **kwargs):
            inputs = kwargs['condition_inputs']
            multi_calls.append(copy.deepcopy(inputs))
            assert [row['role'] for row in inputs] == ['scene_reference', 'source_identity', 'edit_mask']
            assert [row['sha256'] for row in inputs[:2]] == [refinement._sha(reference), refinement._sha(source)]
            assert '<image1>' in prompt and '<image2>' in prompt and '<image3>' in prompt
            if len(multi_calls) == 1:
                assert 'DIAGNOSED MATERIAL CORRECTIONS:' in prompt
            else:
                assert 'DIAGNOSED SCENE AND EFFECT CORRECTIONS:' in prompt
                assert 'placement and hierarchy may change' in prompt
                assert 'Retain subject material colours' in prompt and 'white material regions' not in prompt
            names = ['inputs/cpu-scene.png', 'inputs/cpu-source.png', 'inputs/cpu-mask.png']
            graph = reference_provider.comfyui_adapter.prepare_condition_workflow(
                reference_provider.comfyui_adapter.load_workflow(model_path), names, prompt)
            uploads = [{**row, 'bytes': Path(row['path']).stat().st_size,
                'condition_only': True, 'hard_edit_mask_enforced': False,
                'upload': {'name': name.split('/')[-1], 'subfolder': 'inputs'}}
                for row, name in zip(inputs, names)]
            job = 'cpu-multi-'+str(len(multi_calls))
            events.extend([{'event': 'comfyui_submitted', 'prompt_id': job},
                           {'event': 'comfyui_completed', 'prompt_id': job}])
            return {'id': job, 'comfyui': {'workflow_sha256': refinement._sha(model_path),
                'submitted_workflow_sha256': hashlib.sha256(json.dumps(graph, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest(),
                'condition_inputs': uploads, 'request': {'prompt': graph, 'client_id': 'cpu-multi'}}}, wire_bytes, 'comfyui_view'

        with patch.object(reference_provider,'provider_config',return_value=conf), \
             patch.object(reference_provider.provider_settings,'resolve_credential',return_value=''), \
             patch.object(reference_provider.comfyui_adapter,'request_generation',side_effect=fake_multi):
            result=refinement.refine_reference_once(source,reference,qa,design,root/'generation',original_source=original,
                reference_analysis=analysis,mask_root=root,generation_only=True)
            assert result['status']=='generated' and not result['eligible_for_composition'],result
            assert len(multi_calls)==1 and result['source_display_sent_as_second_image']
            repeated=refinement.refine_reference_once(source,reference,qa,design,root/'generation',original_source=original,
                reference_analysis=analysis,mask_root=root,generation_only=True)
            assert repeated['status']=='budget_exhausted' and len(multi_calls)==1
            rejected=refinement.refine_reference_once(source,reference,qa,design,root/'old-mode',correction_mode='legacy_single',generation_only=True)
            assert rejected['status']=='rejected' and len(multi_calls)==1
            scene_design=copy.deepcopy(design)
            scene_design['design_brief']['primary_effect_plan']={'mode':'scene_effect','description':'Amber smoke behind the shoulder'}
            scene_response=response(True)
            scene_response['checks']['effects_space']['reference']=verdict('reference',status='fail',cause='reference_defect',
                stage='reference_generation',action='Restore the missing amber smoke outside the protected subject silhouette.',panels=['source','reference'])
            scene_sampling=fixture['prepared_native']
            with patch.object(visual.vision_transport,'request_vision_json',multi_fake_transport(scene_response)):
                scene_qa=visual.review_cosplay_post(source,reference,root/'scene-review',scene_design,
                  final=scene_sampling['final_path'],native_crops=scene_sampling['native_crops'],native_selection=scene_sampling['receipt'])
            assert scene_qa['status']=='reviewed',scene_qa
            scene_result=refinement.refine_reference_once(source,reference,root/'scene-review/review.json',scene_design,
                root/'scene-generation',original_source=original,reference_analysis=analysis,
                source_analysis=fixture['source']['binding']['analysis']['path'],mask_root=root,generation_only=True,
                correction_criteria=('identity_makeup','concept_hierarchy','effects_space','scene_integration'))
            assert scene_result['status']=='generated' and not scene_result['eligible_for_composition'],scene_result
            assert len(multi_calls)==2 and refinement._validated_generation(root/'scene-generation')[0]['generation_attempts']==1
        candidate=Path(result['candidate'])
        actions=refinement._bound_review(source,reference,qa,design)[1]
        permitted,_=refinement.ownership.validate_correction_mask(result['material_ownership'],reference,actions)
        assert np.array_equal(np.asarray(Image.open(candidate))[~permitted],np.asarray(Image.open(reference))[~permitted])
        assert refinement._validated_generation(root/'generation')[0]['generation_attempts']==1
        review_run=copy.deepcopy(fixture['sampling_run']);review_run['reference']={'path':str(candidate),'sha256':visual._sha(candidate)}
        review_run['options']['appearance_revision']['design_reconstruction_plan']['candidate_sha256']=visual._sha(candidate)
        sampling=prepare_cosplay_native_review(review_run,root/'candidate-native-sampling',comprehensive=True)
        with patch.object(visual.vision_transport,'request_vision_json',multi_fake_transport(response(True))):
            checked=visual.review_cosplay_post(sampling['source_path'],candidate,root/'candidate-comprehensive',design,
              final=sampling['final_path'],native_crops=sampling['native_crops'],native_selection=sampling['receipt'])
        assert checked['status']=='reviewed'
        evaluation=refinement.evaluate_reference_refinement(root/'generation',candidate_review_path=root/'candidate-comprehensive/review.json')
        assert evaluation['status']=='reviewed' and evaluation['eligible_for_composition'],evaluation
        eligible=refinement.validate_composition_eligibility(root/'generation',candidate=candidate,source_sha256=visual._sha(original),design=design)
        assert eligible['eligible_for_composition'],eligible
        assert not refinement.validate_composition_eligibility(root/'generation',candidate=reference)['eligible_for_composition']
        assert not refinement.validate_composition_eligibility(root/'generation',source_sha256='0'*64)['eligible_for_composition']
    print(json.dumps({'status':'passed','cases_passed':10,'adapter':'actual current three-image registration',
      'scene_prompt_and_source_guarded_mask':True,
      'network_requests':0,'gpu_requests':0,'qualification_validator_mocks':0}))


if __name__=='__main__':main()
