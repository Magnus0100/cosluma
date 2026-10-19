"""Checked direction survives HTTP options, CLI settings and producer entry."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from backend_server import Backend
from creative_task import plan_task, execution_request
from creative_direction_smoke import enhanced
from creative_task_smoke import fixture
from production_coordinator import route_contract
import single_source_workflow as workflow
from task_model_preflight_smoke import mocked_task_capability


def main():
    with tempfile.TemporaryDirectory(prefix='direction-execution-') as temporary:
        root = Path(temporary)
        registry, sources, _ = fixture(root)
        body = {'source':sources[0], 'model_registry':str(registry),
            'provider':'qwen_image_2_1_uncensored_fast',
            'character_name':'祝小红','character_notes':'两只蝴蝶是情感意象，不是技能',
            'input_prompt':'Keep the person and original scene.',
            'creative_task':enhanced()}
        with patch('task_model_preflight.inspect_task_models',side_effect=mocked_task_capability), \
                patch('reference_provider.provider_health',return_value={'ready':True}):
            planned = plan_task(root, body)
            checked = execution_request(root,{**planned['request'],'plan_id':planned['plan']['plan_id']})
        backend = Backend(root,embedded=True)
        argv = backend._one_click_args(checked, checked['source'])
        settings = json.loads(argv[argv.index('--options-json')+1])
        assert settings['creative_direction'] == checked['task_plan']['creative_direction']
        payload = backend._one_click_payload(checked, checked['source'])
        assert payload['creative_direction'] == settings['creative_direction']
        # Existing-reference image generation is opt-in only, and never grants
        # a global retry, a correction or an independent effect submission.
        existing = copy.deepcopy(checked['task_intent'])
        existing['mode'] = 'existing_reference'
        existing['regional_refinement'] = {'maximum_submissions': 1,
            'coordinate_space': 'source_canvas',
            'regions': [{'id': 'new-local-task', 'core_box': [16,16,64,64]}]}
        region_run = {'task_intent': existing,
            'task_plan': {'schema_version': 'cosplay_task_plan.v2',
                'image_budget': {'regional_refinement': 1}}}
        from argparse import Namespace
        assert workflow.task_image_budget(region_run, Namespace(), 'regional_refinement') == 1
        assert workflow.task_image_budget(region_run, Namespace(), 'initial_reference') == 0
        assert workflow.task_image_budget(region_run, Namespace(final_repair_only=True), 'regional_refinement') == 0
        region_run['task_intent']['regional_refinement']['maximum_submissions'] = 0
        assert workflow.task_image_budget(region_run, Namespace(), 'regional_refinement') == 0
        args = workflow.parse_args(['--root',str(root),*argv[1:]])
        workflow.apply_workflow_settings(args,settings)
        contract = route_contract(vars(args))
        assert contract['creative_direction_id'] == checked['creative_direction']['direction_id']
        observed=[]
        def producer(_root,run,supplied):
            observed.append(supplied.prompt)
            assert run['creative_direction'] == checked['creative_direction']
            assert run['options']['input_prompt'] == body['input_prompt']
            assert run['generation_design']['design_origin'] == 'checked_local_creative_direction'
            raise RuntimeError('CPU fixture stops at actual model boundary')
        with patch('runtime_preflight.preflight_for_run',return_value={'status':'ready'}), \
                patch.object(workflow,'generate_candidate',side_effect=producer), \
                patch('beautygrpo_prepare.prepare_cached_for_workbench',return_value={
                    'ok':True,'no_change':True,'result':None,'layer':None,
                    'generation_inferences_this_attempt':0,'cache_hit':True}), \
                patch('character_profile.analyze_character',side_effect=AssertionError('Unexpected external planning')):
            saved = workflow.create_run(args)
        assert observed == [checked['creative_direction']['prompt']]
        assert saved['status'] in {'blocked','failed'}
        assert saved['result_publication']['state']=='pending'
        assert saved['generation_budget']['image_submissions']==1
        copied=copy.deepcopy(saved)
        copied['task_plan']['image_budget']['regional_refinement']=1
        copied['options']['appearance_revision']={'design_reconstruction_plan':{'provenance':{
            'source_scene_enhancement':{'new_model_submissions':1,'provider_ids':['actual-one'],
                'reused_saved_regions':0}}}}
        workflow._record_regional_execution(copied)
        workflow._record_regional_execution(copied)
        assert copied['generation_budget']['regional_refinement_image_submissions']==1
        copied['options']['appearance_revision']['design_reconstruction_plan']['provenance']['source_scene_enhancement']={
            'new_model_submissions':2,'provider_ids':['other','third']}
        try: workflow._record_regional_execution(copied)
        except ValueError: pass
        else: raise AssertionError('Excess regional submissions accepted')
        reservation=root/'outputs/regional-reservations'/saved['run_id']/'regional-refinement.json'
        reservation.parent.mkdir(parents=True)
        copied['generation_budget'].update(origin_run_id=saved['run_id'],regional_refinement_image_budget=1)
        reservation.write_text(json.dumps({'schema':'regional_scene_refinement.v1',
            'state':'unknown','request_started':True,'model_submissions':0}), 'utf8')
        workflow._record_regional_reservation(root,copied)
        assert copied['generation_budget']['regional_submission_count_uncertain']
        assert copied['generation_budget']['regional_refinement_reserved_allowance']==1
        reservation.write_text(json.dumps({'schema':'regional_scene_refinement.v1',
            'state':'completed','request_started':True,'model_submissions':1,'prompt_id':'actual-one'}), 'utf8')
        workflow._record_regional_reservation(root,copied)
        assert copied['generation_budget']['regional_refinement_image_submissions']==1
        assert not copied['generation_budget']['regional_submission_count_uncertain']
    print(json.dumps({'ok':True,'HTTP_CLI_actual_producer_prompt_bound':True,
        'no_external_planning':True,'failed_generation_not_published':True,
        'regional_count_idempotent_and_bounded':True,'actual_model_submissions':0}))


if __name__=='__main__':
    main()
