"""Current exact retry preserves the sent prompt and refuses retired contracts."""
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import single_source_workflow as workflow
from backend_server import Backend
from generation_design import DESIGN_VERSION, prepare_generation_design
from generation_design_smoke import plan_fixture


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay-current-retry-') as folder:
        root=Path(folder);source=root/'source.png'
        Image.new('RGB',(24,32),'#806050').save(source)
        with patch('vision_transport.request_vision_json',return_value=(plan_fixture(dramatic=False),{'fixture':True})):
            design=prepare_generation_design(source,root/'design',{},base_prompt='A photographic cosplay finish.',
                character_research={'status':'unavailable','summary':'visible costume only'})
        design['input_prompt']='A photographic cosplay finish.'
        design['status']='ready'
        exact=design['effective_prompt']+'\nUser final edit: keep the amber sleeve reflection.'
        parent={'kind':'single_input_run','input_mode':'single_source','run_id':'parent','sample':'fixture',
            'source':{'path':str(source)},'generation_design':copy.deepcopy(design),
            'options':{'effective_prompt':exact,'input_prompt':design['input_prompt'],
                'prompt_template_version':'custom','generation_design':copy.deepcopy(design)}}
        directory=root/'outputs/runs/parent';directory.mkdir(parents=True)
        manifest=directory/'manifest.json';manifest.write_text(json.dumps(parent),'utf8');before=manifest.read_bytes()
        backend=Backend(root,embedded=True)
        with patch.object(backend,'one_click',side_effect=lambda body:body):
            request=backend.retry_source_run('parent',{'async':False})
        assert request['effective_prompt']==exact and request['preserve_effective_prompt'] is True
        assert request['generation_design']['schema_version']==DESIGN_VERSION
        submitted=[]
        def generate(_root,run,args):
            submitted.append(args.prompt)
            candidate=args.candidate_out_dir/'candidate.png';candidate.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(24,32),'#807060').save(candidate);return candidate
        args=workflow.parse_args(['--root',str(root),'--source',str(source),'--generate','--run-id','retry'])
        for key,value in request.items():setattr(args,key,value)
        args.source=Path(args.source)
        args.run_id='retry'
        with patch.object(workflow,'generate_candidate',side_effect=generate), \
             patch.object(workflow,'diagnose_cosplay_post',return_value=None), \
             patch.object(workflow,'render_candidate',return_value={'outputs':{},'metadata':{},'review':{}}), \
             patch('character_profile.analyze_character',side_effect=AssertionError('exact retry replanned character')), \
             patch('generation_design.prepare_generation_design',side_effect=AssertionError('exact retry replanned design')):
            child=workflow.create_run(args)
            assert submitted==[exact] and child['generation_design']['effective_prompt']==exact
            args.run_id='retired';args.generation_design={**design,'schema_version':'retired-design'}
            try:workflow.create_run(args)
            except ValueError as error:assert 'complete current generation design' in str(error)
            else:raise AssertionError('retired design accepted')
            assert submitted==[exact]
        for body in ({'alpha_revision':{}},{'appearance_revision':{'parameters':{}}}):
            try:backend.retry_source_run('parent',body)
            except ValueError:pass
            else:raise AssertionError('retired retry fields accepted')
        assert manifest.read_bytes()==before
    print(json.dumps({'ok':True,'exact_prompt_submitted_once':True,'no_replanning':True,
        'retired_design_rejected_before_submission':True,'parent_unchanged':True,'external_requests':0,
        'provider_and_render_mocked':True}))


if __name__=='__main__':main()
