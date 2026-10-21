"""Offline coordinator checks: reject before composition, bounded fresh children."""
from __future__ import annotations

import copy
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import cosplay_post_completion as completion
import single_source_workflow as workflow
import project_cli
from backend_server import Backend
from pipeline_stage_report import format_stage_report
from generation_design import DESIGN_VERSION


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay-completion-smoke-') as folder:
        root=Path(folder)
        source=root/'original.png'; reference=root/'reference.png'; candidate=root/'candidate.png'
        for path,colour in ((source,'#906050'),(reference,'#806040'),(candidate,'#a08060')):
            Image.new('RGB',(24,32),colour).save(path)
        project=root/'project.json'; project.write_text('{"selected":"original"}',encoding='utf-8')
        directory=root/workflow.RUNS_DIR/'initial'; directory.mkdir(parents=True)
        artifacts={}
        for key,suffix in (('auto_draft','.png'),('render_master','.tif'),('render_scene_master','.tif')):
            path=directory/(key+suffix); Image.new('RGB',(24,32),'#806040').save(path)
            artifacts[key]=str(path)
        run={'run_id':'initial','revision':4,'sample':'source','status':'needs_visual_review',
             'source':{'path':str(source),'sha256':workflow.sha256(source)},
             'reference':{'path':str(reference),'sha256':workflow.sha256(reference)},
             'artifacts':artifacts,'generation_design':{'schema_version':DESIGN_VERSION,'status':'ready',
                'effective_prompt':'Full complete scene design'},
             'generation_budget':{'max_image_submissions':2,'image_submissions':1},
             'cosplay_post_review':{'status':'reviewed','binding':{'images':{
                'source':{'path':str(source)},'reference':{'path':str(reference)}}},
                'checks':{'effects_space':{'reference':{'status':'fail',
                    'correction_stage':'reference_generation'}}}}}
        manifest=directory/'manifest.json'; workflow.write_json(manifest,run)
        args=workflow.parse_args(['--root',str(root),'--source',str(source),'--generate',
                                 '--cosplay-correction-budget','1'])
        args.character_profile={'provenance':{'model':'offline-vision'}}
        args.appearance_revision={'design_reconstruction_plan':{'stale_parent_stack':True},
                                  'appearance_controls':{'colour':.8}}
        args.reference_pipeline_inputs={'stale':'parent'}
        provenance=root/'candidate.provenance.json'
        provenance.write_text(json.dumps({'prompt':'Correct diagnosed skin materials'}),encoding='utf-8')
        result={'status':'generated','generation_attempts':1,'candidate':str(candidate),
                'provenance':str(provenance),'generation_evidence':{'response_id':'offline-fixture'},
                'binding':{'original_design_sha256':'fixture-original-design'}}
        original_assets={str(p):workflow.sha256(Path(p)) for p in
                         (source,reference,project,*artifacts.values())}

        def run_case(local_run=None,local_args=None,*,eligible=False,outcome=None,
                     child_status='needs_visual_review',parent_artifacts=False,wrong_pair=False):
            selected=copy.deepcopy(local_run or run); settings=copy.deepcopy(local_args or args)
            workflow.write_json(manifest,selected)
            children=[]
            def child_creator(child_args):
                children.append(copy.deepcopy(child_args))
                assert not child_args.generate and child_args.cosplay_correction_budget==0
                assert child_args.cosplay_review_only and child_args.preserve_project_selection
                assert child_args.source==source and child_args.reference==candidate
                assert child_args.parent_run_id=='initial' and child_args.run_id!='initial'
                assert not child_args.force
                assert 'design_reconstruction_plan' not in child_args.appearance_revision
                assert child_args.appearance_revision['appearance_controls']=={'colour':.8}
                assert child_args.reference_pipeline_inputs is None
                assert child_args.generation_design == run['generation_design']
                assert child_args.preserve_effective_prompt
                child_directory=root/workflow.RUNS_DIR/child_args.run_id
                child_directory.mkdir(parents=True,exist_ok=True)
                child_artifacts={}
                for key,parent_path in artifacts.items():
                    child_path=child_directory/Path(parent_path).name
                    Image.new('RGB',(24,32),'#a08060').save(child_path)
                    child_artifacts[key]=str(child_path)
                child={'run_id':child_args.run_id,'parent_run_id':'initial','revision':5,
                       'status':child_status,'review':{'status':'not_checked'},
                       'cosplay_post_review':{'status':'reviewed','diagnostic_gate':'fail'},
                       'source':copy.deepcopy(run['source']),
                       'reference':{'path':str(candidate),'sha256':workflow.sha256(candidate)},
                       'options':{'appearance_revision':{'design_reconstruction_plan':{
                            'source_sha256':run['source']['sha256'],
                            'candidate_sha256':workflow.sha256(reference if wrong_pair else candidate)}}},
                       'artifacts':copy.deepcopy(artifacts) if parent_artifacts else child_artifacts}
                workflow.write_json(root/workflow.RUNS_DIR/child['run_id']/'manifest.json',child)
                return child
            with patch('reference_refinement.refine_reference_once',
                       side_effect=outcome if isinstance(outcome,BaseException) else None,
                       return_value=outcome or result) as refine, \
                 patch('reference_refinement.validate_composition_eligibility',return_value={
                    'eligible_for_composition':eligible,'status':'valid' if eligible else 'rejected'}) as gate, \
                 patch('reference_refinement.evaluate_reference_refinement',return_value={'status':'reviewed'}) as evaluation, \
                 patch.object(workflow,'create_run',side_effect=child_creator):
                saved=completion.continue_cosplay_post(root,selected,settings)
            for path,digest in original_assets.items():
                assert workflow.sha256(Path(path))==digest
            return saved,refine.call_count,gate.call_count,children

        rejected,calls,gates,children=run_case()
        assert calls==gates==len(children)==1
        assert rejected['cosplay_post_completion']['status']=='correction_rejected'
        assert rejected['generation_budget']['total_image_submissions']==2
        text=format_stage_report({'cosplay_post_completion':rejected['cosplay_post_completion']})
        assert '未通过检查，已保留原草稿' in text and '纠错提交 1/1' in text
        child,calls,gates,children=run_case(eligible=True)
        assert calls==gates==len(children)==1 and child['run_id']=='initial-material-correction'
        assert child['status']=='needs_visual_review' and child['review']['status']=='not_checked'
        assert not child['cosplay_post_completion']['visual_acceptance']
        assert child['generation_budget']['total_image_submissions']==2
        assert child['generation_budget']['origin_run_id']=='initial'
        assert child['generation_budget']['local_child_image_submissions']==0
        failed,calls,gates,children=run_case(eligible=True,child_status='blocked')
        assert failed['run_id']=='initial' and len(children)==1
        assert failed['cosplay_post_completion']['status']=='corrected_local_processing_failed'
        for options in ({'parent_artifacts':True},{'wrong_pair':True}):
            saved,calls,gates,children=run_case(eligible=True,**options)
            assert saved['run_id']=='initial' and len(children)==1
            assert saved['cosplay_post_completion']['status']=='correction_unavailable'
        for status in ('unavailable','budget_exhausted'):
            saved,calls,gates,children=run_case(outcome={'status':status,'generation_attempts':1})
            assert calls==1 and gates==0 and not children
            assert saved['cosplay_post_completion']['status']=='correction_unavailable'
        for field,value in (('generate',False),('cosplay_correction_budget',0),('preserve_effective_prompt',True)):
            settings=copy.deepcopy(args); setattr(settings,field,value)
            saved,calls,gates,children=run_case(local_args=settings)
            assert calls==gates==0 and not children and 'cosplay_post_completion' not in saved
        only_final=copy.deepcopy(run)
        only_final['cosplay_post_review']['checks']['effects_space']['reference']['correction_stage']='subject_appearance'
        saved,calls,gates,children=run_case(local_run=only_final)
        assert calls==gates==0 and not children
        cancelled=copy.deepcopy(args); marker=root/'cancel.requested'; marker.touch()
        cancelled.cancel_request_path=str(marker)
        saved,calls,gates,children=run_case(local_args=cancelled)
        assert calls==gates==0 and not children
        assert saved['cosplay_post_completion']['status']=='correction_cancelled'
        saved,calls,gates,children=run_case(local_run=rejected)
        assert calls==gates==0 and not children
        assert saved['generation_budget']['correction_image_submissions']==1
        # A lost parent outcome may still have the immutable reserved attempt.
        spent=directory/'reference-material-correction/attempt.json'
        spent.parent.mkdir(exist_ok=True); spent.write_text('{"attempt":1,"budget":1}',encoding='utf-8')
        saved,calls,gates,children=run_case(outcome={'status':'budget_exhausted','generation_attempts':0})
        assert saved['generation_budget']['correction_image_submissions']==1
        assert saved['generation_budget']['total_image_submissions']==2
        saved,calls,gates,children=run_case(outcome=OSError('receipt save failed after submission'))
        assert calls==1 and gates==0 and not children
        assert saved['generation_budget']['correction_image_submissions']==1
        assert saved['generation_budget']['total_image_submissions']==2
        # Exercise the public product entry and Backend argv serialization.
        backend=Backend(root,embedded=True)
        for body,expected in (({'source':str(source)},1),
                              ({'source':str(source),'cosplay_correction_budget':0},0),
                              ({'source':str(source),'reference':str(reference),'generate':False},0),
                              ({'source':str(source),'local_only':True},0),
                              ({'source':str(source),'preserve_effective_prompt':True,
                                'effective_prompt':'Exact saved brief'},0)):
            argv=backend._one_click_args(body,str(source))
            parsed=project_cli.build_parser().parse_args(argv)
            submitted=[]
            def capture(settings):
                submitted.append(settings)
                return {'status':'needs_visual_review', 'run_id':'product-entry-fixture', 'revision':1, 'options':{}}
            with patch.object(workflow,'create_run',side_effect=capture), contextlib.redirect_stdout(io.StringIO()):
                assert project_cli.one_click_cmd(parsed,root)==0
            assert getattr(submitted[0],'cosplay_correction_budget',0)==expected
            # Exact retries carry zero correction budget through the product
            # entry as well as the coordinator's independent exclusion.
            assert backend._one_click_payload(body,str(source)).get('cosplay_correction_budget')==body.get('cosplay_correction_budget')
    print(json.dumps({'ok':True,'generation_requests':0,'network_requests':0,
                      'rejects_before_composition':True,'fresh_bound_child':True,
                      'source_and_masters_and_project_preserved':True}))


if __name__=='__main__':
    main()
