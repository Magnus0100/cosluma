"""Product-result negative cases on real current frozen assets; no inference."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from current_stack_fixture import prepare_fixture
from design_reconstruction import plan_for_run
from float_appearance import render_source_float
from native_source_baseline import source_for_plan
from pipeline_stage_report import VISUAL_CHECK_KEYS, build_stage_report, read_stage_report, format_stage_report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); checks=[]
    with TemporaryDirectory(prefix='cosplay-product-result-') as folder:
        run=prepare_fixture(Path(folder)); plan=plan_for_run(run)
        _,operator=render_source_float(source_for_plan(plan),run)
        run.update(status='needs_visual_review',render={'run_id':run['run_id'],'revision':run['revision'],'operator':operator})
        run['generation_design']={'design_brief':{'primary_effect_plan':{'mode':'scene_effect','description':'A visible back-scene effect.'}}}
        run['system_post_completion']={'status':'completed_with_findings','stages':[{'id':'independent_effects','status':'skipped'}]}
        assets={key:True for key in ('source','reference','draft','master','scene_master','alpha','manifest','background','background_validity','reference_appearance_delta','reference_appearance_colour_delta','reference_appearance_lightness_delta','reference_appearance_control_receipt')}
        observations={'assets':assets,'review':{}}
        hashes={'auto_draft':'draft-sha','render_master':'master-sha','render_scene_master':'scene-sha'}
        run['cosplay_post_review']={'status':'reviewed','actual_api_evidence_valid':True,
            'run_binding':{'run_id':run['run_id'],'revision':run['revision'],
                'source_sha256':run['source']['sha256'],'reference_sha256':run['reference']['sha256'],
                'final_sha256':hashes['auto_draft'],'render_artifacts':hashes},
            'checks':{key:{'final':{'status':'pass','reason':'Bound observed evidence.'}} for key in
                ('identity_makeup','concept_hierarchy','effects_space','skin_materials','native_detail','scene_integration')}}
        def report(value): return build_stage_report(value,observations)
        result=report(run)
        assert result['whole_work']['result_state']=='needs_visual_review'
        assert result['whole_work']['required_effects']['fulfillment']=='fulfilled'
        assert not result['visual_approved'];checks.append('observed_pass_is_not_human_acceptance')
        failed=copy.deepcopy(run)
        failed['cosplay_post_review']['checks']['effects_space']['final'].update(status='fail',cause='transfer_incomplete',correction_stage='scene_composition')
        result=report(failed)
        assert result['whole_work']['result_state']=='needs_repair'
        assert result['whole_work']['required_effects']['fulfillment']=='failed'
        assert result['whole_work']['required_effects']['independent_effect_status']=='skipped'
        action=next(row for row in result['whole_work']['recovery_actions'] if row['action_id']=='repair_final_locally')
        assert action['supported'] and action['QA_required'] and not action['generation_required']
        assert not result['cosplay_post_diagnosis']['current_repair_authorized'];checks.append('skipped_does_not_satisfy_required_failed_effect')
        owned=copy.deepcopy(run)
        for stage in ('native_retouch','subject_appearance','reference_generation','unsupported_layering'):
            for cause in ('transfer_incomplete','composition_defect'):
                owned['cosplay_post_review']['checks']['skin_materials']['final'].update(
                    status='fail',cause=cause,correction_stage=stage)
                saved_review=copy.deepcopy(owned['cosplay_post_review'])
                result=report(owned)
                finding=next(row for row in result['whole_work']['open_findings'] if row['criterion']=='skin_materials')
                assert finding['binding_state']=='current' and finding['owner']==stage, (stage,cause,finding)
                assert owned['cosplay_post_review']==saved_review
                assert not result['cosplay_post_diagnosis']['current_repair_authorized']
        checks.append('current_explicit_owner_beats_generic_scene_cause_without_rewriting_review')
        for stage in (None,'unrecognized_stage'):
            fallback=copy.deepcopy(run)
            fallback['cosplay_post_review']['checks']['skin_materials']['final'].update(
                status='fail',cause='transfer_incomplete',correction_stage=stage)
            finding=next(row for row in report(fallback)['whole_work']['open_findings'] if row['criterion']=='skin_materials')
            assert finding['owner']=='scene_composition'
        checks.append('missing_or_unrecognized_owner_keeps_existing_scene_fallback')
        unknown=copy.deepcopy(run);unknown['cosplay_post_review']['checks']['identity_makeup']['final']['status']='unknown'
        assert report(unknown)['whole_work']['result_state']=='needs_evidence';checks.append('identity_unknown_requires_evidence')
        inherited=copy.deepcopy(failed)
        parent_id=inherited['run_id'];inherited['run_id']='copied-child';inherited['revision']+=1
        inherited['render'].update(run_id=inherited['run_id'],revision=inherited['revision'])
        receipt={'schema':'saved_render_copy.v1','copied_from_run_id':parent_id,'copied_from_manifest_sha256':'parent-sha',
            'rendered_run_id':parent_id,'render_executed':False,'copied_master_sha256':hashes}
        inherited['saved_render_copy']=copy.deepcopy(receipt);inherited['render']['saved_render_copy']=copy.deepcopy(receipt)
        result=report(inherited)
        assert result['cosplay_post_diagnosis']['binding_state']=='inherited_same_pixels'
        assert result['whole_work']['result_state']=='needs_evidence'
        assert any(row['criterion']=='effects_space' and row['status']=='fail' for row in result['whole_work']['open_findings'])
        assert not result['cosplay_post_diagnosis']['current_render_binding_matches']
        assert not result['cosplay_post_diagnosis']['current_repair_authorized']
        assert '继承' in format_stage_report(result);checks.append('copy_retains_findings_without_repair_authority')
        historical=copy.deepcopy(inherited)
        historical['cosplay_post_review']['checks']['skin_materials']['final'].update(
            status='fail',cause='transfer_incomplete',correction_stage='native_retouch')
        historical_review=copy.deepcopy(historical['cosplay_post_review'])
        finding=next(row for row in report(historical)['whole_work']['open_findings'] if row['criterion']=='skin_materials')
        assert finding['binding_state']=='inherited_same_pixels' and finding['owner']=='scene_composition'
        assert historical['cosplay_post_review']==historical_review
        checks.append('historical_owner_display_and_saved_review_not_relabelled')
        inherited['saved_render_copy']['copied_master_sha256']['auto_draft']='changed'
        assert report(inherited)['cosplay_post_diagnosis']['binding_state']=='stale';checks.append('tampered_copy_proof_is_stale')
        stale=copy.deepcopy(run);stale['cosplay_post_review']['run_binding']['source_sha256']='other'
        assert report(stale)['whole_work']['result_state']=='needs_evidence';checks.append('stale_source_cannot_fulfill_effect')
        invalid=copy.deepcopy(run);invalid['cosplay_post_review']['actual_api_evidence_valid']=False
        assert report(invalid)['whole_work']['required_effects']['fulfillment']=='unknown';checks.append('invalid_actual_qa_is_not_fulfillment')
        none=copy.deepcopy(run);none['generation_design']['design_brief']['primary_effect_plan']['mode']='none'
        assert report(none)['whole_work']['required_effects']['fulfillment']=='not_required';checks.append('explicit_no_effects_needs_no_added_effect')
        dirty=copy.deepcopy(run);dirty['continuation_edit']={'render_state':'pending'}
        assert report(dirty)['whole_work']['result_state']=='pending_save';checks.append('pending_edit_precedes_visual_findings')
        blocked=copy.deepcopy(run);blocked['status']='blocked'
        result=report(blocked)
        assert result['whole_work']['result_state']=='needs_local_recovery'
        assert any(row['action_id']=='retry_local_processing' and not row['generation_required'] for row in result['whole_work']['recovery_actions'])
        assert not any(row['action_id']=='repair_final_locally' and row['supported'] for row in result['whole_work']['recovery_actions']);checks.append('failed_local_render_has_safe_retry')
        accepted=copy.deepcopy(run);accepted['status']='accepted'
        observations['review']={'run_id':run['run_id'],'revision':run['revision'],'status':'accepted','checks':{key:'pass' for key in VISUAL_CHECK_KEYS}}
        assert report(accepted)['whole_work']['result_state']=='accepted';checks.append('current_explicit_human_gate')
    output={'ok':True,'checks':checks,'gpu_calls':0,'image_requests':0,'qa_requests':0}
    if args.root and args.manifest:
        root=args.root.resolve();manifest=args.manifest.resolve();real=json.loads(manifest.read_text('utf8'))
        protected=[root/'project.json',manifest,*[Path(real['artifacts'][key]) for key in ('auto_draft','render_master','render_scene_master')]]
        before={str(path):hashlib.file_digest(path.open('rb'),'sha256').hexdigest() for path in protected}
        actual=read_stage_report(root,real,manifest)
        assert actual['whole_work']['required_effects']['required'] is True
        assert actual['cosplay_post_diagnosis']['binding_state']=='inherited_same_pixels'
        assert any(row['criterion']=='effects_space' and row['status']=='fail' for row in actual['whole_work']['open_findings'])
        assert not actual['cosplay_post_diagnosis']['current_repair_authorized']
        after={str(path):hashlib.file_digest(path.open('rb'),'sha256').hexdigest() for path in protected}
        assert before==after
        output.update(actual_run_id=real['run_id'],actual_stage_report=actual,protected_sha256=before,protected_unchanged=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps({key:value for key,value in output.items() if key not in {'actual_stage_report','protected_sha256'}},ensure_ascii=False))


if __name__=='__main__':main()
