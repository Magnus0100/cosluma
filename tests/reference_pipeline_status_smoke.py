"""Actual current frozen v2 status, stale consumption and retired rejection."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
from current_stack_fixture import prepare_fixture
from design_reconstruction import plan_for_run
from float_appearance import render_source_float
from native_source_baseline import source_for_plan
from pipeline_stage_report import VISUAL_CHECK_KEYS,build_stage_report,read_stage_report
from raw_support import linear_preview,save_render_master,save_scene_master
from source_appearance_responsibility import prepare


def main():
    checks=[]
    with TemporaryDirectory(prefix='current-status-') as directory:
        root=Path(directory);owned=root/'outputs/runs/current-stack-fixture'
        run=prepare_fixture(owned)
        plan=plan_for_run(run)
        values,operator=render_source_float(source_for_plan(plan),run)
        preview=linear_preview(values,display_transform=operator['display_transform'])
        preview_path=owned/'draft.png';preview.save(preview_path,icc_profile=preview.info['icc_profile'])
        master=owned/'master.tif';scene=owned/'scene.tif';alpha=owned/'alpha.png'
        save_render_master(values,master,display_transform=operator['display_transform'])
        save_scene_master(values,scene,working_domain='display_linear_sRGB')
        Image.new('L',tuple(plan['canvas']['source_oriented_size']),255).save(alpha)
        manifest=owned/'manifest.json';run['manifest']=str(manifest)
        run['status']='needs_visual_review';run['render']={'run_id':run['run_id'],'revision':run['revision'],'operator':operator}
        run['artifacts']={'auto_draft':str(preview_path),'render_master':str(master),'render_scene_master':str(scene),'alpha':str(alpha)}
        manifest.write_text(json.dumps(run),'utf8')
        report=read_stage_report(root,run)
        assert report['reference_appearance_stack']['prepared'] and report['reference_appearance_stack']['applied'],report
        assert report['reference_appearance_stack']['independent_controls']
        assert report['draft_state']=='usable_draft' and not report['visual_approved']
        assert not report['assets_rehashed'] and not report['whole_work']['complete_automatic_pipeline_claim']
        descriptor=prepare(run,plan);plan['source_appearance_responsibility']=descriptor
        checks.append('actual_v2_render_consumption_and_metadata_status')
        for label,mutation in (
            ('missing_render',lambda r:r.pop('render')),
            ('wrong_render_revision',lambda r:r['render'].update(revision=2)),
            ('wrong_render_run',lambda r:r['render'].update(run_id='other')),
            ('pending_edit',lambda r:r.update(continuation_edit={'render_state':'pending'})),
            ('wrong_delta_consumption',lambda r:r['render']['operator']['design_reconstruction_plan']['layers'][0].update(consumed_asset_sha256='other')),
            ('wrong_control_consumption',lambda r:r['render']['operator']['design_reconstruction_plan']['layers'][0]['consumed_control_assets_sha256'].update(control_receipt='other'))):
            changed=copy.deepcopy(run);mutation(changed)
            assert not read_stage_report(root,changed)['reference_appearance_stack']['applied'],label
            checks.append(label)
        for label,mutation in (
            ('v1_rejected',lambda p:p['layers'][0]['params'].update(schema='reference_appearance_stack.v1')),
            ('wrong_domain',lambda p:p['layers'][0]['params'].update(working_domain='scene_linear_sRGB')),
            ('wrong_source',lambda p:p['layers'][0]['params'].update(source_sha256='other')),
            ('missing_control_identity',lambda p:p['layers'][0]['params'].pop('control_receipt_identity_sha256')),
            ('missing_support',lambda p:p['layers'][0]['params'].pop('reference_support'))):
            changed=copy.deepcopy(run);mutation(plan_for_run(changed))
            result=read_stage_report(root,changed)
            assert result['reconstruction_required'] and not result['reference_appearance_stack']['applied'],label
            checks.append(label)
        control=Path(plan['layers'][0]['assets']['control_receipt']['path']);saved=control.read_bytes();control.unlink()
        assert read_stage_report(root,run)['reconstruction_required'];control.write_bytes(saved)
        checks.append('missing_real_control_file')
        zero=copy.deepcopy(run);plan_for_run(zero)['layers'][0].update(colour_strength=0.,lightness_strength=0.)
        zero['render']['operator']['design_reconstruction_plan']['layers'][0].update(colour_strength=0.,lightness_strength=0.)
        result=read_stage_report(root,zero)
        assert result['reference_appearance_stack']['prepared'] and not result['reference_appearance_stack']['applied']
        assert not result['reconstruction_required'];checks.append('zero_current_controls_are_valid')
        observations={'assets':{key:True for key in ('source','reference','draft','master','scene_master','alpha','manifest','background','background_validity','reference_appearance_delta','reference_appearance_colour_delta','reference_appearance_lightness_delta','reference_appearance_control_receipt')},
            'review':{'run_id':run['run_id'],'revision':run['revision'],'status':'accepted','checks':{key:'pass' for key in VISUAL_CHECK_KEYS}}}
        accepted=copy.deepcopy(run);accepted['status']='accepted'
        assert build_stage_report(accepted,observations)['visual_approved']
        observations['review']['revision']=99
        assert not build_stage_report(accepted,observations)['visual_approved'];checks.append('explicit_current_visual_acceptance_only')
    print(json.dumps({'ok':True,'checks':checks,'actual_current_assets':True,'gpu_calls':0,'api_calls':0}))


if __name__=='__main__':main()
