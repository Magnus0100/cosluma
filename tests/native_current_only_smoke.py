"""Source-only display once, current ROI/reopen and portable format boundaries."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from PIL import Image
from current_stack_fixture import prepare_fixture
from design_reconstruction import plan_for_run
from display_transform import DEFAULT,display_linear,source_transform_for_precision,plan_for_run as display_plan
from export_artifact import export_source_run
from float_appearance import render_source_float,render_source_only_float
from live_native_detail import live_native_detail
from native_detail import saved_native_detail
from native_roi_render import render_native_roi
from native_source_baseline import prepare_source_only,source_for_plan
from portable_project import export_project,import_project
from raw_support import RGB_DECODE_VERSION,deterministic_srgb_profile,linear_preview,save_render_master,save_scene_master
from source_preview import SourcePreviewCache


def save(root,run,pixels,operator):
    directory=root/'outputs/runs'/run['run_id'];directory.mkdir(parents=True,exist_ok=True)
    draft=directory/'auto_draft.png';master=directory/'render_master.tif';scene=directory/'render_scene.tif'
    image=linear_preview(pixels,display_transform=display_plan(run));image.save(draft,icc_profile=image.info['icc_profile'])
    save_render_master(pixels,master,display_transform=display_plan(run));save_scene_master(pixels,scene,working_domain='display_linear_sRGB')
    run['artifacts']={'auto_draft':str(draft),'render_master':str(master),'render_scene_master':str(scene)}
    run['render']={'run_id':run['run_id'],'revision':run['revision'],'operator':operator,
        'render_precision':{'working_domain':'display_linear_sRGB','working_precision':'float32_linear','master_encoding':'uint16_srgb'}}
    manifest=directory/'manifest.json';manifest.write_text(json.dumps(run),'utf8');return manifest


def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='native-current-') as temporary:
        root=Path(temporary);source=root/'source.png';Image.new('RGB',(32,24),(210,180,130)).save(source)
        run={'kind':'single_input_run','input_mode':'single_source','run_id':'source-only','sample':'source-only','revision':1,'root':str(root),
            'source':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'oriented_size':[32,24]},'reference':None,
            'options':{'render_mode':'native_source_design'}}
        linear=np.full((24,32,3),(.97,.88,.84),np.float32)
        for precision in ('raw16_linear','raster8_srgb'):
            local=copy.deepcopy(run)
            values,plan=prepare_source_only(linear,local,{'source_precision':precision,'decode_version':RGB_DECODE_VERSION})
            local['options']['source_only_plan']=plan
            result,operator=render_source_only_float(values,local,Image.new('L',(32,24),255))
            transform = source_transform_for_precision(precision)
            np.testing.assert_array_equal(result,display_linear(linear,transform))
            if precision == 'raw16_linear':
                assert not np.array_equal(result,display_linear(result,transform))
            else:
                np.testing.assert_array_equal(result,linear)
            assert operator['source_display_applied_once'] and not operator['candidate_pixels_copied']
            if precision=='raster8_srgb':run=local
            checks.append(precision+'_display_once')
        for label,mutation in (
            ('reference_contamination',lambda r:r.update(reference={'sha256':'other'})),
            ('missing_decoder',lambda r:r['options']['source_only_plan'].pop('decode_version')),
            ('wrong_source_identity',lambda r:r['options']['source_only_plan'].update(source_sha256='other'))):
            changed=copy.deepcopy(run);mutation(changed)
            try:render_source_only_float(values,changed,Image.new('L',(32,24),255))
            except ValueError:pass
            else:raise AssertionError(label+' accepted')
            checks.append(label)
        manifest=save(root,run,result,operator)
        from pipeline_stage_report import read_stage_report
        state=read_stage_report(root,run,manifest)
        assert state['draft_state']=='local_source_draft' and not state['reconstruction_required']
        assert not state['visual_approved'] and state['whole_work']['reference_colour_match_applied'] is False
        assert next(row for row in state['stages'] if row['id']=='save_and_export')['evidence']['native_masters_present']
        checks.append('source_only_progress_and_export_without_reference')
        saved=saved_native_detail(root,run['run_id'],{'edge':128})
        assert saved['working_domain']=='display_linear_sRGB' and not saved['resampled']
        preview=SourcePreviewCache(root,max_edge=16).preview(run['run_id'],run,manifest,{'appearance_revision':{'params':{'exposure':.3}}})
        assert preview['working_domain']=='display_linear_sRGB' and preview['preview_size']==[16,12]
        checks.append('source_only_saved_detail_and_edited_proxy')
        export=export_source_run(root,run['run_id'],directory=str(root/'exports'),format='TIFF16')
        assert Path(export['path']).read_bytes()==Path(run['artifacts']['render_master']).read_bytes()
        archive=root/'current.cosplay';export_project(root,run['run_id'],archive)
        imported=import_project(root/'moved',archive,run_id='moved-source')['run']
        assert imported['options']['source_only_plan']==run['options']['source_only_plan']
        assert Path(imported['artifacts']['render_master']).read_bytes()==Path(run['artifacts']['render_master']).read_bytes()
        checks.append('actual_source_only_export_and_portable_v2')
        old=root/'old.cosplay'
        with zipfile.ZipFile(archive) as original,zipfile.ZipFile(old,'w') as output:
            for name in original.namelist():
                data=original.read(name)
                if name=='project.json':
                    descriptor=json.loads(data);descriptor['format']='cosplay.portable-project.v1';data=json.dumps(descriptor).encode()
                output.writestr(name,data)
        try:import_project(root/'refused',old)
        except ValueError:pass
        else:raise AssertionError('old portable format accepted')
        assert not (root/'refused/outputs/runs').exists();checks.append('old_portable_rejected_before_extraction')
        paired=prepare_fixture(root/'paired',native=(32,24));plan=plan_for_run(paired)
        paired['options']['appearance_revision']['regions']={'subject':{'exposure':.2},'global':{'contrast':.1}}
        source_values=source_for_plan(plan);full,op=render_source_float(source_values,paired)
        crop,_=render_native_roi(paired,(5,4,22,19));np.testing.assert_array_equal(crop,full[4:19,5:22])
        paired_manifest=save(root,paired,full,op)
        saved=saved_native_detail(root,paired['run_id'],{'edge':128});live=live_native_detail(root,paired['run_id'],{'edge':128})
        with Image.open(saved['preview_path']) as a,Image.open(live['preview_path']) as b:assert a.tobytes()==b.tobytes()
        plan['enabled']=False
        off,_=render_source_float(source_values,paired)
        assert not np.array_equal(off,full)
        checks.append('current_paired_full_roi_saved_live_and_design_toggle')
        profile=deterministic_srgb_profile()
        assert profile[24:36]==bytes.fromhex('07d000010001000000000000') and profile[84:100]==bytes(16)
        a=linear_preview(linear,display_transform={'method':'clip_sdr'});b=linear_preview(linear,display_transform={'method':'clip_sdr'})
        a.save(root/'a.png',icc_profile=a.info['icc_profile']);b.save(root/'b.png',icc_profile=b.info['icc_profile'])
        assert (root/'a.png').read_bytes()==(root/'b.png').read_bytes()
        checks.append('canonical_icc_source_bytes')
    print(json.dumps({'ok':True,'checks':checks,'api_calls':0,'gpu_calls':0}))


if __name__=='__main__':main()
