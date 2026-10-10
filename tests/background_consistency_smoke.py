"""CPU numeric regression: observed support, improvement, exact exclusions."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
from PIL import Image
from background_consistency import prepare_consistent_background
from scene_stage_audit import audit_scene_stages, array, file_record, write


def fixture(root, drift=True):
    root=Path(root); root.mkdir(parents=True)
    compact=(96,128); native=(192,256)
    yy,xx=np.mgrid[:compact[1],:compact[0]]
    image=np.uint8(np.clip(95+17*np.sin(xx/13)+8*np.cos(yy/17),0,255))
    image=np.repeat(image[...,None],3,-1)
    reference=root/'reference.png'; Image.fromarray(image).save(reference)
    source=root/'source.png'; Image.new('RGB',native,(44,57,68)).save(source)
    clean=root/'clean.png'; Image.fromarray(image).save(clean)
    removal=np.zeros(image.shape[:2],np.uint8); removal[36:76,30:55]=255
    mask=root/'removal.png'; Image.fromarray(removal).save(mask)
    srgb=image.astype(np.float32)/255
    observed=np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
    p=cv2.resize(observed,native,interpolation=cv2.INTER_NEAREST)
    if drift:p+=.012
    alpha=np.zeros(native[::-1],np.float32); alpha[90:165,73:112]=1
    v=np.ones(native[::-1],np.float32); v[:12]=.25; p*=v[...,None]
    def asset(name,values):
        path=root/(name+'.npz'); np.savez_compressed(path,values=values)
        return file_record(path,size=list(values.shape[1::-1]))
    alpha_asset=asset('alpha',alpha); pv=asset('validity',v); pp=asset('background',p)
    op=asset('observation',observed); ov=asset('observation-validity',np.ones(compact[::-1],np.float32))
    receiver_support=asset('receiver-support',np.ones((20,24),np.float32)); receiver_support['native_box']=[135,212,159,232]
    receiver_path=root/'receiver.json'; write(receiver_path,{'state':'prepared','support_asset':receiver_support,
        'input_background':pp,'output_background':pp})
    clean_npz=asset('clean-values',observed); manifest=root/'clean-manifest.json'
    write(manifest,{'input':str(reference),'clean_png':str(clean),'removal_mask_input':str(mask)})
    stage=root/'clean-binding.json'; write(stage,{'schema_version':'design_background_stage_cache.v1',
        'successful_worker_returncode':0,'request_identity':{
            'input_sha256':file_record(reference)['sha256'],'removal_mask_sha256':file_record(mask)['sha256'],
            'output_contract':'preserve_candidate_RGB_exactly_outside_supplied_mask'},
        'result_manifest':file_record(manifest),'outputs':{'clean_png':file_record(clean),
            'clean_npz':clean_npz,'removal_mask':file_record(mask)}})
    sr={'schema_version':'background_superresolution.v1','assets':{'background':pp,'validity':pv},
        'source_assets':{'background':op,'validity':ov}}
    src=file_record(source); ref=file_record(reference)
    frozen=root/'observation-receipt.json';write(frozen,{'schema':'background_observation.v1',
        'source_sha256':src['sha256'],'reference_sha256':ref['sha256'],'sr_provenance':sr,
        'clean_stage_binding':file_record(stage),'clean_input_asset':file_record(reference)})
    plan={'working_domain':'display_linear_sRGB','source_sha256':src['sha256'],'candidate_sha256':ref['sha256'],
        'canvas':{'source_oriented_size':list(native),'source_proxy_size':list(compact),
            'candidate_mapping':{'scene_registration_supported':True,
                'candidate_processing_proxy_to_final_proxy':np.eye(3).tolist()}},
        'mask_roles':{'foreground_alpha':alpha_asset,
            'candidate_removal':file_record(mask,size=list(compact),coordinate_space='candidate_proxy')},
        'layers':[{'kind':'background_design','enabled':True,'strength':1,'assets':{
            'background':pp,'validity':pv,'compact_observation':op,'compact_validity':ov,
            'background_observation':file_record(frozen),
            'native_receiver_preservation':file_record(receiver_path)}}],
        'provenance':{'background_native_detail':sr,'background_stage_cache':{'binding':str(stage)}},
        'scene_design':{'light':{}}}
    run={'run_id':'synthetic-current','revision':1,'source':src,'reference':ref,
        'options':{'appearance_revision':{'design_reconstruction_plan':plan}}}
    return run,plan


def main():
    cases=[]
    with tempfile.TemporaryDirectory(prefix='scene-consistency-current-') as folder:
        root=Path(folder);run,plan=fixture(root/'input')
        original=array(plan['layers'][0]['assets']['background'])
        audit=audit_scene_stages(root,run,root/'audit')
        assert audit['status']=='repair_eligible',audit
        result=prepare_consistent_background(root,run,plan,root/'result',audit=audit)
        assert result['changed'] and result['receipt']['relative_RMSE_improvement']>.3,result
        current=array(result['design_reconstruction_plan']['layers'][0]['assets']['background'])
        assert not np.array_equal(current,original)
        for key in ('person_coverage_exact','receiver_native_box_exact','outside_qualified_support_exact','invalid_sampling_exact'):
            assert result['receipt'][key],key
        assert np.max(abs(current-original))<=.0250001
        assert array(plan['layers'][0]['assets']['background']).tobytes()==original.tobytes()
        cases.extend(['drift_improved','finite_bounded','all_exclusion_identity','parent_immutable'])
        cacheless=copy.deepcopy(run);cacheless['options']['appearance_revision']['design_reconstruction_plan']['provenance']={}
        assert audit_scene_stages(root,cacheless,root/'cacheless-audit')['status']=='repair_eligible'
        cases.append('owned_observation_without_disposable_cache')
        child=copy.deepcopy(run);child['options']['appearance_revision']['design_reconstruction_plan']=result['design_reconstruction_plan']
        audited=audit_scene_stages(root,child,root/'child-audit')
        assert audited['status']!='unsupported',audited
        cases.append('owned_consistency_ancestry_reopens')
        stale=copy.deepcopy(audit); stale['source_sha256']='stale'
        try:prepare_consistent_background(root,run,plan,root/'stale',audit=stale)
        except ValueError:pass
        else:raise AssertionError('Tampered audit accepted')
        cases.append('tampered_audit_rejected')
        changed=copy.deepcopy(run);changed['options']['appearance_revision']['design_reconstruction_plan']['canvas']['candidate_mapping']['scene_registration_supported']=False
        assert audit_scene_stages(root,changed,root/'unregistered')['status']=='unsupported'
        cases.append('unregistered_no_repair')
        clean=np.asarray(Image.open(root/'input/clean.png')).copy();clean[0,0]=255
        Image.fromarray(clean).save(root/'input/clean.png')
        assert audit_scene_stages(root,run,root/'clean-tamper')['status']=='unsupported'
        cases.append('changed_clean_no_repair')
        run0,plan0=fixture(root/'undrifted',drift=False)
        audit0=audit_scene_stages(root,run0,root/'undrifted-audit')
        assert audit0['status']=='audited_no_repair',audit0
        assert not prepare_consistent_background(root,run0,plan0,root/'unchanged',audit=audit0)['changed']
        cases.append('no_drift_no_writes')
    print(json.dumps({'status':'passed','cases':cases,'API_requests':0,'model_inference_requests':0}))


if __name__=='__main__':main()
