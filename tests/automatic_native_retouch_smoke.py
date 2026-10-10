"""Real loopback approvals, native healing, paired-source replay and ownership."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import automatic_native_retouch as automatic
import native_retouch_producer as producer
from native_retouch_producer_smoke import fixture, Loopback
from native_source_baseline import open_source_for_run, source_for_plan, retouch_identity
from raw_support import open_linear
import vision_transport


def main():
    checks=[]
    loopback=Loopback()
    actual_request=vision_transport.request_vision_json
    try:
        with tempfile.TemporaryDirectory(prefix='automatic-retouch-') as name:
            root=Path(name);manifest,_=fixture(root/'fixture')
            run={'run_id':'parent','source':{'path':manifest['source']['path'],'sha256':manifest['source']['sha256']},'options':{}}
            output=root/'outputs/runs/parent/automatic-native-retouch';output.mkdir(parents=True)
            automatic.write(output/'input.json',run)
            connection_arguments=[]
            def loopback_request(source,out_dir,prompt,**kwargs):
                connection_arguments.append(kwargs)
                candidate_id=re.search(r'(?:Candidate ID |Candidate )([a-f0-9]{16})',prompt).group(1)
                if 'native_retouch_donor_approval.v1' in prompt:
                    loopback.payload={'schema_version':'native_retouch_donor_approval.v1','candidate_id':candidate_id,
                        'decision':'approve','confidence':.99,'same_material':True,'same_surface_scale':True,
                        'same_illumination':True,'donor_has_no_protected_or_undesired_feature':True,
                        'reason':'Synthetic fixture known clean donor from the same surface.'}
                else:
                    loopback.payload={'schema_version':'native_retouch_candidate_approval.v1','candidate_id':candidate_id,
                        'decision':'approve','confidence':.99,'defect_type':'temporary_surface_blemish',
                        'target_location_verified':True,'protected_feature_checks':{key:False for key in producer.FEATURE_CHECKS},
                        'reason':'Synthetic fixture known temporary small defect.'}
                endpoint='http://127.0.0.1:'+str(loopback.server.server_port)+'/v1'
                with patch.object(vision_transport,'_connection',return_value=({'registry_id':'loopback_test_only'},endpoint,
                            'synthetic-key','/chat/completions')):
                    return actual_request(source,out_dir,prompt,vision_model='gpt-6.1-sol')
            with patch.object(vision_transport,'request_vision_json',side_effect=loopback_request):
                automatic.worker(root,output/'input.json',output,analysis_path=root/'fixture/analysis.json',
                    provider='loopback_test_only',api_base='http://127.0.0.1/test',api_url='/chat/completions',api_key_env='SYNTHETIC_KEY_ENV')
            result=automatic.read(output/'result.json');binding=result['layer_record']
            assert binding['approved_patches']==1 and len(loopback.calls)==2
            assert all(arguments['api_base']=='http://127.0.0.1/test' and arguments['api_url']=='/chat/completions'
                and arguments['api_key_env']=='SYNTHETIC_KEY_ENV' for arguments in connection_arguments)
            checks.append('actual_loopback_candidate_and_donor_approvals_consumed_by_worker')
            run['options']['native_source_retouch']=binding
            original,precision=open_linear(Path(run['source']['path']))
            repaired,repaired_precision=open_source_for_run(run,root=root)
            layer=automatic.read(binding['layer']['path'])
            expected=producer.apply_frozen_retouch(original,layer,source_sha256=run['source']['sha256'],verify_original=True)
            np.testing.assert_array_equal(repaired,expected)
            assert np.any(repaired!=original) and repaired_precision['native_retouch_consumptions']==1
            assert automatic.sha(run['source']['path'])==run['source']['sha256']
            checks.append('nonzero_source_delta_consumed_once_before_display_original_file_preserved')
            from display_transform import display_linear, source_transform_for_precision
            from raw_support import RGB_DECODE_VERSION
            from source_foreground import estimate_source_background
            displayed=display_linear(repaired);alpha=np.ones(repaired.shape[:2],np.float32)
            alpha[:, :40]=0
            trimap=np.where(alpha==0,0,255).astype(np.uint8)
            bg,fp,_,_,_,_=estimate_source_background(displayed,alpha,trimap)
            assets={}
            for key,values in [('foreground_premultiplied',fp),('foreground_alpha',alpha),('old_environment',bg)]:
                path=root/(key+'.npz');np.savez_compressed(path,values=values)
                assets[key]={'path':str(path),'sha256':automatic.sha(path),'size':list(alpha.shape[::-1])}
            plan={'schema_version':'design_reconstruction_plan.v1','working_domain':'display_linear_sRGB','execution':{'source_reference':'paired_native_baseline'},'source_baseline':{'schema':'appearance_source_baseline.v1','working_domain':'display_linear_sRGB','assets':assets,
                'native_source_retouch':copy.deepcopy(binding), 'source_sha256':run['source']['sha256'],
                'source_precision':'raster8_srgb', 'decode_version':RGB_DECODE_VERSION,
                'source_display':source_transform_for_precision('raster8_srgb'), 'source_display_applied_once':True}}
            np.testing.assert_allclose(source_for_plan(plan),displayed,atol=1e-7,rtol=0)
            assert not np.array_equal(source_for_plan(plan),display_linear(expected+(expected-original)))
            checks.append('actual_background_decomposition_and_saved_pair_replay_do_not_add_delta_twice')
            run['options']['appearance_revision']={'design_reconstruction_plan':plan}
            child=copy.deepcopy(run);child['run_id']='child';child=automatic.own_binding(root,child)
            child_path=Path(child['options']['native_source_retouch']['layer']['path'])
            assert (root/'outputs/runs/child') in child_path.parents
            checked=producer.validate_frozen_retouch_layer(automatic.read(child_path),source_sha256=run['source']['sha256'])
            assert checked['approved_patches']==1
            np.testing.assert_array_equal(open_source_for_run(child,root=root)[0],repaired)
            checks.append('child_owns_rehashed_complete_causal_graph_and_exact_replay')
            bad=copy.deepcopy(child);bad['options']['native_source_retouch']['working_domain']='scene_linear_sRGB'
            try:open_source_for_run(bad,root=root)
            except ValueError:pass
            else:raise AssertionError('wrong RAW/display domain accepted')
            checks.append('cross_domain_consumption_rejected')
            # A durable result can reopen without producer/model/API calls.
            automatic.write(output/'attempt.json',{'status':'frozen_source_stage_ready'})
            with patch.object(subprocess_fixture(),'Popen',side_effect=AssertionError('saved work must not start a process')):
                reopened=automatic.prepare_once(root,{'run_id':'parent','source':run['source'],'options':{}})
            assert reopened['status']=='reopened_frozen_source_stage'
            checks.append('durable_completed_reopen_has_no_api_or_producer')
            cancelled_run={'run_id':'cancelled','source':run['source'],'options':{}}
            cancelled=automatic.prepare_once(root,cancelled_run,cancel=lambda:True)
            assert cancelled['status']=='cancelled_draft_retained'
            repeated=automatic.prepare_once(root,cancelled_run,cancel=lambda:False)
            assert repeated['status']=='already_attempted' and repeated['layer_record'] is None
            checks.append('durable_cancel_preserves_draft_and_never_retries')
            # Reuse proof runs current code/source/domain and still validates real
            # double approvals rather than trusting the enabled/count fields.
            reuse=root/'outputs/runs/reuse/automatic-native-retouch';reuse.mkdir(parents=True)
            reuse_run={'run_id':'reuse','source':run['source'],'options':{}}
            automatic.write(reuse/'input.json',reuse_run)
            with patch.object(vision_transport,'request_vision_json',side_effect=AssertionError('valid frozen evidence must not call API')):
                automatic.worker(root,reuse/'input.json',reuse,reuse_layer=binding['layer']['path'])
            assert automatic.read(reuse/'result.json')['reused_actual_evidence'] is True
            checks.append('same_code_source_domain_actual_evidence_reuse_has_zero_api')
            # One finite, saved attempt cannot silently retry after worker error.
            failed_run={'run_id':'failed','source':run['source'],'options':{}}
            with patch('local_photo_analysis.local_runtime_paths',return_value={'python':root/'missing-python.exe'}):
                failed=automatic.prepare_once(root,failed_run)
            assert failed['status']=='unavailable_draft_retained'
            assert automatic.prepare_once(root,failed_run)['status']=='already_attempted'
            checks.append('durable_worker_failure_keeps_draft_and_blocks_repeated_api')
            assert automatic.prepare_once(root,failed_run,authorized=False)['status']=='not_authorized_for_new_preparation'
            checks.append('saved_task_without_explicit_authorization_cannot_trigger_attempt')
            # Cancellation after the first real loopback response must prevent
            # submitting the second review even in a directly called worker.
            cancel_dir=root/'cancel-between-requests';cancel_dir.mkdir();signal=cancel_dir/'cancel.txt'
            automatic.write(cancel_dir/'input.json',{'run_id':'cancel-between','source':run['source'],'options':{}})
            before=len(loopback.calls)
            def cancel_after_response(*args,**kwargs):
                response=loopback_request(*args,**kwargs);signal.write_text('cancel');return response
            with patch.object(vision_transport,'request_vision_json',side_effect=cancel_after_response):
                try:automatic.worker(root,cancel_dir/'input.json',cancel_dir,analysis_path=root/'fixture/analysis.json',cancel_request_path=signal)
                except InterruptedError:pass
                else:raise AssertionError('cancel between candidate/donor reviews ignored')
            assert len(loopback.calls)==before+1 and not (cancel_dir/'result.json').exists()
            checks.append('cancel_after_real_candidate_response_prevents_donor_request')
            # Exercise the actual export/import with a complete frozen native
            # retouch graph, rather than only the top-level delta path.
            from portable_project import export_project,import_project
            portable_run=copy.deepcopy(run)
            portable_run.update(kind='single_input_run',sample='fixture',revision=1,input_mode='single_source',
                reference=copy.deepcopy(run['source']))
            portable_run['options']['render_mode']='native_source_design'
            portable_plan=portable_run['options']['appearance_revision']['design_reconstruction_plan']
            portable_plan.update(source_sha256=run['source']['sha256'],candidate_sha256=run['source']['sha256'],
                canvas={'source_oriented_size':[384,384]},mask_roles={'foreground_alpha':assets['foreground_alpha']},layers=[{'id':'background_design','kind':'background_design','enabled':False,'assets':{'foreground_premultiplied':assets['foreground_premultiplied'],'old_environment':assets['old_environment'],'background':{**assets['old_environment'],'representation':'sampling_validity_associated_RGB','validity_associated':True},'validity':assets['foreground_alpha']}}])
            automatic.write(root/'outputs/runs/parent/manifest.json',portable_run)
            archive=root/'retouch.cosplay';export_project(root,'parent',archive)
            imported=import_project(root/'portable-workspace',archive,run_id='imported')['run']
            imported_layer=automatic.read(imported['options']['native_source_retouch']['layer']['path'])
            checked=producer.validate_frozen_retouch_layer(imported_layer,source_sha256=run['source']['sha256'])
            assert checked['approved_patches']==1
            np.testing.assert_array_equal(open_source_for_run(imported,root=root/'portable-workspace')[0],repaired)
            np.testing.assert_array_equal(source_for_plan(imported['options']['appearance_revision']['design_reconstruction_plan']),source_for_plan(plan))
            checks.append('actual_portable_export_import_complete_approval_graph_and_exact_native_replay')
    finally:loopback.close()
    print(json.dumps({'ok':True,'checks':len(checks),'names':checks,'gpu_used':False,'external_api_used':False}))


def subprocess_fixture():
    import subprocess
    return subprocess


if __name__=='__main__':main()
