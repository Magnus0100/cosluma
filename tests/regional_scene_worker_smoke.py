"""Real CPU provider transport with an HTTP fixture; no model/inference."""
from __future__ import annotations
import copy
import io
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import regional_scene_worker as worker
import source_scene_enhancement as scene
from source_scene_enhancement_smoke import fixture


def resume_fixture(root):
    import creative_task as task
    from creative_direction import compile_direction
    from cos_node_registry import compile_workflow
    root=Path(root);old,plan,_,_=fixture(root/'in',size=(1152,1152),regional=True)
    old['status']='blocked'
    old['generation_budget']={'origin_run_id':old['run_id']}
    binding={'source':old['source']['sha256'],'reference':old['reference']['sha256'],
        'core_box':old['task_intent']['regional_refinement']['regions'][0]['core_box'],
        'native_size':plan['canvas']['source_oriented_size'],'checked_plan_id':old['task_plan']['plan_id'],
        'direction_id':old['creative_direction']['direction_id']}
    journal={'schema':worker.SCHEMA,'state':'failed_no_submission','binding':binding,
             'model_submissions':0,'automatic_retry':False,'error':'synthetic missing provider resource'}
    scene.write(root/'outputs/runs'/old['run_id']/'manifest.json',old)
    ledger=root/'outputs/regional-reservations'/old['run_id']/'regional-refinement.json';scene.write(ledger,journal)
    new=copy.deepcopy(old);new['run_id']='run-continuation';new['status']='preparing'
    intent=copy.deepcopy(old['task_intent']);intent['mode']='existing_reference'
    intent['regional_refinement']['resume_from_run_id']=old['run_id']
    intent=task.normalize_task_intent(intent);direction=compile_direction(intent,source_sha256=old['source']['sha256'])
    checked=copy.deepcopy(old['task_plan']);checked.update(mode='existing_reference',intent=intent,
        creative_direction=direction,plan_id='b'*64,image_budget=task.image_budget(intent),
        layer_workflow=compile_workflow('existing_reference'))
    checked['inputs']['reference']=old['reference']
    proof=task.verify_regional_resume(root,intent,checked['inputs'],direction)
    checked['regional_resume_authorization']=proof
    new.update(task_intent=intent,task_plan=checked,creative_direction=direction)
    return old,new,plan,ledger,journal


class Handler(BaseHTTPRequestHandler):
    submissions=0;images=[];output=None;fail=False;wrong_canvas=False;photometric_gain=.95
    def log_message(self,*args):pass
    def reply(self,payload,raw=False):
        body=payload if raw else json.dumps(payload).encode()
        self.send_response(200);self.end_headers();self.wfile.write(body)
    def do_POST(self):
        body=self.rfile.read(int(self.headers['Content-Length']))
        if self.path=='/upload/image':
            data=body.split(b'\r\n\r\n',1)[1].rsplit(b'\r\n--',1)[0]
            type(self).images.append(data);self.reply({'name':f'input-{len(self.images)}.png','subfolder':'','type':'input'})
        elif self.path=='/prompt':
            type(self).submissions+=1
            type(self).graph=json.loads(body)['prompt']
            with Image.open(io.BytesIO(self.images[-1])) as im:values=np.asarray(im.convert('RGB')).copy()
            # Synthetic local subject is only in the core. The changed halo is
            # a bounded colour offset which the real alignment must remove.
            values=np.uint8(np.clip(values.astype(float)*self.photometric_gain,0,255))
            values[440:475,440:475]=[155,120,65]
            if self.wrong_canvas:
                # Full-image composition drawn into a correctly sized local
                # output: a body-sized rectangle and head erase halo content.
                values[:]=[8,5,2];values[160:920,280:800]=[22,75,125]
                yy,xx=np.indices(values.shape[:2]);values[((xx-540)**2+(yy-250)**2)<130**2]=[150,100,75]
            stream=io.BytesIO();Image.fromarray(values).save(stream,format='PNG');type(self).output=stream.getvalue()
            self.reply({'prompt_id':'cpu-fixture-'+str(self.submissions)})
        else:self.send_error(404)
    def do_GET(self):
        if self.path.startswith('/history/'):
            ident=self.path.rsplit('/',1)[-1]
            if self.fail:self.reply({ident:{'status':{'status_str':'error','messages':['synthetic failure']}}})
            else:self.reply({ident:{'status':{'completed':True,'status_str':'success'},
                'outputs':{'8':{'images':[{'filename':'result.png','type':'output','subfolder':''}]}}}})
        elif self.path.startswith('/view?'):self.reply(self.output,raw=True)
        else:self.send_error(404)


class RegionalTests(unittest.TestCase):
    def test_known_returned_candidate_cpu_reuse_zero_submit_and_shifted_permission(self):
        import creative_task as task,reference_provider
        from creative_direction import compile_direction
        from cos_node_registry import compile_workflow
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        Handler.submissions=0;Handler.images=[];Handler.fail=False;Handler.wrong_canvas=False;Handler.photometric_gain=.6
        try:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);old,plan,_,_=fixture(root/'in',size=(1152,1152),regional=True)
                old['task_plan']['inputs']['source']['regional_native_size']=[1152,1152]
                old['options']['api_base']=f'http://127.0.0.1:{server.server_port}'
                old['generation_budget']={'origin_run_id':old['run_id']}
                output=root/'known-original';scene.write(output/'request.json',{'root':str(root),'run':old,'plan':plan})
                with patch.dict(os.environ,{'COSPLAY_PROVIDER_SETTINGS':str(root/'settings.json')}), \
                     patch('start_qwen_image.ensure_running',return_value={'managed':False}), \
                     patch('start_qwen_image.release_for_run',return_value={'state':'idle_models_released'}), \
                     patch.object(reference_provider,'qwen_quality_gate',return_value={'passed':True}), \
                     patch.dict(worker.ALIGNMENT,{'maximum_exposure_EV':.35}):
                    with self.assertRaisesRegex(ValueError,'local canvas halo'):
                        worker.prepare_local(root,old,plan,output)
                self.assertEqual(Handler.submissions,1)
                old['status']='blocked';manifest=root/'outputs/runs/run-fixture/manifest.json';scene.write(manifest,old)
                ledger=root/'outputs/regional-reservations/run-fixture/regional-refinement.json'
                immutable=(manifest.read_bytes(),ledger.read_bytes())
                new=copy.deepcopy(old);new['run_id']='run-cpu-reuse'
                intent=copy.deepcopy(old['task_intent']);intent['mode']='existing_reference'
                intent['regional_refinement'].update(maximum_submissions=0,reuse_generated_from_run_id=old['run_id'])
                intent['regional_refinement']['regions'][0]['core_box']=[96,96,608,608]
                intent=task.normalize_task_intent(intent)
                direction=compile_direction(intent,source_sha256=old['source']['sha256'])
                checked=copy.deepcopy(old['task_plan']);checked.update(mode='existing_reference',intent=intent,
                    creative_direction=direction,image_budget=task.image_budget(intent),plan_id='c'*64,
                    layer_workflow=compile_workflow('existing_reference'))
                checked['inputs']['reference']=old['reference']
                proof=task.verify_regional_candidate_reuse(root,intent,checked['inputs'],direction)
                checked['regional_candidate_reuse_authorization']=proof
                new.update(task_intent=intent,task_plan=checked,creative_direction=direction)
                with patch.object(reference_provider,'main',side_effect=AssertionError('CPU reuse called provider')), \
                     patch('subprocess.Popen',side_effect=AssertionError('CPU reuse spawned process')), \
                     patch('start_qwen_image.release_for_run',side_effect=AssertionError('CPU reuse touched GPU')):
                    result,record=worker.prepare(root,new,plan,root/'cpu-recovery')
                self.assertEqual(record['new_model_submissions'],0)
                self.assertEqual(record['state'],'known generated result reused')
                self.assertEqual(scene.execution_stats(new,result,record)['reused_provider_ids'],['cpu-fixture-1'])
                self.assertEqual((manifest.read_bytes(),ledger.read_bytes()),immutable)
                stored=worker.validate_saved(record,new,result)
                self.assertEqual(stored['binding']['core_box'],[96,96,608,608])
                self.assertEqual(stored['model_binding']['core_box'],[128,128,640,640])
                self.assertEqual(stored['binding']['context_box'],stored['model_binding']['context_box'])
                self.assertEqual(stored['alignment']['profile']['maximum_exposure_EV'],2.)
                with np.load(stored['input_background']['path']) as z:before=z['values'].copy()
                with np.load(stored['output_background']['path']) as z:after=z['values'].copy()
                allowed=np.zeros(before.shape[:2],bool);allowed[96:608,96:608]=True
                self.assertTrue(np.array_equal(before[~allowed],after[~allowed]))
                changed=copy.deepcopy(plan);changed['canvas']['source_oriented_size']=[1151,1152]
                with self.assertRaises(ValueError):worker.prepare(root,new,changed,root/'changed-canvas')
                self.assertEqual(Handler.submissions,1)
        finally:Handler.photometric_gain=.95;server.shutdown();thread.join(timeout=5);server.server_close()

    def test_real_wrong_portrait_and_white_frame_stay_rejected_at_v3(self):
        from float_appearance import srgb_to_linear
        configured=os.environ.get('COSPLAY_REGIONAL_REGRESSION_DIRS','').strip()
        if not configured:
            self.skipTest('Set COSPLAY_REGIONAL_REGRESSION_DIRS for optional private fixtures; synthetic checks remain mandatory')
        folders=[Path(value) for value in configured.split(os.pathsep) if value]
        self.assertTrue(folders)
        for folder in folders:
            self.assertTrue((folder/'provider/regional.png').is_file(), 'Configured regression fixture is missing')
        for p in folders:
            with np.load(p/'input-context.npz') as z:base=z['values']
            with np.load(p/'reliable-halo.npz') as z:reliable=z['values']>0
            permission=np.asarray(Image.open(p/'permission.png'))>0
            raw=srgb_to_linear(np.asarray(Image.open(p/'provider/regional.png').convert('RGB'),np.float32)/255)
            with self.subTest(actual_failure=str(p)),self.assertRaisesRegex(ValueError,'local canvas halo'):
                worker.align_candidate(base,raw,permission,reliable)

    def test_explicit_existing_task_has_independent_origin_and_local_conditions(self):
        import creative_task as task,reference_provider
        from creative_direction import compile_direction
        from cos_node_registry import compile_workflow
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        Handler.submissions=0;Handler.images=[];Handler.fail=False;Handler.wrong_canvas=False
        try:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);run,plan,_,_=fixture(root/'in',size=(1152,1152),regional=True)
                run['run_id']='run-new-explicit';intent=run['task_intent'];intent['mode']='existing_reference'
                intent=task.normalize_task_intent(intent);direction=compile_direction(intent,source_sha256=run['source']['sha256'])
                run['task_intent']=intent;run['creative_direction']=direction
                run['task_plan'].update(mode='existing_reference',intent=intent,image_budget=task.image_budget(intent),
                    creative_direction=direction,layer_workflow=compile_workflow('existing_reference'))
                run['task_plan']['inputs']['reference']=run['reference']
                run['generation_budget']={'origin_run_id':run['run_id'],'image_submissions':0}
                self.assertEqual(run['task_plan']['image_budget'],{'initial_reference':0,'reference_correction':0,
                    'independent_effect':0,'regional_refinement':1})
                worker.authorization(run,root=root)
                inherited=copy.deepcopy(run);inherited['generation_budget']['origin_run_id']='run-consumed-parent'
                with self.assertRaisesRegex(ValueError,'independent run origin'):worker.authorization(inherited,root=root)
                run['options']['api_base']=f'http://127.0.0.1:{server.server_port}'
                with patch.dict(os.environ,{'COSPLAY_PROVIDER_SETTINGS':str(root/'settings.json')}), \
                     patch('start_qwen_image.ensure_running',return_value={'managed':False}), \
                     patch('start_qwen_image.release_for_run',return_value={'state':'idle_models_released'}), \
                     patch.object(reference_provider,'qwen_quality_gate',return_value={'passed':True}):
                    _,record=worker.prepare_local(root,run,plan,root/'out')
                self.assertEqual(Handler.submissions,1);self.assertEqual(record['new_model_submissions'],1)
                for raw in Handler.images:
                    with Image.open(io.BytesIO(raw)) as im:self.assertEqual(im.size,(1024,1024))
                prompt=Handler.graph['5']['inputs']['prompt']
                self.assertNotIn('Global light/design:',prompt)
                self.assertNotIn('CHECKED SOURCE ENHANCEMENT DIRECTION',prompt)
                self.assertIn('ONE supplied image',prompt);self.assertIn('两只小旧纸蝴蝶',prompt)
                self.assertNotIn('<image2>',prompt);self.assertNotIn('<image3>',prompt)
                self.assertIn('x=[256,768), y=[256,768)',prompt)
                self.assertEqual(len(Handler.images),1)
                self.assertNotIn('images.image_2',Handler.graph['5']['inputs'])
                self.assertEqual(Handler.graph['5']['inputs']['resolution'],0)
                manifest=json.loads((root/'out/conditions.json').read_text(encoding='utf8'))
                self.assertFalse(manifest['permission_mask']['transmitted_to_model'])
                self.assertTrue((root/'outputs/regional-reservations/run-new-explicit/regional-refinement.json').is_file())
                self.assertFalse((root/'outputs/regional-reservations/run-consumed-parent').exists())
        finally:server.shutdown();thread.join(timeout=5);server.server_close()

    def test_correct_size_wrong_full_portrait_is_rejected_consumed_not_unknown(self):
        import reference_provider
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        Handler.submissions=0;Handler.images=[];Handler.fail=False;Handler.wrong_canvas=True
        try:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);run,plan,_,_=fixture(root/'in',size=(1152,1152),regional=True)
                run['options']['api_base']=f'http://127.0.0.1:{server.server_port}'
                with patch.dict(os.environ,{'COSPLAY_PROVIDER_SETTINGS':str(root/'settings.json')}), \
                     patch('start_qwen_image.ensure_running',return_value={'managed':False}), \
                     patch('start_qwen_image.release_for_run',return_value={'state':'idle_models_released'}), \
                     patch.object(reference_provider,'qwen_quality_gate',return_value={'passed':True}):
                    with self.assertRaisesRegex(ValueError,'local canvas halo'):
                        worker.prepare_local(root,run,plan,root/'out')
                    with self.assertRaisesRegex(RuntimeError,'already reserved'):
                        worker.prepare_local(root,run,plan,root/'again')
                self.assertEqual(Handler.submissions,1)
                state=json.loads((root/'outputs/regional-reservations/run-fixture/regional-refinement.json').read_text(encoding='utf8'))
                self.assertEqual(state['state'],'generated_rejected');self.assertEqual(state['model_submissions'],1)
                self.assertEqual(state['prompt_id'],'cpu-fixture-1');self.assertTrue(state['provider_returned'])
                self.assertTrue(Path(state['raw_candidate']['path']).is_file())
                self.assertFalse((root/'out/native-patch.npz').exists())
        finally:Handler.wrong_canvas=False;server.shutdown();thread.join(timeout=5);server.server_close()

    def test_dark_single_channel_clipping_uses_luma_and_rejects_wrong_structure(self):
        yy,xx=np.indices((192,192),dtype=np.float32)
        value=.002+.001*np.sin(xx/15)+.0006*np.cos(yy/22)
        base=np.stack((value*2,value,value*.04),-1).astype(np.float32)
        raw=base*.9;raw[...,2]=0  # dark 8-bit blue can legitimately quantize to zero
        permitted=np.zeros((192,192),bool);permitted[48:144,48:144]=True
        raw[86:96,86:96]=.1
        aligned,report=worker.align_candidate(base,raw,permitted,np.ones(permitted.shape,bool))
        self.assertGreater(report['anchor_pixels'],1024)
        self.assertEqual(report['chroma_anchor_pixels'][2],0)
        self.assertGreater(float(aligned[90,90,0]),.1)
        wrong=raw.copy();wrong[~permitted]=raw[::-1,::-1][~permitted]*10
        with self.assertRaisesRegex(ValueError,'local canvas halo'):
            worker.align_candidate(base,wrong,permitted,np.ones(permitted.shape,bool))

    def test_explicit_zero_submission_resume_consumes_once_and_keeps_failure(self):
        import reference_provider
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        Handler.submissions=0;Handler.images=[];Handler.fail=False;Handler.wrong_canvas=False
        try:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);old,run,plan,ledger,previous=resume_fixture(root)
                manifest=root/'outputs/runs/run-fixture/manifest.json';immutable=manifest.read_bytes()
                run['options']['api_base']=f'http://127.0.0.1:{server.server_port}'
                with patch.dict(os.environ,{'COSPLAY_PROVIDER_SETTINGS':str(root/'settings.json')}), \
                     patch('start_qwen_image.ensure_running',return_value={'managed':False}), \
                     patch('start_qwen_image.release_for_run',return_value={'state':'idle_models_released'}), \
                     patch.object(reference_provider,'qwen_quality_gate',return_value={'passed':True}):
                    result,record=worker.prepare_local(root,run,plan,root/'out')
                self.assertEqual(Handler.submissions,1);self.assertEqual(record['new_model_submissions'],1)
                state=json.loads(ledger.read_text('utf8'))
                self.assertEqual(state['state'],'completed');self.assertEqual(state['model_submissions'],1)
                self.assertEqual(state['previous_failure'],previous);self.assertEqual(state['attempt_history'],[previous])
                self.assertEqual(manifest.read_bytes(),immutable)
                self.assertEqual(scene.execution_stats(run,result,record)['provider_ids'],['cpu-fixture-1'])
                with self.assertRaises(ValueError):worker.authorization(run,root=root)
                self.assertEqual(Handler.submissions,1)
        finally:server.shutdown();thread.join(timeout=5);server.server_close()

    def test_two_actual_processes_cannot_spend_one_resume_ticket(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old,run,plan,ledger,previous=resume_fixture(root)
            binding={**previous['binding'],'checked_plan_id':run['task_plan']['plan_id'],
                     'direction_id':run['creative_direction']['direction_id']}
            data=root/'resume-probe.json';scene.write(data,{'root':str(root),'checked':run['task_plan'],'binding':binding,'ledger':str(ledger)})
            probe=root/'probe.py';probe.write_text('''import json,sys
from pathlib import Path
from cache_store import exclusive_file_lock
from regional_scene_worker import resume_failure
from source_scene_enhancement import write
d=json.loads(Path(sys.argv[1]).read_text('utf8'));p=Path(d['ledger'])
try:
 with exclusive_file_lock(p.with_suffix('.lock')):
  old=json.loads(p.read_text('utf8'))
  previous=resume_failure(d['root'],d['checked'],old,d['binding'])
  write(p,{'schema':'regional_scene_refinement.v1','state':'reserved','model_submissions':0,
   'binding':d['binding'],'previous_failure':previous,'attempt_history':[previous]})
 print('WON')
except ValueError:
 print('REJECTED')
''',encoding='utf8')
            children=[subprocess.Popen([sys.executable,'-B',str(probe),str(data)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
            outputs=[]
            for child in children:
                out,err=child.communicate(timeout=30);self.assertEqual(child.returncode,0,err);outputs.append(out.strip())
            self.assertCountEqual(outputs,['WON','REJECTED'])
            self.assertEqual(json.loads(ledger.read_text(encoding='utf8'))['previous_failure'],previous)

    def test_resume_refuses_any_uncertain_or_used_original_reservation(self):
        import creative_task as task
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old,run,plan,ledger,previous=resume_fixture(root)
            worker.authorization(run,root=root)
            for changes in ({'state':'unknown'},{'state':'submitted','model_submissions':1},
                            {'state':'completed'},{'request_started':True},{'prompt_id':'seen'},
                            {'model_submissions':None}):
                scene.write(ledger,{**previous,**changes})
                with self.assertRaises(ValueError):worker.authorization(run,root=root)
            scene.write(ledger,previous)
            changed=copy.deepcopy(run);changed['reference']['sha256']='f'*64
            with self.assertRaises(ValueError):worker.authorization(changed,root=root)

    def test_provider_once_permissions_alignment_replay_and_portable(self):
        import reference_provider,creative_task
        from creative_direction import compile_direction
        from design_reconstruction import materialize_plan
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        Handler.submissions=0;Handler.images=[];Handler.fail=False;Handler.wrong_canvas=False
        try:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);run,plan,_,alpha=fixture(root/'in',size=(1152,1152),regional=True)
                run['options']['api_base']=f'http://127.0.0.1:{server.server_port}'
                # No real service discovery or model launch; the actual adapter
                # still uploads three images, submits one graph, and gets bytes.
                with patch.dict(os.environ,{'COSPLAY_PROVIDER_SETTINGS':str(root/'settings.json')}), \
                     patch('start_qwen_image.ensure_running',return_value={'managed':False}), \
                     patch('start_qwen_image.release_for_run',return_value={'state':'idle_models_released'}) as release, \
                     patch.object(reference_provider,'qwen_quality_gate',return_value={'passed':True}), \
                     patch('regional_scene_worker.prepare',side_effect=lambda root,run,plan,out,**kw:worker.prepare_local(root,run,plan,out)):
                    result=scene.prepare(root,run,plan,root/'cache')
                self.assertEqual(Handler.submissions,1);self.assertEqual(len(Handler.images),1);self.assertEqual(release.call_count,1)
                stats=result['provenance']['source_scene_enhancement'];self.assertEqual(stats['new_model_submissions'],1)
                self.assertEqual(stats['provider_ids'],['cpu-fixture-1'])
                receipt=scene.validate_observation(result,run);record=receipt['regional_refinement']
                r=worker.validate_saved(record,run,result)
                self.assertFalse(r['background_geometry_preservation_verified']);self.assertTrue(r['geometry_review_required'])
                self.assertFalse(r['alignment']['new_core_content_used_as_anchor']);self.assertGreater(r['alignment']['anchor_pixels'],1024)
                with np.load(r['input_background']['path']) as z:before=z['values'].copy()
                with np.load(r['output_background']['path']) as z:after=z['values'].copy()
                core=r['binding']['core_box'];allowed=np.zeros(alpha.shape,bool);allowed[core[1]:core[3],core[0]:core[2]]=True
                self.assertTrue(np.array_equal(before[~allowed],after[~allowed]));self.assertTrue(np.array_equal(before[alpha>0],after[alpha>0]))
                self.assertTrue(np.any(before[allowed]!=after[allowed]))
                run['options']['appearance_revision']['design_reconstruction_plan']=result
                owned=materialize_plan(root,run,result)
                continuation=copy.deepcopy(run);continuation['options']['appearance_revision']['design_reconstruction_plan']=owned
                continuation['task_intent']['mode']='existing_reference'
                continuation['task_intent']['regional_refinement']={'maximum_submissions':0,'coordinate_space':'source_canvas','regions':[]}
                continuation['task_plan']=None
                with patch('subprocess.Popen',side_effect=AssertionError('zero-budget replay invoked model')):
                    replay=scene.prepare(root,continuation,plan,root/'replay')
                self.assertEqual(replay['provenance']['source_scene_enhancement']['new_model_submissions'],0)
                self.assertEqual(replay['provenance']['source_scene_enhancement']['reused_saved_regions'],1)
                self.assertEqual(replay['provenance']['source_scene_enhancement']['reused_provider_ids'],['cpu-fixture-1'])
                self.assertEqual(Handler.submissions,1)
                # Already reserved success cannot silently become another request.
                with self.assertRaises(RuntimeError):
                    changed=copy.deepcopy(run);changed['task_plan']['plan_id']='b'*64
                    worker.prepare_local(root,changed,plan,root/'changed')
                self.assertEqual(Handler.submissions,1)
                shutil.rmtree(root/'cache');worker.validate_saved(scene.validate_observation(owned,run)['regional_refinement'],run,owned)
                altered=copy.deepcopy(plan);altered['canvas']['source_oriented_size']=[1151,1152]
                with self.assertRaises(ValueError):worker.validate_saved(record,run,altered)
        finally:server.shutdown();thread.join(timeout=5);server.server_close()

    def test_unknown_reservation_refuses_resubmit_and_preserves_first_error(self):
        import reference_provider
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run,plan,_,_=fixture(root/'in',size=(1152,1152),regional=True)
            def fail(endpoint,image,workflow,prompt,timeout,events,**kw):
                events.append({'event':'comfyui_submitted','prompt_id':'cpu-unknown'})
                raise RuntimeError('synthetic first transport error')
            with patch('start_qwen_image.ensure_running',return_value={'managed':False}), \
                 patch('start_qwen_image.release_for_run',side_effect=RuntimeError('synthetic handoff error')), \
                 patch.object(reference_provider,'qwen_quality_gate',return_value={'passed':True}), \
                 patch.object(reference_provider.comfyui_adapter,'request_generation',side_effect=fail) as provider, \
                 patch.dict(os.environ,{'COSPLAY_PROVIDER_SETTINGS':str(root/'settings.json')}):
                with self.assertRaisesRegex(RuntimeError,'synthetic first transport error'):
                    worker.prepare_local(root,run,plan,root/'out')
                with self.assertRaisesRegex(RuntimeError,'already reserved'):
                    worker.prepare_local(root,run,plan,root/'out2')
                self.assertEqual(provider.call_count,1)
            state=json.loads((root/'outputs/regional-reservations/run-fixture/regional-refinement.json').read_text(encoding='utf8'))
            self.assertEqual(state['state'],'unknown');self.assertEqual(state['model_submissions'],1)
            self.assertEqual(state['prompt_id'],'cpu-unknown')

    def test_native_geometry_and_one_pixel_guard(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run,plan,_,_=fixture(root/'in',size=(1152,1152),regional=True)
            region=run['task_intent']['regional_refinement']['regions'][0]
            self.assertEqual(worker.geometry(plan,region)[1],[ -128,-128,896,896])
            from native_array_storage import save_native_array
            with np.load(plan['mask_roles']['foreground_alpha']['path']) as z:alpha=z['values'].copy()
            alpha[127,127]=1e-7;p=root/'pixel.npz';save_native_array(p,alpha)
            bad=copy.deepcopy(plan);bad['mask_roles']['foreground_alpha']=scene.asset(p,size=[1152,1152])
            with self.assertRaises(ValueError):worker.geometry(bad,region)
            broken=copy.deepcopy(run);broken['task_plan']['image_budget']['regional_refinement']=0
            with self.assertRaises(ValueError):worker.authorization(broken)

    def test_anchor_alignment_does_not_fit_new_core_and_mask_is_exact(self):
        base=np.full((192,192,3),.2,np.float32);raw=base*.8
        permission=np.zeros((192,192),bool);permission[48:144,48:144]=True
        raw[80:100,80:100]=.7
        aligned,params=worker.align_candidate(base,raw,permission,np.ones(permission.shape,bool))
        self.assertGreater(float(aligned[90,90,0]),.7)
        self.assertLess(abs(float(np.median(aligned[~permission]))-.2),.002)
        after,_=worker.blend_patch(base,aligned,permission)
        self.assertTrue(np.array_equal(after[~permission],base[~permission]))
        with self.assertRaises(ValueError):worker.align_candidate(base,raw,permission,np.zeros(permission.shape,bool))


if __name__=='__main__':unittest.main()
