"""Real loopback/venv processes; a tiny JSON service stands in for GPU inference."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, Mock

import beautygrpo_broker as broker_module
import beautygrpo_prepare as prepare
from beautygrpo_worker import execution_profile
from runtime_memory import BEAUTY_COLD_PHYSICAL_BYTES, BEAUTY_COLD_COMMIT_BYTES


SERVICE = '''import json,sys,time,os
from pathlib import Path
for line in sys.stdin:
 r=json.loads(line);p=Path(r['event_log']);p.parent.mkdir(parents=True,exist_ok=True)
 def event(value):
  with p.open('a',encoding='utf8') as f:f.write(json.dumps(value)+'\\n')
 event({'type':'beautygrpo_progress','request_id':r['request_id'],'stage':'fixture_cpu','message':'CPU fixture only','progress':.4})
 time.sleep(r['run'].get('fixture_wait',.15))
 out=Path(r['output']);out.mkdir(parents=True,exist_ok=True)
 (out/'result.json').write_text(json.dumps({'ok':True,'no_change':True,'result':None,'layer':None,'generation_inferences_this_attempt':0}),encoding='utf8')
 event({'type':'beautygrpo_result','request_id':r['request_id'],'ok':True})
'''

CLIENT = '''import os,sys,json
from pathlib import Path
sys.path.insert(0,os.environ['BROKER_TEST_SCRIPTS'])
import beautygrpo_prepare as p
from beautygrpo_broker import client
root=Path(os.environ['BROKER_TEST_ROOT']);run_id=os.environ['BROKER_TEST_RUN']
run={'run_id':run_id,'source':{'sha256':'a'*64}}
mode=os.environ['BROKER_TEST_MODE']
if mode=='busy':
 try:p.release_idle_for_stage(next_stage_cpu_bytes=0,next_stage='generation',next_stage_gpu=True)
 except p.BeautyServiceBusyError:print(json.dumps({'busy_refused':True,'pid':os.getpid()}))
 else:raise AssertionError('Busy pipeline was released')
elif mode=='park':
 receipt=p.release_idle_for_stage(next_stage_cpu_bytes=0,next_stage='generation',next_stage_gpu=True)
 demand=p.beauty_memory_needed()
 print(json.dumps({'receipt':receipt,'after':demand,'pid':os.getpid()}))
elif mode=='memory':
 print(json.dumps({'demand':p.beauty_memory_needed(),'pid':os.getpid()}))
else:
 demand=p.beauty_memory_needed()
 if mode=='cancel':run['fixture_wait']=30
 pending=p.begin_for_workbench(run,root,root/'outputs/runs'/run_id/'portrait-beauty-preparation')
 progress=[];count=[0]
 def cancel():count[0]+=1;return mode=='cancel' and count[0]>2
 try:result=p.join_for_workbench(pending,cancel=cancel,progress=lambda *x:progress.append(x))
 except InterruptedError:print(json.dumps({'cancelled':True,'pid':os.getpid()}))
 else:print(json.dumps({'demand':demand,'result':result,'progress':progress,'pid':os.getpid(),'service_pid':pending.process.pid}))
'''


class BrokerTests(unittest.TestCase):
    def test_startup_prewarm_handoff_waits_cancellably_without_releasing_busy_service(self):
        with tempfile.TemporaryDirectory(prefix='beauty-startup-handoff-') as directory:
            root = Path(directory); config = root/'installation.json'; config.write_text('{}', encoding='utf8')
            process = Mock(pid=4321); process.poll.return_value = None
            service = {'process':process, 'owner_pid':os.getpid(), 'busy_request_id':None,
                'pipeline_loaded':False, 'code_identity':{'fixture':1},
                'execution_profile':{'offload':'nunchaku_cuda'}}
            started, completed = threading.Event(), threading.Event()
            warm_results, errors = [], []
            def events(path, offset, request_id, progress, **kwargs):
                started.set()
                return offset, {'ok':True} if completed.is_set() else None
            def stopped():
                self.assertNotIn('busy_kind', service)
                self.assertIsNone(service['busy_request_id'])
                process.poll.return_value = 0
            def warm():
                try: warm_results.append(prepare.prewarm_for_workbench(root))
                except BaseException as error: errors.append(error)
            broker = broker_module.BeautyBroker(root)
            token, _ = broker.issue('job-startup', 'run-startup')
            broker.register(token, os.getpid(), broker_module._stamp(os.getpid()))
            rpc = broker_module.BrokerClient(broker.url, token, broker.owner_pid, broker.owner_stamp)
            try:
                with (patch.object(prepare, '_BEAUTY_SERVICE', service),
                      patch.object(prepare, 'installation_path', return_value=config),
                      patch.object(prepare, 'load_installation', return_value={}),
                      patch.object(prepare, '_start_beauty_service', return_value=(service,False)),
                      patch.object(prepare, '_read_service_events', side_effect=events),
                      patch.object(prepare, '_service_code_identity', return_value={'fixture':1}),
                      patch.object(prepare, '_stop_beauty_service', side_effect=stopped) as stop,
                      patch.object(broker_module, 'client', return_value=rpc),
                      patch('runtime_memory.memory_budget', return_value={'fits':True}),
                      patch('runtime_memory.memory_status', return_value={})):
                    thread = threading.Thread(target=warm); thread.start()
                    try:
                        self.assertTrue(started.wait(2))
                        self.assertFalse(prepare._reconcile_service_terminal(service))
                        checks = [0]
                        def cancel(): checks[0] += 1; return checks[0] > 1
                        with self.assertRaises(InterruptedError):
                            prepare.release_idle_for_stage(next_stage_cpu_bytes=0,
                                next_stage='generation', next_stage_gpu=True, cancel=cancel)
                        stop.assert_not_called()
                        self.assertIsNone(process.poll())
                        timer = threading.Timer(.2, completed.set); timer.start()
                        receipt = prepare.release_idle_for_stage(next_stage_cpu_bytes=0,
                            next_stage='generation', next_stage_gpu=True)
                        timer.join(); thread.join(2)
                        self.assertFalse(thread.is_alive())
                        self.assertFalse(errors, errors)
                        self.assertEqual(warm_results[0]['state'], 'pipeline_warm')
                        self.assertEqual(receipt['state'], 'idle_owned_service_stopped')
                        self.assertGreater(receipt['startup_prewarm_wait_seconds'], 0)
                        stop.assert_called_once()
                        # A failed startup observer must not leave a permanent
                        # prewarm marker that strands every later GPU stage.
                        process.poll.return_value = None; service['pipeline_loaded'] = False
                        with patch.object(prepare, '_read_service_events', side_effect=OSError('event log unavailable')):
                            with self.assertRaises(OSError): prepare.prewarm_for_workbench(root)
                        self.assertNotIn('busy_kind', service)
                        self.assertIsNone(service['busy_request_id'])
                        self.assertEqual(stop.call_count, 2)
                    finally:
                        completed.set(); thread.join(2)
            finally: broker.close()

    def test_real_venv_descendant_shared_warm_begin_progress_busy_park_and_cancel(self):
        with tempfile.TemporaryDirectory(prefix='beauty-broker-') as folder:
            root=Path(folder); service_script=root/'cpu-service.py';service_script.write_text(SERVICE,encoding='utf8')
            client_script=root/'client.py';client_script.write_text(CLIENT,encoding='utf8')
            config_path=root/'installation.json';config_path.write_text('{}',encoding='utf8')
            config={'python':sys.executable,'config_path':str(config_path),'config_sha256':prepare.sha(config_path)}
            log=root/'cpu-service.log';stream=log.open('ab')
            service_process=subprocess.Popen([sys.executable,'-B',str(service_script)],stdin=subprocess.PIPE,
                stdout=stream,stderr=stream,text=True,encoding='utf8',bufsize=1,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),start_new_session=os.name!='nt')
            service={'process':service_process,'stream':stream,'config_path':str(config_path.resolve()),
                'config_sha256':prepare.sha(config_path),'process_log':str(log),'owner_pid':os.getpid(),
                'code_identity':prepare._service_code_identity(),'execution_profile':execution_profile(config),
                'pipeline_loaded':True,'busy_request_id':None}
            broker=broker_module.BeautyBroker(root)
            launched=[];serial=[0]
            def child(mode):
                serial[0]+=1;job='job-fixture-'+str(serial[0]);run='run-fixture-'+str(serial[0])
                token,env=broker.issue(job,run,'a'*64)
                environment={**os.environ,**env,'BROKER_TEST_ROOT':str(root),'BROKER_TEST_RUN':run,
                    'BROKER_TEST_MODE':mode,'BROKER_TEST_SCRIPTS':str(Path(__file__).resolve().parents[1]/'scripts')}
                process=subprocess.Popen([sys.executable,'-B','-X','utf8',str(client_script)],env=environment,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8',
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                launched.append(process)
                broker.register(token,process.pid,broker_module._stamp(process.pid))
                stdout,stderr=process.communicate(timeout=15)
                if process.returncode:raise AssertionError(stderr)
                return json.loads(stdout),token,process.pid
            try:
                with patch.object(prepare,'_BEAUTY_SERVICE',service), \
                        patch.object(prepare,'installation_path',return_value=config_path), \
                        patch.object(prepare,'load_installation',return_value=config), \
                        patch('runtime_memory.memory_budget',return_value={'fits':True}):
                    result,token,launcher=child('prepare')
                    self.assertEqual(result['demand']['physical_bytes'],0)
                    self.assertTrue(result['demand']['broker_warm_verified'])
                    self.assertEqual(result['service_pid'],service_process.pid)
                    self.assertTrue(result['progress']);self.assertTrue(result['result']['ok'])
                    if os.name=='nt' and Path(sys.executable).parent.name.lower()=='scripts':
                        self.assertNotEqual(result['pid'],launcher,'Windows venv shim must exercise the real interpreter child')
                    self.assertIs(prepare._BEAUTY_SERVICE,service)
                    self.assertIsNone(service_process.poll())
                    # The warm memory discount has an epoch: an intervening
                    # unload cannot silently turn a warm admission into a cold
                    # allocation without another real memory check.
                    epoch,environment=broker.issue('job-epoch','run-epoch','a'*64)
                    broker.register(epoch,os.getpid(),broker_module._stamp(os.getpid()))
                    rpc=broker_module.BrokerClient(broker.url,epoch,broker.owner_pid,broker.owner_stamp)
                    self.assertTrue(rpc.call('memory')['warm_verified'])
                    service['pipeline_loaded']=False
                    with self.assertRaises(RuntimeError):rpc.begin({'run_id':'run-epoch','source':{'sha256':'a'*64}},root,
                        root/'outputs/runs/run-epoch/portrait-beauty-preparation')
                    self.assertFalse((root/'outputs/runs/run-epoch/portrait-beauty-preparation-input.json').exists())
                    service['pipeline_loaded']=True;broker.revoke(epoch)
                    # A token without actual socket/ancestry ownership cannot
                    # be reused by the test parent or by arbitrary JSON PID.
                    envtoken,env=broker.issue('job-foreign','run-foreign','a'*64)
                    broker.register(envtoken,service_process.pid,broker_module._stamp(service_process.pid))
                    foreign=broker_module.BrokerClient(broker.url,envtoken,broker.owner_pid,broker.owner_stamp)
                    with self.assertRaises(RuntimeError):foreign.call('memory')
                    broker.revoke(envtoken)
                    # Reconcile model/config identity: the same running PID
                    # alone is never sufficient evidence for a warm discount.
                    service['pipeline_loaded']=False
                    cold,cap,_=child('memory')
                    self.assertEqual(cold['demand']['physical_bytes'],BEAUTY_COLD_PHYSICAL_BYTES)
                    self.assertEqual(cold['demand']['commit_bytes'],BEAUTY_COLD_COMMIT_BYTES)
                    self.assertFalse(cold['demand']['broker_warm_verified']);broker.revoke(cap)
                    service['pipeline_loaded']=True
                    direct=prepare.begin_for_workbench({'run_id':'run-owned-local','fixture_wait':.8},root,
                        root/'outputs/runs/run-owned-local/portrait-beauty-preparation')
                    busy,cap,_=child('busy')
                    self.assertTrue(busy['busy_refused']);self.assertIsNone(service_process.poll())
                    prepare.join_for_workbench(direct);broker.revoke(cap)
                    # Park reaches the original GUI-owned service rather than
                    # falsely reporting not_running in a fresh CLI global.
                    parked,cap,_=child('park')
                    self.assertEqual(parked['receipt']['state'],'idle_owned_service_stopped')
                    self.assertTrue(parked['receipt']['process_exit_confirmed'])
                    self.assertIsNotNone(service_process.poll())
                    self.assertEqual(parked['after']['commit_bytes'],BEAUTY_COLD_COMMIT_BYTES)
                    broker.revoke(cap);broker.revoke(token)
                    # A new fixture service exercises cancellation through the
                    # real broker, which can stop only this request's owned tree.
                    stream=log.open('ab')
                    replacement=subprocess.Popen([sys.executable,'-B',str(service_script)],stdin=subprocess.PIPE,
                        stdout=stream,stderr=stream,text=True,encoding='utf8',bufsize=1,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),start_new_session=os.name!='nt')
                    launched.append(replacement)
                    prepare._BEAUTY_SERVICE={**service,'process':replacement,'stream':stream,'busy_request_id':None,
                        'pipeline_loaded':True}
                    cancelled,cap,_=child('cancel')
                    self.assertTrue(cancelled['cancelled']);self.assertIsNotNone(replacement.poll())
                    broker.revoke(cap)
            finally:
                broker.close()
                for process in launched:
                    if process.poll() is None:prepare._stop_process(process)
                if service_process.poll() is None:prepare._stop_process(service_process)
                stream.close()

    def test_capability_revocation_wrong_stamp_and_nonloopback_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix='beauty-broker-auth-') as folder:
            broker=broker_module.BeautyBroker(folder)
            try:
                token,_=broker.issue('job-auth','run-auth')
                broker.register(token,os.getpid(),broker_module._stamp(os.getpid()))
                rpc=broker_module.BrokerClient(broker.url,token,broker.owner_pid,broker.owner_stamp)
                self.assertIn('demand',rpc.call('memory'))
                with self.assertRaises(PermissionError):broker.dispatch(token,{'action':'memory','client_pid':os.getpid(),'client_stamp':'wrong'})
                broker.revoke(token)
                with self.assertRaises(RuntimeError):rpc.call('memory')
                with self.assertRaises(ValueError):broker_module.BrokerClient('http://example.com/rpc','token',os.getpid(),'x')
                with patch.dict(os.environ,{broker_module.PREFIX+'TOKEN':'only'},clear=True):
                    with self.assertRaises(RuntimeError):broker_module.client()
            finally:broker.close()

    def test_historical_generated_receipt_cannot_claim_live_pipeline_residency(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(prefix='beauty-residency-') as folder:
            root=Path(folder);(root/'model').mkdir()
            (root/'result.json').write_text(json.dumps({'ok':True,'generation_inferences_this_attempt':0}),encoding='utf8')
            (root/'model/result.json').write_text(json.dumps({'inference_completed':True}),encoding='utf8')
            service={'pipeline_loaded':False,'execution_profile':{},'code_identity':{},'process_log':'fixture.log'}
            pending=SimpleNamespace(terminal={'ok':True},output=root,service=service,reused=False,
                request_id='fixture',process=SimpleNamespace(pid=os.getpid()),started=time.perf_counter(),event_log=root/'events')
            prepare._collect_request_result(pending,time.perf_counter())
            self.assertFalse(service['pipeline_loaded'])
            pending.terminal['pipeline_loaded']=True
            prepare._collect_request_result(pending,time.perf_counter())
            self.assertTrue(service['pipeline_loaded'])

    def test_calibrated_cold_admission_binds_run_and_real_begin_effective_options(self):
        from types import SimpleNamespace
        from beauty_memory_calibration_smoke import fixture
        import beauty_memory_calibration as calibration
        with tempfile.TemporaryDirectory(prefix='beauty-broker-calibrated-') as folder:
            root=Path(folder);run,config,receipt,path=fixture(root)
            broker=broker_module.BeautyBroker(root)
            token,_=broker.issue('job-calibrated',run['run_id'],run['source']['sha256'])
            broker.register(token,os.getpid(),broker_module._stamp(os.getpid()))
            rpc=broker_module.BrokerClient(broker.url,token,broker.owner_pid,broker.owner_stamp)
            captured={}
            def begin(r,where,output,options,**kwargs):
                captured['options']=options
                pending=SimpleNamespace(request_id='fixture-request',snapshot=root/'request.json',event_log=root/'events',
                    process=SimpleNamespace(pid=os.getpid()))
                captured['pending']=pending;return pending
            def join(pending,**kwargs):
                return {'ok':True,'generation_inferences_this_attempt':0,'effective_max_faces':captured['options']['max_faces']}
            try:
                with patch.object(calibration,'calibration_path',return_value=path), \
                        patch.object(prepare,'load_installation',return_value=config), \
                        patch.object(prepare,'_BEAUTY_SERVICE',None), \
                        patch.object(prepare,'_begin_for_workbench_locked',side_effect=begin) as dispatch, \
                        patch.object(prepare,'join_for_workbench',side_effect=join), \
                        patch.object(prepare,'cleanup_for_workbench',return_value={'state':'fixture_completed'}):
                    wrong={**run,'run_id':'another-run'}
                    with self.assertRaises(RuntimeError):rpc.call('memory',run=wrong)
                    demand=rpc.call('memory',run=run)['demand']
                    self.assertEqual(demand['commit_bytes'],20*1024**3)
                    pending=rpc.begin(run,root,root/'outputs/runs'/run['run_id']/'portrait-beauty-preparation')
                    self.assertEqual(pending.join()['effective_max_faces'],1)
                    self.assertEqual(captured['options'],{'max_faces':1})
                    self.assertEqual(captured['pending'].memory_admission['calibration']['source_sha256'],run['source']['sha256'])
                    before=dispatch.call_count
                    with self.assertRaises(RuntimeError):rpc.begin(run,root,root/'outputs/runs'/run['run_id']/'retry',{'max_faces':8})
                    changed={**run,'options':{'native_source_retouch':{'different':True}}}
                    with self.assertRaises(RuntimeError):rpc.begin(changed,root,root/'outputs/runs'/run['run_id']/'retry')
                    self.assertEqual(dispatch.call_count,before)
                    broker.revoke(token)
            finally:broker.close()


if __name__=='__main__':unittest.main()
