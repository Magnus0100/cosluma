"""CPU-only Qwen ownership/launch attestation; never load or submit a model."""
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import beautygrpo_broker as broker
import qwen_norm_compat as compat
import start_qwen_image as qwen


NORM = '''import torch
class Attention:
    def forward(self, q, k, pe, q_scale, k_scale):
        if True:
            q, k = comfy.quant_ops.ck.rms_rope(q, k, pe, q_scale, k_scale, self.norm_q.eps)
        return q, k
'''

SERVER = '''import json,sys,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
folder=Path(__file__).parent.parent
config=json.loads((folder/'fixture.json').read_text())
time.sleep(config.get('delay',0))
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        if self.path=='/system_stats':body={'system':{'argv':[__file__]},'devices':[]}
        elif self.path=='/queue':body={'queue_running':[1] if (folder/'busy').exists() else [],'queue_pending':[]}
        else:raise AssertionError('Unexpected HTTP request '+self.path)
        data=json.dumps(body).encode();self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_POST(self):raise AssertionError('CPU ownership fixture forbids POST')
ThreadingHTTPServer(('127.0.0.1',config['port']),Handler).serve_forever()
'''


def rejected(call, contains):
    try:call()
    except RuntimeError as error:assert contains in str(error), str(error)
    else:raise AssertionError('Unverified service accepted: '+contains)


def real_launch_cases():
    original_popen=subprocess.Popen
    original_status=qwen.service_status
    processes=[]
    with tempfile.TemporaryDirectory(prefix='cosplay-qwen-owner-') as temporary, ExitStack() as stack:
        folder=Path(temporary)
        python=folder/'runtime/Scripts/python.exe';python.parent.mkdir(parents=True);python.touch()
        base_python=qwen.process_executable(os.getpid())
        (folder/'runtime/pyvenv.cfg').write_text('executable = '+str(base_python)+'\n','utf8')
        main=folder/'ComfyUI/main.py';main.parent.mkdir();main.write_text(SERVER,'utf8')
        target=folder/'ComfyUI'/compat.TARGET;target.parent.mkdir(parents=True);target.write_bytes(NORM.encode())
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
        endpoint='http://127.0.0.1:'+str(port)
        (folder/'fixture.json').write_text(json.dumps({'port':port,'delay':.1}),'utf8')
        calls=[]
        def launch(command,**kwargs):
            if str(main) not in command:return original_popen(command,**kwargs)
            # Actual subprocess with actual OS listener/creation/argv proof.
            assert compat.validate_qwen_norm_compat(folder/'ComfyUI',qwen._prepare_runtime_compat(folder))
            assert 'cosplay_qwen_launch=' in ' '.join(command)
            process=original_popen(command,**kwargs);processes.append(process);calls.append(command)
            return process
        stack.enter_context(patch.object(qwen,'ENDPOINT',endpoint))
        stack.enter_context(patch.object(qwen,'service_status',side_effect=lambda:original_status(endpoint)))
        stack.enter_context(patch.object(qwen,'service_launch',return_value=(
            [str(base_python),'-s',str(main),*qwen.launch_policy()['required_arguments']],dict(os.environ))))
        stack.enter_context(patch.object(qwen,'install_conditioning_node'))
        stack.enter_context(patch.object(compat,'SOURCE_SHA256',compat._sha(NORM.encode())))
        stack.enter_context(patch.object(compat,'PATCHED_SHA256',compat._sha(compat._patch_bytes(NORM.encode()))))
        stack.enter_context(patch.object(qwen.subprocess,'Popen',side_effect=launch))
        try:
            # Cancel immediately after launch identity is durably captured.
            # The caller does not own the shared service's lifetime.
            try:qwen.start_service(folder,cancel=lambda:bool(calls))
            except InterruptedError:pass
            else:raise AssertionError('Post-launch cancellation ignored')
            pending=qwen._read_owner(folder)
            assert pending['state']=='starting' and qwen._same_process(pending['launcher'])
            assert len(calls)==1
            with patch.object(qwen,'deployment_root',return_value=folder), \
                 patch.object(qwen,'model_files',return_value=[]), ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(lambda _:qwen.ensure_running(endpoint=endpoint),range(2)))
            first=results[0]
            assert all(result['managed'] for result in results) and len(calls)==1
            saved=qwen._read_owner(folder)
            assert first['managed'] and saved['state']=='ready'
            assert saved['pid']==qwen._listener_pid()==processes[0].pid
            assert saved['server']==qwen._process_identity(processes[0].pid)
            assert saved['ancestry']==[saved['launcher']]
            assert qwen._owned_service_record(folder,qwen.service_status())==saved
            assert qwen._compatibility_current(folder,saved)
            assert saved['launch_policy']==first['launch_policy']==qwen.launch_policy()
            assert qwen._launch_policy_current(saved)
            assert qwen.start_service(folder)['already_running'] and len(calls)==1
            # Missing/stale policy cannot be added retroactively to a live launch.
            legacy_policy = {key:value for key,value in qwen.launch_policy().items()
                             if key != 'external_runtime_isolation'}
            legacy_policy['schema'] = 'cosplay_qwen_launch_policy.v3'
            for policy in (None, {**qwen.launch_policy(),'pinned_memory':'unverified'},
                           legacy_policy,
                           {**qwen.launch_policy(),'external_runtime_isolation':'unverified'}):
                old=dict(saved)
                if policy is None:old.pop('launch_policy')
                else:old['launch_policy']=policy
                qwen._write_owner(folder/'server-process.json',old)
                assert qwen._owned_service_record(folder,qwen.service_status())==old
                with patch.object(qwen,'_stop_owned_idle_service',side_effect=RuntimeError('policy restart required')):
                    rejected(lambda:qwen.start_service(folder),'policy restart required')
                assert qwen._read_owner(folder) == old, 'A live launch must never be re-signed'
                (folder/'busy').touch()
                rejected(lambda:qwen.start_service(folder),'still generating')
                assert qwen.process_alive(saved['pid']) and len(calls)==1
                assert qwen._read_owner(folder) == old, 'Busy legacy policy must remain unchanged'
                (folder/'busy').unlink()
            qwen._write_owner(folder/'server-process.json',saved)
            # A matching HTTP deployment path alone cannot claim the endpoint.
            for marker in ({},dict(saved,pid=saved['pid']+100000),
                           dict(saved,server=dict(saved['server'],created_at=1)),
                           dict(saved,launch_id='0'*32),
                           dict(saved,ancestry=[])):
                qwen._write_owner(folder/'server-process.json',marker)
                assert qwen.start_service(folder)['managed'] is False
            qwen._write_owner(folder/'server-process.json',saved)
            # Disk patch is valid, but a legacy live launch has no load attestation.
            legacy={key:saved[key] for key in ('pid','endpoint','python','main','started_at')}
            legacy['python']=saved['server']['executable']
            legacy['compatibility']=saved['compatibility']
            qwen._write_owner(folder/'server-process.json',legacy)
            assert qwen._owned_service_record(folder,qwen.service_status())==legacy
            assert not qwen._compatibility_current(folder,legacy)
            (folder/'busy').touch()
            with patch.object(qwen,'_stop_owned_idle_service',wraps=qwen._stop_owned_idle_service) as stop:
                rejected(lambda:qwen.start_service(folder),'still generating')
                assert stop.call_count==1 and qwen.process_alive(saved['pid'])
            (folder/'busy').unlink()
            # Stop/restart paths are mocked: never kill a service outside this fixture.
            with patch.object(qwen,'_stop_owned_idle_service',side_effect=RuntimeError('owned idle restart required')):
                rejected(lambda:qwen.start_service(folder),'owned idle restart required')
            qwen._write_owner(folder/'server-process.json',saved)
            # Cancellation does not stop a shared live process.
            try:qwen.start_service(folder,cancel=lambda:True)
            except InterruptedError:pass
            else:raise AssertionError('Cancellation ignored')
            assert qwen.process_alive(saved['pid']) and len(calls)==1
            # Restore a captured pending launch, not a new Popen, after caller cancellation.
            pending={key:value for key,value in saved.items() if key not in {'server','ancestry'}}
            pending['state']='starting';qwen._write_owner(folder/'server-process.json',pending)
            assert qwen.start_service(folder)['managed'] and len(calls)==1
            # Invalid on-disk compatibility cannot attest the running process.
            target.write_bytes(target.read_bytes()+b'\n# changed after launch\n')
            with patch.object(qwen,'_stop_owned_idle_service',side_effect=RuntimeError('compat restart required')):
                rejected(lambda:qwen.start_service(folder),'compat restart required')
        finally:
            for process in processes:
                if process.poll() is None:process.terminate()
                process.wait(timeout=10)


def simulated_wrapper_cases():
    # Windows venv launcher and base-python listener have different PIDs/exes.
    with tempfile.TemporaryDirectory(prefix='cosplay-qwen-wrapper-') as temporary, ExitStack() as stack:
        folder=Path(temporary);python=folder/'runtime/Scripts/python.exe'
        python.parent.mkdir(parents=True);python.touch()
        main=folder/'ComfyUI/main.py';main.parent.mkdir();main.touch()
        base=folder/'base/python.exe';base.parent.mkdir();base.touch()
        (folder/'runtime/pyvenv.cfg').write_text('executable = '+str(base)+'\n','utf8')
        launcher={'pid':100,'created_at':1000.,'executable':str(python.resolve())}
        server={'pid':101,'created_at':1001.,'executable':str(base.resolve())}
        nonce='a'*32
        identities={100:launcher,101:server}
        args={pid:[identity['executable'],'-X','cosplay_qwen_launch='+nonce,str(main)] for pid,identity in identities.items()}
        stack.enter_context(patch.object(qwen,'_process_identity',side_effect=lambda pid:identities.get(pid)))
        stack.enter_context(patch.object(qwen,'process_created_at',side_effect=lambda pid:(identities.get(pid) or {}).get('created_at')))
        stack.enter_context(patch.object(qwen,'process_alive',side_effect=lambda pid:pid in identities))
        stack.enter_context(patch.object(qwen,'_process_arguments',side_effect=lambda pid:args.get(pid)))
        stack.enter_context(patch.object(qwen,'_listener_pid',return_value=101))
        parents={101:100};stack.enter_context(patch.object(broker,'_process_parents',return_value=parents))
        record={'schema':'cosplay_qwen_service_owner.v2','state':'ready','pid':101,
                'launcher':launcher,'server':server,'ancestry':[server,launcher],
                'launch_id':nonce,'endpoint':qwen.ENDPOINT,'python':str(python.resolve()),
                'runtime_root':str((folder/'runtime').resolve()),'main':str(main.resolve())}
        qwen._write_owner(folder/'server-process.json',record)
        assert qwen._owned_process_record(folder)[0]==record
        # Even a new-looking v2 record cannot claim a policy absent from OS argv.
        record['launch_policy']=qwen.launch_policy()
        # This process-chain fixture models the Windows venv launcher.
        stack.enter_context(patch.object(qwen,'launch_policy',return_value={
            'schema':'cosplay_qwen_launch_policy.v4',
            'external_runtime_isolation':'frozen_bundle_dll_directory_and_path_v1',
            'pinned_memory':'disabled',
            'required_arguments':['--disable-pinned-memory'],
            'forbidden_arguments':['--disable-fast-disk']}))
        record['launch_policy']=qwen.launch_policy()
        assert not qwen._launch_policy_current(record)
        args[101][1:1]=['-X','--disable-pinned-memory']
        assert not qwen._launch_policy_current(record), 'Python option is not a ComfyUI flag'
        del args[101][1:3]
        args[100].append('--disable-pinned-memory')
        assert not qwen._launch_policy_current(record), 'Only the launcher had the flag'
        args[101].append('--disable-pinned-memory')
        assert qwen._launch_policy_current(record)
        args[101].append('--disable-fast-disk')
        assert not qwen._launch_policy_current(record), 'A stale forced-RAM policy cannot be hidden by a new marker'
        args[101].pop()
        for change in ('parent','nonce','reused','foreign'):
            if change=='parent':parents[101]=999
            if change=='nonce':args[101][2]='cosplay_qwen_launch='+'b'*32
            if change=='reused':identities[101]={**server,'created_at':2000.}
            if change=='foreign':identities[101]={**server,'executable':str(folder/'foreign.exe')}
            assert qwen._owned_process_record(folder)[0] is None,change
            parents[101]=100;args[101][2]='cosplay_qwen_launch='+nonce;identities[101]=server
        # A dead saved PID cannot inherit a live endpoint.
        identities.clear();assert qwen._owned_process_record(folder)[1]=='not_running'


def submission_gate_cases():
    with tempfile.TemporaryDirectory(prefix='cosplay-qwen-submit-gate-') as temporary:
        folder=Path(temporary)
        with patch.object(qwen,'deployment_root',return_value=folder), \
             patch.object(qwen,'model_files',return_value=[]), \
             patch.object(qwen,'start_service',return_value={'managed':False,'reason':'foreign'}) as start:
            rejected(lambda:qwen.ensure_running(),'ownership is unverified')
            assert start.call_count==1
        with patch.object(qwen,'start_service',side_effect=AssertionError('External service was touched')):
            assert qwen.ensure_running(endpoint='https://operator.invalid')['managed'] is False


def main():
    simulated_wrapper_cases();real_launch_cases();submission_gate_cases()
    print(json.dumps({'ok':True,'real_CPU_listener_launch_identity':True,'venv_wrapper_parent_chain':True,
                      'foreign_dead_reused_busy_untouched':True,'cold_norm_compat_attestation':True,
                      'cancel_pending_without_duplicate_start':True,'models_or_prompts':0}))


if __name__=='__main__':main()
