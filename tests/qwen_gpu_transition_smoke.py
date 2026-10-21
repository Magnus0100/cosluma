"""Offline ownership/queue checks for the local generation-to-post transition."""
from pathlib import Path
import io,json,os,tempfile
from unittest.mock import patch
import start_qwen_image as qwen
import runtime_memory as memory

def main():
    with tempfile.TemporaryDirectory(prefix='cosplay-qwen-launch-') as temporary:
        deployment = Path(temporary)
        qwen.install_conditioning_node(deployment)
        assert not (deployment/'ComfyUI/custom_nodes/ComfyUI-DeGrid').exists()
        conditioning_node = deployment/'ComfyUI/custom_nodes/cosplay_qwen_conditioning.py'
        conditioning_stat = conditioning_node.stat().st_mtime_ns
        qwen.install_conditioning_node(deployment)
        assert conditioning_node.stat().st_mtime_ns == conditioning_stat
        with patch.dict(os.environ, {'PYTHONPATH':'foreign-runtime', 'PYTHONHOME':'foreign-runtime',
                                    'PYTHONUSERBASE':'foreign-runtime',
                                    'COSPLAY_CONFIG_ROOT':str(deployment/'config')}, clear=True):
            command, environment = qwen.service_launch(Path(temporary))
            assert ('--disable-pinned-memory' in command) == (os.name=='nt')
            assert '--disable-fast-disk' not in command
            assert '--fast-disk' not in command
            assert qwen.launch_policy() == {
                'schema':'cosplay_qwen_launch_policy.v4',
                'external_runtime_isolation':'frozen_bundle_dll_directory_and_path_v1',
                'pinned_memory':'disabled' if os.name=='nt' else 'upstream_default',
                'required_arguments':['--disable-pinned-memory'] if os.name=='nt' else [],
                'forbidden_arguments':['--disable-fast-disk']}
            assert not {'--highvram','--lowvram','--novram','--force-fp16','--force-fp32',
                        '--fp16-unet','--fp8_e4m3fn-unet','--disable-mmap'} & set(command)
            assert '--reserve-vram' not in command and '--bf16-text-enc' not in command
            assert not {'PYTHONPATH','PYTHONHOME','PYTHONUSERBASE'} & environment.keys()
            assert environment['PYTHONNOUSERSITE'] == '1'
            assert environment['OPENBLAS_NUM_THREADS'] == '1'
            assert 'OMP_NUM_THREADS' not in environment
            with patch.dict(os.environ, {'OPENBLAS_NUM_THREADS':'8', 'OMP_NUM_THREADS':'6'}):
                _, explicit_environment = qwen.service_launch(Path(temporary))
                assert explicit_environment['OPENBLAS_NUM_THREADS'] == '8'
                assert explicit_environment['OMP_NUM_THREADS'] == '6'
            for key in ('TEMP','TMP','HF_HOME','HF_HUB_CACHE','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR'):
                assert Path(environment[key]).is_relative_to(temporary)
            with patch.dict(os.environ, {'COSPLAY_QWEN_RESERVE_VRAM':'2'}):
                command, _ = qwen.service_launch(Path(temporary))
                assert command[-2:] == ['--reserve-vram','2.0']
            with patch.dict(os.environ, {'COSPLAY_QWEN_ATTENTION':'ck'}):
                command, _ = qwen.service_launch(Path(temporary))
                assert '--use-ck-attention' in command
            with patch.dict(os.environ, {'COSPLAY_QWEN_ATTENTION':'pytorch'}):
                command, _ = qwen.service_launch(Path(temporary))
                assert '--use-pytorch-cross-attention' in command
            with patch.dict(os.environ, {'COSPLAY_QWEN_ATTENTION':'invalid'}):
                try:qwen.service_launch(Path(temporary))
                except ValueError:pass
                else:raise AssertionError('Invalid attention override accepted')
            for invalid in ('-1','nan','inf','bad'):
                with patch.dict(os.environ, {'COSPLAY_QWEN_RESERVE_VRAM':invalid}):
                    try:qwen.service_launch(Path(temporary))
                    except ValueError:pass
                    else:raise AssertionError('Invalid VRAM reserve accepted')
    with tempfile.TemporaryDirectory(prefix='cosplay-qwen-handoff-') as temporary:
        folder = Path(temporary)
        python = (folder/'runtime/Scripts/python.exe').resolve()
        owner = {'pid':12345, 'endpoint':qwen.ENDPOINT, 'started_at':1000.,
                 'main':str((folder/'ComfyUI/main.py').resolve()), 'python':str(python)}
        (folder/'server-process.json').write_text(json.dumps(owner), encoding='utf8')
        status = {'system':{'argv':[owner['main']]}, 'devices':[
            {'type':'cuda','torch_vram_total':0,'torch_vram_free':0}]}
        run = {'options':{'provider':'qwen_image_2_1_fast'}}
        healthy = {'available_bytes':48*memory.GIB,'available_commit_bytes':48*memory.GIB,
                   'total_bytes':64*memory.GIB}
        pressure = {**healthy,'available_bytes':8*memory.GIB,'available_commit_bytes':2*memory.GIB}

        def exercise(*, before=healthy, after=healthy, queued_reads=0, stop_after=healthy,
                     stop_queued_reads=0,
                     active_initial=False, active_at_stop=False, foreign=False,
                     ownership_changed=False, gpu_busy=False, reused_pid=False,
                     dead_pid=False, wrong_python=False, unknown_created=False,
                     cpu_bytes=memory.BEAUTY_COLD_PHYSICAL_BYTES,
                     commit_bytes=memory.BEAUTY_COLD_COMMIT_BYTES,
                     admission_wait=0, stop_memory_ready_after=None, cancel_at=None):
            from contextlib import ExitStack
            from types import SimpleNamespace
            state = {'free':False,'stopped':False,'reads':0,'stop_reads':0,
                     'queues':0,'clock':0.,'stop_calls':0,'status_after_stop':0,'status_calls':0}
            calls = []
            def clock():return state['clock']
            def sleep(seconds):state['clock'] += seconds
            def snapshot():
                if state['stopped']:
                    state['stop_reads'] += 1
                    if stop_memory_ready_after is not None:
                        return stop_after if state['clock']-state['stopped_at'] >= stop_memory_ready_after else after
                    return after if state['stop_reads'] <= stop_queued_reads else stop_after
                if not state['free']:return before
                state['reads'] += 1
                return before if state['reads'] <= queued_reads else after
            def observed_status():
                state['status_calls'] += 1
                if state['stopped']:state['status_after_stop'] += 1
                observed = dict(status)
                if foreign or (ownership_changed and state['clock'] >= 3):
                    observed['system'] = {'argv':[str(folder/'foreign/main.py')]}
                if gpu_busy and not state['stopped']:
                    observed['devices'] = [{'type':'cuda','torch_vram_total':8*memory.GIB,
                                            'torch_vram_free':0}]
                return observed
            def open_request(request, **kwargs):
                url = request if isinstance(request,str) else request.full_url
                calls.append(url)
                if url.endswith('/queue'):
                    state['queues'] += 1
                    busy = active_initial or (active_at_stop and state['queues'] > 1)
                    return io.BytesIO(json.dumps({'queue_running':[1] if busy else [],
                                                'queue_pending':[]}).encode())
                assert url.endswith('/free')
                state['free'] = True
                required_physical = cpu_bytes + memory.CPU_RESERVE_BYTES
                required_commit = (cpu_bytes if commit_bytes is None else commit_bytes) + memory.CPU_RESERVE_BYTES
                fits = bool(before and before['available_bytes'] >= required_physical
                            and before['available_commit_bytes'] >= required_commit)
                assert json.loads(request.data) == {'unload_models':True, 'free_memory':not fits}
                return io.BytesIO(b'{}')
            def stop(*args, **kwargs):
                state['stop_calls'] += 1
                if os.name == 'nt':
                    assert args[0] == ['taskkill','/PID','12345','/T','/F']
                state['stopped'] = True
                state['stopped_at'] = state['clock']
                return SimpleNamespace(returncode=0)
            with ExitStack() as stack:
                stack.enter_context(patch.object(qwen,'service_status',side_effect=observed_status))
                stack.enter_context(patch.object(qwen,'deployment_root',return_value=folder))
                stack.enter_context(patch.object(qwen,'_listener_pid',return_value=12345))
                stack.enter_context(patch.object(qwen,'_process_main_matches',return_value=True))
                stack.enter_context(patch.object(qwen,'process_executable',
                    return_value=folder/'foreign/python.exe' if wrong_python else python))
                stack.enter_context(patch.object(qwen,'process_created_at',
                    return_value=None if unknown_created else (2000. if reused_pid else 1000.)))
                stack.enter_context(patch.object(qwen,'process_alive',
                    side_effect=lambda pid:not dead_pid and not state['stopped']))
                stack.enter_context(patch.object(qwen.OPENER,'open',side_effect=open_request))
                stack.enter_context(patch.object(qwen.time,'sleep',side_effect=sleep))
                stack.enter_context(patch.object(qwen.time,'monotonic',side_effect=clock))
                stack.enter_context(patch.object(qwen.time,'perf_counter',side_effect=clock))
                stack.enter_context(patch.object(memory,'memory_status',side_effect=snapshot))
                if os.name == 'nt':stack.enter_context(patch.object(qwen.subprocess,'run',side_effect=stop))
                else:stack.enter_context(patch.object(qwen.os,'kill',side_effect=stop))
                try:
                    result = qwen.release_for_run(run,next_stage_cpu_bytes=cpu_bytes,
                                                  next_stage_commit_bytes=commit_bytes,
                                                  next_stage='beautygrpo',memory_admission_wait=admission_wait,
                                                  cancel=(lambda:state['clock']>=cancel_at) if cancel_at is not None else None)
                except (RuntimeError,InterruptedError) as error:
                    result = error
            assert calls.count(qwen.ENDPOINT+'/free') <= 1, 'Release was resubmitted'
            assert state['status_after_stop'] == 0, 'Post-stop observation contacted the service'
            return result,state,calls

        receipt,state,calls = exercise()
        assert receipt['state'] == 'idle_models_released' and receipt['memory_state_verified']
        assert receipt['gpu_unload_verified'] and receipt['node_cache_preservation_requested']
        assert state['clock'] == 0 and state['stop_calls'] == 0
        assert state['status_calls'] == 2
        receipt,state,calls = exercise(admission_wait=10)
        assert 'memory_admission_wait' not in receipt and state['clock']==0

        receipt,state,calls = exercise(before=pressure,queued_reads=2)
        assert receipt['state'] == 'idle_models_released' and receipt['memory_state_verified']
        assert receipt['node_cache_reset_requested'] and state['clock'] > 0
        assert state['stop_calls'] == 0 and calls.count(qwen.ENDPOINT+'/free') == 1

        receipt,state,calls = exercise(before=pressure,after=pressure,queued_reads=100)
        assert receipt['state'] == 'idle_owned_service_stopped' and receipt['process_exit_confirmed']
        assert receipt['memory_after'] == healthy and receipt['budget_fits_after_stop']
        assert state['stop_calls'] == 1 and calls.count(qwen.ENDPOINT+'/free') == 1
        assert calls.count(qwen.ENDPOINT+'/queue') == 2
        assert receipt['post_stop_memory_samples'] == 1
        assert receipt['post_stop_observation_seconds'] == 0

        receipt,state,calls = exercise(before=pressure,after=pressure,stop_after=healthy,
            stop_memory_ready_after=5,admission_wait=10)
        wait=receipt['memory_admission_wait']
        assert wait['state']=='admitted' and wait['elapsed_seconds']>=2
        assert receipt['memory_state_verified'] and receipt['budget_fits_after_stop']
        assert wait['required_available_physical_bytes']==memory.BEAUTY_COLD_PHYSICAL_BYTES+memory.CPU_RESERVE_BYTES
        assert wait['required_available_commit_bytes']==memory.BEAUTY_COLD_COMMIT_BYTES+memory.CPU_RESERVE_BYTES
        assert state['stop_calls']==1 and calls.count(qwen.ENDPOINT+'/free')==1
        assert wait['free_requests']==wait['process_actions']==wait['model_submissions']==0
        error,state,calls = exercise(before=pressure,after=pressure,stop_after=healthy,
            stop_memory_ready_after=30,admission_wait=1)
        assert isinstance(error,RuntimeError) and 'memory budget is still unavailable' in str(error)
        assert error.handoff_receipt['memory_admission_wait']['state']=='timed_out'
        assert error.handoff_receipt['memory_admission_wait']['elapsed_seconds']==1
        assert not error.handoff_receipt['memory_state_verified']
        error,state,calls = exercise(before=pressure,after=pressure,stop_after=healthy,
            stop_memory_ready_after=30,admission_wait=10,cancel_at=6.5)
        assert isinstance(error,InterruptedError)
        assert error.handoff_receipt['memory_admission_wait']['state']=='cancelled'
        assert state['clock']<7 and calls.count(qwen.ENDPOINT+'/free')==1 and state['stop_calls']==1
        receipt,state,calls = exercise(before=pressure,after=pressure,stop_queued_reads=2)
        assert receipt['state'] == 'idle_owned_service_stopped' and receipt['budget_fits_after_stop']
        assert receipt['memory_after'] == healthy and receipt['process_exit_confirmed']
        assert receipt['post_stop_memory_samples'] == state['stop_reads'] == 3
        assert abs(receipt['post_stop_observation_seconds']-.3) < 1e-9
        assert state['stop_calls'] == 1 and calls.count(qwen.ENDPOINT+'/free') == 1
        assert calls.count(qwen.ENDPOINT+'/queue') == 2

        error,state,calls = exercise(before=pressure,after=pressure,stop_after=pressure)
        assert isinstance(error,RuntimeError) and 'memory budget is still unavailable' in str(error)
        assert not error.handoff_receipt['budget_fits_after_stop']
        assert error.handoff_receipt['process_exit_confirmed'] and state['stop_calls'] == 1
        assert error.handoff_receipt['memory_after'] == pressure
        assert error.handoff_receipt['post_stop_memory_samples'] == state['stop_reads'] > 1
        assert error.handoff_receipt['post_stop_observation_seconds'] == 3
        assert calls.count(qwen.ENDPOINT+'/free') == 1
        assert calls.count(qwen.ENDPOINT+'/queue') == 2

        for kwargs in ({'active_initial':True}, {'before':pressure,'after':pressure,'active_at_stop':True}):
            error,state,calls = exercise(admission_wait=10,**kwargs)
            assert isinstance(error,RuntimeError) and 'still generating' in str(error)
            assert state['stop_calls'] == 0
            if kwargs.get('active_initial'):assert qwen.ENDPOINT+'/free' not in calls

        receipt,state,calls = exercise(foreign=True)
        assert receipt['state'] == 'unowned_service_untouched' and not calls
        assert state['stop_calls'] == 0 and state['status_calls'] == 1
        for kwargs in ({'reused_pid':True}, {'wrong_python':True}, {'unknown_created':True}):
            receipt,state,calls = exercise(**kwargs)
            assert receipt['state'] == 'unowned_service_untouched' and not calls
            assert state['stop_calls'] == 0 and state['status_calls'] == 0
        receipt,state,calls = exercise(dead_pid=True)
        assert receipt['state'] == 'not_running' and receipt['memory_state_verified'] and not calls
        assert state['stop_calls'] == 0 and state['status_calls'] == 0
        for marker in (None, {key:value for key,value in owner.items() if key != 'pid'}):
            marker_path = folder/'server-process.json'
            if marker is None:marker_path.unlink()
            else:marker_path.write_text(json.dumps(marker),encoding='utf8')
            receipt,state,calls = exercise()
            assert receipt['state'] == 'not_running' and receipt['memory_state_verified'] and not calls
            assert state['stop_calls'] == 0 and state['status_calls'] == 0
        marker_path.write_text(json.dumps(owner),encoding='utf8')
        error,state,calls = exercise(dead_pid=True,before=pressure)
        assert isinstance(error,RuntimeError) and not error.handoff_receipt['memory_state_verified']
        assert error.handoff_receipt['state'] == 'not_running' and not calls
        assert state['status_calls'] == 0
        error,state,calls = exercise(before=pressure,after=pressure,ownership_changed=True)
        assert isinstance(error,RuntimeError) and 'ownership changed' in str(error)
        assert state['stop_calls'] == 0

        receipt,state,calls = exercise(gpu_busy=True)
        assert receipt['state'] == 'idle_owned_service_stopped' and receipt['gpu_unload_verified']
        assert receipt['stop_reason'] == 'gpu_unload_not_observed' and state['stop_calls'] == 1

        asymmetric = {**healthy, 'available_bytes':40*memory.GIB, 'available_commit_bytes':28*memory.GIB}
        requirements = {'cpu_bytes':32*memory.GIB, 'commit_bytes':23*memory.GIB}
        receipt,state,calls = exercise(before=asymmetric,after=asymmetric,**requirements)
        assert receipt['state'] == 'idle_models_released' and state['stop_calls'] == 0
        assert receipt['node_cache_preservation_requested'] and receipt['memory_state_verified']
        assert receipt['cpu_budget']['required_available_physical_bytes'] == 32*memory.GIB+memory.CPU_RESERVE_BYTES
        assert receipt['cpu_budget']['required_available_commit_bytes'] == 23*memory.GIB+memory.CPU_RESERVE_BYTES
        low_physical = {**asymmetric, 'available_bytes':32*memory.GIB+memory.CPU_RESERVE_BYTES-1}
        low_commit = {**asymmetric, 'available_commit_bytes':23*memory.GIB+memory.CPU_RESERVE_BYTES-1}
        for insufficient in (low_physical,low_commit,None):
            receipt,state,calls = exercise(before=insufficient,after=insufficient,
                                           stop_after=asymmetric,**requirements)
            assert receipt['state'] == 'idle_owned_service_stopped' and state['stop_calls'] == 1
            assert receipt['node_cache_reset_requested'] and receipt['budget_fits_after_stop']
            error,state,calls = exercise(before=insufficient,after=insufficient,
                                        stop_after=insufficient,**requirements)
            assert isinstance(error,RuntimeError) and not error.handoff_receipt['memory_state_verified']
            assert state['stop_calls'] == 1 and calls.count(qwen.ENDPOINT+'/free') == 1
    from contextlib import ExitStack
    gpt_run = {'options':{'provider':'gpt_image'}}
    cold_demand = {'next_stage_cpu_bytes':memory.BEAUTY_COLD_PHYSICAL_BYTES,
                   'next_stage_commit_bytes':memory.BEAUTY_COLD_COMMIT_BYTES,
                   'next_stage':'beautygrpo'}
    low_physical = {**healthy, 'available_bytes':memory.BEAUTY_COLD_PHYSICAL_BYTES+memory.CPU_RESERVE_BYTES-1}
    low_commit = {**healthy, 'available_commit_bytes':memory.BEAUTY_COLD_COMMIT_BYTES+memory.CPU_RESERVE_BYTES-1}
    with ExitStack() as stack:
        untouched = [stack.enter_context(patch.object(target, name,
                     side_effect=AssertionError('GPT handoff touched local service or HTTP')))
                     for target,name in ((qwen,'service_status'),(qwen,'start_service'),
                                         (qwen.OPENER,'open'),(qwen.subprocess,'Popen'),
                                         (qwen.subprocess,'run'),(qwen.os,'kill'))]
        for insufficient in (low_physical,low_commit,None):
            with patch.object(memory,'memory_status',return_value=insufficient):
                try:qwen.release_for_run(gpt_run,**cold_demand)
                except RuntimeError as error:
                    assert 'before model startup or image submission' in str(error)
                    receipt = error.handoff_receipt
                    assert receipt['state'] == 'not_managed_qwen'
                    assert not receipt['memory_state_verified'] and not receipt['local_service_touched']
                    assert receipt['free_requests'] == 0 and receipt['cpu_budget']['snapshot'] == insufficient
                    assert receipt['cpu_budget']['next_stage_cpu_bytes'] == memory.BEAUTY_COLD_PHYSICAL_BYTES
                    assert receipt['cpu_budget']['next_stage_commit_bytes'] == memory.BEAUTY_COLD_COMMIT_BYTES
                    assert receipt['cpu_budget']['reason'] == ('memory_unknown' if insufficient is None
                                                              else 'insufficient_available_memory')
                else:raise AssertionError('GPT admitted an unavailable cold Beauty budget')
        with patch.object(memory,'memory_status',return_value=healthy):
            receipt = qwen.release_for_run(gpt_run,**cold_demand)
            assert receipt['state'] == 'not_managed_qwen' and receipt['memory_state_verified']
            assert not receipt['local_service_touched'] and receipt['free_requests'] == 0
            assert qwen.release_for_run({'options':{'provider':'manual_import'}})['state'] == 'not_managed_qwen'
            assert qwen.release_for_run({'options':{'provider':'qwen_image_2_1_fast',
                'api_base':'https://external.invalid'}})['state'] == 'external_service_untouched'
        warm_headroom = {**pressure, 'available_commit_bytes':8*memory.GIB}
        with patch.object(memory,'memory_status',return_value=warm_headroom):
            receipt = qwen.release_for_run(gpt_run,next_stage_cpu_bytes=0,next_stage_commit_bytes=0,
                                           next_stage='beautygrpo')
            # A warm Beauty adds no allocation, but still keeps the system reserve.
            assert receipt['cpu_budget']['next_stage_cpu_bytes'] == 0
            assert receipt['cpu_budget']['next_stage_commit_bytes'] == 0
            assert receipt['memory_state_verified']
        with patch.object(memory,'memory_status',return_value=low_commit):
            assert qwen.release_for_run(gpt_run,next_stage_cpu_bytes=32*memory.GIB,
                                       next_stage='local_post') == {'state':'not_managed_qwen'}
        for operation in untouched:operation.assert_not_called()
    # No releasable service: wait only for real, jointly adequate OS headroom.
    import workflow_timing
    no_service={'options':{'provider':'manual_import'}}
    near_commit={**healthy,'available_commit_bytes':memory.BEAUTY_COLD_COMMIT_BYTES+memory.CPU_RESERVE_BYTES-38*1024**2}
    low_only_physical={**healthy,'available_bytes':memory.BEAUTY_COLD_PHYSICAL_BYTES+memory.CPU_RESERVE_BYTES-1}
    for samples,limit,cancel_at,expected in (
        ([near_commit,near_commit,low_only_physical,healthy],3,None,'admitted'),
        ([None],1,None,'timed_out'),
        ([near_commit],3,.25,'cancelled')):
        with tempfile.TemporaryDirectory(prefix='admission-wait-') as folder, ExitStack() as stack:
            clock={'value':0.,'reads':0,'heartbeat':False}
            def observed_memory():
                position=min(clock['reads'],len(samples)-1);clock['reads']+=1
                return samples[position]
            def observe_sleep(seconds):
                assert 0<seconds<=1
                clock['value']+=seconds
                ledger=workflow_timing.current()
                if ledger:
                    live=json.loads(ledger.path.read_text('utf8'))
                    clock['heartbeat'] |= any(row['name']=='memory_admission_wait' and row['status']=='running' for row in live['spans'])
            stack.enter_context(patch.object(qwen.time,'monotonic',side_effect=lambda:clock['value']))
            stack.enter_context(patch.object(qwen.time,'perf_counter',side_effect=lambda:clock['value']))
            stack.enter_context(patch.object(qwen.time,'sleep',side_effect=observe_sleep))
            stack.enter_context(patch.object(memory,'memory_status',side_effect=observed_memory))
            for target,name in ((qwen,'service_status'),(qwen,'start_service'),(qwen.OPENER,'open'),
                                (qwen.subprocess,'Popen'),(qwen.subprocess,'run'),(qwen.os,'kill')):
                stack.enter_context(patch.object(target,name,side_effect=AssertionError('Waiting touched a service or process')))
            with workflow_timing.request(Path(folder),'admission-test'):
                try:
                    result=qwen.release_for_run(no_service,**cold_demand,memory_admission_wait=limit,
                        cancel=(lambda:clock['value']>=cancel_at) if cancel_at is not None else None)
                except (RuntimeError,InterruptedError) as error:
                    result=error.handoff_receipt
                    assert isinstance(error,InterruptedError)==(expected=='cancelled')
                assert result['memory_admission_wait']['state']==expected
                assert result['free_requests']==0 and not result['local_service_touched']
                assert result['memory_state_verified']==(expected=='admitted')
                assert clock['value']<=limit and clock['heartbeat']
                if expected=='admitted':assert result['memory_after']==healthy and clock['reads']==4
            saved=json.loads(next((Path(folder)/'outputs/timings').glob('*.json')).read_text('utf8'))
            stage=next(row for row in saved['spans'] if row['name']=='memory_admission_wait')
            assert stage['inclusive_seconds']==clock['value']
            assert workflow_timing.MODULE_LABELS['memory_admission_wait']=='等待系统内存回收'
    # Production opts into the wait and carries the cancellation marker/context.
    import single_source_workflow as workflow
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory(prefix='beauty-admission-') as folder:
        root=Path(folder);marker=root/'cancel';settings=SimpleNamespace(root=root,cancel_request_path=str(marker))
        run={'root':str(root),'options':{}}
        demand={'physical_bytes':memory.BEAUTY_COLD_PHYSICAL_BYTES,'commit_bytes':memory.BEAUTY_COLD_COMMIT_BYTES}
        with patch('beautygrpo_prepare.beauty_memory_needed',return_value=demand) as needed, \
                patch.object(qwen,'release_for_run',return_value={'state':'not_running','memory_state_verified':True}) as release:
            workflow._release_task_beauty_memory(run,settings)
            assert release.call_args.kwargs['memory_admission_wait']==60
            assert release.call_args.kwargs['cancel']() is False
            assert needed.call_args.kwargs=={'run':run,'root':str(root)}
            marker.write_text('cancel')
            assert release.call_args.kwargs['cancel']() is True
        def cancelled(*args,**kwargs):
            error=InterruptedError('CPU fixture cancellation')
            error.handoff_receipt={'memory_state_verified':False,'memory_admission_wait':{'state':'cancelled'}}
            raise error
        with patch('beautygrpo_prepare.beauty_memory_needed',return_value=demand), \
                patch.object(qwen,'release_for_run',side_effect=cancelled):
            try:workflow._release_task_beauty_memory(run,settings)
            except workflow.WorkflowCancelled:pass
            else:raise AssertionError('Production memory wait cancellation did not use the job cancellation path')
            assert run['options']['portrait_beauty_memory_transition']['memory_admission_wait']['state']=='cancelled'
    for invalid in (-1,float('nan'),float('inf'),True,'60'):
        try:qwen.release_for_run(no_service,memory_admission_wait=invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid memory-admission timeout accepted')
    with patch.object(memory,'memory_status',return_value=None):
        assert memory.memory_budget(1)['reason']=='memory_unknown'
    print(json.dumps({'ok':True,'unowned_and_remote_untouched':True,'busy_generation_rejected':True,
                      'absent_dead_or_mismatched_process_before_http':True,
                      'idle_model_handoff':True,'node_cache_preservation_requested':True,
                      'queued_release_completion_verified':True,'owned_stop_budget_verified':True,
                      'delayed_post_stop_memory_recovery_verified':True,
                      'independent_physical_commit_thresholds':True,'unknown_memory_rejected':True,
                      'gpt_cold_budget_before_start_or_http':True,'warm_zero_demand_preserved':True,
                      'bounded_cancelable_OS_memory_admission':True,'wait_does_not_resubmit_or_release':True,
                      'independent_cli_timing_heartbeat':True,'production_wait_context_and_cancel':True,
                      'late_active_queue_protected':True,'hardware_defaults_and_isolated_storage':True}))

if __name__=='__main__':main()
