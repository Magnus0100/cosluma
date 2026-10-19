"""Foreground staging retains full bound assets and current session receipts."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import source_foreground as foreground
import source_foreground_cache as cache
import workflow_timing


def main():
    with tempfile.TemporaryDirectory(prefix='foreground-stage-handoff-') as temporary:
        root = Path(temporary)
        source, saliency = root/'source.png', root/'saliency.png'
        Image.new('RGB',(8,6),(40,80,120)).save(source)
        Image.new('L',(8,6),128).save(saliency)
        model = root/'model';model.mkdir()
        for name in ('model.safetensors','config.json','preprocessor_config.json'):
            (model/name).write_bytes(('identity-only CPU fixture '+name).encode())
        runtime = {'python':Path('D:/selected-runtime/python.exe'),
            'vitmatte_model_root':model,'matting_profile':'vitmatte_small_distinctions'}
        cancelled = lambda:False
        calls = []
        timeline = {'whole_wall_seconds':.5,'spans':[]}
        def execute(python, selected_root, command, directory, cancel, **kwargs):
            calls.append((python, selected_root, command, directory, cancel, kwargs))
            assert python == runtime['python'] and selected_root == root
            assert Path(command[0]).name == 'source_foreground.py'
            assert command[command.index('--source')+1] == source.resolve()
            assert command[command.index('--saliency')+1] == saliency.resolve()
            assert cancel is cancelled and kwargs['stage'] == 'source_foreground'
            alpha = np.full((6,8),.5,np.float32)
            rgb = np.full((6,8,3),.25,np.float32)
            paths = foreground.save_foreground_arrays(directory,[('alpha',alpha),('old-background',rgb),
                ('premultiplied-foreground',rgb),('unknown-edge-support',alpha)])
            sparse = {'flat_index':np.array([0],np.uint32),'fit_error':np.zeros(1,np.float32),
                'separation':np.ones(1,np.float32),'background_projection_delta':np.zeros((1,3),np.float32),
                'overlap_weight':np.ones(1,np.float32),'canvas_size':np.array([8,6],np.int32),
                'fb_normalization_scale':np.array(1,np.float32)}
            descriptor = foreground._write_fb_diagnostics(directory,sparse,source,paths)
            record = {'schema_version':'source_foreground.v1',**paths,
                'fb_diagnostics_path':descriptor['path'],'fb_diagnostics':descriptor,
                'native_values_encoding':foreground.VALUES_ENCODING,'analysis_size':[8,6],
                'provenance':{},'worker_timing':'unused-stale-result-locator'}
            (directory/'result.json').write_text(json.dumps(record),'utf8')
            return {'wall_seconds':.6,'worker_pid':4321,'worker_timeline':timeline,
                    'native_field_execution':{'backend':'fixture-only'},
                    'native_libraw_threading':{'threads':8,'library':'fixture-vcomp'}}
        with patch.object(cache,'local_runtime_paths',return_value=runtime), \
                patch('reference_stage_worker.execute_stage',side_effect=execute):
            with workflow_timing.request(root,'handoff') as ledger:
                result = cache.prepare_source_foreground(root,source,saliency_path=saliency,cancel=cancelled)
                assert result['cache_hit'] is False and len(calls) == 1
                assert cache._valid_assets(result,result['identity'])
                worker = ledger.workers[-1]
                assert worker['name'] == 'source_foreground_worker' and worker['execution_seconds'] == .6
                assert worker['metadata']['worker_pid'] == 4321
                assert worker['metadata']['worker_timeline'] == timeline
                assert worker['metadata']['native_libraw_threading']['threads'] == 8
                reused = cache.prepare_source_foreground(root,source,saliency_path=saliency,cancel=cancelled)
                assert reused['cache_hit'] and len(calls) == 1
                assert cache._valid_assets(reused,reused['identity'])
            try:
                cache.prepare_source_foreground(root,source,saliency_path=saliency,cancel=lambda:True)
            except InterruptedError:pass
            else:raise AssertionError('Cancelled foreground cache request was admitted')
            assert len(calls) == 1
        print(json.dumps({'ok':True,'selected_runtime_root_and_cancel_propagated':True,
            'fresh_native_targets_and_diagnostics_SHA_preserved':True,
            'actual_current_stage_timeline_and_PID_recorded_once':True,
            'complete_bound_bundle_cache_hit_does_not_start_worker':True,'models_GPU_network_used':False}))


if __name__ == '__main__':main()
