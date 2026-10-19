"""Exercise the sole native T+B CLI and its passive timing with tiny CPU data."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import patch

import numpy as np

import native_source_baseline
import portrait_beauty_layer
import render_reference_colour_composite as renderer
import reference_pre_scene_cache
from reference_colour_fit import fit, REGIONS
from reference_colour_blend import FEATURE_CHROMA_POLICY
from runtime_resources import code_path
import workflow_timing


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report_path=None):
    with tempfile.TemporaryDirectory(prefix='cosplay_native_baseline_cli_') as temporary:
        root=Path(temporary);directory=root/'matching';directory.mkdir()
        native=(16,269)
        alpha=np.resize(np.array([0.,.5,1.],np.float32),native[0]*native[1]).reshape(native[1],native[0])
        person=np.full((*alpha.shape,3),[.2,.3,.4],np.float32)
        old_background=np.full_like(person,[.08,.1,.12])
        foreground=alpha[...,None]*person
        source=foreground+(1-alpha[...,None])*old_background
        selected=np.full_like(person,[.16,.18,.2])
        expected=foreground+(1-alpha[...,None])*selected
        def asset(name,values):
            path=root/(name+'.npz');np.savez_compressed(path,values=values)
            return {'path':str(path),'sha256':sha(path)}
        alpha_asset=asset('alpha',alpha)
        background_asset=asset('selected',selected)
        validity_asset=asset('validity',np.ones_like(alpha))
        weights=np.zeros((*alpha.shape,len(REGIONS)),np.float32)
        weights_path=directory/'source-weights.npz';np.savez_compressed(weights_path,values=weights)
        source_path=root/'source.bytes';source_path.write_bytes(b'bound CPU source fixture')
        reference_path=root/'reference.bytes';reference_path.write_bytes(b'bound CPU reference fixture')
        parameters=fit([])
        parameters['region_order']=list(REGIONS)
        parameters['binding']={'source':sha(source_path),'reference':sha(reference_path),
            'alpha':alpha_asset['sha256'],'source_protection':'fixture-no-protection','mapping':{'fixture':'same_grid'},
            'algorithm_sha256':sha(code_path('reference_colour_fit.py')),'source_weights_sha256':sha(weights_path)}
        parameters['protection_policy']={'fixture':'no authored changes'}
        (directory/'parameters.json').write_text(json.dumps(parameters),encoding='utf8')
        (directory/'validation.json').write_text(json.dumps({'native_crop_boxes':{'all':[0,0,*native]}}),encoding='utf8')
        colour=np.zeros((*alpha.shape,6),np.float32);colour[...,2]=1
        colour_asset=asset('colour-field',colour)
        colour_receipt={'schema':'reference_spatial_colour.v2','working_domain':'display_linear_sRGB',
            'feature_chroma_policy':FEATURE_CHROMA_POLICY,
            'native_size':list(native),'proxy_size':list(native),'regions':{},'asset':colour_asset,
            'algorithm_sha256':sha(code_path('reference_colour_blend.py')),
            'binding':{**{role:parameters['binding'][role] for role in ('source','reference','alpha','source_protection','mapping')},
                'parameters_sha256':hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()}}
        colour_path=directory/'reference-colour-layer.json';colour_path.write_text(json.dumps(colour_receipt),encoding='utf8')
        plan={'canvas':{'source_proxy_size':list(native),'source_oriented_size':list(native)},
            'source_baseline':{'assets':{'foreground_alpha':alpha_asset}},
            'layers':[{'kind':'background_design','assets':{'background':background_asset,'validity':validity_asset}}]}
        run={'run_id':'small-native-baseline','source':{'path':str(source_path),'sha256':sha(source_path)},
            'reference':{'path':str(reference_path),'sha256':sha(reference_path)},
            'options':{'appearance_revision':{'design_reconstruction_plan':plan}}}
        run_path=root/'run.json';run_path.write_text(json.dumps(run),encoding='utf8')
        argv=['renderer','--run',str(run_path),'--match-dir',str(directory),'--reference-blend',str(colour_path)]
        encoded=[];save=renderer.save_render_master
        def capture_master(values,path):
            encoded.append(values.copy())
            return save(values,path)
        original_bands=renderer._matching_native_bands
        writes=[]
        class Writer:
            def __init__(self):self.bands=[]
            def write(self,box,values):
                assert threading.current_thread() is threading.main_thread(), 'Cache writes escaped the consumer'
                self.bands.append((box, values.copy()))
            def finish(self):
                assert [box[1] for box, _ in self.bands] == [0,128,256]
                writes.append(self.bands)
                return {'written':True,'tiles':len(self.bands)}
        output=io.StringIO()
        for workers in (1,2):
            with patch.object(native_source_baseline,'source_for_reference_prepare',return_value=(source.copy(),
                    {'source_precision':'raster8_srgb'}, {'method':'clip_sdr'})) as source_read, \
                 patch.object(native_source_baseline,'background_for_reference_prepare',return_value=(old_background.copy(),
                    foreground.copy(),np.ones_like(alpha),0,{'cache_hit':True,'fixture_only':True})) as fb_read, \
                 patch.object(portrait_beauty_layer,'hair_correction_guard_for_run',return_value=None), \
                 patch.object(reference_pre_scene_cache,'open_writer',return_value=Writer()), \
                 patch.object(renderer,'_matching_native_bands',side_effect=lambda *args,**kwargs:
                    original_bands(*args,**kwargs,workers=workers)), \
                 patch.object(renderer,'save_render_master',side_effect=capture_master), \
                 patch.object(sys,'argv',argv),contextlib.redirect_stdout(output):
                renderer.main()
                assert source_read.call_count==fb_read.call_count==1
        assert len(encoded)==2 and all(values.shape==expected.shape for values in encoded)
        assert all(np.array_equal(values.view(np.uint32),expected.view(np.uint32)) for values in encoded)
        assert len(writes)==2
        for (serial_box,serial_values),(parallel_box,parallel_values) in zip(*writes):
            assert serial_box==parallel_box
            assert np.array_equal(serial_values.view(np.uint32),parallel_values.view(np.uint32))
        receipt=json.loads((directory/'composite-receipt.json').read_text('utf8'))
        assert receipt['composition_count']==1 and receipt['reference_spatial_colour_used'] is True
        assert receipt['reference_spatial_lightness_used'] is False and receipt['scene_balance'] is None
        assert receipt['output_sha256']==sha(receipt['output'])
        assert not (directory/'pure-composite.jpg').exists() and not list(directory.glob('composite-native-*.png'))
        ledger=json.loads(Path(receipt['workflow_timing']['path']).read_text('utf8'))
        assert ledger['status']=='complete' and not ledger['persistence_errors']
        assert abs(ledger['reconciliation']['difference_seconds'])<1e-9
        spans={row['name']:row for row in ledger['spans']}
        required={'load','matching_donors','foreground_background','load_selected_background',
            'matching_native','master_encoding','receipt_hashing'}
        assert required <= spans.keys()
        assert all(spans[name]['status']=='complete' and spans[name]['inclusive_seconds']>=0 for name in required)
        assert receipt['workflow_timing']['clock']=='time.perf_counter'
        # Unsupported paths stop at argument parsing, before any preparation.
        with patch.object(native_source_baseline,'source_for_reference_prepare') as blocked_source, \
             patch.object(sys,'argv',argv+['--scene-balance',str(root/'retired-scene.json')]), \
             contextlib.redirect_stderr(io.StringIO()):
            try:renderer.main()
            except SystemExit as error:assert error.code==2
            else:raise AssertionError('Retired full target CLI was accepted')
            assert blocked_source.call_count==0
        with patch.object(native_source_baseline,'source_for_reference_prepare') as blocked_source, \
             patch.object(sys,'argv',argv[:-2]),contextlib.redirect_stderr(io.StringIO()):
            try:renderer.main()
            except SystemExit as error:assert error.code==2
            else:raise AssertionError('Unbound T-only CLI was accepted')
            assert blocked_source.call_count==0
        result={'schema':'reference_native_baseline_cli_smoke.v1','passed':True,'native_size':list(native),
            'actual_T_B_cli_and_bound_colour_validation':True,'exact_single_composition_bits':True,
            'actual_tiff_jpeg_png_encoding':True,'retired_scene_cli_rejected_before_decode':True,
            'bound_colour_required':True,'existing_monotonic_ledger_reconciles':True,
            'spans':sorted(required),'models_or_gpu_executed':False,'large_native_run':False,
            'source_decode_and_FB_producers_substituted_by_CPU_fixture':True}
        result.update(serial_parallel_three_bands_and_tail_exact_bits=True,
                      pre_scene_writes_on_main_thread_in_original_order=True)
    if report_path:Path(report_path).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path)
    main(parser.parse_args().report)
