"""Small scene-output fixture using actual producer save/metadata statements."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import zipfile

import numpy as np

from native_array_storage import save_native_array, VALUES_ENCODING


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf8'))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    foreground_path = scripts/'source_foreground.py'
    sr_path = scripts/'background_superresolution_worker.py'
    cache_path = scripts/'source_foreground_cache.py'
    bits = np.resize(np.array([0x00000000, 0x80000000, 0x00000001, 0x3dcccccd, 0x3f800000], '<u4'), 23*31)
    alpha = bits.reshape(23, 31).view('<f4')
    rgb = np.stack([alpha, alpha*.5, alpha*.25], -1)
    foreground = np.roll(rgb, 2, axis=1)
    support = np.roll(alpha, 1, axis=0)
    input_snapshots = {role: values.tobytes(order='C') for role,values in (
        ('fg-alpha',alpha),('fg-old-background',rgb),('fg-premultiplied',foreground),('fg-support',support),
        ('sr-associated',rgb),('sr-validity',alpha))}
    checks = []
    with tempfile.TemporaryDirectory(prefix='cosplay_scene_storage_fixture_') as name:
        root = Path(name); fg_dir=root/'foreground'; sr_dir=root/'background'; fg_dir.mkdir(); sr_dir.mkdir()
        source=root/'source-fixture.bytes'; source.write_bytes(b'identity-only synthetic scene fixture; no photograph or model')
        namespace = {'np': np, 'Path': Path, 'save_native_array': save_native_array,
            'args': SimpleNamespace(output_dir=fg_dir, source=source, tile=512, halo=64), 'paths': {},
            'alpha': alpha, 'background': rgb, 'foreground': foreground, 'support': support,
            '_sha256_file': sha, 'FB_DIAGNOSTICS_SCHEMA': 'source_fb_diagnostics.sparse.v1',
            '__file__': str(foreground_path), 'fb_backend': 'pymatting_cpu_small_image'}
        fg_tree = ast.parse(foreground_path.read_text(encoding='utf8'))
        fg_main = next(node for node in fg_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        from source_foreground import save_foreground_arrays
        namespace['paths'] = save_foreground_arrays(fg_dir, [('alpha',alpha), ('old-background',rgb),
            ('premultiplied-foreground',foreground), ('unknown-edge-support',support)])
        from source_foreground import _write_fb_diagnostics
        namespace['_write_fb_diagnostics'] = _write_fb_diagnostics
        diagnostics={'flat_index':np.array([0],np.uint32),'fit_error':np.zeros(1,np.float32),
            'separation':np.ones(1,np.float32),'background_projection_delta':np.zeros((1,3),np.float32),
            'overlap_weight':np.ones(1,np.float32),'canvas_size':np.array([31,23],np.int32),
            'fb_normalization_scale':np.array(1,np.float32)}
        descriptor = namespace['_write_fb_diagnostics'](fg_dir, diagnostics, source, namespace['paths'])
        with zipfile.ZipFile(descriptor['path']) as archive:
            assert set(archive.namelist()) == {key+'.npy' for key in diagnostics}
            assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
        with np.load(descriptor['path'],allow_pickle=False) as resource:
            assert resource.files == list(diagnostics)
            for key,values in diagnostics.items():
                decoded = resource[key]
                assert decoded.dtype == values.dtype and decoded.shape == values.shape
                assert decoded.tobytes(order='C') == values.tobytes(order='C')
                assert np.isfinite(decoded).all()
            indices = resource['flat_index']
            assert indices.ndim == 1 and indices.dtype.kind == 'u' and len(indices) == descriptor['sample_count']
            assert np.all(indices[1:] > indices[:-1]) and np.all(indices < 31*23)
            assert resource['canvas_size'].tolist() == descriptor['canvas_size'] == [31,23]
        assert descriptor['representation'] == 'trimap_unknown_sparse_row_major_flat_index'
        assert descriptor['coordinate_space'] == 'oriented_source_pixels'
        assert descriptor['renderer_consumed'] is False and descriptor['support_expanded'] is False
        assert descriptor['sha256'] == sha(descriptor['path'])
        assert descriptor['source_sha256'] == sha(source)
        assert descriptor['paired_asset_sha256'] == {key:sha(path) for key,path in namespace['paths'].items()}
        invalid = {**diagnostics,'fit_error':np.array([object()],object)}
        try:namespace['_write_fb_diagnostics'](fg_dir,invalid,source,namespace['paths'])
        except ValueError:pass
        else:raise AssertionError('Invalid diagnostics were published')
        assert sha(descriptor['path']) == descriptor['sha256'] and not list(fg_dir.glob('*.tmp.npz'))
        namespace['paths']['fb_diagnostics_path'] = descriptor['path']
        namespace.update(fb_diagnostics=descriptor, VALUES_ENCODING=VALUES_ENCODING, alpha_cache=None,
            source=rgb, native_size=[31,23], trimap_method='fixture-only', radius=1,
            alpha_tiles=1, fb_tiles=1, provenance={}, precision='float32', scale=1.,
            started=0., alpha_finished=0., fitted=0., time=SimpleNamespace(perf_counter=lambda:0.),
            limitations=['Synthetic output fixture; no model execution'])
        fg_result_node = next(node for node in fg_main.body if isinstance(node,ast.Assign) and
            any(isinstance(target,ast.Name) and target.id=='result' for target in node.targets))
        exec(compile(ast.Module(body=[fg_result_node],type_ignores=[]),str(foreground_path),'exec'),namespace)
        result = json.loads(json.dumps(namespace['result']))
        assert result['schema_version']=='source_foreground.v1'
        assert result['native_values_encoding']==VALUES_ENCODING
        assert result['native_source_size']==[31,23] and result['analysis_size']==[31,23]
        assert result['coordinate_space']=='oriented_source_pixels'
        assert result['working_space']=='scene-linear_sRGB_float32'
        cache={'Path':Path,'file_identity':sha,'RGB_DECODE_VERSION':'fixture-only',
            'PARAMETERS':{'fb_diagnostics':'source_fb_diagnostics.sparse.v1','native_values_encoding':VALUES_ENCODING},
            'ASSET_KEYS':('alpha_path','old_background_path','premultiplied_foreground_path','unknown_edge_support_path','fb_diagnostics_path'),
            'code_path':lambda name:scripts/name}
        functions(cache_path, {'_identity','_valid_assets'}, cache)
        # Tiny identity stubs exercise the real cache function; they are never model inputs.
        model_root=root/'identity-only-stubs'; model_root.mkdir()
        for filename in ('model.safetensors','config.json','preprocessor_config.json'):
            (model_root/filename).write_bytes(('identity stub '+filename).encode())
        saliency=root/'saliency-fixture.bytes'; saliency.write_bytes(b'identity-only mask')
        runtime={'vitmatte_model_root':model_root,'matting_profile':'vitmatte_small_distinctions'}
        identity=cache['_identity'](root, source, saliency, None, runtime)
        assert cache['_valid_assets'](result, identity)
        stale_producer = json.loads(json.dumps(result))
        stale_producer['fb_diagnostics']['producer_sha256'] = '0'*64
        assert not cache['_valid_assets'](stale_producer,identity)
        wrong_layout = json.loads(json.dumps(result))
        wrong_layout['fb_diagnostics']['canvas_size'] = [23,31]
        assert not cache['_valid_assets'](wrong_layout,identity)
        old_policy=dict(result); old_policy.pop('native_values_encoding')
        assert not cache['_valid_assets'](old_policy, identity)
        changed_writer=root/'writer-identity-variant.py'
        changed_writer.write_bytes((scripts/'native_array_storage.py').read_bytes()+b'\n# identity-only variant\n')
        cache['code_path']=lambda name:changed_writer if name=='native_array_storage.py' else scripts/name
        identity2=cache['_identity'](root, source, saliency, None, runtime)
        assert identity['worker_sha256']==identity2['worker_sha256']
        assert identity['storage_writer_sha256']!=identity2['storage_writer_sha256']
        assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).digest()!=hashlib.sha256(json.dumps(identity2,sort_keys=True).encode()).digest()
        mapped_p=np.lib.format.open_memmap(sr_dir/'rgb.npy',mode='w+',dtype='<f4',shape=rgb.shape); mapped_p[:]=rgb
        mapped_v=np.lib.format.open_memmap(sr_dir/'alpha.npy',mode='w+',dtype='<f4',shape=alpha.shape); mapped_v[:]=alpha
        sr_tree=ast.parse(sr_path.read_text(encoding='utf8'))
        sr_main=next(node for node in sr_tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        output_nodes=[]
        for node in sr_main.body:
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='save_native_array': output_nodes.append(node)
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='descriptors' for t in node.targets): output_nodes.append(node)
            if isinstance(node,ast.For) and isinstance(node.target,ast.Tuple) and [item.id for item in node.target.elts]==['key','filename']: output_nodes.append(node)
        assert len(output_nodes)==4
        import copy
        sr_namespace={'np':np,'out':sr_dir,'native_p':mapped_p,'native_v':mapped_v,
            'save_native_array':save_native_array,'assets':{'background':{'role':'fixture-associated'},'validity':{'role':'fixture-coverage'}},
            'copy':copy,'digest':sha,'target_size':(31,23)}
        exec(compile(ast.Module(body=output_nodes,type_ignores=[]),str(sr_path),'exec'),sr_namespace)
        sr_namespace.update(VALUES_ENCODING=VALUES_ENCODING, run={'run_id':'scene-storage-fixture'},
            WEIGHT_SHA256='identity-only-unused', reflection_regions=[], crops=[], execution_stats={},
            start=0., time=SimpleNamespace(time=lambda:0.))
        sr_manifest_node = next(node for node in sr_main.body if isinstance(node,ast.Assign) and
            any(isinstance(target,ast.Name) and target.id=='manifest' for target in node.targets))
        exec(compile(ast.Module(body=[sr_manifest_node],type_ignores=[]),str(sr_path),'exec'),sr_namespace)
        sr_manifest = json.loads(json.dumps(sr_namespace['manifest']))
        assert sr_manifest['schema_version']=='background_superresolution.v1'
        assert sr_manifest['native_values_encoding']==VALUES_ENCODING
        assert sr_manifest['person_geometry_changed'] is False
        assert sr_manifest['source_run_id']==sr_manifest['producer_run_id']==sr_manifest['consumer_run_id']=='scene-storage-fixture'
        outputs=[('fg-alpha',fg_dir/'alpha.npz',alpha),('fg-old-background',fg_dir/'old-background.npz',rgb),
            ('fg-premultiplied',fg_dir/'premultiplied-foreground.npz',foreground),('fg-support',fg_dir/'unknown-edge-support.npz',support),
            ('sr-associated',sr_dir/'background-associated.npz',mapped_p),('sr-validity',sr_dir/'background-validity.npz',mapped_v)]
        wide=np.arange(17*19,dtype='<f8').reshape(17,19); wide[0,0]=-0.
        strided=wide[::-1,::2]; path=root/'dtype-preserving-f64.npz'
        input_snapshots['dtype-preserving-f64-strided']=strided.tobytes(order='C')
        save_native_array(path,strided)
        outputs.append(('dtype-preserving-f64-strided',path,strided))
        for role,path,expected in outputs:
            before=input_snapshots[role]
            with zipfile.ZipFile(path) as archive:
                assert archive.namelist()==['values.npy']
                assert archive.getinfo('values.npy').compress_type==zipfile.ZIP_STORED
            with np.load(path,allow_pickle=False) as data:
                assert data.files==['values']; decoded=data['values']
            assert decoded.dtype.str==expected.dtype.str and decoded.shape==expected.shape
            assert decoded.tobytes(order='C')==before and expected.tobytes(order='C')==before
            checks.append({'role':role,'dtype':decoded.dtype.str,'shape':list(decoded.shape),'bitwise_equal':True,
                'input_unchanged':True,'standard_np_load':True,'archive_sha256':sha(path),
                'stored':True})
        for role,asset in sr_manifest['assets'].items():
            assert asset['sha256']==sha(asset['path']) and asset['size']==[31,23] and asset['native_size']==[31,23]
            assert asset['coordinate_space']=='source_canvas'
        kept=root/'existing.npz'; save_native_array(kept,alpha); kept_sha=sha(kept)
        try: save_native_array(kept,np.array([object()],object))
        except ValueError: pass
        else: raise AssertionError('Object array was accepted')
        assert sha(kept)==kept_sha and not list(root.glob('*.tmp.npz'))
        mapped_p._mmap.close(); mapped_v._mmap.close()
    manifest=json.loads((scripts/'runtime_manifest.json').read_text(encoding='utf8'))
    assert 'native_array_storage.py' in manifest['files'] and len(manifest['files'])==len(set(manifest['files']))
    for name in ('design_reconstruction_pipeline.py','reference_pipeline.py','source_preview.py'):
        assert 'native_array_storage.py' in (scripts/name).read_text(encoding='utf8')
    assert '_save_native_array' not in (scripts/'reference_appearance_controls.py').read_text(encoding='utf8')
    report={'schema':'scene_native_array_storage_smoke.v1','passed':True,'models_or_gpu_used':False,
        'performance_benchmark':False,'producer_save_statements_executed':True,'cases':checks,
        'actual_producer_metadata_expressions_executed':True,'standard_json_metadata_roundtrip':True,
        'source_foreground_paired_asset_sha_binding':True,'source_foreground_current_policy_required':True,
        'sparse_diagnostics_all_observation_bits_and_sha_preserved':True,
        'sparse_diagnostics_indices_finite_domain_metadata_and_current_producer_verified':True,
        'sparse_diagnostics_fixed_stored_codec_and_atomic_publication':True,
        'writer_only_change_invalidates_cache_identity':True,'background_descriptor_sha_binding':True,
        'failed_write_preserves_destination_and_removes_own_temporary':True,
        'runtime_and_cache_dependency_closure':True,'old_private_writer_alias_present':False}
    if args.report: args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
