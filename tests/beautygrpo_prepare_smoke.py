"""Source-display, cache ownership and desktop cancellation contracts."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import beautygrpo_prepare as prepare
import beautygrpo_worker as worker
from display_transform import display_linear,source_transform_for_precision
from raw_support import RGB_DECODE_VERSION,linear_preview
from portrait_beauty_layer import apply_result


class BeautyPrepareContracts(unittest.TestCase):
    def test_portrait_regions_cache_reuses_bound_masks_without_rerunning_worker(self):
        """A warm retry reuses immutable parser assets and keeps request receipt ownership."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source-display.png'
            Image.new('RGB', (12, 10), (20, 50, 80)).save(source)
            face_model = root/'face.task'; face_model.write_bytes(b'face-model')
            registry = root/'registry.json'; registry.write_text('{}', encoding='utf8')
            config = {'face_registry': str(registry), 'regions_python': str(root/'python.exe'),
                      'regions_model_root': str(root/'segface')}
            record = {'model_input': str(source), 'model_input_sha256': worker.sha(source),
                      'native_size': [12, 10]}

            def fake_worker(command, **_kwargs):
                output = Path(command[command.index('--output') + 1])
                face = output/'face-01'; face.mkdir(parents=True)
                (face/'regions.npz').write_bytes(b'npz')
                prepare.write_json(output/'result.json', {
                    'schema_version': 'portrait_regions.v1',
                    'status': 'ready', 'source_path': str(source),
                    'source_sha256': record['model_input_sha256'],
                    'source_native_size': record['native_size'],
                    'worker_sha256': 'actual-parser-producer',
                    'regions': [{'id': 'face-01', 'npz_path': str(face/'regions.npz')}],
                })
                return SimpleNamespace(returncode=0)

            with patch.dict(os.environ, {'COSPLAY_BEAUTY_CACHE_ROOT': str(root/'cache')}), \
                 patch('current_model_registry.load_registry',
                       return_value={'face_model_path': str(face_model)}), \
                 patch.object(prepare.subprocess, 'run', side_effect=fake_worker) as worker_run:
                first_parent = root/'request-1'; first_parent.mkdir()
                first = prepare._run_regions(config, record, root, first_parent/'regions', 8)
                second_parent = root/'request-2'; second_parent.mkdir()
                second = prepare._run_regions(config, record, root, second_parent/'regions', 8)
                from runtime_resources import code_path
                code_identity = prepare.regions_code_identity(code_path('portrait_regions_worker.py'))
                with patch.object(prepare, 'regions_code_identity',
                                  return_value={**code_identity, 'producer_sha256': 'codec-only-change'}):
                    third_parent = root/'request-3'; third_parent.mkdir()
                    third = prepare._run_regions(config, record, root, third_parent/'regions', 8)
                self.assertEqual(worker_run.call_count, 1)
                third_cached = json.loads(third.read_text(encoding='utf8'))['cache']
                self.assertTrue(third_cached['cache_hit'])
                self.assertEqual(third_cached['producer_provenance'], {
                    'regions_producer_sha256': 'actual-parser-producer',
                    'requested_regions_producer_sha256': 'codec-only-change'})
                with patch.object(prepare, 'regions_code_identity',
                                  return_value={**code_identity, 'algorithm_sha256': 'changed-algorithm'}):
                    fourth_parent = root/'request-4'; fourth_parent.mkdir()
                    fourth = prepare._run_regions(config, record, root, fourth_parent/'regions', 8)
                    self.assertFalse(json.loads(fourth.read_text(encoding='utf8'))['cache']['cache_hit'])

            self.assertEqual(worker_run.call_count, 2)
            self.assertEqual(json.loads(first.read_text(encoding='utf8'))['source_sha256'],
                             record['model_input_sha256'])
            cached = json.loads(second.read_text(encoding='utf8'))
            self.assertTrue(cached['cache']['cache_hit'])
            self.assertEqual(cached['cache']['cache_root'], str((root/'cache').resolve()))
            self.assertTrue(Path(cached['regions'][0]['npz_path']).is_file())
            self.assertEqual(cached['source_path'], str(source.resolve()))
            self.assertEqual(cached['cache']['schema'], prepare.REGIONS_CACHE_SCHEMA)

    def test_generated_cache_binds_algorithm_and_rejects_old_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            face = root/'face.task'; face.write_bytes(b'cpu-fixture')
            config = {'face_registry': 'unused', 'base_manifest_sha256': 'test',
                      'offload': 'sequential_cpu_offload'}
            record = {'source_sha256': 'source', 'source_display': {}, 'source_precision': {},
                      'native_size': [12, 10], 'native_source_retouch': None, 'model_input_sha256': 'display'}
            code_identity = {'schema': 'portrait_regions_algorithm.v1',
                             'algorithm_sha256': 'stable-algorithm', 'producer_sha256': 'before-codec'}
            with (patch('current_model_registry.load_registry', return_value={'face_model_path': str(face)}),
                  patch.object(prepare, 'regions_code_identity', return_value=code_identity)):
                before = prepare._cache_binding(record, config, {'max_faces': 8})
                with patch.object(prepare, 'regions_code_identity',
                                  return_value={**code_identity, 'producer_sha256': 'after-codec'}):
                    self.assertEqual(before, prepare._cache_binding(record, config, {'max_faces': 8}))
                with patch.object(prepare, 'regions_code_identity',
                                  return_value={**code_identity, 'algorithm_sha256': 'new-algorithm'}):
                    self.assertNotEqual(before, prepare._cache_binding(record, config, {'max_faces': 8}))
            directory, cached = prepare._cache_result(root, before)
            self.assertIsNone(cached)
            directory.mkdir(parents=True)
            prepare.write_json(directory/'cache-entry.json', {
                'schema': 'beautygrpo_generated_cache.v1', 'binding': before})
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                prepare._cache_result(root, before)

    def test_generated_pixel_cache_binding_is_distinct_for_execution_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            face = Path(temporary)/'face.task'
            face.write_bytes(b'cpu-fixture')
            config = {'face_registry': 'unused', 'base_manifest_sha256': 'test',
                      'offload': 'sequential_cpu_offload'}
            record = {'source_sha256': 'source', 'source_display': {}, 'source_precision': {},
                      'native_size': [12, 10], 'native_source_retouch': None, 'model_input_sha256': 'display'}
            with patch('current_model_registry.load_registry', return_value={'face_model_path': str(face)}), \
                    patch.dict(os.environ, {worker.OFFLOAD_ENV: ''}):
                current = prepare._cache_binding(record, config, {'max_faces': 8})
                changed = prepare._cache_binding(record, {**config, 'torch_cpu_threads': 4}, {'max_faces': 8})
            self.assertNotEqual(worker._canonical_sha(current), worker._canonical_sha(changed))
            self.assertEqual({key: value for key, value in current.items() if key != 'model_identity'},
                             {key: value for key, value in changed.items() if key != 'model_identity'})

    def test_source_native_initialization_does_not_prepare_or_overlay_beauty(self):
        import argparse
        import single_source_workflow as workflow
        from generated_appearance_render_smoke import ScopedRenderTests
        fixture=ScopedRenderTests(methodName='runTest');fixture.setUp()
        try:
            run=fixture.fixture();before=copy.deepcopy(run)
            run['task_intent']={'mode':'create_reference'}
            with patch.object(prepare,'prepare_for_workbench',side_effect=AssertionError('Unused Beauty model')) as model, \
                    patch.object(prepare,'prepare_cached_for_workbench',side_effect=AssertionError('Unused Beauty cache')) as cache, \
                    patch('start_qwen_image.release_for_run',side_effect=AssertionError('No Beauty inference transition')) as release:
                workflow._prepare_task_portrait_beauty(fixture.root,run,argparse.Namespace())
                rendered=workflow.render_candidate(fixture.root,run,fixture.args)
            model.assert_not_called();cache.assert_not_called();release.assert_not_called()
            self.assertEqual(run['options']['photo_appearance'],before['options']['photo_appearance'])
            self.assertEqual(rendered['metadata']['kind'],'scoped_photo_appearance_render')
            self.assertFalse(rendered['metadata']['operator']['source_subject_RGB_restored'])
        finally:fixture.tearDown()

    def test_predecode_cache_binding_uses_saved_baseline_without_opening_raw(self):
        """A complete appearance baseline is enough to probe a cache hit."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source.ARW'
            source.write_bytes(b'raw-placeholder')
            source_sha = worker.sha(source)
            baseline = {
                'schema': 'appearance_source_baseline.v1',
                'working_domain': 'display_linear_sRGB',
                'source_sha256': source_sha,
                'source_display_applied_once': True,
                'source_precision': 'raw16_linear',
                'decode_version': RGB_DECODE_VERSION,
                'source_display': source_transform_for_precision('raw16_linear'),
                'native_size': [8, 6],
            }
            run = {'source': {'path': str(source), 'sha256': source_sha},
                   'options': {'appearance_revision': {
                       'design_reconstruction_plan': {
                           'schema_version': 'design_reconstruction_plan.v1',
                           'working_domain': 'display_linear_sRGB',
                           'source_baseline': baseline}}}}
            face = root/'face.task'; face.write_bytes(b'cpu-fixture')
            config = {'face_registry': 'unused', 'base_manifest_sha256': 'test',
                      'offload': 'sequential_cpu_offload'}
            source_only = copy.deepcopy(run)
            source_only['options'] = {'source_only_plan': {
                **baseline, 'schema': 'source_only_display_plan.v1'}}
            with (patch('raw_support.open_linear', side_effect=AssertionError('RAW decode is not allowed')),
                  patch('current_model_registry.load_registry', return_value={'face_model_path': str(face)})):
                for request in (run, source_only):
                    with self.subTest(schema=(request['options'].get('source_only_plan') or baseline)['schema']):
                        binding = prepare._predecode_binding(request, config, {'max_faces': 8}, root)
                        self.assertEqual(binding['source_sha256'], source_sha)
                        self.assertEqual(binding['native_size'], [8, 6])
                        self.assertEqual(binding['source_precision']['source_precision'], 'raw16_linear')
                        self.assertNotIn('model_input_sha256', binding)
                        record = {key: binding[key] for key in ('source_sha256', 'source_display',
                            'source_precision', 'native_size', 'native_source_retouch')}
                        self.assertRaises(KeyError, prepare._cache_binding, record, config, {'max_faces': 8})
                        full = prepare._cache_binding({**record, 'model_input_sha256': 'd'*64},
                                                      config, {'max_faces': 8})
                        self.assertEqual(full['model_input_sha256'], 'd'*64)
                        self.assertEqual({key: value for key, value in full.items()
                                          if key != 'model_input_sha256'}, binding)
            facts_only = {'source': {'path': str(source), 'sha256': source_sha,
                                     'mode': 'RAW', 'decode_version': RGB_DECODE_VERSION,
                                     'oriented_size': [8, 6]}, 'options': {}}
            with patch('current_model_registry.load_registry',
                       return_value={'face_model_path': str(face)}):
                binding = prepare._predecode_binding(facts_only, config, {'max_faces': 8}, root)
            self.assertEqual(binding['source_precision']['source_precision'], 'raw16_linear')
            self.assertEqual(binding['native_size'], [8, 6])

    def test_persistent_service_events_are_request_bound_and_read_incrementally(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'events.jsonl'
            events = [
                {'type': 'beautygrpo_progress', 'request_id': 'other', 'stage': 'wrong', 'message': 'wrong', 'progress': 1},
                {'type': 'beautygrpo_progress', 'request_id': 'current', 'stage': 'beauty_face_generation', 'message': 'sampling', 'progress': .4},
            ]
            path.write_text('\n'.join(json.dumps(event) for event in events)+'\n', encoding='utf8')
            observed = []
            def progress(stage, message, amount):
                observed.append((stage, message, amount))
            with path.open('r', encoding='utf8') as stream:
                offset, terminal = prepare._read_service_events(path, 0, 'current', progress, stream=stream)
                self.assertIsNone(terminal)
                self.assertEqual(observed, [('beauty_face_generation', 'sampling', .4)])
                result = {'type': 'beautygrpo_result', 'request_id': 'current', 'ok': True}
                with path.open('a', encoding='utf8') as writer:
                    writer.write(json.dumps(result)+'\n')
                end, terminal = prepare._read_service_events(path, offset, 'current', progress, stream=stream)
                self.assertGreater(end, offset)
                self.assertEqual(terminal, result)
                self.assertEqual(len(observed), 1)

    def test_recomposition_freezes_same_verified_generated_rgb_without_model_runtime(self):
        from beautygrpo_worker_smoke import face_region
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.png';Image.new('RGB',(256,256),(20,50,80)).save(source)
            region,fields=face_region()
            points=np.asarray(region['landmarks']);points[33]=[86,98];points[133]=[108,98]
            points[362]=[150,98];points[263]=[172,98];region['landmarks']=points.tolist()
            fields['labels'][:15]=14
            asset=root/'region.npz';np.savez(asset,**fields);region['npz_path']=str(asset)
            region_path=root/'regions.json';worker.write_json(region_path,{'schema_version':'portrait_regions.v1',
                'source_sha256':worker.sha(source),'source_native_size':[256,256],'regions':[region]})
            box=worker.face_focused_box(region,[256,256],fields)
            generated=root/'generated.png';Image.new('RGB',(box[2]-box[0],box[3]-box[1]),(210,60,100)).save(generated)
            model=root/'model-output.png';Image.new('RGB',(1024,1024),(210,60,100)).save(model)
            oldweight=root/'old.npy';np.save(oldweight,np.zeros((box[3]-box[1],box[2]-box[0]),np.float32))
            face={'id':region['id'],'native_box':box,'generated_native_roi':str(generated),
                'generated_native_roi_sha256':worker.sha(generated),'model_output':str(model),'model_output_sha256':worker.sha(model),
                'model_input':str(model),'model_input_sha256':worker.sha(model),'composition_weight':str(oldweight),
                'composition_weight_sha256':worker.sha(oldweight)}
            record={'schema':prepare.PREPARE_SCHEMA,'inference_completed':True,'working_domain':'display_linear_sRGB',
                'source_path':str(source),'source_sha256':worker.sha(source),'native_size':[256,256],
                'source_precision':{'source_precision':'raster8_srgb','decode_version':RGB_DECODE_VERSION},
                'source_display':source_transform_for_precision('raster8_srgb'),'native_source_retouch':None,
                'source_display_consumptions':1,'model_input':str(source),'model_input_sha256':worker.sha(source),
                'model_identity':worker.model_identity({'base_manifest_sha256':'test','offload':'sequential_cpu_offload'}),
                'regions':{'path':str(region_path),'sha256':worker.sha(region_path)},'faces':[face],
                'model_evidence':{'lora_verified':{'all_matrix_values_verified':True}}}
            path=root/'prepared.json';worker.write_json(path,record);before=path.read_bytes()
            with patch.object(worker,'load_pipeline') as pipeline:
                result=prepare.recompose_prepared(path,root/'recomposed')
            pipeline.assert_not_called()
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(result['generation_inferences_this_attempt'],0)
            self.assertTrue(result['actual_generated_rgb_unchanged'])
            refreshed=json.loads(Path(result['prepared']).read_text(encoding='utf8'))
            self.assertEqual(refreshed['model_evidence'],record['model_evidence'])
            self.assertEqual(refreshed['faces'][0]['model_output_sha256'],face['model_output_sha256'])
            with np.load(result['layer']['faces'][0]['pixels']['path'],allow_pickle=False) as frozen:
                self.assertGreater(frozen['composition_weight'][24,110],0)
                self.assertEqual(frozen['composition_weight'][10,110],0)

    def test_source_uses_existing_display_transform_once_and_binds_native_retouch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.png';Image.new('RGB',(6,4)).save(source)
            transform=source_transform_for_precision('raw16_linear');transform['shoulder_start']=.6
            values=np.full((4,6,3),1.2,np.float32);values[...,2]=.7
            precision={'source_precision':'raw16_linear','working_precision':'float32_linear','decode_version':RGB_DECODE_VERSION,
                       'native_source_retouch':{'layer_sha256':'move-sensitive','delta_sha256':'stable'},'native_retouch_consumptions':1}
            run={'source':{'path':str(source),'sha256':worker.sha(source)},'options':{
                'source_only_plan':{'schema':'source_only_display_plan.v1','working_domain':'display_linear_sRGB',
                    'source_sha256':worker.sha(source),'source_precision':'raw16_linear','decode_version':RGB_DECODE_VERSION,
                    'source_display':transform,'source_display_applied_once':True}}}
            stable={'delta_sha256':'a','source_sha256':worker.sha(source),'working_domain':'scene_linear_sRGB'}
            with patch('native_source_baseline.open_source_for_run',return_value=(values.copy(),precision)), \
                    patch.object(prepare,'stable_retouch_identity',return_value=stable):
                record=prepare.prepare_source(run,root,root)
            self.assertEqual(record['source_display'],transform)
            self.assertEqual(record['source_display_consumptions'],1)
            self.assertEqual(record['source_precision'],precision)
            self.assertEqual(record['native_source_retouch'],stable)
            expected=np.asarray(linear_preview(display_linear(values,transform)))
            actual=np.asarray(Image.open(record['model_input']))
            self.assertTrue(np.array_equal(actual,expected))
            twice=np.asarray(linear_preview(display_linear(display_linear(values,transform),transform)))
            self.assertFalse(np.array_equal(actual,twice))

    def test_explicit_and_per_user_config_discovery_without_repository_or_drive_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);config=root/'installation.json';config.write_text('{}',encoding='utf8')
            selection=root/'selection.json';selection.write_text(json.dumps({'schema':prepare.SELECTION_SCHEMA,'model_config':str(config)}),encoding='utf8')
            with patch.dict(os.environ,{'COSPLAY_BEAUTYGRPO_CONFIG':'','COSPLAY_MODELS_ROOT':str(root/'model')}),patch.object(prepare,'selection_path',return_value=selection):
                self.assertEqual(prepare.installation_path(),config.resolve())
                # A remembered explicit installation must win over a stale
                # portable config beside the launcher.
                (root/'model').mkdir()
                (root/'model/beautygrpo.json').write_text('{}', encoding='utf8')
                self.assertEqual(prepare.installation_path(),config.resolve())
                self.assertEqual(prepare.installation_path({'model_config':str(config)}),config.resolve())
                self.assertRaises(ValueError,prepare.installation_path,{'model_config':'relative.json'})
            self.assertRaises(ValueError,prepare.validate_options,{'steps':2})
            self.assertRaises(ValueError,prepare.validate_options,{'max_faces':True})

    def test_complete_generated_layer_cache_is_owned_and_never_reruns_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.png';Image.new('RGB',(12,10),(20,50,80)).save(source)
            run={'source':{'path':str(source),'sha256':worker.sha(source)},'options':{}}
            oldrun=copy.deepcopy(run)
            config={'base_manifest_sha256':'test','offload':'sequential_cpu_offload'}
            def binding(record,*args):
                return {'source_sha256':record['source_sha256'],'model_input_sha256':record['model_input_sha256'],
                        'source_display':record['source_display'],'model_identity':worker.model_identity(config)}
            def regions(config,record,root,output,max_faces):
                output.mkdir();path=output/'result.json';worker.write_json(path,{'test':'regions'});return path
            def infer(input_path,regions_path,config,output,**kwargs):
                output.mkdir()
                generated=output/'generated.png';Image.new('RGB',(8,8),(210,60,100)).save(generated)
                weights=output/'weight.npy';np.save(weights,np.ones((8,8),np.float32),allow_pickle=False)
                face={'id':'face-01','native_box':[2,1,10,9],'generated_native_roi':str(generated),
                      'generated_native_roi_sha256':worker.sha(generated),'composition_weight':str(weights),
                      'composition_weight_sha256':worker.sha(weights),'model_output':str(generated)}
                result={'inference_completed':True,'faces':[face],'source':{'sha256':worker.sha(input_path),'native_size':[12,10]},
                        'regions':{'path':str(regions_path)},'model_identity':worker.model_identity(config),
                        'base_verified':{},'lora_verified':{},'versions':{}}
                worker.write_json(output/'result.json',result);return result
            with patch.object(prepare,'installation_path',return_value=root/'unused.json'), \
                    patch.object(prepare,'load_installation',return_value=config), \
                    patch.object(prepare,'_cache_binding',side_effect=binding), \
                    patch.object(prepare,'_run_regions',side_effect=regions) as parsing, \
                    patch.object(worker,'run',side_effect=infer) as model:
                first=prepare.prepare(run,root,root/'first')
                second=prepare.prepare(run,root,root/'second')
            self.assertFalse(first['cache_hit']);self.assertTrue(second['cache_hit'])
            self.assertEqual(model.call_count,1);self.assertEqual(parsing.call_count,1)
            self.assertEqual(second['generation_inferences_this_attempt'],0)
            self.assertEqual(run,oldrun,'Model preparation must not mutate the submitted revision')
            for result,directory in ((first,root/'first'),(second,root/'second')):
                face=result['layer']['faces'][0]
                self.assertTrue(Path(face['pixels']['path']).is_relative_to(directory))
                self.assertEqual(worker.sha(face['pixels']['path']),face['pixels']['sha256'])
            values=np.full((10,12,3),.08,np.float32)
            after=apply_result(values,second['layer'])
            self.assertFalse(np.array_equal(after[4,4],values[4,4]))
            self.assertTrue(np.array_equal(after[0],values[0]))
            self.assertFalse(second['layer']['source_texture_restoration'])

    def test_no_face_result_is_a_complete_no_change_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.png';Image.new('RGB',(8,8)).save(source)
            run={'source':{'path':str(source),'sha256':worker.sha(source)},'options':{}}
            config={'base_manifest_sha256':'test','offload':'sequential_cpu_offload'}
            with patch.object(prepare,'installation_path',return_value=root/'unused.json'), \
                    patch.object(prepare,'load_installation',return_value=config), \
                    patch.object(prepare,'_cache_binding',return_value={}), \
                    patch.object(prepare,'_run_regions',return_value=root/'regions.json'), \
                    patch.object(worker,'run',return_value={'inference_completed':False,'faces':[]}), \
                    patch('portrait_beauty_layer.freeze_prepared') as freeze:
                result=prepare.prepare(run,root,root/'output')
            freeze.assert_not_called()
            self.assertTrue(result['ok']);self.assertTrue(result['no_change'])
            self.assertIsNone(result['result']);self.assertIsNone(result['layer'])

    def test_cancelled_desktop_request_never_starts_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch.object(prepare,'installation_path',return_value=root/'unused.json'), \
                    patch.object(prepare,'load_installation',return_value={'python':'unused'}), \
                    patch('subprocess.Popen') as process:
                self.assertRaises(InterruptedError,prepare.prepare_for_workbench,{},root,root/'output',cancel=lambda:True)
            process.assert_not_called()
            self.assertFalse((root/'output').exists())


    def test_resident_gpu_handoff_stops_owned_idle_service_even_when_cpu_budget_fits(self):
        import builtins
        import runtime_memory
        import threading
        from unittest.mock import Mock
        process = Mock(pid=4321); process.poll.return_value = None
        service = {'process': process, 'owner_pid': os.getpid(), 'busy_request_id': None,
                   'code_identity': prepare._service_code_identity(),
                   'execution_profile': {'offload': 'nunchaku_cuda'}}
        original_import = builtins.__import__
        def model_free_import(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'diffusers', 'accelerate'}:
                raise AssertionError('Desktop handoff imported model runtime: ' + name)
            return original_import(name, *args, **kwargs)
        with patch.object(prepare, '_BEAUTY_SERVICE', service), \
                patch.object(prepare, '_BEAUTY_SERVICE_LOCK', threading.Lock()), \
                patch.object(runtime_memory, 'memory_budget', return_value={'fits': True}), \
                patch.object(runtime_memory, 'memory_status', return_value={'available_bytes': 100}), \
                patch.object(prepare, '_stop_beauty_service') as stop, \
                patch('builtins.__import__', side_effect=model_free_import):
            stop.side_effect = lambda: setattr(process.poll, 'return_value', 0)
            for stage, explicit in (('native_analysis', False), ('generation', False), ('future_gpu_stage', True)):
                process.poll.return_value = None
                receipt = prepare.release_idle_for_stage(next_stage_cpu_bytes=0,
                    next_stage=stage, next_stage_gpu=explicit)
                self.assertEqual(receipt['state'], 'idle_owned_service_stopped')
                self.assertEqual(receipt['stop_reason'], 'resident_gpu_handoff')
                self.assertTrue(receipt['process_exit_confirmed'])
                self.assertEqual(receipt['model_requests'], 0)
                self.assertTrue(receipt['cpu_budget']['fits'])
            self.assertEqual(stop.call_count, 3)

    def test_idle_cpu_budget_reuse_checks_full_configured_execution_profile(self):
        import threading
        import runtime_memory
        from unittest.mock import Mock
        config_path = Path(__file__).resolve()
        config = {'base_manifest_sha256': 'test', 'offload': 'sequential_cpu_offload',
                  'execution_offload': 'nunchaku_cuda'}
        process = Mock(pid=4321); process.poll.return_value = None
        with patch.dict(os.environ, {worker.OFFLOAD_ENV: ''}):
            profile = worker.execution_profile(config)
            service = {'process': process, 'owner_pid': os.getpid(), 'busy_request_id': None,
                'code_identity': prepare._service_code_identity(),
                'pipeline_loaded': True, 'config_path': str(config_path),
                'config_sha256': 'fixture', 'execution_profile': profile}
            with patch.object(prepare, '_BEAUTY_SERVICE', service), \
                    patch.object(prepare, '_BEAUTY_SERVICE_LOCK', threading.Lock()), \
                    patch.object(prepare, 'installation_path', return_value=config_path), \
                    patch.object(prepare, 'sha', return_value='fixture'), \
                    patch.object(prepare, '_service_code_identity', return_value=service['code_identity']), \
                    patch.object(prepare, 'load_installation', return_value=config) as load:
                self.assertEqual(prepare.beauty_memory_needed(), {'physical_bytes': 0, 'commit_bytes': 0})
                load.assert_called_once_with(str(config_path))
                with patch.dict(config, {'torch_cpu_threads': 4}):
                    self.assertEqual(prepare.beauty_memory_needed(), {
                        'physical_bytes': runtime_memory.BEAUTY_COLD_PHYSICAL_BYTES,
                        'commit_bytes': runtime_memory.BEAUTY_COLD_COMMIT_BYTES})
                for identity in (None, {'schema': 'beautygrpo_service_code.v1', 'sources': {}}):
                    with self.subTest(code_identity=identity):
                        service['code_identity'] = identity
                        self.assertEqual(prepare.beauty_memory_needed(), {
                            'physical_bytes': runtime_memory.BEAUTY_COLD_PHYSICAL_BYTES,
                            'commit_bytes': runtime_memory.BEAUTY_COLD_COMMIT_BYTES})

    def test_service_identity_binds_each_small_source_and_invalidates_changes(self):
        import hashlib
        import runtime_resources
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ('beautygrpo_worker.py', 'beautygrpo_prepare.py', 'beautygrpo_nunchaku.py')
            for name in names:
                (root/name).write_bytes(b'# fixture\n')
            with patch.object(runtime_resources, 'code_path', side_effect=lambda name: root/name):
                previous = prepare._service_code_identity()
                self.assertEqual(previous, {
                    'schema': 'beautygrpo_service_code.v1',
                    'sources': dict.fromkeys(names, hashlib.sha256(b'# fixture\n').hexdigest())})
                for name in names:
                    with self.subTest(source=name):
                        (root/name).write_bytes(b'# changed fixture\n')
                        current = prepare._service_code_identity()
                        self.assertEqual({key for key in names if previous['sources'][key] != current['sources'][key]},
                                         {name})
                        previous = current

    def test_start_service_reuses_matching_code_and_replaces_stale_owned_idle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {'python': 'unused-python', 'config_path': str(root/'installation.json'),
                      'config_sha256': 'fixture'}
            identity = prepare._service_code_identity()
            existing = {'process': Mock(pid=4321), 'owner_pid': os.getpid(), 'busy_request_id': None,
                        'config_path': str((root/'installation.json').resolve()), 'config_sha256': 'fixture',
                        'execution_profile': worker.execution_profile(config), 'code_identity': identity}
            existing['process'].poll.return_value = None
            with patch.object(prepare, '_BEAUTY_SERVICE', existing), \
                    patch.object(prepare, '_stop_beauty_service') as stop, \
                    patch.object(prepare.subprocess, 'Popen') as launch:
                reused, is_reused = prepare._start_beauty_service(config, root)
                self.assertIs(reused, existing); self.assertTrue(is_reused)
                stop.assert_not_called(); launch.assert_not_called()
                identities = [None]
                for name in identity['sources']:
                    changed = copy.deepcopy(identity); changed['sources'][name] = 'changed'
                    identities.append(changed)
                for stale_identity in identities:
                    with self.subTest(code_identity=stale_identity):
                        existing['code_identity'] = stale_identity
                        with patch.object(prepare, '_BEAUTY_SERVICE', existing):
                            replacement, is_reused = prepare._start_beauty_service(config, root)
                        self.assertFalse(is_reused); self.assertEqual(replacement['code_identity'], identity)
                        replacement['stream'].close()
                self.assertEqual(stop.call_count, len(identities))
                self.assertEqual(launch.call_count, len(identities))

    def test_start_service_never_replaces_busy_or_unowned_process(self):
        process = Mock(pid=4321); process.poll.return_value = None
        config = {'config_path': __file__, 'config_sha256': 'fixture'}
        for owner, request_id in ((os.getpid(), 'active-request'), (-1, None)):
            with self.subTest(owner=owner, request_id=request_id):
                service = {'process': process, 'owner_pid': owner, 'busy_request_id': request_id}
                with patch.object(prepare, '_BEAUTY_SERVICE', service), \
                        patch.object(prepare, '_stop_beauty_service') as stop, \
                        patch.object(prepare.subprocess, 'Popen') as launch, \
                        self.assertRaises(prepare.BeautyServiceBusyError):
                    prepare._start_beauty_service(config, Path(__file__).parent)
                stop.assert_not_called(); launch.assert_not_called()

    def test_stale_idle_handoff_stops_without_sending_old_control_or_budget(self):
        import runtime_memory
        import threading
        identity = prepare._service_code_identity()
        for stale_identity in (None, {**identity, 'sources': {}}):
            with self.subTest(code_identity=stale_identity):
                process = Mock(pid=4321); process.poll.return_value = None
                service = {'process': process, 'owner_pid': os.getpid(), 'busy_request_id': None,
                           'pipeline_loaded': True, 'code_identity': stale_identity,
                           'execution_profile': {'offload': 'nunchaku_cuda'}}
                def stop_process():
                    process.poll.return_value = 0
                with patch.object(prepare, '_BEAUTY_SERVICE', service), \
                        patch.object(prepare, '_BEAUTY_SERVICE_LOCK', threading.Lock()), \
                        patch.object(prepare, '_stop_beauty_service', side_effect=stop_process) as stop, \
                        patch.object(runtime_memory, 'memory_budget') as budget, \
                        patch.object(runtime_memory, 'memory_status', return_value={'available_bytes': 100}):
                    result = prepare.release_idle_for_stage(next_stage_cpu_bytes=10, next_stage='generation')
                self.assertEqual(result['state'], 'idle_owned_service_stopped')
                self.assertEqual(result['stop_reason'], 'service_code_changed')
                self.assertEqual(result['requested_service_code_identity'], identity)
                self.assertEqual(result['service_code_identity'], stale_identity)
                self.assertTrue(result['process_exit_confirmed']); self.assertEqual(result['model_requests'], 0)
                stop.assert_called_once_with(); budget.assert_not_called()

    def test_resident_handoff_preserves_busy_unowned_and_cpu_save_services(self):
        import runtime_memory
        import threading
        from unittest.mock import Mock
        process = Mock(pid=4321); process.poll.return_value = None
        service = {'process': process, 'owner_pid': os.getpid(), 'busy_request_id': None,
                   'code_identity': prepare._service_code_identity(),
                   'execution_profile': {'offload': 'nunchaku_cuda'}}
        lock = threading.Lock()
        with patch.object(prepare, '_BEAUTY_SERVICE', service), \
                patch.object(prepare, '_BEAUTY_SERVICE_LOCK', lock), \
                patch.object(runtime_memory, 'memory_budget', return_value={'fits': True}), \
                patch.object(prepare, '_stop_beauty_service') as stop:
            with lock:
                self.assertEqual(prepare.release_idle_for_stage(next_stage_cpu_bytes=0,
                    next_stage='generation')['state'], 'busy_service_untouched')
            service['busy_request_id'] = 'active'
            self.assertEqual(prepare.release_idle_for_stage(next_stage_cpu_bytes=0,
                next_stage='native_analysis')['state'], 'busy_service_untouched')
            service['busy_request_id'] = None; service['owner_pid'] = -1
            self.assertEqual(prepare.release_idle_for_stage(next_stage_cpu_bytes=0,
                next_stage='native_analysis')['state'], 'unowned_service_untouched')
            service['owner_pid'] = os.getpid()
            self.assertEqual(prepare.release_idle_for_stage(next_stage_cpu_bytes=0,
                next_stage='cpu_save')['state'], 'idle_pipeline_retained')
            stop.assert_not_called()

    def test_resident_handoff_reconciles_only_matching_owned_terminal(self):
        import runtime_memory
        import threading
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temporary:
            events = Path(temporary) / 'events.jsonl'
            process = Mock(pid=4321); process.poll.return_value = None
            service = {'process': process, 'owner_pid': os.getpid(), 'busy_request_id': 'own-request',
                       'code_identity': prepare._service_code_identity(),
                       'busy_event_log': str(events),
                       'execution_profile': {'offload': 'nunchaku_cuda'}}
            with patch.object(prepare, '_BEAUTY_SERVICE', service), \
                    patch.object(prepare, '_BEAUTY_SERVICE_LOCK', threading.Lock()), \
                    patch.object(runtime_memory, 'memory_budget', return_value={'fits': True}), \
                    patch.object(runtime_memory, 'memory_status', return_value=None), \
                    patch.object(prepare, '_stop_beauty_service') as stop:
                prepare._service_event(events, {'type': 'beautygrpo_result', 'request_id': 'another', 'ok': True})
                self.assertEqual(prepare.release_idle_for_stage(next_stage_cpu_bytes=0,
                    next_stage='generation')['state'], 'busy_service_untouched')
                stop.assert_not_called()
                prepare._service_event(events, {'type': 'beautygrpo_result', 'request_id': 'own-request', 'ok': True})
                before = events.read_bytes()
                stop.side_effect = lambda: setattr(process.poll, 'return_value', 0)
                receipt = prepare.release_idle_for_stage(next_stage_cpu_bytes=0, next_stage='generation')
                self.assertEqual(receipt['stop_reason'], 'resident_gpu_handoff')
                self.assertIsNone(service['busy_request_id'])
                self.assertEqual(events.read_bytes(), before)
                stop.assert_called_once()

    def test_generation_releases_beauty_only_for_local_gpu_provider(self):
        from contextlib import redirect_stdout
        import io
        import photo_edit_tools_smoke as tools_fixture
        for kind in ('comfyui','openai-compatible'):
            with self.subTest(kind=kind):
                fixture=tools_fixture.EditingToolsTests(methodName='runTest');fixture.setUp()
                try:
                    fixture.prepare_decision('joint_edit')
                    import reference_provider
                    original=reference_provider.provider_config
                    workflow=fixture.root/'workflow.json';workflow.write_text('{}',encoding='utf8')
                    def provider(**options):
                        actual=original(**options);actual['kind']=kind
                        actual.update(workflow_path=str(workflow),workflow_configured=True)
                        return actual
                    def request(endpoint,image,workflow,prompt,timeout,events,**kwargs):
                        return fixture.transport(endpoint,'fixture',image,'fixture-image',prompt,None,timeout)
                    with patch.object(reference_provider,'provider_config',side_effect=provider), \
                            patch.object(reference_provider,'request_edit',side_effect=fixture.transport), \
                            patch.object(reference_provider.comfyui_adapter,'request_generation',side_effect=request), \
                            patch.object(prepare,'release_idle_for_stage',return_value={'state':'released'}) as release, \
                            redirect_stdout(io.StringIO()):
                        fixture.execute()
                    record=json.loads(next((fixture.root/'tools').glob('*/tool-attempt.json')).read_text('utf8'))
                    if kind=='comfyui':
                        release.assert_called_once()
                        values=dict(release.call_args.kwargs);cancel=values.pop('cancel')
                        self.assertEqual(values,{'next_stage_cpu_bytes':0,'next_stage':'generation','next_stage_gpu':True})
                        self.assertIsNone(cancel());fixture.cancelled=True
                        with self.assertRaises(InterruptedError):cancel()
                        self.assertEqual(record['portrait_generation_memory_transition']['state'],'released')
                    else:
                        release.assert_not_called()
                        self.assertNotIn('portrait_generation_memory_transition',record)
                finally:fixture.tearDown()

    def test_gpu_stage_flag_is_explicit_and_cannot_be_an_integer(self):
        self.assertRaisesRegex(ValueError, 'must be a boolean', prepare.release_idle_for_stage,
            next_stage_cpu_bytes=0, next_stage='generation', next_stage_gpu=1)

    def test_native_preparation_handoff_uses_its_supplied_cancellation(self):
        import single_source_workflow as workflow
        from current_stack_fixture import prepare_fixture
        class ReachedHandoff(Exception): pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = prepare_fixture(root); cancelled = Mock(return_value=False)
            def handoff(**values):
                self.assertIs(values['cancel'], cancelled)
                self.assertFalse(values['cancel']())
                raise ReachedHandoff()
            with (patch('runtime_memory.native_analysis_cpu_bytes', return_value=0),
                  patch.object(prepare, 'release_idle_for_stage', side_effect=handoff)):
                with self.assertRaises(ReachedHandoff):
                    workflow._refresh_design_plan(root, run, cancel=cancelled, preparation_policy='prepare_current')
            self.assertGreaterEqual(cancelled.call_count, 2)

    def test_pressure_releases_only_this_process_idle_service(self):
        from unittest.mock import Mock
        import runtime_memory
        import threading
        process = Mock(pid=4321)
        process.poll.return_value = None
        service = {'process':process,'owner_pid':os.getpid(),'busy_request_id':None,
                   'code_identity':prepare._service_code_identity()}
        lock = threading.Lock()
        with patch.object(prepare,'_BEAUTY_SERVICE',service), patch.object(prepare,'_BEAUTY_SERVICE_LOCK',lock), \
                patch.object(prepare,'_stop_beauty_service') as stop, \
                patch.object(runtime_memory,'memory_budget',return_value={'fits':False}), \
                patch.object(runtime_memory,'memory_status',return_value=None):
            with lock:
                self.assertEqual(prepare.release_idle_for_stage(next_stage_cpu_bytes=1,next_stage='native')['state'],
                                 'busy_service_untouched')
            service['busy_request_id']='active-request'
            self.assertEqual(prepare.release_idle_for_stage(next_stage_cpu_bytes=1,next_stage='native')['state'],
                             'busy_service_untouched')
            service['busy_request_id']=None;service['owner_pid']=-1
            self.assertEqual(prepare.release_idle_for_stage(next_stage_cpu_bytes=1,next_stage='native')['state'],
                             'unowned_service_untouched')
            service['owner_pid']=os.getpid()
            with patch.object(runtime_memory,'memory_budget',return_value={'fits':True}):
                self.assertEqual(prepare.release_idle_for_stage(next_stage_cpu_bytes=1,next_stage='native')['state'],
                                 'idle_pipeline_retained')
            stop.assert_not_called()
            stop.side_effect=lambda: setattr(process.poll,'return_value',0)
            receipt=prepare.release_idle_for_stage(next_stage_cpu_bytes=1,next_stage='native')
            self.assertEqual(receipt['state'],'idle_owned_service_stopped')
            self.assertTrue(receipt['process_exit_confirmed'])
            stop.assert_called_once()

    def test_abandoned_wait_observes_only_its_late_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            events=Path(temporary)/'events.jsonl'
            service={'owner_pid':os.getpid(),'busy_request_id':'old','busy_event_log':str(events)}
            prepare._service_event(events,{'type':'beautygrpo_result','request_id':'unrelated','ok':True})
            self.assertFalse(prepare._reconcile_service_terminal(service))
            self.assertEqual(service['busy_request_id'],'old')
            prepare._service_event(events,{'type':'beautygrpo_result','request_id':'old','ok':False})
            before=events.read_bytes()
            self.assertTrue(prepare._reconcile_service_terminal(service))
            self.assertIsNone(service['busy_request_id'])
            self.assertNotIn('busy_event_log',service)
            self.assertEqual(events.read_bytes(),before)

    def test_desktop_requests_reuse_one_isolated_service_process(self):
        """The expensive configured runtime is started once per model identity."""
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); config_path=root/'installation.json'; config_path.write_text('{}')
            config={'python':'configured-python','config_path':str(config_path),
                    'config_sha256':'config-identity'}
            calls=[]
            class Stdin:
                def write(self, text):
                    request=json.loads(text); output=Path(request['output']); output.mkdir(parents=True,exist_ok=True)
                    prepare.write_json(output/'result.json',{'ok':True,'no_change':True,'result':None,'layer':None})
                    prepare._service_event(request['event_log'],{'type':'beautygrpo_result',
                        'request_id':request['request_id'],'ok':True})
                def flush(self): pass
                def close(self): pass
            class Process:
                pid=4321
                stdin=Stdin()
                returncode=None
                def poll(self): return None
                def wait(self,timeout=None): self.returncode=0; return 0
            def launch(*args,**kwargs):
                calls.append((args,kwargs)); return Process()
            prepare._stop_beauty_service()
            storage = {'TEMP': r'D:\cosplay-tests', 'TMP': r'D:\cosplay-tests',
                'HF_HOME': r'D:\codex-models\cache', 'HF_HUB_CACHE': r'D:\codex-models\cache\hub',
                'TORCH_HOME': r'D:\codex-models\cache\torch',
                'TORCHINDUCTOR_CACHE_DIR': r'D:\codex-models\cache\inductor'}
            with patch.object(prepare,'installation_path',return_value=config_path), \
                    patch.object(prepare,'load_installation',return_value=config), \
                    patch.object(prepare.subprocess,'Popen',side_effect=launch), \
                    patch.object(prepare,'_stop_process'), \
                    patch.dict(os.environ, {**storage, worker.OFFLOAD_ENV: 'nunchaku_cuda',
                        'PYTHONPATH': r'C:\foreign\site-packages',
                        'PYTHONHOME': r'C:\foreign\python', 'PYTHONUSERBASE': r'C:\foreign\user'}):
                first=prepare.prepare_for_workbench({},root,root/'one')
                second=prepare.prepare_for_workbench({},root,root/'two')
                with patch.dict(config, {'torch_cpu_threads': 4}):
                    third=prepare.prepare_for_workbench({},root,root/'three')
                prepare._stop_beauty_service()
            self.assertEqual(len(calls),2)
            self.assertFalse(first['runtime']['service_reused'])
            self.assertTrue(second['runtime']['service_reused'])
            self.assertEqual(first['runtime']['service_pid'],4321)
            self.assertEqual(second['runtime']['pipeline_reused'],True)
            self.assertFalse(third['runtime']['service_reused'])
            self.assertEqual(third['runtime']['execution_profile']['torch_cpu_threads'], 4)
            self.assertEqual(calls[1][1]['env'][worker.OFFLOAD_ENV], 'nunchaku_cuda')
            environment = calls[0][1]['env']
            self.assertEqual(environment[worker.OFFLOAD_ENV], 'nunchaku_cuda')
            for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
                self.assertNotIn(key, environment)
            self.assertEqual(environment['PYTHONNOUSERSITE'], '1')
            self.assertEqual(environment['PYTHONUTF8'], '1')
            self.assertEqual(environment['PYTHONIOENCODING'], 'utf-8')
            self.assertEqual(environment['HF_HUB_OFFLINE'], '1')
            self.assertEqual(environment['TRANSFORMERS_OFFLINE'], '1')
            self.assertEqual({key: environment[key] for key in storage}, storage)


if __name__=='__main__':
    unittest.main()
