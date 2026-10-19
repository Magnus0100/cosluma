"""CPU-only cache completion and unchanged cold handoff contracts."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import beautygrpo_prepare as prepare
import beautygrpo_worker as worker
from portrait_beauty_layer import freeze_prepared
import single_source_workflow as workflow
from runtime_memory import BEAUTY_COLD_PHYSICAL_BYTES, BEAUTY_COLD_COMMIT_BYTES


class BeautyCacheHandoff(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root/'source.png'
        Image.new('RGB',(12,10),(20,50,80)).save(source)
        self.run = {'run_id':'cache-handoff','source':{'path':str(source),
                    'sha256':worker.sha(source),'oriented_size':[12,10]},'options':{}}
        face = self.root/'face.task'; face.write_bytes(b'cpu-fixture')
        self.config = {'base_manifest_sha256':'test','offload':'sequential_cpu_offload',
                       'face_registry':'unused'}
        for target,kwargs in (
            ('beautygrpo_prepare.installation_path',{'return_value':self.root/'unused.json'}),
            ('beautygrpo_prepare.load_installation',{'return_value':self.config}),
            ('current_model_registry.load_registry',{'return_value':{'face_model_path':str(face)}}),
            ('beautygrpo_worker.run',{'side_effect':AssertionError('model inference forbidden')}),
            ('subprocess.Popen',{'side_effect':AssertionError('service start forbidden')}),
        ):
            mock = patch(target,**kwargs); mock.start(); self.addCleanup(mock.stop)

    def publish_fixture(self):
        seed = self.root/'seed'; seed.mkdir()
        record = prepare.prepare_source(self.run,self.root,seed)
        image = seed/'model.png'; Image.new('RGB',(8,8),(180,60,100)).save(image)
        weight = seed/'weight.npy'; np.save(weight,np.ones((8,8),np.float32))
        record.update(model_identity=worker.inference_identity(self.config),faces=[{
            'id':'face-01','native_box':[2,1,10,9],'generated_native_roi':str(image),
            'generated_native_roi_sha256':worker.sha(image),'model_output':str(image),
            'composition_weight':str(weight),'composition_weight_sha256':worker.sha(weight)}])
        result,_layer = freeze_prepared(record,seed/'layer')
        directory,_cached = prepare._cache_result(self.root,prepare._cache_binding(record,self.config,{'max_faces':8}))
        prepare._publish_cache(directory,prepare._cache_binding(record,self.config,{'max_faces':8}),
                               result,record['source_path'])
        return directory

    def test_completed_cache_owns_same_rgb_without_service_or_model_import(self):
        directory = self.publish_fixture(); before = copy.deepcopy(self.run)
        modules = set(sys.modules)
        result = prepare.prepare_cached_for_workbench(self.run,self.root,self.root/'owned')
        self.assertTrue(result['cache_hit']); self.assertEqual(result['generation_inferences_this_attempt'],0)
        self.assertEqual(result['runtime']['additional_model_cpu_bytes'],0)
        self.assertEqual(self.run,before)
        for asset in ('face-01.npz','face-01-model.png'):
            self.assertEqual((self.root/'owned/layer'/asset).read_bytes(),(directory/'layer'/asset).read_bytes())
        self.assertFalse({'torch','diffusers'} & (set(sys.modules)-modules))

    def test_true_miss_continues_same_prepared_source_once_and_rejects_mutation(self):
        output = self.root/'miss'
        with patch.object(prepare,'prepare_source',wraps=prepare.prepare_source) as source:
            self.assertIsNone(prepare.prepare_cached_for_workbench(self.run,self.root,output))
            with patch.object(prepare,'_run_regions',return_value=self.root/'regions.json'), \
                    patch.object(worker,'run',return_value={'inference_completed':False,'faces':[]}):
                result = prepare.prepare(self.run,self.root,output)
            self.assertTrue(result['no_change']); self.assertEqual(source.call_count,1)
        other = self.root/'tamper'
        self.assertIsNone(prepare.prepare_cached_for_workbench(self.run,self.root,other))
        (other/'source-display.png').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            prepare.prepare(self.run,self.root,other)

    def test_cancel_after_owned_copy_never_publishes_complete_receipt(self):
        self.publish_fixture(); state = {'cancel':False}; original = prepare._copy_layer
        def copied(*args):
            result = original(*args); state['cancel'] = True; return result
        with patch.object(prepare,'_copy_layer',side_effect=copied), self.assertRaises(InterruptedError):
            prepare.prepare_cached_for_workbench(self.run,self.root,self.root/'cancel',
                                                cancel=lambda:state['cancel'])
        self.assertFalse((self.root/'cancel/result.json').exists())

    def test_main_flow_hit_skips_release_and_miss_declares_current_memory_demand(self):
        result = {'ok':True,'no_change':True,'result':None,'layer':None,
                  'cache_hit':True,'generation_inferences_this_attempt':0}
        with patch.object(workflow,'_task_portrait_beauty_needed',return_value=True), \
                patch.object(prepare,'prepare_cached_for_workbench',return_value=result), \
                patch('start_qwen_image.release_for_run') as release:
            workflow._prepare_task_portrait_beauty(self.root,self.run,argparse.Namespace())
            release.assert_not_called()
        with patch.object(workflow,'_task_portrait_beauty_needed',return_value=True), \
                patch.object(prepare,'prepare_cached_for_workbench',return_value=None), \
                patch.object(prepare,'prepare_for_workbench',return_value=result), \
                patch('start_qwen_image.release_for_run',return_value={'state':'not_running'}) as release:
            workflow._prepare_task_portrait_beauty(self.root,self.run,argparse.Namespace())
            self.assertEqual(release.call_args.kwargs['next_stage_cpu_bytes'],BEAUTY_COLD_PHYSICAL_BYTES)
            self.assertEqual(release.call_args.kwargs['next_stage_commit_bytes'],BEAUTY_COLD_COMMIT_BYTES)


if __name__ == '__main__':
    unittest.main(verbosity=2)
