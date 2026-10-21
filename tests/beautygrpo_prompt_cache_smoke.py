"""Small CPU-only contracts for installation-bound official prompt embeddings."""
from contextlib import nullcontext
import copy
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import beautygrpo_worker as worker


SHAPES = {'prompt_embeds': [1, 3, 4], 'pooled_prompt_embeds': [1, 2]}
RUNTIME = {'python_path': sys.executable, 'versions': {'fixture': 'cpu'}, 'source': 'fixture'}
DEVICE = {'device': 'cuda:0', 'name': 'fixture', 'math': 'fixture'}


class PromptCacheContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        quantized = patch('beautygrpo_nunchaku.installed_model', return_value=(self.root/'fixture-int4', {}))
        quantized.start(); self.addCleanup(quantized.stop)
        self.config = {'config_path': str(self.root/'installation.json'), 'config_sha256': 'config',
            'python': sys.executable, 'verification_receipt_sha256': 'installation-receipt',
            'base_model_root': str(self.root), 'lora_path': str(self.root/'lora.safetensors'),
            'base_manifest_sha256': 'manifest', 'offload': 'sequential_cpu_offload',
            'verification_receipt': {'manifest': {'files': [
                {'path': 'text_encoder/config.json', 'bytes': 20, 'sha256': 'clip', 'mtime_ns': 1},
                {'path': 'text_encoder_2/config.json', 'bytes': 20, 'sha256': 't5', 'mtime_ns': 1},
                {'path': 'tokenizer/tokenizer_config.json', 'bytes': 20, 'sha256': 'tokenizer', 'mtime_ns': 1}]},
                'lora': {'tensor_keys': [f'transformer.layer{i}.lora_{kind}.weight'
                    for i in range(342) for kind in ('A', 'B')]}}}
        self.tensors = {'prompt_embeds': torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 4),
                        'pooled_prompt_embeds': torch.tensor([[.5, -.5]], dtype=torch.bfloat16)}
        self.shapes = patch.object(worker, 'PROMPT_TENSORS', SHAPES)
        self.shapes.start()
        self.addCleanup(self.shapes.stop)
        self.runtime = patch.object(worker, '_prompt_cache_runtime', return_value=RUNTIME)
        self.runtime.start()
        self.addCleanup(self.runtime.stop)

    def publish(self):
        return worker._publish_prompt_embeddings(self.config, self.tensors, RUNTIME, DEVICE, [])

    def installed(self):
        with patch.object(worker, '_load_weight_installation', return_value=copy.deepcopy(self.config)):
            return worker.load_installation(self.config['config_path'])

    def test_exact_bf16_roundtrip_and_artifact_identity_bind_inference_cache(self):
        receipt = self.publish()
        config = self.installed()
        tensors, evidence = worker.load_prompt_embeddings(config)
        for name in self.tensors:
            self.assertTrue(torch.equal(tensors[name], self.tensors[name]))
            self.assertEqual(tensors[name].dtype, torch.bfloat16)
            self.assertEqual(tensors[name].device.type, 'cpu')
        self.assertEqual(evidence['artifact_sha256'], receipt['artifact']['sha256'])
        self.assertFalse(evidence['encoder_model_loading'])
        self.assertEqual(worker.inference_identity(config)['execution_profile']['prompt_conditioning']
                         ['receipt_sha256'], config['_prompt_embedding_identity'])
        self.assertFalse(list(self.root.glob('*.tmp-*')))

    def test_missing_cache_fails_before_gpu_or_pipeline_construction(self):
        from diffusers import FluxKontextPipeline
        with patch.dict(worker._PIPELINE_CACHE, {}, clear=True), \
                patch.object(torch.cuda, 'is_available') as cuda, \
                patch.object(FluxKontextPipeline, 'from_pretrained') as load:
            self.assertRaisesRegex(ValueError, '--prepare-prompt-embeddings', worker.load_pipeline, self.config)
        cuda.assert_not_called()
        load.assert_not_called()

    def test_prompt_encoder_tokenizer_receipt_and_runtime_changes_are_rejected(self):
        receipt = self.publish()
        for changed in ('prompt', 'tokenizer', 'encoder', 'receipt', 'dtype'):
            config = copy.deepcopy(self.config)
            candidate = copy.deepcopy(receipt)
            if changed == 'prompt':
                candidate['binding']['encoding']['prompt'] += ' changed'
            elif changed in ('tokenizer', 'encoder'):
                config['verification_receipt']['manifest']['files'][2 if changed == 'tokenizer' else 0]['sha256'] = 'changed'
            elif changed == 'receipt':
                config['verification_receipt_sha256'] = 'changed'
            else:
                candidate['binding']['encoding']['dtype'] = 'float32'
            with self.subTest(changed=changed):
                self.assertRaisesRegex(ValueError, 'stale receipt binding',
                    worker._validate_prompt_cache_receipt, config, candidate)
        with patch.object(worker, '_prompt_cache_runtime', return_value={'changed': True}):
            self.assertRaisesRegex(ValueError, 'encoding runtime',
                worker._validate_prompt_cache_receipt, self.config, receipt, runtime=True)

    def test_wrong_dtype_shape_nonfinite_and_modified_bytes_cannot_be_used(self):
        for kind in ('dtype', 'shape', 'nan'):
            tensors = {name: tensor.clone() for name, tensor in self.tensors.items()}
            if kind == 'dtype':
                tensors['prompt_embeds'] = tensors['prompt_embeds'].float()
            elif kind == 'shape':
                tensors['prompt_embeds'] = tensors['prompt_embeds'][:, :2]
            else:
                tensors['prompt_embeds'][0, 0, 0] = float('nan')
            with self.subTest(kind=kind):
                self.assertRaisesRegex(ValueError, 'invalid tensors', worker._publish_prompt_embeddings,
                    self.config, tensors, RUNTIME, DEVICE, [])
        receipt = self.publish()
        path = Path(receipt['artifact']['path'])
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        self.assertRaisesRegex(ValueError, 'changed tensor artifact',
            worker._validate_prompt_cache_receipt, self.config, receipt)

    def test_explicit_prepare_reuses_valid_receipt_without_model_execution(self):
        from diffusers import FluxKontextPipeline
        self.publish()
        with patch.object(worker, '_load_weight_installation', return_value=self.config), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'device_count', return_value=1), \
                patch.object(worker, '_prompt_encoding_device', return_value=DEVICE), \
                patch.object(FluxKontextPipeline, 'from_pretrained') as load:
            result = worker.prepare_prompt_embeddings(self.config['config_path'])
        self.assertTrue(result['prepare_reused'])
        load.assert_not_called()

    def test_explicit_prepare_loads_only_text_components_and_native_encode_prompt(self):
        from diffusers import FluxKontextPipeline
        pipe = SimpleNamespace(text_encoder=SimpleNamespace(dtype=torch.bfloat16),
            text_encoder_2=SimpleNamespace(dtype=torch.bfloat16),
            encode_prompt=Mock(return_value=(self.tensors['prompt_embeds'],
                                           self.tensors['pooled_prompt_embeds'], None)))
        with patch.object(worker, '_load_weight_installation', return_value=self.config), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'device_count', return_value=1), \
                patch.object(torch.cuda, 'device', return_value=nullcontext()), \
                patch.object(torch.random, 'fork_rng', return_value=nullcontext()), \
                patch.object(worker, '_prompt_encoding_device', return_value=DEVICE), \
                patch('accelerate.cpu_offload') as offload, \
                patch.object(FluxKontextPipeline, 'from_pretrained', return_value=pipe) as load:
            result = worker.prepare_prompt_embeddings(self.config['config_path'])
        self.assertFalse(result['prepare_reused'])
        for name in ('transformer', 'vae', 'image_encoder', 'feature_extractor'):
            self.assertIsNone(load.call_args.kwargs[name])
        self.assertTrue(load.call_args.kwargs['local_files_only'])
        self.assertEqual(offload.call_count, 2)
        pipe.encode_prompt.assert_called_once_with(prompt=worker.RECIPE['prompt'], prompt_2=None,
            device=torch.device('cuda:0'), num_images_per_prompt=1, max_sequence_length=512, lora_scale=None)
        self.assertEqual(result['generated_images'], 0)

    def test_runtime_loader_omits_all_encoders_and_preserves_official_lora(self):
        from diffusers import AutoencoderKL, FluxKontextPipeline
        import beautygrpo_nunchaku
        self.publish(); config = self.installed()
        pipe = SimpleNamespace(vae=SimpleNamespace(enable_slicing=Mock(), enable_tiling=Mock()),
            to=Mock())
        pipe.to.return_value = pipe
        config['vae_model_root'] = str(self.root/'ssd'/'vae')
        transformer = object()
        with patch.dict(worker._PIPELINE_CACHE, {}, clear=True), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'device_count', return_value=1), \
                patch.object(torch.cuda, 'set_device'), \
                patch.object(worker, '_prompt_encoding_device', return_value=DEVICE), \
                patch.object(worker, 'verify_base', return_value={'fixture': True}), \
                patch.object(beautygrpo_nunchaku, 'installed_model', return_value=(self.root/'int4.safetensors', {'fixture': True})), \
                patch.object(beautygrpo_nunchaku, 'bind_lora', return_value={'consumed_source_tensors': 684}) as bind, \
                patch.object(beautygrpo_nunchaku, 'load_transformer', return_value=transformer) as transformer_load, \
                patch.object(AutoencoderKL, 'from_pretrained', return_value=pipe.vae) as vae_load, \
                patch.object(FluxKontextPipeline, 'from_pretrained', return_value=pipe) as load:
            actual, evidence = worker.load_pipeline(config)
        self.assertIs(actual, pipe)
        for name in ('text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2'):
            self.assertIsNone(load.call_args.kwargs[name])
        self.assertEqual(load.call_args.kwargs['torch_dtype'], torch.bfloat16)
        self.assertIs(load.call_args.kwargs['transformer'], transformer)
        self.assertIs(load.call_args.kwargs['vae'], pipe.vae)
        self.assertEqual(vae_load.call_args.args, (str((self.root/'ssd'/'vae').resolve()),))
        self.assertEqual(vae_load.call_args.kwargs['torch_dtype'], torch.bfloat16)
        self.assertEqual(evidence['loaded_model_locations']['vae'], str((self.root/'ssd'/'vae').resolve()))
        self.assertEqual(evidence['lora_verified']['consumed_source_tensors'], 684)
        self.assertEqual(bind.call_args.args[0], transformer)
        self.assertEqual(transformer_load.call_args.args, (self.root/'int4.safetensors',))
        self.assertEqual(transformer_load.call_args.kwargs['device'], 'cuda:0')
        self.assertEqual(transformer_load.call_args.kwargs['torch_dtype'], torch.bfloat16)
        self.assertEqual(transformer_load.call_args.kwargs['io_profile'], [])
        pipe.to.assert_called_once_with('cuda:0')
        self.assertIn('installed_prompt_embeddings_loading',
                      [stage['stage'] for stage in evidence['model_load_profile']['stages']])

    def test_native_cached_encode_prompt_does_no_encoder_work(self):
        from diffusers import FluxKontextPipeline
        class Transformer:
            dtype = torch.bfloat16
        transformer = Transformer()
        pipe = FluxKontextPipeline(scheduler=None, vae=None, text_encoder=None, tokenizer=None,
            text_encoder_2=None, tokenizer_2=None, transformer=transformer)
        with patch.object(pipe, '_get_clip_prompt_embeds') as clip, \
                patch.object(pipe, '_get_t5_prompt_embeds') as t5:
            prompt, pooled, text_ids = pipe.encode_prompt(prompt=None, device=torch.device('cpu'),
                prompt_embeds=self.tensors['prompt_embeds'],
                pooled_prompt_embeds=self.tensors['pooled_prompt_embeds'])
        self.assertIs(prompt, self.tensors['prompt_embeds'])
        self.assertIs(pooled, self.tensors['pooled_prompt_embeds'])
        self.assertEqual(list(text_ids.shape), [3, 3])
        self.assertEqual(text_ids.dtype, torch.bfloat16)
        clip.assert_not_called(); t5.assert_not_called()

    def test_component_timing_uses_native_loader_and_restores_on_failure(self):
        from diffusers.pipelines import pipeline_utils
        records = []
        with patch.object(pipeline_utils, 'load_sub_model', side_effect=RuntimeError('fixture')) as native:
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with worker._measure_component_loading(records):
                    pipeline_utils.load_sub_model(name='transformer')
            self.assertIs(pipeline_utils.load_sub_model, native)
        self.assertEqual(records[0]['component'], 'transformer')
        self.assertEqual(records[0]['stage'], 'component_loading')
        self.assertGreaterEqual(records[0]['seconds'], 0)
        self.assertFalse(torch.cuda.is_initialized(), 'CPU contract must never initialize CUDA')

    def test_existing_installation_cli_prepares_prompt_resource_once_after_verification(self):
        import beautygrpo_prepare as prepare
        calls = []
        def verified(path, *, receipt_path):
            calls.append(('verify', path, receipt_path))
            return {'schema': worker.VERIFICATION_SCHEMA, 'base': {'file_count': 25},
                    'lora': {'tensor_count': 684}}
        def encoded(path):
            calls.append(('prepare', path))
            return {'receipt_path': str(worker._prompt_cache_path(self.config)), 'prepare_reused': False}
        path = Path(self.config['config_path'])
        raw_config = {key: value for key, value in self.config.items() if key != 'verification_receipt'}
        with patch.object(sys, 'argv', ['beautygrpo_prepare.py', '--verify-installation', str(path)]), \
                patch.object(prepare, '_read_installation_config', return_value=raw_config), \
                patch.object(prepare, 'verify_installation', side_effect=verified), \
                patch.object(prepare, 'prepare_prompt_embeddings', side_effect=encoded), \
                patch.object(sys, 'stdout', new_callable=StringIO) as stdout:
            prepare.main()
        self.assertEqual([call[0] for call in calls], ['verify', 'prepare'])
        self.assertEqual(calls[0][1], path)
        self.assertEqual(calls[1][1], path)
        self.assertTrue(json.loads(stdout.getvalue())['prompt_embeddings_ready'])


if __name__ == '__main__':
    unittest.main()
