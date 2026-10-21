"""Meaningful CPU contracts for complete-face generation and strict binding."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import beautygrpo_worker as worker


def face_region(center=(128,128),radius=(65,90),box=(0,0,256,256),identifier='face-01',status='full_visible_face'):
    points = np.tile(np.asarray(center,np.float64),(478,1))
    angles = np.linspace(-np.pi/2,3*np.pi/2,len(worker.FACE_OVAL),endpoint=False)
    points[list(worker.FACE_OVAL)] = np.stack((center[0]+radius[0]*np.cos(angles),center[1]+radius[1]*np.sin(angles)),axis=1)
    labels = np.full((box[3]-box[1],box[2]-box[0]),2,np.uint8)
    labels[90:106,86:108] = 8
    labels[90:106,150:172] = 9
    labels[77:87,84:113] = 6
    labels[77:87,148:176] = 7
    labels[174:181,113:143] = 12
    visible = np.ones(labels.shape,np.float32)
    if status=='visible_upper_face_only':
        visible[98:] = 0
        labels[174:181,113:143] = 2
    region = {'id':identifier,'native_box':list(box),'landmarks':points.tolist(),
              'texture_eligibility':{'status':status}}
    fields = {'labels':labels,'skin_weight':np.zeros(labels.shape,np.float32),
              'visible_face_support':visible,'features_gate':np.ones(labels.shape,np.float32)}
    fields['skin_weight'][52:170,80:176] = 1
    return region,fields


class BeautyWorkerContracts(unittest.TestCase):
    def setUp(self):
        quantized = patch('beautygrpo_nunchaku.installed_model', return_value=(Path('fixture-int4'), {}))
        quantized.start(); self.addCleanup(quantized.stop)

    def test_only_nunchaku_execution_preserves_installation_and_rejects_retired_values(self):
        config = {'base_manifest_sha256': 'test', 'offload': 'sequential_cpu_offload'}
        installed = worker.model_identity(config)
        with patch.dict(os.environ, {worker.OFFLOAD_ENV: ''}):
            identity = worker.inference_identity(config)
            self.assertEqual(identity['offload'], 'nunchaku_cuda')
            self.assertEqual(worker.model_identity(config), installed)
            self.assertEqual(identity['recipe'], worker.RECIPE)
            self.assertEqual(identity['quantization'], 'SVDQuant_INT4_W4A4_rank32')
            self.assertFalse(identity['source_texture_restoration'])
            self.assertNotEqual(identity, worker.inference_identity({**config, 'torch_cpu_threads': 4}))
            with patch.object(worker, 'sha', return_value='changed-readonly-loader'):
                self.assertNotEqual(identity, worker.inference_identity(config))
        for policy in ('balanced_cpu_offload', 'sequential_cpu_offload', 'group_cpu_offload', 'disk_offload'):
            with self.subTest(policy=policy):
                with patch.dict(os.environ, {worker.OFFLOAD_ENV: policy}):
                    self.assertRaisesRegex(ValueError, 'nunchaku_cuda', worker.execution_profile, config)
                with patch.dict(os.environ, {worker.OFFLOAD_ENV: 'nunchaku_cuda'}):
                    for field in ('execution_offload', '_execution_offload'):
                        self.assertRaisesRegex(ValueError, 'nunchaku_cuda', worker.execution_profile,
                            {**config, field: policy})





    def test_inference_measurements_preserve_results_and_dynamic_forward_hooks(self):
        import torch
        class Transformer(torch.nn.Module):
            def forward(self, values):
                self.forward = lambda x: x + 1
                return values + 1
        class VAE:
            def encode(self, value): return value
            def decode(self, value): return value
        class Pipe:
            transformer = Transformer()
            vae = VAE()
            def encode_prompt(self): return 'prompt'
            def prepare_latents(self): return 'latents'
        pipe = Pipe()
        with patch.object(worker, '_runtime_counters', return_value={'process_cpu_seconds': 0}):
            with worker._measure_inference(pipe) as evidence:
                self.assertEqual(pipe.encode_prompt(), 'prompt')
                self.assertEqual(pipe.vae.encode('image'), 'image')
                self.assertEqual(pipe.prepare_latents(), 'latents')
                for _ in range(2):
                    self.assertEqual(pipe.transformer(torch.tensor(3)).item(), 4)
                self.assertEqual(pipe.vae.decode('output'), 'output')
        self.assertEqual([row['stage'] for row in evidence['stages']], [
            'prompt_encoding', 'vae_encode', 'latent_preparation', 'transformer_forward',
            'transformer_forward', 'vae_decode'])
        self.assertNotIn('encode_prompt', vars(pipe)); self.assertNotIn('encode', vars(pipe.vae))
        self.assertFalse(pipe.transformer._forward_pre_hooks)
        self.assertFalse(pipe.transformer._forward_hooks)
        with patch.object(worker, '_runtime_counters', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                with worker._measure_inference(pipe):
                    raise RuntimeError('inference failed')
        self.assertNotIn('encode_prompt', vars(pipe))
        self.assertFalse(pipe.transformer._forward_hooks)

    def test_pipeline_cache_refuses_different_execution_profile_before_cuda(self):
        from types import SimpleNamespace
        config = {'base_model_root': '.', 'lora_path': __file__, 'base_manifest_sha256': 'test',
                  'offload': 'sequential_cpu_offload', '_execution_offload': 'nunchaku_cuda',
                  'verification_receipt': {'lora': {'tensor_keys': [
                      f'transformer.layer{i}.lora_{kind}.weight' for i in range(342) for kind in ('A', 'B')]}}}
        key = (str(Path(config['base_model_root']).resolve()), str(Path(config['lora_path']).resolve()),
               worker._canonical_sha(worker.inference_identity(config)))
        cuda = Mock()
        modules = {'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=cuda)),
                   'diffusers': SimpleNamespace(FluxKontextPipeline=Mock()),
                   **{name: SimpleNamespace() for name in ('peft', 'transformers', 'accelerate')}}
        with patch.dict(worker._PIPELINE_CACHE, {key: (object(), {})}, clear=True), \
                patch.dict(sys.modules, modules):
            self.assertRaisesRegex(ValueError, 'cannot replace', worker.load_pipeline,
                {**config, 'torch_cpu_threads': 4})
        cuda.assert_not_called()

    def test_pipeline_requires_complete_receipt_inventory_without_rescanning_lora(self):
        for keys in (None, [], ['transformer.layer0.lora_A.weight'], ['invalid.weight'] * 684):
            with self.subTest(keys=keys), patch.object(worker, 'verify_lora_file') as verify, \
                    patch.object(worker, 'verify_base') as base:
                with self.assertRaisesRegex(ValueError, 'tensor key inventory is incomplete'):
                    worker.load_pipeline({'verification_receipt': {'lora': {'tensor_keys': keys}}})
                verify.assert_not_called()
                base.assert_not_called()

    def test_forehead_extension_follows_skin_to_hairline_without_spreading_to_lower_face_or_hair(self):
        region,fields=face_region()
        points=np.asarray(region['landmarks'])
        points[33]=[86,98];points[133]=[108,98]
        points[362]=[150,98];points[263]=[172,98]
        region['landmarks']=points.tolist()
        # Skin continues above canonical oval, to a hairline at y=15. An
        # adjacent connected skin-labelled prop below the face must stay out.
        fields['labels'][:15]=14
        fields['labels'][:,195:]=3
        fields['labels'][35:43,124:130]=14  # A real hair strand cuts the domain.
        weight,policy=worker.composition_weight(region,fields,[0,0,256,256])
        self.assertGreater(weight[24,110],.99,'Visible upper forehead should keep generated RGB')
        self.assertEqual(weight[10,110],0,'Hair cannot become a forehead extension')
        self.assertEqual(weight[40,127],0,'Hard hair strand remains source-owned')
        self.assertEqual(weight[240,128],0,'Lower skin-labelled props cannot join upward extension')
        self.assertEqual(weight[60,220],0,'Outside the canonical face-width corridor is excluded')
        self.assertGreater(policy['forehead_extension']['added_native_pixels'],0)
        self.assertEqual(policy['forehead_extension']['seed_band_fraction'],.2)
        # Geometry alone is not a reliable seed through a non-face occluder.
        fields['labels'][:75]=3
        withheld,policy=worker.composition_weight(region,fields,[0,0,256,256])
        self.assertEqual(policy['forehead_extension']['added_native_pixels'],0)
        self.assertEqual(withheld[24,110],0)

    def test_installation_is_complete_local_pinned_and_base_tampering_cannot_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);base=root/'base';base.mkdir();regions=root/'regions';regions.mkdir()
            rows=[]
            replaced = sorted(worker.REPLACED_BASE_WEIGHTS)
            for index in range(25):
                name=('model_index.json' if index==0 else 'weight.safetensors' if index==1
                      else 'vae/diffusion_pytorch_model.safetensors' if index==2
                      else 'vae/config.json' if index==3
                      else replaced[index-4] if 4<=index<7 else f'config-{index}.json')
                path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'x')
                rows.append({'path':name,'bytes':1,'local_sha256':worker.sha(path),'verified':True})
            catalog=root/'manifest.json';worker.write_json(catalog,{'status':'ready_for_local_inference',
                'hf_repo':worker.BASE_ID,'hf_revision':worker.BASE_REVISION,'completed_files':rows})
            python=root/'python.exe';python.write_bytes(b'local')
            lora=root/'lora.safetensors';lora.write_bytes(b'wrong')
            registry=root/'registry.json';registry.write_text('{}',encoding='utf8')
            config={'schema':worker.CONFIG_SCHEMA,'ready':True,'python':str(python),'regions_python':str(python),
                    'base_model_root':str(base),'regions_model_root':str(regions),'lora_path':str(lora),
                    'face_registry':str(registry),'base_manifest':str(catalog),'base_manifest_sha256':worker.sha(catalog),
                    'offload':'sequential_cpu_offload','recipe':dict(worker.RECIPE)}
            installation=root/'installation.json';worker.write_json(installation,config)
            # A real INT4 installation never loads these BF16 shards. Its
            # original download evidence still describes all official files.
            for name in replaced:
                (base/name).unlink()
            with patch.dict(worker.BASE_WEIGHT_HASHES,{'weight.safetensors':worker.sha(base/'weight.safetensors')},clear=True):
                keys = [f'transformer.layer{index}.lora_{kind}.weight'
                        for index in range(342) for kind in ('A','B')]
                with patch.object(worker,'verify_lora_file',return_value=keys):
                    receipt=worker.verify_installation(installation,
                        receipt_path=installation.with_suffix('.verification.json'))
                self.assertEqual(receipt['base']['inventory_scope'],worker.BASE_INVENTORY_SCOPE)
                self.assertEqual(receipt['base']['file_count'],22)
                self.assertFalse(set(replaced)&{row['path'] for row in receipt['base']['files']})
                observed=[]
                original_sha=worker.sha
                def track(path):
                    observed.append(Path(path).resolve())
                    return original_sha(path)
                with patch.object(worker,'sha',side_effect=track):
                    loaded=worker.load_installation(installation)
                self.assertNotIn((base/'weight.safetensors').resolve(),observed,
                                 'Runtime receipt validation must not hash base weights')
                self.assertNotIn(lora.resolve(),observed,
                                 'Runtime receipt validation must not hash the LoRA')
                # Runtime code changes invalidate preparation caches, not the
                # installation-time proof of unchanged model weight bytes.
                runtime_identity = worker._runtime_identity
                with patch.object(worker, '_runtime_identity', side_effect=lambda c:
                        {**runtime_identity(c), 'worker_sha256': 'a' * 64}):
                    worker.load_installation(installation)
                with patch.dict(os.environ, {worker.OFFLOAD_ENV: 'nunchaku_cuda'}):
                    current = worker.load_installation(installation)
                    self.assertEqual(current['verification_receipt']['runtime']['offload'], 'sequential_cpu_offload')
                    self.assertEqual(worker.inference_identity(current)['offload'], 'nunchaku_cuda')
                for policy in ('group_cpu_offload', 'sequential_cpu_offload'):
                    with patch.dict(os.environ, {worker.OFFLOAD_ENV: policy}):
                        self.assertRaisesRegex(ValueError, 'nunchaku_cuda', worker.load_installation, installation)
                with patch.object(worker, '_runtime_identity', side_effect=lambda c:
                        {**runtime_identity(c), 'gpu_id': 7}):
                    self.assertRaises(ValueError, worker.load_installation, installation)
                evidence=worker.verify_base(loaded)
                self.assertEqual(evidence['verified_file_count'],22)
                original_receipt=copy.deepcopy(receipt)
                receipt['base'].pop('inventory_scope')
                worker.write_json(installation.with_suffix('.verification.json'),receipt)
                self.assertRaises(ValueError,worker.load_installation,installation)
                worker.write_json(installation.with_suffix('.verification.json'),original_receipt)
                (base/'weight.safetensors').write_bytes(b'y')  # Same size is not hash identity.
                self.assertRaises(ValueError,worker.load_installation,installation)
                with patch.object(worker,'load_pipeline') as load:
                    self.assertRaises(ValueError,worker.verify_lora_file,lora)
                load.assert_not_called()
                config['ready']=False;worker.write_json(installation,config)
                self.assertRaises(ValueError,worker.load_installation,installation)

    def test_generic_complete_face_crop_does_not_shrink_to_forehead(self):
        region,fields = face_region(radius=(50,70),status='visible_upper_face_only')
        fields['skin_weight'][100:] = 0
        box = worker.face_focused_box(region,(256,256),fields)
        self.assertGreaterEqual(box[3],198)
        self.assertEqual(box[2]-box[0],box[3]-box[1])
        moved = copy.deepcopy(region)
        moved['landmarks'] = (np.asarray(moved['landmarks'])+[256,0]).tolist()
        moved['native_box'] = [256,0,512,256]
        shifted = worker.face_focused_box(moved,(512,256),fields)
        self.assertTrue(shifted[0]>=256)
        self.assertEqual(shifted[1:], [box[1],box[2]+256,box[3]])
        self.assertRaises(ValueError,worker.face_focused_box,{**region,'landmarks':[]},(256,256),fields)

    def test_original_generated_features_keep_full_dose_while_occluders_are_excluded(self):
        region,fields = face_region()
        fields['labels'][120:155,125:140] = 3  # A real non-face material label.
        fields['labels'][55:150,50:60] = 14  # Stray hair is retained from source.
        weight,policy = worker.composition_weight(region,fields,[0,0,256,256])
        self.assertGreater(weight[96,95],.99,'Eyes must be part of the generated face')
        self.assertGreater(weight[96,160],.99)
        self.assertGreater(weight[177,122],.99,'Lips must be part of the generated face')
        self.assertEqual(weight[135,130],0,'Observed non-face prop must not be overwritten')
        self.assertEqual(weight[70,5],0)
        self.assertFalse(policy['features_gate_applied'])
        self.assertFalse(policy['skin_only'])
        self.assertFalse(policy['source_texture_restoration'])
        self.assertGreater(policy['face_feature_pixels_included'],0)
        self.assertEqual(weight.dtype,np.float32)

    def test_missing_mouth_withholds_occluding_lower_face_but_retains_whole_visible_eyes(self):
        region,fields = face_region(status='visible_upper_face_only')
        weight,policy = worker.composition_weight(region,fields,[0,0,256,256])
        self.assertGreater(weight[100,95],0,'Visible lower eyelid stays in generated upper face')
        self.assertEqual(weight[130,130],0,'Missing-mouth paper cannot become generated skin')
        self.assertGreater(weight[70,130],.99)
        self.assertEqual(policy['qualification_status'],'visible_upper_face_only')
        self.assertEqual(len(policy['limitations']),2)

    def test_letterbox_roundtrip_preserves_aspect_and_discards_padding(self):
        source = Image.new('RGB',(120,240),(12,23,34))
        crop,model,mapping = worker.prepare_roi(source,[0,0,120,240],size=128)
        self.assertEqual(model.size,(128,128))
        self.assertEqual(mapping['resized_size'],[64,128])
        self.assertEqual(mapping['padding_offset'],[32,0])
        self.assertEqual(model.getpixel((0,64)),(127,127,127))
        self.assertEqual(model.getpixel((64,64)),(12,23,34))
        x,y=mapping['padding_offset'];w,h=mapping['resized_size']
        restored=model.crop((x,y,x+w,y+h)).resize(crop.size,Image.Resampling.LANCZOS)
        self.assertTrue(np.array_equal(np.asarray(restored),np.asarray(crop)))

    def test_no_faces_never_loads_gpu_or_alternate_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.png';Image.new('RGB',(256,256)).save(source)
            regions=root/'regions.json';worker.write_json(regions,{'schema_version':'portrait_regions.v1',
                'source_sha256':worker.sha(source),'source_native_size':[256,256],'regions':[]})
            config={'base_manifest_sha256':'test','offload':'sequential_cpu_offload'}
            with patch.object(worker,'load_pipeline') as load:
                result=worker.run(source,regions,config,root/'output')
            load.assert_not_called()
            self.assertEqual(result['status'],'no_faces_available')
            self.assertFalse(result['inference_completed'])

    def test_multi_face_request_uses_one_pipeline_and_retains_generated_rgb(self):
        import torch
        original_generator=torch.Generator
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.png';Image.new('RGB',(512,256),(35,70,90)).save(source)
            regions=[]
            for index in range(2):
                region,fields=face_region(identifier=f'face-{index+1:02d}')
                if index:
                    region['native_box']=[256,0,512,256]
                    region['landmarks']=(np.asarray(region['landmarks'])+[256,0]).tolist()
                asset=root/f'region-{index}.npz';np.savez(asset,**fields);region['npz_path']=str(asset)
                regions.append(region)
            manifest=root/'regions.json';worker.write_json(manifest,{'schema_version':'portrait_regions.v1',
                'source_sha256':worker.sha(source),'source_native_size':[512,256],'regions':regions})
            class Pipeline:
                calls=0
                def __call__(self,**kwargs):
                    self.calls+=1
                    self.last=kwargs
                    kwargs['callback_on_step_end'](self,0,None,{})
                    return type('Result',(),{'images':[Image.new('RGB',(1024,1024),(220,80,45))]})()
            pipe=Pipeline();config={'base_manifest_sha256':'test','offload':'sequential_cpu_offload','gpu_id':0}
            pipe._execution_device = torch.device('cpu')
            pipe._beauty_prompt_embeddings = {
                'prompt_embeds': torch.zeros((1, 2, 4), dtype=torch.bfloat16),
                'pooled_prompt_embeds': torch.zeros((1, 2), dtype=torch.bfloat16)}
            events=[]
            with patch.object(worker,'load_pipeline',return_value=(pipe,{'lora_verified':{'loaded_tensor_count':684}})) as load, \
                    patch.object(torch.cuda,'device',return_value=nullcontext()), \
                    patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'max_memory_allocated',return_value=1), \
                    patch.object(torch.cuda,'max_memory_reserved',return_value=2), \
                    patch.object(torch,'Generator',side_effect=lambda device:original_generator('cpu')):
                result=worker.run(source,manifest,config,root/'output',progress=lambda *event:events.append(event))
            self.assertEqual(load.call_count,1)
            self.assertEqual(pipe.calls,2)
            self.assertTrue(result['inference_completed'])
            for face in result['faces']:
                with Image.open(face['generated_native_roi']) as image:
                    self.assertEqual(image.getpixel((image.width//2,image.height//2)),(220,80,45))
                self.assertFalse(face['source_texture_restoration'])
                self.assertFalse(face['skin_only_residual'])
                self.assertEqual(worker.sha(face['generated_native_roi']),face['generated_native_roi_sha256'])
            self.assertTrue(events)
            self.assertEqual(pipe.last['guidance_scale'],2.5)
            self.assertEqual(pipe.last['num_inference_steps'],28)
            self.assertNotIn('prompt', pipe.last)
            self.assertEqual(pipe.last['max_sequence_length'],512)
            self.assertTrue(torch.equal(pipe.last['prompt_embeds'], pipe._beauty_prompt_embeddings['prompt_embeds']))
            self.assertEqual(pipe.last['generator'].initial_seed(),42)
            first_stages = [stage['stage'] for stage in result['faces'][0]['inference_profile']['stages']]
            second_stages = [stage['stage'] for stage in result['faces'][1]['inference_profile']['stages']]
            self.assertIn('prompt_embedding_device_transfer', first_stages)
            self.assertIn('prompt_embedding_device_reuse', second_stages)



if __name__=='__main__':
    unittest.main()
