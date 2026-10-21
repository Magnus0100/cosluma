"""Relocated Qwen preflight and real launch arguments select the same weights."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import start_qwen_image as qwen


class StorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = self.root/'config/provider-settings.json'
        self.settings.parent.mkdir()
        self.storage = self.settings.with_name('qwen-model-storage.json')
        self.catalog = self.root/'catalog.json'
        self.catalog.write_text(json.dumps({'models':[
            {'role':'official_transformer','destination':'ComfyUI/models/diffusion_models/model.safetensors','size':3},
            {'role':'vae','destination':'ComfyUI/models/vae/vae.safetensors','size':2}],
            'provider_roles':{'fixture':['official_transformer']}}),encoding='utf8')
        for target, value in [('provider_settings.settings_path',self.settings),
                              ('start_qwen_image.resource_path',self.catalog),
                              ('start_qwen_image.deployment_root',self.root/'deployment')]:
            mock = patch(target,return_value=value);mock.start();self.addCleanup(mock.stop)

    def save(self, models):
        self.storage.write_text(json.dumps({'schema':'cosplay_qwen_model_storage.v1','models':models}),encoding='utf8')

    def test_preflight_and_launched_engine_share_explicit_storage(self):
        target = self.root/'ssd/diffusion_models/model.safetensors'
        target.parent.mkdir(parents=True);target.write_bytes(b'abc')
        self.save({'official_transformer':str(target)})
        files = qwen.model_files('fixture')
        self.assertEqual(files[0]['path'], str(target.resolve()))
        self.assertTrue(files[0]['complete'])
        command, _ = qwen.service_launch(self.root/'deployment')
        index = command.index('--extra-model-paths-config')
        generated = json.loads(Path(command[index+1]).read_text('utf8'))
        self.assertEqual(generated['official_transformer']['diffusion_models'], str(target.parent))
        self.assertTrue(generated['official_transformer']['is_default'])
        self.assertEqual(qwen.model_catalog()['models'][1]['destination'],'ComfyUI/models/vae/vae.safetensors')

    def test_changed_storage_requires_new_service_attestation(self):
        self.save({'official_transformer':str(self.root/'first/model.safetensors')})
        first = qwen.launch_policy()
        self.save({'official_transformer':str(self.root/'second/model.safetensors')})
        self.assertNotEqual(first['weight_storage_sha256'],qwen.launch_policy()['weight_storage_sha256'])

    def test_unknown_role_relative_path_or_wrong_filename_is_rejected(self):
        for models in ({'other':str(self.root/'model.safetensors')},
                       {'official_transformer':'relative/model.safetensors'},
                       {'official_transformer':str(self.root/'other.safetensors')}):
            with self.subTest(models=models):
                self.save(models)
                with self.assertRaises(ValueError):qwen.model_catalog()

    def test_portable_catalog_needs_no_user_profile_but_explicit_config_must_resolve(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch('provider_settings.settings_path', side_effect=RuntimeError('Could not determine home directory.')):
            self.assertEqual(qwen.model_catalog()['models'][0]['destination'],
                             'ComfyUI/models/diffusion_models/model.safetensors')
            os.environ['COSPLAY_CONFIG_ROOT'] = '~'
            with self.assertRaises(RuntimeError):qwen.model_catalog()


if __name__ == '__main__':
    unittest.main()
