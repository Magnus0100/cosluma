"""Component relocation preserves official bytes and stays in its root."""
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

import beautygrpo_worker as worker


class VAEStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root/'original'
        self.vae = self.base/'vae'
        self.vae.mkdir(parents=True)
        rows = []
        for name, data in [('config.json', b'{"sample_size":64}'),
                           ('diffusion_pytorch_model.safetensors', b'fixture official weights')]:
            (self.vae/name).write_bytes(data)
            rows.append({'path':'vae/'+name, 'bytes':len(data),
                         'local_sha256':hashlib.sha256(data).hexdigest()})
        self.config = {'base_model_root':str(self.base), '_manifest_catalog':{'completed_files':rows}}

    def test_relocated_component_is_the_verified_runtime_donor(self):
        original = worker._vae_component(self.config, verify_hash=True)
        relocated = self.root/'ssd'/'vae'
        shutil.copytree(self.vae, relocated)
        self.config['vae_model_root'] = str(relocated)
        shutil.rmtree(self.vae)
        actual = worker._vae_component(self.config, verify_hash=True)
        self.assertEqual(actual['root'], str(relocated.resolve()))
        self.assertEqual([r['sha256'] for r in actual['files']], [r['sha256'] for r in original['files']])
        self.assertEqual(actual, worker._vae_component(self.config))

    def test_equal_size_corrupt_copy_is_rejected(self):
        target = self.vae/'diffusion_pytorch_model.safetensors'
        target.write_bytes(b'x'*target.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'differs from the official manifest'):
            worker._vae_component(self.config, verify_hash=True)

    def test_manifest_cannot_escape_component_root(self):
        outside = self.base/'outside.safetensors'
        outside.write_bytes(b'outside')
        self.config['_manifest_catalog']['completed_files'] = [{
            'path':'vae/../outside.safetensors', 'bytes':7,
            'local_sha256':hashlib.sha256(b'outside').hexdigest()}]
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            worker._vae_component(self.config, verify_hash=True)


if __name__ == '__main__':
    unittest.main()
