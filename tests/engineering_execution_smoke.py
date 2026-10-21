"""Real runtime closure snapshots, isolated imports and atomic failures; no models."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import engineering_execution as execution


class ExecutionSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = Path(__file__).resolve().parents[1]
        cls.temporary = tempfile.TemporaryDirectory(prefix='execution-runtime-closure-')
        cls.fixture = Path(cls.temporary.name)/'real-closure'; cls.fixture.mkdir()
        for path in execution.collect_files(cls.original):
            destination = cls.fixture/path.relative_to(cls.original)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='execution-snapshot-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'mutable checkout'; shutil.copytree(self.fixture, self.source)
        self.snapshot = self.root/'engineering'/'frozen code'
        self.workspace = self.root/'photo project'; self.workspace.mkdir()
        self.request = self.root/'private request.json'; self.request.write_text('{}', encoding='utf-8')

    def create(self):
        return execution.freeze(self.source, self.snapshot)

    def spec(self, frozen, action='task-plan'):
        return execution.command(self.snapshot, expected_id=frozen['snapshot_id'], python=sys.executable,
            workspace=self.workspace, action=action, request_file=self.request)

    def environment(self, spec):
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(self.source/'scripts'), COSPLAY_PACKAGE_ROOT=str(self.source))
        for key in spec['environment']['remove']:
            environment.pop(key, None)
        environment.update(spec['environment']['set'])
        environment.update(COSPLAY_CONFIG_ROOT=str(self.root/'private-config'),
            COSPLAY_MODEL_REGISTRY=str(self.root/'missing-external-registry.json'),
            COSPLAY_CACHE_ROOT=str(self.root/'metadata-cache'),
            COSPLAY_ASSET_IDENTITY_CACHE=str(self.root/'identity-cache'),
            COSPLAY_SOURCE_METADATA_CACHE=str(self.root/'source-metadata-cache'))
        return environment

    def test_real_inventory_bytes_and_private_data_excluded(self):
        private = {'photos/original.png': b'private photograph', 'model/weights.safetensors': b'private weights',
            '.env.local': b'API_KEY=private-fixture', 'config/provider-settings.json': b'private credential config',
            'providers/local-models.current.json': b'private registry', 'providers/qwen-image-2.1-models.json': b'private catalog',
            'providers/workflows/secret-local.json': b'private file next to public workflows',
            'scripts/unregistered_private.py': b'PRIVATE_TOKEN = "never copied"',
            'outputs/runs/old/manifest.json': b'private historical receipt'}
        for name, data in private.items():
            path = self.source/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        frozen = self.create()
        manifest = json.loads((self.snapshot/execution.MANIFEST).read_text(encoding='utf-8'))
        paths = {row['path'] for row in manifest['files']}
        runtime = json.loads((self.source/'scripts/runtime_manifest.json').read_text(encoding='utf-8'))['files']
        self.assertTrue({'scripts/'+name for name in runtime}.issubset(paths))
        self.assertIn('providers/workflows/qwen-image-2.1-effect-rgba.json', paths)
        self.assertGreater(len(paths), 190)
        for name in private:
            self.assertNotIn(name, paths)
            self.assertFalse((self.snapshot/name).exists())
        for row in manifest['files']:
            self.assertEqual((self.snapshot/row['path']).read_bytes(), (self.source/row['path']).read_bytes())
        self.assertEqual(frozen['manifest_sha256'], hashlib.sha256((self.snapshot/execution.MANIFEST).read_bytes()).hexdigest())
        self.assertEqual(execution.verify(self.snapshot, expected_id=frozen['snapshot_id'])['snapshot_id'], frozen['snapshot_id'])

    def test_parent_edits_do_not_change_late_imports_resources_or_graph_producers(self):
        frozen = self.create(); spec = self.spec(frozen)
        code = '''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'scripts'))
from runtime_resources import code_path
from installed_resources import resource_path
import photo_editing
from cos_node_registry import compile_workflow
from scoped_workflow import SOURCE_NATIVE_INITIALIZATION
graph=compile_workflow('create_reference',editing_scope=photo_editing.editing_scope(),visual_review=True,
 editing_initialization=SOURCE_NATIVE_INITIALIZATION)
print(json.dumps({'late_module':photo_editing.__file__,'producer':graph['producer'],
 'graph_id':graph['graph_id'],'code':str(code_path('scoped_workflow.py')),
 'resource':str(resource_path('providers/registry.json')),
 'registry_sha':hashlib.sha256(resource_path('providers/registry.json').read_bytes()).hexdigest()}))'''
        argv = [sys.executable, '-I', '-B', '-X', 'utf8', '-c', code, str(self.snapshot)]
        def probe():
            result = subprocess.run(argv, cwd=spec['cwd'], env=self.environment(spec), capture_output=True,
                text=True, encoding='utf-8', timeout=20, check=True)
            return json.loads(result.stdout)
        before = probe()
        parent_module = self.source/'scripts/photo_editing.py'
        parent_module.write_text(parent_module.read_text(encoding='utf-8')+'\nraise RuntimeError("mutable parent imported")\n', encoding='utf-8')
        graph = self.source/'scripts/scoped_workflow.py'; graph.write_bytes(graph.read_bytes()+b'\n# changed live graph\n')
        registry = self.source/'providers/registry.json'; registry.write_bytes(registry.read_bytes()+b'\n')
        after = probe()
        self.assertEqual(before, after)
        self.assertTrue(Path(after['late_module']).is_relative_to(self.snapshot))
        self.assertTrue(Path(after['resource']).is_relative_to(self.snapshot))
        self.assertTrue(Path(after['code']).is_relative_to(self.snapshot))
        self.assertEqual(self.spec(frozen, 'task-start')['snapshot_id'], spec['snapshot_id'])
        execution.verify(self.snapshot, expected_id=frozen['snapshot_id'])

    def test_commands_are_exact_existing_entrypoints_and_plan_uses_snapshot(self):
        frozen = self.create(); spec = self.spec(frozen)
        self.assertFalse(spec['shell']); self.assertFalse(spec['automatic_retry'])
        self.assertEqual(spec['argv'][5], str(self.snapshot/'scripts/project_cli.py'))
        self.assertEqual(spec['argv'][-3:], ['task-plan', '--request-file', str(self.request)])
        self.assertIn('COSPLAY_PACKAGE_ROOT', spec['environment']['remove'])
        self.assertNotIn('SCOPED_FIXTURE_KEY', json.dumps(spec))
        image = self.workspace/'source.png'; Image.new('RGB', (16, 24), 'gray').save(image)
        self.request.write_text(json.dumps({'source': str(image), 'model_registry': str(self.root/'missing-external-registry.json'),
            'creative_task': {'schema_version': 'cosplay_task_intent.v1', 'mode': 'local_portrait',
                'brief': 'CPU fixture only', 'constraints': {'preserve_background': True, 'no_effects': True},
                'services': {'text_vision': False, 'visual_review': False}}}), encoding='utf-8')
        result = subprocess.run(spec['argv'], cwd=spec['cwd'], env=self.environment(spec),
            capture_output=True, text=True, encoding='utf-8', timeout=25)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)['plan']
        self.assertFalse(plan['ready'])  # Deliberately absent models never become ready.
        self.assertEqual(plan['preflight_network_requests'], 0)
        self.assertEqual(plan['preflight_model_inferences'], 0)
        execution.verify(self.snapshot, expected_id=frozen['snapshot_id'])
        with self.assertRaises(ValueError):
            self.spec(frozen, 'one-click')

    def test_copy_failure_is_atomic_and_existing_snapshot_is_never_replaced(self):
        copied = []
        original = execution._copy_file
        def fail(source, destination):
            copied.append(source)
            if len(copied) == 3:
                raise OSError('Synthetic copy interrupted')
            return original(source, destination)
        with patch.object(execution, '_copy_file', side_effect=fail):
            with self.assertRaises(OSError):
                self.create()
        self.assertFalse(self.snapshot.exists())
        self.assertEqual(list(self.snapshot.parent.glob('*.partial')), [])
        frozen = self.create(); original_manifest = (self.snapshot/execution.MANIFEST).read_bytes()
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual((self.snapshot/execution.MANIFEST).read_bytes(), original_manifest)
        self.assertEqual(execution.verify(self.snapshot, expected_id=frozen['snapshot_id'])['snapshot_id'], frozen['snapshot_id'])

    def test_source_change_during_copy_cannot_publish_mixed_code(self):
        original = execution._copy_file; changed = []
        def changing(source, destination):
            original(source, destination)
            if not changed:
                source.write_bytes(source.read_bytes()+b'\n')
                changed.append(source)
        with patch.object(execution, '_copy_file', side_effect=changing):
            with self.assertRaises(ValueError):
                self.create()
        self.assertFalse(self.snapshot.exists())
        self.assertEqual(list(self.snapshot.parent.glob('*.partial')), [])

    def test_changed_snapshot_extra_files_wrong_identity_and_inside_source_output_rejected(self):
        with self.assertRaises(ValueError):
            execution.freeze(self.source, self.source/'snapshot')
        frozen = self.create()
        with self.assertRaises(ValueError):
            execution.verify(self.snapshot, expected_id='0'*64)
        extra = self.snapshot/'credentials.json'; extra.write_text('private injection', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.spec(frozen)
        extra.unlink()
        module = self.snapshot/'scripts/scoped_workflow.py'
        module.chmod(stat.S_IWRITE | stat.S_IREAD); module.write_bytes(module.read_bytes()+b'\n')
        with self.assertRaises(ValueError):
            self.spec(frozen)


if __name__ == '__main__':
    unittest.main()
