"""Face-shaping installation checks read bounded bytes; no runtimes or models.

Tiny model/Python bytes are deliberately not executable. Real inference and
geometric acceptance are outside this preflight test's evidence.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import creative_task
from creative_task_smoke import snapshot
from current_model_registry import SCHEMA
from photo_editing import editing_scope, identity
import task_model_preflight as preflight

ACTUAL_INSPECT = preflight.inspect_task_models
FACE = {'action':'apply', 'parts':{'face':.3}}
BODY = {'action':'apply', 'parts':{'waist':.2, 'arms':.2, 'legs':.2}}
RESTORE = {'action':'restore'}


def make_face_registry(root):
    python = root/'face-runtime/Scripts/python.exe'; python.parent.mkdir(parents=True)
    python.write_bytes(b'Offline fake executable bytes; never launched.\n')
    model = root/'face_model.task'; model.write_bytes(b'Offline bounded face model bytes; never loaded.\n')
    metadata = python.parent.parent/'Lib/site-packages/mediapipe-0.10.32.dist-info/METADATA'
    metadata.parent.mkdir(parents=True)
    metadata.write_text('Metadata-Version: 2.1\nName: mediapipe\nVersion: 0.10.32\n', encoding='utf8')
    config = {'schema':SCHEMA, 'ready':True, 'face_python':str(python), 'face_model_path':str(model)}
    registry = root/'face-registry.json'; registry.write_text(json.dumps(config), encoding='utf8')
    return registry, config, metadata


class FaceShapingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='face-shaping-preflight-'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.registry, self.config, self.metadata = make_face_registry(self.root)
        self.scope = editing_scope()
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(subprocess, 'Popen', side_effect=AssertionError('Preflight launched a runtime')))
        self.stack.enter_context(patch.object(socket, 'create_connection', side_effect=AssertionError('Preflight sent network')))

    def inspect(self, request=FACE, config=None):
        return ACTUAL_INSPECT(self.config if config is None else config, mode='existing_reference',
            editing_scope=self.scope, shaping_request=request)

    def test_only_explicit_face_apply_adds_portrait_regions(self):
        requests = ((FACE, ['portrait_regions']),
            ({'action':'apply', 'parts':{'face':.3, 'waist':.2}}, ['portrait_regions']),
            (BODY, []), (RESTORE, []), (None, []))
        for request, expected in requests:
            with self.subTest(request=request):
                checked = self.inspect(request)
                self.assertTrue(checked['ready'], checked['blocking_reasons'])
                self.assertEqual(checked['required_capabilities'], expected)
                self.assertEqual(checked['required_model_groups'], ['face_localization'] if expected else [])
                self.assertEqual(checked['required_path_keys'], ['face_model_path', 'face_python'] if expected else [])
                self.assertEqual(checked['model_inferences'], 0); self.assertEqual(checked['network_requests'], 0)
                self.assertFalse(checked['inference_verified'])
        with self.assertRaises(ValueError):
            ACTUAL_INSPECT(self.config, mode='create_reference', editing_scope=self.scope, shaping_request=FACE)

    def test_selected_face_model_python_and_metadata_have_actual_content_digests(self):
        before = snapshot(self.root); modules = set(sys.modules); checked = self.inspect()
        rows = {row['key']:row for row in checked['checks']}
        for key, path in (('face_model_path', Path(self.config['face_model_path'])),
                          ('face_python', Path(self.config['face_python']))):
            row = rows[key+'.observed_identity']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertTrue(row['available']); self.assertEqual(row['path'], str(path))
            self.assertEqual(row['sha256'], digest); self.assertEqual(row['actual_sha256'], digest)
            self.assertEqual(row['bytes'], path.stat().st_size)
        row = rows['face_runtime.mediapipe']; self.assertTrue(row['available'])
        self.assertEqual(row['path'], str(self.metadata)); self.assertEqual(row['version'], '0.10.32')
        self.assertEqual(row['metadata_sha256'], hashlib.sha256(self.metadata.read_bytes()).hexdigest())
        self.assertEqual(row['actual_sha256'], row['metadata_sha256'])
        self.assertEqual(checked['hash_bytes_read'], sum(Path(row['path']).stat().st_size for row in
            (rows['face_model_path.observed_identity'], rows['face_python.observed_identity'], rows['face_runtime.mediapipe'])))
        self.assertTrue(checked['small_file_identity_verified'])
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(any(name=='mediapipe' or name.startswith('mediapipe.') for name in set(sys.modules)-modules))

    def test_tampered_model_python_and_metadata_change_checked_identity(self):
        for path, key, field in ((Path(self.config['face_model_path']), 'face_model_path.observed_identity', 'actual_sha256'),
                (Path(self.config['face_python']), 'face_python.observed_identity', 'actual_sha256'),
                (self.metadata, 'face_runtime.mediapipe', 'metadata_sha256')):
            with self.subTest(key=key):
                original = path.read_bytes(); before = {row['key']:row for row in self.inspect()['checks']}
                path.write_bytes(original+b'\nChanged bounded content.\n')
                after = {row['key']:row for row in self.inspect()['checks']}
                self.assertNotEqual(before[key][field], after[key][field])
                path.write_bytes(original)

    def test_missing_disabled_malformed_and_oversized_face_installations_block(self):
        for config in ({}, {**self.config, 'ready':False}, {**self.config, 'schema':'retired'},
                {**self.config, 'face_python':str(self.root/'missing-python.exe')},
                {**self.config, 'face_model_path':str(self.root/'missing-model.task')},
                {**self.config, 'face_model_path':True}):
            with self.subTest(config=config):
                checked = self.inspect(config=config)
                self.assertFalse(checked['ready']); self.assertTrue(checked['blocking_reasons'])
        original = self.metadata.read_bytes()
        for text in ('Name: mediapipe\n', 'Name: torch\nVersion: 0.10.32\n',
                     'Name: mediapipe\nVersion: 1\nVersion: 2\n'):
            self.metadata.write_text(text, encoding='utf8'); self.assertFalse(self.inspect()['ready'])
        self.metadata.write_bytes(original)
        with patch.object(preflight, 'SMALL_FILE_LIMIT', 4):
            checked = self.inspect(); self.assertFalse(checked['ready'])
            self.assertEqual(checked['hash_bytes_read'], 0)
        for request in (BODY, RESTORE, None):
            self.assertTrue(self.inspect(request, config={})['ready'])

    def test_public_fixture_mock_cannot_skip_actual_face_identity_reads(self):
        from task_model_preflight_smoke import mocked_task_capability
        with patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability):
            checked = preflight.inspect_task_models(self.config, mode='existing_reference',
                editing_scope=self.scope, shaping_request=FACE)
            self.assertTrue(checked['ready']); self.assertGreater(checked['hash_bytes_read'], 0)
            self.assertNotEqual(checked.get('mocked'), True)
            self.metadata.unlink()
            blocked = preflight.inspect_task_models(self.config, mode='existing_reference',
                editing_scope=self.scope, shaping_request=FACE)
            self.assertFalse(blocked['ready'])

    def test_public_plans_discover_face_registry_and_bind_content_without_generation(self):
        import scoped_production_smoke as fixtures
        fx = fixtures.ScopedProductionTests(methodName='runTest'); fx.setUp(); self.addCleanup(fx.doCleanups)
        _, parent = fx.start(fx.request(fx.sources[0]))
        from photo_appearance import validate_asset
        u = parent['options']['photo_appearance']; receipt = validate_asset(u,
            source_sha256=parent['source']['sha256'], scope=parent['task_plan']['editing_scope'],
            design_id=parent['generation_design']['observed_design']['design_id'])
        body = fx.request(fx.sources[0]); body.pop('editing_autofinish')
        body['creative_task'].update(mode='existing_reference', services={'text_vision':True, 'visual_review':True})
        body.update(parent_run_id=parent['run_id'], reference=str(Path(u['path'])/'preview.png'),
            model_registry=str(self.registry), shaping_request=copy.deepcopy(FACE))
        counts = (len(fx.image_requests), len(fx.vision_requests))
        import reference_provider
        with patch('task_model_preflight.inspect_task_models', side_effect=ACTUAL_INSPECT), \
             patch('reference_provider.provider_health', wraps=reference_provider.provider_health) as image_health, \
             patch('runtime_preflight._probe_runtime', side_effect=AssertionError('Unused native runtime probe')):
            before = snapshot(fx.root)
            checked = creative_task.plan_task(fx.root, body); plan = checked['plan']
            self.assertTrue(plan['ready'], plan['blocking_reasons'])
            self.assertEqual(sum(plan['image_budget'].values()),0)
            self.assertFalse(plan['services']['image_generation'])
            self.assertTrue(plan['capabilities']['native_models_required'])
            self.assertEqual(plan['capabilities']['model_preflight']['required_capabilities'], ['portrait_regions'])
            self.assertEqual({row['key'] for row in plan['capabilities']['models']}, {'face_python', 'face_model_path'})
            creative_task.validate_task_plan(plan, plan['intent'])
            self.assertEqual(snapshot(fx.root), before)
            for path in (Path(self.config['face_model_path']), Path(self.config['face_python']), self.metadata):
                original = path.read_bytes(); path.write_bytes(original+b'\nIdentity changed after checking.\n')
                replanned = creative_task.plan_task(fx.root, body)['plan']
                self.assertNotEqual(plan['plan_id'], replanned['plan_id'])
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    creative_task.execution_request(fx.root, {**checked['request'], 'plan_id':plan['plan_id']})
                path.write_bytes(original)
            invalid = self.root/'malformed.json'; invalid.write_text('{broken', encoding='utf8')
            disabled = self.root/'disabled.json'; disabled.write_text(json.dumps({'schema':SCHEMA,'ready':False}), encoding='utf8')
            for registry in (self.root/'missing.json', invalid, disabled):
                blocked = creative_task.plan_task(fx.root, {**body, 'model_registry':str(registry)})['plan']
                self.assertFalse(blocked['ready'])
            # Non-face requests keep the generated-native isolation contract.
            self.assertEqual(image_health.call_count,0,'CPU face planning checked unused image readiness')
            with patch('current_model_registry.registry_status', side_effect=AssertionError('Unused native discovery')):
                body_only = creative_task.plan_task(fx.root, {**body, 'shaping_request':BODY, 'model_registry':str(disabled)})['plan']
                self.assertTrue(body_only['ready'], body_only['blocking_reasons'])
                self.assertEqual(body_only['capabilities']['model_preflight']['required_capabilities'], [])
                ordinary = fx.request(fx.sources[0], model_registry=str(disabled))
                ordinary_plan = creative_task.plan_task(fx.root, ordinary)['plan']
                self.assertTrue(ordinary_plan['ready'], ordinary_plan['blocking_reasons'])
                self.assertFalse(ordinary_plan['capabilities']['native_models_required'])
            # Stub only frozen-parent authorization for this isolated restore
            # preflight test; actual portable restore has a separate fixture.
            restore_auth = copy.deepcopy(plan['shaping_authorization'])
            restore_auth.update(request=RESTORE, manual_provider_target=None, manual_target=None)
            restore_auth['authorization_id'] = identity({k:v for k,v in restore_auth.items() if k!='authorization_id'})
            with patch('photo_shaping.frozen_restore_source', return_value=copy.deepcopy(plan['inputs']['source'])), \
                 patch('photo_shaping.authorize', return_value=restore_auth), \
                 patch('current_model_registry.registry_status', side_effect=AssertionError('Restore discovered native config')), \
                 patch('task_model_preflight.generated_registry_status', side_effect=AssertionError('Restore inspected native metadata')), \
                 patch('reference_provider.provider_health', side_effect=AssertionError('Restore resolved provider')), \
                 patch('photo_request_budget.manual_provider_target', side_effect=AssertionError('Restore resolved credentials')):
                restore = copy.deepcopy(body); restore['shaping_request'] = RESTORE
                restore['creative_task']['services'] = {'text_vision':False, 'visual_review':False}
                restore['model_registry'] = str(self.root/'unavailable-registry.json')
                restored_plan = creative_task.plan_task(fx.root, restore)['plan']
                self.assertTrue(restored_plan['ready'], restored_plan['blocking_reasons'])
                self.assertEqual(restored_plan['capabilities']['model_preflight']['checks'], [])
                self.assertEqual(restored_plan['services']['text_vision_requests_upper_bound'], 0)
                self.assertEqual(sum(restored_plan['image_budget'].values()), 0)
            self.assertEqual((len(fx.image_requests), len(fx.vision_requests)), counts)


if __name__ == '__main__': unittest.main()
