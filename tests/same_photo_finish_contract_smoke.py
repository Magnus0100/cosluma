"""Pure opt-in reference ownership; never assert registration from preflight."""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

import creative_direction as direction
import creative_task as task
from creative_direction_smoke import enhanced
from creative_task_smoke import fixture, snapshot
from task_model_preflight_smoke import mocked_task_capability


class SamePhotoFinishTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='spf-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        registry, sources, reference = fixture(self.root)
        self.body = {'source': sources[0], 'reference': reference,
            'model_registry': str(registry), 'input_prompt': 'Match cool light; preserve the original person.',
            'character_name': '用户声明角色', 'character_notes': '保留当前妆造、发型和乐器',
            'creative_task': enhanced('existing_reference')}
        self.body['creative_task']['brief'] = '冷蓝环境、干净脸肤、发丝冷rim，参考薄雾和前景红叶层次。'
        self.body['creative_task']['constraints']['reference_role'] = 'same_photo_finish'
        self.enterContext(patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability))
        self.enterContext(patch('urllib.request.urlopen', side_effect=AssertionError('Planning attempted network')))
        self.provider = self.enterContext(patch('reference_provider.provider_health', side_effect=AssertionError('No generation provider probe')))

    def test_opt_in_descriptor_is_frozen_zero_budget_and_unregistered(self):
        before = snapshot(self.root)
        checked = task.plan_task(self.root, self.body)
        self.assertTrue(checked['plan']['ready'])
        self.assertEqual(before, snapshot(self.root))
        self.provider.assert_not_called()
        descriptor = checked['plan']['reference_contract']
        self.assertEqual(descriptor, {'schema': 'same_photo_finish_reference.v1', 'role': 'same_photo_finish',
            'source_sha256': task.sha(Path(self.body['source'])),
            'reference_sha256': task.sha(Path(self.body['reference'])), 'geometry_owner': 'source',
            'texture_owner': 'source_native', 'transfer_scope': 'tone_and_source_guided_local_colour_coefficients',
            'reference_rgb_compositing_allowed': False, 'registration_status': 'not_checked'})
        self.assertEqual(checked['plan']['image_budget'], {'initial_reference': 0,
            'reference_correction': 0, 'independent_effect': 0})
        self.assertEqual(checked['plan']['generation_submissions_upper_bound'], 0)
        self.assertFalse(checked['request']['generate'])
        self.assertFalse(checked['request']['creative_task']['constraints']['no_effects'])
        self.assertNotIn('reference_contract', checked['request'])
        self.assertIn('薄雾', checked['request']['creative_direction']['prompt'])
        self.assertIn('红叶', checked['request']['creative_direction']['prompt'])
        self.assertIn('不直接合成参考RGB', checked['request']['creative_direction']['prompt'])
        labels = [row['label'] for row in checked['plan']['stages']]
        self.assertTrue(any('CPU影调' in label for label in labels))
        self.assertTrue(any('系数场校正' in label for label in labels))
        from qt_workbench import creative_task_plan_text
        summary = creative_task_plan_text(checked['plan'])
        self.assertIn('尚未实际配准', summary)
        execute = task.execution_request(self.root, {**checked['request'], 'plan_id': checked['plan']['plan_id']})
        task.validate_task_plan(execute['task_plan'], execute['task_intent'])
        self.assertEqual(execute['task_plan']['reference_contract'], descriptor)
        self.assertEqual(checked['plan']['plan_id'], task.plan_task(self.root, checked['request'])['plan']['plan_id'])

    def test_default_dictionary_and_graph_remain_unchanged(self):
        old_body = copy.deepcopy(self.body)
        old_body['creative_task']['constraints'].pop('reference_role')
        intent = task.normalize_task_intent(old_body['creative_task'])
        self.assertNotIn('reference_role', intent['constraints'])
        self.assertEqual(intent, old_body['creative_task'])
        old = task.plan_task(self.root, old_body)
        new = task.plan_task(self.root, self.body)
        self.assertNotIn('reference_contract', old['plan'])
        self.assertNotIn('reference_role', old['plan']['creative_direction'])
        self.assertEqual(old['plan']['layer_workflow'], new['plan']['layer_workflow'])
        self.assertNotEqual(old['plan']['plan_id'], new['plan']['plan_id'])
        self.assertIsNone(direction.reference_contract_for(intent, old['plan']['inputs']))
        bad = copy.deepcopy(old['plan']); bad['reference_contract'] = new['plan']['reference_contract']
        with self.assertRaises(ValueError):
            task.validate_task_plan(bad, intent)

    def test_strict_descriptor_input_binding_and_saved_replanning(self):
        checked = task.plan_task(self.root, self.body)
        for field, value in (('source_sha256', 'f'*64), ('reference_sha256', 'f'*64),
                             ('geometry_owner', 'reference'), ('texture_owner', 'reference'),
                             ('transfer_scope', 'reference_rgb'), ('registration_status', 'registered'),
                             ('reference_rgb_compositing_allowed', True), ('reference_rgb_compositing_allowed', 0),
                             ('role', 'style'), ('extra', 'unsupported')):
            bad = copy.deepcopy(checked['plan']); bad['reference_contract'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                task.validate_task_plan(bad, bad['intent'])
        bad = copy.deepcopy(checked['plan']); bad.pop('reference_contract')
        with self.assertRaises(ValueError):
            task.validate_task_plan(bad, bad['intent'])
        old_graph = copy.deepcopy(checked['plan']); old_graph['layer_workflow']['graph_id'] = 'historical'
        with self.assertRaises(ValueError):
            task.saved_task_intent_for_replanning(old_graph, old_graph['intent'],
                source_sha256=checked['plan']['inputs']['source']['sha256'],
                reference_sha256=checked['plan']['inputs']['reference']['sha256'])
        old_graph['reference_contract']['registration_status'] = 'registered'
        with self.assertRaises(ValueError):
            task.saved_task_intent_for_replanning(old_graph, old_graph['intent'],
                source_sha256=checked['plan']['inputs']['source']['sha256'],
                reference_sha256=checked['plan']['inputs']['reference']['sha256'])

    def test_conflicting_routes_regional_fields_proofs_and_variants_refused(self):
        for mode in ('create_reference', 'local_portrait'):
            body = copy.deepcopy(self.body); body['creative_task']['mode'] = mode
            with self.assertRaises(ValueError): task.plan_task(self.root, body)
        for role in (None, True, 'style_reference', 'registered'):
            body = copy.deepcopy(self.body); body['creative_task']['constraints']['reference_role'] = role
            with self.assertRaises(ValueError): task.plan_task(self.root, body)
        body = copy.deepcopy(self.body); body['creative_task']['constraints'].pop('background_mode')
        with self.assertRaises(ValueError): task.plan_task(self.root, body)
        for regional in (None, {}, {'maximum_submissions': 0, 'regions': []},
                         {'maximum_submissions': 1, 'regions': [{'id':'old','core_box':[1,2,12,16]}]}):
            body = copy.deepcopy(self.body); body['creative_task']['regional_refinement'] = regional
            with self.assertRaises(ValueError): task.plan_task(self.root, body)
        for extra in ({'regional_resume_authorization': None}, {'regional_candidate_reuse_authorization': {}},
                      {'generate': True}, {'variant_count': 2}, {'variant_styles':['night_scene']},
                      {'defer_reference_selection': True}, {'reference_contract': {'fake':True}}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                task.plan_task(self.root, {**self.body, **extra})

    def test_source_target_role_and_brief_changes_reject_the_previous_plan(self):
        checked = task.plan_task(self.root, self.body)
        for mutate in (lambda b:b['creative_task']['constraints'].pop('reference_role'),
                       lambda b:b['creative_task'].update(brief='Changed art direction'),
                       lambda b:b.update(character_name='Other role')):
            body = copy.deepcopy(checked['request']); mutate(body)
            with self.assertRaises(ValueError):
                task.execution_request(self.root, {**body, 'plan_id':checked['plan']['plan_id']})
        for role in ('source', 'reference'):
            path = Path(self.body[role]); original = path.read_bytes()
            Image.new('RGB',(25,20),'#abcdef').save(path)
            with self.assertRaises(ValueError):
                task.execution_request(self.root,{**checked['request'],'plan_id':checked['plan']['plan_id']})
            path.write_bytes(original)
        body = copy.deepcopy(self.body); body['creative_task']['constraints']['no_effects'] = True
        no_effects = task.plan_task(self.root, body)
        self.assertTrue(no_effects['request']['creative_task']['constraints']['no_effects'])
        self.assertEqual(no_effects['request']['creative_direction']['main_effect']['mode'], 'none')


if __name__ == '__main__':
    unittest.main()
