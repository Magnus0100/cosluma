"""Existing-reference direction and ROI budgets; retired creation is rejected."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import creative_direction as direction
import creative_task as task
from creative_task_smoke import fixture, snapshot
from task_model_preflight_smoke import mocked_task_capability


def legacy(mode='create_reference'):
    return {'schema_version': task.INTENT_SCHEMA, 'mode': mode, 'brief': '保留身份妆造姿态',
        'constraints': {'preserve_background': mode == 'local_portrait', 'no_effects': False},
        'services': {'text_vision': False, 'visual_review': False}}


def enhanced(mode='create_reference', *, regional=False):
    intent = legacy(mode)
    intent['constraints'].update(preserve_background=False, background_mode='enhance_source')
    intent['brief'] = '适当增强原景和主光，不完整换景；主效：两只克制蝴蝶的情感意象。'
    if regional:
        intent['regional_refinement'] = {'maximum_submissions': 1,
            'regions': [{'id': 'bg-1', 'core_box': [1, 2, 12, 16]}]}
    return intent


class DirectionTests(unittest.TestCase):
    def test_existing_reference_has_no_implicit_generation(self):
        old = legacy('existing_reference')
        self.assertEqual(task.normalize_task_intent(old), old)
        self.assertEqual(task.image_budget(old), {
            'initial_reference': 0, 'reference_correction': 0, 'independent_effect': 0})
        self.assertNotIn('background_mode', task.normalize_task_intent(old)['constraints'])

    def test_retired_creation_direction_and_native_region_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Retired creation route'):
            task.normalize_task_intent(enhanced('create_reference', regional=True))
        with tempfile.TemporaryDirectory(prefix='retired-direction-') as folder:
            root = Path(folder)
            registry, sources, _ = fixture(root)
            body = {'source': sources[0], 'model_registry': str(registry),
                'creative_task': enhanced('create_reference')}
            before = snapshot(root)
            with patch('urllib.request.urlopen', side_effect=AssertionError('Rejected plan requested network')):
                for key in ('creative_direction', 'regional_resume_authorization',
                            'regional_candidate_reuse_authorization'):
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Retired creation'):
                        task.plan_task(root, {**body, key: {}})
            self.assertEqual(snapshot(root), before)

    def test_explicit_regional_budget_and_geometry(self):
        current = task.normalize_task_intent(enhanced('existing_reference', regional=True))
        self.assertEqual(current['regional_refinement']['coordinate_space'], 'source_canvas')
        self.assertEqual(task.image_budget(current)['regional_refinement'], 1)
        for key, value in [('maximum_submissions', True), ('maximum_submissions', 2),
                           ('coordinate_space', 'reference_scene'), ('regions', [])]:
            changed = enhanced('existing_reference', regional=True)
            changed['regional_refinement'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                task.normalize_task_intent(changed)
        for box in ([0, 0, 0, 1], [0, 0, 2, True], [-1, 0, 2, 3], [0, 0, 2.5, 3]):
            changed = enhanced('existing_reference', regional=True)
            changed['regional_refinement']['regions'][0]['core_box'] = box
            with self.subTest(box=box), self.assertRaises(ValueError):
                task.normalize_task_intent(changed)

    def test_no_route_or_background_permission_escalation(self):
        for mode in ('local_portrait',):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                task.normalize_task_intent(enhanced(mode, regional=True))
        changed = enhanced('existing_reference', regional=True)
        changed['constraints']['preserve_background'] = True
        with self.assertRaises(ValueError):
            task.normalize_task_intent(changed)
        changed = enhanced('existing_reference', regional=True)
        changed['constraints'].pop('background_mode')
        with self.assertRaises(ValueError):
            task.normalize_task_intent(changed)
        changed = enhanced('existing_reference', regional=True)
        changed['regional_refinement']['regions'][0]['instruction'] = '增强原有后景灯光，不改变建筑。'
        self.assertIn('instruction', task.normalize_task_intent(changed)['regional_refinement']['regions'][0])
        existing = enhanced('existing_reference')
        existing['regional_refinement'] = {'maximum_submissions': 0, 'regions': []}
        self.assertEqual(task.image_budget(existing)['regional_refinement'], 0)
        explicit = enhanced('existing_reference', regional=True)
        self.assertEqual(task.image_budget(explicit), {'initial_reference': 0,
            'reference_correction': 0, 'independent_effect': 0, 'regional_refinement': 1})
        local = legacy('local_portrait')
        local['regional_refinement'] = {'maximum_submissions': 0, 'regions': []}
        self.assertEqual(sum(task.image_budget(local).values()), 0)

    def test_direction_is_user_grounded_and_no_effects_wins(self):
        intent = task.normalize_task_intent(enhanced('existing_reference'))
        result = direction.compile_direction(intent, source_sha256='a'*64,
            character_name='祝小红', character_notes='情感意象，不是角色施法。', input_prompt='User prose')
        self.assertEqual(result['origin'], 'offline_user_contract')
        self.assertFalse(result['character']['research_performed'])
        self.assertIn('两只克制蝴蝶', result['main_effect']['description'])
        self.assertIn('祝小红', result['prompt'])
        self.assertEqual(result, direction.validate_direction(result, intent, source_sha256='a'*64))
        intent['constraints']['no_effects'] = True
        result = direction.compile_direction(intent, source_sha256='a'*64)
        self.assertEqual(result['main_effect']['mode'], 'none')
        intent = enhanced('existing_reference')
        intent['brief'] += '不要添加特效。'
        self.assertTrue(task.normalize_task_intent(intent)['constraints']['no_effects'])
        self.assertTrue(direction.has_no_effects_veto('Do not add any visual effects.'))
        result = direction.compile_direction(task.normalize_task_intent(enhanced('existing_reference')),
            source_sha256='a'*64, input_prompt='Do not add any visual effects.')
        self.assertEqual(result['main_effect']['mode'], 'none')
        self.assertEqual(result, direction.validate_direction(result,
            task.normalize_task_intent(enhanced('existing_reference')), source_sha256='a'*64))
        result = direction.compile_direction(task.normalize_task_intent(enhanced('existing_reference')),
            source_sha256='a'*64, input_prompt='Keep scene' + direction.MARKER + '不要添加特效。')
        self.assertEqual(result['main_effect']['mode'], 'user_described')
        intent = task.normalize_task_intent(enhanced('existing_reference'))
        intent['brief'] = '只调整主光与环境层次'
        result = direction.compile_direction(intent, source_sha256='b'*64)
        self.assertEqual(result['main_effect']['mode'], 'not_specified')
        self.assertNotIn('祝小红', result['prompt'])
        tampered = copy.deepcopy(result)
        tampered['scene'] = 'replace scene'
        with self.assertRaises(ValueError):
            direction.validate_direction(tampered, intent, source_sha256='b'*64)

    def test_checked_plan_roundtrip_and_mutation_rejection(self):
        with tempfile.TemporaryDirectory(prefix='creative-direction-') as folder:
            root = Path(folder)
            registry, sources, reference = fixture(root)
            body = {'source': sources[0], 'model_registry': str(registry),
                'provider':'qwen_image_2_1_uncensored_fast',
                'character_name': '祝小红', 'character_notes': '不发明角色能力',
                'input_prompt': 'Retain the photograph.', 'reference': reference,
                'creative_task': enhanced('existing_reference', regional=True)}
            with patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability), \
                    patch('reference_provider.provider_health', return_value={'ready': True}), \
                    patch('urllib.request.urlopen', side_effect=AssertionError('Pure planning requested network')):
                before = snapshot(root)
                checked = task.plan_task(root, body)
                self.assertTrue(checked['plan']['ready'])
                self.assertEqual(snapshot(root), before)
                self.assertEqual(checked['plan']['generation_submissions_upper_bound'], 1)
                self.assertEqual(checked['plan']['image_budget'], {'initial_reference': 0,
                    'reference_correction': 0, 'independent_effect': 0, 'regional_refinement': 1})
                self.assertEqual(checked['request']['input_prompt'], body['input_prompt'])
                self.assertEqual(checked['request']['effective_prompt'], checked['plan']['creative_direction']['prompt'])
                self.assertEqual(checked['plan']['plan_id'], task.plan_task(root, checked['request'])['plan']['plan_id'])
                for extra in ({'variant_count': 2}, {'variant_styles': ['night_scene']},
                              {'defer_reference_selection': True}):
                    with self.subTest(extra=extra), self.assertRaises(ValueError):
                        task.plan_task(root, {**body, **extra})
                outside=copy.deepcopy(body)
                outside['creative_task']['regional_refinement']['regions'][0]['core_box']=[0,0,26,20]
                with self.assertRaises(ValueError): task.plan_task(root,outside)
                execute = task.execution_request(root, {**checked['request'], 'plan_id': checked['plan']['plan_id']})
                self.assertEqual(execute['cosplay_correction_budget'], 0)
                task.validate_task_plan(execute['task_plan'], execute['task_intent'])
                for mutate in (
                        lambda row: row.update(character_name='Other character'),
                        lambda row: row['creative_task']['regional_refinement']['regions'][0].update(core_box=[2, 3, 13, 17]),
                        lambda row: row['creative_direction'].update(light='unbound light'),
                        lambda row: row['creative_task']['constraints'].update(background_mode='replace')):
                    changed = copy.deepcopy(checked['request'])
                    changed['plan_id'] = checked['plan']['plan_id']
                    mutate(changed)
                    with self.assertRaises(ValueError):
                        task.execution_request(root, changed)
                altered_plan = copy.deepcopy(checked['plan'])
                altered_plan['image_budget']['regional_refinement'] = 2
                with self.assertRaises(ValueError):
                    task.validate_task_plan(altered_plan, checked['plan']['intent'])
                altered_plan['image_budget']['regional_refinement'] = True
                with self.assertRaises(ValueError):
                    task.validate_task_plan(altered_plan, checked['plan']['intent'])
                old_graph = copy.deepcopy(checked['plan'])
                old_graph['layer_workflow']['graph_id'] = 'obsolete'
                with self.assertRaises(ValueError):
                    task.saved_task_intent_for_replanning(old_graph, checked['plan']['intent'],
                        source_sha256=checked['plan']['inputs']['source']['sha256'])
                for change in ('source', 'budget'):
                    old = copy.deepcopy(old_graph)
                    if change == 'budget':
                        old['image_budget']['regional_refinement'] = 0
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        task.saved_task_intent_for_replanning(old, checked['plan']['intent'],
                            source_sha256='f'*64 if change == 'source' else checked['plan']['inputs']['source']['sha256'])
                existing = enhanced('existing_reference')
                existing['regional_refinement'] = {'maximum_submissions': 0, 'regions': []}
                repair = task.plan_task(root, {**body, 'creative_task': existing, 'reference': reference,
                    'final_repair_only': True})
                self.assertEqual(sum(repair['plan']['image_budget'].values()), 0)
                self.assertEqual(repair['plan']['preparation_policy'], 'replay_frozen')
                with self.assertRaises(ValueError):
                    task.plan_batch(root, {key: value for key, value in
                        {**body, 'sources': sources}.items() if key != 'source'})


if __name__ == '__main__':
    unittest.main()
