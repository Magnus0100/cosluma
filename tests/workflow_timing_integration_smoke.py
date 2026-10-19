#!/usr/bin/env python3
"""Exercise the real selection chain with CPU fixtures and isolated workers mocked."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import single_source_workflow as workflow
import workflow_timing as timing


def main():
    with tempfile.TemporaryDirectory(prefix='layered-') as temporary:
        root = Path(temporary)
        source = root / 'source.jpg'
        candidate = root / 'candidate.png'
        source.write_bytes(b'original source fixture')
        candidate.write_bytes(b'candidate fixture')
        source_facts = {'path': str(source), 'sha256': workflow.sha256(source)}
        parent = {'kind': 'single_input_run', 'run_id': 'parent', 'sample': 'fixture', 'revision': 1,
            'source': source_facts, 'reference': None, 'reference_role': 'candidate_pending_selection',
            'options': {'render_mode': 'native_source_design', 'provider': 'fixture'},
            'task_intent': {'mode': 'existing_reference'},
            'task_plan': {'execution_strategy': 'layered_reference_transfer'},
            'workflow_timing': {'request_id': 'historical', 'path': 'historical.json'},
            'variants': [{'variant_id': 'v1', 'candidate_path': str(candidate), 'prompt': 'fixture'}]}
        workflow.write_json(root / 'outputs/runs/parent/manifest.json', parent)
        workflow.write_json(root / 'project.json', {'state': {}})
        order = []
        def beauty(*args, **kwargs):
            order.append('beauty')
            return {'ok': True, 'no_change': True, 'cache_hit': False,
                    'generation_inferences_this_attempt': 1}
        def native(_root, run, *, cancel=None):
            order.append('native')
            return {'cache_hit': False, 'design_reconstruction_plan': {'enabled': True, 'layers': [],
                'source_sha256': run['source']['sha256'], 'candidate_sha256': run['reference']['sha256']}}
        def boundary(_root, run, *args, **kwargs):
            order.append('boundary')
            return {'ok': True, 'no_change': True, 'cache_hit': True,
                'design_reconstruction_plan': run['options']['appearance_revision']['design_reconstruction_plan'],
                'receipt': {'model_tile_inferences_this_attempt': 0}}
        @timing.timed('render_candidate', model_request=False, http_request=False)
        def render(_root, run, args):
            workflow._refresh_design_plan(_root, run)
            with timing.span('native_render', model_request=False, http_request=False):
                order.append('render')
            with timing.span('render_export', model_request=False, http_request=False):
                output = root / 'outputs/runs' / run['run_id'] / 'fixture-render'
                output.mkdir(parents=True)
                draft = output / 'draft.png'
                draft.write_bytes(b'mocked rendered bytes')
            return {'outputs': {'auto_draft': str(draft)}, 'metadata': {'fixture': True}, 'review': {}}
        modules = {
            'start_qwen_image': SimpleNamespace(release_for_run=lambda *a, **k: {'state': 'not_managed_qwen'}),
            'beautygrpo_prepare': SimpleNamespace(prepare_for_workbench=beauty,
                prepare_cached_for_workbench=lambda *a, **k: None,
                beauty_memory_needed=lambda options, **kwargs: {'physical_bytes': 0, 'commit_bytes': 0},
                release_idle_for_stage=lambda **k: {}),
            'portrait_beauty_layer': SimpleNamespace(materialize_for_run=lambda root, run, value: value),
            'design_reconstruction_pipeline': SimpleNamespace(ready=lambda *a, **k: True),
            'reference_pipeline': SimpleNamespace(prepare=native),
            'runtime_memory': SimpleNamespace(native_analysis_cpu_bytes=lambda run: 0),
            'design_reconstruction': SimpleNamespace(materialize_plan=lambda root, run, plan: copy.deepcopy(plan),
                render_mode_for_run=lambda run: 'native_source_design', is_native_run=lambda run: True,
                has_native_foreground=lambda plan: True),
            'scene_design': SimpleNamespace(bind_plan_receiver_intent=lambda run, plan: plan),
            'design_scene_assets': SimpleNamespace(prepare_native_receivers=lambda root, run, plan: plan),
            'boundary_light_coordination_worker': SimpleNamespace(prepare_for_workbench=boundary, KINDS=()),
        }
        with patch.dict(sys.modules, modules), patch.object(workflow, 'render_candidate', render):
            selected = workflow.select_source_variant(root, 'parent', 'v1')
        assert order == ['beauty', 'native', 'boundary', 'render'], order
        report = json.loads(Path(selected['workflow_timing']['path']).read_text(encoding='utf-8'))
        assert report['status'] == 'complete' and abs(report['reconciliation']['difference_seconds']) < 1e-8
        assert selected['workflow_timing']['request_id'] != 'historical'
        saved = workflow.read_json(root / selected['manifest'])
        assert saved['workflow_timing'] == selected['workflow_timing']
        assert workflow.read_json(root / 'outputs/runs/parent/manifest.json')['workflow_timing']['request_id'] == 'historical'
        names = [row['name'] for row in report['spans']]
        assert names.index('portrait_beauty') < names.index('native_preparation') < names.index('boundary_light') < names.index('render_candidate')
        assert 'render_export' in names and 'save_json' in names and 'save_project_state' in names
        cache = [row for row in report['spans'] if row['name'] == 'native_preparation' and row['metadata']['cache_hit']]
        assert len(cache) == 1 and cache[0]['metadata']['execution_seconds'] == 0
        assert cache[0]['metadata']['validation_seconds'] >= 0

        # Failure after the new manifest exists must retain both it and its timing.
        original = RuntimeError('synthetic native failure')
        modules['reference_pipeline'] = SimpleNamespace(prepare=lambda *a, **k: (_ for _ in ()).throw(original))
        try:
            with patch.dict(sys.modules, modules), patch.object(workflow, 'render_candidate', render):
                workflow.select_source_variant(root, 'parent', 'v1')
        except RuntimeError as error:
            assert error is original
        else:
            raise AssertionError('Selection swallowed native failure')
        reports = [json.loads(path.read_text(encoding='utf-8')) for path in (root / 'outputs/timings').glob('*.json')]
        failures = [value for value in reports if value['operation'] == 'select_variant' and value['status'] == 'failed']
        assert len(failures) == 1 and failures[0]['request_id'] != report['request_id']
        assert failures[0]['manifest_paths'] and Path(failures[0]['manifest_paths'][0]).is_file()
        assert next(row for row in failures[0]['spans'] if row['name'] == 'native_preparation')['error']['message'] == str(original)
        assert timing.current() is None
    print(json.dumps({'ok': True, 'layered_stage_order': True, 'historical_parent_unchanged': True,
        'manifest_timing_pointer': True, 'cached_validation_separate': True,
        'failed_selection_persisted': True, 'no_model_or_http_execution': True}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
