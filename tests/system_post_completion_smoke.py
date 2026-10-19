"""Offline dependency-order, child lineage and one-attempt completion contracts."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import single_source_workflow as workflow
import system_post_completion as system
from generation_design import DESIGN_VERSION


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        project = root/'project.json'; project.write_text('{"selection":"original"}')
        digest = workflow.sha256(project)
        source = root/'source.png'; reference = root/'reference.png'
        Image.new('RGB', (20, 24), '#805050').save(source)
        Image.new('RGB', (20, 24), '#806050').save(reference)
        run = {'kind': 'single_input_run', 'run_id': 'initial', 'revision': 1, 'status': 'needs_visual_review',
            'source': {'path': str(source), 'sha256': workflow.sha256(source)},
            'reference': {'path': str(reference), 'sha256': workflow.sha256(reference)},
            'generation_design': {'schema_version': DESIGN_VERSION, 'status': 'ready'},
            'root': str(root), 'artifacts': {'auto_draft': str(reference)},
            'task_intent': {'schema_version': 'cosplay_task_intent.v1', 'mode': 'create_reference',
                            'services': {'text_vision': True, 'visual_review': True}},
            'options': {}, 'generation_budget': {'image_submissions': 0}}
        calls = []
        def review(current):
            current['cosplay_post_review'] = {'status': 'reviewed', 'actual_api_evidence_valid': True,
                'run_binding': {'run_id': current['run_id'], 'revision': current['revision'],
                    'source_sha256': current['source']['sha256'], 'reference_sha256': current['reference']['sha256'],
                    'final_sha256': workflow.sha256(reference),
                    'render_artifacts': {key:workflow.sha256(Path(value)) for key,value in current['artifacts'].items()}},
                'binding': {'images': {'final': {'path': str(reference), 'sha256': workflow.sha256(reference)}}},
                'diagnostic_gate': 'fail', 'checks': {'skin_materials': {'reference': {'status': 'fail'},
                    'final': {'status': 'pass'}}, 'scene_integration': {'reference': {'status': 'fail'},
                    'final': {'status': 'fail'}}}}
        review(run)
        args = SimpleNamespace(cosplay_correction_budget=1, generate=False, complete_post=True,
                               preserve_effective_prompt=False, cancel_request_path=None)
        disabled = copy.deepcopy(run); disabled['run_id'] = 'review-disabled'
        disabled['task_intent']['services'].pop('visual_review')
        with patch('automatic_material_completion.complete_materials_once') as mat, \
             patch('cosplay_post_completion.continue_cosplay_post') as sc, \
             patch('automatic_foreground_effects.complete_effects_once') as fx, \
             patch.object(workflow, 'diagnose_cosplay_post') as qa:
            assert system.complete_once(root, disabled, args) is disabled
            for producer in (mat, sc, fx, qa): producer.assert_not_called()
            assert not (root / workflow.RUNS_DIR / disabled['run_id'] / 'system-post-completion').exists()
        def own(current, name):
            child = copy.deepcopy(current); child['parent_run_id'] = current['run_id']
            child['run_id'] = name; child['revision'] += 1
            child['cosplay_post_review'] = {'status': 'unavailable'}
            workflow.write_json(root/workflow.RUNS_DIR/name/'manifest.json', child)
            return child
        def material(_root, current, _args, **kwargs):
            calls.append('material')
            child = own(current, 'material-child')
            child['cosplay_post_completion'] = {'status': 'qualified_material_child_saved', 'correction_submissions': 0}
            return {'run': child, 'status': 'qualified_material_child_saved', 'handled': True}
        def scene(_root, current, settings):
            calls.append('scene')
            assert current['run_id'] == 'material-child' and system._review_current(current)
            assert settings._material_completion_checked
            child = own(current, 'scene-child')
            child['cosplay_post_completion'] = {'status': 'corrected_child_saved', 'correction_submissions': 1}
            child['generation_budget']['total_image_submissions'] = 1
            return child
        def effect(_root, current, settings, **kwargs):
            calls.append('effects'); assert kwargs == {'fresh_generation': True, 'bounded': True}
            assert current['run_id'] == 'scene-child'
            child = own(current, 'effect-child')
            child['automatic_foreground_effects'] = {'independent_effect_image_submissions': 1, 'actual_visual_gate': 'pass'}
            return {'run': child, 'status': 'effect_child_saved', 'handled': True}
        def diagnose(_root, current, settings):
            calls.append('review:' + current['run_id']); review(current)
        workflow.write_json(root/workflow.RUNS_DIR/run['run_id']/'manifest.json', run)
        with patch('automatic_material_completion.complete_materials_once', side_effect=material) as mat, \
             patch('cosplay_post_completion.continue_cosplay_post', side_effect=scene) as sc, \
             patch('automatic_foreground_effects.complete_effects_once', side_effect=effect) as fx, \
             patch.object(workflow, 'diagnose_cosplay_post', side_effect=diagnose) as qa:
            saved = system.complete_once(root, run, args)
            assert calls == ['material', 'review:material-child', 'scene', 'effects', 'review:effect-child']
            assert saved['run_id'] == 'effect-child'
            receipt = saved['system_post_completion']
            assert receipt['status'] == 'completed_with_findings'
            assert [row['id'] for row in receipt['stages']] == receipt['stage_order']
            assert receipt['failed_checks'] == ['scene_integration']
            assert not receipt['automatic_visual_acceptance']
            assert saved['generation_budget']['total_image_submissions_including_effects'] == 2
            repeated = system.complete_once(root, run, args)
            assert repeated['run_id'] == saved['run_id']
            assert (mat.call_count, sc.call_count, fx.call_count, qa.call_count) == (1, 1, 1, 2)
        assert workflow.sha256(project) == digest
        ordinary = copy.copy(args); ordinary.complete_post = False
        with patch('automatic_foreground_effects.complete_effects_once') as fx:
            assert system.complete_once(root, run, ordinary) == run
            fx.assert_not_called()
        # A successful reference/skin check still dispatches declared FX.
        clean = copy.deepcopy(run); clean['run_id'] = 'clean'
        review(clean); clean['cosplay_post_review']['checks']['skin_materials']['reference']['status'] = 'pass'
        workflow.write_json(root/workflow.RUNS_DIR/clean['run_id']/'manifest.json', clean)
        with patch('automatic_material_completion.complete_materials_once') as mat, \
             patch('cosplay_post_completion.continue_cosplay_post', side_effect=lambda _r, current, _a: current), \
             patch('automatic_foreground_effects.complete_effects_once', side_effect=lambda _r, current, _a, **kw:
                   {'run': current, 'status': 'skipped', 'handled': False, 'reason': 'baked'}) as fx:
            saved = system.complete_once(root, clean, args)
            mat.assert_not_called(); fx.assert_called_once()
            assert saved['system_post_completion']['stages'][3]['status'] == 'skipped'
    print(json.dumps({'ok': True, 'ordered_full_completion': True, 'material_child_does_not_exit_early': True,
        'effects_checked_without_reference_failure': True, 'final_child_independently_reviewed': True,
        'saved_reentry_no_requests': True, 'source_and_selection_preserved': True, 'network_requests': 0}))


if __name__ == '__main__':
    main()
