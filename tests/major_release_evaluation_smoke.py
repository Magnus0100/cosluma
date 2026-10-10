"""Offline evidence counterexamples; no actual photo/model/service processing."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import tifffile

import major_release_evaluation as audit
from raw_support import deterministic_srgb_profile, save_render_master, save_scene_master


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf8')


def fixture(root, number):
    directory = root / 'outputs/runs' / str(number)
    directory.mkdir(parents=True)
    source = root / ('source-' + str(number) + '.png')
    Image.new('RGB', (37, 29), (35 + number, 85, 120)).save(source)
    reference = root / ('reference-' + str(number) + '.png')
    Image.new('RGB', (37, 29), (65, 75 + number, 120)).save(reference)
    values = np.linspace(.02, .75, 37 * 29 * 3, dtype=np.float32).reshape(29, 37, 3)
    draft, master, scene = (directory / name for name in ('auto_draft.png', 'render_master.tif', 'render_scene.tif'))
    Image.new('RGB', (37, 29), (57, 88, 123)).save(draft, icc_profile=deterministic_srgb_profile())
    save_render_master(values, master, display_transform={'method': 'clip_sdr'})
    save_scene_master(values, scene, working_domain='display_linear_sRGB')
    intent = {'schema_version': 'cosplay_task_intent.v1', 'mode': 'existing_reference',
              'brief': '同一主题', 'constraints': {'no_effects': True, 'preserve_background': False},
              'services': {'text_vision': True}}
    plan = {'schema_version': 'cosplay_task_plan.v1', 'mode': intent['mode'], 'intent': intent,
        'ready': True, 'plan_id': 'a' * 64, 'inputs': {'source': {'sha256': audit.sha(source)},
        'reference': {'sha256': audit.sha(reference)}}, 'image_budget': dict.fromkeys(audit.BUDGET_KEYS, 0),
        'services': {'text_vision': True, 'image_generation': False},
        'preflight_network_requests': 0, 'preflight_model_inferences': 0, 'preflight_jobs_created': 0,
        'capabilities': {'model_profiles': audit.MODEL_PROFILES}}
    review = directory / 'visual_review.json'
    write(review, {'run_id': str(number), 'revision': 1, 'status': 'needs_review',
                   'checks': dict.fromkeys(audit.HUMAN_CHECKS, 'not_checked')})
    run = {'run_id': str(number), 'revision': 1, 'status': 'needs_visual_review',
        'software_version': '0.3.4', 'source': {'path': str(source), 'sha256': audit.sha(source),
        'oriented_size': [37, 29]}, 'reference': {'path': str(reference), 'sha256': audit.sha(reference)},
        'task_intent': intent, 'task_plan': plan, 'options': {},
        'render': {'software_version': '0.3.4', 'run_id': str(number), 'revision': 1},
        'artifacts': {'auto_draft': str(draft), 'render_master': str(master),
            'render_scene_master': str(scene), 'visual_review': str(review)}}
    hashes = {key: audit.sha(run['artifacts'][key]) for key in audit.MASTERS}
    run['cosplay_post_review'] = {'status': 'reviewed', 'actual_api_evidence_valid': True,
        'run_binding': {'run_id': str(number), 'revision': 1, 'source_sha256': audit.sha(source),
            'reference_sha256': audit.sha(reference), 'final_sha256': hashes['auto_draft'], 'render_artifacts': hashes},
        'checks': {key: {'final': {'status': 'pass'}} for key in audit.CHECKS},
        'native_sampling_coverage': {'actually_observed_pairs': ['eyes'],
            'unsupported_regions': [{'region': 'lower_clothing', 'status': 'unknown'}]}}
    manifest = directory / 'manifest.json'
    write(manifest, run)
    return {'manifest': str(manifest), 'source_sha256': audit.sha(source),
            'split': 'baseline', 'source_group': 'explicit_same_session'}, run


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='cosplay-major-readonly-') as folder:
        root = Path(folder)
        fixtures = [fixture(root, i) for i in (1, 2, 3)]
        cases = [pair[0] for pair in fixtures]
        protected = {str(path): audit.sha(path) for path in root.rglob('*') if path.is_file()}
        with (patch('urllib.request.urlopen', side_effect=AssertionError('No network')),
              patch('subprocess.Popen', side_effect=AssertionError('No model process'))):
            result = audit.evaluate(cases)
            assert result['storage_input_invariants_met'] and result['independent_saved_source_SHA_count'] == 3
            assert not result['release_evidence_ready'] and not result['human_final_work_approval_granted']
            assert not result['distinct_person_or_scene_count_verified']
            assert all(case['quality']['quality_state'] == 'recorded_scoped_pass_unreplayed' for case in result['cases'])
            assert all(case['quality']['unsupported_regions'] for case in result['cases'])
            assert all(not case['producer']['current_major_producer_qualified'] for case in result['cases'])
            checks.append('three_sources_do_not_grant_major_producer_visual_or_human_pass')
            duplicate = audit.evaluate([cases[0], cases[0], cases[0]])
            assert duplicate['independent_saved_source_SHA_count'] == 1 and not duplicate['storage_input_invariants_met']
            checks.append('duplicate_source_SHA_is_not_independence')
            pending = {'case_id': 'held', 'source': fixtures[2][1]['source']['path'],
                'source_sha256': cases[2]['source_sha256'], 'split': 'held_out', 'source_group': 'explicit_same_session'}
            declared = audit.evaluate(cases + [pending])
            assert declared['registered_held_out_case_count'] == 1 and not declared['held_out_evidence_complete']
            assert declared['cases'][-1]['quality_state'] == 'unknown'
            checks.append('explicit_pending_held_out_does_not_become_saved_pass')
            dev, held = copy.deepcopy(cases[0]), copy.deepcopy(cases[0])
            dev['split'], held['split'] = 'development', 'held_out'
            overlap = audit.evaluate([dev, held], minimum=1)
            assert overlap['held_out_source_SHA_also_labelled_development'] == [cases[0]['source_sha256']]
            assert not overlap['held_out_evidence_complete']
            checks.append('explicit_development_and_held_out_overlap_stays_visible')
            changed = copy.deepcopy(fixtures[0][1])
            changed['task_plan']['image_budget']['independent_effect'] = 1
            assert audit.task_evidence(changed, changed['source']['sha256'], changed['reference']['sha256'])['status'] == 'invalid'
            changed = copy.deepcopy(fixtures[0][1])
            changed['automatic_foreground_effects'] = {'run_id': changed['run_id'], 'image_submissions': 1}
            assert audit.task_evidence(changed, changed['source']['sha256'], changed['reference']['sha256'])['status'] == 'invalid'
            checks.append('zero_image_route_rejects_budget_or_recorded_submission')
            changed = copy.deepcopy(fixtures[0][1])
            changed['cosplay_post_review']['run_binding']['run_id'] = 'ancestor'
            bound = {key: audit.sha(changed['artifacts'][key]) for key in audit.MASTERS}
            assert audit.quality_evidence(changed, root, bound)['status'] == 'missing_or_stale'
            checks.append('old_QA_cannot_certify_new_run')
            assert protected == {str(path): audit.sha(path) for path in root.rglob('*') if path.is_file()}
            checks.append('evaluation_preserves_all_inputs')
            master = Path(fixtures[0][1]['artifacts']['render_master'])
            tifffile.imwrite(master, np.zeros((29, 37, 3), np.uint8), photometric='rgb')
            bad = audit.evaluate(cases)
            assert not bad['storage_input_invariants_met']
            assert bad['cases'][0]['native_masters']['render_master']['status'] == 'invalid_or_missing'
            checks.append('rgb8_TIFF_is_not_native_uint16_master')
            output = root / 'reports/result.json'
            assert audit.main(['--run', cases[1]['manifest'], '--require-min-sources', '1', '--output', str(output)]) == 0
            try:
                audit.main(['--run', cases[1]['manifest'], '--require-min-sources', '1', '--output', str(output)])
            except FileExistsError:
                pass
            else:
                raise AssertionError('Old evaluation evidence was overwritten')
            checks.append('CLI_writes_new_external_evidence_only')
    print(json.dumps({'ok': True, 'checks': checks, 'network_requests': 0, 'model_inferences': 0}))


if __name__ == '__main__':
    main()
