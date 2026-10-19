"""Offline production-hook contract; tiny fixtures, no models or network."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import cosplay_post_review as review
from cosplay_post_review_smoke import fake_transport, multi_fake_transport, response, verdict
from cosplay_native_review_smoke import make_native_fixture
from generation_design import DESIGN_VERSION
from generation_prompt import DEFAULT_PROMPT, STYLE_MARKER
import single_source_workflow as workflow


def assert_pending(run):
    assert run['status'] == 'needs_visual_review'
    assert run['delivery_status'] == 'auto_draft'
    assert run['review']['status'] == 'not_checked'
    assert all(value == 'not_checked' for value in run['review']['checks'].values())
    assert not any(row['status'] in {'accepted', 'delivered'} for row in run['events'])


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay-post-workflow-smoke-') as folder:
        root = Path(folder)
        native_size = (1100, 1400)
        source = root / 'original.png'
        Image.new('RGB', native_size, (8, 20, 35)).save(source)
        profile = {'identification': 'unidentified', 'recommended_styles': [],
                   'provenance': {'model': 'fixture_vision_model'}}
        generated, rendered, planned, snapshots = [], [], [], {}

        def planner(source_path, output, current_profile, **kwargs):
            planned.append(str(source_path))
            return {'schema_version': DESIGN_VERSION, 'status': 'ready',
                    'effective_prompt': DEFAULT_PROMPT,
                    'character_research': {'status': 'offline_fixture'},
                    'design_brief': {'summary': 'Offline complete-design fixture',
                                     'primary_effect_plan': {'mode': 'none',
                                         'description': 'An intentionally restrained portrait finish.'}}}

        def provider(argv):
            output = Path(argv[argv.index('--out-dir') + 1])
            output.mkdir(parents=True, exist_ok=True)
            candidate = output / (argv[argv.index('--sample') + 1] + '.png')
            Image.new('RGB', (12, 18), (90, 60, 20)).save(candidate)
            generated.append(candidate)
            return 0

        def frozen_source(plan, *, size):
            assert plan['execution']['source_reference'] == 'paired_native_baseline'
            assert plan['source_baseline']['fixture_identity'] == 'immutable-frozen-source'
            assert size == native_size
            # Distinct from the imported PNG. Above the usual shoulder threshold
            # to also detect accidental re-development of display-linear data.
            return np.full((native_size[1], native_size[0], 3), .95, dtype=np.float32)

        def snapshot(run):
            paths = [run['source']['path'], run['reference']['path'],
                     *[run['artifacts'][key] for key in
                       ('auto_draft', 'render_master', 'render_scene_master')]]
            return {path: workflow.sha256(Path(path)) for path in paths}

        def renderer(_root, run, _args):
            if run['run_id'] == 'complete-generated':
                # The production tail gets explicit saved task permission.
                from creative_task import normalize_task_intent
                run['task_intent'] = normalize_task_intent({'schema_version': 'cosplay_task_intent.v1',
                    'mode': 'create_reference', 'services': {'text_vision': True, 'visual_review': True}})
            output = root / workflow.RUNS_DIR / run['run_id']
            output.mkdir(parents=True, exist_ok=True)
            fixture = make_native_fixture(output/'frozen-inputs', native_size)
            plan = fixture['options']['appearance_revision']['design_reconstruction_plan']
            assert plan['source_sha256'] == run['source']['sha256']
            plan['source_baseline']['fixture_identity'] = 'immutable-frozen-source'
            run['options'].setdefault('appearance_revision', {})['design_reconstruction_plan'] = plan
            outputs = {}
            for key, suffix in (('auto_draft', '.png'), ('render_master', '.tif'),
                                ('render_scene_master', '.tif')):
                path = output / (key + suffix)
                Image.new('RGB', native_size, (60, 80, 110)).save(path)
                outputs[key] = str(path)
            rendered.append(run['run_id'])
            run['artifacts'] = outputs
            snapshots[run['run_id']] = snapshot(run)
            return {'outputs': outputs, 'metadata': {'mode': 'offline_fixture'},
                    'review': {'status': 'not_checked',
                               'checks': {key: 'not_checked' for key in workflow.VISUAL_CHECK_KEYS}}}

        def args_for(name, *, generate=True, reference=None, planning=False):
            argv = ['--root', str(root), '--source', str(source), '--run-id', name]
            if generate:
                argv.append('--generate')
            if reference is not None:
                argv += ['--reference', str(reference)]
            args = workflow.parse_args(argv)
            args.character_profile = profile
            if planning:
                from creative_task import normalize_task_intent
                args.task_intent = normalize_task_intent({'schema_version':'cosplay_task_intent.v1',
                    'mode':'create_reference','services':{'text_vision':True,'visual_review':True}})
                args.task_plan = {'schema_version':'cosplay_task_plan.v2',
                    'inputs':{'source':{'sha256':workflow.sha256(source)}},
                    'image_budget':{'initial_reference':1,'reference_correction':0,'independent_effect':0}}
            return args

        payload = response(True)
        payload['checks']['native_detail']['final'] = verdict('final', panels=['final', 'native_0', 'native_1'])
        payload['checks']['skin_materials']['final'] = verdict('final', status='fail',
            cause='transfer_incomplete', stage='subject_appearance',
            action='Correct supported material colour transitions.', panels=['reference', 'final'])
        with patch('reference_provider.provider_config',
                   return_value={'provider': 'offline_fixture', 'model': 'fixture_image_model'}), \
             patch('reference_provider.main', side_effect=provider) as image_sender, \
             patch('generation_design.prepare_generation_design', side_effect=planner), \
             patch('creative_task.validate_task_plan', side_effect=lambda plan, _intent: copy.deepcopy(plan)), \
             patch('automatic_native_retouch.prepare_once', side_effect=lambda _root, current, *_args, **_kw:
                   {'run':current,'status':'offline_source_fixture'}), \
             patch.object(workflow, 'render_candidate', side_effect=renderer), \
             patch('native_source_baseline.source_for_plan', side_effect=frozen_source) as source_reader, \
             patch.object(review, 'review_cosplay_post', wraps=review.review_cosplay_post) as reviewer, \
             patch('vision_transport.request_vision_json', side_effect=multi_fake_transport(payload)) as vision:
            # Real create_run must reach its tail hook after rendering, not just
            # pass a direct unit call to the helper. Use a non-default revision.
            parent = root / workflow.RUNS_DIR / 'revision-parent' / 'manifest.json'
            workflow.write_json(parent, {'revision': 6})
            args = args_for('complete-generated', planning=True)
            args.parent_run_id = 'revision-parent'
            run = workflow.create_run(args)
            assert len(generated) == 1 and len(rendered) == 1 and len(planned) == 1
            assert reviewer.call_count == source_reader.call_count == 1 and vision.call_count > 1
            assert run['revision'] == 7 and run['generation_design']['status'] == 'ready'
            assert run['generation_design']['schema_version'] == DESIGN_VERSION
            diagnosis = run['cosplay_post_review']
            assert diagnosis['status'] == 'reviewed' and diagnosis['diagnostic_gate'] == 'fail'
            assert diagnosis['automatic_accepted'] is False and diagnosis['visual_acceptance'] is False
            assert diagnosis['checks']['skin_materials']['final']['correction_stage'] == 'subject_appearance'
            assert_pending(run)
            assert snapshot(run) == snapshots[run['run_id']]
            assert diagnosis['run_binding'] == {'run_id': run['run_id'], 'revision': 7,
                'source_sha256': workflow.sha256(source),
                'reference_sha256': workflow.sha256(Path(run['reference']['path'])),
                'final_sha256': workflow.sha256(Path(run['artifacts']['auto_draft'])),
                'render_artifacts': {key:workflow.sha256(Path(value)) for key,value in run['artifacts'].items()
                    if key in ('auto_draft', 'render_master', 'render_scene_master')}}
            images = diagnosis['binding']['images']
            assert Path(images['reference']['path']) == Path(run['reference']['path'])
            assert Path(images['final']['path']) == Path(run['artifacts']['auto_draft'])
            assert images['reference']['sha256'] == run['reference']['sha256']
            assert images['final']['sha256'] == diagnosis['run_binding']['final_sha256']
            assert Path(images['source']['path']).name == 'source-display.png'
            assert images['source']['sha256'] != run['source']['sha256']
            with Image.open(images['source']['path']) as displayed:
                assert displayed.size == native_size and displayed.getpixel((0, 0)) == (249, 249, 249)
            assert [panel['id'] for panel in diagnosis['comparison']['panels']] == ['source', 'reference', 'final']
            assert diagnosis['native_selection']['status'] == 'ready'
            assert diagnosis['native_selection']['native_size'] == list(native_size)
            assert diagnosis['checks']['native_detail']['final']['status'] == 'pass'
            assert diagnosis['checks']['native_detail']['reference']['status'] == 'unknown'
            assert diagnosis['native_detail_scope'] == 'only_listed_actually_cited_source_final_native_windows'
            assert len(diagnosis['native_crops']) > 3
            assert diagnosis['native_crops'][0]['box'] == diagnosis['native_crops'][1]['box']
            assert all(crop['displayed_at_100_percent'] for crop in diagnosis['native_crops'])
            assert all(max(row['api_input']['size']) <= 1024 for row in diagnosis['reviews'])
            assert diagnosis['native_sampling_coverage']['unobserved_pairs']==[]
            assert diagnosis['whole_image_native_acceptance'] is False
            assert Path(diagnosis['comparison']['path']).is_file()
            manifest = root / workflow.RUNS_DIR / run['run_id'] / 'manifest.json'
            assert workflow.read_json(manifest)['cosplay_post_review'] == diagnosis
            assert run['generation_budget']['image_submissions'] == image_sender.call_count == 1

            # A completion coordinator may explicitly review a freshly composed
            # child. Ordinary imported create_run remains excluded below.
            child_args = args_for('explicit-child-review', generate=False)
            child_args.cosplay_review_only = True
            count = image_sender.call_count
            child_review = workflow.diagnose_cosplay_post(root, run, child_args)
            assert child_review['status'] == 'reviewed' and child_review['native_selection']['status'] == 'ready'
            assert image_sender.call_count == count and snapshot(run) == snapshots[run['run_id']]
            assert_pending(run)

            def offline(*_args, **_kwargs):
                raise OSError('Controlled offline diagnostic fixture')

            # Actual reviewer handles unavailable transport and malformed model
            # output. Both save unknown diagnosis while keeping identical assets.
            malformed = response(True)
            del malformed['checks']['effects_space']
            for index, transport in enumerate((offline, fake_transport(malformed))):
                vision.side_effect = transport
                count = image_sender.call_count
                failure_run = copy.deepcopy(run)
                failure_run['run_id'] = 'fresh-failed-diagnosis-'+str(index)
                unavailable = workflow.diagnose_cosplay_post(root, failure_run, args)
                assert unavailable['status'] == 'unavailable'
                assert unavailable['automatic_accepted'] is False and unavailable['visual_acceptance'] is False
                assert all(value['status'] == 'unknown' for row in unavailable['checks'].values()
                           for value in row.values())
                assert unavailable['run_binding']['final_sha256'] == diagnosis['run_binding']['final_sha256']
                receipt = root / workflow.RUNS_DIR / failure_run['run_id'] / 'cosplay-post-review/review.json'
                assert workflow.read_json(receipt) == unavailable
                assert image_sender.call_count == count
                assert snapshot(run) == snapshots[run['run_id']]
                assert_pending(run)

            # Gate exclusions must prevent the reviewer and source reader from
            # running; a caught mock error would not count as proof of no call.
            exclusions = [('legacy-design', True, 'old-design', 'ready', 'needs_visual_review'),
                          ('non-ready-design', True, DESIGN_VERSION, 'unavailable', 'needs_visual_review'),
                          ('imported-reference', False, DESIGN_VERSION, 'ready', 'needs_visual_review'),
                          ('legacy-exact', True, DESIGN_VERSION, 'legacy_frozen', 'needs_visual_review'),
                          ('blocked-render', True, DESIGN_VERSION, 'ready', 'blocked'),
                          ('awaiting-reference', True, DESIGN_VERSION, 'ready', 'awaiting_reference'),
                          ('delivered', True, DESIGN_VERSION, 'ready', 'delivered')]
            for name, generate, version, design_status, run_status in exclusions:
                excluded = copy.deepcopy(run)
                excluded.pop('cosplay_post_review', None)
                excluded.update(run_id=name, status=run_status)
                excluded['generation_design'].update(schema_version=version, status=design_status)
                excluded_args = args_for(name, generate=generate)
                before = (reviewer.call_count, vision.call_count, source_reader.call_count, image_sender.call_count)
                untouched = copy.deepcopy(excluded)
                assert workflow.diagnose_cosplay_post(root, excluded, excluded_args) == {}
                assert excluded == untouched
                assert before == (reviewer.call_count, vision.call_count, source_reader.call_count, image_sender.call_count)
                assert not (root / workflow.RUNS_DIR / name / 'cosplay-post-review').exists()

            # Imported references add no implicit review. Retired exact retry
            # is rejected before any image submission or diagnosis.
            before = (reviewer.call_count, vision.call_count, source_reader.call_count)
            imported_args = args_for('imported-create-run', generate=False, reference=generated[0])
            imported_args.generation_design = copy.deepcopy(run['generation_design'])
            imported = workflow.create_run(imported_args)
            assert_pending(imported)
            assert imported['cosplay_post_review']['status'] == 'not_requested' and image_sender.call_count == 1
            exact_args = args_for('legacy-exact-create-run')
            exact_args.effective_prompt = 'The exact saved legacy editing text.'
            exact_args.preserve_effective_prompt = True
            try: workflow.create_run(exact_args)
            except ValueError as error: assert 'complete current generation design' in str(error)
            else: raise AssertionError('Retired exact retry was accepted')
            assert len(planned) == 1 and image_sender.call_count == 1
            assert before == (reviewer.call_count, vision.call_count, source_reader.call_count)
            assert snapshot(imported) == snapshots[imported['run_id']]

            # Default complete-post saves real pixels without submitting QA,
            # reserving an attempt, or dispatching diagnostic correction.
            default_args = args_for('default-no-visual-review')
            default_args.complete_post = True
            default_args.cosplay_correction_budget = 1
            full_brief = '丹恒的角色摄影、人像美化和纵深。\n保持真实面孔、妆造与摄影质感。'
            default_args.prompt = default_args.input_prompt = full_brief + STYLE_MARKER + 'OLD STYLE'
            default_args.preset = 'night_scene'
            default_args.variant_count = 2
            default_args.variant_styles = ['night_scene', 'cinematic_backlight']
            from creative_task import normalize_task_intent
            creative_requirements = '一看就是角色主导的 COS 大片，同时保持真实摄影质感。'
            default_args.task_intent = normalize_task_intent({'schema_version':'cosplay_task_intent.v1',
                'mode':'create_reference','brief':creative_requirements,
                'constraints':{'preserve_background':True,'no_effects':True}})
            default_args.task_plan = copy.deepcopy(args.task_plan)
            default_args.task_plan['image_budget']['initial_reference'] = 2
            default_args.character_profile = None
            before = (reviewer.call_count, vision.call_count, source_reader.call_count)
            with patch('system_post_completion.complete_once') as coordinator, \
                    patch('character_profile.analyze_character') as recognizer:
                default_run = workflow.create_run(default_args)
                coordinator.assert_not_called()
                recognizer.assert_not_called()
            assert_pending(default_run)
            from preset_store import get_preset
            assert len(default_run['variants']) == 2
            for variant, style_id in zip(default_run['variants'], default_args.variant_styles):
                actual_prompt = variant['prompt']
                assert full_brief in actual_prompt and get_preset(style_id)['prompt'] in actual_prompt
                assert creative_requirements in actual_prompt and 'OLD STYLE' not in actual_prompt
                assert '保留原背景，不更换或重建场景。' in actual_prompt
                assert '不要添加特效。' in actual_prompt
            assert default_run['generation_design']['status'] == 'not_requested'
            assert default_run['generation_design']['text_vision_submissions'] == 0
            assert default_run['generation_design']['character_research']['status'] == 'not_requested'
            assert len(planned) == 1
            assert snapshot(default_run) == snapshots[default_run['run_id']]
            assert before == (reviewer.call_count, vision.call_count, source_reader.call_count)
            disabled = default_run['cosplay_post_review']
            assert disabled['status'] == 'not_requested' and disabled['human_review_required']
            assert not disabled['actual_api_evidence_valid'] and not disabled['automatic_accepted']
            assert disabled['text_vision_submissions'] == 0 and not disabled['visual_acceptance']
            assert workflow.read_json(root / workflow.RUNS_DIR / default_run['run_id'] / 'manifest.json')['cosplay_post_review'] == disabled
            assert not (root / workflow.RUNS_DIR / default_run['run_id'] / 'cosplay-post-review').exists()
            assert not (root / workflow.RUNS_DIR / default_run['run_id'] / 'system-post-completion').exists()

    print(json.dumps({'ok': True, 'real_create_run_tail_hook': True,
        'frozen_display_source_used': True, 'actual_source_reference_final_run_revision_bound': True,
        'native_source_full_size_and_verified_paired_crops': True, 'internal_child_recheck_explicit_only': True,
        'diagnostic_failure_retains_pending_and_masters': True,
        'offline_and_malformed_unavailable_saved': True, 'gate_exclusions_make_no_request': True,
        'imported_reference_no_implicit_diagnosis_and_retired_retry_rejected': True,
        'diagnosis_adds_no_image_request': True, 'models_used': False, 'network_calls': 0}))


if __name__ == '__main__':
    main()
