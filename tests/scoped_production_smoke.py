"""Checked desktop task -> model boundaries -> real native save/review/export.

Only external image/vision execution and model-installation probes are fixtures.
The planner, checked dispatch, coordinator, generation adapter, frozen assets,
pixel renderer, review board and export encoders run normally on two originals.
Synthetic images and responses are not artistic-quality evidence.
"""
from contextlib import ExitStack
import copy
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from asset_identity import sha256_file
from backend_server import Backend
from creative_task_smoke import fixture, intent, snapshot
from export_artifact import export_source_run
from photo_editing_smoke import proposal
from task_model_preflight_smoke import mocked_task_capability


class ScopedProductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='scoped-production-')
        self.addCleanup(self.temporary.cleanup)
        self.metadata_cache = tempfile.TemporaryDirectory(prefix='scoped-metadata-cache-')
        self.addCleanup(self.metadata_cache.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.registry, sources, _reference = fixture(self.root)
        self.sources = [Path(path) for path in sources[:2]]
        for source, size, color in zip(self.sources, ((64, 96), (120, 80)), ((100, 120, 90), (90, 80, 120))):
            image = Image.new('RGB', size, color)
            ImageDraw.Draw(image).rectangle((size[0]//4, size[1]//4, size[0]*3//4, size[1]*3//4), fill=(170, 130, 110))
            image.save(source)
        self.backend = Backend(self.root, embedded=True)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {'COSPLAY_MODEL_REGISTRY': str(self.registry),
            'COSPLAY_PROVIDER_SETTINGS': str(self.root/'provider-settings.json'),
            'COSPLAY_MODEL_SETTINGS': str(self.root/'model-selection.json'),
            'COSPLAY_CONFIG_ROOT': str(self.root/'config'), 'SCOPED_FIXTURE_KEY': 'fixture-only-never-transmitted',
            'COSPLAY_CACHE_ROOT': str(Path(self.metadata_cache.name)/'cache'),
            'COSPLAY_ASSET_IDENTITY_CACHE': str(Path(self.metadata_cache.name)/'identity'),
            'COSPLAY_SOURCE_METADATA_CACHE': str(Path(self.metadata_cache.name)/'metadata')}))
        self.stack.enter_context(patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability))
        self.stack.enter_context(patch('runtime_preflight._probe_runtime', return_value=None))
        self.vision = self.stack.enter_context(patch('vision_transport.request_vision_json', side_effect=self.vision_request))
        self.generation = self.stack.enter_context(patch('reference_provider.request_edit', side_effect=self.image_request))
        self.stack.enter_context(patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected actual network')))
        self.image_requests = []; self.vision_requests = []
        self.review_failure = None
        self.decision_tool = 'joint_edit'
        self.decision_stop = False

    def request(self, source, *, review=True, **extra):
        creative = intent('create_reference')
        creative['brief'] = 'Preserve the indoor setting and improve coherent character lighting with restrained visible scene edits.'
        creative['constraints']['background_mode'] = 'enhance_source'
        creative['services'] = {'text_vision': True, 'visual_review': review}
        return {'source': str(source), 'model_registry': str(self.registry), 'creative_task': creative,
            'editing_request': {'work_type': 'partial_composite', 'style': 'expressive', 'language': 'en'},
            # This fixture verifies the one-shot preparation/save/export seam.
            # Default automatic finishing has its own model-decision fixtures.
            'editing_autofinish': {'maximum_improvements': 0},
            'provider': 'openai-compatible', 'model': 'gpt-image-2.5-sunburst',
            'api_base': 'https://example.invalid/v1', 'api_key_env': 'SCOPED_FIXTURE_KEY',
            'vision_model': 'synthetic-vision-fixture', **extra}

    def vision_request(self, source, output, prompt, **kwargs):
        source = Path(source).resolve()
        self.vision_requests.append({'source': str(source), 'sha256': sha256_file(source), 'prompt': prompt})
        provenance = {'source': {'path': str(source), 'sha256': sha256_file(source)},
            'network_used': True, 'generation_used': False, 'model': 'actual-fixture-vision-model',
            'input_profile': kwargs.get('input_profile','standard')}
        if source.name == 'review-board.png':
            if 'decision' in str(output):
                if getattr(self,'decision_stop',False):
                    return {'action':'stop','reason':'No useful safe edit identified by the synthetic model.'},provenance
                decision={'action':'tool','tool':getattr(self,'decision_tool','joint_edit'),'operations':['relight'],
                    'instruction':'Improve coherent subject and scene lighting with restrained visible atmosphere.',
                    'reason':'Observed source lighting and requested creative treatment.'}
                if decision['tool']=='edit_region':decision['region']=[.25,.25,.75,.75]
                return decision,provenance
            mapping=json.loads((source.parent/'review-board.json').read_text(encoding='utf8'))
            edited=len(mapping['images'])==3
            if self.review_failure is not None and edited:
                raise self.review_failure
            with Image.open(source) as board:
                self.assertEqual(board.size,(len(mapping['panels'])*524+12,1644 if mapping.get('region') else 828))
            payload = {'credibility': {'status': 'unknown', 'findings': ['Synthetic imagery cannot establish portrait identity']},
                'creative_benefit': {'status': 'pass', 'findings': ['Synthetic edited appearance remains visible']},
                'comparison': 'better' if edited else 'first_candidate',
                'primary_issue': 'Synthetic fixture has no artistic acceptance'}
        else:
            self.assertIn(source, self.sources)
            payload = proposal()
        return payload, provenance

    def image_request(self, endpoint, key, image, model, prompt, size, timeout, retry=None, events=None):
        self.image_requests.append({'image': str(image), 'model': model, 'prompt': prompt})
        with Image.open(image) as source:
            # Deliberately low-resolution edited appearance exposes accidental
            # source RGB restoration and tests honest native-canvas resizing.
            edited = source.convert('RGB').resize((max(1, source.width//2), max(1, source.height//2)))
        values = np.asarray(edited).copy()
        values[:, :, 2] = np.minimum(values[:, :, 2].astype(np.int16)+65, 255).astype(np.uint8)
        edited = Image.fromarray(values)
        ImageDraw.Draw(edited).ellipse((2, 2, max(4, edited.width//3), max(4, edited.height//3)), fill=(120, 175, 235))
        stream = BytesIO(); edited.save(stream, format='PNG')
        return {'id': 'fixture-image-'+str(len(self.image_requests)), 'created': 0}, stream.getvalue(), 'fixture_png_bytes'

    def start(self, request):
        before = snapshot(self.root)
        checked = self.backend.task_plan(request)
        self.assertTrue(checked['plan']['ready'], checked['plan'].get('blockers'))
        self.assertEqual(snapshot(self.root), before, 'Task planning wrote files')
        self.assertEqual(self.image_requests, [])
        response = self.backend.start_task({**checked['request'], 'plan_id': checked['plan']['plan_id'], 'async': False})
        self.assertTrue(response.get('ok'), {'stderr':response.get('stderr'),
            'error':response.get('error'),'run_error':(response.get('run') or {}).get('error')})
        return checked, response['run']

    def test_omitted_creation_scope_has_one_current_default_and_no_planning_calls(self):
        request=self.request(self.sources[0])
        request.pop('editing_request')
        request.pop('editing_autofinish')
        request['creative_task'].pop('services')
        checked=self.backend.task_plan(request)
        plan=checked['plan']
        self.assertTrue(plan['ready'],plan.get('blocking_reasons'))
        self.assertEqual(plan['editing_scope']['work_type'],'partial_composite')
        self.assertEqual(plan['editing_scope']['style'],'expressive')
        self.assertEqual(plan['editing_scope']['target_setting'],'same')
        self.assertEqual(plan['editing_autofinish'],{'maximum_improvements':2,'maximum_zero_image_finishes':1})
        self.assertEqual(plan['image_budget']['initial_reference'],1)
        self.assertEqual(plan['generation_submissions_upper_bound'],3)
        self.assertTrue(plan['intent']['services']['visual_review'])
        self.assertEqual(self.image_requests,[])
        self.assertEqual(self.vision_requests,[])

    def test_retired_creation_permissions_and_forged_scope_less_worker_are_rejected(self):
        from cos_node_registry import compile_workflow
        import creative_task
        import single_source_workflow as workflow
        request=self.request(self.sources[0])
        for key,value in (('reference_role','source_conditioned_design'),
                          ('refinement_profile',{'id':'anything'})):
            body=copy.deepcopy(request);body['creative_task']['constraints'][key]=value
            with self.assertRaisesRegex(ValueError,'Retired creation'):
                self.backend.task_plan(body)
        for key in ('creative_direction','regional_resume_authorization','regional_candidate_reuse_authorization'):
            with self.assertRaisesRegex(ValueError,'Retired creation'):
                self.backend.task_plan({**request,key:{}})
        body=copy.deepcopy(request);body['creative_task']['regional_refinement']={}
        with self.assertRaisesRegex(ValueError,'Retired creation'):
            self.backend.task_plan(body)
        checked=self.backend.task_plan(request)
        forged=copy.deepcopy(checked['plan']);forged.pop('editing_scope')
        with self.assertRaisesRegex(ValueError,'checked editing scope'):
            creative_task.validate_task_plan(forged,forged['intent'])
        with self.assertRaisesRegex(ValueError,'checked editing scope'):
            compile_workflow('create_reference')
        from photo_editing import editing_scope
        with self.assertRaisesRegex(ValueError,'source-native initialization'):
            compile_workflow('create_reference',editing_scope=editing_scope({'work_type':'retouch'}))
        args=workflow.parse_args(['--root',str(self.root/'unchecked'),'--source',str(self.sources[0]),'--generate'])
        with self.assertRaisesRegex(ValueError,'checked scoped task'):
            workflow.create_run(args)
        # The timing entry may retain an error ledger; no project or run is
        # initialized and no model/provider is reached by this rejection.
        self.assertFalse((self.root/'unchecked'/'outputs'/'runs').exists())
        self.assertEqual(self.image_requests,[])
        self.assertEqual(self.vision_requests,[])

    def test_two_originals_use_one_checked_observed_native_pipeline(self):
        for index, source in enumerate(self.sources):
            with self.subTest(source=source.name):
                self.image_requests.clear()
                checked, run = self.start(self.request(source, run_id='scoped-'+str(index)))
                self.assertEqual(len(self.image_requests), 1)
                self.assertEqual(run['status'], 'needs_visual_review')
                self.assertEqual(run['production_contract']['plan_id'], checked['plan']['plan_id'])
                self.assertEqual(run['production_contract']['editing_scope_id'], checked['plan']['editing_scope']['scope_id'])
                self.assertEqual(run['generation_design']['design_origin'], 'observed_scoped_edit')
                self.assertEqual(run['generation_design']['art_direction']['provenance']['model'], 'actual-fixture-vision-model')
                self.assertEqual(self.image_requests[0]['prompt'], run['reference']['prompt'])
                self.assertEqual(run['generation_budget']['image_submissions'], 1)
                self.assertEqual(run['options']['denoise_strength'], 0)
                frozen = run['options']['photo_appearance']
                receipt=json.loads((Path(frozen['path'])/'receipt.json').read_text(encoding='utf8'))
                self.assertEqual(receipt['design_id'], run['generation_design']['observed_design']['design_id'])
                draft = Path(run['artifacts']['auto_draft'])
                with Image.open(source) as original, Image.open(draft) as rendered:
                    self.assertEqual(rendered.size, original.size)
                    self.assertGreater(float(np.mean(np.abs(np.asarray(rendered).astype(float)-np.asarray(original)))), 8)
                self.assertEqual(run['render']['kind'], 'scoped_photo_appearance_render')
                self.assertFalse(run['render']['alpha']['is_source_foreground'])
                self.assertFalse(run['photo_edit_session']['human_approved'])
                self.assertFalse(run['photo_edit_session']['automated_target_met'])
                self.assertFalse(run['photo_edit_session']['native_detail_acceptance'])
                self.assertEqual(run['photo_edit_session']['reserved_budget_usage']['vision_requests'], 3)
                self.assertEqual(run['photo_edit_session']['origin_request_budget']['reserved_vision_requests'], 4)
                for output_format in ('PNG', 'TIFF16', 'TIFF32_LINEAR'):
                    exported = export_source_run(self.root, run['run_id'], directory=str(self.root/'exports'),
                                                 artifact='draft', format=output_format)
                    self.assertTrue(exported['ok'])
                    self.assertFalse(exported['metadata']['approved'])
                    self.assertFalse(exported['metadata']['export_rerendered'])
                self.assertEqual(len(self.image_requests), 1, 'Save/export submitted another image')
                saved = json.loads((self.root/'outputs/runs'/run['run_id']/'manifest.json').read_text(encoding='utf8'))
                self.assertEqual(saved['production_contract'], run['production_contract'])
                self.assertEqual(saved['source']['sha256'], sha256_file(source))
        self.assertEqual(len(self.vision_requests), 8, 'Expected design, baseline review, decision and recheck per original')

    def test_review_veto_and_changed_task_are_enforced_before_external_requests(self):
        checked, run = self.start(self.request(self.sources[0], review=False, run_id='no-review'))
        self.assertEqual(len(self.image_requests), 1)
        self.assertEqual(len(self.vision_requests), 2)
        self.assertEqual(run['photo_edit_session']['status'], 'stopped')
        changed = copy.deepcopy(checked['request'])
        changed['editing_request']['work_type'] = 'full_composite'
        with self.assertRaises(ValueError):
            self.backend.start_task({**changed, 'plan_id': checked['plan']['plan_id'], 'async': False})
        self.assertEqual(len(self.image_requests), 1)
        self.assertEqual(len(self.vision_requests), 2)

    def test_local_initial_edit_keeps_native_float_outside_its_coverage(self):
        from photo_appearance import load_asset
        self.decision_tool='edit_region'
        _checked,run=self.start(self.request(self.sources[1],run_id='local-first'))
        binding={'source_sha256':run['source']['sha256'],'scope':run['task_plan']['editing_scope'],
            'design_id':run['generation_design']['observed_design']['design_id']}
        original=load_asset(run['initial_source_appearance'],**binding)
        current=load_asset(run['options']['photo_appearance'],**binding)
        try:
            untouched=current['edit_coverage']==0
            self.assertTrue(np.any(untouched))
            self.assertTrue(np.any(~untouched))
            self.assertTrue(np.array_equal(current['master'][untouched],original['master'][untouched]))
            self.assertEqual(current['receipt']['kind'],'mixed_regional')
            self.assertEqual(len(self.image_requests),1)
            self.assertEqual(len(self.vision_requests),4)
        finally:
            for item in (original,current):
                if getattr(item['master'],'_mmap',None):item['master']._mmap.close()

    def test_source_baseline_cannot_be_published_as_a_completed_edit(self):
        self.decision_stop=True
        _checked,run=self.start(self.request(self.sources[0],run_id='source-only-result'))
        self.assertEqual(run['status'],'no_edited_candidate')
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertTrue(run['photo_edit_session']['selected_is_source_baseline'])
        self.assertEqual(len(self.image_requests),0)
        self.assertEqual(len(self.vision_requests),3)

    def test_explicit_synthetic_approval_owns_native_assets_before_parent_cleanup(self):
        import shutil
        import single_source_workflow as workflow
        from generated_appearance_render import validate_run
        _checked,run=self.start(self.request(self.sources[0],run_id='delivery-ownership'))
        original_png=Path(run['artifacts']['auto_draft']).read_bytes()
        accepted=workflow.accept_run(self.root,run['run_id'],reviewer='CPU fixture only; no real photo approved',
            checks={key:'pass' for key in workflow.VISUAL_CHECK_KEYS})
        delivery=Path(accepted['artifacts']['delivered']).parent
        frozen=json.loads((delivery/'manifest.json').read_text(encoding='utf8'))
        for asset in (frozen['options']['photo_appearance'],frozen['initial_source_appearance']):
            self.assertTrue(Path(asset['path']).is_relative_to(delivery))
        for asset in (run['options']['photo_appearance'],run['initial_source_appearance']):
            directory=Path(asset['path']).resolve()
            self.assertTrue(directory.is_relative_to(self.root))
            shutil.rmtree(directory)
        self.sources[0].unlink()
        validate_run(self.root,frozen)
        for format in ('PNG','TIFF16','TIFF32_LINEAR'):
            result=export_source_run(self.root,run['run_id'],directory=str(self.root/'exports'),
                artifact='delivery',format=format)
            self.assertTrue(result['metadata']['approved'])
            if format=='PNG':self.assertEqual(Path(result['path']).read_bytes(),original_png)

    def test_observation_is_required_and_failed_design_cannot_submit_an_image(self):
        request = self.request(self.sources[0], review=False)
        request['creative_task']['services']['text_vision'] = False
        prepared = self.backend.task_plan(request)
        self.assertFalse(prepared['plan']['ready'])
        with self.assertRaises(ValueError):
            self.backend.start_task({**prepared['request'], 'plan_id': prepared['plan']['plan_id'], 'async': False})
        self.assertEqual(len(self.vision_requests), 0)
        self.assertEqual(len(self.image_requests), 0)
        request = self.request(self.sources[0], review=False, run_id='bad-design')
        checked = self.backend.task_plan(request)
        original_transport = self.vision_request
        def wrong_identity(source, output, prompt, **kwargs):
            payload, provenance = original_transport(source, output, prompt, **kwargs)
            provenance['source']['sha256'] = '0'*64
            return payload, provenance
        self.vision.side_effect = wrong_identity
        response = self.backend.start_task({**checked['request'], 'plan_id': checked['plan']['plan_id'], 'async': False})
        self.assertFalse(response['ok'])
        self.assertEqual(len(self.vision_requests), 1)
        self.assertEqual(len(self.image_requests), 0)

    def test_review_timeout_preserves_complete_native_output_and_export(self):
        self.review_failure = TimeoutError('Synthetic provider review outcome unknown')
        checked, run = self.start(self.request(self.sources[0], run_id='review-timeout'))
        self.assertEqual(run['status'], 'needs_attention')
        self.assertEqual(run['photo_edit_session']['status'], 'needs_attention')
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertEqual(len(self.image_requests), 1)
        self.assertEqual(len(self.vision_requests), 4)
        exported = export_source_run(self.root, run['run_id'], directory=str(self.root/'exports'), artifact='draft', format='TIFF16')
        self.assertTrue(exported['ok'])
        self.assertFalse(exported['metadata']['approved'])
        self.assertEqual(len(self.vision_requests), 4)
        self.assertEqual(len(self.image_requests), 1)

    def test_source_preset_pixels_cannot_overwrite_generated_light_and_frozen_assets_cannot_change(self):
        historical_preset = {'id': 'synthetic-legacy-preset', 'name': 'Synthetic old source appearance',
            'prompt': 'User selected photographic treatment', 'denoise_strength': .8,
            'appearance': {'exposure': 3.0, 'saturation': 0.0},
            'manual_layers': {'synthetic_source_rgb_override': {'enabled': True}}}
        checked, run = self.start(self.request(self.sources[0], review=False, run_id='preset-freeze',
                                               preset_snapshot=historical_preset))
        self.assertEqual(run['options']['denoise_strength'], 0)
        self.assertFalse(run['options'].get('preset'))
        self.assertFalse(run['options'].get('manual_layers'))
        active_appearance = run['options'].get('appearance_revision') or {}
        self.assertFalse({key: value for key, value in active_appearance.items() if key != 'workflow_graph'})
        self.assertEqual(self.image_requests[0]['prompt'], run['reference']['prompt'])
        frozen = run['options']['photo_appearance']
        receipt_path = Path(frozen['path'])/'receipt.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf8'))
        self.assertFalse(receipt['native_generated_detail'])
        self.assertFalse(receipt['human_approved'])
        appearance = receipt_path.parent/receipt['files']['master']['file']
        original = appearance.read_bytes()
        appearance.write_bytes(original[:-1]+bytes([original[-1]^1]))
        with self.assertRaises(ValueError):
            export_source_run(self.root, run['run_id'], directory=str(self.root/'exports'), artifact='draft', format='PNG')
        self.assertEqual(len(self.image_requests), 1)
        self.assertEqual(len(self.vision_requests), 2)

    def test_deferred_scoped_selection_can_switch_completed_child_without_stale_pixels_or_approval(self):
        import single_source_workflow as workflow
        request = self.request(self.sources[0], run_id='deferred-scoped', variant_count=2,
                               defer_reference_selection=True)
        # This explicit multi-candidate task uses the same scoped editor; the
        # older enhance_source control deliberately offers just one candidate.
        request['creative_task']['constraints'].pop('background_mode')
        original_request = self.image_request
        def distinct_candidates(*args, **kwargs):
            payload, pixels, mode = original_request(*args, **kwargs)
            if len(self.image_requests) == 2:
                with Image.open(BytesIO(pixels)) as image:
                    values = np.asarray(image.convert('RGB')).copy()
                values[:, :, 1] = np.minimum(values[:, :, 1].astype(np.int16)+70, 255).astype(np.uint8)
                stream = BytesIO(); Image.fromarray(values).save(stream, format='PNG'); pixels = stream.getvalue()
            return payload, pixels, mode
        self.generation.side_effect = distinct_candidates
        with (patch('beautygrpo_prepare.prepare_for_workbench', side_effect=AssertionError('Scoped selection ran Beauty')) as beauty,
              patch('beautygrpo_prepare.begin_for_workbench', side_effect=AssertionError('Scoped selection began Beauty')) as begin_beauty,
              patch('design_reconstruction_pipeline.prepare_scene', side_effect=AssertionError('Scoped selection ran source T/B/S')) as tbs):
            checked, parent = self.start(request)
            self.assertEqual(parent['status'], 'awaiting_reference')
            self.assertEqual(parent['reference_role'], 'candidate_pending_selection')
            self.assertEqual(len(self.image_requests), 2)
            self.assertEqual(len(self.vision_requests), 6)
            self.assertEqual(len(parent['variants']), 2)
            expected_design = parent['generation_design']['observed_design']['design_id']
            for variant in parent['variants']:
                self.assertEqual(variant['generation_design']['observed_design']['design_id'], expected_design)
                self.assertEqual(variant['prompt'], variant['reference']['prompt'])
            parent_path = self.root/'outputs/runs'/parent['run_id']/'manifest.json'
            parent_bytes = parent_path.read_bytes()
            selected = self.backend.select_source_variant(parent['run_id'], {'variant_id': 'v1', 'async': False})
            self.assertTrue(selected['ok']); first = selected['run']
            self.assertEqual(first['selected_variant_id'], 'v1')
            self.assertEqual(first['status'], 'needs_visual_review')
            first_pixels = Path(first['artifacts']['auto_draft']).read_bytes()
            self.assertEqual(parent_path.read_bytes(), parent_bytes)
            self.assertFalse(first['photo_edit_session']['human_approved'])
            first_session = first['photo_edit_session']['session_id']

            # Deliberate synthetic prior-review fixture: a new version must not
            # inherit either an independent result or a saved human disposition.
            first_path = self.root/'outputs/runs'/first['run_id']/'manifest.json'
            first['photo_edit_session'] = {'status': 'stopped', 'automated_target_met': True,
                'human_approved': False, 'session_id': 'synthetic-prior-review-only'}
            first['review'].update(status='accepted', disposition='accepted', reviewer='Synthetic fixture; no real artistic approval',
                                   reviewed_at='2026-01-01T00:00:00Z', checks={key: 'pass' for key in workflow.VISUAL_CHECK_KEYS})
            first['status'] = 'accepted'
            workflow.write_json(first_path, first)
            first_bytes = first_path.read_bytes()
            switched = self.backend.select_source_variant(first['run_id'], {'variant_id': 'v2', 'async': False})
            self.assertTrue(switched['ok']); second = switched['run']
            self.assertEqual(second['parent_run_id'], first['run_id'])
            self.assertEqual(second['selected_variant_id'], 'v2')
            self.assertEqual(second['status'], 'needs_visual_review')
            self.assertNotEqual(Path(second['artifacts']['auto_draft']).read_bytes(), first_pixels)
            self.assertEqual(second['generation_design']['observed_design']['design_id'], expected_design)
            self.assertNotEqual(second['photo_edit_session']['session_id'], first_session)
            self.assertNotEqual(second['photo_edit_session']['session_id'], 'synthetic-prior-review-only')
            self.assertFalse(second['photo_edit_session']['human_approved'])
            self.assertFalse(second['photo_edit_session']['automated_target_met'])
            observed = second['photo_edit_session']['independent_evaluation']['observed_images']
            self.assertEqual(observed[1], {'role': 'candidate', 'sha256': sha256_file(second['artifacts']['auto_draft'])})
            self.assertNotEqual(second['artifacts']['photo_edit_session'], first['artifacts']['photo_edit_session'])
            self.assertEqual(second['review']['status'], 'needs_review')
            self.assertIsNone(second['review']['reviewer'])
            self.assertTrue(all(value == 'not_checked' for value in second['review']['checks'].values()))
            frozen = second['options']['photo_appearance']
            self.assertNotEqual(frozen['path'], first['options']['photo_appearance']['path'])
            receipt = json.loads((Path(frozen['path'])/'receipt.json').read_text(encoding='utf8'))
            candidate_v2 = Path(parent['variants'][1]['candidate_path'])
            self.assertEqual(receipt['files']['preview']['sha256'], sha256_file(candidate_v2))
            self.assertEqual(second['reference']['sha256'], sha256_file(candidate_v2))
            self.assertNotEqual(second['reference']['sha256'], first['reference']['sha256'])
            self.assertFalse(receipt['human_approved'])
            self.assertEqual(first_path.read_bytes(), first_bytes)
            self.assertEqual(parent_path.read_bytes(), parent_bytes)
            self.assertEqual(len(self.image_requests), 2, 'Selecting a saved candidate generated another image')
            self.assertEqual(len(self.vision_requests), 6, 'Saved initial sessions already own baseline, decisions and rechecks')
            self.assertFalse(beauty.called or begin_beauty.called or tbs.called)
            exported = export_source_run(self.root, second['run_id'], directory=str(self.root/'exports'), artifact='draft', format='PNG')
            self.assertTrue(exported['ok'])
            self.assertFalse(exported['metadata']['approved'])
            self.assertEqual(Path(exported['path']).read_bytes(), Path(second['artifacts']['auto_draft']).read_bytes())


if __name__ == '__main__':
    unittest.main()
