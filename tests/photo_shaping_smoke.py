"""Public checked shaping, real typed pixels and exact restore; model IO fake.

Synthetic findings verify orchestration and ownership, never natural appearance
or human acceptance. All requests use an isolated temporary workspace.
"""
import copy
from io import BytesIO
import json
from pathlib import Path
import shutil
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from asset_identity import sha256_file
from creative_task_smoke import snapshot
from photo_shaping import normalize_request
import scoped_production_smoke as production_fixture


class ShapingTests(unittest.TestCase):
    def setUp(self):
        self.fx = production_fixture.ScopedProductionTests(methodName='runTest')
        self.fx.setUp(); self.addCleanup(self.fx.doCleanups)
        self.root = self.fx.root; self.phase = 'parent'; self.failure = None
        # A synthetic canonical face has enough native pixels for meaningful
        # hard-protection checks; detector execution remains the only mock.
        yy,xx=np.indices((960,640),dtype=np.float32)
        gradient=np.stack((45+130*xx/640,55+120*yy/960,65+100*(xx+yy)/1600),axis=-1)
        Image.fromarray(np.uint8(gradient)).save(self.fx.sources[0])
        self.fx.decision_tool='edit_region'
        registry=json.loads(self.fx.registry.read_text('utf8'))
        metadata=Path(registry['face_python']).parent.parent/'Lib/site-packages/mediapipe-0.10.20.dist-info/METADATA'
        metadata.parent.mkdir(parents=True,exist_ok=True)
        metadata.write_text('Name: mediapipe\nVersion: 0.10.20\n',encoding='utf8')
        self.detector_calls=[];self.shape_alpha=False
        self.fx.stack.enter_context(patch('photo_shaping_geometry._run_detector',side_effect=self.detect_geometry))
        self.conditioning = []; self.generated = []
        self.cancel_after_image = False; self.cancel_job_id = None; self.job_available = threading.Event()
        self.addCleanup(self.job_available.set)
        self.fx.vision.side_effect = self.vision
        self.fx.generation.side_effect = self.generate
        _checked, self.parent = self.fx.start(self.fx.request(self.fx.sources[0], run_id='shape-U'))
        self.parent_path = self.root/'outputs/runs'/self.parent['run_id']/'manifest.json'
        self.parent_bytes = self.parent_path.read_bytes()
        self.u = copy.deepcopy(self.parent['options']['photo_appearance'])
        self.phase = 'shape'

    def detect_geometry(self,runtime,preview,output,cancel):
        from photo_shaping_geometry_smoke import synthetic_points
        with Image.open(preview) as picture:size=list(picture.size)
        self.detector_calls.append(sha256_file(preview))
        payload={'source_native_size':size,'source_sha256':sha256_file(preview),
            'faces':[{'landmarks':synthetic_points(size=size,scale=.7).tolist()}],
            'method':'mediapipe_offline_multi_face_landmarks','model_sha256':runtime['model_sha256'],
            'mediapipe_version':runtime['mediapipe_version']}
        (output/'detections.json').write_text(json.dumps(payload),encoding='utf8')

    def vision(self, source, output, prompt, **kwargs):
        payload, provenance = self.fx.vision_request(source, output, prompt, **kwargs)
        if self.phase=='parent' and Path(source).name=='review-board.png' and 'decision' in str(output):
            payload['region']=[.4,.3,.65,.65]
        if self.phase == 'shape' and Path(source).name == 'review-board.png':
            if 'decision' in str(output):
                state_path = next(parent/'editing-session/session.json' for parent in Path(output).parents
                    if (parent/'editing-session/session.json').is_file())
                context=json.loads(state_path.read_text('utf8'))['attempts'][-1]['context']
                from photo_shaping import shaping_tool
                auth=context['shaping_authorization'];parts=auth['request']['parts'];tool=shaping_tool(auth)
                payload = {'action': 'tool', 'tool': tool,
                    'operations':['shape_'+part for part in ('face','waist','arms','legs') if part in parts],
                    'instruction': 'Apply only the checked selected contour adjustment while preserving protected features.',
                    'reason': 'Execute the explicit independently checked shaping request.'}
                if tool=='edit_region':payload['region']=[.25,.25,.75,.75]
            else:
                payload['credibility'] = {'status': 'pass', 'findings': ['Synthetic protected features remain unchanged.']}
                payload['creative_benefit'] = {'status': 'pass', 'findings': ['Synthetic original creative treatment remains visible.']}
                payload['primary_issue'] = 'none'
                state_path = next(parent/'editing-session/session.json' for parent in Path(output).parents
                    if (parent/'editing-session/session.json').is_file())
                state = json.loads(state_path.read_text('utf8'))
                context = state['attempts'][-1]['context']
                if not context.get('shaping_baseline'):
                    payload['shaping_fulfillment'] = {'status': 'pass', 'findings': ['Synthetic selected contour adjustment is visible.']}
        return payload, provenance

    def generate(self, *args, **kwargs):
        if self.phase == 'shape':
            image = Path(args[2]); self.conditioning.append({'sha256': sha256_file(image), 'pixels': np.asarray(Image.open(image)).copy()})
            if self.failure is not None:
                self.fx.image_requests.append({'image': str(image), 'model': args[3], 'prompt': args[4]})
                raise self.failure
        result, pixels, mode = self.fx.image_request(*args, **kwargs)
        if self.phase == 'shape':
            with Image.open(BytesIO(pixels)) as image:
                values = np.asarray(image.convert('RGB')).copy()
            values[:, :, 0] = np.uint8(35 + len(self.conditioning)*20)
            self.generated.append(values.copy())
            if self.shape_alpha:
                alpha=np.zeros(values.shape[:2],np.uint8);alpha[:,:alpha.shape[1]//2]=128
                values=np.concatenate((values,alpha[...,None]),axis=-1)
            stream = BytesIO(); Image.fromarray(values).save(stream, format='PNG'); pixels = stream.getvalue()
            if self.cancel_after_image:
                if not self.job_available.wait(5):
                    raise AssertionError('Checked asynchronous task did not expose its job')
                self.fx.backend.cancel_job(self.cancel_job_id)
        return result, pixels, mode

    def request(self, *, parent=None, run_id='shape-A', amount=.3, restore=False, part='waist'):
        parent = parent or self.parent
        body = self.fx.request(self.fx.sources[0], run_id=run_id)
        body.pop('editing_autofinish')
        body['creative_task']['mode'] = 'existing_reference'
        body['creative_task']['services'] = {'text_vision': not restore, 'visual_review': not restore}
        base = (parent['options'].get('shaping_state') or {}).get('base_appearance') or parent['options']['photo_appearance']
        body.update(parent_run_id=parent['run_id'], reference=str(Path(base['path'])/'preview.png'),
            shaping_request={'action': 'restore'} if restore else {'action': 'apply', 'parts': {part: amount}})
        return body

    def plan(self, body):
        before = snapshot(self.root); images = len(self.fx.image_requests); vision = len(self.fx.vision_requests)
        checked = self.fx.backend.task_plan(body)
        self.assertTrue(checked['plan']['ready'], checked['plan'].get('blocking_reasons'))
        self.assertEqual(snapshot(self.root), before, 'Checked planning changed the workspace')
        self.assertEqual(len(self.fx.image_requests), images); self.assertEqual(len(self.fx.vision_requests), vision)
        return checked

    def start(self, body):
        checked = self.plan(body)
        response = self.fx.backend.start_task({**checked['request'], 'plan_id': checked['plan']['plan_id'], 'async': False})
        self.assertTrue(response.get('ok'), {'stderr': response.get('stderr'), 'run': response.get('run')})
        return checked, response['run']

    def binding(self, run):
        return {'source_sha256': run['source']['sha256'], 'scope': run['task_plan']['editing_scope'],
            'design_id': run['generation_design']['observed_design']['design_id']}

    def test_request_values_are_strict_and_zero_cannot_implicitly_enable(self):
        self.assertNotIn('shaping_state', self.parent['options'])
        self.assertNotIn('shaping_request', self.parent['task_plan'])
        self.assertEqual(normalize_request({'action': 'apply', 'parts': {'face': .3, 'waist': 0}}),
            {'action': 'apply', 'parts': {'face': .3}})
        for value in (True, False, float('nan'), float('inf'), -.1, 1.01, '0.3', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_request({'action': 'apply', 'parts': {'face': value}})
        for value in ({'action': 'apply', 'parts': {}}, {'action': 'apply', 'parts': {'face': 0}},
                      {'action': 'restore', 'parts': {}}, {'action': 'apply', 'parts': {'nose': .2}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_request(value)

    def test_public_face_A_B_use_same_U_zero_generation_and_transport_G_with_exact_P_J0(self):
        from appearance_assets import load_asset
        from scipy.ndimage import map_coordinates
        from workbench_store import select_saved_result, saved_result_state
        select_saved_result(self.root,self.parent['run_id'],explicit=True)
        images=len(self.fx.image_requests);visions=len(self.fx.vision_requests)
        checked,a=self.start(self.request(part='face'))
        self.assertEqual(checked['plan']['image_budget']['shaping'],0)
        self.assertEqual(checked['plan']['services']['text_vision_requests_upper_bound'],3)
        self.assertEqual(len(self.fx.image_requests),images);self.assertEqual(len(self.fx.vision_requests)-visions,3)
        self.assertEqual(a['result_publication']['state'],'committed');self.assertTrue(a['shaping_result']['completed'])
        self.assertTrue(a['options']['preserve_project_selection'])
        self.assertEqual(saved_result_state(self.root)['selected_saved_run_id'],self.parent['run_id'])
        select_saved_result(self.root,a['run_id'],explicit=True)
        a_path=self.root/'outputs/runs'/a['run_id']/'manifest.json';a_bytes=a_path.read_bytes()
        _checked,b=self.start(self.request(parent=a,run_id='shape-B',amount=.6,part='face'))
        self.assertEqual(len(self.fx.image_requests),images);self.assertEqual(len(self.fx.vision_requests)-visions,6)
        self.assertEqual(self.conditioning,[],'CPU face deformation called the generator')
        self.assertEqual(saved_result_state(self.root)['selected_saved_run_id'],a['run_id'])
        self.assertNotEqual(a['options']['photo_appearance']['asset_id'],b['options']['photo_appearance']['asset_id'])
        moved_G=False
        for current in (a,b):
            self.assertEqual(current['options']['shaping_state']['base_appearance']['asset_id'],self.u['asset_id'])
            baseline=load_asset(current['options']['shaping_state']['base_appearance'],**self.binding(current))
            candidate=load_asset(current['options']['photo_appearance'],**self.binding(current))
            receipt=candidate['receipt'];self.assertEqual(receipt['kind'],'geometric_transform')
            self.assertIsNone(receipt['generation']);self.assertEqual(receipt['operation']['image_generation_submissions'],0)
            self.assertEqual(receipt['parent']['asset_id'],self.u['asset_id'])
            operation=receipt['operation'];metadata=operation['mapping_metadata'];x0,y0,x1,y1=operation['box']
            with np.load(Path(current['options']['photo_appearance']['path'])/operation['mapping']['file'],allow_pickle=False) as mapping:
                protected=mapping['protected'];outside=~mapping['allowed'];displacement=mapping['displacement']
                edited=candidate['master'][y0:y1,x0:x1];original=baseline['master'][y0:y1,x0:x1]
                np.testing.assert_array_equal(edited[protected|outside],original[protected|outside])
                np.testing.assert_array_equal(candidate['edit_coverage'][y0:y1,x0:x1],mapping['edit_support'])
                self.assertTrue(np.any(mapping['edit_support']));self.assertTrue(np.all(displacement[protected|outside]==0))
                G=baseline['generated_coverage']
                if type(G) is int:G=np.full(baseline['master'].shape[:2],G,np.float32)
                actual=candidate['generated_coverage']
                if type(actual) is int:actual=np.full(baseline['master'].shape[:2],actual,np.float32)
                yy,xx=np.indices(displacement.shape,dtype=np.float64);h=np.asarray(metadata['h'],np.float64)
                expected=map_coordinates(G,[yy+y0+h[1]*displacement,xx+x0+h[0]*displacement],order=1,prefilter=False,mode='nearest')
                np.testing.assert_allclose(actual[y0:y1,x0:x1],expected,rtol=0,atol=2e-7)
                np.testing.assert_array_equal(actual[y0:y1,x0:x1][protected|outside],G[y0:y1,x0:x1][protected|outside])
                moved_G|=bool(np.any(actual!=G))
            state=json.loads(Path(current['photo_edit_session']['path']).read_text('utf8'))
            self.assertEqual([row['kind'] for row in state['attempts']],['evaluate','decide','tool','evaluate'])
            self.assertEqual(state['candidates'][0]['appearance_asset']['asset_id'],self.u['asset_id'])
            decision=next(row['context']['decision'] for row in state['attempts'] if row['kind']=='tool')
            self.assertEqual(decision['tool'],'face_contour_deform');self.assertNotIn('region',decision)
            provenance=state['attempts'][-1]['response']['provenance']
            self.assertEqual(provenance['review_inspection_scope'],'whole_image_and_edit_region')
            self.assertEqual(provenance['input_profile'],'detailed_board')
            observation=provenance['regional_observation']
            self.assertEqual(observation['core_box'],operation['box'])
            self.assertEqual(observation['candidate_asset_id'],current['options']['photo_appearance']['asset_id'])
            self.assertEqual(observation['parent_asset_id'],self.u['asset_id'])
            self.assertEqual(observation['operation_kind'],'geometric_transform')
            self.assertIsNone(observation['generation_context_box'])
            self.assertFalse(provenance['native_detail_acceptance'])
        self.assertTrue(moved_G,'Public fixture must exercise a genuinely nonuniform ownership transport')
        self.assertEqual(self.detector_calls,[sha256_file(Path(self.u['path'])/'preview.png')])
        self.assertEqual(a_path.read_bytes(),a_bytes);self.assertEqual(self.parent_path.read_bytes(),self.parent_bytes)

    def test_face_and_body_combination_is_rejected_during_planning_without_calls(self):
        body=self.request(part='face');body['shaping_request']['parts']['waist']=.2
        before=snapshot(self.root);counts=(len(self.fx.image_requests),len(self.fx.vision_requests),len(self.detector_calls))
        try:
            result=self.fx.backend.task_plan(body)
            self.assertFalse(result['plan']['ready'])
        except ValueError:
            pass
        self.assertEqual(snapshot(self.root),before)
        self.assertEqual((len(self.fx.image_requests),len(self.fx.vision_requests),len(self.detector_calls)),counts)

    def test_body_generation_preserves_true_alpha_and_premultiplied_RGB(self):
        from appearance_assets import load_asset
        from photo_session_evidence import resolve_file
        from photo_edit_tools import _generated_pixels,_mask
        self.shape_alpha=True
        checked,current=self.start(self.request())
        self.assertEqual(checked['plan']['image_budget']['shaping'],1)
        state=json.loads(Path(current['photo_edit_session']['path']).read_text('utf8'))
        provenance=state['candidates'][-1]['provenance']
        composition=json.loads(Path(resolve_file(state,provenance['composition_receipt'])['path']).read_text('utf8'))
        x0,y0,x1,y1=composition['core_box'];bx0,by0,bx1,by1=composition['context_box']
        generated=resolve_file(state,provenance['generated_image'])['path']
        _rgb,associated,alpha=_generated_pixels(Path(generated),(bx1-bx0,by1-by0))
        self.assertIsNotNone(alpha)
        baseline=load_asset(current['options']['shaping_state']['base_appearance'],**self.binding(current))
        changed=load_asset(current['options']['photo_appearance'],**self.binding(current))
        f=_mask((x1-x0,y1-y0));local=(slice(y0-by0,y1-by0),slice(x0-bx0,x1-bx0));c=f*alpha[local]
        expected=baseline['master'][y0:y1,x0:x1]*(1-c[...,None])+associated[local]*f[...,None]
        np.testing.assert_array_equal(changed['edit_coverage'][y0:y1,x0:x1],c)
        np.testing.assert_array_equal(changed['master'][y0:y1,x0:x1][c>0],expected[c>0])
        np.testing.assert_array_equal(changed['master'][y0:y1,x0:x1][c==0],baseline['master'][y0:y1,x0:x1][c==0])

    def test_missing_geometry_stops_before_vision_or_generation_and_preserves_parent(self):
        def no_face(runtime,preview,output,cancel):
            self.detect_geometry(runtime,preview,output,cancel)
            path=output/'detections.json';data=json.loads(path.read_text('utf8'));data['faces']=[]
            path.write_text(json.dumps(data),encoding='utf8')
        checked=self.plan(self.request(part='face'))
        before=(len(self.fx.image_requests),len(self.fx.vision_requests))
        with patch('photo_shaping_geometry._run_detector',side_effect=no_face):
            result=self.fx.backend.start_task({**checked['request'],'plan_id':checked['plan']['plan_id'],'async':False})
        self.assertFalse(result.get('ok'),result)
        self.assertEqual((len(self.fx.image_requests),len(self.fx.vision_requests)),before)
        self.assertEqual(self.parent_path.read_bytes(),self.parent_bytes)

    def test_unadopted_candidate_does_not_become_startup_default_without_explicit_prior_selection(self):
        from backend_server import Backend
        from workbench_store import saved_result_state
        before = saved_result_state(self.root)
        self.assertEqual(before['latest_saved_run_id'], self.parent['run_id'])
        self.assertFalse(before['selection_explicit'])
        _checked, candidate = self.start(self.request())
        self.assertEqual(candidate['result_publication']['state'], 'committed')
        self.assertEqual(saved_result_state(self.root), before)
        Backend(self.root, embedded=True)
        self.assertEqual(saved_result_state(self.root), before)

    def test_state_rejects_other_typed_kinds_and_regional_edits_accumulated_from_A(self):
        import photo_appearance as appearance
        import photo_development as development
        from photo_edit_tools import _close_loaded
        from photo_shaping import validate_state
        _checked, a = self.start(self.request())
        binding = self.binding(a); state = a['options']['shaping_state']
        u = appearance.load_asset(state['base_appearance'], **binding)
        current = appearance.load_asset(a['options']['photo_appearance'], **binding)
        def generation(asset, receipt):
            value = copy.deepcopy(receipt['generation'])
            for key in ('image', 'provider_receipt'):
                value[key] = {'path': str(Path(asset['path'])/value[key]['file']), 'sha256': value[key]['sha256']}
            return value
        try:
            parameters = {'adjustments': [{'region': [0, 0, 1, 1], 'exposure_ev': .1,
                'warmth_ev': 0, 'tint_ev': 0, 'feather': 0}]}
            values, coverage, stats = development.develop_float(u['master'], parameters)
            photographic = appearance.freeze_candidate(self.root/'shape-invalid-photo', kind='photographic_adjustment',
                master=values, parent=state['base_appearance'], coverage=coverage,
                photographic_operation={'operator': development.operator_for(parameters), 'stats': stats,
                    'producer': development.producer_identity()}, **binding)
            full = appearance.freeze_candidate(self.root/'shape-invalid-full', kind='generated_full',
                master=u['master'].copy()+np.float32(.001), parent=state['base_appearance'],
                generation={**generation(state['base_appearance'], u['receipt']),'core_box':None,'context_box':None}, **binding)
            values = current['master'].copy(); coverage = np.zeros(values.shape[:2], np.float32)
            x0, y0, x1, y1 = current['receipt']['generation']['core_box']
            coverage[y0:y1, x0:x1] = .5; values[coverage > 0] *= np.float32(.995)
            accumulated = appearance.freeze_candidate(self.root/'shape-invalid-accumulated', kind='mixed_regional',
                master=values, parent=a['options']['photo_appearance'], coverage=coverage,
                generation=generation(a['options']['photo_appearance'], current['receipt']), **binding)
            for asset in (a['initial_source_appearance'], photographic, full, accumulated):
                receipt = appearance.validate_asset(asset, **binding)
                with self.subTest(kind=receipt['kind'], parent=receipt.get('parent')):
                    changed = copy.deepcopy(state); changed['selected_asset_id'] = asset['asset_id']
                    with self.assertRaises(ValueError):
                        validate_state(changed, current=asset, **binding)
            validate_state(state, current=a['options']['photo_appearance'], **binding)
        finally:
            _close_loaded(u); _close_loaded(current)

    def test_restore_zero_services_returns_exact_owned_U_and_no_inherited_approval(self):
        from appearance_assets import load_asset
        _checked, a = self.start(self.request())
        images = len(self.fx.image_requests); visions = len(self.fx.vision_requests)
        body = self.request(parent=a, run_id='shape-off', restore=True)
        for field in ('provider', 'model', 'vision_model', 'api_base', 'api_key_env'):
            body.pop(field, None)
        with patch('photo_request_budget.manual_provider_target', side_effect=AssertionError('Restore resolved credentials')), \
             patch('raw_support.open_linear', side_effect=AssertionError('Restore decoded original')):
            checked, restored = self.start(body)
        self.assertEqual(checked['plan']['image_budget']['shaping'], 0)
        self.assertEqual(checked['plan']['services']['text_vision_requests_upper_bound'], 0)
        self.assertEqual(len(self.fx.image_requests), images); self.assertEqual(len(self.fx.vision_requests), visions)
        state = restored['options']['shaping_state']; self.assertFalse(state['enabled'])
        original = load_asset(state['base_appearance'], **self.binding(restored))
        current = load_asset(restored['options']['photo_appearance'], **self.binding(restored))
        self.assertTrue(np.array_equal(original['master'], current['master']))
        self.assertEqual(sha256_file(restored['artifacts']['auto_draft']), sha256_file(original['preview_path']))
        self.assertEqual(original['receipt']['edit_coverage'], current['receipt']['edit_coverage'])
        self.assertEqual(original['receipt']['generated_coverage'], current['receipt']['generated_coverage'])
        self.assertIsNone(restored['review']['reviewer']); self.assertFalse(restored['shaping_result']['human_approved'])
        self.assertEqual(restored['result_publication']['state'], 'committed')
        _checked, reapplied = self.start(self.request(parent=restored, run_id='shape-on-again', amount=.5))
        self.assertEqual(reapplied['options']['shaping_state']['base_appearance']['asset_id'], self.u['asset_id'])
        self.assertEqual(self.conditioning[0]['sha256'], self.conditioning[1]['sha256'])

    def test_face_restore_uses_exact_U_without_new_warp_detector_generation_or_vision(self):
        import face_contour_deform
        from appearance_assets import load_asset, clear_proof_cache, proof_cache_info, validate_asset
        _checked,current=self.start(self.request(part='face'))
        original=load_asset(current['options']['shaping_state']['base_appearance'],**self.binding(current))
        expected_master=original['master'].copy();expected_G=copy.deepcopy(original['generated_coverage'])
        sample=face_contour_deform.sample_mapping;validation_calls=[]
        def validate_saved_A(master,G,mapping,**kwargs):
            np.testing.assert_array_equal(master,expected_master)
            np.testing.assert_array_equal(G,expected_G)
            self.assertEqual(mapping['amount'],current['options']['shaping_state']['request']['parts']['face'])
            validation_calls.append(mapping['mapping_id'])
            return sample(master,G,mapping,**kwargs)
        counts=(len(self.fx.image_requests),len(self.fx.vision_requests),len(self.detector_calls))
        body=self.request(parent=current,run_id='face-exact-off',restore=True)
        for key in ('provider','model','vision_model','api_base','api_key_env'):body.pop(key,None)
        # Force the cold reopen boundary: saved A must reconstruct from U once.
        # A warm mathematical proof may subsequently be reused only after a
        # fresh actual-byte audit; requiring a new warp on every read defeats it.
        clear_proof_cache()
        with patch('face_contour_deform.sample_mapping',side_effect=validate_saved_A),\
                patch('photo_shaping_geometry._run_detector',side_effect=AssertionError('Restore ran detector')),\
                patch('vision_transport.request_vision_json',side_effect=AssertionError('Restore called vision')),\
                patch('reference_provider.request_edit',side_effect=AssertionError('Restore generated an image')),\
                patch('raw_support.open_linear',side_effect=AssertionError('Restore decoded original')):
            checked,restored=self.start(body)
        self.assertTrue(validation_calls,'Saved A must still pass strict typed validation')
        before=proof_cache_info()
        with patch('face_contour_deform.sample_mapping',side_effect=AssertionError('Warm restore proof was reconstructed')):
            validate_asset(current['options']['photo_appearance'],**self.binding(current))
        after=proof_cache_info()
        self.assertGreater(after['hits'],before['hits'])
        self.assertGreater(after['bytes_hashed'],before['bytes_hashed'])
        self.assertEqual(checked['plan']['image_budget']['shaping'],0)
        self.assertEqual(checked['plan']['services']['text_vision_requests_upper_bound'],0)
        self.assertEqual((len(self.fx.image_requests),len(self.fx.vision_requests),len(self.detector_calls)),counts)
        u=load_asset(restored['options']['shaping_state']['base_appearance'],**self.binding(restored))
        result=load_asset(restored['options']['photo_appearance'],**self.binding(restored))
        np.testing.assert_array_equal(result['master'],u['master'])
        np.testing.assert_array_equal(result['generated_coverage'],u['generated_coverage'])
        self.assertEqual(sha256_file(restored['artifacts']['auto_draft']),sha256_file(u['preview_path']))
        self.assertFalse(restored['options']['shaping_state']['enabled']);self.assertIsNone(restored['review']['reviewer'])

    def test_unknown_second_attempt_preserves_A_and_cannot_reset_target_with_another_run(self):
        from workbench_store import select_saved_result, saved_result_state
        _checked, a = self.start(self.request())
        select_saved_result(self.root, a['run_id'], explicit=True)
        path = self.root/'outputs/runs'/a['run_id']/'manifest.json'; before = path.read_bytes()
        self.failure = TimeoutError('Synthetic external image outcome remains unknown')
        body = self.request(parent=a, run_id='shape-unknown', amount=.6)
        checked = self.plan(body)
        response = self.fx.backend.start_task({**checked['request'], 'plan_id': checked['plan']['plan_id'], 'async': False})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(saved_result_state(self.root)['selected_saved_run_id'], a['run_id'])
        self.assertEqual((response.get('run') or {})['result_publication']['state'], 'failed')
        count = len(self.fx.image_requests)
        body['run_id'] = 'shape-new-id-same-target'
        try:
            plan = self.fx.backend.task_plan(body)
            self.assertFalse(plan['plan']['ready'])
        except ValueError:
            pass
        self.assertEqual(len(self.fx.image_requests), count)
        self.assertEqual(path.read_bytes(), before)

    def test_changed_strength_after_plan_is_rejected_before_model_submission(self):
        checked = self.plan(self.request())
        count = (len(self.fx.image_requests), len(self.fx.vision_requests))
        body = copy.deepcopy(checked['request']); body['shaping_request']['parts']['waist'] = .7
        try:
            response = self.fx.backend.start_task({**body, 'plan_id': checked['plan']['plan_id'], 'async': False})
            self.assertFalse(response.get('ok'))
        except ValueError:
            pass
        self.assertEqual((len(self.fx.image_requests), len(self.fx.vision_requests)), count)

    def test_actual_original_unknown_ledger_blocks_even_with_committed_parent_summary(self):
        ledger = self.root/'outputs/request-budgets'/self.parent['run_id']/'vision.json'
        original = ledger.read_bytes(); value = json.loads(original)
        self.assertTrue(value['operations'])
        value['operations'][-1]['status'] = 'unknown'
        ledger.write_text(json.dumps(value), encoding='utf8')
        counts = (len(self.fx.image_requests), len(self.fx.vision_requests))
        try:
            try:
                checked = self.fx.backend.task_plan(self.request())
                self.assertFalse(checked['plan']['ready'])
            except ValueError:
                pass
        finally:
            ledger.write_bytes(original)
        self.assertEqual((len(self.fx.image_requests), len(self.fx.vision_requests)), counts)
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)

    def test_late_cancelled_B_keeps_explicit_A_selection_and_parent_pixels(self):
        from workbench_store import select_saved_result, saved_result_state
        _checked, a = self.start(self.request())
        select_saved_result(self.root, a['run_id'], explicit=True)
        parent = self.root/'outputs/runs'/a['run_id']/'manifest.json'; before = parent.read_bytes()
        image_before = len(self.fx.image_requests)
        self.cancel_after_image = True
        checked = self.plan(self.request(parent=a, run_id='shape-cancelled-B', amount=.6))
        response = self.fx.backend.start_task({**checked['request'], 'plan_id': checked['plan']['plan_id'], 'async': True})
        self.cancel_job_id = response['job_id']; self.job_available.set()
        deadline = time.monotonic()+15
        while time.monotonic() < deadline:
            job = self.fx.backend.jobs.get(self.cancel_job_id)
            if job['status'] in {'failed', 'blocked', 'cancelled', 'succeeded'}:
                break
            time.sleep(.01)
        self.assertEqual(job['status'], 'cancelled')
        worker = self.fx.backend._job_workers.get(self.cancel_job_id)
        if worker is not None:
            worker.join(5); self.assertFalse(worker.is_alive())
        self.assertEqual(len(self.fx.image_requests)-image_before, 1)
        self.assertEqual(parent.read_bytes(), before)
        self.assertEqual(saved_result_state(self.root)['selected_saved_run_id'], a['run_id'])

    def test_portable_current_and_U_restore_after_original_workspace_disappears(self):
        self._portable_roundtrip(include_raw=True)

    def test_portable_without_original_bytes_still_restores_exact_U(self):
        self._portable_roundtrip(include_raw=False)

    def _portable_roundtrip(self, *, include_raw):
        from backend_server import Backend
        from portable_project import export_project, import_project
        _checked, a = self.start(self.request(part='face'))
        expected = sha256_file(Path(self.u['path'])/'preview.png')
        archive = self.root.parent/(self.root.name+'-shape.cosplayproj'); self.addCleanup(archive.unlink, missing_ok=True)
        export_project(self.root, a['run_id'], archive, include_raw=include_raw)
        target = self.root.parent/(self.root.name+'-ported'); self.addCleanup(shutil.rmtree, target, True)
        imported = import_project(target, archive, run_id='ported-shape-A')['run']
        hidden = self.root.with_name(self.root.name+'-unavailable')
        self.root.rename(hidden)
        try:
            from photo_edit_session import read_saved
            from photo_session_evidence import resolve_asset,resolve_file,read_json
            from appearance_assets import validate_asset
            state = (read_saved(Path(imported['photo_edit_session']['path']).parent) if include_raw else
                json.loads(Path(imported['photo_edit_session']['path']).read_text('utf8')))
            self.assertTrue(state['binding'].get('shaping_authorization'))
            for candidate in state['candidates']:
                asset=resolve_asset(state,candidate['appearance_asset'])
                receipt=validate_asset(asset,**self.binding(imported))
                if receipt['kind']=='geometric_transform':
                    for key in ('mapping','support','detections','support_masks'):
                        member=receipt['operation'][key]
                        self.assertEqual(sha256_file(Path(asset['path'])/member['file']),member['sha256'])
                geometry=candidate.get('provenance',{}).get('shaping_geometry')
                if geometry is not None:
                    resolve_file(state,geometry['core_masks'])
                    support=read_json(state,geometry['support'])
                    resolve_file(state,support['detections']);resolve_file(state,support['masks'])
            backend = Backend(target, embedded=True)
            from export_artifact import export_source_run
            before=(len(self.fx.image_requests),len(self.fx.vision_requests),len(self.detector_calls))
            for format,artifact in (('PNG','auto_draft'),('TIFF16','render_master'),('TIFF32_LINEAR','render_scene_master')):
                exported=export_source_run(target,imported['run_id'],directory=str(target/'current-exports'),format=format)
                self.assertEqual(sha256_file(exported['path']),sha256_file(imported['artifacts'][artifact]))
                self.assertFalse(exported['metadata']['approved'])
            body = self.request(parent=imported, run_id='ported-shape-off', restore=True)
            body['source'] = imported['source']['path']
            body['sample'] = imported['sample']
            if not include_raw:
                self.assertFalse(Path(body['source']).exists())
            body.pop('model_registry', None)
            for field in ('provider', 'model', 'vision_model', 'api_base', 'api_key_env'):
                body.pop(field, None)
            with patch('raw_support.open_linear', side_effect=AssertionError('Portable restore decoded original')), \
                 patch('photo_request_budget.manual_provider_target', side_effect=AssertionError('Portable restore resolved credentials')):
                checked = backend.task_plan(body)
                self.assertTrue(checked['plan']['ready'], checked['plan'].get('blocking_reasons'))
                response = backend.start_task({**checked['request'], 'plan_id': checked['plan']['plan_id'], 'async': False})
            self.assertTrue(response.get('ok'), response.get('stderr'))
            self.assertEqual((len(self.fx.image_requests),len(self.fx.vision_requests),len(self.detector_calls)),before)
            restored = response['run']
            self.assertEqual(sha256_file(restored['artifacts']['auto_draft']), expected)
            self.assertTrue(Path(restored['options']['shaping_state']['base_appearance']['path']).is_relative_to(target))
            from export_artifact import export_source_run
            for format, artifact in (('PNG', 'auto_draft'), ('TIFF16', 'render_master'), ('TIFF32_LINEAR', 'render_scene_master')):
                exported = export_source_run(target, restored['run_id'], directory=str(target/'exports'), format=format)
                self.assertEqual(sha256_file(exported['path']), sha256_file(restored['artifacts'][artifact]))
                self.assertFalse(exported['metadata']['approved'])
        finally:
            hidden.rename(self.root)


if __name__ == '__main__':
    unittest.main()
