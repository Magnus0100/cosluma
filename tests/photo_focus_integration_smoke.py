"""Real generated ROI -> reviewed working -> CPU focus -> independent review.

Only image/vision transports are synthetic. Source identity, session budgets,
generation receipts, actual current-best pixels and focus assets are real.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from asset_identity import sha256_file
from photo_appearance import load_asset,validate_asset
from photo_edit_tools import _close_loaded
from photo_edit_finish import drive_session
from photo_edit_tools import make_tools
from photo_focus_integration import PARAMETERS, TOOL, _sigma, integrate_increment, validate_focus_working
from photo_session_decision import decide_images
from photo_session_review import evaluate_images
import photo_edit_session as session
import photo_edit_tools_smoke as tools_fixture


class FocusIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fx = tools_fixture.EditingToolsTests(methodName='runTest'); self.fx.setUp(); self.addCleanup(self.fx.tearDown)
        self.root = self.fx.root; self.stage = 'generation'; self.verdict = 'better'; self.relative_sigma = .02
        self.cancel_calls = 0; self.cancel_threshold = None; self.image_requests = 0; self.vision_requests = []
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('reference_provider.request_edit', side_effect=self.image_transport))
        self.stack.enter_context(patch('vision_transport.request_vision_json', side_effect=self.vision_transport))
        with redirect_stdout(io.StringIO()):
            for _ in range(4):
                session.advance(self.fx.directory, tools=self.fx.tools, decide=self.decide,
                    evaluate=self.evaluate, model_identity='actual-fixture-adapters')
        original = session.snapshot(self.fx.directory)
        self.best = copy.deepcopy(original['candidates'][0]); self.working = copy.deepcopy(original['candidates'][1])
        self.best_context = copy.deepcopy(next(a['context'] for a in original['attempts']
            if a.get('response') == self.best['review']))
        self.working_context = copy.deepcopy(next(a['context'] for a in original['attempts']
            if a.get('response') == self.working['review']))
        self.source = {'path': str(self.fx.source), 'sha256': sha256_file(self.fx.source)}
        self.baseline_bytes = Path(self.best['image']['path']).read_bytes()
        self.working_bytes = Path(self.working['image']['path']).read_bytes()
        self.directory = self.root/'focus-session'
        session.open_session(self.directory, source=self.fx.source, scope=self.fx.scope, design=self.fx.design,
            execution_identity='checked-new-CPU-focus-task', producer_identity='fixture-coordinator',
            budget={'image_submissions': 0, 'vision_requests': 2, 'tool_calls': 1, 'improvement_steps': 1},
            appearance_required=True,source_appearance=self.fx.appearance,
            user_intent={'brief': 'Preserve the complete photograph.',
                         'user_feedback': 'Integrate the generated object with its surrounding photographic focus.'})
        session.adopt_reviewed_candidate(self.directory, self.best['image']['path'], review=self.best['review'],
            provenance={'review_context': self.best_context}, baseline_only=True,appearance_asset=self.best['appearance_asset'])
        session.adopt_reviewed_working_candidate(self.directory, self.working['image']['path'],
            review=self.working['review'], provenance=self.working['provenance'], review_context=self.working_context,appearance_asset=self.working['appearance_asset'])
        self.tools = make_tools(self.root, self.root/'focus-tools', scope=self.fx.scope,
            session_directory=self.directory, cancel=self.cancel)
        self.stage = 'focus'; self.images_before = self.image_requests; self.visions_before = len(self.vision_requests)

    def cancel(self):
        if self.cancel_threshold is None:return False
        self.cancel_calls += 1
        return self.cancel_calls >= self.cancel_threshold

    def decide(self, context):
        return decide_images(context, self.root/'decisions'/context['attempt_id'], provider='fixture')

    def evaluate(self, context):
        return evaluate_images(context, self.root/'reviews'/context['attempt_id'], provider='fixture')

    def vision_transport(self, source, output, prompt, **kwargs):
        source = Path(source); self.vision_requests.append({'stage': self.stage, 'prompt': prompt})
        provenance = {'source': {'path': str(source.resolve()), 'sha256': sha256_file(source)},
            'network_used': True, 'generation_used': False, 'model': 'synthetic-vision-transport',
            'input_profile': kwargs.get('input_profile', 'standard')}
        if 'Select the single most useful available tool' in prompt:
            if self.stage == 'generation':
                return {'action': 'tool', 'tool': 'edit_region', 'operations': ['relight'],
                    'instruction': 'Refine the visible local material, retaining current framing and illumination.',
                    'reason': 'The reviewed material needs a local edit.', 'region': [.25, .25, .75, .75]}, provenance
            self.assertIn('working', prompt)
            return {'action': 'tool', 'tool': TOOL, 'operations': ['detail_repair'],
                'instruction': 'Match the existing generated increment to surrounding photographic focus.',
                'reason': 'The reviewed working material is too sharp.',
                'parameters': {'focus_sigma': self.relative_sigma}}, provenance
        mapping = json.loads((source.parent/'review-board.json').read_text('utf-8'))
        has_previous = len(mapping['images']) == 3
        passed = self.stage == 'focus' and self.verdict == 'better'
        return {'credibility': {'status': 'pass' if passed or not has_previous else 'fail',
            'findings': ['Synthetic geometry is preserved.' if passed or not has_previous else 'Synthetic new material focus is incoherent.']},
            'creative_benefit': {'status': 'pass' if has_previous else 'fail',
                'findings': ['Synthetic generated material is visible.' if has_previous else 'A synthetic useful edit remains.']},
            'comparison': ('better' if passed else 'worse') if has_previous else 'first_candidate',
            'primary_issue': 'none' if passed else 'Integrate the generated material focus.'}, provenance

    def image_transport(self, *args, **kwargs):
        self.image_requests += 1
        payload, _, mode = self.fx.transport(*args, **kwargs)
        with Image.open(args[2]) as input_image:
            yy, xx = np.indices((input_image.height, input_image.width))
        tiles = ((xx//2+yy//2) % 2).astype(np.uint8)
        values = np.stack((30+tiles*170, 70+tiles*90, 50+tiles*110), axis=-1)
        stream = io.BytesIO(); Image.fromarray(values).save(stream, format='PNG')
        return payload, stream.getvalue(), mode

    def proof(self,best=None,working=None,provenance=None,working_asset=None):
        return validate_focus_working(self.source,best or self.best['image'],working or self.working['image'],
            provenance or self.working['provenance'],appearance_assets={'best':self.best['appearance_asset'],
                'working':working_asset or self.working['appearance_asset']},scope=self.fx.scope,design_id=self.fx.design['design_id'])

    def run_focus(self):
        return drive_session(self.directory, tools=self.tools, decide=self.decide, evaluate=self.evaluate,
            model_identity='actual-fixture-adapters', cancel=self.cancel)

    def test_actual_zero_image_focus_uses_working_delta_and_true_lineage(self):
        proof = self.proof()
        driven = self.run_focus(); self.assertIsNone(driven['failure'])
        state = driven['state']; result = driven['selected']
        self.assertEqual([a['kind'] for a in state['attempts']],
                         ['adopted_evaluation', 'adopted_working_evaluation', 'decide', 'tool', 'evaluate'])
        self.assertEqual(state['used']['image_submissions'], 0); self.assertEqual(state['used']['vision_requests'], 2)
        self.assertEqual(self.image_requests, self.images_before)
        self.assertEqual(len(self.vision_requests)-self.visions_before, 2)
        self.assertTrue(state['automated_target_met']); self.assertFalse(state['human_approved'])
        self.assertNotEqual(result['image']['sha256'], self.working['image']['sha256'])
        actual = np.asarray(Image.open(result['image']['path']))
        base = np.asarray(Image.open(self.best['image']['path']))
        working = np.asarray(Image.open(self.working['image']['path']))
        x0,y0,x1,y1 = proof['core_box']; outside = np.ones(actual.shape[:2], bool); outside[y0:y1,x0:x1] = False
        np.testing.assert_array_equal(actual[outside], base[outside])
        def energy(values):return np.abs(np.diff(values[y0+8:y1-8,x0+8:x1-8].astype(float),axis=1)).mean()
        self.assertLess(energy(actual), energy(working)/2)
        self.assertEqual(Path(self.best['image']['path']).read_bytes(), self.baseline_bytes)
        self.assertEqual(Path(self.working['image']['path']).read_bytes(), self.working_bytes)
        receipt = json.loads(Path(result['provenance']['composition_receipt']['path']).read_text('utf-8'))
        self.assertEqual(receipt['image_generation_submissions'], 0)
        self.assertFalse(receipt['focus_operator']['base_blurred'])
        self.assertFalse(receipt['native_generated_detail']); self.assertFalse(receipt['camera_detail_claimed'])
        self.assertEqual(receipt['original_generated_image']['sha256'],proof['generated_image']['sha256'])
        self.assertEqual(self.proof(working=result['image'],provenance=result['provenance'],
            working_asset=result['appearance_asset'])['core_box'],proof['core_box'])
        native=load_asset(result['appearance_asset'],**self.fx.binding)
        prior=load_asset(self.best['appearance_asset'],**self.fx.binding)
        np.testing.assert_array_equal(native['master'][outside],prior['master'][outside])
        self.assertFalse(native['receipt']['native_generated_detail'])
        self.assertEqual(native['receipt']['generated_resolution'],
            validate_asset(self.working['appearance_asset'],**self.fx.binding)['generated_resolution'])
        from raw_support import save_render_preview_and_master,save_scene_master
        values=native['master'].copy();_close_loaded(native,prior)
        preview=save_render_preview_and_master(values,self.root/'focus-master16.tif',display_transform={'method':'clip_sdr'})
        save_scene_master(values,self.root/'focus-master32.tif',working_domain='display_linear_sRGB')
        np.testing.assert_array_equal(np.asarray(preview),actual)

    def test_zero_sigma_preserves_working_exact_bytes_without_quality_claim(self):
        self.relative_sigma = 0.; self.verdict = 'worse'
        state = self.run_focus()['state']; focused = state['candidates'][-1]
        self.assertEqual(focused['image']['sha256'], self.working['image']['sha256'])
        self.assertTrue(focused['provenance']['focus_operator']['zero_sigma_noop'])
        self.assertFalse(focused['provenance']['focus_operator']['quality_improvement_claimed'])
        self.assertEqual(state['best_candidate_id'], state['candidates'][0]['candidate_id'])
        self.assertFalse(state['automated_target_met'])
        self.assertEqual(self.image_requests, self.images_before)

    def test_decision_may_observe_working_then_choose_joint_on_the_actual_best(self):
        def joint(context):
            self.assertEqual([row['role'] for row in context['images']],['source','best','working'])
            return {'action':'tool','tool':'joint_edit','operations':['relight'],
                'instruction':'Unify the current portrait illumination while retaining the scene.',
                'reason':'Choose a complete edit instead of focus integration.',
                'observed_images':[{'role':row['role'],'sha256':row['sha256']} for row in context['images']],
                'provenance':{'model':'fixture-vision','model_requests':1,'generation_used':False}}
        other=self.root/'joint-with-working'
        session.open_session(other,source=self.fx.source,scope=self.fx.scope,design=self.fx.design,
            execution_identity='checked-joint-with-working',producer_identity='fixture',appearance_required=True,
            source_appearance=self.fx.appearance,budget={'image_submissions':1,'vision_requests':2,
                'tool_calls':1,'improvement_steps':1},user_intent={'user_feedback':'Improve the shared portrait illumination.'})
        session.adopt_reviewed_candidate(other,self.best['image']['path'],review=self.best['review'],
            provenance={'review_context':self.best_context},baseline_only=True,appearance_asset=self.best['appearance_asset'])
        session.adopt_reviewed_working_candidate(other,self.working['image']['path'],review=self.working['review'],
            provenance=self.working['provenance'],review_context=self.working_context,appearance_asset=self.working['appearance_asset'])
        tools=make_tools(self.root,self.root/'joint-tools',scope=self.fx.scope,session_directory=other,
            provider_options=self.fx.provider_options)
        session.advance(other,tools=tools,decide=joint,evaluate=self.evaluate,model_identity='fixture')
        def transport(endpoint,key,image,model,prompt,size,timeout,**kwargs):
            with Image.open(image) as uploaded:
                stream=io.BytesIO();Image.new('RGB',uploaded.size,(90,100,120)).save(stream,'PNG')
            return {'id':'joint-fixture'},stream.getvalue(),'b64_json'
        with patch('reference_provider.request_edit',side_effect=transport) as request,redirect_stdout(io.StringIO()):
            state=session.advance(other,tools=tools,decide=joint,evaluate=self.evaluate,model_identity='fixture')
        self.assertEqual(request.call_count,1)
        self.assertEqual(state['candidates'][-1]['appearance_kind'],'generated_full')

    def test_missing_wrong_base_or_changed_provenance_rejected_before_model(self):
        counts = self.image_requests, len(self.vision_requests)
        state = session.snapshot(self.directory)
        focus = next(tool for tool in self.tools if tool.name == TOOL)
        session_path = self.directory/'session.json'; before = session_path.read_bytes()
        unpublished = self.directory/'session.json.unpublished'
        session_path.replace(unpublished)
        try:
            self.assertTrue(session.tool_inputs_available(state, focus),
                            'Eligibility reloaded a stale disk session instead of using the supplied actual state')
        finally:
            unpublished.replace(session_path)
        self.assertEqual(session_path.read_bytes(), before)
        absent = copy.deepcopy(state); absent['working_candidate_id'] = None
        self.assertFalse(session.tool_inputs_available(absent, focus))
        corrupted = copy.deepcopy(state)
        row = next(c for c in corrupted['candidates'] if c['candidate_id']==corrupted['working_candidate_id'])
        row['provenance']['composition_receipt']['sha256'] = '0'*64
        self.assertFalse(session.tool_inputs_available(corrupted, focus))
        with self.assertRaises(ValueError):
            self.proof(best=self.working['image'])
        changed = copy.deepcopy(self.working['appearance_asset']);changed['receipt_sha256']='0'*64
        with self.assertRaises(ValueError):self.proof(working_asset=changed)
        changed = copy.deepcopy(self.working['provenance']); changed['composition_receipt']['sha256'] = '0'*64
        with self.assertRaises(ValueError):self.proof(provenance=changed)
        self.assertEqual(counts, (self.image_requests, len(self.vision_requests)))
        self.assertFalse((self.root/'focus-tools').exists())

    def test_late_cpu_cancel_does_not_replace_best_or_publish_partial_candidate(self):
        session.advance(self.directory, tools=self.tools, decide=self.decide, evaluate=self.evaluate,
                        model_identity='actual-fixture-adapters')
        self.cancel_threshold = 5; self.cancel_calls = 0
        with self.assertRaises(InterruptedError):
            session.advance(self.directory, tools=self.tools, decide=self.decide, evaluate=self.evaluate,
                            model_identity='actual-fixture-adapters', cancel=self.cancel)
        state = session.snapshot(self.directory)
        self.assertEqual(state['best_candidate_id'], state['candidates'][0]['candidate_id'])
        self.assertFalse(state['automated_target_met'])
        self.assertEqual(self.image_requests, self.images_before)
        self.assertFalse(list((self.root/'focus-tools').rglob('candidate.png')))
        self.assertFalse(list((self.root/'focus-tools').rglob('*.partial.png')))
        self.assertEqual(Path(self.best['image']['path']).read_bytes(), self.baseline_bytes)

    def test_parameters_are_finite_bounded_numbers_and_not_booleans(self):
        for invalid in (True, float('nan'), float('inf'), -.001, .020001, '0.01'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):_sigma({'focus_sigma': invalid})
        with self.assertRaises(ValueError):_sigma({'focus_sigma': .01, 'extra': 1})
        self.assertEqual(PARAMETERS['focus_sigma']['unit'], 'native_short_edge')

    def test_full_canvas_joint_mixed_asset_is_not_a_bounded_focus_increment(self):
        from photo_appearance import freeze_candidate
        best=working=None
        try:
            best=load_asset(self.best['appearance_asset'],**self.fx.binding)
            working=load_asset(self.working['appearance_asset'],**self.fx.binding)
            prior=working['receipt']['generation'];directory=Path(self.working['appearance_asset']['path'])
            generation={role:{'path':str(directory/prior[role]['file']),'sha256':prior[role]['sha256']}
                for role in ('image','provider_receipt')}
            generation.update(core_box=[0,0,96,128],context_box=[0,0,96,128])
            coverage=np.full((128,96),np.float32(.5))
            values=best['master']*(1-coverage[...,None])+np.float32(.2)*coverage[...,None]
            asset=freeze_candidate(self.root/'joint-mixed',kind='mixed_regional',master=values,
                parent=self.best['appearance_asset'],coverage=coverage,generation=generation,**self.fx.binding)
        finally:_close_loaded(best,working)
        image={'path':str(Path(asset['path'])/'preview.png'),'sha256':sha256_file(Path(asset['path'])/'preview.png')}
        with self.assertRaisesRegex(ValueError,'genuine local edit'):
            self.proof(working=image,working_asset=asset)
        self.assertEqual(self.image_requests,self.images_before)
        self.assertFalse((self.root/'focus-tools').exists())

    def test_increment_blur_does_not_soften_unmodified_base_on_two_canvases(self):
        for width,height in ((100,150),(160,100)):
            yy,xx=np.indices((height,width));base=np.repeat((((xx+yy)%2)*.6+.05)[...,None],3,axis=2).astype(np.float32)
            base+=np.float32(.0000001)
            x0,y0,x1,y1=width//4,height//4,width*3//4,height*3//4
            coverage=np.zeros((height,width),np.float32);coverage[y0+1:y1-1,x0+1:x1-1]=1
            result=integrate_increment(base,base.copy(),coverage,[x0,y0,x1,y1],min(width,height)*.02)
            np.testing.assert_array_equal(result,base)

    def test_impulse_is_integrated_in_linear_light_not_gamma_encoded_rgb(self):
        from float_appearance import srgb_to_linear,linear_to_srgb
        base=np.full((100,100,3),srgb_to_linear(np.float32(16/255)),np.float32)
        working=base.copy();working[50,50]=1
        coverage=np.zeros((100,100),np.float32);coverage[20:80,20:80]=1
        result=integrate_increment(base,working,coverage,[20,20,80,80],1.)
        encoded=linear_to_srgb(result[50,50,0])*255
        self.assertGreater(float(encoded),95);self.assertLess(float(encoded),125)
        np.testing.assert_array_equal(result[:20],base[:20])


if __name__ == '__main__':
    unittest.main()
