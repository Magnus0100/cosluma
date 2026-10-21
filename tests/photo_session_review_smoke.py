"""Real board pixels and transport binding, with synthetic vision responses."""
import copy
import base64
from email.message import Message
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

from asset_identity import sha256_file
from photo_editing import editing_scope, validate_observed_design, identity
import photo_session_review as review
from photo_edit_session_smoke import design


def answer():
    return {'credibility': {'status': 'pass', 'findings': ['Synthetic bound source identity retained']},
        'creative_benefit': {'status': 'fail', 'findings': ['Synthetic design not yet fulfilled']},
        'comparison': 'first_candidate', 'primary_issue': 'Synthetic lighting remains weak'}


class IndependentReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'source.png'; Image.new('RGB', (32, 48), 'red').save(self.source)
        self.candidate = self.root/'candidate.png'; Image.new('RGB', (32, 48), 'blue').save(self.candidate)
        self.images = [{'role': role, 'path': str(path), 'sha256': sha256_file(path)}
                       for role, path in [('source', self.source), ('candidate', self.candidate)]]
        self.scope = editing_scope({'language': 'en'})
        self.context = {'session_id': 'fixture', 'scope': self.scope, 'design': design(self.scope),
                        'images': self.images, 'attempt_id': 'fixture-attempt'}
        self.output = self.root/'review'

    def transport(self, source, output, prompt, **kwargs):
        source = Path(source)
        self.assertEqual(source.name, 'review-board.png')
        self.assertIn('source, candidate', prompt)
        self.assertNotIn('Flawless', prompt)
        with Image.open(source) as board:
            self.assertEqual(board.getpixel((268, 432)), (255, 0, 0))
            self.assertEqual(board.getpixel((792, 432)), (0, 0, 255))
        return answer(), {'source': {'path': str(source), 'sha256': sha256_file(source)},
            'network_used': True, 'generation_used': False, 'model': 'actual-transport-model-fixture',
            'response_id': 'response-fixture'}

    def run_record(self, *, enabled=True):
        intent = {'mode': 'create_reference', 'services': {'text_vision': True, 'visual_review': enabled}}
        return {'run_id': 'fixture-run', 'task_intent': intent,
            'task_plan': {'ready': True, 'mode': 'create_reference', 'plan_id': 'checked-plan-fixture', 'editing_scope': self.scope,
                'image_budget': {'initial_reference': 1, 'reference_correction': 0, 'independent_effect': 0},
                'services': {'text_vision_requests_upper_bound': 2}},
            'source': {'path': str(self.source), 'sha256': sha256_file(self.source)},
            'generation_design': {'observed_design': design(self.scope),
                'source_sha256': sha256_file(self.source), 'text_vision_submissions': 1,
                'art_direction': {'status': 'ready', 'model_requests_attempted': 1,
                    'binding': {'source': {'path': str(self.source), 'sha256': sha256_file(self.source)}},
                    'provenance': {'source': {'path': str(self.source), 'sha256': sha256_file(self.source)},
                        'network_used': True, 'generation_used': False, 'model': 'synthetic-design-history-fixture'}}},
            'generation_budget': {'image_submissions': 1},
            'artifacts': {'auto_draft': str(self.candidate)}, 'status': 'needs_visual_review',
            'review_gate': 'visual_review_required', 'options': {'task_intent': copy.deepcopy(intent)}}

    def typed_regional_fixture(self, *, parent_is_source=False, core=(30, 50, 70, 100), name='typed'):
        """Actual native assets/receipts; only the model-returned pixels are synthetic."""
        from photo_appearance import create_source_baseline, freeze_candidate, load_asset
        from raw_support import open_linear
        directory=self.root/name;directory.mkdir()
        cache=patch.dict('os.environ',{'COSPLAY_ASSET_IDENTITY_CACHE':str(directory/'identity-cache')})
        cache.start();self.addCleanup(cache.stop)
        source=directory/'original.png'
        yy,xx=np.indices((180,120));pixels=np.empty((180,120,3),np.uint8)
        pixels[...,0]=40+xx%50;pixels[...,1]=60+yy%70;pixels[...,2]=90
        Image.fromarray(pixels).save(source)
        binding={'source_sha256':sha256_file(source),'scope':self.scope,'design_id':self.context['design']['design_id']}
        baseline=create_source_baseline(directory/'baseline',source_path=source,**binding)
        def generated(label,kind,parent,core_box=None, *, unchanged=False):
            parent_values=load_asset(parent,**binding)['master'].copy()
            raw=directory/(label+'.png')
            if kind=='generated_full':
                values=pixels.copy();values[...,2]+=35;Image.fromarray(values).save(raw)
                master,_=open_linear(raw);coverage=None;context_box=None
            else:
                context_box=[0,0,120,180]
                if unchanged:
                    raw.write_bytes(Path(load_asset(parent,**binding)['preview_path']).read_bytes())
                else:
                    Image.new('RGB',(120,180),(185,45,170)).save(raw)
                patch_values,_=open_linear(raw);coverage=np.zeros((180,120),np.float32)
                l,t,r,b=core_box
                coverage[t:b,l:r]=.5 if core_box==context_box else 1.
                master=parent_values.copy()
                if not unchanged:
                    master=master*(1-coverage[...,None])+patch_values*coverage[...,None]
            receipt=directory/(label+'.provenance.json')
            receipt.write_text(json.dumps({'status':'reference_candidate','source':{
                'path':str(source),'sha256':binding['source_sha256']},'asset':{
                'path':str(raw),'sha256':sha256_file(raw),'oriented_size':[120,180]},
                'provider':'fixture','model':'synthetic-regional-pixels'}),encoding='utf-8')
            return freeze_candidate(directory/label,kind=kind,master=master,parent=parent,coverage=coverage,
                generation={'image':{'path':str(raw),'sha256':sha256_file(raw)},
                    'provider_receipt':{'path':str(receipt),'sha256':sha256_file(receipt)},
                    'context_box':context_box,'core_box':core_box},**binding)
        prior=baseline if parent_is_source else generated('prior','generated_full',baseline)
        candidate=generated('local','mixed_regional',prior,list(core))
        def context_for(asset,parent=None,*,previous=True):
            assets={'source':baseline,'candidate':asset}
            rows=[]
            for role,current in [('source',baseline),('candidate',asset),*([('previous_best',parent)] if previous and parent else [])]:
                loaded=load_asset(current,**binding)
                rows.append({'role':role,'path':str(loaded['preview_path']),
                    'sha256':loaded['receipt']['files']['preview']['sha256']})
                assets[role]=current
            return {**copy.deepcopy(self.context),'images':rows,'source_original':{
                'path':str(source),'sha256':binding['source_sha256']},'appearance_assets':copy.deepcopy(assets),
                'user_intent':{'brief':'Preserve the person and improve the actual local edit.'}}
        return {'context':context_for(candidate,prior),'baseline':baseline,'parent':prior,'candidate':candidate,
            'binding':binding,'source':source,'context_for':context_for,'generated':generated}

    def test_typed_regional_board_preserves_first_row_and_actual_crop_transport_once(self):
        from vision_transport import _input_image
        from raw_support import open_rgb
        fx=self.typed_regional_fixture();context=fx['context'];original=copy.deepcopy(context)
        region=review.review_region(context)
        self.assertEqual(region['canvas_size'],[120,180]);self.assertEqual(region['core_box'],[30,50,70,100])
        self.assertEqual(region['box'],[26,46,74,104]);self.assertEqual(region['generation_context_box'],[0,0,120,180])
        self.assertEqual(region['actual_parent_role'],'previous_best')
        self.assertEqual(region['parent_asset_id'],fx['parent']['asset_id'])
        self.assertEqual(region['candidate_asset_id'],fx['candidate']['asset_id'])
        self.assertEqual(region['candidate_receipt_sha256'],fx['candidate']['receipt_sha256'])
        whole=review.prepare_board(context['images'],self.root/'whole-only')
        def inspect(source,output,prompt,**kwargs):
            self.assertEqual(kwargs.get('input_profile'),'detailed_board')
            self.assertIn('first row remains the complete photographs',prompt)
            self.assertIn('Only core_box is the permitted edit',prompt)
            self.assertIn('not native 1:1 certification',prompt)
            mapping=json.loads((Path(source).parent/'review-board.json').read_text('utf-8'))
            self.assertEqual(mapping['region'],region)
            with Image.open(source) as board,Image.open(whole['board']['path']) as first:
                self.assertEqual(board.size,(1584,1644))
                np.testing.assert_array_equal(np.asarray(board)[:828],np.asarray(first))
                for logical in mapping['images']:
                    roi=logical['region'];self.assertEqual(roi['native_box'],region['box'])
                    self.assertEqual(roi['core_box'],region['core_box']);self.assertEqual(roi['crop_size'],[48,58])
                    x,y,w,h=roi['board_rect'];decoded=open_rgb(Path(logical['path']))
                    expected=ImageOps.contain(decoded.crop(region['box']),(512,768),Image.Resampling.LANCZOS)
                    np.testing.assert_array_equal(np.asarray(board)[y:y+h,x:x+w],np.asarray(expected))
                    self.assertEqual(roi['native_pixels_per_board_pixel'],[48/w,58/h]);self.assertTrue(roi['upscaled'])
                    decoded.close();expected.close()
            api,metadata=_input_image(Path(source),Path(output),input_profile=kwargs['input_profile'])
            self.assertEqual(metadata['size'],[1584,1644])
            self.assertEqual(metadata['decode']['long_edge_limit'],2048)
            with Image.open(api) as jpeg,Image.open(source) as board:
                for logical in mapping['images']:
                    x,y,w,h=logical['region']['board_rect'];point=(x+w//2,y+h//2)
                    self.assertTrue(all(abs(a-b)<=4 for a,b in zip(jpeg.getpixel(point),board.getpixel(point))))
            payload=answer();payload['comparison']='better'
            return payload,{'source':{'path':str(source),'sha256':sha256_file(source)},'api_input':metadata,
                'input_profile':'detailed_board','network_used':True,'generation_used':False,'model':'fixture'}
        with patch('vision_transport.request_vision_json',side_effect=inspect) as transport:
            result=review.evaluate_images(context,self.output)
            self.assertEqual(review.evaluate_images(context,self.output),result)
        self.assertEqual(transport.call_count,1);self.assertEqual(context,original)
        saved=json.loads((self.output/'review.json').read_text('utf-8'))
        self.assertEqual(saved['model_requests_attempted'],1);self.assertEqual(saved['image_submissions'],0)
        self.assertEqual(result['observed_images'],[{k:r[k] for k in ('role','sha256')} for r in context['images']])
        self.assertEqual(result['provenance']['review_inspection_scope'],'whole_image_and_edit_region')
        self.assertFalse(result['provenance']['native_detail_acceptance'])
        self.assertFalse(saved['human_approved'])

    def test_regional_review_uses_real_transport_profile_jpeg_and_terminal_evidence(self):
        fx = self.typed_regional_fixture()
        payload = answer(); payload['comparison'] = 'better'

        class Response(BytesIO):
            def __init__(self):
                super().__init__(json.dumps({'id': 'offline-regional-response', 'status': 'completed',
                    'output_text': json.dumps(payload)}).encode('utf-8'))
                self.headers = Message(); self.headers['Content-Type'] = 'application/json'
                self.status = 200

        def opened(request, *, timeout):
            self.assertEqual(request.full_url, 'https://fixture.invalid/v1/responses')
            body = json.loads(request.data)
            content = body['input'][0]['content']
            image = next(row['image_url'] for row in content if row['type'] == 'input_image')
            jpeg = base64.b64decode(image.split(',', 1)[1])
            with Image.open(BytesIO(jpeg)) as decoded:
                self.assertEqual(decoded.size, (1584, 1644))
            self.assertIn('detailed_board', next(row['text'] for row in content if row['type'] == 'input_text'))
            return Response()

        with patch('vision_transport._connection', return_value=(
                {'registry_id': 'offline-fixture'}, 'https://fixture.invalid/v1', 'fixture-secret', None)), \
                patch('vision_transport.urllib.request.urlopen', side_effect=opened) as http:
            result = review.evaluate_images(fx['context'], self.output, vision_model='fixture-vision')
            self.assertEqual(review.evaluate_images(fx['context'], self.output, vision_model='fixture-vision'), result)
        self.assertEqual(http.call_count, 1)
        provenance = result['provenance']
        self.assertEqual(provenance['input_profile'], 'detailed_board')
        self.assertEqual(provenance['api_input']['decode']['long_edge_limit'], 2048)
        self.assertEqual(provenance['api_input']['size'], [1584, 1644])
        binding = json.loads((self.output/'transport/request-binding.json').read_text('utf-8'))
        self.assertEqual(binding['input_profile'], 'detailed_board')
        observations = list((self.output/'transport').rglob('observation.json'))
        self.assertEqual(len(observations), 1)
        observed = json.loads(observations[0].read_text('utf-8'))
        self.assertEqual(observed['response_id'], 'offline-regional-response')
        self.assertFalse(provenance['native_detail_acceptance'])

    def test_regional_source_parent_aliases_share_both_rows_and_full_core_has_no_roi(self):
        fx=self.typed_regional_fixture(parent_is_source=True)
        context=fx['context'];alias=self.root/'source-parent-alias.png'
        alias.write_bytes(Path(context['images'][0]['path']).read_bytes())
        context['images'][2].update(path=str(alias),sha256=sha256_file(alias))
        region=review.review_region(context)
        self.assertEqual(region['actual_parent_role'],'previous_best')
        board=review.prepare_board(context['images'],self.output,region=region)
        self.assertEqual(len(board['panels']),2)
        self.assertEqual([r['panel_index'] for r in board['images']],[0,1,0])
        self.assertEqual(board['images'][0]['region'],board['images'][2]['region'])
        from raw_support import open_rgb
        before=alias.read_bytes()
        def mutate_alias(path):
            result=open_rgb(path);alias.write_bytes(b'changed regional alias');return result
        with patch('raw_support.open_rgb',side_effect=mutate_alias), \
             patch('vision_transport.request_vision_json') as forbidden:
            self.assertRaisesRegex(ValueError,'bytes changed',review.evaluate_images,context,self.root/'changed-regional-alias')
        self.assertFalse(forbidden.called)
        self.assertFalse((self.root/'changed-regional-alias/board/review-board.png').exists())
        alias.write_bytes(before)
        without_previous=fx['context_for'](fx['candidate'],previous=False)
        self.assertEqual(review.review_region(without_previous)['actual_parent_role'],'source')
        baseline=fx['context_for'](fx['baseline'],previous=False)
        self.assertIsNone(review.review_region(baseline))
        full=fx['generated']('transparent-full-core','mixed_regional',fx['parent'],[0,0,120,180])
        self.assertIsNone(review.review_region(fx['context_for'](full,fx['parent'])))

    def test_global_and_photographic_candidates_do_not_inherit_regional_inspection(self):
        from photo_appearance import load_asset,freeze_candidate
        from photo_development import develop_float,operator_for,producer_identity
        fx=self.typed_regional_fixture()
        self.assertIsNone(review.review_region(fx['context_for'](fx['parent'],fx['baseline'])))
        parent=load_asset(fx['candidate'],**fx['binding'])
        parameters={'adjustments':[{'region':[0,0,1,1],'exposure_ev':.1,'warmth_ev':0.,'tint_ev':0.,'feather':0.}]}
        values,coverage,stats=develop_float(parent['master'],parameters)
        cpu=freeze_candidate(self.root/'cpu',kind='photographic_adjustment',master=values,parent=fx['candidate'],
            coverage=coverage,photographic_operation={'operator':operator_for(parameters),'stats':stats,
                'producer':producer_identity()},**fx['binding'])
        self.assertIsNone(review.review_region(fx['context_for'](cpu,fx['candidate'])))

    def test_region_binding_parent_source_and_decoded_canvas_fail_before_transport(self):
        from raw_support import open_rgb
        fx=self.typed_regional_fixture();context=fx['context']
        invalid=[]
        changed=copy.deepcopy(context);changed['appearance_assets']['candidate']['receipt_sha256']='0'*64;invalid.append(changed)
        changed=copy.deepcopy(context);changed['source_original']['sha256']='0'*64;invalid.append(changed)
        changed=copy.deepcopy(context);changed['images'][1]=copy.deepcopy(changed['images'][2]);changed['images'][1]['role']='candidate';invalid.append(changed)
        changed=copy.deepcopy(context);changed['images'][0]={**copy.deepcopy(changed['images'][2]),'role':'source'};invalid.append(changed)
        changed=copy.deepcopy(context)
        changed['images'][2]={**copy.deepcopy(changed['images'][0]),'role':'previous_best'}
        changed['appearance_assets']['previous_best']=copy.deepcopy(changed['appearance_assets']['source']);invalid.append(changed)
        with patch('vision_transport.request_vision_json') as forbidden:
            for index,changed in enumerate(invalid):
                with self.subTest(index=index):
                    self.assertRaises(ValueError,review.evaluate_images,changed,self.root/('invalid-'+str(index)))
            def mismatch(path):
                image=open_rgb(path)
                return image.resize((60,90)) if Path(path)==Path(context['images'][0]['path']) else image
            with patch('raw_support.open_rgb',side_effect=mismatch):
                self.assertRaisesRegex(ValueError,'native canvas',review.evaluate_images,context,self.root/'wrong-canvas')
        self.assertFalse(forbidden.called)

    def test_regional_receipt_and_preview_tampering_are_not_downgraded_to_whole_only(self):
        fx=self.typed_regional_fixture();context=fx['context']
        receipt=Path(fx['candidate']['path'])/'receipt.json';original=receipt.read_bytes()
        changed=json.loads(original);changed['generation']['core_box']=[1,1,2,2]
        receipt.write_text(json.dumps(changed),encoding='utf-8')
        with patch('vision_transport.request_vision_json') as forbidden:
            self.assertRaises(ValueError,review.evaluate_images,context,self.root/'bad-receipt')
            receipt.write_bytes(original)
            candidate=Path(context['images'][1]['path']);candidate.write_bytes(candidate.read_bytes()+b'changed')
            self.assertRaises(ValueError,review.evaluate_images,context,self.root/'bad-preview')
        self.assertFalse(forbidden.called)

    def test_regional_profile_evidence_required_and_region_facts_affect_review_key(self):
        fx=self.typed_regional_fixture();context=fx['context'];prompt=review.review_prompt(context)
        same_pixels=fx['generated']('same-pixels-one','mixed_regional',fx['parent'],[30,50,70,100],unchanged=True)
        other_box=fx['generated']('same-pixels-two','mixed_regional',fx['parent'],[25,45,75,105],unchanged=True)
        one=fx['context_for'](same_pixels,fx['parent']);two=fx['context_for'](other_box,fx['parent'])
        self.assertEqual([{k:r[k] for k in ('role','sha256')} for r in one['images']],
                         [{k:r[k] for k in ('role','sha256')} for r in two['images']])
        self.assertEqual(one['images'][1]['sha256'],two['images'][1]['sha256'])
        from photo_request_budget import OriginHooks
        hooks=SimpleNamespace(connection={'vision_model':'fixture'},user_intent={})
        first_key=OriginHooks._review_key(hooks,one);second_key=OriginHooks._review_key(hooks,two)
        self.assertEqual(first_key['candidate_sha256'],second_key['candidate_sha256'])
        self.assertNotEqual(first_key['prompt_sha256'],second_key['prompt_sha256'])
        self.assertNotEqual(first_key,second_key)
        self.assertIn('detailed_board',prompt)
        for profile in ('standard', None):
            with self.subTest(profile=profile):
                output = self.output/str(profile)
                def wrong_profile(source,output,prompt,**kwargs):
                    payload=answer();payload['comparison']='better'
                    provenance={'source':{'path':str(source),'sha256':sha256_file(source)},
                        'network_used':True,'generation_used':False,'model':'fixture'}
                    if profile is not None:provenance['input_profile']=profile
                    return payload,provenance
                with patch('vision_transport.request_vision_json',side_effect=wrong_profile) as transport:
                    self.assertRaises(ValueError,review.evaluate_images,context,output)
                    self.assertRaises(RuntimeError,review.evaluate_images,context,output)
                self.assertEqual(transport.call_count,1)

    def test_board_has_actual_images_and_model_cannot_supply_provenance(self):
        with patch('vision_transport.request_vision_json', side_effect=self.transport) as call:
            result = review.evaluate_images(self.context, self.output, vision_model='selected-fixture')
            again = review.evaluate_images(self.context, self.output, vision_model='selected-fixture')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result, again)
        self.assertEqual(result['provenance']['model'], 'actual-transport-model-fixture')
        self.assertEqual(result['observed_images'], [{k: row[k] for k in ('role', 'sha256')} for row in self.images])
        self.assertFalse(result['provenance']['native_detail_acceptance'])
        mapping = json.loads((self.output/'board'/'review-board.json').read_text())
        self.assertEqual([row['role'] for row in mapping['images']], ['source', 'candidate'])
        self.assertEqual(mapping['board']['sha256'], result['provenance']['source']['sha256'])

    def test_alias_roles_share_actual_panel_and_transport_pixels_but_keep_all_bindings(self):
        from vision_transport import _input_image
        alias=self.root/'previous-best.png';alias.write_bytes(self.source.read_bytes())
        images=[*copy.deepcopy(self.images),{'role':'previous_best','path':str(alias),'sha256':sha256_file(alias)}]
        context={**self.context,'images':images};original=copy.deepcopy(context)
        def inspect(source,output,prompt,**kwargs):
            self.assertIn('panels from left to right are source / previous_best, candidate.',prompt)
            self.assertIn('share one panel',prompt)
            with Image.open(source) as board:
                self.assertEqual(board.size,(1060,828))
                self.assertEqual(board.getpixel((268,432)),(255,0,0))
                self.assertEqual(board.getpixel((792,432)),(0,0,255))
            api,receipt=_input_image(Path(source),Path(output))
            self.assertEqual(receipt['size'],[1024,800])
            with Image.open(api) as board:
                for native,expected in (((268,432),(255,0,0)),((792,432),(0,0,255))):
                    pixel=board.getpixel((round(native[0]*1024/1060),round(native[1]*800/828)))
                    self.assertTrue(all(abs(a-b)<=2 for a,b in zip(pixel,expected)))
            payload=answer();payload['comparison']='better'
            return payload,{'source':{'path':str(source),'sha256':sha256_file(source)},
                'network_used':True,'generation_used':False,'model':'fixture'}
        with patch('vision_transport.request_vision_json',side_effect=inspect) as call:
            result=review.evaluate_images(context,self.output)
            self.assertEqual(review.evaluate_images(context,self.output),result)
        self.assertEqual(call.call_count,1);self.assertEqual(context,original)
        self.assertEqual(result['observed_images'],[{key:row[key] for key in ('role','sha256')} for row in images])
        saved=json.loads((self.output/'review.json').read_text('utf8'));board=saved['board']
        self.assertEqual(saved['binding']['images'],images);self.assertFalse(saved['human_approved'])
        self.assertEqual([row['roles'] for row in board['panels']],[['source','previous_best'],['candidate']])
        self.assertEqual([{key:row[key] for key in ('role','path','sha256')} for row in board['images']],images)
        self.assertEqual([row['panel_index'] for row in board['images']],[0,1,0])
        self.assertEqual(board['images'][0]['board_rect'],board['images'][2]['board_rect'])

    def test_single_shared_panel_and_different_bytes_keep_native_tile_sizes(self):
        shared=[self.images[0],{**self.images[0],'role':'candidate'}]
        board=review.prepare_board(shared,self.root/'shared')
        self.assertEqual([row['roles'] for row in board['panels']],[['source','candidate']])
        with Image.open(board['board']['path']) as image:
            self.assertEqual(image.size,(536,828));self.assertEqual(image.getpixel((268,432)),(255,0,0))
        self.assertEqual(board['images'][0]['board_rect'],[12,48,512,768])
        near=self.root/'near.png'
        with Image.open(self.source) as image:
            image.putpixel((0,0),(254,0,0));image.save(near)
        board=review.prepare_board([self.images[0],{'role':'candidate','path':str(near),'sha256':sha256_file(near)}],self.root/'near-board')
        self.assertEqual(len(board['panels']),2)
        self.assertEqual([row['board_rect'][2:] for row in board['images']],[[512,768],[512,768]])
        with Image.open(board['board']['path']) as image:self.assertEqual(image.size,(1060,828))

    def test_same_bytes_in_different_decoder_branches_are_separate_panels(self):
        raw=self.root/'same.ARW';raw.write_bytes(self.source.read_bytes())
        images=[self.images[0],{'role':'candidate','path':str(raw),'sha256':sha256_file(raw)}]
        from raw_support import open_rgb
        def decode(path):return Image.new('RGB',(32,48),'blue') if Path(path)==raw else open_rgb(path)
        with patch('raw_support.open_rgb',side_effect=decode) as call:
            board=review.prepare_board(images,self.output)
        self.assertEqual(call.call_count,2)
        self.assertEqual([row['decode_branch'] for row in board['panels']],['raster','raw'])
        with Image.open(board['board']['path']) as image:
            self.assertEqual(image.size,(1060,828));self.assertEqual(image.getpixel((792,432)),(0,0,255))

    def test_every_alias_is_verified_before_render_after_render_and_on_ready_reuse(self):
        from raw_support import open_rgb
        alias=self.root/'alias.png';alias.write_bytes(self.source.read_bytes())
        images=[*copy.deepcopy(self.images),{'role':'previous_best','path':str(alias),'sha256':sha256_file(alias)}]
        context={**self.context,'images':images}
        alias.unlink()
        with patch('vision_transport.request_vision_json') as forbidden:
            self.assertRaises(FileNotFoundError,review.evaluate_images,context,self.root/'missing')
        self.assertFalse(forbidden.called)
        alias.write_bytes(self.source.read_bytes())
        def changed_during_decode(path):
            image=open_rgb(path);alias.write_bytes(b'changed alias');return image
        with patch('raw_support.open_rgb',side_effect=changed_during_decode),patch('vision_transport.request_vision_json') as forbidden:
            self.assertRaisesRegex(ValueError,'bytes changed',review.evaluate_images,context,self.root/'changed')
        self.assertFalse(forbidden.called);self.assertFalse((self.root/'changed'/'board'/'review-board.png').exists())
        alias.write_bytes(self.source.read_bytes())
        def inspect(source,output,prompt,**kwargs):
            payload=answer();payload['comparison']='equivalent'
            return payload,{'source':{'path':str(source),'sha256':sha256_file(source)},
                'network_used':True,'generation_used':False,'model':'fixture'}
        with patch('vision_transport.request_vision_json',side_effect=inspect) as call:
            review.evaluate_images(context,self.output);before=(self.output/'review.json').read_bytes()
            for value in (b'changed alias',None):
                if value is None:alias.unlink()
                else:alias.write_bytes(value)
                self.assertRaises((ValueError,FileNotFoundError),review.evaluate_images,context,self.output)
        self.assertEqual(call.call_count,1);self.assertEqual((self.output/'review.json').read_bytes(),before)

    def test_transport_receives_delivery_severity_and_preservation_review_standards(self):
        context = copy.deepcopy(self.context)
        context['user_intent'] = {'brief': 'Unify motivated scene illumination while preserving expression.'}
        def inspect_prompt(source, output, prompt, **kwargs):
            # Inspect the actual adapter boundary, not a replacement reviewer.
            # Synthetic findings cannot prove that a vision model follows it.
            for standard in (
                'Pass means the major deliverable objectives for that axis are achieved',
                'not merely\nthe absence of anatomical errors, broken masks',
                'Judge whole-image light integration',
                'Motivated mixed lighting and deliberate colour contrast are acceptable',
                'do not require every surface to share the same hue',
                'A major problem inherited from the source is not exempt',
                'Distinguish small acceptable imperfections and optional polish',
                'A visibly unresolved\nmajor objective is fail',
                'insufficient visual evidence is unknown',
                'Do not give an axis pass while',
                'Check expression, gaze direction and\nlip opening separately from identity resemblance',
                'Better than the source or\nprevious_best does not by itself mean',
                'benefit relative to the source, not the number of new objects',
                'Repeating an existing blurred background motif',
                'Retouching does not require added scenery or effects',
                context['user_intent']['brief'],
            ):
                self.assertIn(standard, prompt)
            self.assertIn('native-resolution detail has not been inspected', prompt)
            return self.transport(source, output, prompt, **kwargs)
        with patch('vision_transport.request_vision_json', side_effect=inspect_prompt) as call:
            result = review.evaluate_images(context, self.output)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(set(result), {'credibility', 'creative_benefit', 'comparison', 'primary_issue',
                                      'observed_images', 'provenance'})
        self.assertEqual(result['observed_images'], [{key: row[key] for key in ('role', 'sha256')} for row in self.images])

    def planner_design(self,marker):
        return validate_observed_design({'summary':marker+'_SUMMARY',
            'observation':{'setting':'indoor','description':marker+'_OBSERVATION'},
            'changes':[{'operation':'relight','target':marker+'_TARGET','instruction':marker+'_INSTRUCTION',
                'purpose':marker+'_PURPOSE'}], 'lighting':marker+'_LIGHTING',
            'keep':[marker+'_KEEP'],'uncertain':[marker+'_UNCERTAIN']},self.scope)

    def test_actual_transport_objective_is_identical_across_different_planner_narratives(self):
        user={'brief':'USER_VISIBLE_WINDOW_PAPER with coherent lighting.',
            'character_name':'USER_CHARACTER','character_notes':'USER_CHARACTER_FACTS',
            'user_feedback':'USER_FEEDBACK_KEEP_FACE_LIGHT'}
        prompts=[];bindings=[]
        def inspect(source,output,prompt,**kwargs):
            prompts.append(prompt)
            objective=json.JSONDecoder().raw_decode(prompt.split('OBJECTIVE\n',1)[1])[0]
            self.assertEqual(objective,{'scope':self.scope,'user_intent':user})
            for marker in ('PLANNER_A','PLANNER_B'):
                self.assertNotIn(marker,prompt)
            return self.transport(source,output,prompt,**kwargs)
        with patch('vision_transport.request_vision_json',side_effect=inspect) as call:
            contexts=[]
            for marker in ('PLANNER_A','PLANNER_B'):
                context={**copy.deepcopy(self.context),'design':self.planner_design(marker),'user_intent':copy.deepcopy(user)}
                original=copy.deepcopy(context);contexts.append(context)
                directory=self.root/marker
                result=review.evaluate_images(context,directory)
                self.assertEqual(context,original)
                self.assertEqual(result['observed_images'],[{key:row[key] for key in ('role','sha256')} for row in self.images])
                bindings.append(json.loads((directory/'review.json').read_text('utf8'))['binding'])
        self.assertEqual(call.call_count,2)
        self.assertEqual(prompts[0],prompts[1])
        self.assertNotEqual(contexts[0]['design']['design_id'],contexts[1]['design']['design_id'])
        self.assertNotEqual(identity(contexts[0]),identity(contexts[1]))
        self.assertNotEqual(bindings[0]['context_sha256'],bindings[1]['context_sha256'])

    def test_actual_transport_preserves_each_original_user_field_and_feedback_changes_objective(self):
        user={'brief':'USER_WINDOW_PAPER_REQUIRED','character_name':'USER_CHARACTER',
            'character_notes':'USER_DECLARED_FACTS','user_feedback':'USER_NEW_FEEDBACK'}
        prompts=[]
        def inspect(source,output,prompt,**kwargs):
            prompts.append(prompt)
            return self.transport(source,output,prompt,**kwargs)
        with patch('vision_transport.request_vision_json',side_effect=inspect) as call:
            review.evaluate_images({**self.context,'user_intent':copy.deepcopy(user)},self.root/'original-user')
            for key in user:
                changed={**user,key:user[key]+'_CHANGED'}
                review.evaluate_images({**self.context,'user_intent':changed},self.root/('changed-'+key))
                objective=json.JSONDecoder().raw_decode(prompts[-1].split('OBJECTIVE\n',1)[1])[0]
                self.assertEqual(objective['user_intent'],changed)
                self.assertNotEqual(prompts[0],prompts[-1])
        self.assertEqual(call.call_count,5)
        for value in user.values():self.assertIn(value,prompts[0])

    def test_prior_producer_review_receipt_is_not_relabelled_or_resubmitted(self):
        with patch('vision_transport.request_vision_json',side_effect=self.transport) as call:
            review.evaluate_images(self.context,self.output)
            path=self.output/'review.json';saved=json.loads(path.read_text('utf8'))
            saved['binding']['producer']['photo_edit_session.py']='0'*64
            saved['prompt']+='\nHISTORICAL_PLANNER_OBJECTIVE'
            path.write_text(json.dumps(saved),encoding='utf8');historical=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'other images, objective or code'):
                review.evaluate_images(self.context,self.output)
        self.assertEqual(call.call_count,1);self.assertEqual(path.read_bytes(),historical)

    def test_old_prompt_origin_cache_is_not_reused_under_current_qa_objective(self):
        from photo_source_session_smoke import NativeFixture
        from photo_request_budget import make_hooks
        from photo_edit_finish import _connection
        import photo_edit_session as session
        fixture=NativeFixture(self,k=0);current=review.review_prompt
        def historical_prompt(context):
            return current(context)+'\nHISTORICAL_PLANNER_OBJECTIVE '+json.dumps(context['design'])
        with patch.object(review,'review_prompt',side_effect=historical_prompt):
            variant=fixture.start()[0]
        directory=Path(variant['photo_edit_session']['path']).parent
        state=session.snapshot(directory)
        baseline=next(row['context'] for row in state['attempts'] if row['kind']=='evaluate')
        hooks=make_hooks(fixture.root,fixture.run,_connection(fixture.run,fixture.args))
        path=Path(hooks.locator['path']);before=path.read_bytes()
        calls=len(fixture.calls),len(fixture.fx.image_requests)
        self.assertIsNone(hooks.cached_review(baseline))
        self.assertEqual((len(fixture.calls),len(fixture.fx.image_requests)),calls)
        self.assertEqual(path.read_bytes(),before)

    def test_working_board_purpose_cannot_be_mistaken_for_an_independent_review(self):
        images = [copy.deepcopy(self.images[0]), {**self.images[1], 'role': 'best'},
                  {**self.images[1], 'role': 'working'}]
        self.assertRaises(ValueError, review.prepare_board, images, self.output)
        board = review.prepare_board(images, self.root/'decision-board', decision=True)
        self.assertEqual([row['role'] for row in board['images']], ['source', 'best', 'working'])
        with patch('vision_transport.request_vision_json') as forbidden:
            invalid = {**self.context, 'images': images}
            self.assertRaises(ValueError, review.evaluate_images, invalid, self.root/'invalid-review')
        self.assertFalse(forbidden.called)

    def test_payload_cannot_inject_execution_evidence_or_wrong_image_identity(self):
        def injected(source, output, prompt, **kwargs):
            payload, provenance = self.transport(source, output, prompt, **kwargs)
            payload['provenance'] = {'model': 'invented-by-model'}
            return payload, provenance
        with patch('vision_transport.request_vision_json', side_effect=injected) as call:
            self.assertRaises(ValueError, review.evaluate_images, self.context, self.output)
            self.assertRaises(RuntimeError, review.evaluate_images, self.context, self.output)
        self.assertEqual(call.call_count, 1)
        saved = json.loads((self.output/'review.json').read_text())
        self.assertIn('proposed_review', saved)
        def wrong_board(source, output, prompt, **kwargs):
            payload, provenance = self.transport(source, output, prompt, **kwargs)
            provenance['source']['sha256'] = 'wrong'
            return payload, provenance
        with patch('vision_transport.request_vision_json', side_effect=wrong_board):
            self.assertRaises(ValueError, review.evaluate_images, self.context, self.root/'wrong-board')

    def test_raw_source_uses_actual_decoder_and_frozen_mapping_detects_changes(self):
        raw = self.root/'original.ARW'; raw.write_bytes(b'synthetic-raw')
        context = copy.deepcopy(self.context)
        context['images'][0] = {'role': 'source', 'path': str(raw), 'sha256': sha256_file(raw)}
        from raw_support import open_rgb
        def decoder(path):
            return Image.new('RGB', (32, 48), 'red') if Path(path) == raw else open_rgb(path)
        with patch('raw_support.open_rgb', side_effect=decoder) as decode, \
                patch('vision_transport.request_vision_json', side_effect=self.transport) as call:
            review.evaluate_images(context, self.output)
            self.assertTrue(any(Path(args[0]) == raw for args, _kwargs in decode.call_args_list))
            raw.write_bytes(b'changed original')
            self.assertRaises(ValueError, review.evaluate_images, context, self.output)
        self.assertEqual(call.call_count, 1)

    def test_saved_run_reviews_once_and_preserves_incomplete_draft_and_human_gate(self):
        run = self.run_record(); before = self.candidate.read_bytes()
        with patch('vision_transport.request_vision_json', side_effect=self.transport) as call:
            review.review_saved_run(self.root, run, SimpleNamespace())
            review.review_saved_run(self.root, run, SimpleNamespace())
        self.assertEqual(call.call_count, 1)
        self.assertEqual(run['photo_edit_session']['reserved_budget_usage']['vision_requests'], 1)
        self.assertEqual(run['photo_edit_session']['reserved_budget_usage']['image_submissions'], 0)
        self.assertEqual(run['photo_edit_session']['historical_design_requests'], 1)
        self.assertEqual(run['photo_edit_session']['historical_image_submissions'], 1)
        self.assertEqual(run['photo_edit_session']['origin_request_budget']['reserved_vision_requests'], 2)
        self.assertEqual(run['photo_edit_session']['status'], 'stopped')
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertFalse(run['photo_edit_session']['human_approved'])
        self.assertEqual(run['status'], 'needs_visual_review')
        self.assertEqual(run['review_gate'], 'visual_review_required')
        self.assertEqual(self.candidate.read_bytes(), before)

    def test_service_veto_and_missing_budget_never_call_transport(self):
        run = self.run_record(enabled=False)
        with patch('vision_transport.request_vision_json') as call:
            review.review_saved_run(self.root, run, SimpleNamespace())
            self.assertEqual(run['photo_edit_session']['status'], 'not_requested')
            run = self.run_record(); run['task_plan']['services']['text_vision_requests_upper_bound'] = None
            review.review_saved_run(self.root, run, SimpleNamespace())
            self.assertEqual(run['photo_edit_session']['status'], 'needs_attention')
        self.assertFalse(call.called)

    def test_unknown_request_is_retained_and_never_implicitly_resubmitted(self):
        run = self.run_record(); before = self.candidate.read_bytes()
        with patch('vision_transport.request_vision_json', side_effect=TimeoutError('Provider outcome unknown')) as call:
            review.review_saved_run(self.root, run, SimpleNamespace())
            review.review_saved_run(self.root, run, SimpleNamespace())
        self.assertEqual(call.call_count, 1)
        self.assertEqual(run['photo_edit_session']['status'], 'needs_attention')
        self.assertEqual(self.candidate.read_bytes(), before)
        self.assertTrue(Path(run['photo_edit_session']['path']).is_file())


if __name__ == '__main__':
    unittest.main()
