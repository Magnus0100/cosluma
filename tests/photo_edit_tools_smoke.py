"""Actual session reservations/provider registration/local pixels; fake transport only."""
from __future__ import annotations

import copy
import hashlib
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image,ImageDraw

from asset_identity import sha256_file
import photo_edit_session as session
import photo_edit_tools
from photo_edit_tools import make_tools
from photo_editing import editing_scope,validate_observed_design
from photo_appearance import create_source_baseline,load_asset
from float_appearance import srgb_to_linear


class EditingToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='cosplay_edit_tools_')
        self.root=Path(self.temporary.name)
        self.environment=patch.dict(os.environ,{'COSPLAY_ASSET_IDENTITY_CACHE':str(self.root/'identities'),
            'COSPLAY_CONFIG_ROOT':str(self.root/'config'),'TOOL_TEST_KEY':'synthetic-unused-secret'})
        self.environment.start()
        self.source=self.root/'source.png';Image.new('RGB',(96,128),(30,45,60)).save(self.source)
        y,x=np.indices((128,96));self.base=np.stack(((x*2+20)%255,(y+40)%255,(x+y+60)%255),axis=-1).astype(np.uint8)
        self.best=self.root/'best.png';Image.fromarray(self.base).save(self.best)
        self.initial=self.root/'initial-generated.png';Image.fromarray(self.base).resize((48,64)).save(self.initial)
        self.scope=editing_scope({'language':'en'})
        self.design=validate_observed_design({'summary':'A coherent indoor portrait with meaningful scene edits',
            'observation':{'setting':'indoor','description':'Portrait in an existing interior'},
            'changes':[{'operation':'relight','target':'person and scene','instruction':'Unify key illumination.',
                        'purpose':'Credible subject emphasis'}], 'lighting':'A coherent soft key and restrained ambient fill',
            'keep':['Identity, body shape, costume, props and indoor setting'], 'uncertain':[]},self.scope)
        Image.fromarray(self.base).save(self.source)
        self.source_float=srgb_to_linear(np.float32(self.base)/255)+np.float32(.0000001)
        self.binding={'source_sha256':sha256_file(self.source),'scope':self.scope,'design_id':self.design['design_id']}
        with patch('raw_support.open_linear',return_value=(self.source_float.copy(),{
                'source_precision':'raw16_linear','decode_version':'oriented-srgb-icc-linearraw-v3'})):
            self.appearance=create_source_baseline(self.root/'baseline',source_path=self.source,**self.binding)
        baseline=load_asset(self.appearance,**self.binding)
        self.source_float=baseline['master'].copy();self.best=baseline['preview_path']
        from photo_edit_tools import _close_loaded
        _close_loaded(baseline)
        self.directory=self.root/'session'
        session.open_session(self.directory,source=self.source,scope=self.scope,design=self.design,
            execution_identity='checked-tool-fixture',producer_identity='coordinator-fixture',
            editing_initialization=session.INITIALIZATION,variant_id='v1',source_appearance=self.appearance,
            budget={'image_submissions':1,'vision_requests':4,'local_repairs':1,'tool_calls':1,'improvement_steps':1})
        session.seed_source_baseline(self.directory,self.appearance)
        self.cancelled=False
        self.provider_options={'provider':'openai-compatible','model':'fixture-image',
            'api_base':'https://example.invalid/v1','api_key_env':'TOOL_TEST_KEY'}
        self.tools=make_tools(self.root,self.root/'tools',scope=self.scope,session_directory=self.directory,
            provider_options=self.provider_options,cancel=lambda:self.cancelled)
        self.uploads=[]

    def tearDown(self):
        self.environment.stop();self.temporary.cleanup()

    def evidence(self,context):
        return {'observed_images':[{'role':row['role'],'sha256':row['sha256']} for row in context['images']],
            'provenance':{'model':'fixture-vision','model_requests':1,'generation_used':False}}

    def review(self,context):
        return {'credibility':{'status':'pass','findings':['Synthetic observed preserved geometry']},
            'creative_benefit':{'status':'fail','findings':['Synthetic missing local lighting edit']},
            'comparison':'first_candidate','primary_issue':'The local material remains unlit',**self.evidence(context)}

    def prepare_decision(self,kind='edit_region',region=None):
        session.advance(self.directory,tools=self.tools,decide=lambda c:None,evaluate=self.review,model_identity='fixture')
        def decide(context):
            result={'action':'tool','tool':kind,'operations':['relight'],
                'instruction':'Improve the observed material illumination and retain the existing scene.',
                'reason':'The independent review identified the unlit material.',**self.evidence(context)}
            if kind=='edit_region':result['region']=[.25,.25,.75,.75] if region is None else region
            return result
        session.advance(self.directory,tools=self.tools,decide=decide,evaluate=self.review,model_identity='fixture')

    def transport(self,endpoint,key,image,model,prompt,size,timeout,**kwargs):
        state=json.loads((self.directory/'session.json').read_text('utf8'))
        attempt=state['attempts'][-1]
        self.assertEqual(attempt['status'],'reserved')
        self.assertEqual(attempt['costs']['image_submissions'],1)
        self.assertEqual(state['used']['image_submissions'],1)
        own=json.loads((self.root/'tools'/attempt['attempt_id']/'tool-attempt.json').read_text('utf8'))
        self.assertEqual(own['state'],'submitting')
        with Image.open(image) as actual:
            actual.load();self.uploads.append(np.asarray(actual).copy())
            result=Image.new('RGB',actual.size,(220,100,70))
        stream=io.BytesIO();result.save(stream,'PNG')
        return {'id':'synthetic-response','data':[{'b64_json':'transport-is-mocked'}]},stream.getvalue(),'b64_json'

    def execute(self):
        with redirect_stdout(io.StringIO()):
            return session.advance(self.directory,tools=self.tools,decide=lambda c:None,evaluate=self.review,model_identity='fixture')

    def picture_transport(self,picture):
        stream=io.BytesIO();picture.save(stream,'PNG',exif=picture.getexif())
        pixels=stream.getvalue();self.raw_response_sha=hashlib.sha256(pixels).hexdigest()
        def transport(*args,**kwargs):
            payload,_,mode=self.transport(*args,**kwargs)
            return payload,pixels,mode
        return transport

    def execute_picture(self,picture,kind='edit_region'):
        self.prepare_decision(kind)
        with patch('reference_provider.request_edit',side_effect=self.picture_transport(picture)) as call:
            state=self.execute()
        self.assertEqual(call.call_count,1)
        return state,state['candidates'][-1]

    def test_low_resolution_alpha_is_premultiplied_before_registration_without_colour_halos(self):
        from photo_edit_tools import _mask,_generated_pixels,_close_loaded
        values=np.empty((48,40,4),np.uint8);values[:]=[255,0,255,0]
        values[:,:20]=[204,102,51,128]
        state,row=self.execute_picture(Image.fromarray(values))
        native=load_asset(row['appearance_asset'],**self.binding)
        try:
            alpha=np.asarray(Image.fromarray(values[...,3].astype(np.float32)/255).resize((80,96),Image.Resampling.BILINEAR))
            coverage=np.zeros((128,96),np.float32);coverage[32:96,24:72]=alpha[16:80,16:64]*_mask((48,64))
            colour=srgb_to_linear(np.asarray([204,102,51],np.float32)/255)
            expected=self.source_float*(1-coverage[...,None])+colour*coverage[...,None]
            np.testing.assert_allclose(native['master'],expected,rtol=0,atol=1e-7)
            np.testing.assert_array_equal(native['edit_coverage'],coverage)
            np.testing.assert_array_equal(native['generated_coverage'],coverage)
            np.testing.assert_array_equal(native['master'][coverage==0],self.source_float[coverage==0])
            self.assertTrue(np.any((coverage>0)&(coverage<1)))
            registered=Path(row['provenance']['generated_image']['path'])
            with Image.open(registered) as image:
                self.assertEqual(image.mode,'RGBA');np.testing.assert_array_equal(np.asarray(image)[...,3],values[...,3])
            receipt=json.loads(Path(row['provenance']['provider_receipt']['path']).read_text('utf8'))
            self.assertTrue(receipt['alpha']['preserved'])
            self.assertEqual(receipt['provider_contract']['response']['output_sha256'],self.raw_response_sha)
            self.assertEqual(native['receipt']['generation']['image']['sha256'],sha256_file(registered))
            other=values.copy();other[other[...,3]==0,:3]=0
            black=self.root/'hidden-black.png';Image.fromarray(other).save(black)
            _,associated_a,alpha_a=_generated_pixels(registered,(80,96))
            _,associated_b,alpha_b=_generated_pixels(black,(80,96))
            np.testing.assert_array_equal(associated_a,associated_b);np.testing.assert_array_equal(alpha_a,alpha_b)
        finally:_close_loaded(native)
        self.assertEqual(state['used']['image_submissions'],1)

    def test_joint_alpha_composes_over_parent_and_owned_reopen_keeps_true_generated_coverage(self):
        from photo_edit_tools import _generated_pixels,_close_loaded
        from photo_session_evidence import freeze_session_evidence,resolve_asset
        from current_model_registry import atomic_json
        values=np.empty((64,48,4),np.uint8);values[:]=[255,0,255,0];values[:,:24]=[180,115,60,128]
        state,row=self.execute_picture(Image.fromarray(values),'joint_edit')
        native=load_asset(row['appearance_asset'],**self.binding)
        try:
            registered=Path(row['provenance']['generated_image']['path'])
            _,associated,alpha=_generated_pixels(registered,(96,128))
            expected=self.source_float.copy();active=alpha>0
            expected[active]=(self.source_float*(1-alpha[...,None])+associated)[active]
            np.testing.assert_array_equal(native['master'],expected)
            np.testing.assert_array_equal(native['generated_coverage'],alpha)
            np.testing.assert_array_equal(native['source_coverage'],1-alpha)
            np.testing.assert_array_equal(native['master'][~active],self.source_float[~active])
            self.assertEqual(native['receipt']['kind'],'mixed_regional')
            self.assertEqual(native['receipt']['generation']['core_box'],[0,0,96,128])
            self.assertEqual(native['receipt']['generation']['context_box'],[0,0,96,128])
            self.assertEqual(row['resources']['local_repairs'],0)
            composition=json.loads(Path(row['provenance']['composition_receipt']['path']).read_text('utf8'))
            self.assertEqual(composition['edit_permission'],'full_canvas')
            self.assertEqual(composition['composition_mode'],'linear_premultiplied_rgba_over_parent')
            saved= native['master'].copy();generated=native['generated_coverage'].copy()
        finally:_close_loaded(native)
        def accept(context):
            return {'credibility':{'status':'pass','findings':['Synthetic completed RGBA composition.']},
                'creative_benefit':{'status':'pass','findings':['Synthetic visible edit.']},
                'comparison':'better','primary_issue':'none',**self.evidence(context)}
        state=session.advance(self.directory,tools=self.tools,decide=lambda c:None,evaluate=accept,model_identity='fixture')
        self.assertEqual(session.selected_output(self.directory)['candidate']['appearance_asset']['asset_id'],row['appearance_asset']['asset_id'])
        owned=self.root/'owned';owned.mkdir()
        state['evidence']=freeze_session_evidence(self.directory,owned/'evidence',state=state)
        atomic_json(owned/'session.json',state)
        shutil.rmtree(self.root/'tools');shutil.rmtree(self.root/'baseline');shutil.rmtree(self.directory);self.source.unlink()
        with patch('raw_support.open_linear',side_effect=AssertionError('Frozen appearance must not decode the deleted parent')):
            reopened=session.snapshot(owned);actual=load_asset(resolve_asset(reopened,row['appearance_asset']),**self.binding)
        try:
            self.assertEqual(reopened['best_candidate_id'],row['candidate_id'])
            np.testing.assert_array_equal(actual['master'],saved);np.testing.assert_array_equal(actual['generated_coverage'],generated)
            self.assertEqual(reopened['used']['image_submissions'],1)
        finally:_close_loaded(actual)

    def test_joint_near_opaque_alpha_does_not_claim_full_generated_ownership(self):
        from photo_edit_tools import _close_loaded
        values=np.full((64,48,4),[180,100,65,254],np.uint8);values[:,::2,3]=253
        _,row=self.execute_picture(Image.fromarray(values),'joint_edit')
        native=load_asset(row['appearance_asset'],**self.binding)
        try:
            self.assertEqual(native['receipt']['kind'],'mixed_regional')
            self.assertTrue(np.all(native['generated_coverage']>0));self.assertTrue(np.all(native['generated_coverage']<1))
            self.assertTrue(np.all(native['source_coverage']>0))
        finally:_close_loaded(native)

    def test_opaque_rgba_matches_existing_rgb_resampling_exactly(self):
        from photo_edit_tools import _resample,_close_loaded
        values=np.full((64,48,4),[180,100,65,255],np.uint8)
        _,row=self.execute_picture(Image.fromarray(values),'joint_edit')
        native=load_asset(row['appearance_asset'],**self.binding)
        try:
            self.assertEqual(native['receipt']['kind'],'generated_full')
            self.assertEqual(native['generated_coverage'],1)
            np.testing.assert_array_equal(native['master'],_resample(Image.fromarray(values[...,:3]),(96,128)))
        finally:_close_loaded(native)

    def test_fully_transparent_response_retains_paid_evidence_and_cannot_retry(self):
        self.prepare_decision();picture=Image.new('RGBA',(80,96),(255,0,255,0))
        with patch('reference_provider.request_edit',side_effect=self.picture_transport(picture)) as call:
            with self.assertRaisesRegex(ValueError,'visible coverage'):self.execute()
            state=self.execute()
        self.assertEqual(call.call_count,1);self.assertEqual(len(state['candidates']),1)
        self.assertEqual(state['used']['image_submissions'],1);self.assertEqual(state['status'],'needs_attention')
        attempt=next((self.root/'tools').glob('*/tool-attempt.json'))
        record=json.loads(attempt.read_text('utf8'));self.assertEqual(record['state'],'needs_attention')
        self.assertEqual(sha256_file(record['generated']['path']),record['generated']['sha256'])
        self.assertTrue(Path(record['provider_receipt']['path']).is_file());self.assertFalse((attempt.parent/'appearance').exists())

    def alpha_format(self,mode):
        alpha=np.zeros((40,48),np.uint8);alpha[:,:24]=128
        if mode=='P':
            picture=Image.fromarray((alpha>0).astype(np.uint8),'P')
            picture.putpalette([255,0,255,180,115,60]+[0,0,0]*254)
            picture.info['transparency']=bytes([0,128])
        else:picture=Image.fromarray(np.stack((np.full_like(alpha,77),alpha),axis=-1),'LA')
        picture.getexif()[274]=6
        _,row=self.execute_picture(picture)
        path=Path(row['provenance']['generated_image']['path'])
        with Image.open(path) as image:
            self.assertEqual(image.mode,'RGBA');self.assertEqual(image.size,(40,48))
            np.testing.assert_array_equal(np.asarray(image)[...,3],np.rot90(alpha,k=3))
        receipt=json.loads(Path(row['provenance']['provider_receipt']['path']).read_text('utf8'))
        self.assertTrue(receipt['alpha']['preserved'])
        self.assertEqual(receipt['provider_contract']['response']['output_sha256'],self.raw_response_sha)

    def test_palette_transparency_registration_keeps_oriented_alpha(self):self.alpha_format('P')

    def test_luminance_alpha_registration_keeps_oriented_alpha(self):self.alpha_format('LA')

    def test_region_is_current_best_context_and_outside_pixels_are_exact(self):
        self.prepare_decision()
        parent=session.snapshot(self.directory)['candidates'][0]['image']
        with patch('reference_provider.request_edit',side_effect=self.transport) as call:
            state=self.execute()
        self.assertEqual(call.call_count,1)
        self.assertEqual(len(state['candidates']),2)
        row=state['candidates'][-1]
        with Image.open(row['image']['path']) as actual:result=np.asarray(actual)
        inside=np.zeros(self.base.shape[:2],bool);inside[32:96,24:72]=True
        np.testing.assert_array_equal(result[~inside],self.base[~inside])
        self.assertTrue(np.any(result[inside]!=self.base[inside]))
        np.testing.assert_array_equal(self.uploads[0],self.base[16:112,8:88])
        self.assertEqual(sha256_file(parent['path']),parent['sha256'])
        composition=json.loads(Path(row['provenance']['composition_receipt']['path']).read_text('utf8'))
        self.assertTrue(composition['outside_core_pixels_identical_to_current_best'])
        self.assertFalse(composition['source_subject_RGB_restored'])
        native=load_asset(row['appearance_asset'],**self.binding)
        self.assertEqual(native['receipt']['kind'],'mixed_regional')
        self.assertFalse(native['receipt']['native_generated_detail'])
        untouched=native['master'][~inside].copy()
        from photo_edit_tools import _close_loaded
        _close_loaded(native)
        np.testing.assert_array_equal(untouched,self.source_float[~inside])
        self.assertEqual(native['receipt']['source_origin']['source_precision'],'raw16_linear')
        self.assertTrue(np.any(native['source_coverage']<1))
        from raw_support import open_linear
        roundtrip,_=open_linear(self.best)
        self.assertFalse(np.array_equal(untouched,roundtrip[~inside]))

    def test_joint_edit_uses_the_reviewed_best_and_one_submission(self):
        self.provider_options['api_base']='https://mutated.invalid/v1'
        self.prepare_decision('joint_edit')
        with patch('reference_provider.request_edit',side_effect=self.transport) as call:state=self.execute()
        self.assertEqual(call.call_count,1)
        self.assertEqual(call.call_args.args[0],'https://example.invalid/v1/images/edits')
        submitted=call.call_args.args[4]
        self.assertIn('Improve the observed material illumination',submitted)
        self.assertNotIn(self.design['summary'],submitted)
        self.assertNotIn(self.design['lighting'],submitted)
        self.assertNotIn('Retain established illumination',submitted)
        self.assertIn('expression',submitted)
        np.testing.assert_array_equal(self.uploads[0],self.base)
        row=state['candidates'][-1]
        self.assertEqual(row['size'],[96,128])
        native=load_asset(row['appearance_asset'],**self.binding)
        self.assertEqual(native['receipt']['kind'],'generated_full')
        self.assertEqual(native['source_coverage'],0)
        self.assertFalse(native['receipt']['native_generated_detail'])
        self.assertEqual(state['used']['image_submissions'],1)

    def test_initial_pass_pass_baseline_still_allows_real_decision_without_fake_failed_review(self):
        def passed(context):
            return {'credibility':{'status':'pass','findings':['Visible source is credible']},
                'creative_benefit':{'status':'pass','findings':['Its original composition is useful']},
                'comparison':'first_candidate','primary_issue':'none',**self.evidence(context)}
        session.advance(self.directory,tools=self.tools,decide=lambda c:None,evaluate=passed,model_identity='fixture')
        self.assertFalse(session.snapshot(self.directory)['automated_target_met'])
        def decide(context):
            return {'action':'tool','tool':'edit_region','operations':['scene_edit'],
                'instruction':'Create a restrained translucent local material behind the existing frame.',
                'reason':'The checked user design has not yet been executed.','region':[.25,.25,.75,.75],**self.evidence(context)}
        session.advance(self.directory,tools=self.tools,decide=decide,evaluate=passed,model_identity='fixture')
        with patch('reference_provider.request_edit',side_effect=self.transport) as call:self.execute()
        state=session.snapshot(self.directory)
        self.assertEqual(call.call_count,1)
        self.assertEqual(state['used']['improvement_steps'],0)
        self.assertEqual(state['candidates'][0]['review']['primary_issue'],'none')
        self.assertEqual(state['candidates'][-1]['appearance_kind'],'mixed_regional')

    def test_retouch_does_not_advertise_whole_image_redrawing_that_asset_permissions_reject(self):
        scope=editing_scope({'work_type':'retouch'})
        tools=make_tools(self.root,self.root/'retouch-tools',scope=scope,session_directory=self.directory)
        self.assertNotIn('joint_edit',[tool.name for tool in tools])
        regional=next(tool for tool in tools if tool.name=='edit_region')
        self.assertTrue(regional.requires_region)
        self.assertNotIn('scene_edit',regional.operations)
        self.assertNotIn('effect',regional.operations)

    def test_duplicate_callback_for_same_reserved_attempt_never_resubmits(self):
        self.prepare_decision()
        def duplicate(*args,**kwargs):
            current=json.loads((self.directory/'session.json').read_text('utf8'))
            context=current['attempts'][-1]['context']
            with self.assertRaises(FileExistsError):self.tools[1].execute(context)
            return self.transport(*args,**kwargs)
        with patch('reference_provider.request_edit',side_effect=duplicate) as call:self.execute()
        self.assertEqual(call.call_count,1)

    def test_timeout_reservation_cannot_be_reused_or_retried(self):
        self.prepare_decision()
        with patch('reference_provider.request_edit',side_effect=TimeoutError('unknown outcome')) as call:
            with self.assertRaises(TimeoutError):self.execute()
            repeated=self.execute()
            self.assertEqual(repeated['status'],'needs_attention')
        self.assertEqual(call.call_count,1)
        state=session.snapshot(self.directory)
        self.assertEqual(state['status'],'needs_attention')
        self.assertEqual(state['used']['image_submissions'],1)
        self.assertEqual(len(state['candidates']),1)

    def test_late_cancellation_keeps_response_evidence_and_old_best(self):
        self.prepare_decision()
        before=session.snapshot(self.directory)['candidates'][0]['image']
        def late(*args,**kwargs):
            result=self.transport(*args,**kwargs);self.cancelled=True;return result
        with patch('reference_provider.request_edit',side_effect=late):
            with self.assertRaises(InterruptedError):self.execute()
        state=session.snapshot(self.directory)
        self.assertEqual(len(state['candidates']),1)
        self.assertEqual(sha256_file(before['path']),before['sha256'])
        own=next((self.root/'tools').glob('*/tool-attempt.json'))
        record=json.loads(own.read_text('utf8'))
        self.assertEqual(record['state'],'interrupted')
        self.assertTrue(record.get('generated'))
        self.assertFalse((own.parent/'candidate.png').exists())

    def test_bad_provider_framing_and_bad_image_never_publish_a_complete_candidate(self):
        self.prepare_decision()
        def malformed(*args,**kwargs):return {'id':'bad'},b'not an image','b64_json'
        with patch('reference_provider.request_edit',side_effect=malformed) as call:
            with self.assertRaises(Exception):self.execute()
        self.assertEqual(call.call_count,1)
        state=session.snapshot(self.directory)
        self.assertEqual(len(state['candidates']),1)
        self.assertEqual(state['used']['image_submissions'],1)

    def test_forged_or_changed_unreserved_context_is_rejected_without_transport(self):
        from photo_edit_tools import edit_region
        self.prepare_decision()
        current=session.snapshot(self.directory)
        # The next tool call has not been reserved yet. No caller may derive
        # permission from a model decision or remaining_budget alone.
        context=current['attempts'][-1]['context']
        config={'scope':self.scope,'session_directory':str(self.directory),'root':str(self.root),'output':str(self.root/'tools')}
        with patch('reference_provider.request_edit',side_effect=AssertionError('submitted')):
            with self.assertRaises(ValueError):edit_region(context,config)

    def test_changed_best_during_transport_prevents_candidate_publication(self):
        self.prepare_decision()
        def changed(*args,**kwargs):
            response=self.transport(*args,**kwargs)
            current=json.loads((self.directory/'session.json').read_text('utf8'))
            best=current['attempts'][-1]['context']['images'][1]
            Image.new('RGB',(96,128),'black').save(best['path'])
            return response
        with patch('reference_provider.request_edit',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'changed during generation'):self.execute()
        self.assertFalse(list((self.root/'tools').glob('*/candidate.png')))

    def test_invalid_region_and_incompatible_provider_shape_never_publish(self):
        from photo_edit_tools import _region,_fit_context
        for region in ([0,0,1,1],[.8,.2,.2,.7],[0,0,float('nan'),.5],[-.1,0,.5,.5],[0,0,.001,.001]):
            with self.subTest(region=region),self.assertRaises(ValueError):_region(region,(96,128))
        core,context=_region([.25,.25,.75,.75],(96,128))
        fitted=_fit_context(core,context,(96,128),2/3)
        self.assertLessEqual(fitted[0],core[0]);self.assertGreaterEqual(fitted[2],core[2])
        self.assertLessEqual(fitted[1],core[1]);self.assertGreaterEqual(fitted[3],core[3])
        self.prepare_decision()
        def wrong(*args,**kwargs):
            image=io.BytesIO();Image.new('RGB',(120,24),'red').save(image,'PNG')
            return {'id':'bad-framing'},image.getvalue(),'b64_json'
        with patch('reference_provider.request_edit',side_effect=wrong):
            with self.assertRaisesRegex(ValueError,'framing'):self.execute()
        self.assertFalse(list((self.root/'tools').glob('*/candidate.png')))

    def test_two_local_objects_do_not_receive_full_photo_portrait_goals(self):
        for name,instruction in (
                ('ceramic-cup','Soften the excessive specular patch on the visible ceramic cup and match its existing defocus.'),
                ('window-frame','Repair the bright scratch on the visible wooden window frame while matching the adjacent grain.')):
            with self.subTest(object=name):
                folder=self.root/name;folder.mkdir()
                picture=Image.new('RGB',(96,128),(65,60,55));draw=ImageDraw.Draw(picture)
                if name=='ceramic-cup':
                    draw.ellipse((32,40,64,52),fill=(190,170,135));draw.rectangle((32,46,64,82),fill=(180,165,135))
                    draw.ellipse((55,50,74,74),outline=(190,170,135),width=3)
                else:
                    draw.rectangle((28,28,68,98),outline=(150,105,65),width=6)
                    draw.line((48,31,48,96),fill=(150,105,65),width=4)
                best=folder/'object-best.png';picture.save(best)
                self.directory=folder/'session'
                proposal={key:copy.deepcopy(self.design[key]) for key in ('summary','observation','changes','lighting','keep','uncertain')}
                proposal['summary']='A full-length woman in a blue qipao holding a brown handbag in an elaborate garden.'
                proposal['keep']=['Preserve the full-length woman, blue qipao, brown handbag and distant garden.']
                design=validate_observed_design(proposal,self.scope)
                tools=make_tools(self.root,self.root/'tools',scope=self.scope,session_directory=self.directory,
                    provider_options=self.provider_options)
                binding={'source_sha256':sha256_file(best),'scope':self.scope,'design_id':design['design_id']}
                baseline=create_source_baseline(folder/'baseline',source_path=best,**binding)
                session.open_session(self.directory,source=best,scope=self.scope,design=design,
                    editing_initialization=session.INITIALIZATION,variant_id='v1',source_appearance=baseline,
                    execution_identity='checked-'+name,producer_identity='fixture',
                    budget={'image_submissions':1,'vision_requests':3,'local_repairs':1,'tool_calls':1,'improvement_steps':1})
                session.seed_source_baseline(self.directory,baseline)
                session.advance(self.directory,tools=tools,decide=lambda c:None,evaluate=self.review,model_identity='fixture')
                def decide(context):
                    return {'action':'tool','tool':'edit_region','operations':['detail_repair'],
                        'instruction':instruction,'reason':'Independent review found the selected local material defect.',
                        'region':[.25,.25,.75,.75],**self.evidence(context)}
                session.advance(self.directory,tools=tools,decide=decide,evaluate=self.review,model_identity='fixture')
                prompts=[]
                def transport(*args,**kwargs):
                    prompts.append(args[4]);return self.transport(*args,**kwargs)
                with patch('reference_provider.request_edit',side_effect=transport),redirect_stdout(io.StringIO()):
                    state=session.advance(self.directory,tools=tools,decide=decide,evaluate=self.review,model_identity='fixture')
                self.assertEqual(len(prompts),1)
                self.assertIn(instruction,prompts[0])
                self.assertIn('cut-off objects',prompts[0])
                for leaked in ('full-length','woman','qipao','handbag','garden'):
                    self.assertNotIn(leaked,prompts[0])
                self.assertEqual(len(state['candidates']),2)

    def test_local_prompt_ignores_global_design_in_both_model_families_and_languages(self):
        from photo_edit_tools import _prompt
        for language in ('zh','en'):
            for family in ('qwen_image_2_1','gpt_image'):
                with self.subTest(language=language,family=family):
                    scope=editing_scope({'language':language})
                    decision={'instruction':'修复可见物体的局部反光。' if language=='zh' else 'Repair the visible local specular patch.'}
                    local=_prompt(scope,decision,[.1,.1,.8,.8],family)
                    self.assertNotIn('GLOBAL_',local);self.assertNotIn('UNSEEN_',local)
                    self.assertIn(decision['instruction'],local)
                    if family=='qwen_image_2_1':self.assertNotIn('\n',local)

    def test_local_prompt_retains_actual_scope_veto_and_setting_without_global_keep_lists(self):
        for language in ('zh','en'):
            for family in ('qwen_image_2_1','gpt_image'):
                decision={'instruction':'Adjust the visible local material only.'}
                scope=editing_scope({'language':language,'work_type':'partial_composite'})
                nofx=editing_scope({'language':language,'work_type':'partial_composite'},no_effects=True)
                retouch=editing_scope({'language':language,'work_type':'retouch'})
                from photo_edit_tools import _prompt
                normal=_prompt(scope,decision,[.1,.1,.8,.8],family)
                for restricted in (nofx,retouch):
                    text=_prompt(restricted,decision,[.1,.1,.8,.8],family)
                    self.assertNotEqual(text,normal)
                    self.assertIn('不新增叙事特效' if language=='zh' else 'Do not add narrative effects',text)
                    self.assertIn('室内外属性' if language=='zh' else 'indoor/outdoor setting',text)
                    self.assertNotIn('scene intervention or effect',text)
                    self.assertNotIn('GLOBAL_KEEP',text)
                self.assertIn('不重建场景' if language=='zh' else 'without scene reconstruction',
                    _prompt(retouch,decision,[.1,.1,.8,.8],family))
                for target in ('indoor','outdoor'):
                    scoped=editing_scope({'language':language,'work_type':'full_composite','target_setting':target})
                    for region in ([.1,.1,.8,.8],None):
                        text=_prompt(scoped,decision,region,family)
                        self.assertIn(('目标场景为'+('室内' if target=='indoor' else '室外')) if language=='zh'
                            else 'target setting is '+target,text)

    def test_joint_correction_can_change_light_or_focus_inside_each_scope(self):
        from photo_edit_tools import _prompt
        for language in ('zh','en'):
            for family in ('qwen_image_2_1','gpt_image'):
                for work in ('retouch','partial_composite','full_composite'):
                    with self.subTest(language=language,family=family,work=work):
                        scope=editing_scope({'language':language,'work_type':work})
                        instruction=('改变人物与场景受光，柔化后景，保留可读的衣料层次。' if language=='zh' else
                            'Change subject and scene lighting, soften the background and retain readable costume folds.')
                        prompt=_prompt(scope,{'instruction':instruction},None,family)
                        self.assertIn(instruction,prompt)
                        for conflict in ('保留当前整体受光','照明和景深保持','Retain established illumination',
                                         'depth of field unchanged','OBJECTIVE','旧审片'):
                            self.assertNotIn(conflict,prompt)
                        for protected in (('身份','表情','体型','道具','室内外') if language=='zh' else
                                          ('identity','expression','body shape','props','indoor/outdoor')):
                            self.assertIn(protected,prompt)
                        if work=='retouch':self.assertIn('不新增叙事特效' if language=='zh' else 'Do not add narrative effects',prompt)
                        if work=='partial_composite':self.assertIn('主要结构' if language=='zh' else 'main structure',prompt)
                        if family=='qwen_image_2_1':self.assertNotIn('\n',prompt)

    def test_subject_matte_comes_from_a_real_segmentation_and_is_recorded(self):
        from photo_edit_tools import _current_canvas_subject
        with Image.open(self.best) as raster:
            base=raster.convert('RGB')
        y,x=np.indices((base.height,base.width),np.float32)
        score=np.clip(1.2-(((x-base.width*.5)/(base.width*.3))**2
            +((y-base.height*.55)/(base.height*.25))**2),0.,1.).astype(np.float32)
        calls=[]
        def provider(*,root,canvas_path,image):
            calls.append((Path(root),Path(canvas_path),image.size))
            return score,{'producer':'fixture-subject-segmentation','model_sha256':'0'*64}
        config={'root':str(self.root),'subject_matte':provider}
        best={'path':str(self.best),'sha256':sha256_file(self.best)}
        matte,record=_current_canvas_subject(config,best,base)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1],self.best)
        self.assertTrue(record['available'])
        self.assertEqual(record['producer'],'fixture-subject-segmentation')
        self.assertEqual(record['current_canvas_sha256'],best['sha256'])
        self.assertFalse(record['matte_is_the_edit_coverage'])
        self.assertEqual(record['canvas_size'],[base.width,base.height])
        self.assertGreater(record['subject_fraction_of_canvas'],.1)
        self.assertLess(record['subject_fraction_of_canvas'],.9)
        self.assertEqual(matte.shape,(base.height,base.width))
        self.assertEqual(matte.dtype,np.float32)

    def test_the_composition_receipt_always_says_whether_the_reading_was_taken(self):
        # This fixture's region is below photographic-join support, so there is no
        # subject matte to compare against. The key must still be present and
        # explicitly null: 'not measured here' is a fact the reader can see.
        self.prepare_decision()
        with patch('reference_provider.request_edit',side_effect=self.transport):
            state=self.execute()
        row=state['candidates'][-1]
        composition=json.loads(Path(row['provenance']['composition_receipt']['path']).read_text('utf8'))
        self.assertIn('provider_readings',composition)
        self.assertIsNone(composition['provider_readings'])
        note=Path(row['provenance']['composition_receipt']['path']).with_name('quality-readings.md')
        self.assertTrue(note.exists())
        note_text=note.read_text('utf8')
        self.assertIn('本次未测',note_text)
        self.assertNotIn('判定：',note_text)
        self.assertNotIn('verdict',composition)
        self.assertTrue(composition['needs_visual_review'])
        self.assertFalse(composition['human_approved'])

    def test_the_recorded_readings_say_what_the_generated_picture_did(self):
        from photo_edit_tools import _recorded_provider_readings
        rng=np.random.default_rng(11)
        native=(rng.random((450,320,3),np.float32)*.4).astype(np.float32)
        patch=(rng.random((384,256,3),np.float32)*.4).astype(np.float32)
        box=[20,30,276,414];core=[60,80,236,364]
        matte=np.zeros((450,320),np.float32);matte[100:400,60:250]=1.
        readings=_recorded_provider_readings(native,patch,box,matte,core)
        self.assertEqual(readings['role'],'recorded_not_judged_no_calibrated_bound')
        self.assertIn(readings['reading_relative_to_passing_sample']['provider_subject_structure_ncc'],
            ('within','above','below'))
        self.assertGreaterEqual(readings['provider_detail_outside_matte_share'],0.)
        self.assertLessEqual(readings['provider_detail_outside_matte_share'],1.)
        self.assertGreaterEqual(readings['provider_subject_structure_ncc'],-1.)
        self.assertLessEqual(readings['provider_subject_structure_ncc'],1.)
        self.assertIn('not judged',readings['role'].replace('_',' '))
        # Without a subject matte there is nothing to read, and nothing is invented.
        self.assertIsNone(_recorded_provider_readings(native,patch,box,None,core))
        self.assertIsNone(_recorded_provider_readings(native,patch,box,matte,None))
        # A patch that does not fit its box is a caller error, not a composite failure.
        self.assertIsNone(_recorded_provider_readings(native,patch[:8,:8],box,matte,core))

    def test_edit_coverage_cannot_stand_in_for_a_subject_matte(self):
        from photo_edit_tools import _current_canvas_subject
        with Image.open(self.best) as raster:
            base=raster.convert('RGB')
        best={'path':str(self.best),'sha256':sha256_file(self.best)}
        def provider(value):
            return lambda *,root,canvas_path,image:(value,{'producer':'fixture-subject-segmentation'})
        for name,value,reason in (
                ('whole canvas coverage',np.ones((base.height,base.width),np.float32),'ValueError'),
                ('no subject at all',np.zeros((base.height,base.width),np.float32),'ValueError'),
                ('uncalibrated weights',np.full((base.height,base.width),2.,np.float32),'ValueError'),
                ('wrong canvas',np.ones((16,16),np.float32),'ValueError')):
            with self.subTest(name):
                matte,record=_current_canvas_subject({'root':str(self.root),
                    'subject_matte':provider(value)},best,base)
                self.assertIsNone(matte)
                self.assertFalse(record['available'])
                self.assertTrue(record['reason'].startswith(reason))
                self.assertIn('matte_is_the_edit_coverage',record)
                self.assertFalse(record['matte_is_the_edit_coverage'])
        # The production route asks the registered silhouette inference; a
        # missing model or runtime is recorded as missing instead of raised.
        with patch('source_alpha.source_silhouette',
                side_effect=RuntimeError('no registered silhouette model')):
            matte,record=_current_canvas_subject({'root':str(self.root)},best,base)
        self.assertIsNone(matte)
        self.assertFalse(record['available'])
        self.assertTrue(record['reason'].startswith('RuntimeError'))
        self.assertFalse(record['matte_is_the_edit_coverage'])
        # And it can be switched off explicitly without calling anything.
        with patch('source_alpha.source_silhouette',
                side_effect=AssertionError('the subject matte was not disabled')):
            matte,record=_current_canvas_subject({'root':str(self.root),'subject_matte':False},
                best,base)
        self.assertIsNone(matte)
        self.assertFalse(record['available'])
        self.assertEqual(record['reason'],'subject_matte_disabled_by_configuration')

    def test_the_rounds_authorized_operations_decide_who_owns_the_scene_side(self):
        """Ownership follows what the round was authorized to do, not the subject."""
        from photo_edit_tools import _scene_write_trigger
        composite=editing_scope({'work_type':'partial_composite'})
        retouch=editing_scope({'work_type':'retouch'})
        full=editing_scope({'work_type':'full_composite','target_setting':'outdoor'})
        cases=(
            (composite,['scene_edit'],False,'authorized_scene_operation:scene_edit'),
            (composite,['relight'],False,'authorized_scene_operation:relight'),
            (composite,['scene_cleanup'],False,'authorized_scene_operation:scene_cleanup'),
            (full,['scene_composite'],False,'authorized_scene_operation:scene_composite'),
            # A scene operation decides even where the scope preserves the scene.
            (retouch,['scene_cleanup'],False,'authorized_scene_operation:scene_cleanup'),
            # The person-only round is the only narrowing branch: no scene
            # operation at all, and a scope that preserves the scene.
            (retouch,['portrait_retouch'],True,'person_only_round:portrait_retouch'),
            (retouch,['detail_repair'],True,'person_only_round:detail_repair'),
            (retouch,['portrait_retouch','detail_repair'],True,
                'person_only_round:detail_repair,portrait_retouch'),
            # A scope that does not preserve the scene keeps the scene side
            # authorized even when the round names only person work.
            (composite,['portrait_retouch'],False,'scope_allows_scene_rebuild:local_edit'),
            (full,['portrait_retouch'],False,'scope_allows_scene_rebuild:rebuild'))
        for scope,operations,subject_only,trigger in cases:
            with self.subTest(operations=operations,policy=scope['scene_policy']):
                self.assertEqual(_scene_write_trigger(scope,{'operations':operations}),
                    (subject_only,trigger))
        # No operations at all is not person work: the scope decides.
        self.assertEqual(_scene_write_trigger(retouch,{}),(True,'person_only_round:'))
        self.assertEqual(_scene_write_trigger(composite,{}),
            (False,'scope_allows_scene_rebuild:local_edit'))
        # The declared semantics and the branch agree.
        self.assertEqual(set(photo_edit_tools.SUBJECT['scene_operations']),
            set(photo_edit_tools.SCENE_OPERATIONS))
        for name in photo_edit_tools.SCENE_OPERATIONS:
            self.assertEqual(photo_edit_tools.SUBJECT['operation_semantics'][name]
                .startswith('authorized_'),True)

    def test_regional_edit_records_the_subject_matte_boundary_it_used(self):
        self.prepare_decision()
        with patch('reference_provider.request_edit',side_effect=self.transport):
            state=self.execute()
        row=state['candidates'][-1]
        composition=json.loads(Path(row['provenance']['composition_receipt']['path']).read_text('utf8'))
        subject=composition['subject_matte']
        self.assertFalse(subject['available'])
        self.assertEqual(subject['boundary_source'],'authorized_core_below_photographic_join_support')
        self.assertFalse(subject['matte_is_the_edit_coverage'])
        # The join below its photographic support writes the reserved rectangle
        # without claiming a boundary of its own, so the receipt keeps the
        # declared reason instead of a generic rectangle claim.
        self.assertNotIn('boundary_source',composition['join']['subject'])
        # The round authorized relighting, which is a joint cross-boundary solve:
        # the scene side is content of this round, so the write is not narrowed
        # to a subject and the receipt records the ground it decided on.
        self.assertEqual(subject['scene_write_trigger'],'authorized_scene_operation:relight')
        self.assertEqual(subject['scene_operations_authorized'],['relight'])
        self.assertEqual(subject['scene_policy'],'local_edit')
        self.assertFalse(subject['subject_only_round'])
        self.assertEqual(subject['ownership_rule'],photo_edit_tools.SUBJECT['ownership_rule'])
        # A join that makes no boundary claim cannot be passed off as the source
        # of the receipt's boundary: the declaration stays authoritative.
        self.assertEqual(subject['authoritative'],'pre_join_declaration')
        self.assertEqual(subject['acquisition']['boundary_source'],
            'authorized_core_below_photographic_join_support')
        # Coverage is still bounded edit permission, not an inferred subject matte.
        self.assertEqual(composition['coverage_semantics'],
            'current_canvas_edit_coverage_not_source_subject_alpha')


if __name__=='__main__':unittest.main()
