"""Immutable history relocation, not a second session or model runner."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from asset_identity import sha256_file
from photo_editing import identity
from photo_session_evidence import freeze_session_evidence,resolve_file,resolve_asset,read_json,adjacent_file
from photo_appearance import validate_asset
from photo_source_session_smoke import NativeFixture


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.fx=NativeFixture(self,k=0)
        self.variant=self.fx.start()[0]
        self.directory=Path(self.variant['photo_edit_session']['path']).parent
        self.state=json.loads((self.directory/'session.json').read_text(encoding='utf8'))
        self.original=copy.deepcopy(self.state)
        self.destination=self.fx.root/'owned-evidence'

    def freeze(self):
        locator=freeze_session_evidence(self.directory,self.destination,state=self.state)
        self.assertEqual(self.state,self.original)
        self.state['evidence']=locator
        return locator

    def test_owned_closure_reads_after_original_run_removal_without_resigning_history(self):
        locator=self.freeze()
        parent=self.fx.root/'outputs/runs'/self.fx.run['run_id']
        self.assertTrue(parent.is_dir());shutil.rmtree(parent)
        binding={'source_sha256':self.state['binding']['source']['sha256'],
            'scope':self.state['scope'],'design_id':self.state['binding']['design_id']}
        for row in self.state['candidates']:
            actual=resolve_file(self.state,row['image'])
            self.assertEqual(actual['sha256'],row['image']['sha256'])
            self.assertTrue(Path(actual['path']).is_relative_to(self.destination))
            receipt=validate_asset(resolve_asset(self.state,row['appearance_asset']),**binding)
            self.assertEqual(receipt['files']['preview']['sha256'],row['image']['sha256'])
            if row.get('review'):
                board=row['review']['provenance']['review_board']
                self.assertEqual(sha256_file(resolve_file(self.state,board)['path']),board['sha256'])
                mapping=read_json(self.state,adjacent_file(self.state,board,'review-board.json'))
                self.assertEqual(mapping['board'],board)
        for old,new in zip(self.original['attempts'],self.state['attempts']):
            self.assertEqual(old,new)
            self.assertEqual(new['context_sha256'],identity(new['context']))
        self.assertEqual(self.original['binding'],self.state['binding'])
        self.assertEqual(locator['sha256'],sha256_file(locator['path']))

    def test_tampered_owned_image_and_index_do_not_fall_back_to_live_parent(self):
        self.freeze();row=self.state['candidates'][-1]
        path=Path(resolve_file(self.state,row['image'])['path']);path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaises(ValueError):resolve_file(self.state,row['image'])
        index=Path(self.state['evidence']['path']);index.write_bytes(index.read_bytes()+b'changed')
        with self.assertRaises(ValueError):resolve_asset(self.state,row['appearance_asset'])

    def test_partial_freeze_failure_never_publishes_or_mutates_parent(self):
        before=(self.directory/'session.json').read_bytes()
        with patch('photo_session_evidence.shutil.copyfile',side_effect=OSError('synthetic disk error')):
            with self.assertRaises(OSError):self.freeze()
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.destination.parent.glob('.owned-evidence.partial-*')))
        self.assertEqual((self.directory/'session.json').read_bytes(),before)

    def test_repeated_freeze_does_not_overwrite_evidence_or_refund_usage(self):
        locator=self.freeze();before=Path(locator['path']).read_bytes()
        with self.assertRaises(FileExistsError):
            freeze_session_evidence(self.directory,self.destination,state=self.state)
        self.assertEqual(Path(locator['path']).read_bytes(),before)
        self.assertEqual(self.state['used'],self.original['used'])

    def test_owned_locator_cannot_escape_even_with_a_recomputed_index_identity(self):
        self.freeze();row=self.state['candidates'][-1]
        index=Path(self.state['evidence']['path']);document=json.loads(index.read_text(encoding='utf8'))
        document['files'][row['image']['sha256']]['file']='../outside.png'
        document['evidence_id']=identity({k:v for k,v in document.items() if k!='evidence_id'})
        index.write_text(json.dumps(document),encoding='utf8')
        self.state['evidence'].update(sha256=sha256_file(index),evidence_id=document['evidence_id'])
        with self.assertRaises(ValueError):resolve_file(self.state,row['image'])

    def test_omitting_original_does_not_remove_required_visual_evidence_or_add_a_new_call(self):
        calls=len(self.fx.fx.image_requests),len(self.fx.fx.vision_requests)
        locator=freeze_session_evidence(self.directory,self.destination,state=self.state,include_source=False)
        self.state['evidence']=locator
        document=json.loads(Path(locator['path']).read_text(encoding='utf8'))
        visual={row['sha256'] for attempt in self.state['attempts'] for row in attempt['context']['images']}
        original=self.state['binding']['source']['sha256']
        if original not in visual:self.assertNotIn(original,document['files'])
        for row in self.state['candidates']:resolve_file(self.state,row['image'])
        self.assertEqual(calls,(len(self.fx.fx.image_requests),len(self.fx.fx.vision_requests)))

    def test_provider_receipt_is_opaque_even_when_it_describes_an_expired_quality_probe(self):
        from current_model_registry import atomic_json
        receipt=self.fx.root/'registered-provider-receipt.json'
        document={'status':'reference_candidate','asset':{'sha256':self.state['candidates'][-1]['image']['sha256']},
            'provider_contract':{'quality':{'metrics':{
                'path':str(self.fx.root/'expired-quality-probe.png'),'sha256':'a'*64}}}}
        atomic_json(receipt,document)
        self.state['archived_provider_receipt']={'path':str(receipt),'sha256':sha256_file(receipt)}
        atomic_json(self.directory/'session.json',self.state);self.original=copy.deepcopy(self.state)
        self.freeze()
        self.assertEqual(read_json(self.state,self.state['archived_provider_receipt']),document)
        index=json.loads(Path(self.state['evidence']['path']).read_text(encoding='utf8'))
        self.assertNotIn('a'*64,index['files'])


class PublicEvidenceTests(unittest.TestCase):
    def test_snapshot_copy_survives_parent_run_removal_and_starts_new_feedback(self):
        import scoped_production_smoke as fixture
        import photo_edit_session as session
        from single_source_workflow import copy_source_run
        fx=fixture.ScopedProductionTests(methodName='runTest');fx.setUp();self.addCleanup(fx.doCleanups)
        _,parent=fx.start(fx.request(fx.sources[0],run_id='copy-evidence-origin'))
        original=session.snapshot(Path(parent['photo_edit_session']['path']).parent)
        original_manifest=fx.root/'outputs/runs'/parent['run_id']/'manifest.json'
        parent_bytes=original_manifest.read_bytes()
        copied=copy_source_run(fx.root,parent['run_id'])
        self.assertEqual(original_manifest.read_bytes(),parent_bytes)
        self.assertEqual(copied['photo_edit_session']['independent_evaluation'],parent['photo_edit_session']['independent_evaluation'])
        shutil.rmtree(original_manifest.parent)
        state=session.snapshot(Path(copied['photo_edit_session']['path']).parent)
        self.assertEqual(state['binding'],original['binding']);self.assertEqual(state['attempts'],original['attempts'])
        body=fx.request(fx.sources[0],run_id='feedback-from-owned-copy',sample=copied['sample'])
        body.pop('editing_autofinish');body['creative_task']['mode']='existing_reference'
        body.update(parent_run_id=copied['run_id'],reference=copied['artifacts']['auto_draft'],
            editing_iteration={'maximum_improvements':1,'user_feedback':'Improve coherent scene lighting while preserving the scene.'})
        counts=len(fx.image_requests),len(fx.vision_requests)
        checked=fx.backend.task_plan(body);self.assertTrue(checked['plan']['ready'])
        started=fx.backend.start_task({**checked['request'],'plan_id':checked['plan']['plan_id'],'async':False})
        self.assertTrue(started.get('ok'),started.get('stderr'))
        self.assertEqual((len(fx.image_requests)-counts[0],len(fx.vision_requests)-counts[1]),(1,2))
        self.assertEqual(started['run']['result_publication']['state'],'committed')
        self.assertFalse(started['run']['photo_edit_session']['human_approved'])

    def test_selected_child_owns_history_then_feedback_runs_after_entire_origin_run_deletion(self):
        import scoped_production_smoke as fixture
        import photo_edit_session as session
        from export_artifact import export_source_run
        fx=fixture.ScopedProductionTests(methodName='runTest');fx.setUp();self.addCleanup(fx.doCleanups)
        _,parent=fx.start(fx.request(fx.sources[0],run_id='evidence-origin',variant_count=2,defer_reference_selection=True))
        old_directory=Path(parent['variants'][0]['photo_edit_session']['path']).parent
        original=session.snapshot(old_directory)
        child=fx.backend.select_source_variant(parent['run_id'],{'variant_id':'v1','async':False})['run']
        directory=Path(child['photo_edit_session']['path']).parent
        owned=session.snapshot(directory)
        for old,new in zip(original['attempts'],owned['attempts']):
            self.assertEqual(old,new)
        parent_dir=fx.root/'outputs/runs'/parent['run_id']
        self.assertTrue(parent_dir.resolve().is_relative_to(fx.root/'outputs/runs'))
        shutil.rmtree(parent_dir)
        self.assertEqual(session.snapshot(directory)['session_id'],original['session_id'])
        exported=export_source_run(fx.root,child['run_id'],directory=str(fx.root/'exports'),format='PNG')
        self.assertTrue(exported['ok'])
        body=fx.request(fx.sources[0],run_id='feedback-without-origin');body.pop('editing_autofinish')
        body['creative_task']['mode']='existing_reference'
        body.update(parent_run_id=child['run_id'],reference=child['artifacts']['auto_draft'],
            editing_iteration={'maximum_improvements':1,'user_feedback':'Improve coherent lighting across the composition.'})
        before=len(fx.image_requests),len(fx.vision_requests)
        checked=fx.backend.task_plan(body);self.assertTrue(checked['plan']['ready'])
        started=fx.backend.start_task({**checked['request'],'plan_id':checked['plan']['plan_id'],'async':False})
        self.assertTrue(started.get('ok'),started.get('stderr'))
        fresh=started['run'];self.assertEqual(fresh['result_publication']['state'],'committed')
        self.assertEqual((len(fx.image_requests)-before[0],len(fx.vision_requests)-before[1]),(1,2))
        self.assertFalse(fresh['photo_edit_session']['human_approved'])
        shutil.rmtree(fx.root/'outputs/runs'/child['run_id'])
        session.snapshot(Path(fresh['photo_edit_session']['path']).parent)
        self.assertTrue(export_source_run(fx.root,fresh['run_id'],directory=str(fx.root/'fresh-export'),format='PNG')['ok'])

    def test_reviewed_working_cpu_uses_owned_evidence_after_original_parent_run_deletion(self):
        from photo_focus_iteration_smoke import FocusIterationTests
        import photo_edit_session as session
        fx=FocusIterationTests(methodName='runTest');fx.setUp();self.addCleanup(fx.doCleanups)
        original=fx.fx.parent
        original_dir=fx.root/'outputs/runs'/original['run_id']
        self.assertNotEqual(original['run_id'],fx.parent['run_id'])
        shutil.rmtree(original_dir)
        state=session.snapshot(Path(fx.parent['photo_edit_session']['path']).parent)
        before=copy.deepcopy(state['attempts'])
        body=fx.request(parent=fx.parent,run_id='owned-working-cpu')
        body['editing_iteration']['maximum_image_submissions']=0
        checked=fx.fx.plan(body)
        self.assertEqual(sum(checked['plan']['image_budget'].values()),0)
        response=fx.fx.start(checked);fx.fx.assert_started(response);child=response['run']
        self.assertEqual(child['result_publication']['state'],'committed')
        self.assertEqual(len(fx.fx.fx.image_requests),fx.initial_images)
        self.assertEqual(len(fx.fx.fx.vision_requests)-fx.initial_vision,2)
        self.assertTrue(child['photo_edit_session']['automated_target_met'])
        self.assertEqual(session.snapshot(Path(fx.parent['photo_edit_session']['path']).parent)['attempts'],before)


if __name__=='__main__':unittest.main()
