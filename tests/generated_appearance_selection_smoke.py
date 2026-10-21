"""Current typed selection/snapshot/save with real sessions and transport seams."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import numpy as np
import tifffile

import generated_appearance_render as renderer
import photo_edit_session as session
from photo_appearance import load_asset,validate_asset
from photo_edit_finish import reviewed_candidate_record,save_selected_candidate
from photo_edit_tools import _close_loaded
from photo_source_session_smoke import NativeFixture
from asset_identity import sha256_file
from current_model_registry import atomic_json


class AppearanceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.fx=NativeFixture(self,k=0)
        with redirect_stdout(io.StringIO()):self.variant=self.fx.start()[0]
        self.root=self.fx.root;self.run=self.fx.run;self.args=self.fx.args
        self.directory=Path(self.variant['photo_edit_session']['path']).parent
        state=session.snapshot(self.directory)
        self.selected=session.selected_output(self.directory)['candidate']
        baseline=next(row for row in state['candidates'] if row['candidate_id']==state['baseline_candidate_id'])
        self.old=copy.deepcopy(baseline['appearance_asset'])
        self.old_receipt=(Path(self.old['path'])/'receipt.json').read_bytes()
        self.run['options']['photo_appearance']=copy.deepcopy(self.old)
        self.run['reference']=reviewed_candidate_record(self.root,baseline,self.run['options'])
        self.run['revision']=1
        self.save(self.run,renderer.render_candidate(self.root,self.run,self.args))
        self.old_preview=Path(self.run['artifacts']['auto_draft']).read_bytes()
        self.old_review=Path(self.run['artifacts']['visual_review']).read_bytes()
        self.binding={'source_sha256':self.run['source']['sha256'],
            'scope':self.run['task_plan']['editing_scope'],
            'design_id':self.run['generation_design']['observed_design']['design_id']}

    def save(self,run,result):
        run.update(auto_draft=result,render=result['metadata'],review=result['review'],artifacts=result['outputs'],
            status='needs_visual_review',root=str(self.root),input_mode='single_source',kind='single_input_run')
        atomic_json(self.root/'outputs/runs'/run['run_id']/'manifest.json',run)

    def test_actual_selected_float_is_saved_without_mutating_source_baseline_or_parent(self):
        before=copy.deepcopy(self.run)
        staged=renderer.prepare_selected_appearance(self.root,self.run,self.selected,session_directory=self.directory)
        self.assertEqual(self.run,before)
        self.assertEqual(staged['options']['photo_appearance'],self.selected['appearance_asset'])
        result=renderer.render_candidate(self.root,staged,self.args);self.save(staged,result)
        self.assertNotEqual(staged['options']['photo_appearance']['asset_id'],self.old['asset_id'])
        self.assertEqual((Path(self.old['path'])/'receipt.json').read_bytes(),self.old_receipt)
        self.assertEqual(Path(self.run['artifacts']['auto_draft']).read_bytes(),self.old_preview)
        self.assertEqual(sha256_file(result['outputs']['auto_draft']),self.selected['image']['sha256'])
        actual=load_asset(self.selected['appearance_asset'],**self.binding)
        try:np.testing.assert_array_equal(tifffile.imread(result['outputs']['render_scene_master']),actual['master'])
        finally:_close_loaded(actual)
        self.assertEqual(validate_asset(self.old,**self.binding)['kind'],'source_baseline')
        self.assertEqual(staged['appearance_history'][0]['photo_appearance']['asset_id'],self.old['asset_id'])

    def test_snapshot_copy_materializes_current_and_baseline_without_parent_dependency(self):
        from single_source_workflow import copy_source_run
        from export_artifact import export_source_run
        saved=save_selected_candidate(self.root,self.run,self.args,selected=self.selected,session_directory=self.directory)
        self.assertTrue(saved['changed']);self.save(self.run,saved['rendered'])
        child=copy_source_run(self.root,self.run['run_id'])
        for key,asset in (('current',child['options']['photo_appearance']),('baseline',child['initial_source_appearance'])):
            self.assertTrue(Path(asset['path']).is_relative_to(self.root/'outputs/runs'/child['run_id']),key)
            validate_asset(asset,**self.binding)
        self.assertEqual(child['options']['photo_appearance']['asset_id'],self.selected['appearance_asset']['asset_id'])
        self.assertEqual(child['initial_source_appearance']['asset_id'],self.old['asset_id'])
        self.assertEqual(child['reference']['photo_appearance'],child['options']['photo_appearance'])
        self.assertEqual(child['render']['initial_source_appearance'],child['initial_source_appearance'])
        self.assertEqual(sha256_file(child['artifacts']['auto_draft']),self.selected['image']['sha256'])
        self.assertIsNone(child['review']['reviewer'])
        self.assertEqual(child['review']['status'],'needs_review')
        for format in ('PNG','TIFF16','TIFF32_LINEAR'):
            exported=export_source_run(self.root,child['run_id'],directory=str(self.root/'export'),format=format)
            self.assertTrue(exported['ok']);self.assertFalse(exported['metadata']['approved'])
        parent_master=Path(self.selected['appearance_asset']['path'])/'master.npy'
        parent_master.write_bytes(parent_master.read_bytes()+b'parent external mutation')
        validate_asset(child['options']['photo_appearance'],**self.binding)
        self.assertRaises(ValueError,validate_asset,self.selected['appearance_asset'],**self.binding)

    def test_failed_selected_save_keeps_prior_complete_run_review_and_source_asset(self):
        before=copy.deepcopy(self.run)
        with patch('raw_support.save_render_preview_and_master',side_effect=OSError('simulated disk failure')):
            self.assertRaises(OSError,save_selected_candidate,self.root,self.run,self.args,
                selected=self.selected,session_directory=self.directory)
        self.assertEqual(self.run,before)
        self.assertEqual(Path(self.run['artifacts']['auto_draft']).read_bytes(),self.old_preview)
        self.assertEqual(Path(self.run['artifacts']['visual_review']).read_bytes(),self.old_review)
        self.assertEqual((Path(self.old['path'])/'receipt.json').read_bytes(),self.old_receipt)
        retried=save_selected_candidate(self.root,self.run,self.args,selected=self.selected,session_directory=self.directory)
        self.assertTrue(retried['changed'])
        self.assertEqual(sha256_file(retried['rendered']['outputs']['auto_draft']),self.selected['image']['sha256'])
        self.assertEqual(len(self.fx.fx.image_requests),1)

    def test_nonselected_and_mismatched_asset_do_not_gain_selection_permission(self):
        for changed in ('candidate_id','appearance_asset'):
            modified=copy.deepcopy(self.selected)
            if changed=='candidate_id':modified['candidate_id']='not-selected'
            else:modified['appearance_asset']=copy.deepcopy(self.old)
            with self.subTest(field=changed),self.assertRaises(ValueError):
                renderer.prepare_selected_appearance(self.root,self.run,modified,session_directory=self.directory)
        Path(self.selected['image']['path']).write_bytes(b'changed reviewed preview')
        self.assertRaises(ValueError,renderer.prepare_selected_appearance,self.root,self.run,self.selected,
            session_directory=self.directory)

    def test_typed_master_tamper_is_not_hidden_by_unchanged_selected_png(self):
        master=Path(self.selected['appearance_asset']['path'])/'master.npy'
        before=Path(self.selected['image']['path']).read_bytes();data=bytearray(master.read_bytes());data[-1]^=1
        master.write_bytes(data)
        self.assertEqual(Path(self.selected['image']['path']).read_bytes(),before)
        self.assertRaises(ValueError,renderer.prepare_selected_appearance,self.root,self.run,self.selected,
            session_directory=self.directory)
        self.assertEqual(Path(self.run['artifacts']['auto_draft']).read_bytes(),self.old_preview)

    def test_known_generated_pixels_survive_initial_qa_timeout_without_claiming_review_pass(self):
        from export_artifact import export_source_run
        other=NativeFixture(self,k=0)
        original=other.vision
        def timeout(source,output,prompt,**kwargs):
            if Path(source).name=='review-board.png' and 'decision' not in str(output) and other.fx.image_requests:
                raise TimeoutError('Synthetic initial QA response is unknown')
            return original(source,output,prompt,**kwargs)
        other.fx.vision.side_effect=timeout
        with redirect_stdout(io.StringIO()):item=other.start()[0]
        directory=Path(item['photo_edit_session']['path']).parent
        selected=session.selected_output(directory)
        self.assertEqual(item['status'],'needs_attention')
        self.assertTrue(selected['unreviewed_initial_draft'])
        self.assertIsNone(selected['candidate']['review'])
        self.assertFalse(selected['automated_target_met'])
        self.assertEqual(len(other.fx.image_requests),1)
        run=other.run;run.update(revision=1,kind='single_input_run',status='preparing')
        staged=renderer.prepare_selected_appearance(other.root,run,selected['candidate'],session_directory=directory)
        rendered=renderer.render_candidate(other.root,staged,other.args)
        staged.update(auto_draft=rendered,render=rendered['metadata'],artifacts=rendered['outputs'],
            review=rendered['review'],status='needs_visual_review')
        atomic_json(other.root/'outputs/runs'/run['run_id']/'manifest.json',staged)
        exported=export_source_run(other.root,run['run_id'],directory=str(other.root/'exports'),format='PNG')
        self.assertEqual(exported['metadata']['render_sha256'],selected['candidate']['image']['sha256'])
        self.assertFalse(exported['metadata']['approved'])
        before=len(other.calls)
        self.assertTrue(session.selected_output(directory)['unreviewed_initial_draft'])
        self.assertEqual(len(other.calls),before)


if __name__=='__main__':unittest.main()
