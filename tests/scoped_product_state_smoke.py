"""Product execution, artistic findings and measured time are separate facts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime,timezone

from types import SimpleNamespace
import numpy as np
import photo_appearance
import photo_appearance_smoke as appearance_fixture
import scoped_workflow_smoke as graph_fixture
from raw_support import save_render_master,save_scene_master
from pipeline_stage_report import read_stage_report,build_stage_report,format_stage_report
from scoped_product_state import observe,timing_summary
from job_source_progress import SourceJobProgress
from photo_editing import editing_scope


class ScopedProductTests(unittest.TestCase):
    def fixture(self):
        # These are persisted-observer CPU fixtures, not real model calls.
        # Public task execution is covered by source-native production tests.
        fixture=graph_fixture.ScopedWorkflowTests(methodName='runTest');fixture.setUp();self.addCleanup(fixture.tearDown)
        root=fixture.root;run=fixture.run_fixture(True);scope=run['task_plan']['editing_scope']
        binding={'source_sha256':run['source']['sha256'],'scope':scope,'design_id':run['generation_design']['observed_design']['design_id']}
        baseline=photo_appearance.create_source_baseline(root/'baseline',source_path=run['source']['path'],**binding)
        prior=photo_appearance.load_asset(baseline,**binding);values=prior['master'].copy()
        coverage=np.zeros(values.shape[:2],np.float32);coverage[5:11,6:13]=1;values[coverage>0]+=.000123
        fixture.source=Path(run['source']['path']);fixture.source_sha=run['source']['sha256']
        edited=photo_appearance.freeze_candidate(root/'edited',kind='mixed_regional',master=values,parent=baseline,
            coverage=coverage,generation=appearance_fixture.AppearanceTests.generation(fixture),**binding)
        receipt=photo_appearance.validate_asset(edited,**binding)
        source_receipt=photo_appearance.validate_asset(baseline,**binding)
        image={'path':str(Path(edited['path'])/'preview.png'),'sha256':receipt['files']['preview']['sha256']}
        review={'credibility':{'status':'fail','findings':['Synthetic observer quality finding']},
            'creative_benefit':{'status':'pass','findings':['Synthetic observer only']},'primary_issue':'Fixture finding',
            'provenance':{'network_used':True,'generation_used':False,'model':'synthetic-observer-fixture'},
            'observed_images':[{'role':'source','sha256':source_receipt['files']['preview']['sha256']},
                {'role':'candidate','sha256':image['sha256']}]}
        session_binding={'source':run['source'],'scope_id':scope['scope_id'],'execution_identity':run['task_plan']['plan_id'],
            'editing_initialization':run['task_plan']['editing_initialization'],'variant_id':'v1'}
        from photo_editing import identity
        state={'session_id':identity(session_binding),'binding':session_binding,'scope':scope,'status':'stopped',
            'phase':'stopped','source_appearance':baseline,'baseline_candidate_id':'source','best_candidate_id':'edited',
            'candidates':[{'candidate_id':'edited','image':image,'appearance_asset':edited,'attempt_id':'actual-tool',
                'review':review,'review_objective':'current'}],
            'attempts':[{'attempt_id':'actual-tool','kind':'tool','status':'completed','operation_stage':'initial',
                'costs':{'image_submissions':0,'tool_calls':1},'wall_seconds':.01,
                'response':{'appearance_asset':edited,'resources':{'image_submissions':0}}}]}
        path=root/'session.json';path.write_text(json.dumps(state),encoding='utf8')
        run.update(initial_source_appearance=baseline,reference=image,status='needs_visual_review',
            result_publication={'state':'committed'},photo_edit_session={'path':str(path),'independent_evaluation':review,
                'selected_is_source_baseline':False,'status':'stopped'})
        run['options']['photo_appearance']=edited
        master,scene=root/'render.tif',root/'scene.tif'
        save_render_master(values,master,display_transform={'method':'clip_sdr'})
        save_scene_master(values,scene,working_domain='display_linear_sRGB')
        run['artifacts']={'auto_draft':image['path'],'render_master':str(master),'render_scene_master':str(scene)}
        run['render']={'kind':'scoped_photo_appearance_render','run_id':run['run_id'],'photo_appearance':edited,
            'editing_scope_id':scope['scope_id'],
            'render_precision':{'working_domain':'display_linear_sRGB','working_precision':'float32_linear','master_encoding':'uint16_srgb'}}
        timing=root/'timing.json'
        ledger={'schema':'cosplay_workflow_timing.v1','request_id':'current','run_ids':[run['run_id']],
            'status':'complete','whole_wall_seconds':.02,'spans':[{'name':'root','parent_id':None,'exclusive_seconds':.01},
                {'name':'typed_cpu_fixture','parent_id':'0','exclusive_seconds':.01}]}
        timing.write_text(json.dumps(ledger),encoding='utf8');run['workflow_timing']={'path':str(timing),'request_id':'current'}
        manifest=root/'outputs/runs'/run['run_id']/'manifest.json';manifest.parent.mkdir(parents=True,exist_ok=True)
        manifest.write_text(json.dumps(run),encoding='utf8')
        del prior
        return SimpleNamespace(root=root,parent=run,parent_path=manifest,fx=SimpleNamespace(image_requests=[],vision_requests=[]))

    def test_committed_creation_with_failed_quality_is_not_execution_failure(self):
        fx=self.fixture();run=fx.parent
        before=(fx.parent_path.read_bytes(),len(fx.fx.image_requests),len(fx.fx.vision_requests))
        report=read_stage_report(fx.root,run)
        work=report['whole_work']
        self.assertEqual(work['execution_state'],'complete')
        self.assertEqual(work['result_state'],'needs_repair')
        self.assertTrue(work['saved_current_revision'])
        self.assertFalse(report['reconstruction_required']);self.assertFalse(report['visual_approved'])
        self.assertEqual({row['id']:row['status'] for row in report['cosplay_post_diagnosis']['checks']}['credibility'],'fail')
        self.assertFalse({row['id'] for row in report['stages']} & {'portrait_beauty','background_reconstruction','reference_colour_match','contact_and_edges'})
        text=format_stage_report(report)
        self.assertIn('相机原生细节',text);self.assertIn('本次实测总用时',text)
        self.assertNotIn('尚未准备生成式美化',text)
        self.assertEqual((fx.parent_path.read_bytes(),len(fx.fx.image_requests),len(fx.fx.vision_requests)),before)

    def test_failed_publication_does_not_become_saved_from_existing_pixels_or_review(self):
        fx=self.fixture();run=copy.deepcopy(fx.parent);obs=observe(fx.root,run)
        run['result_publication']['state']='pending'
        obs['workflow_timing'].update(status='failed')
        report=build_stage_report(run,obs)
        self.assertEqual(report['whole_work']['execution_state'],'failed')
        self.assertFalse(report['whole_work']['saved_current_revision'])
        self.assertEqual(next(row['status'] for row in report['stages'] if row['id']=='save_appearance'),'partial')
        self.assertNotEqual(report['whole_work']['result_state'],'needs_local_recovery')

    def test_actual_typed_save_and_validation_spans_are_displayed_without_legacy_names(self):
        fx=self.fixture();run=fx.parent;observed=observe(fx.root,run)
        observed['workflow_timing']['whole_wall_seconds']=3.5
        observed['workflow_timing']['spans']=[{'name':'root','parent_id':None,'exclusive_seconds':.25},
            {'name':'photo_appearance_render','parent_id':'root','exclusive_seconds':2.5},
            {'name':'photo_appearance_freeze','parent_id':'root','exclusive_seconds':.5},
            {'name':'photo_appearance_validation','parent_id':'root','exclusive_seconds':.25}]
        report=build_stage_report(run,observed)
        stages={row['id']:row for row in report['stages']}
        self.assertEqual(stages['save_appearance']['measured_seconds'],2.5)
        self.assertEqual(stages['freeze_appearance']['measured_seconds'],.75)
        self.assertEqual(sum(row['seconds'] for row in report['timing']['modules']),3.25)

    def test_known_saved_draft_with_unknown_review_remains_viewable_and_is_not_total_failure(self):
        fx=self.fixture();run=fx.parent
        run['status']='needs_attention';run['photo_edit_session']['status']='needs_attention'
        run['editing_autofinish_result']={'failure':{'type':'TimeoutError','message':'Inspect the existing review'}}
        run['photo_edit_session']['independent_evaluation']=None
        report=read_stage_report(fx.root,run)
        self.assertEqual(report['whole_work']['execution_state'],'complete')
        self.assertEqual(report['whole_work']['result_state'],'needs_request_attention')
        self.assertTrue(report['whole_work']['saved_current_revision'])
        self.assertTrue(next(action['supported'] for action in report['whole_work']['recovery_actions'] if action['action_id']=='inspect_native_evidence'))
        self.assertFalse(report['visual_approved'])
        self.assertIn('不自动重发',report['next_action'])

    def test_saved_source_baseline_is_not_an_edited_output_even_with_passed_axes(self):
        fx=self.fixture();run=fx.parent;scope=run['task_plan']['editing_scope'];baseline=run['initial_source_appearance']
        loaded=photo_appearance.load_asset(baseline,source_sha256=run['source']['sha256'],scope=scope,
            design_id=run['generation_design']['observed_design']['design_id'])
        preview={'path':str(Path(baseline['path'])/'preview.png'),'sha256':loaded['receipt']['files']['preview']['sha256']}
        values=loaded['master'].copy();del loaded
        save_render_master(values,Path(run['artifacts']['render_master']),display_transform={'method':'clip_sdr'})
        save_scene_master(values,Path(run['artifacts']['render_scene_master']),working_domain='display_linear_sRGB')
        run['reference']=preview;run['artifacts']['auto_draft']=preview['path'];run['options']['photo_appearance']=baseline
        run['render']['photo_appearance']=baseline
        review=copy.deepcopy(run['photo_edit_session']['independent_evaluation'])
        review['credibility']['status']='pass'
        review['observed_images'][-1]['sha256']=preview['sha256']
        path=Path(run['photo_edit_session']['path']);state=json.loads(path.read_text(encoding='utf8'))
        state.update(best_candidate_id='source',baseline_candidate_id='source',attempts=[],candidates=[{
            'candidate_id':'source','image':preview,'appearance_asset':baseline,'review':review,'review_objective':'current'}])
        path.write_text(json.dumps(state),encoding='utf8')
        run['photo_edit_session'].update(independent_evaluation=review,selected_is_source_baseline=True)
        report=read_stage_report(fx.root,run)
        self.assertTrue(report['whole_work']['saved_current_revision'])
        self.assertFalse(report['whole_work']['edited_output'])
        self.assertEqual(report['whole_work']['result_state'],'needs_editing')
        self.assertIn('尚无获准',report['headline'])

    def test_origin_actual_request_totals_are_not_local_attempt_counts(self):
        fx=self.fixture();run=fx.parent
        run['photo_edit_session']['origin_request_budget']={'origin_run_id':run['run_id'],
            'origin_plan_id':run['task_plan']['plan_id'],'confirmed_image_submissions':0,'reserved_image_requests':0,
            'checked_image_limit':1,'confirmed_vision_requests':2,'reserved_vision_requests':2,'checked_vision_limit':4}
        run['photo_edit_session']['reserved_budget_usage']={'vision_requests':3,'tool_calls':1}
        usage=read_stage_report(fx.root,run)['request_usage']
        self.assertEqual(usage['vision_requests'],{'confirmed':2,'reserved':2,'limit':4})
        self.assertEqual(usage['image_submissions']['confirmed'],0)
        self.assertFalse(usage['local_session_reservations_are_actual_http_totals'])

    def test_native_nodes_do_not_invent_whole_generation_or_use_parent_timing(self):
        fx=self.fixture();run=fx.parent;report=read_stage_report(fx.root,run)
        ids={row['id'] for row in report['stages']}
        self.assertIn('source_native_baseline',ids);self.assertIn('selected_edit_tool',ids)
        self.assertNotIn('generate_edit',ids);self.assertNotIn('portrait_beauty',ids)
        self.assertFalse(report['timing']['parent_timing_included'])
        self.assertFalse(report['whole_work']['native_detail_acceptance'])
        tool=next(row for row in report['stages'] if row['id']=='selected_edit_tool')
        self.assertEqual(tool['measured_seconds'],.01)
        self.assertEqual(tool['timing_basis'],'persisted_current_session_attempt_wall_seconds')
        changed=copy.deepcopy(observe(fx.root,run));changed['workflow_timing']['request_id']='parent'
        self.assertFalse(build_stage_report(run,changed)['timing']['available'])

    def test_exclusive_wall_time_does_not_add_parent_inclusive_and_child_again(self):
        run={'run_id':'owned','workflow_timing':{'request_id':'own'},'task_plan':{}}
        ledger={'schema':'cosplay_workflow_timing.v1','request_id':'own','run_ids':['owned'],'status':'complete',
            'whole_wall_seconds':10,'spans':[{'name':'root','parent_id':None,'exclusive_seconds':3},
                {'name':'generation','parent_id':'0','exclusive_seconds':2,'inclusive_seconds':7},
                {'name':'provider_transport','parent_id':'1','exclusive_seconds':5,'inclusive_seconds':5}]}
        summary=timing_summary(run,ledger)
        self.assertEqual(sum(row['seconds'] for row in summary['modules']),7)
        self.assertEqual(summary['whole_seconds'],10)
        self.assertFalse(summary['nested_durations_additive'])

    def test_eta_workload_identity_separates_model_scope_and_iteration(self):
        from workflow_timing import Ledger
        with tempfile.TemporaryDirectory(prefix='scoped-timing-id-') as directory:
            base={'kind':'single_input_run','run_id':'test','source':{'sha256':'a'*64},
                'task_intent':{'mode':'create_reference'},'options':{'provider':'one','model':'model-a'},
                'task_plan':{'editing_scope':editing_scope(),'layer_workflow':{'graph_id':'graph-a'}}}
            first=Ledger(directory,'production');first.bind_run(copy.deepcopy(base))
            variants=[]
            other=copy.deepcopy(base);other['options']['provider']='two';variants.append(other)
            other=copy.deepcopy(base);other['options']['model']='model-b';variants.append(other)
            other=copy.deepcopy(base);other['task_plan']['editing_scope']=editing_scope({'style':'dramatic'});variants.append(other)
            other=copy.deepcopy(base);other['task_plan']['editing_iteration']={'maximum_improvements':1};variants.append(other)
            for other in variants:
                ledger=Ledger(directory,'production');ledger.bind_run(other)
                self.assertNotEqual(first.workload_key,ledger.workload_key)

    def test_running_scoped_job_reports_actual_module_and_no_invented_percent(self):
        with tempfile.TemporaryDirectory(prefix='scoped-progress-') as directory:
            root=Path(directory);source=root/'source.png';source.write_bytes(b'synthetic metadata only')
            observer=SourceJobProgress(root,'owned',source,source_sha256='a'*64,owns_new_run=True,clock=lambda:10.)
            folder=root/'outputs/runs/owned';folder.mkdir(parents=True)
            target=root/'outputs/timings/workflow-own.json';target.parent.mkdir(parents=True)
            start=datetime.now(timezone.utc).timestamp()
            ledger={'schema':'cosplay_workflow_timing.v1','request_id':'own','run_ids':['owned'],'status':'running',
                'created_at':datetime.fromtimestamp(start,timezone.utc).isoformat(),'whole_wall_seconds':14,
                'spans':[{'id':'0','name':'production_workflow','parent_id':None,'status':'running','start_offset_seconds':0,'inclusive_seconds':14},
                         {'id':'1','name':'provider_transport','parent_id':'0','status':'running','start_offset_seconds':14,'inclusive_seconds':0}]}
            target.write_text(json.dumps(ledger),encoding='utf8')
            run={'run_id':'owned','source':{'path':str(source),'sha256':'a'*64},'status':'generating',
                'task_plan':{'editing_scope':editing_scope(),'editing_iteration':{'maximum_improvements':1}},
                'workflow_timing':{'path':str(target)},'events':[{'at':'now','status':'generating','detail':'model_selected_improvement:tool'}]}
            (folder/'manifest.json').write_text(json.dumps(run),encoding='utf8')
            with patch('workflow_timing.time.time',return_value=start+40):events=observer.poll()
            self.assertTrue(events)
            self.assertTrue(all(row.get('progress') is None for row in events))
            self.assertTrue(all(row.get('progress_basis')=='indeterminate_scoped_workflow' for row in events))
            self.assertEqual(events[-1]['timing']['module_label'],'执行选定图像编辑')
            self.assertEqual(events[-1]['timing']['stage_elapsed_seconds'],26)
            self.assertIn('editing_tool',[row['stage'] for row in events])

    def test_qt_scoped_route_and_task_table_use_indeterminate_progress(self):
        import qt_workbench as qt
        from design_reconstruction_ui_smoke import Window,Pool
        app=qt.QApplication.instance() or qt.QApplication([])
        with tempfile.TemporaryDirectory(prefix='scoped-qt-progress-') as directory:
            window=Window(Path(directory),'http://127.0.0.1:1');window.pool=Pool()
            window.active_checked_task={'request':{'editing_request':editing_scope()}}
            window.simple_creation={'source':'source.png','mode':'create_cos','stage':'production'}
            value=window._quick_progress(.25,'generating','one_click',timing={'module_label':'生成完整编辑图','elapsed_seconds':25})
            self.assertIsNone(value[0]);self.assertEqual(value[1],'生成完整编辑图')
            window.populate_jobs({'jobs':[{'job_id':'scope-job','status':'running','stage':'generating','progress':.25,
                'editing_request':{'work_type':'partial_composite'}}]})
            self.assertEqual(window.jobs_tree.topLevelItem(0).text(3),'—')
            window.simple_creation=None;window.active_checked_task=None;window.session_timer.stop();window.close();app.processEvents()

    def test_source_native_k0_reports_real_first_operation_and_keeps_distinct_phases(self):
        from scoped_workflow import SOURCE_NATIVE_INITIALIZATION
        from asset_identity import sha256_file
        with tempfile.TemporaryDirectory(prefix='native-first-progress-') as directory:
            root=Path(directory);source=root/'source.bin';source.write_bytes(b'CPU fixture')
            observer=SourceJobProgress(root,'owned',source,source_sha256=sha256_file(source),owns_new_run=True)
            run={'run_id':'owned','source':{'path':str(source),'sha256':sha256_file(source)},
                'task_plan':{'editing_scope':editing_scope(),'editing_initialization':SOURCE_NATIVE_INITIALIZATION,
                    'editing_autofinish':{'maximum_improvements':0}}}
            self.assertTrue(observer._bound_record(run))
            events=[observer._workflow_event({'status':'preparing','at':'same',
                'detail':'source_native:initial:'+phase},root/'log') for phase in ('decide','tool','evaluate')]
            self.assertEqual([row['stage'] for row in events],['editing_decision','editing_tool','editing_review'])
            self.assertTrue(all(row.get('progress') is None for row in events))
            self.assertEqual(len({row['workflow_event']['phase'] for row in events}),3)
            observer._workflow_event({'status':'preparing','detail':{'unsupported':'payload'}},root/'log')

    def test_checked_scoped_capabilities_override_unused_global_analysis_setup(self):
        import qt_workbench as qt
        from design_reconstruction_ui_smoke import Window,Pool
        app=qt.QApplication.instance() or qt.QApplication([])
        from scoped_production_smoke import ScopedProductionTests
        fixture=ScopedProductionTests(methodName='runTest');fixture.setUp();self.addCleanup(fixture.doCleanups)
        checked=fixture.backend.task_plan(fixture.request(fixture.sources[0]))
        fx=SimpleNamespace(root=fixture.root,fx=SimpleNamespace(sources=fixture.sources))
        window=Window(fx.root,'http://127.0.0.1:1');window.pool=Pool()
        window.set_simple_source(str(fx.fx.sources[0]))
        window.model_setup_status={'configured':False,'ready':False}
        window.startup_readiness='initializing'
        window.creative_task_plan={'plan':checked['plan'],'request':checked['request'],
            'revision':window.creative_task_revision,'callback':'existing-design','batch':False}
        window.refresh_cos_workspace()
        self.assertTrue(window.simple_workspace.create_cos_button.isEnabled())
        self.assertTrue(window.simple_workspace.start_plan_button.isEnabled())
        window.start_planned_creative_task()
        self.assertEqual(window.pool.tasks[-1].path,'/api/tasks/start')
        window.pool.tasks[-1].signals.error.emit('Captured submission only')
        window.session_timer.stop();window.close();app.processEvents()


if __name__=='__main__':unittest.main()
