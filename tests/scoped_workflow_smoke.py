"""Current source-native contracts and saved graph facts; CPU assets only."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication,QGraphicsItem
import creative_task
from cos_node_registry import compile_workflow
from cos_workflow_graph import build_graph,revision_for_template,preset
from photo_editing import editing_scope,identity
from scoped_workflow import SOURCE_NATIVE_INITIALIZATION
from production_coordinator import route_contract
import generated_appearance_render_smoke as fixture
import photo_appearance


def compile_current(scope,review=False,k=0,n=1):
    return compile_workflow('create_reference',editing_scope=scope,visual_review=review,
        editing_autofinish={'maximum_improvements':k},editing_initialization=SOURCE_NATIVE_INITIALIZATION,
        initial_variant_count=n)


class ScopedWorkflowTests(unittest.TestCase):
    def setUp(self):fixture.ScopedRenderTests.setUp(self)
    def tearDown(self):fixture.ScopedRenderTests.tearDown(self)
    def run_fixture(self,review=False):
        run=fixture.ScopedRenderTests.fixture(self)
        run.update(root=str(self.root),execution_strategy='layered_reference_transfer')
        intent=creative_task.normalize_task_intent({'schema_version':creative_task.INTENT_SCHEMA,
            'mode':'create_reference','services':{'text_vision':True,'visual_review':review}})
        scope=run['task_plan']['editing_scope'];request={key:scope[key] for key in ('work_type','style','target_setting','scene_target','language')}
        run['task_intent']=intent
        run['task_plan'].update(intent=intent,mode='create_reference',schema_version=creative_task.PLAN_SCHEMA,
            execution_strategy='layered_reference_transfer',editing_request=request,plan_id='a'*64,
            editing_initialization=copy.deepcopy(SOURCE_NATIVE_INITIALIZATION),editing_autofinish={'maximum_improvements':0,'maximum_zero_image_finishes':0},
            image_budget=creative_task.image_budget(intent,1,scope,autofinish={'maximum_improvements':0}),
            services={'text_vision_requests_upper_bound':4 if review else 2},layer_workflow=compile_current(scope,review))
        return run

    def test_checked_coordinator_contract_contains_source_baseline_and_model_decision(self):
        for work in ('retouch','partial_composite','full_composite'):
            for review in (False,True):
                scope=editing_scope({'work_type':work})
                graph=compile_current(scope,review)
                self.assertNotIn('generate_edit',graph['order'])
                self.assertNotIn('save_best',graph['order'])
                self.assertEqual('baseline_review' in graph['order'],review)
                self.assertEqual('independent_recheck' in graph['order'],review)
                self.assertLess(graph['order'].index('source_native_baseline'),graph['order'].index('decide_edit'))
                self.assertTrue(graph['initial_candidates']['serial'])
                self.assertEqual(graph['request_limits'],{'image_submissions':1,'vision_requests':4 if review else 2})
                self.assertFalse(set(graph['order'])&{'matte','beauty','colour','background','effects'})
                self.assertTrue(all(node['alpha_role']=='not_applicable' for node in graph['nodes']))
                tool=next(node for node in graph['nodes'] if node['id']=='selected_edit_tool')
                self.assertIn('photo_development.py',tool['controls']['producer'])
                self.assertEqual(tool['controls']['available_tools']['develop_photograph']['costs'].get('image_submissions',0),0)
                self.assertEqual('joint_edit' in tool['controls']['available_tools'],work!='retouch')
        run=self.run_fixture(True);plan=run['task_plan'];intent=run['task_intent']
        self.assertEqual(creative_task.validate_task_plan(plan,intent),plan)
        self.assertEqual(route_contract({'task_plan':plan,'task_intent':intent})['graph_id'],plan['layer_workflow']['graph_id'])

    def test_scope_count_policy_and_producer_change_identity_without_imported_changes(self):
        partial=editing_scope();full=editing_scope({'work_type':'full_composite'})
        first=compile_current(partial)
        self.assertEqual(first,compile_current(partial))
        for other in (compile_current(full),compile_current(partial,True),compile_current(partial,True,1),compile_current(partial,False,0,3)):
            self.assertNotEqual(first['graph_id'],other['graph_id'])
        from asset_identity import sha256_file
        def changed(path,*args,**kwargs):return 'b'*64 if Path(path).name=='photo_appearance.py' else sha256_file(path,*args,**kwargs)
        with patch('asset_identity.sha256_file',side_effect=changed):self.assertNotEqual(first['graph_id'],compile_current(partial)['graph_id'])
        for node in first['nodes']:
            if node['id'] in {'decide_edit','selected_edit_tool','select_best','save_appearance'}:
                self.assertIn('photo_session_evidence.py',node['controls']['producer'])
        self.assertEqual(compile_workflow('existing_reference'),compile_workflow('existing_reference',editing_scope=partial))
        with self.assertRaisesRegex(ValueError,'initialization'):
            compile_workflow('create_reference',editing_scope=partial)
        with self.assertRaisesRegex(ValueError,'read-only'):
            compile_workflow('create_reference',{'beauty':{'enabled':True,'strength':1}},editing_scope=partial,editing_initialization=SOURCE_NATIVE_INITIALIZATION)

    def test_face_shaping_graph_has_actual_geometry_preparation_and_its_producer_closure(self):
        scope=editing_scope();face={'action':'apply','parts':{'face':.3}}
        graph=compile_workflow('existing_reference',editing_scope=scope,visual_review=True,shaping_request=face)
        self.assertLess(graph['order'].index('face_geometry'),graph['order'].index('baseline_review'))
        self.assertLess(graph['order'].index('shaping_inputs'),graph['order'].index('face_geometry'))
        self.assertIn({'from':'face_geometry','output':'support','to':'selected_edit_tool','input':'geometry'},graph['connections'])
        self.assertEqual(graph['request_limits'],{'image_submissions':0,'vision_requests':3})
        tool=next(row for row in graph['nodes'] if row['id']=='selected_edit_tool')
        self.assertEqual(set(tool['controls']['available_tools']),{'face_contour_deform'})
        self.assertFalse(tool['controls']['available_tools']['face_contour_deform']['requires_region'])
        for name in ('photo_shaping_geometry.py','portrait_regions_worker.py','face_contour_deform.py','geometric_appearance.py','task_model_preflight.py'):
            self.assertIn(name,graph['producer'])
        node=next(row for row in graph['nodes'] if row['id']=='face_geometry')
        self.assertEqual(node['controls']['execution_resources'],{'cpu_face_localization':1,'image_submissions':0,'vision_requests':0})
        self.assertFalse(node['controls']['geometry_is_visual_acceptance'])
        for request in ({'action':'apply','parts':{'waist':.3}},{'action':'restore'}):
            other=compile_workflow('existing_reference',editing_scope=scope,visual_review=request['action']=='apply',shaping_request=request)
            self.assertNotIn('face_geometry',other['order']);self.assertEqual(other['preparation_stages'],[])
        from asset_identity import sha256_file
        for name in ('photo_shaping_geometry.py','portrait_regions_worker.py','face_contour_deform.py','geometric_appearance.py','task_model_preflight.py'):
            def changed(path,*args,**kwargs):return 'b'*64 if Path(path).name==name else sha256_file(path,*args,**kwargs)
            with self.subTest(producer=name),patch('asset_identity.sha256_file',side_effect=changed):
                self.assertNotEqual(graph['graph_id'],compile_workflow('existing_reference',editing_scope=scope,visual_review=True,shaping_request=face)['graph_id'])

    def geometry_run(self):
        import photo_shaping_geometry as geometry
        from photo_shaping_geometry_smoke import GeometryTests
        worker=GeometryTests(methodName='runTest');worker.setUp();self.addCleanup(worker.doCleanups)
        run=fixture.ScopedRenderTests.fixture(self,name='shape-graph',size=(512,512))
        run.update(root=str(self.root),execution_strategy='layered_reference_transfer')
        scope=run['task_plan']['editing_scope'];request={'action':'apply','parts':{'face':.3}}
        base=run['options']['photo_appearance'];design=run['generation_design']['observed_design']['design_id']
        run['task_intent']=creative_task.normalize_task_intent({'schema_version':creative_task.INTENT_SCHEMA,
            'mode':'existing_reference','services':{'text_vision':True,'visual_review':True}})
        run['task_plan'].update(mode='existing_reference',plan_id='a'*64,shaping_request=request,
            shaping_authorization={'base_appearance':copy.deepcopy(base)})
        run['task_plan'].pop('editing_initialization',None)
        run['task_plan']['layer_workflow']=compile_workflow('existing_reference',editing_scope=scope,visual_review=True,shaping_request=request)
        output=self.root/'outputs/runs'/run['run_id']/'shaping-geometry'
        with patch.object(geometry,'_run_detector',side_effect=worker.detector):
            support=geometry.prepare_geometry(self.root,base,source_sha256=run['source']['sha256'],scope=scope,
                design_id=design,output=output,registry_path=worker.registry,preflight=worker.preflight())
        run['shaping_geometry']=support
        return run

    def test_face_geometry_graph_reads_owned_proof_without_arrays_runtime_or_new_models(self):
        import photo_shaping_geometry as geometry
        from photo_session_evidence import freeze_session_evidence
        run=self.geometry_run();scope=run['task_plan']['editing_scope']
        def node():return next(row for row in build_graph(run)['nodes'] if row['id']=='face_geometry')
        with patch.object(geometry,'_run_detector',side_effect=AssertionError('graph started detector')),\
                patch.object(geometry,'_runtime',side_effect=AssertionError('graph inspected runtime')),\
                patch.object(geometry,'_geometry',side_effect=AssertionError('graph derived large masks')),\
                patch('numpy.load',side_effect=AssertionError('graph loaded pixels')):
            self.assertEqual(node()['status'],'complete')
            self.assertFalse(node()['execution']['planes_rederived']);self.assertFalse(node()['execution']['runtime_reinspected'])
            self.assertFalse(node()['execution']['naturalness_verified'])
        directory=self.root/'outputs/runs'/run['run_id']/'editing-session';directory.mkdir()
        binding={'source':run['source'],'scope_id':scope['scope_id'],'design_id':run['generation_design']['observed_design']['design_id'],
            'execution_identity':run['task_plan']['plan_id'],'geometry_support':copy.deepcopy(run['shaping_geometry']),
            'shaping_base_appearance':copy.deepcopy(run['options']['photo_appearance'])}
        state={'session_id':identity(binding),'binding':binding,'scope':scope,'status':'active','attempts':[],'candidates':[]}
        session_file=directory/'session.json';session_file.write_text(json.dumps(state),encoding='utf8')
        state['evidence']=freeze_session_evidence(directory,directory.parent/'owned-session-evidence',state=state)
        session_file.write_text(json.dumps(state),encoding='utf8')
        run['photo_edit_session']={'path':str(session_file),'session_id':state['session_id']}
        run['result_publication']={'state':'committed'}
        original=Path(run['shaping_geometry']['path']).parent;hidden=original.with_name('original-geometry-unavailable')
        original.rename(hidden)
        try:
            with patch.object(geometry,'_runtime',side_effect=AssertionError('portable graph inspected runtime')),\
                    patch.object(geometry,'_geometry',side_effect=AssertionError('portable graph derived masks')),\
                    patch('numpy.load',side_effect=AssertionError('portable graph loaded planes')):
                fact=node();self.assertEqual(fact['status'],'complete');self.assertTrue(fact['execution']['owned_session_evidence'])
        finally:hidden.rename(original)

    def test_face_geometry_missing_wrong_u_stale_producer_and_changed_mask_need_attention(self):
        from asset_identity import sha256_file
        run=self.geometry_run()
        def status():return next(row['status'] for row in build_graph(run)['nodes'] if row['id']=='face_geometry')
        support=Path(run['shaping_geometry']['path']);saved=support.read_bytes();descriptor=copy.deepcopy(run['shaping_geometry'])
        run['shaping_geometry']={'path':str(support.with_name('missing.json')),'sha256':descriptor['sha256']}
        self.assertEqual(status(),'needs_attention');run['shaping_geometry']=copy.deepcopy(descriptor)
        for changed in ('wrong-u','stale-producer'):
            document=json.loads(saved)
            if changed=='wrong-u':document['binding']['base']['asset_id']='f'*64
            else:document['binding']['producer']['photo_shaping_geometry.py']='f'*64
            document['support_id']=identity({key:value for key,value in document.items() if key!='support_id'})
            support.write_text(json.dumps(document),encoding='utf8');run['shaping_geometry']['sha256']=sha256_file(support)
            self.assertEqual(status(),'needs_attention')
        support.write_bytes(saved);run['shaping_geometry']=copy.deepcopy(descriptor)
        mask=Path(json.loads(saved)['masks']['path']);values=mask.read_bytes();mask.write_bytes(values[:-1]+bytes([values[-1]^1]))
        self.assertEqual(status(),'needs_attention')
        mask.write_bytes(values);self.assertEqual(status(),'complete')
        run['result_publication']={'state':'committed'}
        self.assertEqual(status(),'needs_attention','A saved flag cannot replace owned session evidence')

    def test_real_baseline_is_not_generation_edit_completion_or_a_cache_hit(self):
        run=self.run_fixture();scope=run['task_plan']['editing_scope'];history=copy.deepcopy(run['task_plan'])
        run['initial_source_appearance']=photo_appearance.create_source_baseline(self.root/'baseline',source_path=run['source']['path'],
            source_sha256=run['source']['sha256'],scope=scope,design_id=run['generation_design']['observed_design']['design_id'])
        with patch('numpy.load',side_effect=AssertionError('graph loaded pixels')):
            graph=build_graph(run)
        states={row['id']:row['status'] for row in graph['nodes']}
        self.assertEqual(states['source_native_baseline'],'complete')
        self.assertEqual(states['selected_edit_tool'],'pending')
        self.assertEqual(states['save_appearance'],'pending')
        self.assertFalse(graph['edited_output']);self.assertFalse(graph['cache_hits_inferred'])
        self.assertEqual(run['task_plan'],history)
        preview=Path(run['initial_source_appearance']['path'])/'preview.png';hidden=preview.with_suffix('.hidden');preview.rename(hidden)
        try:self.assertEqual(next(row['status'] for row in build_graph(run)['nodes'] if row['id']=='source_native_baseline'),'needs_attention')
        finally:hidden.rename(preview)
        old=copy.deepcopy(run);old['task_plan'].pop('editing_initialization');before=copy.deepcopy(old)
        with self.assertRaisesRegex(ValueError,'Unsupported retired creation'):build_graph(old)
        self.assertEqual(old,before)

    def test_unknown_real_session_tool_cannot_become_completed_generation_or_saved(self):
        run=self.run_fixture(True);scope=run['task_plan']['editing_scope']
        binding={'scope_id':scope['scope_id'],'execution_identity':run['task_plan']['plan_id'],'source':run['source'],
            'editing_initialization':SOURCE_NATIVE_INITIALIZATION,'variant_id':'v1'}
        state={'session_id':identity(binding),'binding':binding,'scope':scope,'status':'needs_attention',
            'attempts':[{'kind':'tool','status':'unknown','operation_stage':'initial','costs':{'image_submissions':1,'tool_calls':1}}]}
        path=self.root/'unknown-session.json';path.write_text(json.dumps(state),encoding='utf8')
        run['photo_edit_session']={'path':str(path),'session_id':state['session_id']}
        nodes={row['id']:row for row in build_graph(run)['nodes']}
        self.assertEqual(nodes['selected_edit_tool']['status'],'needs_attention')
        self.assertEqual(nodes['save_appearance']['status'],'pending')
        self.assertEqual(nodes['selected_edit_tool']['execution']['actual_image_submissions'],None)

    def test_zero_image_tool_needs_actual_typed_candidate_and_preserves_initial_stage(self):
        import numpy as np
        from photo_appearance_smoke import AppearanceTests
        run=self.run_fixture();scope=run['task_plan']['editing_scope']
        binding={'source_sha256':run['source']['sha256'],'scope':scope,
            'design_id':run['generation_design']['observed_design']['design_id']}
        baseline=photo_appearance.create_source_baseline(self.root/'baseline',source_path=run['source']['path'],**binding)
        self.source=Path(run['source']['path']);self.source_sha=run['source']['sha256']
        generation=AppearanceTests.generation(self)
        loaded=photo_appearance.load_asset(baseline,**binding);values=loaded['master'].copy()
        coverage=np.zeros(values.shape[:2],np.float32);coverage[5:11,6:13]=1
        values[coverage>0]+=np.float32(.000123)
        edited=photo_appearance.freeze_candidate(self.root/'cpu-edited',kind='mixed_regional',master=values,
            parent=baseline,coverage=coverage,generation=generation,**binding)
        receipt=photo_appearance.validate_asset(edited,**binding)
        session_binding={'scope_id':scope['scope_id'],'execution_identity':run['task_plan']['plan_id'],
            'source':run['source'],'editing_initialization':SOURCE_NATIVE_INITIALIZATION,'variant_id':'v1'}
        attempt={'attempt_id':'tool-one','kind':'tool','status':'completed','operation_stage':'initial','wall_seconds':.2,
            'costs':{'image_submissions':0,'tool_calls':1,'improvement_steps':0},
            'response':{'appearance_asset':edited,'resources':{'image_submissions':0}}}
        candidate={'candidate_id':'edited','attempt_id':'tool-one','appearance_asset':edited,
            'image':{'path':str(Path(edited['path'])/'preview.png'),'sha256':receipt['files']['preview']['sha256']}}
        state={'session_id':identity(session_binding),'binding':session_binding,'scope':scope,'status':'waiting_selection',
            'best_candidate_id':'edited','baseline_candidate_id':'baseline','attempts':[attempt],'candidates':[candidate]}
        path=self.root/'session.json';path.write_text(json.dumps(state),encoding='utf8')
        run['initial_source_appearance']=baseline;run['options']['photo_appearance']=edited
        run['photo_edit_session']={'path':str(path),'session_id':state['session_id']}
        graph=build_graph(run);tool=next(row for row in graph['nodes'] if row['id']=='selected_edit_tool')
        self.assertEqual(tool['status'],'complete');self.assertEqual(tool['execution']['actual_image_submissions'],0)
        self.assertEqual(tool['execution']['zero_image_tool_calls'],1)
        self.assertIn('首次操作 1',tool['detail']);self.assertTrue(graph['edited_output'])
        self.assertEqual(tool['measured_seconds'],.2);self.assertNotIn('generate_edit',graph['layer_workflow']['order'])
        # A status flag without its actual bound typed output is insufficient.
        state['candidates'][0]['image']['sha256']='f'*64;path.write_text(json.dumps(state),encoding='utf8')
        graph=build_graph(run)
        self.assertEqual(next(row['status'] for row in graph['nodes'] if row['id']=='selected_edit_tool'),'needs_attention')
        self.assertFalse(graph['edited_output'])

    def test_offscreen_canvas_has_actual_typed_ports_and_no_legacy_controls(self):
        from cos_workflow_canvas import CosWorkflowCanvas
        app=QApplication.instance() or QApplication([]);run=self.run_fixture(True)
        canvas=CosWorkflowCanvas();canvas.resize(1400,640);canvas.set_run(run);canvas.show();app.processEvents()
        events=[];actions=[];canvas.templateChanged.connect(events.append);canvas.actionRequested.connect(actions.append)
        try:
            canvas.node_items['freeze_appearance'].setSelected(True);app.processEvents()
            self.assertFalse(canvas.enabled.isEnabled());self.assertTrue(canvas.enabled.isHidden())
            self.assertTrue(canvas.strength.isHidden());self.assertFalse(canvas.action.isEnabled())
            self.assertFalse(canvas.presets.isEnabled())
            self.assertFalse(bool(canvas.node_items['freeze_appearance'].flags()&QGraphicsItem.ItemIsMovable))
            self.assertIn('原生与生成来源',canvas.ports.text())
            canvas.strength.setValue(20);canvas._action();self.assertFalse(events);self.assertFalse(actions)
            self.assertEqual(canvas.template()['nodes'],{})
            with self.assertRaisesRegex(ValueError,'只读'):canvas._apply_template(preset('cos_scene'))
            with self.assertRaisesRegex(ValueError,'read-only'):revision_for_template(run,preset('cos_scene'))
            canvas.grab()
        finally:canvas.close()


if __name__=='__main__':unittest.main()
