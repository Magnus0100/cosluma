"""Refinement dependency graphs follow the actual bounded coordinator tools.

Contracts use real producers; saved typed float assets and synthetic persisted
observer records drive offscreen Qt. Public execution is tested separately.
"""
from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QGraphicsItem

from asset_identity import sha256_file
from cos_node_registry import compile_workflow
from cos_workflow_graph import build_graph, revision_for_template, preset
from photo_editing import editing_scope
import scoped_product_state_smoke as product_fixture
import json


class IterationGraphTests(unittest.TestCase):
    def fixture(self):
        # Actual frozen assets plus synthetic persisted observer records. The
        # separate public production suite owns model transport execution.
        owner=product_fixture.ScopedProductTests(methodName='runTest');self.addCleanup(owner.doCleanups)
        fx=owner.fixture();run=fx.parent;scope=run['task_plan']['editing_scope']
        plan=run['task_plan'];plan.pop('editing_initialization');plan.pop('editing_autofinish')
        run['task_intent']['mode']='existing_reference';plan['mode']='existing_reference'
        plan['editing_iteration']={'maximum_improvements':1,'reason':'Repair the reviewed issue'}
        plan['inputs']['reference']=copy.deepcopy(run['reference'])
        plan['image_budget']={'initial_reference':0,'reference_correction':0,'independent_effect':0,'editing_refinement':1}
        plan['editing_refinement_authorization']={'scope_id':scope['scope_id'],'source_sha256':run['source']['sha256'],
            'base_image_sha256':run['reference']['sha256'],'review_sha256':'b'*64}
        plan['layer_workflow']=self.compile(scope)
        run['editing_iteration_result']={'improved':True}
        path=Path(run['photo_edit_session']['path']);state=json.loads(path.read_text(encoding='utf8'))
        state['binding'].pop('editing_initialization');state['binding']['appearance_required']=True
        from photo_editing import identity
        state['session_id']=identity(state['binding']);run['photo_edit_session']['session_id']=state['session_id']
        state['attempts'][0].update(context={'decision':{'tool':'edit_region'}},operation_stage='improvement')
        state['attempts'].extend([{'kind':'decide','status':'completed','response':{'action':'tool'},'operation_stage':'improvement'},
            {'kind':'evaluate','status':'completed','operation_stage':'improvement'}])
        path.write_text(json.dumps(state),encoding='utf8');fx.state=state;fx.session_path=path
        fx.parent_path.write_text(json.dumps(run),encoding='utf8')
        return fx

    def compile(self,scope=None,**kwargs):
        return compile_workflow('existing_reference',editing_scope=scope or editing_scope(),
            visual_review=True,editing_iteration=kwargs or {'maximum_improvements':1,'reason':'Repair the main reviewed issue'})

    def test_scope_iteration_review_and_actual_tool_producers_own_the_signature(self):
        scope=editing_scope()
        graph=self.compile(scope)
        self.assertEqual(graph,self.compile(scope))
        self.assertEqual(graph['order'],['reviewed_inputs','decide_edit','selected_edit_tool','independent_recheck',
            'select_best','freeze_appearance','save_appearance','export'])
        self.assertFalse(set(graph['order']) & {'observe_design','generate_edit','beauty','matte','colour','background','boundary','effects'})
        self.assertTrue(graph['readonly'])
        for node in graph['nodes']:
            if node['id'] in {'reviewed_inputs','decide_edit','selected_edit_tool','independent_recheck','select_best','save_appearance'}:
                self.assertIn('photo_session_evidence.py',node['controls']['producer'])
        self.assertTrue(graph['bounded_dynamic_loop']['stop_allowed'])
        self.assertEqual(graph['bounded_dynamic_loop']['maximum_improvements'],1)
        tool=next(row for row in graph['nodes'] if row['id']=='selected_edit_tool')
        from photo_edit_tools import make_tools
        expected=make_tools('.', '.', scope=scope,session_directory='.')
        actual=tool['controls']['available_tools']
        self.assertEqual(set(actual),{item.name for item in expected})
        for item in expected:
            self.assertEqual(actual[item.name]['operations'],list(item.operations))
            self.assertEqual(actual[item.name]['costs'],item.costs)
            self.assertEqual(actual[item.name]['requires_region'],item.requires_region)
        self.assertTrue(all(row['alpha_role']=='not_applicable' for row in graph['nodes']))
        self.assertNotEqual(graph['graph_id'],self.compile(maximum_improvements=2,reason='Repair the main reviewed issue')['graph_id'])
        self.assertNotEqual(graph['graph_id'],self.compile(editing_scope({'style':'dramatic'}))['graph_id'])
        self.assertNotEqual(graph['graph_id'],self.compile(reason='A different checked feedback',maximum_improvements=1)['graph_id'])
        def changed(path,*args,**kwargs):
            return 'c'*64 if Path(path).name=='photo_edit_tools.py' else sha256_file(path,*args,**kwargs)
        with patch('asset_identity.sha256_file',side_effect=changed):
            self.assertNotEqual(graph['graph_id'],self.compile(scope)['graph_id'])
        self.assertEqual(compile_workflow('existing_reference'),compile_workflow('existing_reference',editing_scope=scope))
        with self.assertRaisesRegex(ValueError,'independent review'):
            compile_workflow('existing_reference',editing_scope=scope,editing_iteration={'maximum_improvements':1})
        for mode,permissions in (('create_reference',scope),('local_portrait',scope),('existing_reference',None)):
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                compile_workflow(mode,editing_scope=permissions,visual_review=True,editing_iteration={'maximum_improvements':1})
        with self.assertRaisesRegex(ValueError,'read-only'):
            compile_workflow('existing_reference',{'beauty':{'enabled':True,'strength':1}},editing_scope=scope,
                visual_review=True,editing_iteration={'maximum_improvements':1})

    def test_saved_iteration_facts_do_not_rewrite_historical_checked_graphs(self):
        fixture=self.fixture();run=fixture.parent;history=copy.deepcopy(run['task_plan'])
        with patch('numpy.load',side_effect=AssertionError('Canvas loaded pixel arrays')):graph=build_graph(run)
        states={row['id']:row['status'] for row in graph['nodes']}
        for key in ('reviewed_inputs','decide_edit','selected_edit_tool','independent_recheck','select_best','freeze_appearance','save_appearance'):
            self.assertEqual(states[key],'complete',key)
        self.assertEqual(run['task_plan'],history);self.assertFalse(graph['cache_hits_inferred'])
        self.assertIn('edit_region',next(row for row in graph['nodes'] if row['id']=='selected_edit_tool')['detail'])
        run['task_plan']['layer_workflow']=compile_workflow('existing_reference');old=copy.deepcopy(run['task_plan'])
        display=build_graph(run);self.assertEqual(display['historical_execution_graph_id'],old['layer_workflow']['graph_id'])
        self.assertEqual(run['task_plan'],old)
        legacy=copy.deepcopy(run);legacy['options'].pop('photo_appearance')
        with self.assertRaisesRegex(ValueError,'Unsupported retired creation'):build_graph(legacy)

    def test_parent_review_cannot_be_counted_as_this_task_evaluation(self):
        fixture=self.fixture();run=fixture.parent;state=fixture.state
        state['attempts']=[{'kind':'decide','status':'completed','response':{'action':'stop'}}]
        fixture.session_path.write_text(json.dumps(state),encoding='utf8')
        states={row['id']:row['status'] for row in build_graph(run)['nodes']}
        self.assertEqual(states['decide_edit'],'complete')
        self.assertEqual(states['selected_edit_tool'],'not_requested');self.assertEqual(states['independent_recheck'],'not_requested')

    def test_unknown_tool_is_attention_and_cannot_be_a_completed_recheck(self):
        fixture=self.fixture();run=fixture.parent;state=fixture.state
        state['status']='needs_attention';state['attempts']=[{'kind':'tool','status':'unknown','costs':{'image_submissions':1,'tool_calls':1}}]
        fixture.session_path.write_text(json.dumps(state),encoding='utf8')
        nodes={row['id']:row for row in build_graph(run)['nodes']}
        self.assertEqual(nodes['selected_edit_tool']['status'],'needs_attention')
        self.assertEqual(nodes['independent_recheck']['status'],'not_requested')
        self.assertIsNone(nodes['selected_edit_tool']['execution']['actual_image_submissions'])

    def test_refinement_canvas_is_readonly_and_shows_typed_ownership(self):
        from cos_workflow_canvas import CosWorkflowCanvas
        app=QApplication.instance() or QApplication([]);fixture=self.fixture();run=fixture.parent
        canvas=CosWorkflowCanvas();canvas.set_run(run);canvas.show();app.processEvents()
        events=[];canvas.templateChanged.connect(events.append)
        try:
            canvas.node_items['selected_edit_tool'].setSelected(True);app.processEvents()
            self.assertIn('模型决定的下一项操作',canvas.ports.text());self.assertIn('原生与生成来源',canvas.ports.text())
            self.assertFalse(canvas.action.isEnabled());self.assertTrue(canvas.strength.isHidden());self.assertFalse(canvas.presets.isEnabled())
            self.assertFalse(bool(canvas.node_items['selected_edit_tool'].flags()&QGraphicsItem.ItemIsMovable))
            with self.assertRaisesRegex(ValueError,'read-only'):revision_for_template(run,preset('cos_scene'))
            canvas.strength.setValue(25);self.assertFalse(events);canvas.grab()
        finally:canvas.close()

    def test_workbench_bridge_displays_actual_typed_iteration_readonly(self):
        from design_reconstruction_ui_smoke import Window,Pool
        app=QApplication.instance() or QApplication([]);fixture=self.fixture();run=fixture.parent
        window=Window(fixture.root,'http://127.0.0.1:1');window.pool=Pool()
        with patch('sys.excepthook') as unhandled:
            try:
                window.load_source_run(run);window.set_professional_mode(True);window.show();app.processEvents()
                canvas=window.workflow_canvas
                self.assertTrue(canvas.isEnabled());self.assertEqual(canvas.graph['run_id'],run['run_id'])
                self.assertIn('selected_edit_tool',canvas.node_items);self.assertNotIn('observe_design',canvas.node_items)
                self.assertFalse(window.local_editor_tabs.isEnabled());self.assertFalse(window.portrait_beauty_prepare.isEnabled())
                self.assertFalse(window.simple_workspace.save_button.isEnabled());self.assertTrue(window.simple_workspace.export_button.isEnabled())
                self.assertEqual(window.previews['final'].path,Path(run['artifacts']['auto_draft']))
                unhandled.assert_not_called()
            finally:window.session_timer.stop();window.close();app.processEvents()


if __name__=='__main__':unittest.main()
