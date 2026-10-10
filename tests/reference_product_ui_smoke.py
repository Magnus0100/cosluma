"""Capture planned reference batches, isolated retry and persisted split controls."""
from __future__ import annotations
import copy, json, os, tempfile
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PIL import Image
import creative_task as contract
from creative_task_ui_smoke import model_capability_fixture, model_registry_fixture, service_fixture
import qt_workbench as qt

class Window(qt.Workbench):
    def refresh(self): pass

class Pool:
    def __init__(self): self.tasks=[]
    def start(self,task): self.tasks.append(task)

def main():
    app=qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='cosplay-product-ui-') as folder:
        root=Path(folder);window=Window(root,'http://127.0.0.1:1');pool=Pool();window.pool=pool
        window.source_run_id='parent'; window.source_run_source_sha256='source'; window.source_run_candidate_sha256='reference'
        window._paired_subject_state=lambda *args:{'reconstruction_required':False}
        from current_stack_fixture import prepare_fixture
        fixture=prepare_fixture(root,native=(60,90))
        current=fixture['options']['appearance_revision']['design_reconstruction_plan']
        current['layers'][0]['strength']=.8
        window.source_run_source_sha256=fixture['source']['sha256'];window.source_run_candidate_sha256=fixture['reference']['sha256']
        assert not hasattr(window,'reference_control_upgrade')
        support={'schema':'reference_support.v1','regions':{'left_arm':{
            'geometry':{'status':'unsupported','reason':'pose_mismatch'},
            'material':{'status':'supported'},'statistics':{'status':'partial'},
            'spatial_colour':{'status':'unsupported'},'lightness':{'status':'unsupported'},'donor':{'status':'partial'}}}}
        v2=copy.deepcopy(current)
        with patch('source_appearance_responsibility.reference_stack_state',return_value={'reference_support':support}):
            window.load_design_reconstruction(v2)
            assert all(spin.isEnabled() for spin in window.reference_control_spins.values())
            assert '左臂' in window.reference_support_text.toPlainText() and '降级' in window.reference_support_text.toPlainText()
            window.reference_control_spins['colour_strength'].setValue(.3)
            window.reference_control_spins['lightness_strength'].setValue(.6);window.source_light_timer.stop()
            payload=window._source_edit_payload()['appearance_revision']['design_reconstruction_plan']['layers'][0]
            assert payload['strength']==.8 and payload['colour_strength']==.3 and payload['lightness_strength']==.6
            snapshot=window._session_snapshot()
            override=snapshot['design_overrides']['layers'][0]
            assert override['colour_strength']==.3 and override['lightness_strength']==.6
            window.load_design_reconstruction(v2);window._restore_session_payload(snapshot)
            restored=window.design_reconstruction_plan['layers'][0]
            assert restored['colour_strength']==.3 and restored['lightness_strength']==.6
        assert v2['layers'][0]['colour_strength']==1, 'Do not mutate backend plan'
        damaged=copy.deepcopy(v2);damaged['layers'][0]['params']['reference_support']=support
        with patch('source_appearance_responsibility.reference_stack_state',return_value={'reference_support':None}):
            window.load_design_reconstruction(damaged)
            assert '未通过' in window.reference_support_text.toPlainText()
            assert '左臂' not in window.reference_support_text.toPlainText()
        second_source=root/'second-source.png';Image.new('RGB',(60,90),'#756585').save(second_source)
        second_reference=root/'second-reference.png';Image.new('RGB',(60,90),'#455765').save(second_reference)
        window.local_batch_pairs=[{'source':fixture['source']['path'],'reference':fixture['reference']['path']},
            {'source':str(second_source),'reference':str(second_reference)}]
        with patch('current_model_registry.registry_status',return_value=model_registry_fixture()), \
             patch('task_model_preflight.inspect_task_models',side_effect=model_capability_fixture), \
             patch.object(contract,'_services',side_effect=service_fixture):
            window.run_local_reference_batch();request=pool.tasks[-1]
            assert request.path=='/api/tasks/batch/plan' and request.body['creative_task']['mode']=='existing_reference'
            assert request.body['items']==window.local_batch_pairs
            assert not {'appearance_revision','manual_layers','generation_design','character_profile'} & request.body.keys()
            count=len(pool.tasks);window.start_planned_creative_task();assert len(pool.tasks)==count
            files_before=sorted(str(path.relative_to(root)) for path in root.rglob('*'))
            checked=contract.plan_batch(root,request.body);plan=checked['batch_plan']
            assert files_before==sorted(str(path.relative_to(root)) for path in root.rglob('*'))
            assert plan['ready'] and sum(plan['image_budget_total'].values())==0
            assert not plan['per_image_geometry_shared']
            for item,pair in zip(plan['items'],window.local_batch_pairs):
                assert item['plan']['mode']=='existing_reference'
                for kind in ('source','reference'):
                    assert item['plan']['inputs'][kind]['sha256']==contract.sha(Path(pair[kind]))
                capabilities=item['plan']['capabilities']
                assert capabilities['model_preflight']['test_model_capability_mocked'] is True
                assert capabilities['inference_verified'] is False
                execution=contract.execution_request(root,{**item['request'],'plan_id':item['plan']['plan_id']})
                assert execution['complete_post'] is True and execution['generate'] is False
            assert len({item['plan']['inputs']['source']['sha256'] for item in plan['items']})==2
            request.signals.result.emit(checked)
            assert window.creative_start_button.isEnabled()
            assert '模型文件/小文件身份与配置检查；权重内容和推理在执行验证' in window.creative_plan_summary.toPlainText()
            window.creative_start_button.click();started=pool.tasks[-1]
            assert started.path=='/api/tasks/batch' and started.body['plan_id']==plan['plan_id']
            assert contract.plan_batch(root,started.body)['batch_plan']['plan_id']==plan['plan_id']
        record={'kind':'creative_task_batch','shared_task':{'mode':'existing_reference'},'batch_id':'batch-fixture','items':[
            {'index':0,**window.local_batch_pairs[0],'job':{'status':'failed'}},
            {'index':1,**window.local_batch_pairs[1],'job':{'status':'succeeded'}}]}
        started.signals.result.emit({'batch':record})
        window.local_batch_id=record['batch_id'];window.show_local_batch(record)
        window.local_batch_tree.setCurrentItem(window.local_batch_tree.topLevelItem(0));window.retry_local_batch_item()
        assert pool.tasks[-1].path.endswith('/retry-item') and pool.tasks[-1].body=={'index':0}
        retry=pool.tasks[-1];before=len(pool.tasks);window.retry_local_batch_item();assert len(pool.tasks)==before
        retry.signals.result.emit({'batch':record})
        before=len(pool.tasks);window.local_batch_tree.setCurrentItem(window.local_batch_tree.topLevelItem(1));window.retry_local_batch_item()
        assert len(pool.tasks)==before
        window.local_batch_restore_button.click()
        assert pool.tasks[-1].method=='GET' and pool.tasks[-1].path=='/api/batch'
        window.result('local-batch-restore',{'batches':[{'kind':'generation','created_at':'2099'},
            {**record,'created_at':'2026-10-07'},{**record,'batch_id':'batch-newer','created_at':'2026-10-08'}]})
        assert window.local_batch_id=='batch-newer' and pool.tasks[-1].path=='/api/batch/batch-newer'
        window.result('recovery',{'source_runs':[{'run_id':'one','sample':'one','auto_draft':True,'status':'needs_visual_review'},
            {'run_id':'two','sample':'two','auto_draft':True,'status':'needs_visual_review'}]})
        for i in range(2):window.saved_export_versions.topLevelItem(i).setCheckState(0,qt.Qt.Checked)
        window.export_directory.setText(str(root/'exports'));window.export_artifact.setCurrentIndex(0)
        window._unsaved_session_blocks_export=lambda:False
        window.export_selected_versions();request=pool.tasks[-1]
        assert request.path=='/api/source-runs/export-batch'
        assert request.body['items']==[{'run_id':'one','artifact':'draft'},{'run_id':'two','artifact':'draft'}]
        before=len(pool.tasks);window._unsaved_session_blocks_export=lambda:True;window.export_selected_versions()
        assert len(pool.tasks)==before
        assert not any(task.method=='POST' and task.path in {'/api/one-click','/api/batch','/api/batch/local-reference'} for task in pool.tasks)
        window.close();app.processEvents()
        print(json.dumps({'ok':True,'retired_upgrade_absent':True,'v2_split_controls_and_session_restore':True,
            'regional_degradation_display':True,'planned_pair_batch_and_failed_retry_actions':True,
            'safe_request_plan_id_start':True,
            'local_batch_restart_restore_action':True,'cross_photo_saved_export_selection':True,'unsaved_edit_export_blocked':True,'no_generation':True,
            'real_Qt_controls':True,'current_task_contract':True,'real_input_hashes':True,
            'synthetic_current_assets':True,'transport_and_support_metadata_mocked':True,
            'test_model_capability_mocked':True,'inference_verified':False,
            'network_requests':0,'model_inferences':0}))

if __name__=='__main__':main()
