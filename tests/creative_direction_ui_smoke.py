"""Shared art intent, explicit one-region budget and saved-native comparison UI."""
from __future__ import annotations
import copy
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
import qt_workbench as qt
import creative_task as contract
from creative_task_ui_smoke import model_registry_fixture, model_capability_fixture, service_fixture
from design_reconstruction_ui_smoke import Window, Pool, last


def main():
    app = qt.QApplication.instance() or qt.QApplication([])
    with patch('sys.excepthook') as unhandled, tempfile.TemporaryDirectory(prefix='cosplay-direction-ui-') as folder:
        root = Path(folder); source = root/'source.png'; reference = root/'reference.png'
        Image.new('RGB', (4688,7028), '#476d8a').save(source)
        Image.new('RGB', (832,1248), '#6a7c7d').save(reference)
        preview = root/'source-proxy.jpg'; Image.new('RGB',(800,1200),'#476d8a').save(preview)
        window = Window(root,'http://127.0.0.1:1'); window.pool = Pool()
        window.show(); window.set_simple_source(str(source)); app.processEvents()
        quick = window.simple_workspace
        # Current default is automatic per-image direction; the old
        # source_refinement recommendation is opt-in and must not be injected
        # into a fresh task.
        assert window.preset.currentData()==''
        assert not quick.character_name.text() and not window.regional_enabled
        quick.character_name.setText('祝小红'); quick.character_notes.setText('纸嫁衣，原旗袍妆造')
        quick.brief.setText('保留原景不换；主效：两只克制小纸蝴蝶的情感意象，非施法')
        quick.task_mode.setCurrentIndex(quick.task_mode.findData('create_reference'))
        assert window.character_name.text() == '祝小红' and window.creative_task_brief.toPlainText() == quick.brief.text()
        assert window.creative_task_payload()['constraints'] == {'background_mode':'enhance_source','preserve_background':False,
            'no_effects':False}
        assert window.creative_task_payload()['brief']==quick.brief.text()
        window.preset.setCurrentIndex(window.preset.findData('stage_light'))
        assert 'refinement_profile' not in window.creative_task_payload()['constraints']
        window.preset.setCurrentIndex(window.preset.findData('source_refinement'))
        assert not window.creative_text_vision.isEnabled() and window.creative_text_vision.isChecked()
        controls = window.editing_scope_panels[-1]
        controls['style'].setCurrentIndex(controls['style'].findData('dramatic'))
        assert window.creative_task_payload()['constraints']['background_mode'] == 'enhance_source'
        assert window.editing_request_payload()['target_setting'] == 'same'
        assert window.creative_task_payload()['services']['text_vision'] is True
        # The display proxy is low resolution; native dimensions come only from source facts.
        window.result('source-preview', {'source':str(source), 'proxy_path':str(preview),'width':4688,'height':7028})
        assert window.source_native_size() == [4688,7028]
        # The old manually boxed source-native regional tool remains available
        # only for an imported reference, separate from scoped generation.
        assert quick.regional.isHidden()
        quick.task_mode.setCurrentIndex(quick.task_mode.findData('existing_reference'))
        window.creative_task_reference.setText(str(reference))
        window.refresh_cos_workspace(); quick.regional.enabled.setChecked(True)
        for key, value in (('left',256),('top',512),('width',768),('height',768)):
            quick.regional.edits[key].setValue(value)
        quick.regional.apply.click()
        assert window.regional_core_box == [256,512,1024,1280]
        assert quick.source_image.selection == [256/4688,512/7028,1024/4688,1280/7028]
        # Letterbox-aware drag coordinates map to the actual oriented source canvas.
        panel = quick.source_image; app.processEvents(); panel._read_image()
        rect = panel.image_rect()
        point = panel.image_point(rect.center())
        assert abs(point[0]-.5)<1e-6 and abs(point[1]-.5)<1e-6
        quick.select_region.setChecked(True)
        start=QPoint(round(rect.left()+rect.width()*.05), round(rect.top()+rect.height()*.08))
        end=QPoint(round(rect.left()+rect.width()*.15), round(rect.top()+rect.height()*.18))
        QTest.mousePress(panel,Qt.LeftButton,pos=start); QTest.mouseMove(panel,end); QTest.mouseRelease(panel,Qt.LeftButton,pos=end)
        assert window.regional_core_box and not quick.select_region.isChecked()
        assert window.regional_core_box[2]-window.regional_core_box[0] <= 768
        window.set_normalized_regional_box([256/4688,512/7028,1024/4688,1280/7028])
        assert window.regional_core_box == [256,512,1024,1280]
        with patch('current_model_registry.registry_status', return_value=model_registry_fixture()), \
             patch('task_model_preflight.inspect_task_models', side_effect=model_capability_fixture), \
             patch.object(contract, '_services', side_effect=service_fixture), \
             patch('reference_provider.provider_config', return_value={'kind':'comfyui','registry_id':'qwen_image_2_1_uncensored_fast'}), \
             patch('reference_provider.provider_health', return_value={'ready':True}):
            quick.check_plan_button.click(); pending = window.pool.tasks[-1]
            assert pending.path == '/api/tasks/plan'
            assert pending.body['character_name'] == '祝小红'
            region = pending.body['creative_task']['regional_refinement']
            assert region == {'maximum_submissions':1,'coordinate_space':'source_canvas','regions':[{'id':'bg-1','core_box':[256,512,1024,1280]}]}
            checked = contract.plan_task(root,pending.body)
            pending.signals.result.emit(checked)
            assert '祝小红' in quick.direction_summary.text() and '小纸蝴蝶' in quick.direction_summary.text()
            assert '局部' in window.creative_plan_summary.toPlainText() and quick.start_plan_button.isEnabled()
            quick.start_plan_button.click(); started = window.pool.tasks[-1]
            assert started.path == '/api/tasks/start'
            assert started.body['creative_direction'] == checked['plan']['creative_direction']
            # An open saved version must not replace the running checked task's direction
            # or make an explicitly budgeted region appear to be zero-generation.
            window.product_run = {'run_id':'current-r4','options':{
                'execution_strategy': 'layered_reference_transfer',
                'task_intent': {'mode': 'existing_reference'}}}
            started.signals.result.emit({'ok':True,'async':True,'job_id':'job-direction-running'})
            window.refresh_cos_workspace()
            assert window.active_checked_task['job_id']=='job-direction-running'
            assert '本次制作已冻结方向' in quick.direction_summary.text()
            assert '祝小红' in quick.direction_summary.text() and '小纸蝴蝶' in quick.direction_summary.text()
            assert '局部精修 1' in quick.regional.note.text() and '不新增' not in quick.regional.note.text()
            expected_total = sum(checked['plan']['image_budget'].values())
            assert f'总上限：{expected_total}' in quick.regional.note.text() and '不是外部付费' in quick.regional.note.text()
            assert not quick.regional.enabled.isEnabled()
            window.result('job-status', {'job':{'job_id':'job-direction-running','kind':'one_click',
                'status':'failed','message':'captured terminal, no production'}})
            window.refresh_cos_workspace()
            assert window.active_checked_task is None and '旧版本不会补写' in quick.direction_summary.text()
            # A different route never inherits an old positive regional budget.
            for mode in ('local_portrait','existing_reference'):
                window.creative_task_mode.setCurrentIndex(window.creative_task_mode.findData(mode))
                payload = window.creative_task_request_options(mode)['creative_task']
                assert not (payload.get('regional_refinement') or {}).get('maximum_submissions')
            # Existing-reference enhancement may explicitly authorize one NEW
            # regional request while continuing to reuse its global reference.
            window.creative_task_mode.setCurrentIndex(window.creative_task_mode.findData('existing_reference'))
            window.set_background_mode('enhance_source'); window.creative_task_reference.setText(str(reference))
            assert '默认零生图' in quick.task_mode.currentText()
            assert '默认零生图' in window.creative_task_mode.currentText()
            assert '复用参考' in quick.existing_reference_button.text()
            assert '主动授权' in quick.existing_reference_button.toolTip()
            assert not window.regional_enabled and quick.regional.enabled.isEnabled()
            quick.regional.enabled.setChecked(True)
            assert window.creative_task_payload()['regional_refinement']['maximum_submissions']==1
            request_body=window.creative_task_request_options('existing_reference')
            request_body.update(source=str(source),reference=str(reference))
            local_checked=contract.plan_task(root,request_body)
            assert local_checked['plan']['image_budget']['initial_reference']==0
            assert local_checked['plan']['image_budget']['regional_refinement']==1
            assert '总上限：1' in qt.creative_task_plan_text(local_checked['plan'])
            window.creative_task_mode.setCurrentIndex(window.creative_task_mode.findData('create_reference'))
            assert not window.regional_enabled
            # Changing any shared intent invalidates a checked plan.
            quick.check_plan_button.click(); request = window.pool.tasks[-1]
            request.signals.result.emit(contract.plan_task(root,request.body))
            quick.character_name.setText('另一角色')
            assert window.creative_task_plan is None and not quick.start_plan_button.isEnabled()
        # Saved old versions remain honest: no invented frozen character or region budget.
        window.apply_creative_task_options({'schema_version':'cosplay_task_intent.v1','mode':'existing_reference',
            'brief':'旧方向','constraints':{'preserve_background':False,'no_effects':False},
            'services':{'text_vision':False,'visual_review':False}})
        assert window.creative_background_mode.currentData() == 'replace' and not window.regional_enabled
        window.source_run_id='saved'; window.source_revision=4; window.source_status='needs_visual_review'
        window.source_run_has_reference=False
        window.source_run_source=str(source); window.native_detail_button.setEnabled(True)
        window.creative_task_mode.setCurrentIndex(window.creative_task_mode.findData('create_reference'))
        window.set_background_mode('enhance_source'); window._session_dirty=False
        window.check_creative_task_plan()
        assert window.pool.tasks[-1].body['parent_run_id']=='saved'
        window.creative_task_plan_request=None
        window.request_native_detail([.3,.2]); task=window.pool.tasks[-1]
        assert task.body['include_source'] is True and task.body['center']==[.3,.2]
        crop=root/'crop.png'; base=root/'base.png'
        Image.new('RGB',(128,128),'red').save(crop); Image.new('RGB',(128,128),'blue').save(base)
        window.show_native_detail({'run_id':'saved','revision':4,'native_size':[4688,7028],
            'crop_box':[100,200,228,328],'preview_path':str(crop),'source_preview_path':str(base),
            'source_comparison':{'basis':'saved_pre_beauty_native_baseline','resampled':False}})
        labels=[label.text() for label in window.native_detail_dialog.findChildren(qt.QLabel)]
        assert '原片基线（美化前）' in labels and '已保存成片' in labels
        assert len([label for label in window.native_detail_dialog.findChildren(qt.QLabel) if label.pixmap() and not label.pixmap().isNull()])==2
        window.native_detail_dialog.close()
        window._session_dirty=False; window.render_pending=False
        window.creative_task_start_pending=False; window.job_rows={}
        window.regional_core_box=[192,448,960,1216]
        window.regional_controls.sync([4688,7028],False,window.regional_core_box,cpu_reuse=True)
        assert window.regional_controls.geometry.isEnabled()
        assert not window.regional_controls.enabled.isChecked()
        assert '不新增生图' in window.regional_controls.note.text()
        window.run_product_action({'action_id':'retry_local_processing','supported':True,
            'regional_candidate_reuse':True,'generation_required':False})
        assert window.pool.tasks[-1].body=={'generate':False,'async':True,
            'regional_reuse_core_box':[192,448,960,1216]}
        window.session_timer.stop(); window.close()
        # A saved no-profile task must not acquire the newly selected GUI
        # default, even when a preset shares the recommendation's identifier.
        legacy = Window(root, 'http://127.0.0.1:1');legacy.pool=Pool()
        old={'schema_version':'cosplay_task_intent.v1','mode':'create_reference','brief':'原有要求',
            'constraints':{'preserve_background':False,'no_effects':False,
                'background_mode':'enhance_source','reference_role':'source_conditioned_design'},
            'services':{'text_vision':False,'visual_review':False}}
        legacy.apply_creative_task_options(old)
        assert legacy.preset.currentData()==''
        assert 'refinement_profile' not in legacy.creative_task_payload()['constraints']
        legacy.session_timer.stop();legacy.close()
    # A failed-run continuation reuses the global reference but still grants
    # exactly one local image-model submission. Offline does not mean zero GPU calls.
    unhandled.assert_not_called()
    resume_budget = {'initial_reference':0,'reference_correction':0,'independent_effect':0,'regional_refinement':1}
    text = qt.creative_task_plan_text({'mode':'existing_reference','ready':True,'image_budget':resume_budget,
        'services':{'offline':True,'text_vision':False}})
    assert '新增图像模型提交总上限：1' in text
    assert '全局参考 0 / 参考纠错 0 / 独立效果 0 / 局部精修 1' in text
    assert '本地 GPU' in text and '不是外部付费图像请求' in text
    assert '总上限：0' in qt.image_submission_budget_text({})
    print(json.dumps({'ok':True,'shared_direction':True,'native_region_not_proxy_size':True,
        'regional_opt_in_explicit_reference_tasks_only':True,'frozen_plan_transmitted':True,'same_coordinate_native_pair':True}))


if __name__=='__main__': main()
