"""Verify the quick Cos route exposes monotonic overall/local progress."""
from __future__ import annotations

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch

import qt_workbench as qt
import creative_task
from creative_direction_smoke import enhanced, legacy
from creative_task_smoke import fixture
from task_model_preflight_smoke import mocked_task_capability
from simple_cos_workspace import SimpleCosWorkspace
from design_reconstruction_ui_smoke import Window, Pool


def checked_routes(window):
    """Real pure plans drive the labels; no worker or model is started."""
    app=qt.QApplication.instance() or qt.QApplication([])
    quick=SimpleCosWorkspace()
    assert '4 个阶段' not in quick.progress.toolTip()
    assert len(quick.progress.SEGMENTS)==1, 'Plan items cannot be drawn as invented time partitions'
    with tempfile.TemporaryDirectory(prefix='checked-progress-') as folder:
        root=Path(folder);registry,sources,reference=fixture(root)
        base={'source':sources[0],'model_registry':str(registry),
            'provider':'qwen_image_2_1_uncensored_fast','input_prompt':'Keep source.'}
        cases=[('local-source-first',enhanced(),{},'comfyui'),
            ('remote',legacy(),{},'openai-compatible'),
            ('multiple',legacy(),{'variant_count':2,'defer_reference_selection':True},'comfyui'),
            ('existing',enhanced('existing_reference'),{'reference':reference},'comfyui'),
            ('portrait',legacy('local_portrait'),{},'comfyui')]
        with patch('task_model_preflight.inspect_task_models',side_effect=mocked_task_capability), \
                patch('urllib.request.urlopen',side_effect=AssertionError('No network in UI progress')):
            for name,intent,extra,kind in cases:
                with patch('reference_provider.provider_health',return_value={'ready':True,'kind':kind}):
                    checked=creative_task.plan_task(root,{**base,**extra,'creative_task':intent})
                stages=checked['plan']['stages'];labels=[row['label'] for row in stages]
                window.active_checked_task={'progress_stages':copy.deepcopy(stages)}
                window.creative_task_plan=None
                window.simple_creation={'source':sources[0],'mode':'beauty' if name=='portrait' else 'create_cos','stage':'production'}
                window._reset_quick_progress()
                assert window._quick_progress_stages()==stages
                quick.set_progress_stages(window._quick_progress_stages())
                assert f'共 {len(stages)} 项' in quick.progress.toolTip()
                assert quick.progress.toolTip().index(labels[0])<quick.progress.toolTip().index(labels[-1])
                if name=='local-source-first':
                    assert labels.index('原片人像美化并冻结（本地生图前）')<labels.index('创作风格参考')
                else:
                    assert not any('本地生图前' in label for label in labels)
                if name=='existing':assert '复用已有参考' in labels
                if name=='portrait':assert len(stages)==3 and not any('参考' in label for label in labels)
                actual=window._quick_progress(.25,'generating','one_click',timing={
                    'module':'portrait_beauty','module_label':'原片人像美化','elapsed_seconds':12})
                assert actual[1]=='原片人像美化' and actual[3] is None and actual[4] is None
                quick.set_job_progress(actual[0],actual[1],local_progress=actual[3],stage_index=actual[4])
                assert f'本次路线 {len(stages)} 项' in quick.progress_detail.text()
                assert '当前：原片人像美化' in quick.progress_detail.text()
                assert '本步骤' not in quick.progress_detail.text(), 'whole-job percent is not a phase fraction'
                newer=window._quick_progress(.7,'rendering','one_click',timing={'module_label':'原尺寸渲染','elapsed_seconds':100})
                older=window._quick_progress(.4,'generating','one_click',timing={'module_label':'创作参考','elapsed_seconds':80})
                assert older[0]==newer[0], 'poll/SSE must not regress milestone percentage'
                assert older[1]=='原尺寸渲染', 'older timing events cannot regress the observed module'
                assert '不表示耗时比例' in quick.progress.toolTip()
                if name=='multiple':
                    window.simple_creation['stage']='awaiting_variant'
                    waiting=window._quick_progress(.4,'awaiting_reference','one_click')
                    assert waiting[1]=='awaiting_reference' and '请选择' in waiting[2]
    window.active_checked_task=None;window.creative_task_plan=None;window.simple_creation=None
    assert window._quick_progress_stages()==[]
    quick.set_progress_stages([]);quick.set_job_progress(None,'workflow')
    assert '4' not in quick.progress_detail.text() and '尚未确认' in quick.progress.toolTip()
    # Exercise actual polling and SSE result handlers, not just the mapper.
    with tempfile.TemporaryDirectory(prefix='progress-events-') as folder:
        observed=Window(Path(folder),'http://127.0.0.1:1');observed.pool=Pool()
        observed.simple_creation={'source':'source.png','mode':'create_cos','stage':'production'}
        observed.active_job_id='progress-job'
        observed.active_checked_task={'callback':'one-click','job_id':'progress-job','progress_stages':[
            {'id':'1','label':'源Beauty先行'},{'id':'2','label':'创作风格参考'},
            {'id':'3','label':'原生保存'},{'id':'4','label':'人工审片'}]}
        job={'job_id':'progress-job','kind':'one_click','status':'running','stage':'generating',
            'progress':.25,'timing':{'request_id':'timing','module_label':'原片人像美化','elapsed_seconds':20}}
        observed.result('job-status',{'job':job})
        assert '当前：原片人像美化' in observed.simple_workspace.progress_detail.text()
        job.update(progress=.4,timing={'request_id':'timing','module_label':'创作参考','elapsed_seconds':50})
        observed.result('job-stream',{'data':job})
        assert '当前：创作参考' in observed.simple_workspace.progress_detail.text()
        assert '本步骤' not in observed.simple_workspace.progress_detail.text()
        assert '源Beauty先行 → 创作风格参考' in observed.simple_workspace.progress.toolTip()
        job.update(progress=.25,timing={'request_id':'timing','module_label':'原片人像美化','elapsed_seconds':20})
        observed.result('job-status',{'job':job})
        assert '当前：创作参考' in observed.simple_workspace.progress_detail.text()
        observed.job_poll_timer.stop();observed.simple_creation=None;observed.active_checked_task=None;observed.close()
    quick.close();app.processEvents()


def main():
    window = qt.Workbench.__new__(qt.Workbench)
    window.simple_creation = {'source': 'source.png', 'mode': 'create_cos', 'stage': 'colour'}
    window._quick_progress_key = None
    window._quick_progress_value = 0.0
    first = window._quick_progress(1.0, 'reference_complete', '')
    stale = window._quick_progress(0.1, 'reference_started', '')
    assert abs(first[0] - .69) < 1e-9 and stale[0] == first[0], 'stale worker events must not move the route backwards'
    assert first[3] == 1.0 and first[4] == 2, 'fusion must report its local and overall stage index'
    assert stale[3] == 1.0, 'a stale refresh cannot move local progress backwards either'
    empty_refresh = window._quick_progress(None, '', '')
    assert empty_refresh[3] == 1.0

    window.simple_creation['stage'] = 'boundary'
    boundary = window._quick_progress(.5, 'boundary_light', 'native_design_reconstruction')
    assert boundary[0] >= first[0] and boundary[3] == .5 and boundary[4] == 2

    window.simple_creation['stage'] = 'post_save_wait'
    saved = window._quick_progress(0.0, 'saving', '')
    assert saved[0] >= first[0] and saved[4] == 3
    assert saved[3] == 0.0, 'a new stage must start with its own local fraction'
    window.simple_creation['stage'] = 'post'
    post = window._quick_progress(.25, 'effects', '')
    assert post[0] >= saved[0] and post[3] == .25
    window.simple_creation['stage'] = 'beauty'
    window.active_job_id = 'old-selection'
    transition = window._quick_progress(1., 'complete', 'select_reference_variant')
    assert transition[3] is None, 'completed selection cannot make the new Beauty stage 100%'
    window.active_job_id = 'new-beauty'
    fresh = window._quick_progress(.02, 'preparing', 'prepare_portrait_beauty')
    assert fresh[3] == .02 and fresh[4] == 1
    assert fresh[0] >= saved[0], 'switching stage/job within the route cannot reset overall progress'
    checked_routes(window)
    print({'ok': True, 'monotonic_overall': True, 'local_percent_available': True,
           'boundary_light_stage': True, 'fusion_to_post_no_regression': True,
           'checked_routes_source_first_remote_multiple_existing_portrait':True,
           'actual_module_not_whole_job_phase_percent':True,'no_fixed_four_stages':True})


if __name__ == '__main__':
    main()
