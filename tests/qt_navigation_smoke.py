#!/usr/bin/env python3
"""Exercise desktop navigation with captured tasks; no backend or network."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import qt_workbench as qt
from generation_prompt import DEFAULT_PROMPT


def main() -> int:
    assert qt.QT_AVAILABLE, "PySide6 required for desktop interaction smoke"
    app = qt.QApplication.instance() or qt.QApplication([])

    class CapturedTasks:
        def __init__(self): self.tasks = []
        def start(self, task): self.tasks.append(task)

    class Workbench(qt.Workbench):
        def refresh(self):
            if getattr(self, "capture_ready", False): super().refresh()

    temporary_workspace = tempfile.TemporaryDirectory(prefix='cosplay_navigation_')
    window = Workbench(Path(temporary_workspace.name), "http://127.0.0.1:1")
    pool = CapturedTasks(); window.pool = pool
    def start_checked_plan(request):
        """Transport fixture; the dedicated task smoke checks the real compiler."""
        import copy
        batch = request.path == '/api/tasks/batch/plan'
        assert request.path in {'/api/tasks/plan', '/api/tasks/batch/plan'}
        mode = request.body['creative_task']['mode']
        plan = {'schema_version': 'cosplay_batch_task_plan.v1' if batch else 'cosplay_task_plan.v1',
                'plan_id': 'c'*64, 'mode': mode, 'ready': True, 'items': [], 'image_budget_total': {}}
        safe_request = copy.deepcopy(request.body)
        if not batch:
            safe_request.update(generate=mode == 'create_reference', local_only=mode == 'local_portrait',
                                complete_post=mode != 'local_portrait')
        request.signals.result.emit({'ok': True, 'batch_plan' if batch else 'plan': plan, 'request': safe_request})
        window.creative_start_button.click()
        started = pool.tasks[-1]
        assert started.path == ('/api/tasks/batch' if batch else '/api/tasks/start')
        assert started.body['plan_id'] == plan['plan_id']
        return started
    assert window.prompt.toPlainText() == DEFAULT_PROMPT
    assert window.tabs.widget(0).isAncestorOf(window.prompt)
    assert not window.log_toggle.isChecked() and not window.tabs.tabBar().isVisible()
    assert not window.final_prompt_toggle.isChecked()
    window.navigation.setCurrentRow(1)
    assert window.tabs.currentIndex() == 1
    window.tabs.setCurrentIndex(2)
    assert window.navigation.currentRow() == 2
    window.tabs.setCurrentIndex(0)
    assert window.variant_count.value() == 1 and window.preset.currentData() == ""
    assert not window.character_name.text() and not window.character_notes.text()
    assert window.source_retry.text() == "重新生图" and "不会自动" in window.source_retry.toolTip()
    before_error = len(pool.tasks)
    window.error("request timed out", "one-click")
    assert len(pool.tasks) == before_error and "不会自动重试生成" in window.operation_state.text()
    window.capture_ready = True
    window.refresh_btn.click()
    assert [(task.method, task.path, task.body, task.label) for task in pool.tasks] == [
        ("GET", "/api/status", None, "status"),
        ("GET", "/api/recovery", None, "recovery"),
        ("GET", "/api/jobs", None, "jobs-list"),
        ("GET", "/api/providers", None, "providers"),
        ("GET", "/api/provider-settings", None, "provider-settings"),
        ("GET", "/api/templates", None, "templates"),
        ("GET", "/api/recent", None, "recent"),
        ('GET', '/api/model-setup', None, 'model-setup'),
    ]
    window.capture_ready = False

    def click_button(text):
        button = next(button for button in window.findChildren(qt.QPushButton) if button.text() == text)
        count = len(pool.tasks)
        button.click()
        assert len(pool.tasks) == count + 1, f"{text} must dispatch one request"
        return pool.tasks[-1]

    request = click_button("查看可清理空间")
    assert (request.method, request.path, request.body, request.label) == ("GET", "/api/cleanup", None, "cleanup-preview")
    original_question = qt.QMessageBox.question
    try:
        qt.QMessageBox.question = lambda *args, **kwargs: qt.QMessageBox.StandardButton.No
        before = len(pool.tasks)
        next(button for button in window.findChildren(qt.QPushButton) if button.text() == "清理可重建文件").click()
        assert len(pool.tasks) == before
        qt.QMessageBox.question = lambda *args, **kwargs: qt.QMessageBox.StandardButton.Yes
        request = click_button("清理可重建文件")
        assert (request.method, request.path, request.body, request.label) == ("POST", "/api/cleanup", {"apply": True}, "cleanup-apply")
    finally:
        qt.QMessageBox.question = original_question
    window.provider_discover.click()
    request = pool.tasks[-1]
    assert (request.method, request.path, request.body, request.label) == ("GET", "/api/provider-settings/discover", None, "provider-discovery")
    window.result("recent", {"recent": [{"sample": "demo", "run_id": "recent-run"}]})
    request = click_button("打开最近运行")
    assert (request.method, request.path, request.label) == ("GET", "/api/source-runs/recent-run", "history-run-selected")
    window.source_edit.setText("C:/fixture/source.arw")
    assert not window.character_recognize.isEnabled() and not window.prepare_design_button.isEnabled()
    window.creative_task_mode.setCurrentIndex(window.creative_task_mode.findData('create_reference'))
    window.character_recognize.click()
    request = pool.tasks[-1]
    assert request.path == "/api/character/recognize" and request.body["source"] == "C:/fixture/source.arw"
    assert request.body["character_name"] == "" and request.body["character_notes"] == ""
    assert request.body['creative_task']['services']['text_vision'] is True
    profile = {"character_name": "", "identification": "unknown", "confidence": "low",
               "visual_cues": ["blue costume", "night setting"], "recommended_styles": [
                   {"id": "night_scene", "reason": "Cool costume matches evening light", "english_direction": "Use a restrained violet evening atmosphere."}]}
    request.signals.result.emit({"ok": True, "profile": profile})
    assert "未知角色" in window.character_result.text() and "blue costume" in window.character_result.text()
    assert window.recommended_styles.count() == 1 and not window.character_name.text()
    window.character_apply_style.click()
    assert window.preset.currentData() == "night_scene" and profile["recommended_styles"][0]["english_direction"] in window.prompt.toPlainText()
    pool.tasks.clear()
    catalog = {"providers": [{"id": "comfyui", "model": None, "status": "missing_workflow"}]}
    window.result("providers", catalog)
    window.model.setText("C:/custom/workflow.json")
    window.result("providers", catalog)
    assert window.model.text() == "C:/custom/workflow.json"

    window.populate_jobs({"jobs": [
        {"job_id": "A", "status": "running", "progress": .25, "run_id": "run-A"},
        {"job_id": "B", "status": "succeeded", "progress": 1, "run_id": "run-B"},
    ]})
    window.jobs_tree.setCurrentItem(window.jobs_tree.topLevelItem(0))
    stale = list(pool.tasks)
    window.jobs_tree.setCurrentItem(window.jobs_tree.topLevelItem(1))
    current = pool.tasks[-1]
    assert window.active_job_id == "B" and current.label == "job-run-loaded"
    current.signals.result.emit({"run": {"run_id": "run-B", "sample": "demo", "status": "awaiting_reference"}})
    app.processEvents()
    assert window.source_run_id == "run-B"
    state = window.operation_state.text()
    for task in stale:
        if getattr(task, "label", None) == "job-run-loaded":
            task.signals.result.emit({"run": {"run_id": "run-A", "sample": "demo"}})
        elif getattr(task, "label", None) == "job-status":
            task.signals.result.emit({"job": {"status": "running"}})
        else:
            task.signals.result.emit({"data": {"status": "succeeded", "run_id": "run-A"}})
        task.signals.error.emit("obsolete response")
        task.signals.finished.emit()
    app.processEvents()
    assert window.source_run_id == "run-B" and window.operation_state.text() == state

    run = {"run_id": "child", "sample": "demo", "revision": 2, "status": "needs_visual_review",
           "source": {"path": "C:/fixture/source.arw"},
           "artifacts": {"alpha": "C:/fixture/trust.png", "silhouette": "C:/fixture/silhouette.png", "structure_safe": "C:/fixture/structure_safe.png"}, "options": {
        "appearance_revision": {"params": {"exposure": .25, "saturation": .1}},
        "manual_layers": {"outline_light": {"enabled": True, "opacity": .2, "width": 9}},
        "prompt": DEFAULT_PROMPT + "\nKeep a clean neutral portrait finish.",
    }}
    from current_stack_fixture import prepare_fixture
    fixture=prepare_fixture(window.root_path/'test-current-stack')
    run['source']['sha256']=fixture['source']['sha256'];run['reference']=fixture['reference']
    run['input_mode']='single_source';run['options']['render_mode']='native_source_design'
    run['options']['appearance_revision']['design_reconstruction_plan']=fixture['options']['appearance_revision']['design_reconstruction_plan']
    window.load_source_run(run)
    assert window.prompt.toPlainText() == run["options"]["prompt"]
    assert json.loads(window.appearance.toPlainText())["params"] == {"exposure": .25, "saturation": .1}
    assert window.appearance_sliders["exposure"].value() == 25
    assert not hasattr(window, "align_scale")
    assert json.loads(window.manual_layers.toPlainText())["layers"]["outline_light"]["width"] == 9
    window.source_copy.click()
    assert (pool.tasks[-1].method, pool.tasks[-1].path) == ("POST", "/api/source-runs/child/copy")
    assert all(box.currentText() == "not_checked" for box in window.visual_checks.values())
    pool.tasks[-1].signals.result.emit({"ok": True, "run": run})
    assert not hasattr(window, 'samples') and window.sample() == run['sample']
    assert window.previews["silhouette"].path == Path(run["artifacts"]["silhouette"])
    assert "structure_safe" not in window.previews
    assert window.alpha_canvas.source == Path(run["artifacts"]["alpha"])
    for box in window.visual_checks.values(): box.setCurrentText("pass")
    assert window.source_accept.isEnabled()
    window._checkpoint_session = lambda **kw: None
    window._session_checkpoint_failed = False
    window.save_revision("appearance")
    request = pool.tasks[-1]
    assert request.path == "/api/source-runs/child/adjust"
    assert request.body["appearance_revision"]["params"]["global"]["exposure"] == .25
    assert request.body["appearance_revision"]["params"]["global"]["saturation"] == .1
    assert set(request.body) == {"appearance_revision", "manual_layers"}
    assert window.render_pending and not window.source_accept.isEnabled()
    assert "正在高清处理" in window.operation_state.text()
    request.signals.error.emit("render failed")
    assert not window.render_pending and window.source_accept.isEnabled()
    window.save_revision("appearance")
    request = pool.tasks[-1]
    assert "alignment_revision" not in request.body
    request.signals.result.emit({"ok": True, "run": {**run, "run_id": "child-r3", "revision": 3}})
    assert not window.render_pending and window.source_run_id == "child-r3"
    assert "revision 3" in window.review_title.text()
    assert all(box.currentText() == "not_checked" for box in window.visual_checks.values())
    window.preset.setCurrentIndex(window.preset.findData("night_scene"))
    assert window.preset.currentData() == "night_scene"
    style_prompt = window.preset_rows["night_scene"]["prompt"]
    assert style_prompt in window.prompt.toPlainText() and DEFAULT_PROMPT in window.prompt.toPlainText()
    window.restore_prompt.click()
    assert window.prompt.toPlainText() == DEFAULT_PROMPT
    window.apply_style_prompt.click()
    assert style_prompt in window.prompt.toPlainText()
    edited_prompt = window.prompt.toPlainText() + "\nAdd a subtle purple accent on the background architecture."
    window.prompt.setPlainText(edited_prompt)
    window.appearance_sliders["exposure"].setValue(30)
    window.save_revision("appearance")
    regional_request = pool.tasks[-1]
    assert regional_request.body["appearance_revision"]["params"]["global"]["exposure"] == .3
    regional_request.signals.error.emit("render failed")
    window.source_edit.setText("C:/fixture/source.arw")
    window.character_name.setText("Original night heroine"); window.character_notes.setText("Moonlit costume edition"); window.variant_count.setValue(2)
    before_local = len(pool.tasks)
    window.local_source_button.click()
    local_request = pool.tasks[-1]
    assert len(pool.tasks) == before_local + 1 and local_request.path == '/api/tasks/plan'
    assert local_request.body['source'] == 'C:/fixture/source.arw'
    assert local_request.body['creative_task']['mode'] == 'local_portrait'
    assert local_request.body['creative_task']['services']['text_vision'] is False and not window.render_pending
    local_request = start_checked_plan(local_request)
    assert local_request.label == 'local-only' and window.render_pending and not window.local_source_button.isEnabled()
    local_request.signals.error.emit("local runtime not installed")
    assert len(pool.tasks) == before_local + 2 and not window.render_pending and window.local_source_button.isEnabled()
    assert "未触发生图" in window.operation_state.text()
    window.run_source_button.click()
    request = pool.tasks[-1]
    assert request.body["algorithm_profile"] == "dinov3_vitl16"
    assert request.path == '/api/tasks/plan' and request.body["prompt"] == edited_prompt
    assert request.body["effective_prompt"] == edited_prompt
    assert request.body["preset"] == "night_scene" and request.body['creative_task']['mode'] == 'create_reference'
    assert request.body["character_name"] == "Original night heroine" and request.body["character_notes"] == "Moonlit costume edition" and request.body["variant_count"] == 2
    shared_keys = ('provider', 'model', 'preset', 'character_name', 'character_notes', 'variant_count', 'prompt', 'input_prompt', 'creative_task')
    shared_options = {key: request.body[key] for key in shared_keys}
    request = start_checked_plan(request)
    request.signals.error.emit("no backend")
    window.batch_sources = ["C:/fixture/source.arw", "C:/fixture/source2.arw"]
    window.batch_button.click()
    request = pool.tasks[-1]
    assert request.path == '/api/tasks/batch/plan'
    assert {key: request.body[key] for key in shared_keys} == shared_options
    assert not {'appearance_revision', 'manual_layers', 'generation_design', 'character_profile'} & request.body.keys()
    request = start_checked_plan(request); request.signals.error.emit('no backend')
    window.source_edit.setText("C:/fixture/unrelated.jpg")
    window.gen.click()
    request = pool.tasks[-1]
    assert request.path == '/api/tasks/plan' and request.body["source"] == "C:/fixture/source.arw"
    assert request.body["parent_run_id"] == "child-r3" and request.body["prompt"] == edited_prompt
    request = start_checked_plan(request); request.signals.error.emit("no backend")
    window.source_retry.click()
    request = pool.tasks[-1]
    assert request.path == '/api/tasks/plan' and request.body["prompt"] == edited_prompt
    assert request.body['parent_run_id'] == 'child-r3' and request.body['creative_task']['mode'] == 'create_reference'
    assert 'appearance_revision' not in request.body and 'manual_layers' not in request.body
    request = start_checked_plan(request)
    request.signals.result.emit({"ok": True, "async": True, "job_id": "retry-job"})
    assert window.active_job_id == "retry-job" and window.source_run_id == "child-r3" and window.render_pending
    assert any(getattr(task, "path", None) == "/api/jobs/retry-job" for task in pool.tasks)
    window.result("job-status", {"job": {"job_id": "retry-job", "status": "failed"}})
    assert not window.render_pending
    window.provider.addItem("ccswitch_sub2api")
    window.result("provider-settings", {"default_provider": "ccswitch_sub2api", "providers": {"ccswitch_sub2api": {"endpoint": "http://gateway/v1", "model": "gpt-image-2.5-sunburst", "credential_source": "ccswitch"}}})
    window.result("providers", {"providers": [{"id": "ccswitch_sub2api", "status": "configured", "endpoint": "http://gateway/v1"}, {"id": "comfyui", "model": None, "status": "missing_workflow"}]})
    assert window.provider.currentText() == "ccswitch_sub2api" and window.provider_credential.currentText() == "ccswitch"
    window.provider_endpoint.setText("http://new/v1")
    window.save_provider_settings()
    request = pool.tasks[-1]
    assert request.body["providers"]["ccswitch_sub2api"]["endpoint"] == "http://new/v1"
    window.export_directory.setText("C:/fixture/export")
    window.export_format.setCurrentText("TIFF")
    window.export_source_run()
    request = pool.tasks[-1]
    assert request.path == "/api/source-runs/child-r3/export" and request.body["format"] == "TIFF" and request.body["artifact"] == "draft"
    window.result("source-export", {"path": "C:/fixture/export/image.tiff", "metadata": {"width": 4688, "height": 7028, "actual_bit_depth": 8, "icc_profile_embedded": False, "color_space": "sRGB (pipeline declaration)", "format": "TIFF", "status": "needs_visual_review"}})
    assert "4688" in window.export_state.text() and "8-bit" in window.export_state.text()
    with tempfile.TemporaryDirectory(prefix="cosplay_qt_variants_") as folder:
        paths = [Path(folder) / "v1.png", Path(folder) / "v2.png"]
        Image.new("RGB", (31, 47), "navy").save(paths[0]); Image.new("RGB", (31, 47), "purple").save(paths[1])
        variants = [{"variant_id": f"v{index}", "index": index, "style_id": "night_scene", "style_name": "电影夜景",
                     "candidate_path": str(path), "provenance_path": str(path.with_suffix(".json"))} for index, path in enumerate(paths, 1)]
        variant_run = {**run, "run_id": "child-r3", "variants": variants, "selected_variant_id": "v1", "character_profile": profile,
                       "options": {**run["options"], "character_name": "Original night heroine", "character_notes": "Moonlit costume edition", "variant_count": 2}}
        window.load_source_run(variant_run)
        assert window.variant_list.count() == 2 and not window.variant_list.item(0).icon().isNull()
        assert window.variant_preview.path == paths[0] and window.variant_count.value() == 2
        before = len(pool.tasks)
        window.variant_list.setCurrentRow(1)
        assert window.variant_preview.path == paths[1] and len(pool.tasks) == before
        window.variant_use.click()
        request = pool.tasks[-1]
        assert request.path == "/api/source-runs/child-r3/select-variant" and request.body == {"variant_id": "v2"}
        assert window.render_pending and not window.variant_use.isEnabled()
        request.signals.result.emit({"ok": True, "run": {**variant_run, "run_id": "variant-child", "selected_variant_id": "v2"}})
        assert window.source_run_id == "variant-child" and not window.render_pending
        assert window.variant_list.currentItem().data(qt.Qt.UserRole)["variant_id"] == "v2"
        assert window.variant_preview.path == paths[1] and all(box.currentText() == "not_checked" for box in window.visual_checks.values())
    template_prompt = DEFAULT_PROMPT + "\nUse a muted violet evening palette."
    window.result("templates", {"templates": [{"name": "prompt-roundtrip", "parameters": {
        "preset": "night_scene", "prompt": template_prompt,
        "appearance_revision": {"params": {"global": {"exposure": .12}}}, "manual_layers": {}}}]})
    window.template_selector.setCurrentIndex(1)
    next(button for button in window.findChildren(qt.QPushButton) if button.text() == "套用模板").click()
    assert window.prompt.toPlainText() == template_prompt and window.preset.currentData() == "night_scene"
    window.source_edit.setText("C:/fixture/source.arw")
    window.character_name.setText("Original night heroine"); window.character_notes.setText("Blue costume")
    input_prompt = window.prompt.toPlainText()
    window.prepare_design_button.click()
    request = pool.tasks[-1]
    assert request.path == "/api/generation/prepare" and request.body["input_prompt"] == input_prompt
    assert request.body['creative_task']['services']['text_vision'] is True
    assert request.body["source"] == "C:/fixture/source.arw" and request.body["preset"] == "night_scene"
    assert not window.prepare_design_button.isEnabled()
    researched_profile = {**profile, "character_name": "Original night heroine", "identification": "user_supplied", "confidence": "high"}
    design = {"selected_style_id": "night_scene", "input_prompt": input_prompt, "effective_prompt": input_prompt + "\nIMAGE-SPECIFIC ART DIRECTION\nPlace the cool key light at the upper left.",
              "character_research": {"status": "researched", "summary": "角色的冷色造型适合克制的夜景灯光。",
                                     "sources": [{"title": "Reference fixture", "url": "https://example.org/characters/heroine"}], "provenance": {"search_used": True}},
              "design_brief": {"summary": "本图以冷色环境突出脸部和织物。", "scene_analysis": "原图背景偏暗。", "reference_application": "借鉴角色配色，保留本图的服装版本。"}}
    request.signals.result.emit({"ok": True, "character_profile": researched_profile, "generation_design": design})
    assert window.prepare_design_button.isEnabled() and window.prompt.toPlainText() == input_prompt
    assert window.final_prompt.toPlainText() == design["effective_prompt"] and "本图以冷色" in window.design_summary.toPlainText()
    window.final_prompt_toggle.setChecked(True)
    assert window.final_prompt.verticalScrollBar().value() == 0
    window.final_prompt_toggle.setChecked(False)
    assert window.research_sources.openExternalLinks() and 'href="https://example.org/characters/heroine"' in window.research_sources.text()
    assert "user_supplied" not in window.character_result.text() and "用户提供角色名" in window.character_result.text() and "可信度：高" in window.character_result.text()
    edited_final = design["effective_prompt"] + "\nKeep warm reflections localized."
    window.final_prompt.setPlainText(edited_final)
    window.run_source_button.click()
    request = pool.tasks[-1]
    assert request.body["input_prompt"] == input_prompt and request.body["effective_prompt"] == edited_final
    assert request.body["generation_design"]["final_prompt_edited"] and request.body["generation_design"]["generated_effective_prompt"] == design["effective_prompt"]
    assert request.body["character_profile"]["character_name"] == "Original night heroine"
    request = start_checked_plan(request); request.signals.error.emit("no backend")
    window.batch_button.click()
    request = pool.tasks[-1]
    assert "generation_design" not in request.body and "character_profile" not in request.body
    assert request.body["input_prompt"] == input_prompt and 'effective_prompt' not in request.body
    request = start_checked_plan(request); request.signals.error.emit('no backend')
    # Changes to any design input drop prepared data. Final English edits are
    # deliberately separate from this editable photographic base direction.
    mutations = [lambda: window.source_edit.setText("C:/fixture/another.arw"),
                 lambda: window.character_name.setText("Another heroine"),
                 lambda: window.character_notes.setText("Another costume"),
                 lambda: window.preset.setCurrentIndex(0),
                 lambda: window.prompt.setPlainText(input_prompt + "\nChange the palette.")]
    for mutate in mutations:
        window.show_generation_design(design, researched_profile)
        mutate()
        assert window.prepared_design is None and "generation_design" not in window.generation_request_options()
    window.prepare_design_button.click()
    request = pool.tasks[-1]
    window.character_notes.setText("Changed during research")
    request.signals.result.emit({"ok": True, "character_profile": researched_profile, "generation_design": design})
    assert window.prepared_design is None and window.prepare_design_button.isEnabled()
    restored = {**run, "run_id": "prepared-run", "generation_design": design, "character_profile": researched_profile,
                "options": {**run["options"], "character_name": "Original night heroine", "character_notes": "Blue costume", "preset": "night_scene",
                            "effective_prompt": design["effective_prompt"], "input_prompt": input_prompt}}
    window.load_source_run(restored)
    assert window.prompt.toPlainText() == input_prompt and window.final_prompt.toPlainText() == design["effective_prompt"]
    assert window.generation_request_options()["generation_design"]["input_prompt"] == input_prompt
    unavailable = {**design, "character_research": {"status": "unavailable", "summary": "暂时无法获取联网资料。", "provenance": {"search_used": False}}}
    window.show_generation_design(unavailable, researched_profile)
    assert "已联网研究" not in window.research_summary.text() and "联网研究未完成" in window.research_summary.text()
    assert "电影夜景" in window.design_style.text() or "自动" in window.design_style.text()
    with tempfile.TemporaryDirectory(prefix="cosplay_source_preview_") as folder:
        paths = [Path(folder) / "source.png", Path(folder) / "another.png"]
        for path in paths: Image.new("RGB", (31, 47), "navy").save(path)
        original_dialog = qt.QFileDialog.getOpenFileName
        try:
            qt.QFileDialog.getOpenFileName = lambda *args, **kwargs: (str(paths[0]), "")
            window.choose_source_button.click()
        finally:
            qt.QFileDialog.getOpenFileName = original_dialog
        preview_task = pool.tasks[-1]
        assert isinstance(preview_task, qt.SourcePreviewTask) and window.source_edit.text() == str(paths[0])
        preview_task.signals.result.emit({"source": str(paths[0]), "proxy_path": str(paths[0]), "width": 31, "height": 47})
        assert window.project_preview.path == paths[0] and window.project_preview.proxy_path == paths[0]
        assert window.canvas_title.text() == "source.png" and str(paths[0]) == window.canvas_title.toolTip()
        window.request_source_preview(paths[1])
        preview_task.signals.result.emit({"source": str(paths[0]), "proxy_path": str(paths[0]), "width": 31, "height": 47})
        assert window.project_images["source"] == paths[1] and "source" not in window.project_proxies
        window.project_proxies["reference"] = paths[0]
        window.project_view_selector.setCurrentIndex(1)
        window.load_source_run({**run, "source": {"path": str(paths[0]), "oriented_size": [31, 47]},
                                "artifacts": {"reference_candidate": str(paths[1]), "auto_draft": str(paths[0])}})
        assert "reference" not in window.project_proxies and window.project_preview.proxy_path is None
        candidate_footer = window.project_image_info.text()
        assert "another.png" in candidate_footer and "31 × 47" in candidate_footer and "GPT 候选" in candidate_footer
        window.result("source-preview", {"source": str(paths[0]), "proxy_path": str(paths[0]), "width": 600, "height": 900})
        assert window.project_image_info.text() == candidate_footer
        window.project_view_selector.setCurrentIndex(0)
        assert "source.png" in window.project_image_info.text() and "600 × 900" in window.project_image_info.text()
        window.project_view_selector.setCurrentIndex(2)
        assert "source.png" in window.project_image_info.text() and "31 × 47" in window.project_image_info.text() and "高清草稿" in window.project_image_info.text()
    with tempfile.TemporaryDirectory(prefix='cosplay-qt-source-only-') as folder:
        import hashlib
        import numpy as np
        from native_source_baseline import prepare_source_only
        from raw_support import RGB_DECODE_VERSION
        source=Path(folder)/'source.png';Image.new('RGB',(31,47),'navy').save(source)
        local_run={'kind':'single_input_run','input_mode':'single_source','run_id':'source-only-current',
            'sample':'local','revision':1,'source':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
            'artifacts':{'auto_draft':str(source)},'options':{'render_mode':'native_source_design'}}
        _,local=prepare_source_only(np.full((47,31,3),.2,np.float32),local_run,
            {'decode_version':RGB_DECODE_VERSION,'source_precision':'raster8_srgb'})
        local_run['options']['source_only_plan']=local
        window.load_source_run(local_run)
        assert not window._paired_subject_state()['reconstruction_required']
        window.add_source_light();window.source_light_timer.stop();window.update_source_light_actions()
        assert window.source_light_preview.isEnabled() and window.source_light_save.isEnabled()
        window.source_run_only_plan['source_sha256']='wrong-source'
        assert not window._allow_current_render()
    window.close()
    temporary_workspace.cleanup()
    print(json.dumps({"ok": True, "provider_refresh_preserves_input": True,
                      "refresh_dispatches_recent": True, "cleanup_button_dispatch": True,
                      "cleanup_cancel_preserved": True, "provider_discovery_dispatch": True,
                      "recent_project_button_dispatch": True,
                      "homepage_default_english_prompt": True, "preset_prompt_editable": True,
                      "single_batch_generation_options_shared": True, "source_run_generation_is_not_legacy": True,
                      "editable_prompt_async_retry": True,
                      "restored_run_and_template_prompts_preserved": True,
                      "optional_character_auto_recognition": True, "recommended_style_application": True,
                      "character_and_variant_count_shared": True, "thumbnail_variant_browser": True,
                      "variant_selection_creates_review_child": True,
                      "research_prepare_request": True, "research_sources_clickable": True,
                      "editable_final_prompt_is_separate": True, "prepared_design_inputs_invalidate": True,
                      "batch_designs_not_shared": True, "saved_design_restores_input_and_final": True,
                      "research_unavailable_reported_honestly": True,
                      "modern_navigation_and_collapsed_logs": True, "import_queues_proxy_worker": True,
                      "source_preview_ignores_stale_decode": True,
                      "stale_job_callbacks_ignored": True, "missing_checks_loads": True,
                      "revision_parameters_restored": True, "copy_action": True,
                      "source_only_edit_requests": True, "render_pending_and_failure_restore": True,
                      "retired_alignment_and_transfer_controls_absent": True, "preset_prompt_switch": True, "persistent_provider_selection": True, "format_export": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
