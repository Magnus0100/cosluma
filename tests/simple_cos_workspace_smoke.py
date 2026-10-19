"""Ordinary-user desktop states and host-routed actions without backend or models."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QMimeData, QUrl, QPoint, QPointF, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel
from PIL import Image

import simple_cos_workspace as simple
from workbench_theme import apply_theme


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([]); apply_theme(app)
    with tempfile.TemporaryDirectory(prefix='cosplay-simple-ui-') as temporary:
        root = Path(temporary)
        source, result, raw = root/'原片.png', root/'当前效果.png', root/'原片.ARW'
        # Distinct prepared raster previews make source/result replacement observable.
        Image.new('RGB', (900, 1350), '#476d8a').save(source)
        Image.new('RGB', (900, 1350), '#b98b84').save(result)
        raw.write_bytes(b'not a raster preview')
        window = simple.SimpleCosWorkspace(); window.resize(1280, 820)
        actions, chosen, paths, strengths = [], [], [], []
        window.actionRequested.connect(actions.append)
        window.sourceChosen.connect(lambda: chosen.append(True))
        window.sourcePathChanged.connect(paths.append)
        window.strengthChanged.connect(strengths.append)
        window.show(); app.processEvents()
        assert window.source_button.isEnabled() and window.professional_button.isEnabled()
        assert not any(button.isEnabled() for button in (window.beauty_button, window.create_cos_button,
            window.compare_button, window.save_button, window.export_button))
        assert window.post_button.isHidden()
        assert window.strength_group.isHidden() and window.progress.isHidden()
        window.source_button.click(); assert chosen == [True] and not actions
        window.set_state(str(raw), False, False, False, False)
        assert window.beauty_button.isEnabled() and window.create_cos_button.isEnabled()
        assert window.existing_reference_button.isEnabled()
        assert not window.save_button.isEnabled() and not window.compare_button.isEnabled()
        assert window.source_label.text() == raw.name
        window.beauty_button.click(); window.create_cos_button.click(); window.existing_reference_button.click()
        assert actions == ['beauty', 'create_cos', 'existing_reference'], 'Production actions must be routed to the host'
        assert not strengths and paths == []

        window.set_images(str(source), str(result)); QTest.qWait(120)
        assert not window.source_image.image.isNull() and not window.result_image.image.isNull()
        assert window.source_image.image.pixelColor(0, 0).name() == '#476d8a'
        assert window.result_image.image.pixelColor(0, 0).name() == '#b98b84'
        assert window.source_image.image.width() <= window.source_image.width() * window.devicePixelRatioF()
        assert window.source_image.image.height() <= window.source_image.height() * window.devicePixelRatioF()
        window.set_state(str(raw), True, False, True, True)
        assert window.strength_group.isVisible() and window.strength_slider.isEnabled()
        assert window.strength_slider.value() == 100 and window.strength_label.text() == '100%'
        original_actions = list(actions)
        window.set_strength(57); window.set_compare(True)
        assert window.strength_slider.value() == 57 and window.compare_button.isChecked()
        assert not strengths and actions == original_actions, 'Host synchronization must not launch actions'
        window.strength_slider.setValue(43); assert strengths == [43]
        window.compare_button.click(); window.save_button.click(); window.export_button.click(); window.professional_button.click()
        assert actions[-4:] == ['compare', 'save', 'export', 'professional']
        window.set_post_continuation(True, dirty=True); assert not window.post_button.isHidden() and window.post_button.isEnabled()
        window.post_button.click(); assert actions[-1] == 'continue_post'
        window.set_post_continuation(False)
        window.set_compare(False)
        assert not window.compare_button.isChecked() and window.compare_button.text() == '对比原片'
        window.set_message('当前效果已更新。'); assert window.state_label.text() == '当前效果已更新。'

        window.set_state(str(raw), True, True, True, True)
        assert window.progress.isVisible() and not window.source_button.isEnabled()
        window.set_job_progress(.70, 'rendering', '正在渲染原尺寸成片。')
        assert window.progress.minimum() == 0 and window.progress.maximum() == 100 and window.progress.value() == 70
        assert not any(button.isEnabled() for button in (window.beauty_button, window.create_cos_button,
            window.compare_button, window.save_button, window.export_button))
        assert not window.strength_slider.isEnabled() and window.professional_button.isEnabled()
        original_actions = list(actions)
        window.beauty_button.click(); window.save_button.click()
        window.strength_slider.setValue(7)
        assert actions == original_actions and strengths == [43]
        window.set_candidate_pending(True, 4)
        assert window.candidate_panel.isVisible() and window.candidate_button.isEnabled()
        window.set_state(str(raw), True, False, True, True, progress_active=True)
        window.set_progress_stages(['生成候选', '选择候选', '制作后期', '保存与审片'])
        window.set_job_progress(.40, 'awaiting_reference', '候选设计已生成；请选择一张后继续。')
        assert window.progress.isVisible() and window.progress.value() == 40
        assert '任务里程碑 40%' in window.progress_detail.text() and '本次路线 4 项' in window.progress_detail.text()
        assert '本步骤' not in window.progress_detail.text()
        window.candidate_button.click(); assert actions[-1] == 'select_candidate'
        window.set_candidate_pending(False)
        window.set_state(str(raw), True, True, True, True)

        mime = QMimeData(); mime.setUrls([QUrl.fromLocalFile(str(raw))])
        busy_drop = QDropEvent(QPointF(20, 20), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        window.dropEvent(busy_drop); assert not busy_drop.isAccepted() and not paths
        window.set_state(str(raw), True, False, True, True)
        enter = QDragEnterEvent(QPoint(20, 20), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        window.dragEnterEvent(enter); assert enter.isAccepted()
        drop = QDropEvent(QPointF(20, 20), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        window.dropEvent(drop); assert drop.isAccepted() and paths == [str(raw.resolve())]
        unsupported = root/'notes.txt'; unsupported.write_text('not an image')
        mime.setUrls([QUrl.fromLocalFile(str(unsupported))])
        drop = QDropEvent(QPointF(20, 20), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        window.dropEvent(drop); assert not drop.isAccepted() and len(paths) == 1
        mime.setUrls([QUrl.fromLocalFile(str(source)), QUrl.fromLocalFile(str(result))])
        drop = QDropEvent(QPointF(20, 20), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        window.dropEvent(drop); assert not drop.isAccepted() and len(paths) == 1

        original_reader = simple.QImageReader
        try:
            def no_raw_decoder(path):
                assert Path(path).suffix.lower() not in simple._RAW, 'Ordinary UI attempted to decode a source RAW'
                return original_reader(path)
            simple.QImageReader = no_raw_decoder
            window.set_images(str(raw), str(result))
            assert window.source_image.image.isNull() and window.source_image.empty_text == '原片预览准备中…'
            window.set_images(str(root/'missing.png'), str(result))
            assert window.source_image.image.isNull() and window.source_image.empty_text == '预览暂时无法显示'
        finally:
            simple.QImageReader = original_reader
        # A preview rewritten at the same path must appear on the next host refresh.
        Image.new('RGB', (600, 900), '#688985').save(result)
        window.set_images(str(source), str(result)); QTest.qWait(120)
        assert window.result_image.image.pixelColor(0, 0).name() == '#688985'
        window.set_strength(100)
        window.set_state(str(raw), True, False, True, True)
        window.resize(980, 680); QTest.qWait(120)
        assert window.source_card.isVisible() and window.result_card.isVisible()
        assert window.source_image.width() >= 180 and window.result_image.width() >= 180
        # User-facing copy is limited to photography actions, without backend parameter names.
        labels = ' '.join(widget.text() for widget in window.findChildren(QLabel))
        assert not any(term in labels for term in ('LoRA', 'CUDA', 'SAM3', '阈值', 'display_linear', 'GRPO'))
        if args.snapshot:
            window.resize(1280, 820); QTest.qWait(120)
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            assert window.grab().save(str(args.snapshot))
        window.set_images(None, None); window.set_state('', False, False, False, False)
        assert window.source_image.image.isNull() and window.result_image.image.isNull()
        assert window.strength_group.isHidden() and window.compare_button.isEnabled() is False
        window.close(); app.processEvents()
    print(json.dumps({'ok': True, 'host_routed_actions': True, 'empty_source_result_busy_states': True,
        'source_only_start_supported': True, 'host_sync_has_no_action_feedback': True,
        'raw_display_uses_prepared_proxy_only': True, 'aspect_fit_and_refreshed_preview': True,
        'local_single_photo_drop': True, 'no_backend_or_model_invocation': True}))


if __name__ == '__main__':
    main()
