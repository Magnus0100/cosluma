"""Small desktop: select originals, name the character, produce composites."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from PySide6.QtCore import QSettings, QThread, Signal, Slot, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMainWindow, QPushButton,
    QPlainTextEdit, QVBoxLayout, QWidget)

from photo_pipeline import finish_batch
from settings import default_settings_path, load_settings
from app_version import VERSION


class Worker(QThread):
    progress = Signal(str)
    saved = Signal(dict)
    failed = Signal(str)

    def __init__(self, sources, character, brief, root, config):
        super().__init__()
        self.sources, self.character, self.brief = sources, character, brief
        self.root, self.config = root, config
        self.succeeded = 0
        self.errors = []
        self.batch_error = ''
        self.output_directory = None

    def run(self):
        try:
            for result in finish_batch(self.sources, self.character, self.root, self.config,
                    brief=self.brief, progress=self.progress.emit):
                self.output_directory = result.get('batch_directory',self.output_directory)
                if result['status'] == 'complete':
                    self.succeeded += 1
                    self.saved.emit(result)
                else:
                    message = f'{Path(result["source"]).name}：{result["error"]}'
                    self.errors.append(message)
                    self.failed.emit(message)
        except Exception as error:
            self.batch_error = str(error)
            self.failed.emit(self.batch_error)


class Workbench(QMainWindow):
    def __init__(self, root=None, config=None):
        super().__init__()
        self.preferences = QSettings(str(Path.home()/'CosplayWorkbench/config/desktop.ini'),QSettings.IniFormat)
        root = root if root is not None else self.preferences.value('output_root',str(Path.home()/'CosLuma/output'),type=str)
        config = config or os.environ.get('COSLUMA_CONFIG') or self.preferences.value('config_path',str(default_settings_path()),type=str)
        self.setWindowTitle('CosLuma '+VERSION)
        self.resize(1000, 760)
        self.worker = None
        self.close_after_finish = False
        self.current_output = None
        self.preview_pixmap = None
        self.batch_directory = None
        self.sources = []
        panel = QWidget()
        self.setCentralWidget(panel)
        layout = QVBoxLayout(panel)
        title = QLabel('选照片、填角色，自动设计整组成片')
        title.setStyleSheet('font-size: 22px; font-weight: 600; margin-bottom: 8px')
        layout.addWidget(title)
        row = QHBoxLayout()
        select = QPushButton('选择照片')
        select.clicked.connect(self.select_sources)
        row.addWidget(select)
        self.count = QLabel('尚未选择照片')
        row.addWidget(self.count, 1)
        layout.addLayout(row)
        self.listing = QListWidget()
        self.listing.setMaximumHeight(110)
        layout.addWidget(self.listing)
        form = QFormLayout()
        self.character = QLineEdit(self.preferences.value('character','',type=str))
        self.character.setPlaceholderText('例如：鸣潮 · 守岸人')
        form.addRow('角色', self.character)
        self.brief = QPlainTextEdit()
        self.brief_character = self.character.text().strip()
        if self.brief_character:
            self.brief.setPlainText(self.preferences.value('briefs/'+self.brief_character,'',type=str))
        self.character.editingFinished.connect(self.switch_character)
        self.brief.setPlaceholderText('可以留空；填写后记住当前角色的偏好，自动设计场景、光色和单张特效')
        self.brief.setMaximumHeight(70)
        form.addRow('补充要求', self.brief)
        self.root = QLineEdit(str(root))
        form.addRow('保存目录', self.path_row(self.root, directory=True))
        layout.addLayout(form)
        advanced = QGroupBox('高级设置')
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_layout = QVBoxLayout(advanced)
        advanced_panel = QWidget()
        advanced_form = QFormLayout(advanced_panel)
        self.config = QLineEdit(str(config))
        advanced_form.addRow('配置文件', self.path_row(self.config))
        self.edge = QLineEdit(self.preferences.value('output_edge','',type=str))
        self.edge.setPlaceholderText('留空使用配置，默认长边 6000；以单次超分覆盖范围为准')
        advanced_form.addRow('请求长边像素', self.edge)
        advanced_layout.addWidget(advanced_panel)
        advanced_panel.setVisible(False)
        advanced.toggled.connect(advanced_panel.setVisible)
        layout.addWidget(advanced)
        self.start = QPushButton('开始制作')
        self.start.clicked.connect(self.produce)
        layout.addWidget(self.start)
        self.status = QLabel('补充要求可留空；整组共享场景与光色，单张依据构图处理。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.preview = QLabel('成片将在这里显示')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(220)
        layout.addWidget(self.preview, 1)
        self.output_info = QLabel('')
        self.output_info.setWordWrap(True)
        layout.addWidget(self.output_info)
        open_folder = QPushButton('打开保存目录')
        open_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(
            str(self.batch_directory or (Path(self.current_output).parent if self.current_output else Path(self.root.text()))))))
        layout.addWidget(open_folder)

    def switch_character(self):
        character = self.character.text().strip()
        if character == self.brief_character:
            return
        if self.brief_character:
            self.preferences.setValue('briefs/'+self.brief_character,self.brief.toPlainText())
        self.brief_character = character
        self.brief.setPlainText(self.preferences.value('briefs/'+character,'',type=str) if character else '')

    def remember_preferences(self):
        self.switch_character()
        self.preferences.setValue('character',self.brief_character)
        if self.brief_character:
            self.preferences.setValue('briefs/'+self.brief_character,self.brief.toPlainText())
        for key,value in (('output_root',self.root.text()),('config_path',self.config.text()),('output_edge',self.edge.text())):
            if value.strip() or key=='output_edge':
                self.preferences.setValue(key,value.strip())
        self.preferences.sync()

    def path_row(self, edit, directory=False):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        browse = QPushButton('选择')
        def choose():
            selected = QFileDialog.getExistingDirectory(self, '保存目录', edit.text()) if directory else (
                QFileDialog.getOpenFileName(self, '配置文件', edit.text(), 'JSON (*.json)')[0])
            if selected:
                edit.setText(selected)
        browse.clicked.connect(choose)
        layout.addWidget(browse)
        return widget

    def select_sources(self):
        paths, _ = QFileDialog.getOpenFileNames(self, '选择原片', '',
            '照片 (*.jpg *.jpeg *.png *.tif *.tiff *.arw *.cr2 *.cr3 *.nef *.dng *.raf *.orf *.rw2)')
        if paths:
            self.sources = paths
            self.listing.clear()
            self.listing.addItems([Path(path).name for path in paths])
            self.count.setText(f'{len(paths)} 张照片')

    def produce(self):
        try:
            if not self.sources or not self.character.text().strip():
                raise ValueError('请选择原片并填写角色')
            if not self.root.text().strip():
                raise ValueError('请选择保存目录')
            config = load_settings(self.config.text())
            if self.edge.text().strip():
                config['output_long_edge'] = int(self.edge.text())
            self.remember_preferences()
            self.worker = Worker(list(self.sources), self.character.text().strip(), self.brief.toPlainText(),
                self.root.text(), config)
            self.worker.progress.connect(self.status.setText)
            self.worker.saved.connect(self.show_saved)
            self.worker.failed.connect(self.show_failure)
            self.worker.finished.connect(self.batch_finished)
            self.current_output = None
            self.preview_pixmap = None
            self.batch_directory = None
            self.preview.clear()
            self.preview.setText('正在设计整组场景与光色')
            self.output_info.clear()
            self.start.setEnabled(False)
            self.worker.start()
        except Exception as error:
            self.status.setText('无法开始：' + str(error))

    @Slot(dict)
    def show_saved(self, result):
        path = result.get('exports',{}).get('JPEG',result['files']['jpeg'])
        self.current_output = path
        self.preview_pixmap = QPixmap(path)
        self.batch_directory = result.get('batch_directory')
        self.update_preview()
        width, height = result['output_dimensions']
        message = f'已保存 {width}×{height}；16 位 PNG 与 JPEG 位于：{self.batch_directory}'
        if result.get('output_adjustment'):
            requested = result['requested_dimensions']
            message += f'\n请求 {requested[0]}×{requested[1]}，按实际可用尺寸输出。'
        self.output_info.setText(message)
        self.status.setText(f'已保存 {self.worker.succeeded} / {len(self.worker.sources)} 张，继续制作')

    @Slot(str)
    def show_failure(self, message):
        self.status.setText('制作失败：' + message)

    @Slot()
    def batch_finished(self):
        worker, self.worker = self.worker, None
        if worker.output_directory:
            self.batch_directory = worker.output_directory
        if worker.batch_error:
            summary = f'批次失败：{worker.batch_error}；已保存 {worker.succeeded} 张。'
        elif worker.errors:
            summary = f'批次未全部完成：成功 {worker.succeeded} 张，失败 {len(worker.errors)} 张。'
        else:
            summary = f'批次完成：成功 {worker.succeeded} 张。'
        self.status.setText('\n'.join([summary, *worker.errors]))
        worker.deleteLater()
        self.start.setEnabled(True)
        if self.close_after_finish:
            self.close()
            QApplication.instance().quit()

    def closeEvent(self, event):
        self.remember_preferences()
        if self.worker is not None and self.worker.isRunning():
            self.close_after_finish = True
            QApplication.instance().setQuitOnLastWindowClosed(False)
            self.hide()
            event.ignore()
        else:
            super().closeEvent(event)

    def update_preview(self):
        if self.preview_pixmap is not None:
            self.preview.setPixmap(self.preview_pixmap.scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'preview'):
            self.update_preview()

def main():
    if sys.argv[1:2]==['--cli']:
        del sys.argv[1]
        from project_cli import main as command_line
        return command_line()
    if sys.argv[1:2]==['--server']:
        del sys.argv[1]
        from backend_server import main as server
        return server()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',action='version',version='CosLuma '+VERSION)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    if sys.platform == 'win32':
        font = Path(os.environ['SystemRoot']) / 'Fonts/msyh.ttc'
        font_id = QFontDatabase.addApplicationFont(str(font))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 10))
    window = Workbench(args.root, args.config)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
