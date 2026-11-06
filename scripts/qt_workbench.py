"""Small desktop: select originals, name the character, produce composites."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import json
from urllib.parse import urlsplit

from PySide6.QtCore import QSettings, QThread, Signal, Slot, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QListWidget, QMainWindow, QPushButton,
    QPlainTextEdit, QVBoxLayout, QWidget)

from photo_pipeline import finish_batch
from settings import default_settings_path, editable_settings, load_settings, save_settings
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


class ServiceSettings(QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = Path(path).expanduser().resolve()
        self.values = editable_settings(self.path)
        self.fields = {}
        self.credentials = {}
        self.setWindowTitle('服务设置')
        self.resize(720, 500)
        layout = QVBoxLayout(self)
        description = QLabel('填写服务提供方给出的地址和模型名。保存只写入本机配置，不连接模型服务。\n'
            '密钥使用环境变量或已有密钥文件，配置里只保存名称或路径。')
        description.setWordWrap(True)
        layout.addWidget(description)
        destination = QLabel('配置保存到：' + str(self.path))
        destination.setWordWrap(True)
        destination.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(destination)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        library = QWidget()
        form = QFormLayout(library)
        self.library = QLineEdit(self.values.get('prompt_library', str(self.path.parent / 'prompts.json')))
        row = QHBoxLayout()
        row.addWidget(self.library, 1)
        browse = QPushButton('选择已有库')
        browse.clicked.connect(self.choose_library)
        row.addWidget(browse)
        form.addRow('提示词库 JSON', row)
        hint = QLabel('首次保存时，会在默认位置创建随附的通用提示词库；可换成你自己的库。\n'
            '每条包含 title、text、source，已有文件不会被覆盖。')
        hint.setWordWrap(True)
        form.addRow(hint)
        self.tabs.addTab(library, '提示词库')
        for service, title in (('llm', '提示词服务'), ('image', '图像编辑服务')):
            panel = QWidget()
            form = QFormLayout(panel)
            self.add_field(form, service, 'url', '完整接口地址')
            self.add_field(form, service, 'model', '模型名')
            provider = self.values.get(service, {})
            mode = QComboBox()
            for text, key in (('环境变量名', 'api_key_env'), ('密钥文件路径', 'api_key_file'), ('无需密钥', '')):
                mode.addItem(text, key)
            key = 'api_key_file' if 'api_key_file' in provider else ('api_key_env' if provider.get('api_key_env') else '')
            mode.setCurrentIndex(mode.findData(key))
            value = QLineEdit(provider.get(key, '') if key else '')
            value.setPlaceholderText('填写变量名或文件路径，请勿粘贴密钥')
            value.setEnabled(bool(key))
            mode.currentIndexChanged.connect(lambda _index, mode=mode, value=value: value.setEnabled(bool(mode.currentData())))
            self.credentials[service] = (mode, value)
            form.addRow('凭据来源', mode)
            form.addRow('名称或路径', value)
            if service == 'image':
                self.parameters = QPlainTextEdit(json.dumps(provider.get('parameters', {}), ensure_ascii=False, indent=2))
                self.parameters.setMaximumHeight(120)
                form.addRow('服务参数 JSON', self.parameters)
            self.tabs.addTab(panel, title)
        panel = QWidget()
        form = QFormLayout(panel)
        for service, title in (('head', '整头分割'), ('sr', '超分')):
            self.add_field(form, service, 'url', title + '服务地址')
            self.add_field(form, service, 'model', title + '权重名')
        hint = QLabel('使用已准备好的 ComfyUI 服务。分割权重位于 models/checkpoints，'
            '超分权重位于 models/upscale_models；安装要求见随包 README。')
        hint.setWordWrap(True)
        form.addRow(hint)
        self.tabs.addTab(panel, '分割与超分')
        self.message = QLabel('')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QDialogButtonBox()
        buttons.addButton('保存配置', QDialogButtonBox.AcceptRole)
        buttons.addButton('取消', QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_field(self, form, service, key, label):
        edit = QLineEdit(self.values.get(service, {}).get(key, ''))
        self.fields[service, key] = edit
        form.addRow(label, edit)

    def choose_library(self):
        filename, _ = QFileDialog.getOpenFileName(self, '选择提示词库', self.library.text(), 'JSON (*.json)')
        if filename:
            self.library.setText(filename)

    def save(self):
        try:
            library = self.library.text().strip()
            if not library:
                raise ValueError('请填写提示词库路径')
            for (service, key), edit in self.fields.items():
                value = edit.text().strip()
                if not value:
                    raise ValueError('请填写服务地址和模型名')
                if key == 'url':
                    parsed = urlsplit(value)
                    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.hostname == 'api.example.com':
                        raise ValueError('请填写服务提供方的完整 HTTP/HTTPS 地址，替换示例地址')
                self.values.setdefault(service, {})[key] = value
            for service, (mode, value) in self.credentials.items():
                key = mode.currentData()
                if key and not value.text().strip():
                    raise ValueError('请填写凭据的环境变量名或密钥文件路径')
                provider = self.values[service]
                provider.pop('api_key_env', None)
                provider.pop('api_key_file', None)
                if key:
                    provider[key] = value.text().strip()
            parameters = json.loads(self.parameters.toPlainText())
            if not isinstance(parameters, dict):
                raise ValueError('图像服务参数必须是 JSON 对象')
            self.values['image']['parameters'] = parameters
            self.values['prompt_library'] = library
            save_settings(self.path, self.values)
            self.accept()
        except Exception as error:
            self.message.setText('无法保存：' + str(error))


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
        setup_row = QHBoxLayout()
        self.setup_hint = QLabel('')
        self.setup_hint.setWordWrap(True)
        setup_row.addWidget(self.setup_hint, 1)
        self.setup_button = QPushButton('服务设置')
        self.setup_button.clicked.connect(self.configure_services)
        setup_row.addWidget(self.setup_button)
        layout.addLayout(setup_row)
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
        self.config.textChanged.connect(self.update_setup_hint)
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
        self.update_setup_hint()

    def update_setup_hint(self):
        exists = Path(self.config.text()).expanduser().is_file()
        self.setup_hint.setText('服务配置已选择；需要更换服务时打开“服务设置”。' if exists else
            '首次使用：请打开“服务设置”，填写模型服务后保存；通用提示词库随包提供。')

    def configure_services(self):
        try:
            dialog = ServiceSettings(self.config.text(), self)
            if dialog.exec() == QDialog.Accepted:
                self.config.setText(str(dialog.path))
                self.remember_preferences()
                self.update_setup_hint()
                self.status.setText('配置已保存。选择照片、填写角色后即可开始；连接错误会显示在对应制作步骤。')
        except Exception as error:
            self.status.setText('无法打开服务设置：' + str(error))

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
            if not Path(self.config.text()).expanduser().is_file():
                self.configure_services()
                if not Path(self.config.text()).expanduser().is_file():
                    return
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
            self.setup_button.setEnabled(False)
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
        self.setup_button.setEnabled(True)
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
