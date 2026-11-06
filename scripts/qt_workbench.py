"""Desktop setup, photo queue and previews around the single finishing chain."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit, urlunsplit

from PySide6.QtCore import QSettings, QThread, Signal, Slot, Qt, QUrl, QSize
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QImageReader, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog,
    QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QPushButton, QPlainTextEdit, QProgressBar,
    QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget)

from app_version import VERSION
from image_io import IMAGE_EXTENSIONS
from photo_pipeline import finish_batch
from settings import default_settings_path, editable_settings, load_settings, save_settings


def endpoint_url(value, service):
    """Expand an explicitly entered compatible API base when saving settings."""
    parsed = urlsplit(value.strip())
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.hostname == 'api.example.com':
        raise ValueError('请填写实际服务的 HTTP/HTTPS 地址，替换示例地址')
    if service in ('llm', 'image'):
        path = parsed.path.rstrip('/')
        if path in ('', '/v1'):
            path = path or '/v1'
            path += '/chat/completions' if service == 'llm' else '/images/edits'
            parsed = parsed._replace(path=path)
    else:
        parsed = parsed._replace(path=parsed.path.rstrip('/'))
    return urlunsplit(parsed)


class Worker(QThread):
    progress = Signal(str)
    photo_stage = Signal(str, str)
    result_ready = Signal(dict)
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
                    brief=self.brief, progress=self.progress.emit,
                    photo_progress=self.photo_stage.emit):
                self.output_directory = result.get('batch_directory', self.output_directory)
                if result['status'] == 'complete':
                    self.succeeded += 1
                else:
                    self.errors.append(f'{Path(result["source"]).name}：{result["error"]}')
                self.result_ready.emit(result)
        except Exception as error:
            self.batch_error = str(error)
            self.failed.emit(self.batch_error)


class ServiceSettings(QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = Path(path).expanduser().resolve()
        self.existing = self.path.is_file()
        self.values = editable_settings(self.path)
        self.fields = {}
        self.credentials = {}
        self.setWindowTitle('连接模型服务')
        self.resize(760, 590)
        layout = QVBoxLayout(self)
        title = QLabel('先连接服务，以后只需照片和角色')
        title.setStyleSheet('font-size: 18px; font-weight: 600')
        layout.addWidget(title)
        hint = QLabel('需要图文提示词服务、图像编辑服务，以及已准备好分割和超分的 ComfyUI。\n'
            '填写服务方给你的地址、模型名和密钥。密钥存于本机独立文本文件，请勿共享配置目录。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        for service, title in (('llm', '1  提示词服务'), ('image', '2  图像编辑')):
            panel = QWidget()
            form = QFormLayout(panel)
            form.setVerticalSpacing(12)
            self.add_field(form, service, 'url', '服务地址', '例如 https://服务地址/v1，或完整接口地址')
            self.add_field(form, service, 'model', '模型名称', '填写服务方实际提供的模型名')
            if not self.existing:
                self.fields[service, 'url'].clear()
                self.fields[service, 'model'].clear()
            provider = self.values.get(service, {})
            mode = QComboBox()
            mode.addItem('粘贴密钥，保存在本机', 'inline')
            if self.existing and (provider.get('api_key_file') or provider.get('api_key_env')):
                mode.addItem('保留当前凭据', 'keep')
            mode.addItem('从环境变量读取', 'api_key_env')
            mode.addItem('选择已有密钥文件', 'api_key_file')
            mode.addItem('此服务无需密钥', '')
            value = QLineEdit()
            browse = QPushButton('选择文件')
            browse.clicked.connect(lambda _checked=False, value=value: self.choose_file(value, '选择密钥文件', '所有文件 (*)'))
            row = QHBoxLayout()
            row.addWidget(value, 1)
            row.addWidget(browse)
            def change_mode(_index, mode=mode, value=value, browse=browse, provider=provider):
                key = mode.currentData()
                value.clear()
                value.setEchoMode(QLineEdit.Password if key == 'inline' else QLineEdit.Normal)
                value.setEnabled(key not in ('', 'keep'))
                browse.setVisible(key == 'api_key_file')
                if key == 'inline':
                    value.setPlaceholderText('粘贴服务密钥；不会写入配置 JSON 或运行记录')
                elif key == 'keep':
                    value.setPlaceholderText('保留已配置的凭据，不读取或显示其内容')
                elif key == 'api_key_env':
                    value.setText(provider.get(key, ''))
                    value.setPlaceholderText('例如 COSLUMA_API_KEY，填写变量名')
                elif key == 'api_key_file':
                    value.setText(provider.get(key, ''))
                    value.setPlaceholderText('选择仓库外已有的密钥文本文件')
            mode.currentIndexChanged.connect(change_mode)
            self.credentials[service] = (mode, value)
            if self.existing:
                mode.setCurrentIndex(mode.findData('keep') if mode.findData('keep') >= 0 else mode.findData(''))
            change_mode(0)
            form.addRow('密钥来源', mode)
            form.addRow('密钥', row)
            if service == 'image':
                options = QGroupBox('图像参数（可选）')
                options.setCheckable(True)
                options.setChecked(False)
                options_layout = QVBoxLayout(options)
                content = QWidget()
                parameters = QFormLayout(content)
                self.image_options = {}
                for key, label in (('quality', '质量'), ('size', '生成尺寸'), ('output_format', '图片格式')):
                    edit = QLineEdit(str(provider.get('parameters', {}).get(key, '')) if self.existing else '')
                    edit.setPlaceholderText('留空则不发送，由服务决定')
                    self.image_options[key] = edit
                    parameters.addRow(label, edit)
                options_layout.addWidget(content)
                content.hide()
                options.toggled.connect(content.setVisible)
                form.addRow(options)
            description = QLabel('地址填写到 /v1 时，保存会补全兼容接口路径；填写完整地址时原样使用。')
            description.setWordWrap(True)
            form.addRow(description)
            self.tabs.addTab(panel, title)
        panel = QWidget()
        form = QFormLayout(panel)
        form.setVerticalSpacing(12)
        for service, title in (('head', '整头分割'), ('sr', '超分')):
            self.add_field(form, service, 'url', title + '地址', 'ComfyUI 根地址，例如 http://127.0.0.1:8188')
            self.add_field(form, service, 'model', title + '模型', '填写服务内实际权重文件名')
        info = QLabel('这两个阶段由你的 ComfyUI 服务完成，可以使用同一地址。\n'
            '模型权重由服务部署者准备，软件不下载权重；安装约定见 README 的服务部署部分。')
        info.setWordWrap(True)
        form.addRow(info)
        self.tabs.addTab(panel, '3  分割与超分')
        panel = QWidget()
        form = QFormLayout(panel)
        self.library_mode = QComboBox()
        self.library_mode.addItems(['使用随附通用提示词库', '选择自己的提示词库'])
        default_library = self.path.parent / 'prompts.json'
        self.library = QLineEdit(self.values.get('prompt_library', str(default_library)))
        self.library_custom = self.library.text()
        row = QHBoxLayout()
        row.addWidget(self.library, 1)
        browse = QPushButton('选择文件')
        browse.clicked.connect(lambda: self.choose_file(self.library, '选择提示词库', 'JSON (*.json)'))
        row.addWidget(browse)
        library_row = QWidget()
        library_row.setLayout(row)
        def switch_library(index):
            if index == 0:
                self.library_custom = self.library.text()
                self.library.setText(str(default_library))
            else:
                self.library.setText(self.library_custom)
            library_row.setVisible(index == 1)
        self.library_mode.currentIndexChanged.connect(switch_library)
        custom = Path(self.library.text()).expanduser().resolve() != default_library
        self.library_mode.setCurrentIndex(1 if custom else 0)
        library_row.setVisible(custom)
        form.addRow('提示词参考', self.library_mode)
        form.addRow(library_row)
        info = QLabel('通用库可直接使用，首次保存时创建个人副本，已有文件不会被覆盖。\n'
            '自定义库须是已有 JSON 文件；只有确认需要自己的参考库时才切换。')
        info.setWordWrap(True)
        form.addRow(info)
        location = QLabel('配置保存到：' + str(self.path))
        location.setWordWrap(True)
        location.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow(location)
        self.tabs.addTab(panel, '提示词与存储')
        self.message = QLabel('')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QDialogButtonBox()
        buttons.addButton('保存设置', QDialogButtonBox.AcceptRole)
        buttons.addButton('取消', QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_field(self, form, service, key, label, placeholder):
        edit = QLineEdit(self.values.get(service, {}).get(key, ''))
        edit.setPlaceholderText(placeholder)
        self.fields[service, key] = edit
        form.addRow(label, edit)

    def choose_file(self, edit, title, pattern):
        filename, _ = QFileDialog.getOpenFileName(self, title, edit.text(), pattern)
        if filename:
            edit.setText(filename)

    def reject(self):
        for mode, edit in self.credentials.values():
            if mode.currentData() == 'inline':
                edit.clear()
        super().reject()

    def save(self):
        try:
            values = json.loads(json.dumps(self.values))
            for (service, key), edit in self.fields.items():
                value = edit.text().strip()
                if not value:
                    self.tabs.setCurrentIndex({'llm': 0, 'image': 1, 'head': 2, 'sr': 2}[service])
                    edit.setFocus()
                    raise ValueError('请填写服务地址和模型名称')
                try:
                    values.setdefault(service, {})[key] = endpoint_url(value, service) if key == 'url' else value
                except ValueError:
                    self.tabs.setCurrentIndex({'llm': 0, 'image': 1, 'head': 2, 'sr': 2}[service])
                    edit.setFocus()
                    raise
            keys = {}
            for service, (mode, edit) in self.credentials.items():
                source, value = mode.currentData(), edit.text().strip()
                if source == 'keep':
                    continue
                if source and not value:
                    self.tabs.setCurrentIndex(0 if service == 'llm' else 1)
                    edit.setFocus()
                    raise ValueError('请填写密钥，或选择其他凭据来源')
                if source == 'inline':
                    keys[service] = value
                    continue
                provider = values[service]
                provider.pop('api_key_env', None)
                provider.pop('api_key_file', None)
                if source:
                    provider[source] = value
            parameters = values['image'].setdefault('parameters', {})
            for key, edit in self.image_options.items():
                if edit.text().strip():
                    parameters[key] = edit.text().strip()
                else:
                    parameters.pop(key, None)
            values['prompt_library'] = self.library.text().strip()
            self.values = save_settings(self.path, values, keys)
            for _mode, edit in self.credentials.values():
                edit.clear()
            self.accept()
        except Exception as error:
            self.message.setText('未保存：' + str(error))


class Workbench(QMainWindow):
    def __init__(self, root=None, config=None):
        super().__init__()
        self.preferences = QSettings(str(Path.home() / 'CosplayWorkbench/config/desktop.ini'), QSettings.IniFormat)
        root = root if root is not None else self.preferences.value('output_root', str(Path.home() / 'CosLuma/output'), type=str)
        config = config or os.environ.get('COSLUMA_CONFIG') or self.preferences.value('config_path', str(default_settings_path()), type=str)
        self.setWindowTitle('CosLuma ' + VERSION)
        self.resize(1160, 850)
        self.setAcceptDrops(True)
        self.worker = None
        self.close_after_finish = False
        self.current_output = None
        self.preview_pixmap = None
        self.batch_directory = None
        self.sources = []
        self.results = {}
        self.rows = {}
        self.inputs = []
        panel = QWidget()
        self.setCentralWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel('CosLuma')
        title.setStyleSheet('font-size: 26px; font-weight: 600')
        header.addWidget(title)
        header.addWidget(QLabel('照片与角色，交给整组摄影设计'), 1)
        header.addWidget(QLabel(VERSION))
        layout.addLayout(header)
        setup = QHBoxLayout()
        self.setup_hint = QLabel('')
        self.setup_hint.setWordWrap(True)
        setup.addWidget(self.setup_hint, 1)
        self.import_button = QPushButton('导入已有配置')
        self.import_button.clicked.connect(self.import_config)
        self.setup_button = QPushButton('连接设置')
        self.setup_button.clicked.connect(self.configure_services)
        setup.addWidget(self.import_button)
        setup.addWidget(self.setup_button)
        layout.addLayout(setup)
        toolbar = QHBoxLayout()
        for title, action in (('添加照片', self.select_sources), ('添加文件夹', self.select_folder),
                              ('移除选中', self.remove_selected), ('清空', self.clear_sources)):
            button = QPushButton(title)
            button.clicked.connect(action)
            toolbar.addWidget(button)
            self.inputs.append(button)
        self.count = QLabel('尚未添加照片')
        toolbar.addWidget(self.count, 1)
        layout.addLayout(toolbar)
        splitter = QSplitter(Qt.Horizontal)
        self.queue_pages = QStackedWidget()
        empty = QLabel('把照片或文件夹拖到这里\n\n支持 JPEG、PNG、TIFF 和相机 RAW')
        empty.setAlignment(Qt.AlignCenter)
        empty.setStyleSheet('background: #f3f5f7; color: #526170; border: 1px dashed #b8c2cc; padding: 25px')
        self.queue_pages.addWidget(empty)
        self.listing = QTableWidget(0, 2)
        self.listing.setHorizontalHeaderLabels(['照片', '制作进度'])
        self.listing.verticalHeader().hide()
        self.listing.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.listing.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.listing.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.listing.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listing.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.listing.itemSelectionChanged.connect(self.show_selected)
        self.queue_pages.addWidget(self.listing)
        splitter.addWidget(self.queue_pages)
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview = QLabel('选择照片查看预览\n制作完成后可逐张查看成片')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(280, 200)
        self.preview.setStyleSheet('background: #f3f5f7; color: #526170; border: 1px solid #d6dde3')
        self.preview.setWordWrap(True)
        preview_layout.addWidget(self.preview, 1)
        self.output_info = QLabel('')
        self.output_info.setWordWrap(True)
        self.output_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        preview_layout.addWidget(self.output_info)
        self.open_photo_button = QPushButton('打开选中成片')
        self.open_photo_button.setEnabled(False)
        self.open_photo_button.clicked.connect(self.open_selected_output)
        preview_layout.addWidget(self.open_photo_button)
        splitter.addWidget(preview_panel)
        splitter.setSizes([500, 600])
        splitter.splitterMoved.connect(lambda *_: self.update_preview())
        layout.addWidget(splitter, 1)
        form = QFormLayout()
        form.setVerticalSpacing(10)
        self.character = QLineEdit(self.preferences.value('character', '', type=str))
        self.character.setPlaceholderText('例如：鸣潮 · 守岸人')
        form.addRow('角色', self.character)
        self.brief = QPlainTextEdit()
        self.brief_character = self.character.text().strip()
        if self.brief_character:
            self.brief.setPlainText(self.preferences.value('briefs/' + self.brief_character, '', type=str))
        self.character.editingFinished.connect(self.switch_character)
        self.brief.setPlaceholderText('可以留空；有希望保留的景物、场景想法或偏好时再补充')
        self.brief.setMaximumHeight(65)
        form.addRow('补充要求', self.brief)
        self.root = QLineEdit(str(root))
        output_row = self.path_row(self.root, directory=True)
        form.addRow('保存到', output_row)
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
        self.edge = QLineEdit(self.preferences.value('output_edge', '', type=str))
        self.edge.setPlaceholderText('留空使用配置；默认请求长边 6000')
        advanced_form.addRow('请求长边', self.edge)
        advanced_layout.addWidget(advanced_panel)
        advanced_panel.hide()
        advanced.toggled.connect(advanced_panel.setVisible)
        layout.addWidget(advanced)
        controls = QHBoxLayout()
        self.start = QPushButton('开始制作')
        self.start.setMinimumHeight(38)
        self.start.clicked.connect(self.produce)
        controls.addWidget(self.start, 1)
        self.open_folder_button = QPushButton('打开本次结果')
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self.open_results)
        controls.addWidget(self.open_folder_button)
        layout.addLayout(controls)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('等待开始')
        layout.addWidget(self.progress_bar)
        self.status = QLabel('添加照片，填写角色后开始。每张照片的进度会显示在列表中。')
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.inputs.extend([self.character, self.brief, output_row, advanced, self.setup_button, self.import_button])
        self.update_setup_hint()

    @staticmethod
    def source_key(source):
        return os.path.normcase(str(Path(source).expanduser().resolve()))

    def update_setup_hint(self):
        exists = Path(self.config.text()).expanduser().is_file()
        self.setup_hint.setText('已选择服务配置，可随时在“连接设置”中修改。' if exists else
            '首次使用：有现成配置可直接导入；否则打开“连接设置”，填写模型服务。')

    def import_config(self):
        filename, _ = QFileDialog.getOpenFileName(self, '导入已有服务配置', self.config.text(), 'JSON (*.json)')
        if not filename:
            return
        try:
            load_settings(filename)
            self.config.setText(filename)
            self.remember_preferences()
            self.status.setText('已选择已有配置。选择照片和角色后开始制作。')
        except Exception as error:
            self.status.setText('无法导入配置：' + str(error))

    def configure_services(self):
        try:
            dialog = ServiceSettings(self.config.text(), self)
            if dialog.exec() == QDialog.Accepted:
                self.config.setText(str(dialog.path))
                self.remember_preferences()
                self.update_setup_hint()
                self.status.setText('连接设置已保存。服务会在实际制作时调用。')
        except Exception as error:
            self.status.setText('无法打开连接设置：' + str(error))

    def switch_character(self):
        character = self.character.text().strip()
        if character == self.brief_character:
            return
        if self.brief_character:
            self.preferences.setValue('briefs/' + self.brief_character, self.brief.toPlainText())
        self.brief_character = character
        self.brief.setPlainText(self.preferences.value('briefs/' + character, '', type=str) if character else '')

    def remember_preferences(self):
        self.switch_character()
        self.preferences.setValue('character', self.brief_character)
        if self.brief_character:
            self.preferences.setValue('briefs/' + self.brief_character, self.brief.toPlainText())
        for key, value in (('output_root', self.root.text()), ('config_path', self.config.text()), ('output_edge', self.edge.text())):
            if value.strip() or key == 'output_edge':
                self.preferences.setValue(key, value.strip())
        self.preferences.sync()

    def path_row(self, edit, directory=False):
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        browse = QPushButton('选择')
        def choose():
            selected = QFileDialog.getExistingDirectory(self, '选择保存目录', edit.text()) if directory else (
                QFileDialog.getOpenFileName(self, '选择配置文件', edit.text(), 'JSON (*.json)')[0])
            if selected:
                edit.setText(selected)
        browse.clicked.connect(choose)
        row.addWidget(browse)
        return widget

    def add_sources(self, paths):
        if self.worker is not None:
            return
        candidates = []
        try:
            for value in paths:
                path = Path(value).expanduser().resolve()
                candidates.extend(sorted(path.iterdir()) if path.is_dir() else [path])
        except OSError as error:
            self.status.setText('无法添加照片：' + str(error))
            return
        known = {self.source_key(source) for source in self.sources}
        for path in candidates:
            key = self.source_key(path)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and key not in known:
                self.sources.append(str(path))
                known.add(key)
        self.refresh_queue()

    def select_sources(self):
        paths, _ = QFileDialog.getOpenFileNames(self, '添加原片', '',
            '照片 (' + ' '.join('*' + suffix for suffix in sorted(IMAGE_EXTENSIONS)) + ')')
        self.add_sources(paths)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, '添加文件夹中的照片')
        if folder:
            self.add_sources([folder])

    def remove_selected(self):
        if self.worker is not None:
            return
        rows = {item.row() for item in self.listing.selectedItems()}
        self.sources = [source for row, source in enumerate(self.sources) if row not in rows]
        self.refresh_queue()

    def clear_sources(self):
        if self.worker is not None:
            return
        self.sources.clear()
        self.results.clear()
        self.refresh_queue()

    def refresh_queue(self):
        self.rows.clear()
        self.listing.setRowCount(len(self.sources))
        for row, source in enumerate(self.sources):
            key = self.source_key(source)
            self.rows[key] = row
            name = QTableWidgetItem(Path(source).name)
            name.setData(Qt.UserRole, source)
            name.setToolTip(source)
            self.listing.setItem(row, 0, name)
            result = self.results.get(key)
            stage = ('完成' if result['status'] == 'complete' else '失败') if result else '等待制作'
            self.listing.setItem(row, 1, QTableWidgetItem(stage))
        self.queue_pages.setCurrentIndex(1 if self.sources else 0)
        self.count.setText(f'{len(self.sources)} 张照片' if self.sources else '尚未添加照片')
        if self.worker is None:
            self.progress_bar.setRange(0, max(1, len(self.sources)))
            self.progress_bar.setValue(sum(self.source_key(source) in self.results for source in self.sources))
            self.progress_bar.setFormat('已处理 %v / %m 张' if self.results else '等待开始')
        if self.sources:
            self.listing.selectRow(0)
        else:
            self.current_output = None
            self.preview_pixmap = None
            self.preview.setText('选择照片查看预览\n制作完成后可逐张查看成片')
            self.output_info.clear()
            self.open_photo_button.setEnabled(False)

    def dragEnterEvent(self, event):
        if self.worker is None and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        try:
            self.add_sources([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
            event.acceptProposedAction()
        except OSError as error:
            self.status.setText('无法添加照片：' + str(error))

    def show_selected(self):
        row = self.listing.currentRow()
        if row < 0 or row >= len(self.sources):
            return
        source = self.sources[row]
        result = self.results.get(self.source_key(source))
        self.current_output = None
        self.open_photo_button.setEnabled(False)
        if result and result['status'] == 'complete':
            path = result.get('exports', {}).get('JPEG', result['files']['jpeg'])
            self.current_output = path
            width, height = result['output_dimensions']
            self.output_info.setText(f'{Path(source).name} · 成片 {width} × {height}\n16 位 PNG 和 JPEG 已保存')
            if result.get('output_adjustment'):
                self.output_info.setText(self.output_info.text() + '\n已按模型实际覆盖范围输出。')
            self.open_photo_button.setEnabled(True)
        else:
            path = source
            self.output_info.setText(f'{Path(source).name} · 原片预览' if not result else
                f'{Path(source).name} · 制作失败\n{result["error"]}')
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            size.scale(QSize(1200, 1200), Qt.KeepAspectRatio)
            reader.setScaledSize(size)
        image = reader.read()
        self.preview_pixmap = QPixmap.fromImage(image) if not image.isNull() else None
        if self.preview_pixmap is None:
            self.preview.setText('此文件暂时无法预览\n可从保存目录打开成片' if self.current_output else
                '原片已加入队列\n完成后将在这里显示成片')
        else:
            self.update_preview()

    def produce(self):
        try:
            if not self.sources or not self.character.text().strip():
                raise ValueError('请添加照片并填写角色')
            if not self.root.text().strip():
                raise ValueError('请选择保存目录')
            if not Path(self.config.text()).expanduser().is_file():
                self.configure_services()
                if not Path(self.config.text()).expanduser().is_file():
                    return
            config = load_settings(self.config.text())
            if self.edge.text().strip():
                edge = int(self.edge.text())
                if edge <= 0:
                    raise ValueError('请求长边应为正整数')
                config['output_long_edge'] = edge
            self.remember_preferences()
            self.results.clear()
            self.batch_directory = None
            self.refresh_queue()
            self.progress_bar.setRange(0, len(self.sources))
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat('已处理 %v / %m 张')
            self.open_folder_button.setEnabled(False)
            self.worker = Worker(list(self.sources), self.character.text().strip(), self.brief.toPlainText(),
                self.root.text(), config)
            self.worker.progress.connect(self.status.setText)
            self.worker.photo_stage.connect(self.set_photo_stage)
            self.worker.result_ready.connect(self.show_result)
            self.worker.failed.connect(self.show_failure)
            self.worker.finished.connect(self.batch_finished)
            for widget in self.inputs:
                widget.setEnabled(False)
            self.start.setEnabled(False)
            self.status.setText('正在设计整组场景与光色')
            self.worker.start()
        except Exception as error:
            self.status.setText('无法开始：' + str(error))

    @Slot(str, str)
    def set_photo_stage(self, source, stage):
        key = self.source_key(source)
        if key in self.rows and key not in self.results:
            self.listing.item(self.rows[key], 1).setText(stage)

    @Slot(dict)
    def show_result(self, result):
        key = self.source_key(result['source'])
        self.results[key] = result
        self.batch_directory = result.get('batch_directory', self.batch_directory)
        self.listing.item(self.rows[key], 1).setText('完成' if result['status'] == 'complete' else '失败')
        self.listing.item(self.rows[key], 1).setToolTip('' if result['status'] == 'complete' else str(result['error']))
        self.progress_bar.setValue(len(self.results))
        self.open_folder_button.setEnabled(bool(self.batch_directory))
        if len(self.results) == 1:
            self.listing.selectRow(self.rows[key])
        self.show_selected()

    @Slot(str)
    def show_failure(self, message):
        self.status.setText('制作失败：' + message)

    @Slot()
    def batch_finished(self):
        worker, self.worker = self.worker, None
        self.batch_directory = worker.output_directory or self.batch_directory
        if worker.batch_error:
            for key, row in self.rows.items():
                if key not in self.results:
                    self.listing.item(row, 1).setText('未完成')
            summary = f'批次失败：{worker.batch_error}'
        elif worker.errors:
            summary = f'批次未全部完成：成功 {worker.succeeded} 张，失败 {len(worker.errors)} 张。'
        else:
            summary = f'批次完成：成功 {worker.succeeded} 张。'
        self.status.setText(summary + (' 点击失败照片可查看具体原因。' if worker.errors else ''))
        for widget in self.inputs:
            widget.setEnabled(True)
        self.start.setEnabled(True)
        self.open_folder_button.setEnabled(bool(self.batch_directory))
        worker.deleteLater()
        if self.close_after_finish:
            self.close()
            QApplication.instance().quit()

    def open_selected_output(self):
        if self.current_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.current_output))

    def open_results(self):
        if self.batch_directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.batch_directory)))

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
    if sys.argv[1:2] == ['--cli']:
        del sys.argv[1]
        from project_cli import main as command_line
        return command_line()
    if sys.argv[1:2] == ['--server']:
        del sys.argv[1]
        from backend_server import main as server
        return server()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='CosLuma ' + VERSION)
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
