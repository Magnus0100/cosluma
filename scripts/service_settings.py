"""Guided local configuration for the desktop's existing model services."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLineEdit, QListWidget, QPushButton, QScrollArea,
    QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from settings import editable_settings, resource_path, save_settings
from model_downloads import ModelDownloads
from prompt_library import load_library
from studio_widgets import ui_label


def endpoint_url(value, service):
    """Expand an explicitly entered compatible API base when saving settings."""
    parsed = urlsplit(value.strip())
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.hostname == 'api.example.com':
        raise ValueError('请填写服务方提供的 HTTP/HTTPS 地址')
    if service in ('llm', 'image'):
        path = parsed.path.rstrip('/')
        if path in ('', '/v1'):
            path = (path or '/v1') + ('/chat/completions' if service == 'llm' else '/images/edits')
            parsed = parsed._replace(path=path)
    else:
        parsed = parsed._replace(path=parsed.path.rstrip('/'))
    return urlunsplit(parsed)


class ServiceSettings(QDialog):
    service_names = {'llm':'摄影 Agent', 'image':'整图生图', 'head':'整头分割', 'sr':'超分'}
    service_pages = {'llm':0, 'image':1, 'head':2, 'sr':2}

    def __init__(self, path, parent=None, *, output_root=None, project_output=True):
        super().__init__(parent)
        self.path = Path(path).expanduser().resolve()
        self.existing = self.path.is_file()
        self.values = editable_settings(self.path)
        self.fields = {}
        self.credentials = {}
        self.setWindowTitle('CosLuma 连接设置')
        self.resize(920, 650)
        self.setMinimumSize(760, 530)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)
        layout.addWidget(ui_label('连接你的创作工具', 'title'))
        note = ui_label('使用你自己的模型服务。这里只保存设置；开始选片、讨论或试样时才会调用服务。')
        note.setWordWrap(True)
        layout.addWidget(note)
        body = QHBoxLayout()
        body.setSpacing(20)
        rail = QWidget()
        rail.setFixedWidth(180)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        self.steps = QListWidget()
        self.steps.setObjectName('setupSteps')
        self.steps.addItems(['1  摄影 Agent', '2  整图生图', '3  分割与超分', '4  核对与保存'])
        rail_layout.addWidget(self.steps, 1)
        downloads = QPushButton('模型下载')
        downloads.clicked.connect(lambda: ModelDownloads(self).exec())
        rail_layout.addWidget(downloads)
        help_button = QPushButton('查看服务部署说明')
        help_button.setProperty('kind', 'quiet')
        help_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(resource_path('README.md')))))
        rail_layout.addWidget(help_button)
        body.addWidget(rail)
        self.tabs = QStackedWidget()
        body.addWidget(self.tabs, 1)
        layout.addLayout(body, 1)
        descriptions = {
            'llm':('让 Agent 看懂你的照片', '用于选片、讨论摄影方向和编写提示词。请选择支持图片输入和工具调用的图文模型，Agent 会按需查阅参考库。'),
            'image':('连接整图生图服务', '用于制作完整真实样片与后续成片，也承担用户主动开启的修发和人物光照设计。请选择支持原片和风格参考图输入的图像编辑模型。')}
        for service in ('llm', 'image'):
            form = self.add_page(*descriptions[service])
            if service == 'image':
                self.same_service = QPushButton('沿用摄影 Agent 的地址和密钥')
                self.same_service.clicked.connect(self.use_prompt_service)
                form.addRow(self.same_service)
            self.add_field(form, service, 'url', '服务地址', '服务方提供的地址，例如 https://服务地址/v1')
            self.add_field(form, service, 'model', '模型名称', '填写服务方实际提供的模型名')
            if not self.existing:
                self.fields[service, 'url'].clear()
                self.fields[service, 'model'].clear()
            self.add_credentials(form, service)
            description = ui_label('可填写 /v1 或完整接口地址，保存时会补全标准接口路径。')
            description.setWordWrap(True)
            form.addRow(description)
            if service == 'image':
                self.add_image_options(form)
        form = self.add_page('保留本人身份与原片细节',
            '整头分割提取原片头部，修发时另外识别需要保留的脸部与头饰；超分补充生成区域的清晰度。这两项由已经部署好模型的 ComfyUI 提供。')
        for service, label in (('head','整头分割'), ('sr','超分')):
            form.addRow(ui_label(label, 'heading'))
            if service == 'sr':
                reuse = QPushButton('超分使用相同的 ComfyUI 地址')
                reuse.clicked.connect(lambda: self.fields['sr','url'].setText(self.fields['head','url'].text()))
                form.addRow(reuse)
            self.add_field(form, service, 'url', 'ComfyUI 地址', '例如 http://127.0.0.1:8188')
            self.add_field(form, service, 'model', '模型文件名', '填写服务机器上实际安装的权重文件名')
            download = QPushButton('下载 SAM3.1 分割模型' if service == 'head' else '下载 Nomos2 超分模型')
            download.clicked.connect(lambda _checked=False, service=service: ModelDownloads(self, model=service).exec())
            form.addRow(download)
        info = ui_label('ComfyUI 地址可指向本机或远端。软件包不包含模型服务和权重，请按部署说明准备。')
        info.setWordWrap(True)
        form.addRow(info)
        form = self.add_page('核对设置，开始第一组照片',
            '确认服务用途和保存位置。保存不测试连接，也不生成图片；实际服务错误会在调用步骤显示。')
        self.summary = {}
        for service, label in self.service_names.items():
            row = QHBoxLayout()
            summary = ui_label('')
            summary.setWordWrap(True)
            row.addWidget(summary, 1)
            edit = QPushButton('修改')
            edit.setProperty('kind', 'quiet')
            edit.clicked.connect(lambda _checked=False, service=service: self.tabs.setCurrentIndex(self.service_pages[service]))
            row.addWidget(edit)
            self.summary[service] = summary
            form.addRow(label, row)
        count = sum(not entry.get('overview') for entry in load_library())
        builtin_description = f'已内置 {count} 条摄影参考，按任务分类、由 Agent 按需查阅；无需下载或选择词库文件。'
        self.library_status = ui_label('当前使用自定义参考库：\n'+self.values['prompt_library']
            if self.values.get('prompt_library') else builtin_description)
        self.library_status.setWordWrap(True)
        form.addRow('参考词库', self.library_status)
        if self.values.get('prompt_library'):
            builtin = QPushButton('改用内置参考库')
            def use_builtin():
                self.values['prompt_library'] = None
                self.library_status.setText(builtin_description)
                builtin.setVisible(False)
            builtin.clicked.connect(use_builtin)
            form.addRow(builtin)
        self.output_root = QLineEdit(str(output_root or Path.home()/'CosLuma/output'))
        if not project_output:
            note = ui_label('新建项目时选择保存位置，默认在项目文件夹的 work 中保存制作结果。')
            note.setWordWrap(True)
            form.addRow('项目保存位置', note)
        else:
            row = QHBoxLayout()
            row.addWidget(self.output_root, 1)
            browse = QPushButton('选择文件夹')
            browse.clicked.connect(self.choose_output)
            row.addWidget(browse)
            form.addRow('本项目制作结果保存到', row)
        location = ui_label(str(self.path))
        location.setWordWrap(True)
        location.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow('服务配置保存到', location)
        note = ui_label('密钥另存在本机文件中，配置 JSON 只记录路径。可选美肤与瘦脸在出片后再设置。')
        note.setWordWrap(True)
        form.addRow(note)
        self.message = ui_label('')
        self.message.setObjectName('setupMessage')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        navigation = QHBoxLayout()
        self.cancel = QPushButton('取消' if self.existing else '稍后配置')
        self.cancel.setProperty('kind', 'quiet')
        self.cancel.clicked.connect(self.reject)
        navigation.addWidget(self.cancel)
        navigation.addStretch(1)
        self.previous = QPushButton('上一步')
        self.previous.clicked.connect(lambda: self.tabs.setCurrentIndex(self.tabs.currentIndex()-1))
        navigation.addWidget(self.previous)
        self.next = QPushButton('下一步')
        self.next.setProperty('kind', 'primary')
        self.next.clicked.connect(self.next_step)
        navigation.addWidget(self.next)
        self.save_button = QPushButton('保存设置' if self.existing else '保存并开始')
        self.save_button.setProperty('kind', 'primary')
        self.save_button.clicked.connect(self.save)
        navigation.addWidget(self.save_button)
        layout.addLayout(navigation)
        self.steps.currentRowChanged.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self.update_navigation)
        self.steps.setCurrentRow(0)
        self.update_navigation()

    def add_page(self, title, description):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 0, 12, 12)
        layout.setSpacing(14)
        layout.addWidget(ui_label(title, 'heading'))
        note = ui_label(description)
        note.setWordWrap(True)
        layout.addWidget(note)
        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(12)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.addWidget(content)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(panel)
        self.tabs.addWidget(scroll)
        return form

    def add_field(self, form, service, key, label, placeholder):
        edit = QLineEdit(self.values.get(service, {}).get(key, ''))
        edit.setPlaceholderText(placeholder)
        self.fields[service, key] = edit
        form.addRow(label, edit)

    def add_credentials(self, form, service):
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
        browse.clicked.connect(lambda: self.choose_file(value, '选择密钥文件', '所有文件 (*)'))
        row = QHBoxLayout()
        row.addWidget(value, 1)
        row.addWidget(browse)
        def change_mode(_index):
            key = mode.currentData()
            value.clear()
            value.setEchoMode(QLineEdit.Password if key == 'inline' else QLineEdit.Normal)
            value.setEnabled(key not in ('', 'keep'))
            browse.setVisible(key == 'api_key_file')
            placeholders = {'inline':'粘贴服务密钥', 'keep':'沿用当前凭据，不读取或显示密钥内容',
                'api_key_env':'填写变量名，例如 COSLUMA_API_KEY', 'api_key_file':'选择已有的密钥文本文件',
                '':'服务不要求密钥'}
            value.setPlaceholderText(placeholders[key])
            if key in ('api_key_env', 'api_key_file'):
                value.setText(provider.get(key, ''))
        mode.currentIndexChanged.connect(change_mode)
        self.credentials[service] = (mode, value)
        if self.existing:
            mode.setCurrentIndex(mode.findData('keep') if mode.findData('keep') >= 0 else mode.findData(''))
        change_mode(0)
        form.addRow('密钥来源', mode)
        form.addRow('密钥', row)

    def add_image_options(self, form):
        options = QWidget()
        layout = QVBoxLayout(options)
        layout.setContentsMargins(0, 0, 0, 0)
        self.image_options_toggle = QToolButton()
        self.image_options_toggle.setCheckable(True)
        self.image_options_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        layout.addWidget(self.image_options_toggle, 0, Qt.AlignLeft)
        content = QWidget()
        parameters = QFormLayout(content)
        parameters.setContentsMargins(0, 0, 0, 0)
        self.image_options = {}
        for key, label in (('quality','质量'), ('size','生成尺寸'), ('output_format','图片格式')):
            value = self.values.get('image', {}).get('parameters', {}).get(key, '') if self.existing else ''
            edit = QLineEdit(str(value))
            edit.setPlaceholderText('留空则不发送，由服务决定')
            self.image_options[key] = edit
            parameters.addRow(label, edit)
        layout.addWidget(content)
        def show_parameters(expanded):
            content.setVisible(expanded)
            self.image_options_toggle.setText('收起图像参数' if expanded else '图像参数（可选）')
            self.image_options_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        expanded = any(edit.text().strip() for edit in self.image_options.values())
        self.image_options_toggle.setChecked(expanded)
        self.image_options_toggle.toggled.connect(show_parameters)
        show_parameters(expanded)
        note = ui_label('仅填写服务支持的参数。收起保留已填值，留空才不发送。')
        note.setWordWrap(True)
        layout.addWidget(note)
        form.addRow(options)

    def update_navigation(self):
        index = self.tabs.currentIndex()
        self.steps.setCurrentRow(index)
        self.previous.setEnabled(index > 0)
        self.next.setVisible(index < self.tabs.count()-1)
        self.save_button.setVisible(index == self.tabs.count()-1)
        self.next.setDefault(index < self.tabs.count()-1)
        self.save_button.setDefault(index == self.tabs.count()-1)
        self.message.clear()
        if index == 3:
            for service, summary in self.summary.items():
                model = self.fields[service,'model'].text().strip() or '尚未填写模型'
                address = self.fields[service,'url'].text().strip() or '尚未填写地址'
                summary.setText(model+'\n'+address)

    def provider_settings(self, service):
        provider = deepcopy(self.values.get(service, {}))
        for key, label in (('url','服务地址'), ('model','模型名称')):
            edit = self.fields[service,key]
            value = edit.text().strip()
            try:
                if not value:
                    raise ValueError('请填写'+self.service_names[service]+'的'+label)
                provider[key] = endpoint_url(value, service) if key == 'url' else value
            except ValueError:
                self.tabs.setCurrentIndex(self.service_pages[service])
                edit.setFocus()
                raise
        secret = None
        if service in self.credentials:
            mode, edit = self.credentials[service]
            source, value = mode.currentData(), edit.text().strip()
            if source == 'keep':
                return provider, secret
            if source and not value:
                self.tabs.setCurrentIndex(self.service_pages[service])
                edit.setFocus()
                raise ValueError('请填写'+self.service_names[service]+'的密钥，或选择其他密钥来源')
            if source == 'inline':
                secret = value
            else:
                provider.pop('api_key_env', None)
                provider.pop('api_key_file', None)
                if source:
                    provider[source] = value
        return provider, secret

    def next_step(self):
        try:
            index = self.tabs.currentIndex()
            for service, page in self.service_pages.items():
                if page == index:
                    self.provider_settings(service)
            self.tabs.setCurrentIndex(index+1)
        except ValueError as error:
            self.message.setText(str(error))

    def use_prompt_service(self):
        try:
            provider, secret = self.provider_settings('llm')
            url = urlsplit(provider['url'])
            if not url.path.endswith('/chat/completions'):
                raise ValueError('摄影 Agent 使用自定义路径，请分别填写两个完整接口地址')
            self.fields['image','url'].setText(urlunsplit(url._replace(
                path=url.path[:-len('/chat/completions')]+'/images/edits')))
            target_mode, target_value = self.credentials['image']
            if secret is not None:
                mode, value = 'inline', secret
            elif provider.get('api_key_file'):
                mode, value = 'api_key_file', provider['api_key_file']
            elif provider.get('api_key_env'):
                mode, value = 'api_key_env', provider['api_key_env']
            else:
                mode, value = '', ''
            target_mode.setCurrentIndex(target_mode.findData(mode))
            target_value.setText(value)
            self.tabs.setCurrentIndex(1)
            self.message.setText('地址与密钥来源已沿用。请填写图像编辑模型名称。')
        except ValueError as error:
            self.message.setText(str(error))

    def choose_file(self, edit, title, pattern):
        filename, _ = QFileDialog.getOpenFileName(self, title, edit.text(), pattern)
        if filename:
            edit.setText(filename)

    def choose_output(self):
        directory = QFileDialog.getExistingDirectory(self, '选择照片结果保存文件夹', self.output_root.text())
        if directory:
            self.output_root.setText(directory)

    def reject(self):
        for mode, edit in self.credentials.values():
            if mode.currentData() == 'inline':
                edit.clear()
        super().reject()

    def save(self):
        try:
            values, keys = deepcopy(self.values), {}
            for service in self.service_pages:
                provider, secret = self.provider_settings(service)
                values[service] = provider
                if secret is not None:
                    keys[service] = secret
            parameters = values['image'].setdefault('parameters', {})
            for key, edit in self.image_options.items():
                if edit.text().strip():
                    parameters[key] = edit.text().strip()
                else:
                    parameters.pop(key, None)
            self.tabs.setCurrentIndex(3)
            if not self.output_root.text().strip():
                self.output_root.setFocus()
                raise ValueError('请选择照片结果保存文件夹')
            self.values = save_settings(self.path, values, keys)
            for _mode, edit in self.credentials.values():
                edit.clear()
            self.accept()
        except Exception as error:
            self.message.setText('未保存：'+str(error))
