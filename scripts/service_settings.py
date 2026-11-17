"""Connection settings for the desktop's existing model services."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (QComboBox, QCompleter, QDialog, QFileDialog, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QLineEdit, QListWidget, QPushButton, QScrollArea,
    QSizePolicy, QStackedWidget, QStyle, QToolButton, QVBoxLayout, QWidget)

from settings import credential, editable_settings, resource_path, save_settings
from model_downloads import ModelDownloads
from prompt_library import load_library
from studio_widgets import StatusText, ui_label
from relighting import RELIGHT_DEFAULTS


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


def model_list_url(value, service):
    parsed = urlsplit(endpoint_url(value, service))
    path = parsed.path.rstrip('/')
    if service in ('llm', 'image'):
        for suffix in ('/chat/completions', '/images/edits', '/models'):
            if path.endswith(suffix):
                path = path[:-len(suffix)]
                break
        path += '/models'
    else:
        node = 'CheckpointLoaderSimple' if service == 'head' else 'UpscaleModelLoader'
        path += '/object_info/'+node
    return urlunsplit(parsed._replace(path=path, fragment=''))


class ModelPicker(QWidget):
    """Read service metadata asynchronously; model selection remains an explicit edit."""

    def __init__(self, value, connection, service, parent=None):
        super().__init__(parent)
        self.connection, self.service = connection, service
        self.network = QNetworkAccessManager(self)
        self.reply = None
        self.dirty = True
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(350)
        self.timer.timeout.connect(self.load_models)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.NoInsert)
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(12)
        self.combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo.setMaxVisibleItems(12)
        self.combo.setCurrentText(value)
        completer = self.combo.completer()
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        self.edit = self.combo.lineEdit()
        self.edit.setPlaceholderText('选择或输入模型名称')
        row.addWidget(self.combo, 1)
        self.refresh = QToolButton()
        self.refresh.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh.setToolTip('重新读取模型')
        self.refresh.setAccessibleName('重新读取模型')
        self.refresh.clicked.connect(self.load_models)
        row.addWidget(self.refresh)
        layout.addLayout(row)
        self.status = StatusText('')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def invalidate(self):
        self.cancel()
        self.dirty = True
        current = self.edit.text()
        self.combo.clear()
        self.combo.setCurrentText(current)
        self.status.clear()

    def schedule(self):
        if self.dirty and self.isVisible():
            self.timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        self.schedule()

    def cancel(self):
        self.timer.stop()
        if self.reply is not None:
            reply, self.reply = self.reply, None
            reply.abort()
        self.refresh.setEnabled(True)

    def load_models(self):
        self.cancel()
        try:
            connection = self.connection()
            if connection is None:
                self.status.clear()
                return
            url, key = connection
            request = QNetworkRequest(QUrl(url))
            request.setTransferTimeout(15000)
            request.setRawHeader(b'Accept', b'application/json')
            if key:
                request.setRawHeader(b'Authorization', ('Bearer '+key).encode('utf-8'))
            self.dirty = False
            self.status.setText('正在读取模型…')
            self.refresh.setEnabled(False)
            reply = self.network.get(request)
            self.reply = reply
            reply.finished.connect(lambda: self.loaded(reply))
        except (OSError, ValueError, KeyError) as error:
            self.dirty = False
            self.status.setText('模型列表：'+str(error))

    def loaded(self, reply):
        try:
            if self.reply is not reply:
                return
            self.reply = None
            self.refresh.setEnabled(True)
            raw = bytes(reply.readAll()).decode('utf-8-sig', errors='replace')
            if reply.error() != QNetworkReply.NoError:
                code = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
                detail = 'HTTP '+str(code) if code else reply.errorString()
                self.status.setText('模型列表读取失败：'+detail)
                self.status.setToolTip(raw or reply.errorString())
                return
            payload = json.loads(raw)
            if self.service in ('llm', 'image'):
                names = [item['id'] for item in payload['data']]
            else:
                node, field = ('CheckpointLoaderSimple', 'ckpt_name') if self.service == 'head' else (
                    'UpscaleModelLoader', 'model_name')
                names = payload[node]['input']['required'][field][0]
            names = sorted(set(names), key=str.casefold)
            current = self.edit.text()
            self.combo.clear()
            self.combo.addItems(names)
            self.combo.setCurrentText(current)
            self.status.setText(f'{len(names)} 个模型' if names else '暂无模型')
            self.status.setToolTip('')
        except (ValueError, KeyError, TypeError) as error:
            self.status.setText('模型列表格式无法读取，可手动填写')
            self.status.setToolTip(str(error))
        finally:
            reply.deleteLater()


class ServiceSettings(QDialog):
    service_names = {'llm':'摄影助手', 'image':'图像编辑', 'head':'整头分割', 'sr':'超分'}
    service_pages = {'llm':0, 'image':1, 'head':2, 'sr':2}

    def __init__(self, path, parent=None, *, output_root=None, project_output=True):
        super().__init__(parent)
        self.path = Path(path).expanduser().resolve()
        self.existing = self.path.is_file()
        self.values = editable_settings(self.path)
        self.fields = {}
        self.credentials = {}
        self.model_pickers = {}
        self.setWindowTitle('CosLuma 连接设置')
        self.resize(920, 650)
        self.setMinimumSize(760, 530)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)
        layout.addWidget(ui_label('连接设置', 'title'))
        body = QHBoxLayout()
        body.setSpacing(20)
        rail = QWidget()
        rail.setFixedWidth(160)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        self.steps = QListWidget()
        self.steps.setObjectName('setupSteps')
        self.steps.addItems(['摄影助手', '图像编辑', '分割与超分', '保存位置'])
        rail_layout.addWidget(self.steps, 1)
        downloads = QPushButton('模型下载')
        downloads.clicked.connect(lambda: ModelDownloads(self).exec())
        rail_layout.addWidget(downloads)
        help_button = QPushButton('部署说明')
        help_button.setProperty('kind', 'quiet')
        help_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(resource_path('README.md')))))
        rail_layout.addWidget(help_button)
        body.addWidget(rail)
        self.tabs = QStackedWidget()
        self.tabs.setProperty('role','panel')
        body.addWidget(self.tabs, 1)
        layout.addLayout(body, 1)
        descriptions = {
            'llm':('摄影助手', '需要支持图片输入和工具调用。'),
            'image':('图像编辑', '需要支持原片与参考图输入。')}
        for service in ('llm', 'image'):
            form = self.add_page(*descriptions[service])
            if service == 'image':
                self.same_service = QPushButton('沿用摄影助手连接')
                self.same_service.clicked.connect(self.use_prompt_service)
                form.addRow(self.same_service)
            self.add_field(form, service, 'url', '服务地址', '服务方提供的地址，例如 https://服务地址/v1')
            self.add_field(form, service, 'model', '模型名称', '填写服务方实际提供的模型名')
            if not self.existing:
                self.fields[service, 'url'].clear()
                self.fields[service, 'model'].clear()
            self.add_credentials(form, service)
            self.fields[service,'url'].setToolTip('支持 /v1 或完整接口地址；保存时补全标准接口路径。')
            if service == 'image':
                self.add_image_options(form)
        form = self.add_page('分割与超分','填写 ComfyUI 地址与已安装的模型文件名。')
        for service, label in (('head','整头分割'), ('sr','超分')):
            form.addRow(ui_label(label, 'heading'))
            if service == 'sr':
                reuse = QPushButton('沿用分割服务地址')
                reuse.clicked.connect(lambda: self.fields['sr','url'].setText(self.fields['head','url'].text()))
                reuse.clicked.connect(lambda: self.model_pickers['sr'].schedule())
                form.addRow(reuse)
            self.add_field(form, service, 'url', 'ComfyUI 地址', '例如 http://127.0.0.1:8188')
            self.add_field(form, service, 'model', '模型文件名', '填写服务机器上实际安装的权重文件名')
            download = QPushButton('下载 SAM3.1 分割模型' if service == 'head' else '下载 Nomos2 超分模型')
            download.clicked.connect(lambda _checked=False, service=service: ModelDownloads(self, model=service).exec())
            form.addRow(download)
        for service, picker in self.model_pickers.items():
            address = self.fields[service, 'url']
            address.textChanged.connect(lambda _text, picker=picker: picker.invalidate())
            address.editingFinished.connect(picker.schedule)
            if service in self.credentials:
                mode, value = self.credentials[service]
                mode.currentIndexChanged.connect(lambda _index, picker=picker: picker.invalidate())
                mode.currentIndexChanged.connect(lambda _index, picker=picker: picker.schedule())
                value.textChanged.connect(lambda _text, picker=picker: picker.invalidate())
                value.editingFinished.connect(picker.schedule)
        self.relight_group = QGroupBox('人物重打光 · PIXLRelight')
        self.relight_group.setCheckable(True)
        self.relight_group.setChecked(bool(self.values.get('relight')))
        lighting_layout = QVBoxLayout(self.relight_group)
        lighting_body = QWidget()
        lighting_form = QFormLayout(lighting_body)
        lighting_layout.addWidget(lighting_body)
        lighting_body.setVisible(self.relight_group.isChecked())
        self.relight_group.toggled.connect(lighting_body.setVisible)
        self.relight_fields = {}
        lighting = {**RELIGHT_DEFAULTS, **self.values.get('relight',{})}
        for key,label in (('url','ComfyUI 地址'),('model_directory','PIXLRelight 目录'),
                ('intrinsics_directory','Marigold 目录')):
            field = QLineEdit(lighting.get(key,self.values.get('head',{}).get('url','')))
            self.relight_fields[key] = field
            lighting_form.addRow(label,field)
        self.relight_device = QComboBox()
        self.relight_device.addItems(['cuda:0','cpu'])
        device = str(lighting['device'])
        hint = None
        if self.relight_device.findText(device) < 0:
            # The relight node only accepts cuda:0 and cpu; keep a persisted value visible and
            # unselected instead of replacing it, and save it back until the user picks one.
            message = '当前配置 '+device+' 不在节点支持项中；未改选时保留原值'
            self.relight_device.setPlaceholderText(message)
            self.relight_device.setCurrentIndex(-1)
            hint = ui_label(message)
            hint.setWordWrap(True)
        else:
            self.relight_device.setCurrentText(device)
        lighting_form.addRow('推理设备',self.relight_device)
        if hint is not None:
            lighting_form.addRow(hint)
            self.relight_device.currentIndexChanged.connect(
                lambda _index: hint.setVisible(self.relight_device.currentIndex() < 0))
        download = QPushButton('模型与安装说明')
        download.clicked.connect(lambda: ModelDownloads(self,model='relight').exec())
        lighting_form.addRow(download)
        self.relight_group.setToolTip('仅在制作时主动开启重打光才调用；这里保存连接，模型目录位于服务机器。')
        form.addRow(self.relight_group)
        form = self.add_page('保存位置','')
        count = sum(not entry.get('overview') for entry in load_library())
        builtin_description = f'内置参考库（{count} 条）'
        self.library_status = ui_label('自定义参考库：\n'+self.values['prompt_library']
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
            note = ui_label('项目文件夹 / work')
            note.setWordWrap(True)
            form.addRow('项目保存位置', note)
        else:
            row = QHBoxLayout()
            row.addWidget(self.output_root, 1)
            browse = QPushButton('选择文件夹')
            browse.clicked.connect(self.choose_output)
            row.addWidget(browse)
            form.addRow('成片位置', row)
        location = ui_label(str(self.path))
        location.setWordWrap(True)
        location.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow('配置文件', location)
        self.message = StatusText('')
        self.message.setObjectName('setupMessage')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        navigation = QHBoxLayout()
        self.cancel = QPushButton('取消')
        self.cancel.setProperty('kind', 'quiet')
        self.cancel.clicked.connect(self.reject)
        navigation.addWidget(self.cancel)
        navigation.addStretch(1)
        self.save_button = QPushButton('保存')
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
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)
        layout.addWidget(ui_label(title, 'heading'))
        if description:
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
        if key == 'model':
            picker = ModelPicker(self.values.get(service, {}).get(key, ''),
                lambda: self.model_connection(service), service)
            self.model_pickers[service] = picker
            self.fields[service, key] = picker.edit
            form.addRow(label, picker)
            return
        edit = QLineEdit(self.values.get(service, {}).get(key, ''))
        edit.setPlaceholderText(placeholder)
        self.fields[service, key] = edit
        form.addRow(label, edit)

    def model_connection(self, service):
        address = self.fields[service, 'url'].text().strip()
        if not address:
            return None
        key = None
        if service in self.credentials:
            mode, value = self.credentials[service]
            source, text = mode.currentData(), value.text().strip()
            if source == 'keep':
                key = credential(self.values.get(service, {}))
            elif source == 'inline':
                if not text:
                    return None
                key = text
            elif source:
                if not text:
                    return None
                if source == 'api_key_file':
                    path = Path(text).expanduser()
                    text = str((path if path.is_absolute() else self.path.parent / path).resolve())
                key = credential({source:text})
        return model_list_url(address, service), key

    def add_credentials(self, form, service):
        provider = self.values.get(service, {})
        mode = QComboBox()
        mode.setToolTip('粘贴的密钥保存在独立本机文件中，配置文件仅记录路径。')
        mode.addItem('粘贴密钥', 'inline')
        if self.existing and (provider.get('api_key_file') or provider.get('api_key_env')):
            mode.addItem('保留当前凭据', 'keep')
        mode.addItem('环境变量', 'api_key_env')
        mode.addItem('密钥文件', 'api_key_file')
        mode.addItem('无需密钥', '')
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
            placeholders = {'inline':'粘贴服务密钥', 'keep':'保留当前凭据',
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
            self.image_options_toggle.setText('图像参数' if expanded else '图像参数')
            self.image_options_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        expanded = any(edit.text().strip() for edit in self.image_options.values())
        self.image_options_toggle.setChecked(expanded)
        self.image_options_toggle.toggled.connect(show_parameters)
        show_parameters(expanded)
        self.image_options_toggle.setToolTip('仅填写服务支持的参数；收起保留已填值，留空不发送。')
        form.addRow(options)

    def update_navigation(self):
        index = self.tabs.currentIndex()
        self.steps.setCurrentRow(index)
        self.message.clear()

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

    def use_prompt_service(self):
        try:
            provider, secret = self.provider_settings('llm')
            url = urlsplit(provider['url'])
            if not url.path.endswith('/chat/completions'):
                raise ValueError('摄影助手使用自定义路径，请分别填写完整接口地址')
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
            self.model_pickers['image'].schedule()
            self.message.setText('已沿用连接，请选择图像模型')
        except ValueError as error:
            self.message.setText(str(error))

    def choose_file(self, edit, title, pattern):
        filename, _ = QFileDialog.getOpenFileName(self, title, edit.text(), pattern)
        if filename:
            edit.setText(filename)
            for service, (_mode, value) in self.credentials.items():
                if value is edit:
                    self.model_pickers[service].schedule()

    def choose_output(self):
        directory = QFileDialog.getExistingDirectory(self, '选择照片结果保存文件夹', self.output_root.text())
        if directory:
            self.output_root.setText(directory)

    def reject(self):
        for mode, edit in self.credentials.values():
            if mode.currentData() == 'inline':
                edit.clear()
        super().reject()

    def done(self, result):
        for picker in self.model_pickers.values():
            picker.cancel()
        super().done(result)

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
            if self.relight_group.isChecked():
                lighting = {**RELIGHT_DEFAULTS, **values.get('relight',{})}
                for key,field in self.relight_fields.items():
                    value = field.text().strip()
                    if not value:
                        self.tabs.setCurrentIndex(2)
                        field.setFocus()
                        raise ValueError('请填写 PIXLRelight 服务地址和实际模型目录')
                    lighting[key] = endpoint_url(value,'relight') if key=='url' else value
                if self.relight_device.currentIndex() >= 0:
                    lighting['device'] = self.relight_device.currentText()
                values['relight'] = lighting
            else:
                values.pop('relight',None)
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
