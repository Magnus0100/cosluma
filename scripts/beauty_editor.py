"""In-page, optional face beauty controls for an already completed photograph."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from PySide6.QtCore import QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QSplitter,
    QScrollArea, QVBoxLayout, QWidget)

from face_beauty import (BEAUTY_DEFAULTS, BeautyParameters, BeautySession,
    prepare_face, prepare_skin, render_beauty, save_beauty)
from image_io import display_image, open_photo, save_json
from settings import load_settings
from studio_widgets import PhotoPreview, ui_label


class BeautyConnection(QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.config = load_settings(path)
        values = dict(BEAUTY_DEFAULTS)
        values['url'] = self.config.get('head',{}).get('url', values['url'])
        values.update(self.config.get('beauty',{}))
        self.setWindowTitle('人脸美化连接')
        self.resize(640, 480)
        layout = QVBoxLayout(self)
        layout.addWidget(ui_label('连接美化服务', 'heading'))
        note = ui_label('使用部署了 CosLuma 人脸分析节点和美肤模型的 ComfyUI。保存连接后，由你主动生成人脸与美肤基础。')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.fields = {}
        for key,label in (('url','ComfyUI 地址'),('analysis_model','FaceXFormer 模型'),
                ('landmark_model','面部定位模型'),('model','完整 Kontext 模型目录'),('lora','BeautyGRPO LoRA')):
            field = QLineEdit(str(values[key]))
            form.addRow(label,field)
            self.fields[key] = field
        self.device = QComboBox()
        for device in ('cuda:0','cpu'):
            self.device.addItem(device)
        self.device.setCurrentText(values['analysis_device'])
        form.addRow('皮肤分析设备',self.device)
        self.memory = QComboBox()
        self.memory.addItem('完整 BF16 · 分组卸载到内存','group_offload')
        self.memory.addItem('完整 BF16 · 全部放入显存','cuda')
        self.memory.setCurrentIndex(self.memory.findData(values['memory']))
        form.addRow('美肤内存布局',self.memory)
        layout.addLayout(form)
        self.message = ui_label('Kontext 目录包含完整未量化的 Diffusers 模型、编码器、VAE 和调度器；美肤使用 cuda:0，内存布局由你明确选择。')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def save(self):
        try:
            values = {key:field.text().strip() for key,field in self.fields.items()}
            parsed = urlsplit(values['url'])
            if parsed.scheme not in ('http','https') or not parsed.hostname:
                raise ValueError('请填写 ComfyUI 的 HTTP/HTTPS 根地址')
            if any(not value for value in values.values()):
                raise ValueError('请填写实际模型文件名')
            values['url'] = values['url'].rstrip('/')
            values['analysis_device'] = self.device.currentText()
            values['inference_device'] = 'cuda:0'
            values['memory'] = self.memory.currentData()
            self.config['beauty'] = values
            save_json(self.path,self.config)
            self.accept()
        except (OSError,ValueError,KeyError) as error:
            self.message.setText('保存失败：'+str(error))


class BeautyWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, phase, *, context, config, session=None, pixels=None, parameters=None):
        super().__init__()
        self.phase, self.context, self.config = phase,context,config
        self.session,self.pixels,self.parameters = session,pixels,parameters

    def run(self):
        try:
            if self.phase == 'analysis':
                session,pixels = prepare_face(self.context['base'],self.context['region'],
                    self.context['directory'],self.config,self.progress.emit)
                result = (session,pixels,display_image(pixels))
            elif self.phase == 'resume':
                session = BeautySession(self.context['session'])
                pixels = open_photo(self.context['base'])
                result = (session,pixels,display_image(pixels))
            elif self.phase == 'skin':
                result = prepare_skin(self.session,self.config,self.progress.emit)
            elif self.phase == 'render':
                result = display_image(render_beauty(self.pixels,self.session,self.parameters))
            elif self.phase == 'save':
                self.progress.emit('保存 16 位 PNG 和 JPEG 美化版')
                result = save_beauty(self.pixels,self.session,self.parameters)
            self.ready.emit(result)
        except Exception as error:
            self.failed.emit((getattr(error,'provider_message','') or str(error))+
                '。工作文件：'+str(self.context['directory']))


class BeautyEditor(QWidget):
    back_requested = Signal()
    saved = Signal(dict)
    busy_changed = Signal()
    connection_requested = Signal()

    def __init__(self, context, config, parent=None):
        super().__init__(parent)
        self.context, self.config = dict(context),dict(config)
        self.session = None
        self.pixels = None
        self.before = None
        self.worker = None
        self.pending = None
        self.rendered = BeautyParameters()
        self.failed = False
        self.restored = BeautyParameters()
        self.draft_timer = QTimer(self)
        self.draft_timer.setSingleShot(True)
        self.draft_timer.setInterval(150)
        self.draft_timer.timeout.connect(self.request_preview)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        header = QHBoxLayout()
        header.addWidget(ui_label('人脸美化 · '+context['name'],'heading'),1)
        self.connection_button = QPushButton('美化连接')
        self.connection_button.setProperty('kind','quiet')
        self.connection_button.clicked.connect(self.connection_requested.emit)
        header.addWidget(self.connection_button)
        back = QPushButton('返回成片')
        back.setProperty('kind','quiet')
        back.clicked.connect(self.back_requested.emit)
        header.addWidget(back)
        layout.addLayout(header)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.preview = PhotoPreview(source_label='美化前',output_label='美化后')
        self.preview.set_photo(context['base'])
        splitter.addWidget(self.preview)
        panel = QFrame()
        panel.setProperty('role','panel')
        controls = QVBoxLayout(panel)
        controls.setContentsMargins(18,18,18,18)
        controls.setSpacing(14)
        controls.addWidget(ui_label('按自己的审美调整','heading'))
        note = ui_label('默认不改变原成片。滑杆归零可还原，满意后另存美化版。')
        note.setWordWrap(True)
        controls.addWidget(note)
        self.sliders = {}
        self.values = {}
        for key,label in (('skin','美肤'),('slim','瘦脸')):
            row = QHBoxLayout()
            row.addWidget(ui_label(label,'heading'),1)
            value = ui_label('0')
            row.addWidget(value)
            controls.addLayout(row)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0,100)
            slider.setEnabled(False)
            slider.valueChanged.connect(self.parameters_changed)
            self.sliders[key],self.values[key] = slider,value
            controls.addWidget(slider)
            if key=='skin':
                self.protect_features = QCheckBox('美肤保护五官')
                self.protect_features.setChecked(True)
                self.protect_features.setToolTip('开启时仅调整皮肤，保留眼睛、眉毛与嘴唇；关闭后允许模型调整这些部位。瘦脸仍固定五官位置。')
                self.protect_features.toggled.connect(self.parameters_changed)
                controls.addWidget(self.protect_features)
                explanation = ui_label('关闭后允许美肤调整眼睛、眉毛和嘴唇；切换复用当前基础。')
                explanation.setWordWrap(True)
                controls.addWidget(explanation)
        self.skin_button = QPushButton('生成美肤基础')
        self.skin_button.setEnabled(False)
        self.skin_button.clicked.connect(lambda:self.start('skin'))
        controls.addWidget(self.skin_button)
        self.source_note = ui_label('生成完整 BF16 模型输出后，再调整应用强度。模型原始输出单独保留，可直接查看。')
        self.source_note.setWordWrap(True)
        controls.addWidget(self.source_note)
        self.raw_button = QPushButton('查看模型原始输出')
        self.raw_button.setProperty('kind','quiet')
        self.raw_button.setEnabled(False)
        self.raw_button.clicked.connect(self.open_raw)
        controls.addWidget(self.raw_button)
        self.reset_button = QPushButton('全部归零')
        self.reset_button.setProperty('kind','quiet')
        self.reset_button.clicked.connect(self.reset_parameters)
        controls.addWidget(self.reset_button)
        controls.addStretch(1)
        self.analyze_button = QPushButton('重新分析人脸')
        self.analyze_button.setProperty('kind','quiet')
        self.analyze_button.clicked.connect(lambda:self.start('analysis'))
        controls.addWidget(self.analyze_button)
        self.save_button = QPushButton('另存美化版')
        self.save_button.setProperty('kind','primary')
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(lambda:self.start('save'))
        controls.addWidget(self.save_button)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(270)
        scroll.setMaximumWidth(330)
        scroll.setWidget(panel)
        splitter.addWidget(scroll)
        splitter.setSizes([900,280])
        layout.addWidget(splitter,1)
        self.message = ui_label('正在准备主体人脸，原成片保持可查看。')
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.message)
        QTimer.singleShot(0,self.restore_or_prepare)

    def is_busy(self):
        return self.worker is not None

    def parameters(self):
        return BeautyParameters(skin=self.sliders['skin'].value()/100,
            slim=self.sliders['slim'].value()/100,protect_features=self.protect_features.isChecked())

    def restore_or_prepare(self):
        path = Path(self.context['directory'])/'beauty-session.json'
        try:
            if path.is_file():
                state = json.loads(path.read_text(encoding='utf-8'))
                self.context['session'] = str(Path(self.context['directory'])/state['session'])
                self.restored = BeautyParameters(**state['parameters'])
                self.start('resume')
            else:
                self.start('analysis')
        except (OSError,ValueError,KeyError) as error:
            self.message.setText('读取美化草稿失败：'+str(error))

    def start(self, phase):
        if self.worker is not None:
            return
        self.draft_timer.stop()
        self.failed = False
        worker = BeautyWorker(phase, context=self.context,config=self.config,
            session=self.session,pixels=self.pixels,parameters=self.parameters())
        worker.ready.connect(self.action_ready)
        worker.failed.connect(self.action_failed)
        worker.progress.connect(self.message.setText)
        worker.finished.connect(self.action_finished)
        self.worker = worker
        self.update_controls()
        self.busy_changed.emit()
        worker.start()

    def action_ready(self, result):
        phase = self.worker.phase
        if phase in ('analysis','resume'):
            self.session,self.pixels,self.before = result
            self.preview.set_image_pair(self.before,self.before)
            parameters = self.restored if phase == 'resume' else BeautyParameters()
            self.set_parameters(parameters)
            self.rendered = BeautyParameters()
            self.pending = parameters if parameters != self.rendered else None
            self.message.setText('人脸已准备。调整瘦脸，或主动生成美肤基础。')
            self.save_draft()
        elif phase == 'skin':
            self.session = result
            if self.sliders['skin'].value() == 0:
                self.sliders['skin'].setValue(30)
            self.pending = self.parameters()
            self.save_draft()
            self.message.setText('美肤基础已生成，可以调整强度并对比。')
        elif phase == 'render':
            if self.worker.parameters == self.parameters():
                self.preview.set_image_pair(self.before,result)
                self.rendered = self.worker.parameters
                self.message.setText('预览已更新。满意后另存美化版；全部归零恢复原成片。')
        elif phase == 'save':
            self.save_draft()
            self.message.setText('美化版已另存为 16 位 PNG 和 JPEG。')
            self.saved.emit(result)

    def action_failed(self, error):
        self.failed = True
        self.pending = None
        self.message.setText('人脸美化失败：'+error+'。当前画面保留上次成功预览。')

    def action_finished(self):
        worker,self.worker = self.worker,None
        worker.deleteLater()
        if self.pending is not None and not self.failed:
            self.pending = None
            self.start('render')
            return
        self.update_controls()
        self.busy_changed.emit()

    def set_parameters(self, parameters):
        for key in ('skin','slim'):
            self.sliders[key].blockSignals(True)
            self.sliders[key].setValue(round(getattr(parameters,key)*100))
            self.sliders[key].blockSignals(False)
            self.values[key].setText(str(self.sliders[key].value()))
        self.protect_features.blockSignals(True)
        self.protect_features.setChecked(parameters.protect_features)
        self.protect_features.blockSignals(False)

    def reset_parameters(self):
        self.set_parameters(BeautyParameters(protect_features=self.protect_features.isChecked()))
        self.parameters_changed()

    def open_raw(self):
        if self.session is not None and self.session.state.get('skin_ready'):
            path = self.session.directory/self.session.state['skin_file']
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def parameters_changed(self):
        for key,slider in self.sliders.items():
            self.values[key].setText(str(slider.value()))
        self.save_button.setEnabled(False)
        self.draft_timer.start()

    def request_preview(self):
        self.save_draft()
        if self.session is None:
            return
        if self.worker is not None:
            self.pending = self.parameters()
        else:
            self.start('render')

    def save_draft(self):
        if self.session is None:
            return
        try:
            parameters = self.parameters()
            save_json(Path(self.context['directory'])/'beauty-session.json',
                {'session':self.session.directory.name,'parameters':{'skin':parameters.skin,'slim':parameters.slim,
                    'protect_features':parameters.protect_features}})
        except OSError as error:
            self.message.setText('保存美化草稿失败：'+str(error))

    def update_controls(self):
        busy = self.worker is not None
        rendering = busy and self.worker.phase == 'render'
        self.sliders['slim'].setEnabled(self.session is not None and (not busy or rendering))
        self.sliders['skin'].setEnabled(self.session is not None and self.session.delta is not None and (not busy or rendering))
        self.protect_features.setEnabled(self.session is not None and self.session.delta is not None and (not busy or rendering))
        self.skin_button.setEnabled(self.session is not None and not busy)
        self.skin_button.setText('重新生成美肤基础' if self.session is not None and self.session.delta is not None else '生成美肤基础')
        self.raw_button.setEnabled(self.session is not None and self.session.delta is not None)
        if self.session is not None and self.session.delta is not None:
            official = self.session.state.get('skin',{}).get('engine')=='diffusers.FluxKontextPipeline'
            self.source_note.setText('当前基础：官方 BF16 推理。滑杆调整应用强度；模型原始输出单独保留。' if official else
                '当前基础来自此前的推理配置。主动重新生成后使用官方 BF16 方法，旧文件保留。')
        else:
            self.source_note.setText('生成完整 BF16 模型输出后，再调整应用强度。模型原始输出单独保留，可直接查看。')
        self.analyze_button.setEnabled(not busy)
        self.connection_button.setEnabled(not busy)
        self.reset_button.setEnabled(self.session is not None and (not busy or rendering))
        parameters = self.parameters()
        self.save_button.setEnabled(not busy and parameters == self.rendered and bool(parameters.skin or parameters.slim))
