"""In-page, optional face beauty controls for an already completed photograph."""
from __future__ import annotations

import json
import hashlib
import os
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import uuid
from urllib.parse import urlsplit

from PySide6.QtCore import QPointF, QRectF, QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QFrame, QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QSplitter,
    QMenu, QPlainTextEdit, QScrollArea, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from beauty_chat import discuss_beauty
from face_beauty import (BEAUTY_DEFAULTS, BeautyParameters, BeautySession,
    prepare_face, prepare_skin, render_beauty_crop, save_beauty)
from image_io import display_image, image_size, open_display, open_photo, save_json
from settings import load_settings
from model_downloads import ModelDownloads
from studio_widgets import PhotoCanvas, PhotoPreview, StatusText, chat_shortcuts, chat_composer, ConversationView, ui_label


class FaceRangeCanvas(PhotoCanvas):
    selected = Signal()

    def __init__(self):
        super().__init__()
        self.setDragMode(QGraphicsView.NoDrag)
        self.setCursor(Qt.CrossCursor)
        self.anchor = None
        self.selection = self.scene().addRect(QRectF(),QPen(QColor('#50d9c9'),2))
        self.selection.setZValue(1)
        self.setToolTip('拖动框选人脸；滚轮缩放，双击适应窗口。')

    def image_point(self, event):
        point = self.mapToScene(event.position().toPoint())
        bounds = self.photo.boundingRect()
        return QPointF(max(0,min(bounds.width(),point.x())),max(0,min(bounds.height(),point.y())))

    def mousePressEvent(self, event):
        if event.button()==Qt.LeftButton:
            self.anchor = self.image_point(event)
            self.selection.setRect(QRectF(self.anchor,self.anchor))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.anchor is not None and event.buttons()&Qt.LeftButton:
            self.selection.setRect(QRectF(self.anchor,self.image_point(event)).normalized())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button()==Qt.LeftButton and self.anchor is not None:
            self.selection.setRect(QRectF(self.anchor,self.image_point(event)).normalized())
            self.anchor = None
            self.selected.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class FaceRangeDialog(QDialog):
    def __init__(self, context, parent=None, *, initial=None):
        super().__init__(parent)
        self.setWindowTitle('选择本张人脸范围')
        self.resize(850,740)
        layout = QVBoxLayout(self)
        note = ui_label('在整张成片上框住眉毛到下巴及左右脸颊，留少量边缘；可先滚轮放大再拖动。头饰和大面积头发不必框入。')
        note.setWordWrap(True)
        layout.addWidget(note)
        native = image_size(context['base'])
        display = open_display(context['base'],max_edge=2000)
        self.scale = (native[0]/display.width(),native[1]/display.height())
        # The canvas is the whole completed photograph, shown only shrunk. The
        # range is therefore not limited by the head box of an earlier run.
        self.offset = (0,0)
        self.canvas = FaceRangeCanvas()
        self.canvas.setPixmap(QPixmap.fromImage(display))
        layout.addWidget(self.canvas,1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        self.analyze = buttons.button(QDialogButtonBox.Ok)
        self.analyze.setText('分析选定范围')
        self.analyze.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.canvas.selected.connect(self.selection_changed)
        if initial is not None:
            left,top,right,bottom = initial
            self.canvas.selection.setRect(QRectF(
                left/self.scale[0]-self.offset[0],top/self.scale[1]-self.offset[1],
                (right-left)/self.scale[0],(bottom-top)/self.scale[1]))
            self.selection_changed()
        layout.addWidget(buttons)

    def selection_changed(self):
        self.analyze.setEnabled(not self.canvas.selection.rect().isEmpty())

    def region(self):
        rect = self.canvas.selection.rect()
        return [round((value+offset)*scale) for value,offset,scale in zip(
            (rect.left(),rect.top(),rect.right(),rect.bottom()),self.offset*2,self.scale*2)]


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
                ('landmark_model','FGTBT 关键点模型'),
                ('model','完整 Kontext 模型目录'),('lora','BeautyGRPO LoRA')):
            field = QLineEdit(str(values[key]))
            form.addRow(label,field)
            self.fields[key] = field
        self.device = QComboBox()
        for device in ('cuda:0','cpu'):
            self.device.addItem(device)
        self.device.setCurrentText(values['analysis_device'])
        form.addRow('皮肤与关键点分析设备',self.device)
        self.memory = QComboBox()
        self.memory.addItem('NF4 底座 · 完整 LoRA / BF16 文本与 VAE','bnb_nf4')
        self.memory.addItem('完整 BF16 · 全部放入显存','cuda')
        self.memory.setCurrentIndex(self.memory.findData(values['memory']))
        form.addRow('美肤内存布局',self.memory)
        layout.addLayout(form)
        self.message = ui_label('Kontext 目录包含完整未量化的 Diffusers 模型、编码器、VAE 和调度器；美肤使用 cuda:0，内存布局由你明确选择。')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        downloads = QPushButton('下载美化模型与查看安装方法')
        downloads.clicked.connect(lambda: ModelDownloads(self, model='kontext').exec())
        layout.addWidget(downloads)
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

    def __init__(self, phase, *, context, config, session=None, pixels=None, parameters=None, before=None,
            llm_config=None, inputs=None, directory=None):
        super().__init__()
        self.phase, self.context, self.config = phase,context,config
        self.session,self.pixels,self.parameters = session,pixels,parameters
        self.before = before
        self.llm_config = llm_config
        self.inputs,self.directory = inputs,directory

    def run(self):
        try:
            if self.phase == 'analysis':
                session,pixels = prepare_face(self.context['base'],self.context['analysis_region'],
                    self.context['directory'],self.config,self.progress.emit)
                result = (session,pixels,display_image(pixels))
            elif self.phase == 'resume':
                session = BeautySession(self.context['session'])
                pixels = open_photo(self.context['base'])
                result = (session,pixels,display_image(pixels))
            elif self.phase == 'skin':
                result = prepare_skin(self.session,self.config,self.progress.emit,pixels=self.pixels,
                    llm_config=self.llm_config)
            elif self.phase == 'chat':
                result = discuss_beauty(self.pixels,self.session,config=self.llm_config,
                    directory=self.directory,**self.inputs)
            elif self.phase == 'render':
                result = self.before
                if self.parameters.skin or self.parameters.slim:
                    crop = render_beauty_crop(self.pixels,self.session,self.parameters)
                    result = self.before.copy()
                    painter = QPainter(result)
                    painter.drawImage(self.session.box[0],self.session.box[1],display_image(crop))
                    painter.end()
            elif self.phase == 'save':
                self.progress.emit('保存 16 位 PNG 和 JPEG 美化版')
                result = save_beauty(self.pixels,self.session,self.parameters,
                    parent_version=self.context.get('parent_version'))
            self.ready.emit(result)
        except Exception as error:
            self.failed.emit((getattr(error,'provider_message','') or str(error))+
                '。工作文件：'+str(self.context['directory']))


class BeautyEditor(QWidget):
    back_requested = Signal()
    saved = Signal(dict)
    busy_changed = Signal()
    connection_requested = Signal()

    def __init__(self, context, config, parent=None, *, llm_config=None):
        super().__init__(parent)
        self.context, self.config = dict(context),dict(config)
        self.llm_config = llm_config
        key = os.path.relpath(context.get('edit_id',context['base']),context['directory'])
        identifier = hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]
        self.draft_directory = Path(context['directory'])/'beauty-drafts'/identifier
        self.session = None
        self.pixels = None
        self.before = None
        self.worker = None
        self.pending = None
        self.rendered = BeautyParameters()
        self.failed = False
        self.failure_message = ''
        self.needs_restore = True
        self.restored = BeautyParameters()
        self.messages = []
        self.pending_chat = None
        self.pending_save = False
        self.operation_message = None
        self.discussion_timer = QTimer(self)
        self.discussion_timer.setSingleShot(True)
        self.discussion_timer.setInterval(250)
        self.discussion_timer.timeout.connect(self.save_draft)
        self.draft_timer = QTimer(self)
        self.draft_timer.setSingleShot(True)
        self.draft_timer.setInterval(150)
        self.draft_timer.timeout.connect(self.request_preview)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        header = QHBoxLayout()
        header.addWidget(ui_label('人脸美化 · '+context['name'],'heading'),1)
        more = QToolButton()
        more.setText('更多')
        more.setProperty('kind','quiet')
        more.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(more)
        self.connection_button = menu.addAction('连接设置',self.connection_requested.emit)
        more.setMenu(menu)
        header.addWidget(more)
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
        panel = QWidget()
        side = QVBoxLayout(panel)
        side.setContentsMargins(0,0,0,0)
        self.editor_tabs = QTabWidget()
        manual = QWidget()
        controls = QVBoxLayout(manual)
        controls.setContentsMargins(14,14,14,14)
        controls.setSpacing(12)
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
        self.skin_button = QPushButton('生成美肤基础')
        self.skin_button.setEnabled(False)
        self.skin_button.clicked.connect(lambda:self.start('skin'))
        controls.addWidget(self.skin_button)
        self.source_note = ui_label('未生成美肤基础')
        self.source_note.setWordWrap(True)
        controls.addWidget(self.source_note)
        self.raw_button = menu.addAction('查看模型原始输出',self.open_raw)
        self.raw_button.setEnabled(False)
        self.reset_button = QPushButton('全部归零')
        self.reset_button.setProperty('kind','quiet')
        self.reset_button.clicked.connect(self.reset_parameters)
        controls.addWidget(self.reset_button)
        controls.addStretch(1)
        self.chat_panel = QWidget()
        chat_column = QVBoxLayout(self.chat_panel)
        chat_column.setContentsMargins(0,0,0,0)
        self.chat_history = ConversationView()
        chat_column.addWidget(self.chat_history,1)
        self.chat_input = QPlainTextEdit()
        self.chat_input.setPlaceholderText('输入美化要求')
        self.chat_input.setFixedHeight(54)
        self.chat_input.textChanged.connect(self.discussion_timer.start)
        self.chat_send = QPushButton('发送')
        self.chat_send.clicked.connect(self.send_message)
        self.send_shortcuts = chat_shortcuts(self.chat_input,self.send_message)
        chat_column.addWidget(chat_composer(self.chat_input,self.chat_send))
        self.editor_tabs.addTab(self.chat_panel,'对话')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(manual)
        self.editor_tabs.addTab(scroll,'手动调整')
        side.addWidget(self.editor_tabs,1)
        self.analyze_button = QPushButton('选择人脸并分析')
        self.analyze_button.setProperty('kind','quiet')
        self.analyze_button.clicked.connect(self.choose_face)
        side.addWidget(self.analyze_button)
        self.save_button = QPushButton('另存美化版')
        self.save_button.setProperty('kind','primary')
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(lambda:self.start('save'))
        side.addWidget(self.save_button)
        panel.setMinimumWidth(290)
        panel.setMaximumWidth(360)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0,1)
        splitter.setSizes([900,320])
        layout.addWidget(splitter,1)
        self.message = StatusText('请选择人脸')
        self.message.setMaximumHeight(48)
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.message)
        self.restore_discussion()
        self.editor_tabs.currentChanged.connect(lambda _index:self.discussion_timer.start())

    def is_busy(self):
        return self.worker is not None

    def parameters(self):
        return BeautyParameters(skin=self.sliders['skin'].value()/100,
            slim=self.sliders['slim'].value()/100,protect_features=self.protect_features.isChecked())

    def refresh_discussion(self):
        entries = []
        for row in self.messages:
            label = ''
            if row.get('context'):
                mode = {'source':'美化前','output':'美化后','compare':'左右比较'}[row['context']['view_mode']]
                values = row['context']['parameters']
                label = f'{mode} · 美肤 {round(values["skin"]*100)} / 瘦脸 {round(values["slim"]*100)}'
            entries.append(dict(row,context={'label':label} if label else None))
        self.chat_history.show_messages(entries)

    def restore_discussion(self):
        path = self.draft_directory/'discussion.json'
        if not path.is_file():
            return
        try:
            state = json.loads(path.read_text(encoding='utf-8'))
            self.messages = state['messages']
            self.chat_input.setPlainText(state['input'])
            self.editor_tabs.setCurrentIndex(state.get('tab',0))
            self.refresh_discussion()
        except (OSError,ValueError,KeyError) as error:
            self.message.setText('读取美化讨论失败：'+str(error))

    def chat_context(self):
        return {'base':self.context['base'],'session':str(self.session.directory) if self.session else None,
            'face_box':list(self.session.box) if self.session else None,
            'analysis_region':list(self.session.analysis_region) if self.session else None,
            'skin_ready':self.session is not None and self.session.delta is not None,
            'parameters':asdict(self.parameters()),'preview_parameters':asdict(self.rendered),
            'view_mode':self.preview.mode.currentData()}

    def send_message(self):
        text = self.chat_input.toPlainText().strip()
        if not text or self.is_busy():
            return
        self.draft_timer.stop()
        self.pending = self.parameters() if self.parameters()!=self.rendered else None
        context = self.chat_context()
        self.messages.append({'role':'user','content':text,'context':context})
        self.chat_input.clear()
        self.refresh_discussion()
        if self.save_draft():
            directory = Path(self.context['directory'])/('beauty-chat-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
            self.start('chat',inputs={'context':context,'conversation':deepcopy(self.messages)},directory=directory)

    def operation_status(self, text, *, terminal=False):
        if self.operation_message is not None:
            self.operation_message['operation_status'] = text
            self.refresh_discussion()
            self.save_draft()
            if terminal:
                self.operation_message = None

    def execute_chat(self, row):
        operation = row['operation']
        self.operation_message = row
        self.set_parameters(BeautyParameters(**operation['parameters']))
        self.pending_save = operation['save']
        self.pending = None
        if not self.save_draft():
            self.pending_save = False
            self.operation_status(self.message.text(),terminal=True)
            return
        if operation['action']=='generate_skin':
            self.operation_status('正在生成美肤基础')
            self.start('skin')
        elif self.pending_save:
            self.pending_save = False
            self.operation_status('正在另存美化版')
            self.start('save')
        elif self.isVisible():
            self.operation_status('正在更新美化预览')
            self.start('render')
        else:
            self.operation_status('美化参数已存草稿，返回时显示',terminal=True)
            self.release_pixels()

    def restore_or_prepare(self):
        if not self.isVisible() or self.worker is not None or not self.needs_restore:
            return
        self.needs_restore = False
        path = self.draft_directory/'session.json'
        try:
            if path.is_file():
                state = json.loads(path.read_text(encoding='utf-8'))
                self.context['session'] = str(Path(self.context['directory'])/state['session'])
                self.restored = BeautyParameters(**state['parameters'])
                self.start('resume')
            elif self.context.get('session'):
                self.restored = BeautyParameters(**self.context['beauty_parameters'])
                self.start('resume')
            else:
                self.message.setText('点击“选择人脸并分析”，框住本张脸部后开始。')
        except (OSError,ValueError,KeyError) as error:
            self.message.setText('读取美化草稿失败：'+str(error))

    def choose_face(self):
        if self.worker is not None:
            return
        try:
            initial = self.session.analysis_region if self.session is not None else None
            dialog = FaceRangeDialog(self.context,self,initial=initial)
            if dialog.exec()==QDialog.Accepted:
                self.context['analysis_region'] = dialog.region()
                self.start('analysis')
        except (OSError,ValueError) as error:
            self.message.setText('选择人脸范围失败：'+str(error))

    def start(self, phase, *, inputs=None, directory=None):
        if self.worker is not None:
            return
        self.draft_timer.stop()
        self.needs_restore = False
        if phase!='chat':
            self.failed = False
        if phase not in ('resume','chat'):
            self.failure_message = ''
        worker = BeautyWorker(phase, context=self.context,config=self.config,
            session=self.session,pixels=self.pixels,parameters=self.parameters(),before=self.before,
            llm_config=self.llm_config,inputs=inputs,directory=directory)
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
            self.pending = parameters if parameters != self.rendered and not self.failure_message else None
            if self.failure_message:
                self.failed = True
                self.message.setText(self.failure_message+'\n已载入基础成片；调整参数后更新预览。')
            else:
                self.message.setText('人脸已就绪')
            self.save_draft()
        elif phase == 'skin':
            self.session = result
            self.pending = self.parameters()
            self.save_draft()
            self.message.setText('美肤基础已生成')
        elif phase == 'render':
            if self.worker.parameters == self.parameters():
                self.preview.set_image_pair(self.before,result)
                self.rendered = self.worker.parameters
                self.message.setText('预览已更新')
            self.operation_status('美化预览已更新，尚未另存',terminal=True)
        elif phase == 'chat':
            row = {'role':'assistant','content':result['reply']}
            if result['operation'] is not None:
                row.update(operation=result['operation'],operation_status='操作已登记')
            self.messages.append(row)
            self.refresh_discussion()
            if self.save_draft() and result['operation'] is not None:
                self.pending_chat = row
            self.message.setText(result['reply'])
        elif phase == 'save':
            self.save_draft()
            self.message.setText('美化版已另存为 16 位 PNG 和 JPEG。')
            self.operation_status('已另存美化版：'+result['files']['final'],terminal=True)
            self.saved_result = result

    def action_failed(self, error):
        self.pending_save = False
        self.pending_chat = None
        if self.worker.phase=='chat':
            self.messages.append({'role':'assistant','content':'本次美化对话失败：'+error})
            self.refresh_discussion()
            self.save_draft()
            self.message.setText('本次美化对话失败：'+error)
            return
        self.failed = True
        self.pending = None
        self.failure_message = '人脸美化失败：'+error
        self.message.setText(self.failure_message+'。当前画面保留上次成功预览。')
        self.operation_status(self.failure_message,terminal=True)

    def action_finished(self):
        worker,self.worker = self.worker,None
        worker.pixels = worker.session = worker.before = None
        worker.deleteLater()
        if worker.phase=='save' and not self.failed:
            result = self.saved_result
            del self.saved_result
            self.saved.emit(result)
        if self.pending_chat is not None:
            row,self.pending_chat = self.pending_chat,None
            self.execute_chat(row)
            if self.worker is not None:
                return
        if worker.phase=='skin' and not self.failed and self.operation_message is not None:
            if self.pending_save:
                self.pending_save = False
                self.pending = None
                self.operation_status('美肤基础已生成，正在另存美化版')
                self.start('save')
                return
            self.operation_status('美肤基础已生成，尚未另存',
                terminal=not self.isVisible() or self.pending==self.rendered)
        if not self.isVisible():
            self.release_pixels()
            self.update_controls()
            self.busy_changed.emit()
            return
        if self.pending is not None and self.pending!=self.rendered and not self.failed:
            self.pending = None
            self.start('render')
            return
        self.pending = None
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
        if self.session is None or not self.isVisible():
            return
        if self.worker is not None:
            self.pending = self.parameters()
        else:
            self.start('render')

    def save_draft(self):
        try:
            self.draft_directory.mkdir(parents=True,exist_ok=True)
            save_json(self.draft_directory/'discussion.json',
                {'base':self.context['base'],'messages':self.messages,'input':self.chat_input.toPlainText(),
                    'tab':self.editor_tabs.currentIndex()})
            if self.session is not None:
                save_json(self.draft_directory/'session.json',
                    {'session':self.session.directory.name,'parameters':asdict(self.parameters()),'base':self.context['base']})
        except OSError as error:
            self.message.setText('保存美化草稿失败：'+str(error))
            return False
        return True

    def release_pixels(self):
        self.draft_timer.stop()
        self.discussion_timer.stop()
        self.pending = None
        if not self.save_draft():
            return
        if self.worker is not None:
            return
        self.session = self.pixels = self.before = None
        self.needs_restore = True
        self.preview.release_images()
        self.update_controls()

    def hideEvent(self, event):
        self.release_pixels()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if self.pixels is not None:
            self.preview.set_image_pair(self.before,self.before)
            self.rendered = BeautyParameters()
            if not self.failed:
                self.request_preview()
        QTimer.singleShot(0,self.restore_or_prepare)

    def update_controls(self):
        busy = self.worker is not None
        rendering = busy and self.worker.phase == 'render' and self.operation_message is None
        self.sliders['slim'].setEnabled(self.session is not None and (not busy or rendering))
        self.sliders['skin'].setEnabled(self.session is not None and self.session.delta is not None and (not busy or rendering))
        self.protect_features.setEnabled(self.session is not None and self.session.delta is not None and (not busy or rendering))
        self.skin_button.setEnabled(self.session is not None and not busy)
        self.skin_button.setText('重新生成美肤基础' if self.session is not None and self.session.delta is not None else '生成美肤基础')
        self.raw_button.setEnabled(self.session is not None and self.session.delta is not None)
        if self.session is not None and self.session.delta is not None:
            self.source_note.setText('已有美肤基础')
        else:
            self.source_note.setText('未生成美肤基础')
        self.analyze_button.setEnabled(not busy)
        self.analyze_button.setText('选择人脸并分析' if self.session is None else '重新选择人脸')
        self.connection_button.setEnabled(not busy)
        self.reset_button.setEnabled(self.session is not None and (not busy or rendering))
        parameters = self.parameters()
        self.save_button.setEnabled(self.pixels is not None and not busy and
            parameters == self.rendered and bool(parameters.skin or parameters.slim))
        self.chat_input.setEnabled(not busy)
        self.chat_send.setEnabled(not busy)
