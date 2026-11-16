"""Manual and conversational tone controls sharing the same fixed image operations."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (QDoubleSpinBox, QFormLayout, QHBoxLayout, QPlainTextEdit,
    QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget)

from image_io import display_image, open_photo, save_json
from settings import load_settings
from studio_widgets import PhotoPreview, ui_label
from tone_edit import ToneParameters, discuss_tone, render_tone, save_tone


class ToneWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, phase, context, parameters, pixels=None, *, config_path=None, conversation=(), directory=None):
        super().__init__()
        self.phase,self.context,self.parameters,self.pixels = phase,context,parameters,pixels
        self.config_path,self.conversation,self.directory = config_path,conversation,directory

    def run(self):
        try:
            if self.phase=='load':
                pixels = open_photo(self.context['base'])
                self.ready.emit({'pixels':pixels,'before':display_image(pixels),
                    'after':display_image(render_tone(pixels,self.parameters))})
            elif self.phase=='render':
                self.ready.emit(display_image(render_tone(self.pixels,self.parameters)))
            elif self.phase=='save':
                pixels = self.pixels if self.pixels is not None else open_photo(self.context['base'])
                self.ready.emit(save_tone(pixels,self.parameters,self.context))
            else:
                config = load_settings(self.config_path)
                self.ready.emit(discuss_tone(render_tone(self.pixels,self.parameters),self.parameters,
                    self.conversation,config['llm'],self.directory))
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            self.pixels = None


class ToneEditor(QWidget):
    back_requested = Signal()
    saved = Signal(dict)
    busy_changed = Signal()
    external_changed = Signal()

    def __init__(self, context, config_path, parent=None):
        super().__init__(parent)
        self.context,self.config_path = context,config_path
        identifier = hashlib.sha256(context['base'].encode('utf-8')).hexdigest()[:16]
        self.directory = Path(context['directory'])/('tone-draft-'+identifier)
        self.draft_path = self.directory/'draft.json'
        self.worker = self.pixels = self.before = self.after = None
        self.rendered = None
        self.pending_save = False
        self.save_message = None
        self.external_message = None
        self.messages = []
        self.draft_timer = QTimer(self)
        self.draft_timer.setSingleShot(True)
        self.draft_timer.setInterval(250)
        self.draft_timer.timeout.connect(self.save_draft)
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(160)
        self.render_timer.timeout.connect(self.render)
        layout = QVBoxLayout(self)
        heading = QHBoxLayout()
        heading.addWidget(ui_label('曝光与明度曲线 · '+context['name']))
        heading.addStretch()
        back = QPushButton('返回作品')
        back.clicked.connect(self.back_requested.emit)
        heading.addWidget(back)
        layout.addLayout(heading)
        split = QSplitter(Qt.Horizontal)
        self.preview = PhotoPreview(source_label='编辑前版本',output_label='光调预览')
        split.addWidget(self.preview)
        panel = QWidget()
        column = QVBoxLayout(panel)
        note = ui_label('整图统一调整光调。参数相对本次打开的版本；恢复初始值可还原该版本像素。')
        note.setWordWrap(True)
        column.addWidget(note)
        form = QFormLayout()
        self.controls = {}
        self.sliders = {}
        for name,label,low,high,value,step in (
                ('exposure_ev','曝光（EV）',-2,2,0,.05),
                ('shadows','曲线：暗部 25% →',.05,.4,.25,.01),
                ('midtones','曲线：中间调 50% →',.4,.6,.5,.01),
                ('highlights','曲线：亮部 75% →',.6,.95,.75,.01)):
            control = QDoubleSpinBox()
            control.setRange(low,high)
            control.setDecimals(3)
            control.setSingleStep(step)
            control.setValue(value)
            control.valueChanged.connect(self.parameters_changed)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(round(low*1000),round(high*1000))
            slider.setValue(round(value*1000))
            slider.valueChanged.connect(lambda value,spin=control:spin.setValue(value/1000))
            control.valueChanged.connect(lambda value,bar=slider:bar.setValue(round(value*1000)))
            self.controls[name] = control
            self.sliders[name] = slider
            row = QHBoxLayout()
            row.addWidget(slider,1)
            row.addWidget(control)
            form.addRow(label,row)
        column.addLayout(form)
        buttons = QHBoxLayout()
        self.reset = QPushButton('恢复初始值')
        self.reset.clicked.connect(lambda:self.set_parameters(ToneParameters()))
        buttons.addWidget(self.reset)
        self.save = QPushButton('另存光调版')
        self.save.setProperty('kind','primary')
        self.save.clicked.connect(self.save_version)
        buttons.addWidget(self.save)
        column.addLayout(buttons)
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        column.addWidget(self.history,1)
        self.chat_input = QPlainTextEdit()
        self.chat_input.setPlaceholderText('例如：整体提亮半档，稍微压住亮部。满意后可说“另存这一版”。')
        self.chat_input.setMaximumHeight(90)
        self.chat_input.textChanged.connect(self.draft_timer.start)
        column.addWidget(self.chat_input)
        self.send = QPushButton('发送')
        self.send.clicked.connect(self.send_message)
        column.addWidget(self.send)
        split.addWidget(panel)
        split.setStretchFactor(0,3)
        split.setStretchFactor(1,1)
        split.setSizes([900,350])
        layout.addWidget(split,1)
        self.message = ui_label('手动调整无需模型连接；对话使用摄影 Agent 连接。')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.restore_draft()
        self.update_controls()

    def parameters(self):
        return ToneParameters(**{name:control.value() for name,control in self.controls.items()})

    def set_parameters(self, parameters):
        for name,value in parameters.record().items():
            self.controls[name].blockSignals(True)
            self.sliders[name].blockSignals(True)
            self.controls[name].setValue(value)
            self.sliders[name].setValue(round(value*1000))
            self.controls[name].blockSignals(False)
            self.sliders[name].blockSignals(False)
        self.parameters_changed()

    def parameters_changed(self):
        self.draft_timer.start()
        self.render_timer.start()

    def is_busy(self):
        return self.worker is not None

    def update_controls(self):
        busy = self.is_busy()
        editing = not busy or self.worker.phase=='render'
        for control in (*self.controls.values(),*self.sliders.values()):
            control.setEnabled(editing)
        self.reset.setEnabled(editing)
        self.save.setEnabled(not busy and self.pixels is not None)
        self.send.setEnabled(not busy and self.pixels is not None)
        self.chat_input.setEnabled(not busy)

    def start(self, phase, *, directory=None):
        self.worker_failed = False
        worker = ToneWorker(phase,self.context,self.parameters(),self.pixels,
            config_path=self.config_path,conversation=list(self.messages),directory=directory)
        self.worker = worker
        worker.ready.connect(self.ready)
        worker.failed.connect(self.failed)
        worker.finished.connect(self.finished)
        self.update_controls()
        self.busy_changed.emit()
        self.message.setText({'load':'正在读取已保存版本…','render':'正在更新光调预览…',
            'save':'正在从原尺寸浮点像素另存光调版…','chat':'摄影 Agent 正在理解光调要求…'}[phase])
        worker.start()

    def render(self):
        if self.isVisible() and not self.is_busy() and self.pixels is not None:
            self.start('render')

    def save_version(self):
        if not self.is_busy() and self.save_draft():
            self.save_message = None
            self.render_timer.stop()
            self.start('save')

    def accept_delivery(self, message, user_text):
        """Apply the delivery Agent's already parsed request without another model call."""
        operation = message['operation']
        self.external_message = message
        self.set_parameters(ToneParameters(**operation['parameters']))
        local = {'role':'assistant','content':message['content'],
            'operation':{'parameters':operation['parameters'],'save':operation['save']},
            'operation_status':'parameters_applied'}
        self.messages.extend([{'role':'user','content':user_text},local])
        self.pending_save = operation['save']
        self.save_message = local if self.pending_save else None
        self.refresh_history()
        if not self.save_draft():
            self.pending_save = False
            self.update_external(self.message.text(),terminal=True)
            return False
        self.update_external('已接收光调参数，正在处理')
        if self.isVisible() and not self.is_busy() and self.pixels is not None:
            self.render_timer.stop()
            if self.pending_save:
                self.pending_save = False
                self.start('save')
            else:
                self.start('render')
        return True

    def update_external(self, status, *, terminal=False):
        if self.external_message is not None:
            self.external_message['operation_status'] = status
            self.external_changed.emit()
            if terminal:
                self.external_message = None

    def send_message(self):
        text = self.chat_input.toPlainText().strip()
        if not text or self.is_busy():
            return
        self.render_timer.stop()
        self.messages.append({'role':'user','content':text})
        self.chat_input.clear()
        self.refresh_history()
        if self.save_draft():
            directory = self.directory/('chat-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
            self.start('chat',directory=directory)

    def ready(self, result):
        phase = self.worker.phase
        if phase=='load':
            self.pixels,self.before,self.after = result['pixels'],result['before'],result['after']
            self.rendered = self.worker.parameters
        elif phase=='render':
            self.after = result
            self.rendered = self.worker.parameters
        elif phase=='chat':
            message = {'role':'assistant','content':result['reply']}
            operation = result['operation']
            if operation is not None:
                message['operation'] = operation
            self.messages.append(message)
            self.refresh_history()
            if self.save_draft() and operation is not None:
                self.set_parameters(ToneParameters(**operation['parameters']))
                self.pending_save = operation['save']
                self.save_message = message if self.pending_save else None
                message['operation_status'] = 'parameters_applied'
                self.refresh_history()
                self.save_draft()
        else:
            self.saved_result = result
        if phase in ('load','render') and self.isVisible():
            self.preview.set_image_pair(self.before,self.after)
            self.message.setText('预览已更新；另存时从原尺寸浮点像素处理。')
        if phase in ('load','render') and self.worker.parameters==self.parameters():
            self.update_external('光调参数已应用，正在另存' if self.pending_save else '已更新光调预览，尚未另存',
                terminal=not self.pending_save)

    def failed(self, error):
        self.worker_failed = True
        self.pending_save = False
        self.update_external('光调失败：'+error,terminal=True)
        self.message.setText('光调'+self.worker.phase+'失败：'+error)
        if self.worker.phase=='chat':
            self.messages.append({'role':'assistant','content':'本次对话失败：'+error})
            self.refresh_history()
            self.save_draft()
        elif self.worker.phase=='save' and self.save_message is not None:
            self.save_message['operation_status'] = 'save_failed: '+error
            self.save_message = None
            self.refresh_history()
            self.save_draft()

    def finished(self):
        worker,self.worker = self.worker,None
        phase = worker.phase
        worker.deleteLater()
        if phase=='save' and not self.worker_failed:
            result = self.saved_result
            del self.saved_result
            self.update_external('已另存光调版：'+result['files']['final'],terminal=True)
            if self.save_message is not None:
                self.save_message['operation_status'] = 'saved: '+result['files']['final']
                self.save_message = None
                self.refresh_history()
                self.save_draft()
            self.saved.emit(result)
        elif not self.worker_failed and self.pending_save:
            self.pending_save = False
            self.render_timer.stop()
            self.start('save')
            return
        elif not self.worker_failed and self.isVisible() and self.pixels is not None and self.rendered!=self.parameters():
            self.render_timer.stop()
            self.start('render')
            return
        if not self.isVisible():
            self.release_pixels()
        self.update_controls()
        self.busy_changed.emit()

    def refresh_history(self):
        lines = []
        for row in self.messages:
            text = ('你：' if row['role']=='user' else '摄影 Agent：')+row['content']
            status = row.get('operation_status','')
            if status=='parameters_applied':
                text += '\n已应用光调参数。'
            elif status.startswith('saved: '):
                text += '\n已另存光调版：'+status[len('saved: '):]
            elif status.startswith('save_failed: '):
                text += '\n另存失败：'+status[len('save_failed: '):]
            lines.append(text)
        self.history.setPlainText('\n\n'.join(lines))
        bar = self.history.verticalScrollBar()
        bar.setValue(bar.maximum())

    def save_draft(self):
        try:
            self.directory.mkdir(parents=True,exist_ok=True)
            save_json(self.draft_path,{'base':self.context['base'],'parameters':self.parameters().record(),
                'messages':self.messages,'input':self.chat_input.toPlainText()})
            return True
        except OSError as error:
            self.message.setText('保存光调草稿失败：'+str(error))
            return False

    def restore_draft(self):
        if not self.draft_path.is_file():
            return
        try:
            state = json.loads(self.draft_path.read_text(encoding='utf-8'))
            self.set_parameters(ToneParameters(**state['parameters']))
            self.messages = state['messages']
            self.chat_input.setPlainText(state['input'])
            self.refresh_history()
        except (OSError,ValueError,KeyError,TypeError) as error:
            self.message.setText('读取光调草稿失败：'+str(error))

    def release_pixels(self):
        self.render_timer.stop()
        if self.is_busy():
            return
        self.pixels = self.before = self.after = self.rendered = None
        self.preview.release_images()

    def hideEvent(self, event):
        self.draft_timer.stop()
        self.save_draft()
        self.release_pixels()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if self.pixels is None and not self.is_busy():
            self.start('load')
