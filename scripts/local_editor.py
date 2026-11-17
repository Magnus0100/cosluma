"""Optional painted edits and explicit saved versions, after production."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QGraphicsView, QHBoxLayout, QLineEdit, QPlainTextEdit, QPushButton,
    QMenu, QScrollArea, QSlider, QSplitter, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from image_io import display_image, open_photo, save_image, save_json
from local_retouch import (H3_DEFAULTS, LOCAL_DEFAULTS, LOCAL_ENGINES, LOCAL_KEYS, TASKS,
    generate_local, local_engine, retouch_crop, save_local)
from local_chat import discuss_local
from model_downloads import ModelDownloads
from service_settings import endpoint_url
from settings import load_settings
from studio_widgets import PhotoCanvas, PhotoPreview, StatusText, chat_shortcuts, chat_composer, ConversationView, ui_label


def _pixmap(image):
    pixels = np.array(image.convert('RGBA'))
    return QPixmap.fromImage(QImage(pixels.data,image.width,image.height,pixels.strides[0],QImage.Format_RGBA8888).copy())


class MaskCanvas(PhotoCanvas):
    changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.mask = self.guide = self.protected = None
        self.tool = 'mask'
        self.brush = 36
        self.last = None
        self.overlay = self.scene().addPixmap(QPixmap())
        self.overlay.setZValue(1)
        self.marks = self.scene().addPixmap(QPixmap())
        self.marks.setZValue(2)
        self.setToolTip('滚轮缩放；移动查看可拖动。粉色涂选，蓝色保护，青色标记。发束造型与衣物按实际新旧轮廓写回。')

    def set_base(self, image, mask=None, guide=None, protected=None):
        image = image.scaled(1600,1600,Qt.KeepAspectRatio,Qt.SmoothTransformation) if max(image.width(),image.height())>1600 else image
        self.setPixmap(QPixmap.fromImage(image))
        size = (image.width(),image.height())
        self.mask = mask.resize(size,Image.Resampling.NEAREST) if mask else Image.new('L',size)
        self.guide = guide.resize(size,Image.Resampling.NEAREST) if guide else Image.new('RGBA',size)
        self.protected = protected.resize(size,Image.Resampling.NEAREST) if protected else Image.new('L',size)
        self.refresh_overlay()
        self.set_tool(self.tool)

    def set_tool(self, tool):
        self.tool = tool
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if tool=='move' else QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.OpenHandCursor if tool=='move' else Qt.CrossCursor)

    def refresh_overlay(self):
        if self.mask is None:
            return
        overlay = Image.new('RGBA',self.mask.size,(236,92,145,0))
        overlay.putalpha(self.mask.point(lambda value:round(value*.36)))
        protection = Image.new('RGBA',self.mask.size,(78,141,236,0))
        protection.putalpha(self.protected.point(lambda value:round(value*.44)))
        overlay.alpha_composite(protection)
        self.overlay.setPixmap(_pixmap(overlay))
        self.marks.setPixmap(_pixmap(self.guide))

    def paint_to(self, event):
        if self.mask is None:
            return
        point = self.mapToScene(event.position().toPoint())
        xy = (round(point.x()),round(point.y()))
        previous = self.last or xy
        if self.tool in ('mask','protect','erase_mask'):
            draw = ImageDraw.Draw(self.mask)
            color = 255 if self.tool=='mask' else 0
        else:
            draw = ImageDraw.Draw(self.guide)
            color = (53,220,204,255) if self.tool=='shape' else (0,0,0,0)
        draw.line((previous,xy),fill=color,width=self.brush)
        radius = self.brush/2
        draw.ellipse((xy[0]-radius,xy[1]-radius,xy[0]+radius,xy[1]+radius),fill=color)
        if self.tool in ('mask','protect','erase_mask'):
            exclusion = ImageDraw.Draw(self.protected)
            keep = 255 if self.tool=='protect' else 0
            exclusion.line((previous,xy),fill=keep,width=self.brush)
            exclusion.ellipse((xy[0]-radius,xy[1]-radius,xy[0]+radius,xy[1]+radius),fill=keep)
        self.last = xy
        self.refresh_overlay()

    def mousePressEvent(self, event):
        if event.button()==Qt.LeftButton and self.tool!='move' and self.mask is not None:
            self.last = None
            self.paint_to(event)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons()&Qt.LeftButton and self.tool!='move' and self.mask is not None:
            self.paint_to(event)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button()==Qt.LeftButton and self.tool!='move' and self.mask is not None:
            self.last = None
            self.changed.emit('mask' if self.tool in ('mask','protect','erase_mask') else 'guide')
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def clear_selection(self):
        if self.mask is not None:
            self.mask.paste(0,(0,0,*self.mask.size))
            self.protected.paste(0,(0,0,*self.protected.size))
            self.refresh_overlay()
            self.changed.emit('mask')

    def release(self):
        self.setText('打开局部修饰后读取已保存版本')
        self.overlay.setPixmap(QPixmap())
        self.marks.setPixmap(QPixmap())
        self.mask = self.guide = self.protected = None


class LocalConnection(QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.config = load_settings(path)
        stored = self.config.get('local_edit',{})
        fallback = self.config.get('head',{}).get('url',LOCAL_DEFAULTS['url'])
        self.engine = stored.get('engine') if stored.get('engine') in LOCAL_ENGINES else 'qwen'
        # Both engines keep their own values; only the selected one runs, and neither is rewritten.
        self.values = {'qwen':{key:stored.get(key,LOCAL_DEFAULTS[key]) for key in LOCAL_KEYS},
            'h3':{key:(stored.get('h3') or {}).get(key,H3_DEFAULTS[key]) for key in LOCAL_KEYS}}
        self.values['qwen']['url'] = stored.get('url',fallback)
        self.setWindowTitle('局部修饰连接')
        self.resize(680,470)
        layout = QVBoxLayout(self)
        note = ui_label('使用支持所选引擎的 ComfyUI。两个引擎分别保存地址与权重文件名，三个修饰任务共用；保存连接不开始生成。')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.engine_field = QComboBox()
        for key,engine in LOCAL_ENGINES.items():
            self.engine_field.addItem(engine['label'],key)
        self.engine_field.setCurrentIndex(self.engine_field.findData(self.engine))
        self.engine_field.currentIndexChanged.connect(self.change_engine)
        form.addRow('局部编辑引擎',self.engine_field)
        self.fields = {}
        for key,label in (('url','ComfyUI 地址'),('model','生成模型文件'),
                ('clip','文本编码器文件'),('vae','VAE 文件')):
            field = QLineEdit(str(self.values[self.engine][key]))
            self.fields[key] = field
            form.addRow(label,field)
        layout.addLayout(form)
        self.note = ui_label('')
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.show_note()
        download = QPushButton('模型下载与部署方法')
        download.clicked.connect(self.open_download)
        layout.addWidget(download)
        self.message = ui_label('')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def open_download(self):
        if self.engine=='h3':
            QDesktopServices.openUrl(QUrl(LOCAL_ENGINES['h3']['card']))
            return
        ModelDownloads(self,model='local').exec()

    def field_values(self):
        return {**self.values[self.engine],
            **{key:field.text().strip() for key,field in self.fields.items()}}

    def change_engine(self):
        self.values[self.engine] = self.field_values()
        self.engine = self.engine_field.currentData()
        for key,field in self.fields.items():
            field.setText(str(self.values[self.engine][key]))
        self.show_note()

    def show_note(self):
        values = self.values[self.engine]
        extra = {'qwen':'文本编码器在 CUDA 计算并卸载到 CPU，无损 KV 缓存放在内存，编码后释放文本编码器。',
            'h3':'步数由权重蒸馏决定，不符时服务端直接报错；提示词以 <Picture N> 指代输入图片，不使用提示词增强模型。'}[self.engine]
        self.note.setText(f"最低面积分辨率 {values['resolution']}、{values['steps']} 步、seed {values['seed']}。"+extra+
            '原生输出不足时使用“分割与超分”中已配置的模型超分一次，再缩小到修改区域。')

    def save(self):
        try:
            self.values[self.engine] = self.field_values()
            for key in ('url','model','clip','vae'):
                if not self.values[self.engine][key]:
                    raise ValueError('请填写实际服务地址和模型文件名')
            self.values['qwen']['url'] = endpoint_url(self.values['qwen']['url'],'local_edit')
            self.values['h3']['url'] = endpoint_url(self.values['h3']['url'],'local_edit')
            self.config['local_edit'] = {'engine':self.engine,
                **{key:self.values['qwen'][key] for key in LOCAL_KEYS},
                'h3':{key:self.values['h3'][key] for key in LOCAL_KEYS}}
            save_json(self.path,self.config)
            self.accept()
        except (OSError,ValueError) as error:
            self.message.setText('保存失败：'+str(error))


class LocalWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, phase, context, config, *, pixels=None, before=None, directory=None, inputs=None,
            sr_config=None, llm_config=None, head_config=None, library_path=None):
        super().__init__()
        self.phase,self.context,self.config = phase,context,config
        self.pixels,self.before,self.directory,self.inputs = pixels,before,directory,inputs
        self.sr_config = sr_config
        self.llm_config = llm_config
        self.head_config = head_config
        self.library_path = library_path

    def run(self):
        try:
            if self.phase=='load':
                pixels = open_photo(self.context['base'])
                result = (pixels,display_image(pixels))
            elif self.phase=='generate':
                directory,_ = generate_local(self.context['base'],self.context['directory'],
                    **self.inputs,config=self.config,progress=self.progress.emit,pixels=self.pixels,
                    parent_version=self.context.get('parent_version'),sr_config=self.sr_config,
                    llm_config=self.llm_config,head_config=self.head_config,library_path=self.library_path)
                result = self.comparison(directory)
            elif self.phase=='recompose':
                from local_retouch import revise_writeback
                self.progress.emit('按当前范围重新合成已有生成结果')
                result = self.comparison(revise_writeback(self.directory,self.inputs['mask'],
                    protected=self.inputs['protected']))
            elif self.phase=='compare':
                result = self.comparison(self.directory)
            elif self.phase=='save':
                self.progress.emit('另存原尺寸 16 位 PNG 与 JPEG')
                result = save_local(self.pixels,self.directory)
            elif self.phase=='chat':
                result = discuss_local(self.pixels,config=self.llm_config,directory=self.directory,**self.inputs)
            self.ready.emit(result)
        except Exception as error:
            self.failed.emit(str(error)+'。工作目录：'+self.context['directory'])

    def comparison(self, directory):
        crop,state = retouch_crop(self.pixels,directory)
        preview = self.before.copy()
        painter = QPainter(preview)
        painter.drawImage(state['box'][0],state['box'][1],display_image(crop))
        painter.end()
        return directory,preview,state


class LocalRetouchEditor(QWidget):
    back_requested = Signal()
    connection_requested = Signal()
    busy_changed = Signal()
    saved = Signal(dict)

    def __init__(self, context, config, parent=None, *, sr_config=None, llm_config=None, head_config=None,
            library_path=None):
        super().__init__(parent)
        self.context,self.config = context,config
        self.sr_config = sr_config
        self.llm_config = llm_config
        self.head_config = head_config
        self.library_path = library_path
        self.worker = None
        self.pixels = self.before = None
        self.result_directory = None
        self.writeback_dirty = False
        self.generation_dirty = False
        self.messages = []
        self.pending_chat = None
        self.pending_save = False
        self.operation_message = None
        self.failed = False
        self.failure_phase = None
        self.failure_message = ''
        self.restoring = False
        self.task_key = 'hair_repair'
        relative = os.path.relpath(context['base'],context['directory'])
        identifier = hashlib.sha256(relative.encode('utf-8')).hexdigest()[:12]
        self.draft_directory = Path(context['directory'])/'retouch-drafts'/identifier
        self.drafts = {}
        self.draft_timer = QTimer(self)
        self.draft_timer.setSingleShot(True)
        self.draft_timer.setInterval(250)
        self.draft_timer.timeout.connect(self.save_draft)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        header = QHBoxLayout()
        header.addWidget(ui_label('局部修饰 · '+context['name'],'heading'),1)
        more = QToolButton()
        more.setText('更多')
        more.setProperty('kind','quiet')
        more.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(more)
        self.connection = menu.addAction('连接设置',self.connection_requested.emit)
        more.setMenu(menu)
        header.addWidget(more)
        back = QPushButton('返回成片')
        back.clicked.connect(self.back_requested.emit)
        header.addWidget(back)
        layout.addLayout(header)
        splitter = QSplitter(Qt.Horizontal)
        self.pages = QTabWidget()
        self.canvas = MaskCanvas()
        self.pages.addTab(self.canvas,'涂画修改范围')
        self.preview = PhotoPreview(source_label='所选版本',output_label='本次修饰')
        self.pages.addTab(self.preview,'比较本次结果')
        self.pages.currentChanged.connect(self.page_changed)
        splitter.addWidget(self.pages)
        panel = QWidget()
        side = QVBoxLayout(panel)
        side.setContentsMargins(0,0,0,0)
        self.task = QComboBox()
        for key,(label,_) in TASKS.items():
            self.task.addItem(label,key)
        self.task.currentIndexChanged.connect(self.change_task)
        side.addWidget(self.task)
        self.note = ui_label('')
        self.note.setWordWrap(True)
        self.note.setMaximumHeight(42)
        side.addWidget(self.note)
        self.tool = QComboBox()
        for text,value in (('涂选修改范围','mask'),('保护区域','protect'),('擦除涂选与保护','erase_mask'),
                ('画形状标记','shape'),('擦除形状标记','erase_shape'),('移动查看','move')):
            self.tool.addItem(text,value)
        self.tool.currentIndexChanged.connect(lambda:self.canvas.set_tool(self.tool.currentData()))
        side.addWidget(self.tool)
        brush_row = QHBoxLayout()
        brush_row.addWidget(ui_label('画笔'))
        self.brush = QSlider(Qt.Horizontal)
        self.brush.setRange(4,180)
        self.brush.setValue(36)
        self.brush.valueChanged.connect(lambda value:setattr(self.canvas,'brush',value))
        brush_row.addWidget(self.brush,1)
        self.clear = QPushButton('清空修改范围')
        self.clear.clicked.connect(self.canvas.clear_selection)
        self.clear.setText('清空')
        self.clear.setProperty('kind','quiet')
        brush_row.addWidget(self.clear)
        side.addLayout(brush_row)
        self.editor_tabs = QTabWidget()
        manual = QWidget()
        controls = QVBoxLayout(manual)
        controls.addWidget(ui_label('具体要求','heading'))
        self.brief = QPlainTextEdit()
        self.brief.setPlaceholderText('可写部位、方向和幅度。')
        self.brief.setMaximumHeight(150)
        self.brief.textChanged.connect(self.generation_edited)
        controls.addWidget(self.brief)
        self.brief_hint = ui_label('简短描述希望得到的效果。')
        self.brief_hint.setWordWrap(True)
        self.brief_hint.setVisible(False)
        controls.addWidget(self.brief_hint)
        self.reference = QLineEdit()
        self.reference.setPlaceholderText('可选：外观参考图')
        self.reference.textChanged.connect(self.generation_edited)
        controls.addWidget(self.reference)
        self.reference_button = QPushButton('选择参考图')
        self.reference_button.clicked.connect(self.choose_reference)
        controls.addWidget(self.reference_button)
        note = ui_label('粉色涂选，蓝色保护，青色标记。发束造型与衣物按实际轮廓提取，蓝色区域保留。')
        note.setWordWrap(True)
        controls.addWidget(note)
        controls.addStretch(1)
        self.chat_panel = QWidget()
        chat_column = QVBoxLayout(self.chat_panel)
        chat_column.setContentsMargins(0,0,0,0)
        self.chat_history = ConversationView()
        chat_column.addWidget(self.chat_history,1)
        self.chat_input = QPlainTextEdit()
        self.chat_input.setPlaceholderText('输入修饰要求')
        self.chat_input.setFixedHeight(54)
        self.chat_input.textChanged.connect(self.discussion_edited)
        self.chat_send = QPushButton('发送')
        self.chat_send.clicked.connect(self.send_message)
        self.send_shortcuts = chat_shortcuts(self.chat_input,self.send_message)
        chat_column.addWidget(chat_composer(self.chat_input,self.chat_send))
        self.editor_tabs.addTab(self.chat_panel,'对话')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(manual)
        self.editor_tabs.addTab(scroll,'手动设置')
        self.editor_tabs.currentChanged.connect(lambda _index:self.discussion_edited())
        side.addWidget(self.editor_tabs,1)
        self.generate = QPushButton('生成本次修饰')
        self.generate.setProperty('kind','primary')
        self.generate.clicked.connect(lambda:self.start('generate'))
        side.addWidget(self.generate)
        self.recompose = QPushButton('按当前范围重新合成')
        self.recompose.clicked.connect(lambda:self.start('recompose'))
        side.addWidget(self.recompose)
        self.result_info = ui_label('')
        self.result_info.setWordWrap(True)
        self.result_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result_info.setMaximumHeight(42)
        side.addWidget(self.result_info)
        self.raw_button = menu.addAction('查看模型原始输出',self.open_raw)
        self.reload_button = menu.addAction('重新读取所选版本',lambda:self.start('load'))
        self.save_button = QPushButton('另存这个修饰版本')
        self.save_button.clicked.connect(lambda:self.start('save'))
        side.addWidget(self.save_button)
        panel.setMinimumWidth(290)
        panel.setMaximumWidth(360)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0,1)
        splitter.setSizes([900,320])
        layout.addWidget(splitter,1)
        self.message = StatusText('正在读取照片')
        self.message.setMaximumHeight(48)
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.message)
        self.canvas.changed.connect(self.selection_changed)
        state = self.draft_directory/'draft.json'
        if state.is_file():
            saved = json.loads(state.read_text(encoding='utf-8'))
            self.drafts = saved.get('tasks',{})
            self.task_key = saved['task'] if saved.get('task') in TASKS else 'hair_repair'
            self.task.blockSignals(True)
            self.task.setCurrentIndex(self.task.findData(self.task_key))
            self.task.blockSignals(False)
        self.apply_request(context)
        self.update_controls()

    def apply_request(self, context):
        if self.is_busy():
            return
        task = context.get('local_task')
        if task is None:
            return
        if self.canvas.mask is not None and not self.save_draft():
            return
        self.task_key = task
        self.task.blockSignals(True)
        self.task.setCurrentIndex(self.task.findData(task))
        self.task.blockSignals(False)
        if context.get('local_brief'):
            draft = self.drafts.setdefault(task,{})
            draft['brief'] = '\n'.join(text for text in (draft.get('brief','').rstrip(),context['local_brief']) if text)
            draft['generation_dirty'] = bool(draft.get('result'))
        if self.before is not None:
            self.restore_task()
            self.save_draft()

    def is_busy(self):
        return self.worker is not None

    def discussion_edited(self):
        if not self.restoring:
            self.draft_timer.start()

    def refresh_discussion(self):
        entries = []
        for row in self.messages:
            label = ''
            if row.get('context'):
                mode = {'paint':'涂画范围','source':'只看底图','output':'只看修饰结果','compare':'左右比较'}[row['context']['view_mode']]
                label = row['context']['task_label']+' · '+mode
            entries.append(dict(row,context={'label':label} if label else None))
        self.chat_history.show_messages(entries)

    def chat_context(self):
        return {'base':self.context['base'],'task':self.task_key,'task_label':TASKS[self.task_key][0],
            'engine':local_engine(self.config)['engine'],'brief':self.brief.toPlainText(),
            'mask_dimensions':list(self.canvas.mask.size),'selection_box':self.canvas.mask.getbbox(),
            'selection_sha256':hashlib.sha256(self.canvas.mask.tobytes()).hexdigest(),
            'protected_sha256':hashlib.sha256(self.canvas.protected.tobytes()).hexdigest(),
            'guide_sha256':hashlib.sha256(self.canvas.guide.tobytes()).hexdigest(),
            'reference':self.reference.text().strip() or None,
            'result':self.result_directory if not self.failed else None,
            'generation_dirty':self.generation_dirty,
            'writeback_dirty':self.writeback_dirty,
            'view_mode':self.preview.mode.currentData() if self.pages.currentIndex()==1 else 'paint'}

    def send_message(self):
        text = self.chat_input.toPlainText().strip()
        if not text or self.is_busy():
            return
        context = self.chat_context()
        self.messages.append({'role':'user','content':text,'context':context})
        self.chat_input.clear()
        self.refresh_discussion()
        self.start('chat',inputs={'mask':self.canvas.mask.copy(),'guide':self.canvas.guide.copy(),
            'protected':self.canvas.protected.copy(),
            'context':context,'conversation':deepcopy(self.messages)})

    def operation_status(self, text, *, terminal=False):
        if self.operation_message is not None:
            self.operation_message['operation_status'] = text
            self.refresh_discussion()
            if not self.save_draft():
                self.operation_message['operation_status'] = self.message.text()
                self.refresh_discussion()
            if terminal:
                self.operation_message = None

    def execute_chat(self, row):
        operation = row['operation']
        self.operation_message = row
        if operation['action']!='save' and operation['brief'] is not None:
            self.brief.setPlainText(operation['brief'])
        action = operation['action']
        if action=='draft':
            self.operation_status('具体要求已存草稿，尚未生成',terminal=True)
        else:
            self.pending_save = operation['save']
            phase = action if action in ('generate','recompose') else 'save'
            if self.start(phase):
                self.operation_status({'generate':'正在生成本次修饰','recompose':'正在按当前范围重新合成',
                    'save':'正在另存当前修饰结果'}[phase])
            else:
                self.pending_save = False
                self.operation_status(self.message.text(),terminal=True)

    def choose_reference(self):
        path,_ = QFileDialog.getOpenFileName(self,'选择参考图',self.reference.text())
        if path:
            self.reference.setText(path)

    def open_raw(self):
        if self.result_directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.result_directory)/'model-output.png')))

    def page_changed(self, index):
        if index!=1 or self.before is None or self.worker is not None:
            return
        if self.result_directory and not self.failed:
            self.start('compare')
        else:
            self.preview.set_image_pair(self.before,self.before)

    def generation_edited(self):
        if self.restoring:
            return
        self.generation_dirty = bool(self.result_directory)
        if self.generation_dirty:
            self.message.setText('草稿已更新；当前结果仍为上次生成。')
        self.draft_timer.start()
        self.update_controls()

    def selection_changed(self, kind):
        if self.restoring:
            return
        if kind=='guide' or not self.result_directory:
            self.generation_edited()
            return
        self.writeback_dirty = True
        self.draft_timer.start()
        self.message.setText('范围已修改，可直接重新合成已有结果。')
        self.update_controls()

    def change_task(self):
        if not self.save_draft():
            self.task.blockSignals(True)
            self.task.setCurrentIndex(self.task.findData(self.task_key))
            self.task.blockSignals(False)
            return
        self.task_key = self.task.currentData()
        self.restore_task()

    def restore_task(self):
        self.failed = False
        self.failure_phase = None
        self.failure_message = ''
        self.restoring = True
        try:
            state = self.drafts.get(self.task_key,{})
            self.brief.setPlainText(state.get('brief',''))
            self.reference.setText(state.get('reference',''))
            discussion = state.get('discussion',{})
            self.messages = deepcopy(discussion.get('messages',[]))
            self.chat_input.setPlainText(discussion.get('input',''))
            self.editor_tabs.setCurrentIndex(discussion.get('tab',0))
            self.refresh_discussion()
            mask_path = self.draft_directory/(self.task_key+'-mask.png')
            guide_path = self.draft_directory/(self.task_key+'-guide.png')
            protected_path = self.draft_directory/(self.task_key+'-protected.png')
            mask = Image.open(mask_path).convert('L') if mask_path.is_file() else None
            guide = Image.open(guide_path).convert('RGBA') if guide_path.is_file() else None
            protected = Image.open(protected_path).convert('L') if protected_path.is_file() else None
            self.canvas.set_base(self.before,mask,guide,protected)
            self.preview.set_image_pair(self.before,self.before)
            self.result_directory = str(Path(self.context['directory'])/state['result']) if state.get('result') else None
            self.generation_dirty = state.get('generation_dirty',False)
            self.writeback_dirty = state.get('writeback_dirty',False)
            self.result_info.setText('')
            self.note.setText({'hair_repair':'修补露网、假发缺口与断层，延续相邻发色和发束。',
                'hair_shape':'涂选要调整的发束，描述希望的效果；形状标记可选。',
                'clothing':'涂选要调整的衣料；形状标记可选，蓝色区域保留。'}[self.task_key])
            self.pages.setCurrentIndex(0)
        finally:
            self.restoring = False
        self.update_controls()

    def save_draft(self):
        self.draft_timer.stop()
        if self.canvas.mask is None:
            return True
        try:
            self.draft_directory.mkdir(parents=True,exist_ok=True)
            save_image(self.canvas.mask,self.draft_directory/(self.task_key+'-mask.png'))
            save_image(self.canvas.guide,self.draft_directory/(self.task_key+'-guide.png'))
            save_image(self.canvas.protected,self.draft_directory/(self.task_key+'-protected.png'))
            self.drafts[self.task_key] = {'brief':self.brief.toPlainText(),'reference':self.reference.text(),
                'result':os.path.relpath(self.result_directory,self.context['directory']) if self.result_directory else None,
                'writeback_dirty':self.writeback_dirty,
                'generation_dirty':self.generation_dirty,
                'discussion':{'messages':self.messages,'input':self.chat_input.toPlainText(),
                    'tab':self.editor_tabs.currentIndex()}}
            save_json(self.draft_directory/'draft.json',{'task':self.task_key,'tasks':self.drafts})
            return True
        except (OSError,ValueError) as error:
            self.message.setText('保存局部修饰草稿失败：'+str(error))
            return False

    def start(self, phase, *, inputs=None):
        if self.worker is not None or (phase=='load' and not self.isVisible()):
            return False
        if phase!='load' and not self.save_draft():
            return False
        if phase in ('generate','recompose') and not self.canvas.mask.getbbox():
            self.message.setText('请先涂出本次要修改的范围。')
            return False
        if phase=='save' and self.writeback_dirty:
            self.message.setText('请先按当前范围重新合成，再另存。')
            return False
        if phase=='recompose':
            inputs = {'mask':self.canvas.mask.copy(),'protected':self.canvas.protected.copy()}
        if phase=='generate':
            inputs = {'mask':self.canvas.mask.copy(),'guide':self.canvas.guide.copy(),
                'protected':self.canvas.protected.copy(),
                'task':self.task_key,'brief':self.brief.toPlainText(),'reference':self.reference.text().strip() or None}
        if phase=='load':
            self.loaded_failure = self.failed and self.failure_phase!='load'
            self.loaded_failure_state = (self.failure_phase,self.failure_message)
        self.worker_failed = False
        if phase!='chat':
            self.failed = False
        directory = (self.draft_directory/('chat-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
            if phase=='chat' else self.result_directory)
        self.worker = LocalWorker(phase,self.context,self.config,pixels=self.pixels,before=self.before,
            directory=directory,inputs=inputs,sr_config=self.sr_config,llm_config=self.llm_config,
            head_config=self.head_config,library_path=self.library_path)
        self.worker.progress.connect(self.message.setText)
        self.worker.ready.connect(self.action_ready)
        self.worker.failed.connect(self.action_failed)
        self.worker.finished.connect(self.action_finished)
        self.update_controls()
        self.busy_changed.emit()
        if phase=='chat':
            self.message.setText('Agent 正在讨论本次局部修饰，涂选范围和任务保持发送时状态。')
        self.worker.start()
        return True

    def action_ready(self, result):
        phase = self.worker.phase
        if phase=='load':
            self.pixels,self.before = result
            self.restore_task()
            if self.loaded_failure:
                self.failed = True
                self.failure_phase,self.failure_message = self.loaded_failure_state
            self.message.setText(self.failure_message+'。已重新读取所选版本，下一次生成仍需主动点击。' if self.loaded_failure else
                '请涂选修改范围')
        elif phase=='chat':
            row = {'role':'assistant','content':result['reply'],'operation':result['operation']}
            self.messages.append(row)
            self.refresh_discussion()
            self.message.setText('已回复')
            if self.save_draft() and row['operation'] is not None:
                self.pending_chat = row
        elif phase in ('generate','compare','recompose'):
            self.result_directory,preview,state = result
            if phase in ('generate','recompose'):
                self.writeback_dirty = False
            if phase=='generate':
                self.generation_dirty = False
            if phase in ('generate','recompose'):
                # Show the region actually composed, so subsequent manual adjustments start from
                # the whole part instead of silently replacing it with the earlier intent strokes.
                with Image.open(Path(self.result_directory)/'mask.png') as actual_mask:
                    full_mask = Image.new('L',tuple(state['source_dimensions']))
                    full_mask.paste(actual_mask.convert('L'),tuple(state['box'][:2]))
                    self.canvas.mask = full_mask.resize(self.canvas.mask.size,Image.Resampling.NEAREST)
                self.canvas.refresh_overlay()
            native = state.get('model_output_dimensions',[])
            detail = state.get('detail_dimensions')
            x0,y0,x1,y1 = state['box']
            label = LOCAL_ENGINES.get(state.get('engine'),{}).get('label')
            info = (label+' · ' if label else '')+(f'模型原生 {native[0]} × {native[1]}' if native else '已有模型输出')
            if state.get('superresolution'):
                info += f'\n一次超分 {detail[0]} × {detail[1]}'
            # The label has to say which pictures the engine actually received.
            scope = state.get('input_scope')
            if scope=='full_saved_photo':
                area = {'garment_union_minus_protected':'衣物部件范围写回',
                    'hair_union_minus_protected':'发束新旧轮廓写回'}.get(state.get('writeback_scope'),'仅涂选像素写回')
                info += f'\n处理画布 {x1-x0} × {y1-y0}；'+area
            elif scope=='detail_crop_with_full_photo_context':
                area = {'garment_union_minus_protected':'衣物部件范围写回',
                    'hair_union_minus_protected':'发束新旧轮廓写回'}.get(state.get('writeback_scope'),'仅涂选像素写回')
                info += f'\n编辑视图 {x1-x0} × {y1-y0}；完整照片作参考；'+area
            else:
                info += f'\n写回局部 {x1-x0} × {y1-y0}'
            width,height = state['source_dimensions']
            info += f'\n另存整图 {width} × {height} · 16 位 PNG'
            self.result_info.setText(info)
            # A completed generation is a direct answer to the user's click, so
            # show it; merely reading an existing result must not take the tab
            # away from the painting page the user has open.
            if phase in ('generate','recompose') and self.isVisible():
                self.pages.setCurrentIndex(1)
            if self.isVisible() and self.pages.currentIndex()==1:
                self.preview.set_image_pair(self.before,preview)
            self.save_draft()
            self.message.setText(TASKS[self.task_key][0]+('本次结果已读取，可切到比较页查看后另存。'
                if phase=='compare' else '已生成。比较后可另存，范围外沿用所选版本像素。'))
            if phase=='recompose':
                self.message.setText('已按当前范围重新合成，未调用生图或超分。')
            if self.generation_dirty:
                self.message.setText('当前为上次生成结果；草稿中的新要求尚未生成。')
            if phase in ('generate','recompose'):
                done = '本次范围已重新合成' if phase=='recompose' else '本次修饰已生成'
                self.operation_status(done+('，正在另存' if self.pending_save else '，尚未另存'),
                    terminal=not self.pending_save)
        elif phase=='save':
            self.message.setText('已另存：'+result['files']['final'])
            self.operation_status('已另存修饰版：'+result['files']['final'],terminal=True)
            self.saved.emit(result)

    def action_failed(self, error):
        self.worker_failed = True
        self.pending_chat = None
        self.pending_save = False
        if self.worker.phase=='chat':
            self.messages.append({'role':'assistant','content':'本次局部修饰讨论失败：'+error})
            self.refresh_discussion()
            self.save_draft()
            self.message.setText('本次局部修饰讨论失败：'+error)
        else:
            self.failed = True
            self.failure_phase = self.worker.phase
            self.failure_message = '局部修饰失败：'+error
            self.message.setText(self.failure_message+'。已有文件和涂画保留。')
            self.operation_status(self.failure_message,terminal=True)

    def action_finished(self):
        worker,self.worker = self.worker,None
        worker.pixels = worker.before = worker.inputs = None
        worker.deleteLater()
        row,self.pending_chat = self.pending_chat,None
        if row is not None:
            self.execute_chat(row)
            if self.worker is not None:
                return
        elif not self.worker_failed and worker.phase in ('generate','recompose') and self.pending_save:
            self.pending_save = False
            if self.start('save'):
                return
            self.operation_status(self.message.text(),terminal=True)
        if not self.isVisible():
            self.release_pixels()
        elif (worker.phase=='load' and self.result_directory and not self.failed
                and self.pages.currentIndex()==1):
            self.start('compare')
        self.update_controls()
        self.busy_changed.emit()

    def refresh_engine_note(self):
        """H3 regenerates the editing view and the selection only decides the writeback, so its brief
        asks for the final appearance only.

        Only the hint and the placeholder change; draft or user text is never rewritten, and the
        Qwen wording stays exactly as before.
        """
        h3 = (self.config or {}).get('engine')=='h3'
        self.brief_hint.setVisible(h3)
        self.brief.setPlaceholderText('描述希望得到的最终效果。'
            if h3 else '可写部位、方向和幅度。')

    def update_controls(self):
        busy = self.is_busy()
        ready = self.pixels is not None
        self.refresh_engine_note()
        for widget in (self.task,self.tool,self.brush,self.clear,self.brief,self.reference,
                self.reference_button,self.canvas):
            widget.setEnabled(ready and not busy)
        self.connection.setEnabled(not busy)
        self.chat_send.setEnabled(ready and not busy)
        self.chat_input.setEnabled(not busy)
        self.generate.setEnabled(ready and not busy)
        self.recompose.setVisible(bool(self.result_directory))
        self.recompose.setEnabled(ready and not busy and bool(self.result_directory) and self.writeback_dirty)
        self.reload_button.setEnabled(not busy)
        self.save_button.setEnabled(ready and not busy and not self.failed and bool(self.result_directory) and not self.writeback_dirty)
        self.raw_button.setEnabled(not busy and bool(self.result_directory))

    def release_pixels(self):
        if not self.save_draft() or self.worker is not None:
            return
        self.pixels = self.before = None
        self.canvas.release()
        self.preview.release_images()

    def hideEvent(self, event):
        self.release_pixels()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if self.pixels is None and self.worker is None:
            if not (self.failed and self.failure_phase=='load'):
                QTimer.singleShot(0,lambda:self.start('load'))
