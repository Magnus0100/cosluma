"""Optional painted edits and explicit saved versions, after production."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QGraphicsView, QHBoxLayout, QLineEdit, QPlainTextEdit, QPushButton,
    QScrollArea, QSlider, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from image_io import display_image, open_photo, save_image, save_json
from local_retouch import LOCAL_DEFAULTS, TASKS, generate_local, retouch_crop, save_local
from model_downloads import ModelDownloads
from service_settings import endpoint_url
from settings import load_settings
from studio_widgets import PhotoCanvas, PhotoPreview, ui_label


def _pixmap(image):
    pixels = np.array(image.convert('RGBA'))
    return QPixmap.fromImage(QImage(pixels.data,image.width,image.height,pixels.strides[0],QImage.Format_RGBA8888).copy())


class MaskCanvas(PhotoCanvas):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.mask = self.guide = None
        self.tool = 'mask'
        self.brush = 36
        self.last = None
        self.overlay = self.scene().addPixmap(QPixmap())
        self.overlay.setZValue(1)
        self.marks = self.scene().addPixmap(QPixmap())
        self.marks.setZValue(2)
        self.setToolTip('先选择工具；滚轮缩放。“移动查看”可拖动画面。粉色是实际修改范围，青色是形状标记。')

    def set_base(self, image, mask=None, guide=None):
        image = image.scaled(1600,1600,Qt.KeepAspectRatio,Qt.SmoothTransformation) if max(image.width(),image.height())>1600 else image
        self.setPixmap(QPixmap.fromImage(image))
        size = (image.width(),image.height())
        self.mask = mask.resize(size,Image.Resampling.NEAREST) if mask else Image.new('L',size)
        self.guide = guide.resize(size,Image.Resampling.NEAREST) if guide else Image.new('RGBA',size)
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
        self.overlay.setPixmap(_pixmap(overlay))
        self.marks.setPixmap(_pixmap(self.guide))

    def paint_to(self, event):
        if self.mask is None:
            return
        point = self.mapToScene(event.position().toPoint())
        xy = (round(point.x()),round(point.y()))
        previous = self.last or xy
        if self.tool in ('mask','protect'):
            draw = ImageDraw.Draw(self.mask)
            color = 255 if self.tool=='mask' else 0
        else:
            draw = ImageDraw.Draw(self.guide)
            color = (53,220,204,255) if self.tool=='shape' else (0,0,0,0)
        draw.line((previous,xy),fill=color,width=self.brush)
        radius = self.brush/2
        draw.ellipse((xy[0]-radius,xy[1]-radius,xy[0]+radius,xy[1]+radius),fill=color)
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
            self.changed.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def clear_selection(self):
        if self.mask is not None:
            self.mask.paste(0,(0,0,*self.mask.size))
            self.refresh_overlay()
            self.changed.emit()

    def release(self):
        self.setText('打开局部修饰后读取已保存版本')
        self.overlay.setPixmap(QPixmap())
        self.marks.setPixmap(QPixmap())
        self.mask = self.guide = None


class LocalConnection(QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.config = load_settings(path)
        values = {**LOCAL_DEFAULTS,'url':self.config.get('head',{}).get('url',LOCAL_DEFAULTS['url']),
            **self.config.get('local_edit',{})}
        self.setWindowTitle('局部修饰连接')
        self.resize(680,430)
        layout = QVBoxLayout(self)
        note = ui_label('使用支持 Qwen-Image-2.1 的 ComfyUI。三个修饰任务共用以下固定模型，保存连接不开始生成。')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.fields = {}
        for key,label in (('url','ComfyUI 地址'),('model','Qwen-Image-2.1 模型文件'),
                ('clip','Qwen3-VL 编码器文件'),('vae','Qwen-Image-2.1 VAE')):
            field = QLineEdit(str(values[key]))
            self.fields[key] = field
            form.addRow(label,field)
        layout.addLayout(form)
        note = ui_label('固定 1024 分辨率、40 步、seed 42。文本编码器和无损 KV 缓存放在内存，编码后释放文本编码器。原生输出不足时使用“分割与超分”中已配置的模型超分一次，再缩小到修改区域。')
        note.setWordWrap(True)
        layout.addWidget(note)
        download = QPushButton('模型下载与部署方法')
        download.clicked.connect(lambda:ModelDownloads(self,model='local').exec())
        layout.addWidget(download)
        self.message = ui_label('')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def save(self):
        try:
            values = {key:field.text().strip() for key,field in self.fields.items()}
            if any(not value for value in values.values()):
                raise ValueError('请填写实际服务地址和模型文件名')
            values['url'] = endpoint_url(values['url'],'local_edit')
            self.config['local_edit'] = {**LOCAL_DEFAULTS,**values}
            save_json(self.path,self.config)
            self.accept()
        except (OSError,ValueError) as error:
            self.message.setText('保存失败：'+str(error))


class LocalWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, phase, context, config, *, pixels=None, before=None, directory=None, inputs=None, sr_config=None):
        super().__init__()
        self.phase,self.context,self.config = phase,context,config
        self.pixels,self.before,self.directory,self.inputs = pixels,before,directory,inputs
        self.sr_config = sr_config

    def run(self):
        try:
            if self.phase=='load':
                pixels = open_photo(self.context['base'])
                result = (pixels,display_image(pixels))
            elif self.phase=='generate':
                directory,_ = generate_local(self.context['base'],self.context['directory'],
                    **self.inputs,config=self.config,progress=self.progress.emit,pixels=self.pixels,
                    parent_version=self.context.get('parent_version'),sr_config=self.sr_config)
                result = self.comparison(directory)
            elif self.phase=='compare':
                result = self.comparison(self.directory)
            elif self.phase=='save':
                self.progress.emit('另存原尺寸 16 位 PNG 与 JPEG')
                result = save_local(self.pixels,self.directory)
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

    def __init__(self, context, config, parent=None, *, sr_config=None):
        super().__init__(parent)
        self.context,self.config = context,config
        self.sr_config = sr_config
        self.worker = None
        self.pixels = self.before = None
        self.result_directory = None
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
        self.connection = QPushButton('局部修饰连接')
        self.connection.clicked.connect(self.connection_requested.emit)
        header.addWidget(self.connection)
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
        panel = QFrame()
        panel.setProperty('role','panel')
        controls = QVBoxLayout(panel)
        controls.setContentsMargins(18,18,18,18)
        self.task = QComboBox()
        for key,(label,_) in TASKS.items():
            self.task.addItem(label,key)
        self.task.currentIndexChanged.connect(self.change_task)
        controls.addWidget(self.task)
        self.note = ui_label('')
        self.note.setWordWrap(True)
        controls.addWidget(self.note)
        self.tool = QComboBox()
        for text,value in (('涂选修改范围','mask'),('保护／擦除范围','protect'),
                ('画发束形状标记','shape'),('擦除形状标记','erase_shape'),('移动查看','move')):
            self.tool.addItem(text,value)
        self.tool.currentIndexChanged.connect(lambda:self.canvas.set_tool(self.tool.currentData()))
        controls.addWidget(self.tool)
        controls.addWidget(ui_label('画笔大小'))
        self.brush = QSlider(Qt.Horizontal)
        self.brush.setRange(4,180)
        self.brush.setValue(36)
        self.brush.valueChanged.connect(lambda value:setattr(self.canvas,'brush',value))
        controls.addWidget(self.brush)
        self.clear = QPushButton('清空修改范围')
        self.clear.clicked.connect(self.canvas.clear_selection)
        controls.addWidget(self.clear)
        controls.addWidget(ui_label('这次具体要求','heading'))
        self.brief = QPlainTextEdit()
        self.brief.setPlaceholderText('可留空；发束造型可写方向、轮廓和幅度。')
        self.brief.setMaximumHeight(150)
        self.brief.textChanged.connect(self.invalidate_result)
        controls.addWidget(self.brief)
        self.reference = QLineEdit()
        self.reference.setPlaceholderText('可选：发色或发束参考图')
        self.reference.textChanged.connect(self.invalidate_result)
        controls.addWidget(self.reference)
        self.reference_button = QPushButton('选择参考图')
        self.reference_button.clicked.connect(self.choose_reference)
        controls.addWidget(self.reference_button)
        note = ui_label('粉色范围决定最终写回区域；脸、头饰、衣领和配饰请留在范围外。青色线只表达形状，需另涂相应修改范围。涂画预览最长边 1600，比较结果可看原尺寸。')
        note.setWordWrap(True)
        controls.addWidget(note)
        controls.addStretch(1)
        self.generate = QPushButton('生成本次修饰')
        self.generate.setProperty('kind','primary')
        self.generate.clicked.connect(lambda:self.start('generate'))
        controls.addWidget(self.generate)
        self.result_info = ui_label('')
        self.result_info.setWordWrap(True)
        self.result_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        controls.addWidget(self.result_info)
        self.raw_button = QPushButton('查看模型原始输出')
        self.raw_button.clicked.connect(self.open_raw)
        controls.addWidget(self.raw_button)
        self.reload_button = QPushButton('重新读取所选版本')
        self.reload_button.clicked.connect(lambda:self.start('load'))
        controls.addWidget(self.reload_button)
        self.save_button = QPushButton('另存这个修饰版本')
        self.save_button.clicked.connect(lambda:self.start('save'))
        controls.addWidget(self.save_button)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(290)
        scroll.setMaximumWidth(340)
        scroll.setWidget(panel)
        splitter.addWidget(scroll)
        splitter.setSizes([900,310])
        layout.addWidget(splitter,1)
        self.message = ui_label('正在读取已保存版本；打开界面不调用模型。')
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.message)
        self.canvas.changed.connect(self.invalidate_result)
        state = self.draft_directory/'draft.json'
        if state.is_file():
            saved = json.loads(state.read_text(encoding='utf-8'))
            self.drafts = saved.get('tasks',{})
            self.task_key = saved.get('task','hair_repair')
            self.task.blockSignals(True)
            self.task.setCurrentIndex(self.task.findData(self.task_key))
            self.task.blockSignals(False)
        self.update_controls()

    def is_busy(self):
        return self.worker is not None

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

    def invalidate_result(self):
        if self.restoring:
            return
        self.result_directory = None
        self.result_info.setText('')
        if self.before is not None:
            self.preview.set_image_pair(self.before,self.before)
        self.draft_timer.start()
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
            mask_path = self.draft_directory/(self.task_key+'-mask.png')
            guide_path = self.draft_directory/(self.task_key+'-guide.png')
            mask = Image.open(mask_path).convert('L') if mask_path.is_file() else None
            guide = Image.open(guide_path).convert('RGBA') if guide_path.is_file() else None
            self.canvas.set_base(self.before,mask,guide)
            self.preview.set_image_pair(self.before,self.before)
            self.result_directory = str(Path(self.context['directory'])/state['result']) if state.get('result') else None
            self.result_info.setText('')
            self.note.setText({'hair_repair':'修补露网、假发缺口与断层，延续相邻发色和发束。',
                'hair_shape':'按标记或参考改变指定发束；大幅轮廓变化先比较本次结果。',
                'neck':'仅修饰下颌下方与颈部；面部轮廓、衣领与配饰留在范围外。'}[self.task_key])
            self.pages.setCurrentIndex(0)
        finally:
            self.restoring = False
        self.update_controls()
        if self.result_directory and self.worker is None and self.pages.currentIndex()==1:
            self.start('compare')

    def save_draft(self):
        self.draft_timer.stop()
        if self.canvas.mask is None:
            return True
        try:
            self.draft_directory.mkdir(parents=True,exist_ok=True)
            save_image(self.canvas.mask,self.draft_directory/(self.task_key+'-mask.png'))
            save_image(self.canvas.guide,self.draft_directory/(self.task_key+'-guide.png'))
            self.drafts[self.task_key] = {'brief':self.brief.toPlainText(),'reference':self.reference.text(),
                'result':os.path.relpath(self.result_directory,self.context['directory']) if self.result_directory else None}
            save_json(self.draft_directory/'draft.json',{'task':self.task_key,'tasks':self.drafts})
            return True
        except (OSError,ValueError) as error:
            self.message.setText('保存局部修饰草稿失败：'+str(error))
            return False

    def start(self, phase):
        if self.worker is not None or (phase=='load' and not self.isVisible()):
            return
        if phase!='load' and not self.save_draft():
            return
        if phase=='generate' and not self.canvas.mask.getbbox():
            self.message.setText('请先涂出本次要修改的范围。')
            return
        inputs = {'mask':self.canvas.mask.copy(),'guide':self.canvas.guide.copy(),
            'task':self.task_key,'brief':self.brief.toPlainText(),'reference':self.reference.text().strip() or None} if phase=='generate' else None
        if phase=='load':
            self.loaded_failure = self.failed and self.failure_phase!='load'
            self.loaded_failure_state = (self.failure_phase,self.failure_message)
        self.failed = False
        self.worker = LocalWorker(phase,self.context,self.config,pixels=self.pixels,before=self.before,
            directory=self.result_directory,inputs=inputs,sr_config=self.sr_config)
        self.worker.progress.connect(self.message.setText)
        self.worker.ready.connect(self.action_ready)
        self.worker.failed.connect(self.action_failed)
        self.worker.finished.connect(self.action_finished)
        self.update_controls()
        self.busy_changed.emit()
        self.worker.start()

    def action_ready(self, result):
        phase = self.worker.phase
        if phase=='load':
            self.pixels,self.before = result
            self.restore_task()
            if self.loaded_failure:
                self.failed = True
                self.failure_phase,self.failure_message = self.loaded_failure_state
            self.message.setText(self.failure_message+'。已重新读取所选版本，下一次生成仍需主动点击。' if self.loaded_failure else
                '先涂范围并填写要求，再主动生成；基础成片保留。')
        elif phase in ('generate','compare'):
            self.result_directory,preview,state = result
            native = state.get('model_output_dimensions',[])
            detail = state.get('detail_dimensions')
            x0,y0,x1,y1 = state['box']
            info = f'模型原生 {native[0]} × {native[1]}' if native else '已有模型输出'
            if state.get('superresolution'):
                info += f'\n一次超分 {detail[0]} × {detail[1]}'
            info += f'\n写回局部 {x1-x0} × {y1-y0}'
            width,height = state['source_dimensions']
            info += f'\n另存整图 {width} × {height} · 16 位 PNG'
            self.result_info.setText(info)
            # A completed generation is a direct answer to the user's click, so
            # show it; merely reading an existing result must not take the tab
            # away from the painting page the user has open.
            if phase=='generate' and self.isVisible():
                self.pages.setCurrentIndex(1)
            if self.isVisible() and self.pages.currentIndex()==1:
                self.preview.set_image_pair(self.before,preview)
            self.save_draft()
            self.message.setText(TASKS[self.task_key][0]+('本次结果已读取，可切到比较页查看后另存。'
                if phase=='compare' else '已生成。比较后可另存，范围外沿用所选版本像素。'))
        elif phase=='save':
            self.message.setText('已另存：'+result['files']['final'])
            self.saved.emit(result)

    def action_failed(self, error):
        self.failed = True
        self.failure_phase = self.worker.phase
        self.failure_message = '局部修饰失败：'+error
        self.message.setText(self.failure_message+'。已有文件和涂画保留。')

    def action_finished(self):
        worker,self.worker = self.worker,None
        worker.pixels = worker.before = worker.inputs = None
        worker.deleteLater()
        if not self.isVisible():
            self.release_pixels()
        elif (worker.phase=='load' and self.result_directory and not self.failed
                and self.pages.currentIndex()==1):
            self.start('compare')
        self.update_controls()
        self.busy_changed.emit()

    def update_controls(self):
        busy = self.is_busy()
        ready = self.pixels is not None
        for widget in (self.task,self.tool,self.brush,self.clear,self.brief,self.reference,
                self.reference_button,self.canvas):
            widget.setEnabled(ready and not busy)
        self.connection.setEnabled(not busy)
        self.generate.setEnabled(ready and not busy)
        self.reload_button.setEnabled(not busy)
        self.save_button.setEnabled(ready and not busy and not self.failed and bool(self.result_directory))
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
