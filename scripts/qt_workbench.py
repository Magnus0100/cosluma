"""Desktop setup, photo queue and previews around the single finishing chain."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import uuid
from urllib.parse import urlsplit, urlunsplit

from PySide6.QtCore import QItemSelectionModel, QSettings, QThread, Signal, Slot, Qt, QUrl, QSize, QTimer
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QIcon, QImageReader, QKeySequence, QPalette, QPixmap, QShortcut, QTextDocument
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog,
    QDialogButtonBox, QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QMainWindow, QMenu, QPushButton, QPlainTextEdit, QProgressBar,
    QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QStyle, QSystemTrayIcon, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton,
    QVBoxLayout, QWidget)

from app_version import VERSION
from image_io import IMAGE_EXTENSIONS, save_json
from model_clients import discuss_style
from photo_pipeline import finish_batch, load_style, make_style_samples
from settings import default_settings_path, editable_settings, load_settings, save_settings


def apply_theme(app):
    """Keep the desktop, creative workspace and native popup colours consistent."""
    app.setStyle('Fusion')
    palette = QPalette()
    for role, colour in ((QPalette.Window,'#eef1f4'),(QPalette.WindowText,'#243244'),
            (QPalette.Base,'#ffffff'),(QPalette.AlternateBase,'#f2f5f8'),
            (QPalette.Text,'#243244'),(QPalette.Button,'#ffffff'),
            (QPalette.ButtonText,'#243244'),(QPalette.Highlight,'#315f87'),
            (QPalette.HighlightedText,'#ffffff'),(QPalette.PlaceholderText,'#778197'),
            (QPalette.ToolTipBase,'#ffffff'),(QPalette.ToolTipText,'#243244')):
        palette.setColor(role,QColor(colour))
    palette.setColor(QPalette.Disabled,QPalette.Text,QColor('#96a1ae'))
    palette.setColor(QPalette.Disabled,QPalette.ButtonText,QColor('#96a1ae'))
    app.setPalette(palette)
    app.setStyleSheet('''
        QWidget { color: #243244; font-size: 13px; }
        QMainWindow, QDialog { background: #eef1f4; }
        QLabel { background: transparent; }
        QLabel[role="title"] { font-size: 24px; font-weight: 600; }
        QLabel[role="heading"] { font-size: 16px; font-weight: 600; }
        QLabel[role="muted"] { color: #657588; }
        QLabel[role="eyebrow"] { color: #56799b; font-size: 11px; font-weight: 600; }
        QLabel[role="badge"] { color: #315f87; background: #e5eef6; border-radius: 8px; padding: 5px 10px; }
        QFrame[role="panel"], QWidget[role="panel"] { background: #ffffff; border: 1px solid #dbe2e9; border-radius: 12px; }
        QFrame[role="composer"] { background: #ffffff; border: 1px solid #cfd9e3; border-radius: 12px; }
        QFrame[role="userBubble"] { background: #e2edf6; border-radius: 11px; }
        QFrame[role="agentBubble"] { background: #f1f4f7; border-radius: 11px; }
        QFrame[role="welcome"] { background: #f2f6fa; border: 1px solid #d8e3ed; border-radius: 12px; }
        QFrame[role="sampleCard"] { background: #f7f9fb; border: 1px solid #d5dfe8; border-radius: 11px; }
        QLabel[role="message"] { font-size: 13px; }
        QLabel[role="messageAuthor"] { color: #6a7c8e; font-size: 11px; font-weight: 600; }
        QPushButton, QToolButton { background: #ffffff; border: 1px solid #cfd9e3; border-radius: 8px; padding: 8px 12px; }
        QPushButton:hover, QToolButton:hover { background: #edf3f8; border-color: #a4b8c9; }
        QPushButton:pressed, QToolButton:pressed { background: #e1eaf2; }
        QPushButton:focus, QToolButton:focus { border-color: #426f98; }
        QPushButton:disabled, QToolButton:disabled { color: #8d9baa; background: #eaf0f4; border-color: #dbe4eb; }
        QPushButton[kind="primary"] { background: #315f87; color: #ffffff; border: 1px solid #315f87; font-weight: 600; }
        QPushButton[kind="primary"]:hover { background: #3c719e; }
        QPushButton[kind="primary"]:pressed { background: #244a6c; }
        QPushButton[kind="primary"]:disabled { background: #dce5ec; color: #8b9ba9; border-color: #d1dce5; }
        QPushButton[kind="quiet"], QToolButton[kind="quiet"] { background: transparent; border-color: transparent; color: #586e83; }
        QPushButton[kind="quiet"]:hover, QToolButton[kind="quiet"]:hover { color: #234c70; background: #e5edf4; }
        QLineEdit, QPlainTextEdit, QComboBox { background: #ffffff; border: 1px solid #d2dce6; border-radius: 8px; padding: 8px; selection-background-color: #315f87; }
        QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus { border-color: #426f98; }
        QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled { color: #8596a5; border-color: #ffffff; }
        QPlainTextEdit[role="composer"] { border: none; background: transparent; padding: 0; }
        QPlainTextEdit[role="summary"] { border: none; background: #f1f5f8; padding: 10px; }
        QComboBox { padding-right: 26px; }
        QComboBox::drop-down { width: 22px; border: none; }
        QComboBox QAbstractItemView { background: #ffffff; border: 1px solid #b7c9d8; selection-background-color: #d8e8f5; padding: 4px; outline: 0; }
        QScrollArea { border: none; background: transparent; }
        QScrollArea > QWidget > QWidget { background: transparent; }
        QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
        QScrollBar::handle:vertical { background: #b7c7d5; border-radius: 3px; min-height: 28px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        QScrollBar:horizontal { background: transparent; height: 8px; }
        QScrollBar::handle:horizontal { background: #b7c7d5; border-radius: 3px; min-width: 28px; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        QGroupBox { border: 1px solid #d7e1ea; border-radius: 10px; margin-top: 12px; padding: 15px 12px 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #4b657d; }
        QTableWidget, QListWidget { background: #ffffff; border: 1px solid #d5e0e9; border-radius: 10px; gridline-color: #e9eef3; outline: 0; }
        QTableWidget::item, QListWidget::item { padding: 9px; border: none; }
        QTableWidget::item:selected, QListWidget::item:selected { background: #e3edf6; color: #234b6e; }
        QHeaderView::section { background: #f7f9fb; color: #647a8e; border: none; border-bottom: 1px solid #dce5ed; padding: 9px; }
        QTabWidget::pane { border: 1px solid #d9e3ec; border-radius: 10px; padding: 10px; }
        QTabBar::tab { background: #e8eef4; color: #5d7287; padding: 10px 14px; border-radius: 7px; margin: 2px; }
        QTabBar::tab:selected { background: #ffffff; color: #254f73; }
        QProgressBar { background: #e1e9f0; border: none; border-radius: 5px; text-align: center; color: #345573; min-height: 16px; }
        QProgressBar::chunk { background: #88a9c4; border-radius: 5px; }
        QSplitter::handle { background: transparent; width: 10px; }
        QSplitter::handle:hover { background: #cfdfec; }
        QMenu { background: #ffffff; border: 1px solid #d4dfe8; padding: 5px; }
        QMenu::item { padding: 8px 18px; border-radius: 5px; }
        QMenu::item:selected { background: #e6eff6; }
        QToolTip { color: #243244; background: #ffffff; border: 1px solid #c1d1df; padding: 7px; }
        QToolButton[role="samplePreview"] { background: #12151d; border: 1px solid #303849; border-radius: 9px; color: #a7b8c8; padding: 8px; }
        QToolButton[role="samplePreview"]:hover { border-color: #8cb4d4; }
        QToolButton[role="samplePreview"]:checked { border: 2px solid #5e9bca; background: #243b50; }
        QLabel[role="photoCanvas"] { background: #202934; color: #a9b9c8; border: 1px solid #202934; border-radius: 10px; padding: 8px; }
    ''')


def ui_label(text, role='muted'):
    label = QLabel(text)
    label.setProperty('role',role)
    label.setTextFormat(Qt.PlainText)
    return label


class StatusText(QLabel):
    def __init__(self, text='', parent=None):
        super().__init__(text,parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.PlainText)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setProperty('role','muted')

    def setPlainText(self, text):
        self.setText(text)
        self.setToolTip(text)

    def toPlainText(self):
        return self.text()


class SamplePreview(QToolButton):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setIconSize(QSize(max(80,self.width()-22),max(40,self.height()-24)))


class SampleCard(QFrame):
    def __init__(self, number, parent=None):
        super().__init__(parent)
        self.setProperty('role','sampleCard')
        self.column = QVBoxLayout(self)
        self.column.setContentsMargins(10,10,10,10)
        self.column.setSpacing(6)
        self.title = ui_label(f'方案 {number}','heading')
        self.full_title = f'方案 {number}'
        self.title.setFixedHeight(20)
        self.column.addWidget(self.title)

    def setTitle(self, text):
        self.full_title = text
        self.title.setText(self.title.fontMetrics().elidedText(text,Qt.ElideRight,max(60,self.width()-22)))
        self.title.setToolTip(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setTitle(self.full_title)


class ConversationView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.content = QWidget()
        self.rows = QVBoxLayout(self.content)
        self.rows.setContentsMargins(0,0,6,0)
        self.rows.setSpacing(14)
        self.setWidget(self.content)
        self.show_messages([])

    def show_messages(self, messages, pending=None):
        while self.rows.count():
            item = self.rows.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        if not messages and pending is None:
            welcome = QFrame()
            welcome.setProperty('role','welcome')
            column = QVBoxLayout(welcome)
            column.setContentsMargins(18,20,18,20)
            column.setSpacing(12)
            column.addWidget(ui_label('从这组照片聊起','heading'))
            note = ui_label('说说你想保留的细节、角色的气质，或让 Agent 先看看原片。')
            note.setWordWrap(True)
            column.addWidget(note)
            self.rows.addWidget(welcome)
        for row in [*messages,*([pending] if pending else [])]:
            user = row['role']=='user'
            bubble = QFrame()
            bubble.setProperty('role','userBubble' if user else 'agentBubble')
            column = QVBoxLayout(bubble)
            column.setContentsMargins(14,12,14,14)
            column.setSpacing(7)
            column.addWidget(ui_label('你' if user else '摄影 Agent','messageAuthor'))
            body = QLabel()
            body.setProperty('role','message')
            body.setWordWrap(True)
            body.setTextInteractionFlags(Qt.TextSelectableByMouse|Qt.LinksAccessibleByMouse)
            if user:
                body.setTextFormat(Qt.PlainText)
                body.setText(row['content'])
            else:
                document = QTextDocument()
                document.setDefaultFont(QApplication.font())
                document.setMarkdown(row['content'])
                body.setTextFormat(Qt.RichText)
                body.setText(document.toHtml())
            column.addWidget(body)
            self.rows.addWidget(bubble)
        self.rows.addStretch(1)
        QTimer.singleShot(0,lambda:self.verticalScrollBar().setValue(self.verticalScrollBar().maximum()))


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
    batch_created = Signal(str)
    result_ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, sources, character, brief, root, config, *, style=None):
        super().__init__()
        self.sources, self.character, self.brief = sources, character, brief
        self.root, self.config = root, config
        self.style = style
        self.succeeded = 0
        self.errors = []
        self.batch_error = ''
        self.output_directory = None

    def run(self):
        def started(directory):
            self.output_directory = directory
            self.batch_created.emit(directory)
        try:
            for result in finish_batch(self.sources, self.character, self.root, self.config,
                    brief=self.brief, progress=self.progress.emit,
                    photo_progress=self.photo_stage.emit, batch_started=started, style=self.style):
                self.output_directory = result.get('batch_directory', self.output_directory)
                if result['status'] == 'complete':
                    self.succeeded += 1
                else:
                    self.errors.append(f'{Path(result["source"]).name}：{result["error"]}')
                self.result_ready.emit(result)
        except Exception as error:
            self.batch_error = str(error)
            self.failed.emit(self.batch_error)


class StyleWorker(QThread):
    progress = Signal(str)
    plans_ready = Signal(list, str)
    sample_stage = Signal(int, str)
    sample_ready = Signal(dict)
    directory_created = Signal(str)
    failed = Signal(str)

    def __init__(self, sources, sample_source, character, brief, root, config, *, creative_direction=''):
        super().__init__()
        self.sources, self.sample_source = sources, sample_source
        self.character, self.brief = character, brief
        self.root, self.config = root, config
        self.creative_direction = creative_direction

    def run(self):
        try:
            for option in make_style_samples(self.sources,self.sample_source,self.character,self.root,self.config,
                    brief=self.brief,progress=self.progress.emit,planned=self.plans_ready.emit,
                    sample_progress=self.sample_stage.emit,started=self.directory_created.emit,
                    creative_direction=self.creative_direction):
                self.sample_ready.emit(option)
        except Exception as error:
            self.failed.emit(str(error))


class StyleChatWorker(QThread):
    reply_ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, sources, character, brief, config, directory, messages, direction, samples, focus):
        super().__init__()
        self.arguments = (sources,character,brief,config['prompt_library'],config['llm'],Path(directory),messages)
        self.direction, self.samples, self.focus = direction, samples, focus

    def run(self):
        try:
            result = discuss_style(*self.arguments,direction=self.direction,samples=self.samples,focus=self.focus)
            self.reply_ready.emit(result)
        except Exception as error:
            self.failed.emit(getattr(error,'provider_message','') or str(error))


class PhotoPreview(QWidget):
    """Read display previews independently of the untouched production images."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.output = None
        self.choice = 'compare'
        self.cache = {}
        self.images = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel('照片预览'), 1)
        self.mode = QComboBox()
        for label, value in (('成片', 'output'), ('原片', 'source'), ('原片 / 成片对比', 'compare')):
            self.mode.addItem(label, value)
        self.mode.currentIndexChanged.connect(self.change_mode)
        toolbar.addWidget(self.mode)
        layout.addLayout(toolbar)
        self.pages = QStackedWidget()
        self.preview = self.image_label()
        self.pages.addWidget(self.preview)
        comparison = QWidget()
        row = QHBoxLayout(comparison)
        row.setContentsMargins(0, 0, 0, 0)
        self.before = self.image_label()
        self.after = self.image_label()
        for label, image in (('原片', self.before), ('成片', self.after)):
            column = QVBoxLayout()
            caption = QLabel(label)
            caption.setAlignment(Qt.AlignCenter)
            column.addWidget(caption)
            column.addWidget(image, 1)
            row.addLayout(column, 1)
        self.pages.addWidget(comparison)
        layout.addWidget(self.pages, 1)
        self.set_photo()

    @staticmethod
    def image_label():
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(100, 160)
        label.setWordWrap(True)
        label.setProperty('role','photoCanvas')
        return label

    def read_preview(self, path):
        if not path:
            return None
        try:
            key = (str(path), Path(path).stat().st_mtime_ns)
        except OSError:
            return None
        if key not in self.cache:
            reader = QImageReader(str(path))
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid():
                size.scale(QSize(1600, 1600), Qt.KeepAspectRatio)
                reader.setScaledSize(size)
            image = reader.read()
            self.cache[key] = QPixmap.fromImage(image) if not image.isNull() else None
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    def set_photo(self, source=None, output=None):
        self.source, self.output = source, output
        self.images = {'source': self.read_preview(source), 'output': self.read_preview(output)}
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(self.mode.findData(self.choice if output else 'source'))
        self.mode.setEnabled(bool(output))
        self.mode.blockSignals(False)
        self.update_preview()

    def change_mode(self):
        if self.output:
            self.choice = self.mode.currentData()
        self.update_preview()

    def update_preview(self):
        mode = self.mode.currentData()
        page = 1 if mode == 'compare' and self.output else 0
        if self.pages.currentIndex() != page:
            self.pages.setCurrentIndex(page)
            QTimer.singleShot(0, self.update_preview)
        targets = ((self.before, 'source'), (self.after, 'output')) if self.pages.currentIndex() else (
            (self.preview, mode if self.output else 'source'),)
        for label, kind in targets:
            pixels = self.images.get(kind)
            if pixels is not None:
                label.setPixmap(pixels.scaled(label.size() - QSize(10, 10), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            elif not self.source:
                label.setText('选择照片查看预览\n完成后可对比原片与成片')
            elif kind == 'output':
                label.setText('成片暂时无法预览\n可打开成片或批次文件夹查看文件')
            elif not Path(self.source).is_file():
                label.setText('原片不在原来的位置\n已保存的成片仍可查看')
            else:
                label.setText('此原片暂不支持预览\n仍可制作，也可使用“打开原片”查看')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.update_preview)


class StylePicker(QDialog):
    """Discuss photographs, compare real samples, then explicitly settle the batch direction."""
    def __init__(self, sources, sample_source, character, brief, root, config, parent=None, *, previous=None, discussion=None):
        super().__init__(parent)
        self.setWindowTitle('与摄影 Agent 讨论风格 · '+character)
        self.resize(1400, 880)
        self.setMinimumSize(1080,720)
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)
        self.sources, self.character, self.brief = sources, character, brief
        self.config, self.root = config, root
        self.worker = None
        self.chat_worker = None
        self.messages = []
        self.creative_direction = ''
        self.samples_direction = ''
        self.discussion_directory = None
        self.sample_directories = []
        self.pending_message = None
        self.frozen_direction = ''
        self.chosen_file = ''
        self.options = {}
        self.cards = {}
        self.current_option = 0
        self.selected_style = None
        self.directory = None
        self.failure_message = ''
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24,22,24,18)
        layout.setSpacing(18)
        header = QHBoxLayout()
        heading = QVBoxLayout()
        heading.setSpacing(5)
        heading.addWidget(ui_label('创作工作室','title'))
        heading.addWidget(ui_label('从原片出发，一起找到这组照片的方向。'))
        header.addLayout(heading,1)
        context = ui_label(character,'badge')
        context.setToolTip('当前角色：'+character)
        header.addWidget(context,0,Qt.AlignVCenter)
        layout.addLayout(header)
        workspace = QSplitter(Qt.Horizontal)
        workspace.setHandleWidth(14)
        workspace.setChildrenCollapsible(False)
        sample_panel = QFrame()
        sample_panel.setProperty('role','panel')
        sample_layout = QVBoxLayout(sample_panel)
        sample_layout.setContentsMargins(18,18,18,18)
        sample_layout.setSpacing(12)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(ui_label('代表原片'))
        self.sample_source = QComboBox()
        for source in sources:
            self.sample_source.addItem(Path(source).name, source)
            self.sample_source.setItemData(self.sample_source.count()-1,source,Qt.ToolTipRole)
        self.sample_source.setCurrentIndex(max(0,self.sample_source.findData(sample_source)))
        row.addWidget(self.sample_source,1)
        original = QPushButton('放大对比')
        original.setProperty('kind','quiet')
        original.clicked.connect(lambda:self.inspect(self.current_option))
        row.addWidget(original)
        self.generate = QPushButton('按当前讨论生成三个样片')
        self.generate.setProperty('kind','primary')
        self.generate.setMinimumHeight(38)
        self.generate.clicked.connect(self.generate_samples)
        self.change_source = QPushButton('更换原片')
        self.change_source.setProperty('kind','quiet')
        self.change_source.clicked.connect(self.change_sample_source)
        row.addWidget(self.change_source)
        sample_layout.addLayout(row)
        self.canvas = PhotoPreview()
        self.canvas.set_photo(self.sample_source.currentData())
        sample_layout.addWidget(self.canvas,1)
        self.active_caption = ui_label('原片')
        self.active_caption.setWordWrap(True)
        self.active_caption.setMaximumHeight(40)
        sample_layout.addWidget(self.active_caption)
        self.inspector = QDialog(self)
        self.inspector.setWindowTitle('样片与原片大图对比')
        self.inspector.resize(1100,760)
        inspector_layout = QVBoxLayout(self.inspector)
        self.preview = PhotoPreview()
        self.preview.set_photo(self.sample_source.currentData())
        self.sample_source.currentIndexChanged.connect(lambda:self.show_option(self.current_option))
        inspector_layout.addWidget(self.preview,1)
        self.inspector_details = QPlainTextEdit()
        self.inspector_details.setReadOnly(True)
        self.inspector_details.setMaximumHeight(70)
        inspector_layout.addWidget(self.inspector_details)
        back = QPushButton('返回三个方案')
        back.clicked.connect(self.inspector.accept)
        inspector_layout.addWidget(back)
        self.sample_strip = QWidget()
        self.sample_strip.setFixedHeight(228)
        choices = QHBoxLayout(self.sample_strip)
        choices.setContentsMargins(0,0,0,0)
        choices.setSpacing(10)
        for index in range(3):
            panel = SampleCard(index+1)
            card = panel.column
            thumbnail = SamplePreview()
            thumbnail.setProperty('role','samplePreview')
            thumbnail.setText('等待生成样片')
            thumbnail.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            thumbnail.setFixedHeight(92)
            thumbnail.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
            thumbnail.setCheckable(True)
            thumbnail.clicked.connect(lambda _checked=False,index=index:self.show_option(index))
            card.addWidget(thumbnail)
            summary = QLabel('分析当前角色与照片后，再确定此方向')
            summary.setWordWrap(True)
            summary.setTextFormat(Qt.PlainText)
            summary.setFixedHeight(28)
            stage = ui_label('尚未分析')
            stage.setWordWrap(True)
            stage.setTextFormat(Qt.PlainText)
            stage.setFixedHeight(16)
            card.addWidget(summary)
            card.addWidget(stage)
            button = QPushButton('放大对比')
            button.setProperty('kind','quiet')
            button.setFixedHeight(28)
            button.setStyleSheet('padding: 2px 8px')
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False,index=index:self.inspect(index))
            card.addWidget(button)
            self.cards[index] = (summary,stage,button,thumbnail,panel)
            choices.addWidget(panel,1)
        self.sample_strip.hide()
        sample_layout.addWidget(self.sample_strip)
        self.empty_hint = ui_label('还没有样片。先聊想法，或让 Agent 根据角色和原片探索三个方向。')
        self.empty_hint.setWordWrap(True)
        sample_layout.addWidget(self.empty_hint)
        workspace.addWidget(sample_panel)
        chat_panel = QFrame()
        chat_panel.setProperty('role','panel')
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(18,18,18,18)
        chat_layout.setSpacing(12)
        chat_layout.addWidget(ui_label('一起构思','heading'))
        chat_layout.addWidget(ui_label('摄影 Agent 会结合原片与样片回应。'))
        self.chat_history = ConversationView()
        chat_layout.addWidget(self.chat_history,1)
        chat_layout.addWidget(ui_label('当前创作要求'))
        self.discussion_summary = QPlainTextEdit()
        self.discussion_summary.setProperty('role','summary')
        self.discussion_summary.setReadOnly(True)
        self.discussion_summary.setPlaceholderText('讨论形成的创作要求会显示在这里')
        self.discussion_summary.setFixedHeight(72)
        chat_layout.addWidget(self.discussion_summary)
        composer = QFrame()
        composer.setProperty('role','composer')
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(12,12,12,10)
        composer_layout.setSpacing(8)
        self.chat_input = QPlainTextEdit()
        self.chat_input.setProperty('role','composer')
        self.chat_input.setPlaceholderText('聊聊你想要的感觉，或指出样片里想改的地方…')
        self.chat_input.setFixedHeight(66)
        self.chat_input.textChanged.connect(self.update_creative_actions)
        composer_layout.addWidget(self.chat_input)
        send_row = QHBoxLayout()
        send_row.addWidget(ui_label('Ctrl + Enter 发送'))
        send_row.addStretch(1)
        self.send = QPushButton('发送')
        self.send.clicked.connect(self.send_message)
        send_row.addWidget(self.send)
        composer_layout.addLayout(send_row)
        chat_layout.addWidget(composer)
        self.send_shortcut = QShortcut(QKeySequence('Ctrl+Return'),self.chat_input)
        self.send_shortcut.activated.connect(self.send_message)
        self.keypad_send_shortcut = QShortcut(QKeySequence('Ctrl+Enter'),self.chat_input)
        self.keypad_send_shortcut.activated.connect(self.send_message)
        chat_layout.addWidget(self.generate)
        chat_panel.setMinimumWidth(330)
        sample_panel.setMinimumWidth(620)
        workspace.addWidget(chat_panel)
        workspace.setStretchFactor(0,2)
        workspace.setStretchFactor(1,1)
        workspace.setSizes([900,400])
        layout.addWidget(workspace,1)
        footer = QHBoxLayout()
        self.message = StatusText('先讨论和试样，满意后再定稿。')
        self.message.setMaximumHeight(42)
        footer.addWidget(self.message,1)
        self.open_folder = QPushButton('打开样片文件夹')
        self.open_folder.setProperty('kind','quiet')
        self.open_folder.setEnabled(False)
        self.open_folder.clicked.connect(lambda:BatchResults.open_path(self.directory))
        footer.addWidget(self.open_folder)
        self.cancel = QPushButton('返回照片队列')
        self.cancel.clicked.connect(self.reject)
        footer.addWidget(self.cancel)
        self.finalize = QPushButton('定稿并用于批量')
        self.finalize.setProperty('kind','primary')
        self.finalize.setMinimumHeight(38)
        self.finalize.clicked.connect(lambda:self.choose(self.current_option))
        footer.addWidget(self.finalize)
        layout.addLayout(footer)
        if discussion is not None:
            try:
                self.restore_discussion(discussion)
            except (OSError,ValueError,KeyError,TypeError) as error:
                self.message.setPlainText('已有创作对话无法读取：'+str(error))
        elif previous is not None:
            try:
                self.restore_samples(previous)
                self.creative_direction = self.samples_direction
                self.render_conversation()
            except (OSError,ValueError,KeyError,TypeError) as error:
                self.change_sample_source()
                self.message.setPlainText('已有样片记录无法读取，请重新选择原片后手动生成：'+str(error))
        self.update_creative_actions()

    def is_busy(self):
        return self.worker is not None or self.chat_worker is not None

    def ensure_discussion(self):
        if self.discussion_directory is None:
            directory = Path(self.root).expanduser().resolve()/('creative-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
            directory.mkdir(parents=True,exist_ok=False)
            self.discussion_directory = str(directory)

    def save_discussion(self):
        if self.discussion_directory is not None:
            save_json(Path(self.discussion_directory)/'conversation.json',{
                'character':self.character,'brief':self.brief,'sources':self.sources,
                'messages':self.messages,'direction':self.creative_direction,
                'samples_directory':self.directory,'sample_directories':self.sample_directories,
                'selected_file':self.chosen_file,'frozen_direction':self.frozen_direction,
                'draft':self.pending_message['content'] if self.pending_message else self.chat_input.toPlainText()})

    def restore_discussion(self, directory):
        state = json.loads((Path(directory)/'conversation.json').read_text(encoding='utf-8'))
        if (state['character'],state['brief']) != (self.character,self.brief):
            raise ValueError('此对话的角色或补充要求与当前输入不同')
        self.discussion_directory = str(directory)
        self.messages = state['messages']
        self.creative_direction = state.get('direction','')
        self.sample_directories = state.get('sample_directories',[])
        self.chosen_file = state.get('selected_file','')
        self.frozen_direction = state.get('frozen_direction','')
        self.chat_input.setPlainText(state.get('draft',''))
        if state.get('samples_directory'):
            self.restore_samples(state['samples_directory'])
        self.render_conversation()

    def render_conversation(self):
        self.chat_history.show_messages(self.messages)
        self.discussion_summary.setPlainText(self.creative_direction)

    def update_creative_actions(self):
        if not hasattr(self,'cancel'):
            return
        busy = self.is_busy()
        self.send.setEnabled(not busy and bool(self.chat_input.toPlainText().strip()))
        self.chat_input.setEnabled(not busy)
        self.generate.setEnabled(not busy)
        self.change_source.setEnabled(not busy)
        self.cancel.setEnabled(not busy)
        self.sample_source.setEnabled(not busy and not self.options)
        for index,card in self.cards.items():
            card[2].setEnabled(index in self.options)
        self.finalize.setEnabled(not busy and self.samples_direction==self.creative_direction
            and self.options.get(self.current_option,{}).get('status')=='complete')

    def send_message(self):
        text = self.chat_input.toPlainText().strip()
        if self.is_busy() or not text:
            return
        try:
            self.ensure_discussion()
            self.pending_message = {'role':'user','content':text}
            self.save_discussion()
            conversation = [*self.messages,self.pending_message]
            represented = list(dict.fromkeys([self.sample_source.currentData(),*self.sources]))
            focus = self.options.get(self.current_option,{}).get('title')
            self.chat_worker = StyleChatWorker(represented,self.character,self.brief,self.config,
                self.discussion_directory,conversation,self.creative_direction,list(self.options.values()),focus)
            self.chat_worker.reply_ready.connect(self.receive_reply)
            self.chat_worker.failed.connect(self.chat_failed)
            self.chat_worker.finished.connect(self.chat_finished)
            self.chat_history.show_messages(conversation,{'role':'assistant','content':'正在查看原片与样片…'})
            self.chat_input.clear()
            self.message.setPlainText('Agent 正在结合角色、原片与已有样片回复；这一步只讨论。')
            self.update_creative_actions()
            self.chat_worker.start()
        except Exception as error:
            self.chat_worker = None
            self.message.setPlainText('无法开始对话：'+str(error))
            self.update_creative_actions()

    def receive_reply(self, reply):
        self.messages.extend([self.pending_message,{'role':'assistant','content':reply['reply']}])
        self.pending_message = None
        self.creative_direction = reply['direction']
        self.render_conversation()
        try:
            self.save_discussion()
        except OSError as error:
            self.message.setPlainText('回复已收到，保存对话失败：'+str(error))
            return
        self.message.setPlainText('创作要求已更新。可继续讨论；按当前讨论生成新样片后，再定稿用于批量。' if
            self.options and self.samples_direction!=self.creative_direction else
            '可以继续讨论、查看已有样片，或按当前讨论主动生成样片。由你决定何时定稿。')

    def chat_failed(self, error):
        if self.pending_message is not None:
            self.chat_input.setPlainText(self.pending_message['content'])
        self.pending_message = None
        self.render_conversation()
        self.message.setPlainText('Agent 回复失败：'+error+'。已保留输入与原有样片，可修改后手动发送。')
        try:
            self.save_discussion()
        except OSError as saved_error:
            self.message.setPlainText(self.message.toPlainText()+'\n保存对话失败：'+str(saved_error))

    def chat_finished(self):
        worker,self.chat_worker = self.chat_worker,None
        self.update_creative_actions()
        worker.deleteLater()

    def inspect(self, index):
        self.show_option(index)
        option = self.options.get(index)
        self.inspector_details.setPlainText(option['summary'] if option and option['status']=='complete' else
            option['error'] if option else '这是选中的代表原片，尚未完成此方案。')
        self.inspector.exec()

    def change_sample_source(self):
        if self.is_busy():
            return
        self.options.clear()
        self.directory = None
        self.sample_strip.hide()
        self.empty_hint.show()
        self.open_folder.setEnabled(False)
        self.sample_source.setEnabled(True)
        self.generate.setText('按当前讨论生成三个样片')
        for index,(summary,stage,button,thumbnail,panel) in self.cards.items():
            panel.setTitle(f'方案 {index+1}')
            summary.setText('分析当前角色与照片后，再确定此方向')
            summary.setToolTip('')
            stage.setText('尚未分析')
            button.setEnabled(False)
            thumbnail.setIcon(QIcon())
            thumbnail.setText('等待生成样片')
        self.preview.set_photo(self.sample_source.currentData())
        self.canvas.set_photo(self.sample_source.currentData())
        self.active_caption.setText('原片')
        self.message.setPlainText('选择另一张代表原片后，再点击生成。之前的样片文件仍然保留。')
        self.save_discussion()
        self.update_creative_actions()

    def restore_samples(self, directory):
        directory = Path(directory)
        state = json.loads((directory/'styles.json').read_text(encoding='utf-8'))
        self.samples_direction = state.get('creative_direction','')
        index = self.sample_source.findData(state['sample_source'])
        if index < 0:
            self.sample_source.addItem(Path(state['sample_source']).name+'（已有样片原片）',state['sample_source'])
            index = self.sample_source.count()-1
        self.sample_source.setCurrentIndex(index)
        self.sample_source.setEnabled(False)
        self.show_plans(json.loads((directory/'style-options.json').read_text(encoding='utf-8')),str(directory))
        for saved in state['options'].values():
            index = saved['index']
            try:
                option = load_style(directory/f'style-{index+1}.json') if saved['status']=='complete' else saved
            except (OSError, ValueError, KeyError) as error:
                option = {**saved,'status':'failed','error':str(error)}
            self.show_sample(option)
            self.cards[index][2].setEnabled(option['status']=='complete')
        self.generate.setText('按当前讨论再做三个样片')
        self.message.setPlainText('正在查看为此角色与原片保存的样片，没有调用模型。可直接选择；重新设计会另行制作三张。')

    def generate_samples(self):
        if self.is_busy():
            return
        try:
            self.ensure_discussion()
        except OSError as error:
            self.message.setPlainText('无法创建创作文件夹：'+str(error))
            return
        self.options.clear()
        self.failure_message = ''
        self.directory = None
        self.samples_direction = self.creative_direction
        self.sample_strip.show()
        self.empty_hint.hide()
        self.open_folder.setEnabled(False)
        for index,(summary,stage,button,thumbnail,panel) in self.cards.items():
            panel.setTitle(f'方案 {index+1}')
            summary.setText('正在分析当前角色与照片')
            summary.setToolTip('')
            stage.setText('设计方向中')
            stage.setToolTip('')
            button.setEnabled(False)
            thumbnail.setIcon(QIcon())
            thumbnail.setText('等待生成样片')
        self.preview.set_photo(self.sample_source.currentData())
        self.canvas.set_photo(self.sample_source.currentData())
        self.active_caption.setText('原片 · 正在设计三个方向')
        self.generate.setEnabled(False)
        self.change_source.setEnabled(False)
        self.sample_source.setEnabled(False)
        self.cancel.setEnabled(False)
        represented = list(dict.fromkeys([self.sample_source.currentData(),*self.sources]))
        self.worker = StyleWorker(represented,self.sample_source.currentData(),self.character,
            self.brief,self.root,self.config,creative_direction=self.creative_direction)
        self.worker.progress.connect(self.message.setPlainText)
        self.worker.plans_ready.connect(self.show_plans)
        self.worker.sample_stage.connect(self.show_stage)
        self.worker.sample_ready.connect(self.show_sample)
        self.worker.directory_created.connect(self.directory_ready)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.samples_finished)
        self.message.setPlainText('正在结合角色“'+self.character+'”、代表原片与补充要求设计三个方向。其余照片尚未提交。')
        self.update_creative_actions()
        self.worker.start()

    def directory_ready(self, directory):
        self.directory = directory
        self.open_folder.setEnabled(True)
        if directory not in self.sample_directories:
            self.sample_directories.append(directory)
        try:
            self.save_discussion()
        except OSError as error:
            self.message.setPlainText('样片已开始制作，保存对话目录失败：'+str(error))

    def show_plans(self, plans, directory):
        self.directory = directory
        self.open_folder.setEnabled(True)
        self.sample_strip.show()
        self.empty_hint.hide()
        for index,plan in enumerate(plans):
            self.cards[index][4].setTitle(f'方案 {index+1} · '+plan['title'])
            self.cards[index][0].setText(plan['summary'])
            self.cards[index][0].setToolTip(plan['summary'])

    def show_stage(self, index, stage):
        self.cards[index][1].setText(stage)
        self.cards[index][1].setToolTip(stage)

    def show_sample(self, option):
        self.options[option['index']] = option
        summary,stage,_button,thumbnail,_panel = self.cards[option['index']]
        summary.setText(option['summary'])
        stage.setText('样片完成' if option['status']=='complete' else '样片失败：'+option['error'])
        stage.setToolTip(stage.text())
        if option['status']=='complete':
            pixels = self.preview.read_preview(option['result']['files']['jpeg'])
            if pixels is not None:
                thumbnail.setIcon(QIcon(pixels))
            thumbnail.setText('' if pixels is not None else '无法预览，点击查看')
            thumbnail.setToolTip('点击在画布中查看；“放大对比”可打开大图')
        else:
            thumbnail.setText('样片未完成，点击查看原因')
        if self.current_option == option['index']:
            self.show_option(option['index'])

    def show_option(self, index):
        self.current_option = index
        for number,card in self.cards.items():
            card[3].setChecked(number==index)
        option = self.options.get(index)
        output = option['result']['files']['jpeg'] if option and option['status']=='complete' else None
        self.preview.set_photo(self.sample_source.currentData(),output)
        self.canvas.set_photo(self.sample_source.currentData(),output)
        self.active_caption.setText(f'方案 {index+1} · '+option['title']+'  '+option['summary'] if option else '原片')
        if option and option['status']!='complete':
            self.message.setPlainText(option['error'])
        self.update_creative_actions()

    def failed(self, error):
        self.failure_message = error
        self.message.setPlainText('样片未完成：'+error)

    def samples_finished(self):
        worker,self.worker = self.worker,None
        self.cancel.setEnabled(True)
        self.change_source.setEnabled(True)
        self.generate.setText('按当前讨论再做三个样片')
        self.generate.setEnabled(True)
        complete = sum(option['status']=='complete' for option in self.options.values())
        for index,option in self.options.items():
            self.cards[index][2].setEnabled(option['status']=='complete')
        for index in self.cards:
            if index not in self.options:
                self.cards[index][1].setText('未完成')
        if self.options:
            self.message.setPlainText(f'已完成 {complete} 个真实样片。可以放大对比、继续与 Agent 讨论修改，满意后再定稿；未定稿前不会批量制作。' if complete else
                '三个样片都未完成。点击方案查看真实错误，已有文件保留；可以返回修改设置后手动再发起。')
        if self.failure_message:
            self.message.setPlainText(self.message.toPlainText()+'\n样片未全部完成：'+self.failure_message)
        self.update_creative_actions()
        worker.deleteLater()

    def choose(self, index):
        if self.is_busy() or self.options.get(index,{}).get('status')!='complete':
            return
        if self.samples_direction != self.creative_direction:
            self.message.setPlainText('当前讨论已修改创作要求，请先按当前讨论生成样片，再定稿。已有样片仍可对比查看。')
            return
        try:
            self.ensure_discussion()
            option = self.options[index]
            self.chosen_file = option['selection_file']
            self.frozen_direction = option['art_direction']['direction']
            save_json(Path(self.discussion_directory)/'chosen-direction.json',{
                'character':self.character,'brief':self.brief,'selection_file':self.chosen_file,
                'title':option['title'],'direction':self.frozen_direction,
                'creative_direction':self.creative_direction,
                'sample_source':option['sample_source']})
            self.save_discussion()
        except OSError as error:
            self.message.setPlainText('无法保存定稿：'+str(error))
            return
        self.selected_style = option
        self.accept()

    def reject(self):
        if not self.is_busy():
            try:
                self.save_discussion()
            except OSError as error:
                self.message.setPlainText('保存对话失败：'+str(error))
                return
            super().reject()

    def closeEvent(self, event):
        if self.is_busy():
            self.message.setPlainText('Agent 回复或样片制作仍在进行，可以最小化等待；已有文件会保留。')
            event.ignore()
        else:
            super().closeEvent(event)


class BatchResults(QDialog):
    """View the most recent on-disk batch without changing the current queue."""
    def __init__(self, directory, parent=None):
        super().__init__(parent)
        self.directory = Path(directory).expanduser().resolve()
        state = json.loads((self.directory / 'batch.json').read_text(encoding='utf-8'))
        self.items = state['items']
        self.sources = state.get('sources') or list(self.items)
        if not isinstance(self.items, dict) or not isinstance(self.sources, list) or not self.sources or not all(
                isinstance(source, str) for source in self.sources):
            raise ValueError('上次制作记录中没有有效照片')
        self.error = state.get('error', '')
        self.setWindowTitle('上次制作 · ' + state.get('character', 'CosLuma'))
        self.resize(1050, 700)
        layout = QVBoxLayout(self)
        complete = sum(result.get('status') == 'complete' for result in self.items.values())
        failed = sum(result.get('status') == 'failed' for result in self.items.values())
        summary = QLabel(f'完成 {complete} 张，失败 {failed} 张，未完成 {len(self.sources)-complete-failed} 张。仅查看已保存结果，不发起制作。')
        summary.setWordWrap(True)
        layout.addWidget(summary)
        splitter = QSplitter(Qt.Horizontal)
        self.listing = QListWidget()
        for source in self.sources:
            result = self.items.get(source, {})
            stage = {'complete': '完成', 'failed': '失败'}.get(result.get('status'), '未完成')
            self.listing.addItem(f'{Path(source).name} · {stage}')
            self.listing.item(self.listing.count()-1).setToolTip(source)
        splitter.addWidget(self.listing)
        self.preview = PhotoPreview()
        splitter.addWidget(self.preview)
        splitter.setSizes([250, 800])
        layout.addWidget(splitter, 1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(92)
        layout.addWidget(self.details)
        controls = QHBoxLayout()
        self.output = QPushButton('打开成片')
        self.output.clicked.connect(lambda: self.open_path(self.output_path))
        controls.addWidget(self.output)
        self.work = QPushButton('打开此照片文件夹')
        self.work.clicked.connect(lambda: self.open_path(self.work_directory))
        controls.addWidget(self.work)
        folder = QPushButton('打开本批次文件夹')
        folder.clicked.connect(lambda: self.open_path(str(self.directory)))
        controls.addWidget(folder)
        controls.addStretch(1)
        close = QPushButton('返回当前照片')
        close.clicked.connect(self.accept)
        controls.addWidget(close)
        layout.addLayout(controls)
        self.listing.currentRowChanged.connect(self.show_photo)
        self.listing.setCurrentRow(0)

    @staticmethod
    def open_path(path):
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def saved_path(self, result, value):
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            return str(self.directory / path)
        previous = result.get('batch_directory')
        if previous:
            try:
                return str(self.directory / path.relative_to(Path(previous)))
            except ValueError:
                pass
        return str(path)

    def show_photo(self, row):
        if row < 0:
            return
        source = self.sources[row]
        result = self.items.get(source, {})
        self.output_path = self.saved_path(result,
            result.get('exports', {}).get('JPEG') or result.get('files', {}).get('jpeg'))
        if result.get('status') != 'complete':
            self.output_path = None
        self.work_directory = self.saved_path(result, result.get('directory')) or str(self.directory)
        self.preview.set_photo(source, self.output_path)
        self.output.setEnabled(bool(self.output_path) and Path(self.output_path).is_file())
        self.work.setEnabled(Path(self.work_directory).is_dir())
        self.work.setVisible(bool(result.get('directory')))
        if result.get('status') == 'complete':
            width, height = result['output_dimensions']
            message = f'{Path(source).name} · 成片 {width} × {height} · 16 位 PNG 和 JPEG'
            if not self.output.isEnabled():
                message += '\n成片文件已移动或删除，请在照片文件夹中查找。'
        else:
            message = f'{Path(source).name}\n' + str(result.get('error') or self.error or '未保存完整成片。')
        self.details.setPlainText(message)


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
            if service == 'image':
                self.same_service = QPushButton('使用提示词服务的地址和凭据')
                self.same_service.clicked.connect(self.use_prompt_service)
                form.addRow(self.same_service)
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
        navigation = QHBoxLayout()
        self.previous = QPushButton('上一步')
        self.next = QPushButton('下一步')
        self.previous.clicked.connect(lambda: self.tabs.setCurrentIndex(self.tabs.currentIndex()-1))
        self.next.clicked.connect(lambda: self.tabs.setCurrentIndex(self.tabs.currentIndex()+1))
        navigation.addWidget(self.previous)
        navigation.addStretch(1)
        navigation.addWidget(self.next)
        layout.addLayout(navigation)
        self.tabs.currentChanged.connect(self.update_navigation)
        self.update_navigation()
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

    def update_navigation(self):
        self.previous.setEnabled(self.tabs.currentIndex() > 0)
        self.next.setEnabled(self.tabs.currentIndex() < self.tabs.count()-1)

    def use_prompt_service(self):
        try:
            url = urlsplit(endpoint_url(self.fields['llm', 'url'].text(), 'llm'))
            if not url.path.endswith('/chat/completions'):
                raise ValueError('提示词接口使用自定义路径，请分别填写两个完整接口地址。')
            self.fields['image', 'url'].setText(urlunsplit(url._replace(
                path=url.path[:-len('/chat/completions')] + '/images/edits')))
            origin_mode, origin_value = self.credentials['llm']
            target_mode, target_value = self.credentials['image']
            mode = origin_mode.currentData()
            if mode == 'keep':
                provider = self.values['llm']
                mode = 'api_key_file' if provider.get('api_key_file') else 'api_key_env'
                value = provider[mode]
            else:
                value = origin_value.text()
            if mode and not value.strip():
                raise ValueError('请先填写提示词服务的密钥，或选择正确的凭据来源。')
            target_mode.setCurrentIndex(target_mode.findData(mode))
            target_value.setText(value)
            self.message.setText('已填入同一服务的图像编辑地址和凭据；请填写图像模型名称。')
        except ValueError as error:
            self.message.setText(str(error))

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
        self.resize(1160, 800)
        self.setAcceptDrops(True)
        self.worker = None
        self.close_after_finish = False
        self.tray = None
        self.selected_style = None
        self.style_context = None
        self.style_dialog = None
        self.style_samples_directory = None
        self.style_sample_source = None
        self.style_samples_context = None
        self.style_discussion_directory = None
        self.style_discussion_context = None
        self.current_output = None
        self.batch_directory = None
        self.last_batch_directory = self.preferences.value('last_batch_directory', '', type=str)
        self.batch_error = ''
        self.photo_errors = {}
        self.stages = {}
        self.preview_key = None
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
        header.addWidget(ui_label('照片、角色与共同创作'), 1)
        header.addWidget(ui_label(VERSION,'badge'))
        layout.addLayout(header)
        setup = QHBoxLayout()
        self.setup_hint = QLabel('')
        self.setup_hint.setProperty('role','muted')
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
        self.last_batch_button = QToolButton()
        self.last_batch_button.setText('查看已有结果')
        self.last_batch_button.setPopupMode(QToolButton.MenuButtonPopup)
        saved_menu = QMenu(self)
        self.last_batch_action = saved_menu.addAction('查看上次制作', self.open_last_batch)
        saved_menu.addAction('选择其他批次文件夹', self.choose_saved_batch)
        self.last_batch_button.setMenu(saved_menu)
        self.last_batch_button.clicked.connect(self.open_last_batch)
        toolbar.addWidget(self.last_batch_button)
        layout.addLayout(toolbar)
        self.inputs.append(self.last_batch_button)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(16)
        self.queue_pages = QStackedWidget()
        empty = QLabel('把照片或文件夹拖到这里\n\n支持 JPEG、PNG、TIFF 和相机 RAW')
        empty.setAlignment(Qt.AlignCenter)
        empty.setStyleSheet('background: #f7f9fb; color: #6c8094; border: 1px dashed #c7d5e0; border-radius: 12px; padding: 25px')
        self.queue_pages.addWidget(empty)
        self.listing = QTableWidget(0, 2)
        self.listing.setHorizontalHeaderLabels(['照片', '制作进度'])
        self.listing.verticalHeader().hide()
        self.listing.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.listing.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.listing.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.listing.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listing.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.listing.setAcceptDrops(False)
        self.listing.viewport().setAcceptDrops(False)
        self.listing.itemSelectionChanged.connect(self.show_selected)
        self.queue_pages.addWidget(self.listing)
        queue_panel = QFrame()
        queue_panel.setProperty('role','panel')
        queue_layout = QVBoxLayout(queue_panel)
        queue_layout.setContentsMargins(14,14,14,14)
        queue_layout.setSpacing(12)
        queue_layout.addWidget(self.queue_pages, 1)
        splitter.addWidget(queue_panel)
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.photo_preview = PhotoPreview()
        preview_layout.addWidget(self.photo_preview, 1)
        self.output_info = QLabel('')
        self.output_info.setWordWrap(True)
        self.output_info.setTextFormat(Qt.PlainText)
        self.output_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details = QScrollArea()
        details.setWidgetResizable(True)
        details.setFrameShape(QScrollArea.NoFrame)
        details.setMinimumHeight(56)
        details.setMaximumHeight(88)
        details.setWidget(self.output_info)
        preview_layout.addWidget(details)
        photo_controls = QHBoxLayout()
        self.open_source_button = QPushButton('打开原片')
        self.open_source_button.setEnabled(False)
        self.open_source_button.clicked.connect(self.open_source)
        photo_controls.addWidget(self.open_source_button)
        self.open_work_button = QPushButton('打开此照片文件夹')
        self.open_work_button.setEnabled(False)
        self.open_work_button.clicked.connect(self.open_work_directory)
        photo_controls.addWidget(self.open_work_button)
        self.open_photo_button = QPushButton('打开选中成片')
        self.open_photo_button.setEnabled(False)
        self.open_photo_button.clicked.connect(self.open_selected_output)
        photo_controls.addWidget(self.open_photo_button)
        preview_layout.addLayout(photo_controls)
        splitter.addWidget(preview_panel)
        splitter.setSizes([390, 710])
        splitter.splitterMoved.connect(lambda *_: self.photo_preview.update_preview())
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
        self.character.textChanged.connect(self.update_actions)
        self.brief.setPlaceholderText('可以留空；有希望保留的景物、场景想法或偏好时再补充')
        self.brief.setMaximumHeight(65)
        self.brief.textChanged.connect(self.update_actions)
        form.addRow('补充要求', self.brief)
        self.root = QLineEdit(str(root))
        self.root.textChanged.connect(self.update_actions)
        output_row = self.path_row(self.root, directory=True)
        form.addRow('保存到', output_row)
        queue_layout.addLayout(form)
        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_toggle = QToolButton()
        advanced_toggle.setText('更多设置')
        advanced_toggle.setCheckable(True)
        advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        advanced_toggle.setArrowType(Qt.RightArrow)
        advanced_layout.addWidget(advanced_toggle)
        advanced_panel = QWidget()
        advanced_form = QFormLayout(advanced_panel)
        self.config = QLineEdit(str(config))
        self.config.textChanged.connect(self.update_setup_hint)
        self.config.textChanged.connect(self.update_actions)
        advanced_form.addRow('配置文件', self.path_row(self.config))
        self.edge = QLineEdit(self.preferences.value('output_edge', '', type=str))
        self.edge.textChanged.connect(self.update_actions)
        self.edge.setPlaceholderText('留空使用配置；默认请求长边 6000')
        advanced_form.addRow('请求长边', self.edge)
        advanced_layout.addWidget(advanced_panel)
        advanced_panel.hide()
        advanced_toggle.toggled.connect(advanced_panel.setVisible)
        advanced_toggle.toggled.connect(lambda checked: advanced_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow))
        queue_layout.addWidget(advanced)
        style_box = QWidget()
        style_layout = QVBoxLayout(style_box)
        style_layout.setContentsMargins(0, 0, 0, 0)
        self.style_hint = QLabel('先填写角色、添加照片，与 Agent 讨论并试样，定稿后再批量制作。')
        self.style_hint.setWordWrap(True)
        self.style_hint.setTextFormat(Qt.PlainText)
        style_layout.addWidget(self.style_hint)
        self.style_button = QPushButton('与 Agent 讨论风格')
        self.style_button.setProperty('kind','primary')
        self.style_button.clicked.connect(self.choose_style)
        style_layout.addWidget(self.style_button)
        queue_layout.addWidget(style_box)
        controls = QHBoxLayout()
        self.start = QPushButton('开始制作')
        self.start.setProperty('kind','primary')
        self.start.setMinimumHeight(38)
        self.start.clicked.connect(lambda: self.produce(only_pending=True))
        controls.addWidget(self.start, 1)
        self.remaining_button = QPushButton('全部另做一批')
        self.remaining_button.clicked.connect(self.produce)
        self.remaining_button.hide()
        controls.addWidget(self.remaining_button)
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
        self.status = QLabel('添加照片、填写角色，与 Agent 讨论方向并比较真实样片，定稿后再制作整组。')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.inputs.extend([self.character, self.brief, output_row, advanced, self.setup_button, self.import_button])
        self.inputs.append(style_box)
        self.update_setup_hint()
        self.restore_style_selection()
        self.update_actions()
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = QSystemTrayIcon(self.style().standardIcon(QStyle.SP_ComputerIcon), self)
            self.tray.setToolTip('CosLuma')
            menu = QMenu(self)
            menu.addAction('显示制作窗口', self.restore_window)
            self.tray.setContextMenu(menu)
            self.tray.activated.connect(self.tray_activated)

    @staticmethod
    def source_key(source):
        return os.path.normcase(str(Path(source).expanduser().resolve()))

    def current_style_context(self):
        return (self.character.text().strip(),self.brief.toPlainText())

    def style_is_current(self):
        return self.selected_style is not None and self.style_context == self.current_style_context()

    def restore_style_selection(self):
        try:
            discussion_state = None
            discussion = self.preferences.value('style/discussion','',type=str)
            if discussion and (Path(discussion)/'conversation.json').is_file():
                state = json.loads((Path(discussion)/'conversation.json').read_text(encoding='utf-8'))
                self.style_discussion_directory = discussion
                self.style_discussion_context = (state['character'],state['brief'])
                discussion_state = state
            directory = self.preferences.value('style/samples', '', type=str)
            if directory and (Path(directory)/'styles.json').is_file():
                state = json.loads((Path(directory)/'styles.json').read_text(encoding='utf-8'))
                self.style_samples_directory = directory
                self.style_samples_context = (state['character'],state['brief'])
            chosen = self.preferences.value('style/selected', '', type=str)
            if chosen:
                self.selected_style = load_style(chosen)
                self.style_context = (self.selected_style['result']['character'],self.selected_style['result']['brief'])
                if discussion_state is not None and self.style_discussion_context==self.style_context and (
                        discussion_state.get('direction','') != self.selected_style['art_direction'].get('creative_direction','')):
                    self.selected_style = None
                    self.style_context = None
                    self.preferences.remove('style/selected')
        except (OSError, ValueError, KeyError, TypeError):
            self.status.setText('上次选中的风格样片已移动或缺失，请重新打开样片选择。')

    def desktop_config(self):
        config = load_settings(self.config.text())
        if self.edge.text().strip():
            edge = int(self.edge.text())
            if edge <= 0:
                raise ValueError('请求长边应为正整数')
            config['output_long_edge'] = edge
        return config

    def choose_style(self):
        if self.worker is not None:
            return
        try:
            if not self.sources or not self.character.text().strip():
                raise ValueError('请先添加照片并填写角色')
            if not self.root.text().strip():
                raise ValueError('请选择保存目录')
            if not Path(self.config.text()).expanduser().is_file():
                self.configure_services()
                if not Path(self.config.text()).expanduser().is_file():
                    return
            config = self.desktop_config()
            context = self.current_style_context()
            row = self.listing.currentRow()
            sample = self.sources[row] if 0 <= row < len(self.sources) else self.sources[0]
            self.remember_preferences()
            previous = self.style_samples_directory if self.style_samples_context == context else None
            discussion = self.style_discussion_directory if self.style_discussion_context == context else None
            dialog = StylePicker(list(self.sources),sample,self.character.text().strip(),
                self.brief.toPlainText(),self.root.text(),config,self,previous=previous,discussion=discussion)
            self.style_dialog = dialog
            accepted = dialog.exec() == QDialog.Accepted
            if dialog.discussion_directory is not None:
                self.style_discussion_directory = dialog.discussion_directory
                self.style_discussion_context = context
                self.preferences.setValue('style/discussion',dialog.discussion_directory)
            if self.selected_style is not None and (dialog.creative_direction != self.selected_style['art_direction'].get('creative_direction','')):
                self.selected_style = None
                self.style_context = None
                self.preferences.remove('style/selected')
            if dialog.directory is not None and dialog.options:
                self.style_samples_directory = dialog.directory
                self.style_samples_context = context
                self.style_sample_source = dialog.sample_source.currentData()
                self.preferences.setValue('style/samples',dialog.directory)
            if accepted:
                self.selected_style = dialog.selected_style
                self.style_context = context
                self.preferences.setValue('style/selected',self.selected_style['selection_file'])
                self.status.setText('已定稿“'+self.selected_style['title']+'”。点击批量制作后，沿用此方向并直接复用选中的完整样片。')
            dialog.deleteLater()
            self.style_dialog = None
            self.preferences.sync()
            self.update_actions()
        except Exception as error:
            self.style_dialog = None
            self.status.setText('无法生成风格样片：'+str(error))

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
            self.status.setText('已选择已有配置。添加照片、填写角色后，可与 Agent 讨论风格并试样。')
        except Exception as error:
            self.status.setText('无法导入配置：' + str(error))

    def open_last_batch(self):
        if self.worker is not None:
            return
        if not self.last_batch_directory:
            self.choose_saved_batch()
            return
        self.show_saved_batch(self.last_batch_directory)

    def choose_saved_batch(self):
        if self.worker is not None:
            return
        directory = QFileDialog.getExistingDirectory(self, '选择一次制作的结果文件夹', self.root.text())
        if directory:
            self.show_saved_batch(directory)

    def show_saved_batch(self, directory):
        try:
            dialog = BatchResults(directory, self)
            dialog.exec()
            dialog.deleteLater()
        except Exception as error:
            self.status.setText('无法查看结果，请选择含 batch.json 的批次文件夹：' + str(error))

    def configure_services(self):
        try:
            dialog = ServiceSettings(self.config.text(), self)
            if dialog.exec() == QDialog.Accepted:
                self.config.setText(str(dialog.path))
                self.remember_preferences()
                self.update_setup_hint()
                self.status.setText('连接设置已保存。添加照片和角色后，可与 Agent 讨论风格并试样。')
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
        if self.worker is not None or not paths:
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
        added = duplicates = skipped = 0
        for path in candidates:
            key = self.source_key(path)
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped += 1
            elif key in known:
                duplicates += 1
            else:
                self.sources.append(str(path))
                known.add(key)
                added += 1
        self.refresh_queue()
        notes = [f'添加 {added} 张照片']
        if duplicates:
            notes.append(f'{duplicates} 张已在队列中')
        if skipped:
            notes.append(f'{skipped} 项不是支持的照片，已跳过')
        self.status.setText('；'.join(notes) + '。')

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
        selected = [self.listing.item(row, 0).data(Qt.UserRole) for row in sorted(
            {item.row() for item in self.listing.selectedItems()}) if self.listing.item(row, 0)]
        current = self.listing.item(self.listing.currentRow(), 0)
        current = current.data(Qt.UserRole) if current else None
        self.listing.blockSignals(True)
        self.rows.clear()
        keys = {self.source_key(source) for source in self.sources}
        self.results = {key: result for key, result in self.results.items() if key in keys}
        self.stages = {key: stage for key, stage in self.stages.items() if key in keys}
        self.photo_errors = {key: error for key, error in self.photo_errors.items() if key in keys}
        self.listing.clearSelection()
        self.listing.setRowCount(len(self.sources))
        for row, source in enumerate(self.sources):
            key = self.source_key(source)
            self.rows[key] = row
            name = QTableWidgetItem(Path(source).name)
            name.setData(Qt.UserRole, source)
            name.setToolTip(source)
            self.listing.setItem(row, 0, name)
            result = self.results.get(key)
            stage = ('完成' if result['status'] == 'complete' else '失败') if result else self.stages.get(key, '等待制作')
            cell = QTableWidgetItem(stage)
            if result and result.get('error'):
                cell.setToolTip(str(result['error']))
            self.listing.setItem(row, 1, cell)
        self.queue_pages.setCurrentIndex(1 if self.sources else 0)
        self.count.setText(f'{len(self.sources)} 张照片' if self.sources else '尚未添加照片')
        if self.worker is None:
            self.progress_bar.setRange(0, max(1, len(self.sources)))
            self.progress_bar.setValue(sum(self.source_key(source) in self.results for source in self.sources))
            self.progress_bar.setFormat('已处理 %v / %m 张' if self.results else '等待开始')
        self.preview_key = None
        selected_keys = {self.source_key(source) for source in selected}
        if self.sources:
            row = self.rows.get(self.source_key(current), 0) if current else 0
            self.listing.setCurrentCell(row, 0, QItemSelectionModel.NoUpdate)
            selection = self.listing.selectionModel()
            for key, selected_row in self.rows.items():
                if key in selected_keys:
                    selection.select(self.listing.model().index(selected_row, 0),
                        QItemSelectionModel.Select | QItemSelectionModel.Rows)
            if not self.listing.selectedItems():
                self.listing.selectRow(row)
        self.listing.blockSignals(False)
        self.show_selected()
        self.update_actions()

    def update_actions(self):
        if not hasattr(self,'style_hint'):
            return
        running = self.worker is not None
        has_photos = bool(self.sources)
        has_character = bool(self.character.text().strip())
        has_previous = bool(self.results or any(stage == '未完成' for stage in self.stages.values()))
        pending = sum(self.source_key(source) not in self.results and self.source_key(source) not in self.stages
            for source in self.sources)
        style_ready = self.style_is_current()
        self.start.setEnabled(not running and pending > 0 and has_character and style_ready)
        self.start.setToolTip('请先添加照片' if not has_photos else
            '请填写角色' if not has_character else
            '请先与 Agent 讨论、查看真实样片并定稿' if not style_ready else
            '只制作新加入的照片，已有成功与失败照片不会再次提交' if pending else
            '本组照片已处理；添加新照片，或明确选择全部另做一批')
        self.start.setText(f'按选定风格制作新增 {pending} 张' if has_previous and pending else
            '本组已处理' if has_previous else '按选定风格批量制作')
        self.remaining_button.setVisible(has_previous)
        self.remaining_button.setEnabled(not running and has_photos and has_character and style_ready)
        self.style_button.setEnabled(not running and has_photos and has_character)
        has_samples = self.style_samples_directory is not None and self.style_samples_context == self.current_style_context()
        self.style_button.setText('查看定稿 / 继续讨论' if style_ready else '继续与 Agent 讨论' if has_samples or
            self.style_discussion_context==self.current_style_context() else '与 Agent 讨论风格')
        self.style_hint.setText('已选择：'+self.selected_style['title']+'。'+self.selected_style['summary'] if style_ready else
            '为“'+self.character.text().strip()+'”讨论摄影方向，满意后定稿。'
            if has_character and has_photos else '先填写角色、添加照片，与 Agent 讨论并试样，定稿后再批量制作。')
        self.last_batch_button.setText('查看上次制作' if self.last_batch_directory else '查看已有结果')
        self.last_batch_button.setEnabled(not running)
        self.last_batch_action.setEnabled(bool(self.last_batch_directory))

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
            self.preview_key = None
            self.current_output = None
            self.photo_preview.set_photo()
            self.output_info.clear()
            self.open_source_button.setEnabled(False)
            self.open_work_button.setEnabled(False)
            self.open_photo_button.setEnabled(False)
            return
        source = self.sources[row]
        result = self.results.get(self.source_key(source))
        self.current_output = None
        self.open_photo_button.setEnabled(False)
        self.open_source_button.setEnabled(Path(source).is_file())
        if result and result['status'] == 'complete':
            path = result.get('exports', {}).get('JPEG', result['files']['jpeg'])
            self.current_output = path
            width, height = result['output_dimensions']
            self.output_info.setText(f'{Path(source).name} · 成片 {width} × {height}\n16 位 PNG 和 JPEG 已保存')
            if result.get('output_adjustment'):
                self.output_info.setText(self.output_info.text() + '\n已按模型实际覆盖范围输出。')
            self.open_photo_button.setEnabled(Path(path).is_file())
            if not self.open_photo_button.isEnabled():
                self.output_info.setText(self.output_info.text() + '\n成片文件已移动或删除，请在照片文件夹中查找。')
        else:
            path = source
            error = result.get('error') if result else self.photo_errors.get(self.source_key(source))
            self.output_info.setText(f'{Path(source).name} · 制作失败\n{error}' if error else
                f'{Path(source).name} · {self.stages.get(self.source_key(source), "等待制作")}')
        work_directory = result.get('directory') if result else None
        self.open_work_button.setText('打开此照片文件夹' if work_directory else '打开本批次文件夹')
        self.open_work_button.setEnabled(bool(work_directory or (result or {}).get('batch_directory') or self.batch_directory))
        preview_key = (source, self.current_output)
        if preview_key != self.preview_key:
            self.photo_preview.set_photo(source, self.current_output)
            self.preview_key = preview_key

    def produce(self, _checked=False, *, only_pending=False):
        if self.worker is not None:
            return
        try:
            if not self.sources or not self.character.text().strip():
                raise ValueError('请添加照片并填写角色')
            if not self.root.text().strip():
                raise ValueError('请选择保存目录')
            if not self.style_is_current():
                self.choose_style()
                return
            if not Path(self.config.text()).expanduser().is_file():
                self.configure_services()
                if not Path(self.config.text()).expanduser().is_file():
                    return
            config = self.desktop_config()
            load_style(self.selected_style['selection_file'])
            self.remember_preferences()
            sources = [source for source in self.sources if not only_pending or (
                self.source_key(source) not in self.results and self.source_key(source) not in self.stages)]
            if not sources:
                return
            if not only_pending:
                self.results.clear()
                self.stages.clear()
                self.photo_errors.clear()
            self.batch_error = ''
            self.batch_directory = None
            self.refresh_queue()
            self.active_sources = {self.source_key(source) for source in sources}
            self.active_processed = 0
            self.progress_bar.setRange(0, len(sources))
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat('已处理 %v / %m 张')
            self.open_folder_button.setEnabled(False)
            self.worker = Worker(sources, self.character.text().strip(), self.brief.toPlainText(),
                self.root.text(), config,style=self.selected_style['selection_file'])
            self.worker.progress.connect(self.status.setText)
            self.worker.photo_stage.connect(self.set_photo_stage)
            self.worker.batch_created.connect(self.batch_started)
            self.worker.result_ready.connect(self.show_result)
            self.worker.failed.connect(self.show_failure)
            self.worker.finished.connect(self.batch_finished)
            for widget in self.inputs:
                widget.setEnabled(False)
            self.start.setEnabled(False)
            self.remaining_button.setEnabled(False)
            self.last_batch_button.setEnabled(False)
            self.status.setText('沿用选中的摄影方向，开始制作本批次')
            self.worker.start()
        except Exception as error:
            self.status.setText('无法开始：' + str(error))

    @Slot(str, str)
    def set_photo_stage(self, source, stage):
        key = self.source_key(source)
        if key in self.rows and key not in self.results:
            self.stages[key] = stage
            self.listing.item(self.rows[key], 1).setText(stage)
            if self.listing.currentRow() == self.rows[key]:
                self.show_selected()

    @Slot(str)
    def batch_started(self, directory):
        self.batch_directory = directory
        self.last_batch_directory = directory
        self.preferences.setValue('last_batch_directory', directory)
        self.preferences.sync()
        self.open_folder_button.setEnabled(True)
        self.show_selected()

    @Slot(dict)
    def show_result(self, result):
        key = self.source_key(result['source'])
        self.results[key] = result
        self.batch_directory = result.get('batch_directory', self.batch_directory)
        self.listing.item(self.rows[key], 1).setText('完成' if result['status'] == 'complete' else '失败')
        self.listing.item(self.rows[key], 1).setToolTip('' if result['status'] == 'complete' else str(result['error']))
        self.active_processed += 1
        self.progress_bar.setValue(self.active_processed)
        self.open_folder_button.setEnabled(bool(self.batch_directory))
        self.show_selected()

    @Slot(str)
    def show_failure(self, message):
        self.status.setText('制作失败：' + message)

    @Slot()
    def batch_finished(self):
        worker, self.worker = self.worker, None
        self.batch_directory = worker.output_directory or self.batch_directory
        if worker.batch_error:
            self.batch_error = worker.batch_error
            for key, row in self.rows.items():
                if key in self.active_sources and key not in self.results:
                    self.stages[key] = '未完成'
                    self.photo_errors[key] = worker.batch_error
                    self.listing.item(row, 1).setText('未完成')
            summary = '本批次未完成。选择照片查看失败步骤与原因，已生成文件保留在批次文件夹。'
        elif worker.errors:
            summary = f'批次未全部完成：成功 {worker.succeeded} 张，失败 {len(worker.errors)} 张。'
        else:
            summary = f'批次完成：成功 {worker.succeeded} 张。'
        self.status.setText(summary + (' 点击失败照片可查看具体原因。' if worker.errors else ''))
        for widget in self.inputs:
            widget.setEnabled(True)
        self.update_actions()
        self.open_folder_button.setEnabled(bool(self.batch_directory))
        self.show_selected()
        worker.deleteLater()
        if self.tray is not None:
            self.tray.hide()
        if self.close_after_finish:
            self.close()
            QApplication.instance().quit()

    def open_selected_output(self):
        if self.current_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.current_output))

    def open_source(self):
        row = self.listing.currentRow()
        if 0 <= row < len(self.sources):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.sources[row]))

    def open_work_directory(self):
        row = self.listing.currentRow()
        if 0 <= row < len(self.sources):
            result = self.results.get(self.source_key(self.sources[row]), {})
            directory = result.get('directory') or result.get('batch_directory') or self.batch_directory
            if directory:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def open_results(self):
        if self.batch_directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.batch_directory)))

    def restore_window(self):
        self.close_after_finish = False
        QApplication.instance().setQuitOnLastWindowClosed(True)
        self.showNormal()
        self.raise_()
        self.activateWindow()
        if self.tray is not None:
            self.tray.hide()

    def tray_activated(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.restore_window()

    def closeEvent(self, event):
        self.remember_preferences()
        if self.style_dialog is not None and self.style_dialog.is_busy():
            self.style_dialog.showNormal()
            self.style_dialog.message.setPlainText('Agent 回复或样片制作仍在进行，请在此查看进度；可以最小化等待。')
            event.ignore()
            return
        if self.worker is not None and self.worker.isRunning():
            if self.tray is not None:
                self.close_after_finish = True
                QApplication.instance().setQuitOnLastWindowClosed(False)
                self.tray.show()
                self.hide()
                self.tray.showMessage('CosLuma 正在制作', '本批次会继续制作并保存。点击托盘图标可返回窗口；完成后退出。')
            else:
                self.status.setText('本批次仍在制作，完成后可以关闭。可最小化窗口，不会中断制作。')
                self.showMinimized()
            event.ignore()
        else:
            if self.tray is not None:
                self.tray.hide()
            super().closeEvent(event)

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
    apply_theme(app)
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
