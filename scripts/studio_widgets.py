"""Shared desktop theme, conversation display and native-pixel photo canvas."""
from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
import os
import sys
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot, QRectF
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QKeySequence, QPainter, QPainterPath, QPen, QPalette, QPixmap, QPixmapCache, QShortcut, QTextDocument, QTransform
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QScrollArea, QStackedWidget, QToolButton,
    QSizePolicy, QVBoxLayout, QWidget)
from shiboken6 import isValid

from image_io import open_display


def studio_icon(name, colour='#b9ceca'):
    """Small vector-drawn navigation marks; no image decoding or asset dependency."""
    pixmap = QPixmap(48,48)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(2,2)
    painter.setPen(QPen(QColor(colour),1.6,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin))
    if name=='projects':
        painter.drawRoundedRect(QRectF(3,4,7,16),2,2)
        painter.drawRoundedRect(QRectF(14,4,7,7),2,2)
        painter.drawRoundedRect(QRectF(14,15,7,5),2,2)
    elif name in ('photos','works'):
        painter.drawRoundedRect(QRectF(3,4,18,16),2,2)
        painter.drawEllipse(QRectF(14,7,3,3))
        path = QPainterPath()
        path.moveTo(4,17); path.lineTo(9,11); path.lineTo(14,17)
        painter.drawPath(path)
        if name=='works':
            painter.setBrush(QColor('#182f33'))
            painter.drawEllipse(QRectF(14,14,9,9))
            path = QPainterPath()
            path.moveTo(16,18); path.lineTo(18,20); path.lineTo(21,16)
            painter.drawPath(path)
    elif name=='create':
        path = QPainterPath()
        path.moveTo(5,4); path.lineTo(19,4); path.quadTo(21,4,21,6)
        path.lineTo(21,15); path.quadTo(21,17,19,17)
        path.lineTo(10,17); path.lineTo(5,21); path.lineTo(5,17)
        path.quadTo(3,17,3,15); path.lineTo(3,6); path.quadTo(3,4,5,4)
        painter.drawPath(path)
        painter.drawLine(8,10,16,10); painter.drawLine(12,6,12,14)
    else:
        for x,y in ((5,8),(12,16),(19,10)):
            painter.drawLine(x,3,x,21)
            painter.setBrush(QColor('#182f33'))
            painter.drawEllipse(QRectF(x-2,y-2,4,4))
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)


def apply_theme(app):
    """Keep the desktop, creative workspace and native popup colours consistent."""
    if sys.platform == 'win32':
        font_id = QFontDatabase.addApplicationFont(str(Path(os.environ['SystemRoot'])/'Fonts/msyh.ttc'))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0],10))
    app.setStyle('Fusion')
    palette = QPalette()
    for role, colour in ((QPalette.Window,'#f4f6f5'),(QPalette.WindowText,'#20332f'),
            (QPalette.Base,'#ffffff'),(QPalette.AlternateBase,'#f0f3f1'),
            (QPalette.Text,'#20332f'),(QPalette.Button,'#ffffff'),
            (QPalette.ButtonText,'#20332f'),(QPalette.Highlight,'#16796f'),
            (QPalette.HighlightedText,'#ffffff'),(QPalette.PlaceholderText,'#86948e'),
            (QPalette.ToolTipBase,'#ffffff'),(QPalette.ToolTipText,'#20332f')):
        palette.setColor(role,QColor(colour))
    palette.setColor(QPalette.Disabled,QPalette.Text,QColor('#98a39d'))
    palette.setColor(QPalette.Disabled,QPalette.ButtonText,QColor('#98a39d'))
    app.setPalette(palette)
    app.setStyleSheet('''
        QWidget { color: #20332f; font-size: 13px; }
        QMainWindow, QDialog { background: #f4f6f5; }
        QLabel { background: transparent; }
        QLabel[role="title"] { font-size: 26px; font-weight: 600; }
        QLabel[role="heading"] { font-size: 16px; font-weight: 600; }
        QLabel[role="muted"] { color: #6c7c75; }
        QLabel[role="eyebrow"] { color: #577e74; font-size: 11px; font-weight: 600; }
        QLabel[role="badge"] { color: #16796f; background: #e3f0eb; border-radius: 8px; padding: 5px 10px; }
        QFrame[role="panel"], QWidget[role="panel"] { background: #ffffff; border: none; border-radius: 12px; }
        QFrame[role="composer"] { background: #ffffff; border: 1px solid #d8e1db; border-radius: 12px; }
        QFrame[role="userBubble"] { background: #e1f0e9; border-radius: 11px; }
        QFrame[role="agentBubble"] { background: #ffffff; border: 1px solid #e3eae4; border-radius: 10px; }
        QFrame[role="welcome"] { background: transparent; border: none; }
        QFrame[role="sampleCard"] { background: #f8faf8; border: 1px solid #e0e7e1; border-radius: 11px; }
        QLabel[role="message"] { font-size: 13px; }
        QLabel[role="messageAuthor"] { color: #71847a; font-size: 11px; font-weight: 600; }
        QPushButton, QToolButton { background: #ffffff; border: 1px solid #d8e1db; border-radius: 7px; padding: 8px 12px; }
        QPushButton:hover, QToolButton:hover { background: #eff5f0; border-color: #9ab5a6; }
        QPushButton:pressed, QToolButton:pressed { background: #e4eee7; }
        QPushButton:focus, QToolButton:focus { border-color: #45968a; }
        QPushButton[role="workspaceTab"] { background: transparent; border: none; border-bottom: 2px solid transparent; border-radius: 0; font-size: 14px; padding: 10px 18px; color: #6c7c75; }
        QPushButton[role="workspaceTab"]:hover { color: #20332f; background: #e5eee7; }
        QPushButton[role="workspaceTab"]:checked { color: #12665e; border-bottom-color: #16796f; font-weight: 600; }
        QFrame[role="decisionBar"] { background: #ffffff; border: 1px solid #cbd8e3; border-radius: 10px; }
        QLabel[role="chosenPreview"] { background: #252e2b; color: #c0ccd7; border-radius: 6px; }
        QFrame[role="dropArea"] { background: #f8faf8; border: 1px dashed #c3d0c6; border-radius: 8px; }
        QPushButton:disabled, QToolButton:disabled { color: #8a9b90; background: #edf2ee; border-color: #e0e8e1; }
        QPushButton[kind="primary"] { background: #16796f; color: #ffffff; border: 1px solid #16796f; font-weight: 600; }
        QPushButton[kind="primary"]:hover { background: #208b7f; }
        QPushButton[kind="primary"]:pressed { background: #115d55; }
        QPushButton[kind="primary"]:disabled { background: #e1eae4; color: #8c9e92; border-color: #d6e1d9; }
        QPushButton[kind="quiet"], QToolButton[kind="quiet"] { background: transparent; border-color: transparent; color: #647b70; }
        QPushButton[kind="quiet"]:hover, QToolButton[kind="quiet"]:hover { color: #12665e; background: #e5eee7; }
        QToolButton[kind="quiet"]:checked { color: #12665e; background: #e5eee7; border-color: #c0d1c5; }
        QLineEdit, QPlainTextEdit, QComboBox { background: #ffffff; border: 1px solid #dce4de; border-radius: 8px; padding: 8px; selection-background-color: #16796f; }
        QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus { border-color: #45968a; }
        QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled { color: #869a8d; border-color: #ffffff; }
        QPlainTextEdit[role="composer"] { border: none; background: transparent; padding: 0; }
        QPlainTextEdit[role="summary"] { border: none; background: #eef3ef; padding: 10px; }
        QPlainTextEdit[role="batchDetails"] { border: none; background: transparent; padding: 4px 0; }
        QComboBox { padding-right: 26px; }
        QComboBox::drop-down { width: 22px; border: none; }
        QComboBox QAbstractItemView { background: #ffffff; border: 1px solid #c0d1c5; selection-background-color: #dceee3; padding: 4px; outline: 0; }
        QScrollArea { border: none; background: transparent; }
        QScrollArea > QWidget > QWidget { background: transparent; }
        QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
        QScrollBar::handle:vertical { background: #c3cec6; border-radius: 3px; min-height: 28px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        QScrollBar:horizontal { background: transparent; height: 8px; }
        QScrollBar::handle:horizontal { background: #c3cec6; border-radius: 3px; min-width: 28px; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        QGroupBox { border: 1px solid #dce5de; border-radius: 10px; margin-top: 12px; padding: 15px 12px 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #5b7668; }
        QTableWidget, QListWidget { background: #ffffff; border: 1px solid #e0e7e1; border-radius: 10px; gridline-color: #edf1ed; outline: 0; }
        QTableWidget::item, QListWidget::item { padding: 9px; border: none; }
        QTableWidget::item:selected, QListWidget::item:selected { background: #e3f0e7; color: #175e50; }
        QListWidget#deliverySheet { background: #f8faf8; padding: 6px; }
        QListWidget#deliverySheet::item { background: #ffffff; border: 1px solid transparent; border-radius: 8px; padding: 6px; }
        QListWidget#deliverySheet::item:hover { border-color: #c0d1c5; }
        QListWidget#deliverySheet::item:selected { background: #e3f0e7; border-color: #74a38c; }
        QListWidget#setupSteps { background: transparent; border: none; border-radius: 0; }
        QListWidget#setupSteps::item { padding: 14px 10px; border-left: 3px solid transparent; }
        QListWidget#setupSteps::item:selected { background: #e3f0e7; border-left-color: #16796f; color: #175e50; }
        QHeaderView::section { background: #f8faf8; color: #708377; border: none; border-bottom: 1px solid #dde7df; padding: 9px; }
        QTabWidget::pane { border: none; padding: 0; }
        QTabBar::tab { background: transparent; color: #6f8176; padding: 8px 12px; border-bottom: 2px solid transparent; min-width: 64px; margin-bottom: 8px; }
        QTabBar::tab:selected { color: #12665e; border-bottom-color: #16796f; }
        QTabBar::tab:hover { background: #e5eee7; }
        QProgressBar { background: #e0e9e2; border: none; border-radius: 5px; text-align: center; color: #306650; min-height: 16px; }
        QProgressBar::chunk { background: #71a38a; border-radius: 5px; }
        QSlider::groove:horizontal { height: 4px; background: #dde7df; border-radius: 2px; }
        QSlider::sub-page:horizontal { background: #16796f; border-radius: 2px; }
        QSlider::handle:horizontal { background: #ffffff; border: 2px solid #16796f; width: 14px; height: 14px; border-radius: 9px; margin: -7px 0; }
        QSlider::handle:horizontal:disabled { border-color: #c3d0c6; background: #f4f6f5; }
        QSlider::sub-page:horizontal:disabled { background: #c3d0c6; }
        QSplitter::handle { background: transparent; width: 10px; }
        QSplitter::handle:hover { background: #d1e4d8; }
        QMenu { background: #ffffff; border: 1px solid #dae4dc; padding: 5px; }
        QMenu::item { padding: 8px 18px; border-radius: 5px; }
        QMenu::item:selected { background: #e5efe7; }
        QToolTip { color: #20332f; background: #ffffff; border: 1px solid #c7d9cc; padding: 7px; }
        QToolButton[role="samplePreview"] { background: #12151d; border: 1px solid #303849; border-radius: 9px; color: #a7b8c8; padding: 8px; }
        QToolButton[role="samplePreview"]:hover { border-color: #8cb4d4; }
        QToolButton[role="samplePreview"]:checked { border: 2px solid #5e9bca; background: #243b50; }
        QGraphicsView[role="photoCanvas"] { background: #252e2b; border: 1px solid #252e2b; border-radius: 10px; }

        QFrame#workspaceRail { background: #182f33; border: none; }
        QLabel[role="brand"] { color: #eff6f2; font-size: 14px; font-weight: 600; }
        QToolButton[role="railTab"] { color: #b9ceca; background: transparent; border: none; border-radius: 9px; padding: 8px 4px; font-size: 12px; }
        QToolButton[role="railTab"]:hover { background: #254247; color: #ffffff; }
        QToolButton[role="railTab"]:checked { background: #2b4e50; color: #ffffff; font-weight: 600; }
        QToolButton[role="railTab"]:disabled { color: #607b7c; background: transparent; }
        QToolButton[role="railTab"]:focus { border: 1px solid #a6c6bc; }
        QToolButton[role="railTab"]::menu-indicator { image: none; }
        QWidget#workspaceHeader { border-bottom: 1px solid #e0e7e1; }
        QLabel[role="operationStatus"] { color: #647b70; font-size: 11px; }
        QListWidget#projectList { background: transparent; border: none; padding: 0; }
        QListWidget#projectList::item { padding: 16px 14px; border-bottom: 1px solid #e3e9e3; border-radius: 8px; }
        QListWidget#projectList::item:selected { background: #e1ede3; color: #205b46; }
        QFrame#projectDetail { background: #ffffff; border: none; border-radius: 12px; }
        QFrame#projectWelcome { background: transparent; border: none; }
        QLabel[role="projectStage"] { color: #247454; background: #e3f0e5; padding: 6px 12px; border-radius: 6px; }
    ''')



def ui_label(text, role='muted'):
    label = QLabel(text)
    label.setProperty('role',role)
    label.setTextFormat(Qt.PlainText)
    return label


def chat_shortcuts(editor, send):
    editor.setToolTip('Ctrl + Enter 发送；Enter 换行')
    shortcuts = []
    for key in ('Ctrl+Return','Ctrl+Enter'):
        shortcut = QShortcut(QKeySequence(key),editor)
        shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        shortcut.activated.connect(send)
        shortcuts.append(shortcut)
    return shortcuts


def chat_composer(editor, button):
    frame = QFrame()
    frame.setProperty('role','composer')
    column = QVBoxLayout(frame)
    column.setContentsMargins(12,12,10,10)
    column.setSpacing(8)
    editor.setProperty('role','composer')
    column.addWidget(editor)
    row = QHBoxLayout()
    row.addStretch(1)
    button.setProperty('kind','primary')
    row.addWidget(button)
    column.addLayout(row)
    return frame



class StatusText(QLabel):
    def __init__(self, text='', parent=None):
        super().__init__(text,parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.PlainText)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setProperty('role','muted')
        if not text:
            self.hide()

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)
        self.setVisible(bool(text))

    def setPlainText(self, text):
        self.setText(text)

    def clear(self):
        self.setText('')

    def toPlainText(self):
        return self.text()




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
        self.entries = []
        self.bubbles = []
        self.following_latest = True
        self.updating_messages = False
        self.next_follow = True
        self.next_position = 0
        self.latest_button = QToolButton(self.viewport())
        self.latest_button.setText('查看最新消息')
        self.latest_button.clicked.connect(self.scroll_to_latest)
        self.latest_button.hide()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.content = QWidget()
        self.rows = QVBoxLayout(self.content)
        self.rows.setContentsMargins(0,0,6,0)
        self.rows.setSpacing(14)
        self.welcome = QFrame()
        self.welcome.setProperty('role','welcome')
        welcome_layout = QVBoxLayout(self.welcome)
        welcome_layout.setContentsMargins(12,14,12,14)
        label = ui_label('暂无对话')
        label.setAlignment(Qt.AlignCenter)
        welcome_layout.addWidget(label)
        self.rows.addWidget(self.welcome)
        self.rows.addStretch(1)
        self.setWidget(self.content)
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.finish_message_update)
        self.verticalScrollBar().valueChanged.connect(self.scroll_position_changed)
        self.verticalScrollBar().rangeChanged.connect(self.scroll_range_changed)

    def make_bubble(self, entry):
        user = entry['role']=='user'
        failed = entry.get('status')=='failed'
        bubble = QFrame()
        bubble.setProperty('role','userBubble' if user else 'agentBubble')
        column = QVBoxLayout(bubble)
        column.setContentsMargins(14,12,14,14)
        column.setSpacing(7)
        header = QHBoxLayout()
        author = '你' if user else '助手'
        header.addWidget(ui_label(author+'（发送失败）' if failed else author,'messageAuthor'),1)
        bubble.copy_button = None
        if not entry.get('_pending'):
            bubble.copy_button = QToolButton()
            bubble.copy_button.setText('复制')
            bubble.copy_button.setToolTip('复制这条消息的完整原文')
            bubble.copy_button.setProperty('kind','quiet')
            bubble.copy_button.setFixedHeight(22)
            bubble.copy_button.setStyleSheet('padding: 2px 6px')
            bubble.copy_button.clicked.connect(lambda _checked=False,text=entry['content']:QApplication.clipboard().setText(text))
            header.addWidget(bubble.copy_button)
        column.addLayout(header)
        bubble.context_label = None
        if user and entry.get('context'):
            bubble.context_label = ui_label('发送时：'+entry['context']['label'])
            bubble.context_label.setWordWrap(True)
            bubble.context_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            bubble.context_label.setStyleSheet('font-size: 11px')
            column.addWidget(bubble.context_label)
        bubble.body = QLabel()
        bubble.body.setProperty('role','message')
        bubble.body.setWordWrap(True)
        bubble.body.setTextInteractionFlags(Qt.TextSelectableByMouse|Qt.LinksAccessibleByMouse)
        if user:
            bubble.body.setTextFormat(Qt.PlainText)
            bubble.body.setText(entry['content'])
        else:
            document = QTextDocument()
            document.setDefaultFont(QApplication.font())
            document.setMarkdown(entry['content'])
            bubble.body.setTextFormat(Qt.RichText)
            bubble.body.setText(document.toHtml())
        column.addWidget(bubble.body)
        if not user and entry.get('direction_changed'):
            evidence = ui_label('本次要求更新依据：'+entry['direction_evidence'])
            evidence.setWordWrap(True)
            evidence.setTextInteractionFlags(Qt.TextSelectableByMouse)
            column.addWidget(evidence)
        if entry.get('direction_reverted'):
            column.addWidget(ui_label('此要求更新已撤回，保留原讨论记录。'))
        if entry.get('operation_status'):
            status = ui_label(entry['operation_status'],'operationStatus')
            status.setWordWrap(True)
            status.setTextInteractionFlags(Qt.TextSelectableByMouse)
            column.addWidget(status)
        bubble.error_label = None
        if failed:
            bubble.error_label = ui_label(entry['error'])
            bubble.error_label.setWordWrap(True)
            bubble.error_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            bubble.error_label.setStyleSheet('color: #a2443e')
            column.addWidget(bubble.error_label)
        return bubble

    def show_messages(self, messages, pending=None, *, scroll_to_bottom=False):
        entries = [dict(entry) for entry in messages]
        if pending is not None:
            entries.append(dict(pending,_pending=True))
        if entries==self.entries:
            if scroll_to_bottom:
                self.scroll_to_latest()
            return
        self.next_follow = scroll_to_bottom or not self.entries or self.following_latest
        self.next_position = self.verticalScrollBar().value()
        self.updating_messages = True
        prefix = 0
        while prefix<min(len(entries),len(self.entries)) and entries[prefix]==self.entries[prefix]:
            prefix += 1
        for bubble in self.bubbles[prefix:]:
            self.rows.removeWidget(bubble)
            bubble.hide()
            bubble.deleteLater()
        del self.bubbles[prefix:]
        self.entries = entries
        self.welcome.setVisible(not entries)
        for entry in entries[prefix:]:
            bubble = self.make_bubble(entry)
            self.rows.insertWidget(self.rows.count()-1,bubble)
            self.bubbles.append(bubble)
        self.update_timer.start(0)

    def finish_message_update(self):
        self.following_latest = self.next_follow
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum() if self.next_follow else self.next_position)
        self.updating_messages = False
        self.scroll_position_changed()

    def scroll_to_latest(self):
        self.following_latest = True
        self.next_follow = True
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        self.update_latest_button()

    def scroll_position_changed(self):
        if not self.updating_messages:
            scrollbar = self.verticalScrollBar()
            self.following_latest = scrollbar.value()==scrollbar.maximum()
        self.update_latest_button()

    def scroll_range_changed(self, _minimum, maximum):
        if not self.updating_messages and self.following_latest:
            self.verticalScrollBar().setValue(maximum)
        self.update_latest_button()

    def update_latest_button(self):
        scrollbar = self.verticalScrollBar()
        self.latest_button.setVisible(scrollbar.maximum()>0 and not self.following_latest)
        self.latest_button.adjustSize()
        self.latest_button.move(self.viewport().width()-self.latest_button.width()-8,
            self.viewport().height()-self.latest_button.height()-8)
        self.latest_button.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_latest_button()



class PhotoCanvas(QGraphicsView):
    zoom_changed = Signal()
    pan_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty('role','photoCanvas')
        self.setMinimumSize(100,160)
        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(QColor('#252e2b'))
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setToolTip('滚轮缩放，按住鼠标拖动；双击或按 0 适应窗口，按 1 查看 100%')
        self.setScene(QGraphicsScene(self))
        self.photo = self.scene().addPixmap(QPixmap())
        self.photo.setTransformationMode(Qt.SmoothTransformation)
        self.display = self.scene().addPixmap(QPixmap())
        self.display.setTransformationMode(Qt.SmoothTransformation)
        self.display.hide()
        self.display_key = None
        self.fitting = True
        self.zoom_intent = None
        self.empty_message = ''

    def setPixmap(self, pixels, *, preserve_view=False):
        if self.photo.pixmap().cacheKey() == pixels.cacheKey():
            return
        keep_view = preserve_view and not self.fitting and self.photo.pixmap().size() == pixels.size()
        self.empty_message = ''
        self.photo.setPixmap(pixels)
        self.setSceneRect(self.photo.boundingRect())
        if not keep_view:
            self.fit_photo()
        self.update_display_copy()

    def setText(self, text):
        self.photo.setPixmap(QPixmap())
        self.setSceneRect(0,0,0,0)
        self.empty_message = text
        self.update_display_copy()
        self.viewport().update()

    def fit_photo(self):
        self.fitting = True
        self.zoom_intent = 'fit'
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        if not self.photo.pixmap().isNull():
            self.fitInView(self.photo,Qt.KeepAspectRatio)
            native = 1.0/self.devicePixelRatioF()
            if self.transform().m11()>native:
                self.resetTransform()
                self.scale(native,native)
                self.centerOn(self.photo)
        self.update_display_copy()
        self.zoom_changed.emit()

    def set_zoom(self, scale):
        if self.photo.pixmap().isNull():
            return
        self.fitting = False
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        factor = max(.01,min(8.0/self.devicePixelRatioF(),scale))/self.transform().m11()
        self.scale(factor,factor)
        self.update_display_copy()
        self.zoom_changed.emit()

    def actual_size(self):
        self.zoom_intent = 'actual'
        self.set_zoom(1.0/self.devicePixelRatioF())

    def update_display_copy(self):
        """Draw a smoothly reduced copy while the whole photo fits; native pixels stay untouched."""
        pixels = self.photo.pixmap()
        key = placement = None
        dpr = self.devicePixelRatioF()
        scale = self.transform().m11()*dpr
        if self.isVisible() and not pixels.isNull() and 0<scale<1:
            ratio = pixels.devicePixelRatio()
            width, height = pixels.width()/ratio,pixels.height()/ratio
            target = max(1,round(width*scale)),max(1,round(height*scale))
            viewport = self.viewport().size()
            if target[0]<=viewport.width()*dpr and target[1]<=viewport.height()*dpr and target[0]<pixels.width():
                key = (pixels.cacheKey(),*target)
                placement = width/target[0],height/target[1]
        if key == self.display_key:
            return
        self.display_key = key
        if key is None:
            self.display.setPixmap(QPixmap())
            self.display.hide()
            return
        copy = pixels.scaled(key[1],key[2],Qt.IgnoreAspectRatio,Qt.SmoothTransformation)
        copy.setDevicePixelRatio(1)
        self.display.setPixmap(copy)
        self.display.setTransform(QTransform.fromScale(*placement))
        self.display.show()

    def wheelEvent(self, event):
        if not self.photo.pixmap().isNull() and event.angleDelta().y():
            self.set_zoom(self.transform().m11()*1.2**(event.angleDelta().y()/120))
            event.accept()
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.fit_photo()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if event.buttons() & Qt.LeftButton and not self.photo.pixmap().isNull():
            self.pan_changed.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_0:
            self.fit_photo()
        elif event.key() == Qt.Key_1:
            self.actual_size()
        elif event.key() in (Qt.Key_Plus,Qt.Key_Equal):
            self.set_zoom(self.transform().m11()*1.2)
        elif event.key() == Qt.Key_Minus:
            self.set_zoom(self.transform().m11()/1.2)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self,'photo') and self.fitting:
            self.fit_photo()
        elif hasattr(self,'display'):
            self.update_display_copy()

    def hideEvent(self, event):
        self.display_key = None
        # Interpreter teardown can delete the scene items before this view is
        # hidden; only an overlay that still exists is cleared then.
        if isValid(self.display):
            self.display.setPixmap(QPixmap())
            self.display.hide()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.update_display_copy()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter,rect)
        if self.empty_message:
            painter.save()
            painter.resetTransform()
            painter.setPen(QColor('#a9b9c8'))
            painter.drawText(self.viewport().rect().adjusted(16,16,-16,-16),
                Qt.AlignCenter|Qt.TextWordWrap,self.empty_message)
            painter.restore()



class PreviewSignals(QObject):
    loaded = Signal(object)
    finished = Signal()


# Fit view never shows more than the viewport, so it reads a bounded copy instead
# of the whole original. This is the same bound the face-range dialog passes to
# open_display; the file's own pixels stay untouched until 100% asks for them.
PREVIEW_MAX_EDGE = 2000


class PreviewLoader(QRunnable):
    """Decode display pixels in the pool; create pixmaps on the GUI thread."""
    def __init__(self, keys, max_edge=PREVIEW_MAX_EDGE):
        super().__init__()
        self.keys = keys
        self.max_edge = max_edge
        self.signals = PreviewSignals()
        self.cancelled = Event()

    def run(self):
        for key in self.keys:
            if self.cancelled.is_set():
                break
            try:
                image, error = open_display(key[0], max_edge=self.max_edge), ''
            except Exception as exception:
                image, error = None, str(exception)
            if not self.cancelled.is_set() and isValid(self.signals):
                self.signals.loaded.emit({key:(image,error)})
            del image
        # Interpreter teardown can delete the signal owner while this pool
        # thread still runs, so only a live signal source is notified.
        if isValid(self.signals):
            self.signals.finished.emit()



class PhotoPreview(QWidget):
    """Read display previews independently of the untouched production images."""
    errors = OrderedDict()

    @staticmethod
    def cached_preview(key):
        return QPixmapCache.find('CosLuma:'+repr(key))

    @staticmethod
    def store_preview(key, pixels, error=''):
        if pixels is not None:
            QPixmapCache.insert('CosLuma:'+repr(key),pixels)
            PhotoPreview.errors.pop(key,None)
        elif error:
            PhotoPreview.errors[key] = error
            while len(PhotoPreview.errors)>32:
                PhotoPreview.errors.popitem(last=False)

    @staticmethod
    def clear_cache():
        QPixmapCache.clear()
        PhotoPreview.errors.clear()

    def __init__(self, parent=None, *, source_label='原片', output_label='成片'):
        super().__init__(parent)
        QPixmapCache.setCacheLimit(64*1024)
        self.source = None
        self.output = None
        self.direct_images = False
        self.direct_sources = {}
        self.native_requested = False
        self.native_zoom = None
        self.image_edges = {}
        self.preserve_view = False
        self.saved_views = {}
        self.choice = 'output'
        self.images = {}
        self.keys = {}
        self.loader = None
        self.syncing = False
        self.load_timer = QTimer(self)
        self.load_timer.setSingleShot(True)
        self.load_timer.setInterval(80)
        self.load_timer.timeout.connect(self.load_pending)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        toolbar.addStretch(1)
        self.mode = QComboBox()
        self.mode.setAccessibleName('查看方式')
        for label, value in ((output_label, 'output'), (source_label, 'source'), ('对比', 'compare')):
            self.mode.addItem(label, value)
        self.mode.currentIndexChanged.connect(self.change_mode)
        self.source_caption = None
        self.output_caption = None
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
        for label, image in ((source_label, self.before), (output_label, self.after)):
            column = QVBoxLayout()
            caption = QLabel(label)
            caption.setAlignment(Qt.AlignCenter)
            if image is self.before:
                self.source_caption = caption
            else:
                self.output_caption = caption
            column.addWidget(caption)
            column.addWidget(image, 1)
            row.addLayout(column, 1)
        self.pages.addWidget(comparison)
        layout.addWidget(self.pages, 1)
        tools = QHBoxLayout()
        self.zoom_label = ui_label('滚轮缩放，拖动查看')
        tools.addWidget(self.zoom_label,1)
        self.linked = QToolButton()
        self.linked.setProperty('kind','quiet')
        self.linked.setText('联动查看')
        self.linked.setCheckable(True)
        self.linked.setChecked(True)
        self.linked.setToolTip('按画面比例同步两侧缩放与查看位置；关闭后可分别检查')
        self.linked.toggled.connect(self.link_views)
        tools.addWidget(self.linked)
        fit = QToolButton()
        fit.setProperty('kind','quiet')
        fit.setText('适应窗口')
        fit.clicked.connect(lambda:self.zoom_views('fit_photo'))
        tools.addWidget(fit)
        actual = QToolButton()
        actual.setProperty('kind','quiet')
        actual.setText('100%')
        actual.setToolTip('按图片原始像素查看细节')
        actual.clicked.connect(lambda:self.zoom_views('actual_size'))
        tools.addWidget(actual)
        layout.addLayout(tools)
        for view in (self.preview,self.before,self.after):
            view.zoom_changed.connect(self.update_zoom_label)
            view.zoom_changed.connect(lambda view=view:self.view_mode_changed(view))
        for view in (self.before,self.after):
            view.zoom_changed.connect(lambda view=view:self.synchronize(view,zoom=True))
            view.pan_changed.connect(lambda view=view:self.synchronize(view,zoom=False))
        self.set_photo()

    def set_labels(self, source_label='原片', output_label='成片'):
        self.source_caption.setText(source_label)
        self.output_caption.setText(output_label)
        self.mode.setItemText(0,output_label)
        self.mode.setItemText(1,source_label)
        self.mode.setItemText(2,'对比')

    @staticmethod
    def image_label():
        return PhotoCanvas()

    def read_preview(self, path, *, full=False):
        if not path:
            return None
        edge = None if full else 360
        try:
            key = (str(path), Path(path).stat().st_mtime_ns,edge)
        except OSError:
            return None
        pixels = self.cached_preview(key)
        if pixels is None and key not in self.errors:
            try:
                pixels = QPixmap.fromImage(open_display(path, max_edge=edge))
                self.store_preview(key,pixels)
            except Exception as error:
                self.store_preview(key,None,str(error))
        return pixels

    def request_edge(self):
        """Read size for the current view: file pixels only for an explicit 100% view."""
        return None if self.native_requested else PREVIEW_MAX_EDGE

    def rebuild_keys(self):
        """Rebuild the cached keys for the current read size, keeping paths only."""
        self.keys = {}
        edge = self.request_edge()
        for kind,path in (('source',self.source),('output',self.output)):
            if path:
                try:
                    self.keys[kind] = (str(path),Path(path).stat().st_mtime_ns,edge)
                except OSError:
                    self.keys[kind] = None

    def image_ready(self, kind):
        """True only while the canvas holds pixels for this key and read size."""
        key = self.keys.get(kind)
        return key is not None and kind in self.images and self.image_edges.get(kind)==key[2]

    def set_photo(self, source=None, output=None, *, preserve_view=False):
        self.direct_images = False
        self.direct_sources = {}
        self.preserve_view = preserve_view
        if not preserve_view:
            # The canvas is fitted to this picture, so the previous picture's
            # 100% request must not make the bounded preview read native pixels.
            self.native_requested = False
            self.native_zoom = None
        self.saved_views = {}
        if preserve_view:
            for view in (self.preview,self.before,self.after):
                if not view.fitting and not view.photo.pixmap().isNull():
                    self.saved_views[view] = (view.photo.pixmap().size(),view.transform().m11(),
                        view.mapToScene(view.viewport().rect().center()))
        previous_keys, previous_images = self.keys,self.images
        self.source, self.output = source, output
        self.rebuild_keys()
        self.images = {kind:previous_images[kind] for kind,key in self.keys.items()
            if self.isVisible() and previous_keys.get(kind)==key and kind in previous_images}
        self.image_edges = {kind:edge for kind,edge in self.image_edges.items() if kind in self.images}
        if self.loader is not None and previous_keys!=self.keys:
            self.loader.cancelled.set()
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(self.mode.findData(self.choice if output else 'source'))
        self.mode.setEnabled(bool(output))
        self.mode.blockSignals(False)
        self.update_preview()
        if self.isVisible():
            self.load_timer.start()

    def direct_images_at(self, native):
        """Canvas pixmaps for the in-memory editor images at the current read size.

        The editor keeps the 16-bit working image itself; this only decides how
        many display pixels the canvas holds.
        """
        edge = None if native else PREVIEW_MAX_EDGE
        images = {}
        built = {}
        for kind, image in self.direct_sources.items():
            if image is None:
                continue
            # The editor often passes the same image for both sides; the pixmap
            # is built once and shared, so the canvas holds one buffer.
            if id(image) not in built:
                display = image
                if edge is not None and max(display.width(), display.height()) > edge:
                    display = display.scaled(edge, edge, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                built[id(image)] = QPixmap.fromImage(display)
            images[kind] = built[id(image)]
        return images

    def set_image_pair(self, before, after):
        """Show the beauty editor's in-memory images with the same native canvas."""
        self.load_timer.stop()
        self.direct_images = True
        self.direct_sources = {'source':before, 'output':after}
        self.images = self.direct_images_at(self.native_requested) if self.isVisible() else {}
        self.image_edges = {kind:self.request_edge() for kind in self.images}
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(self.mode.findData(self.choice))
        self.mode.setEnabled(True)
        self.mode.blockSignals(False)
        self.update_preview()

    def load_pending(self):
        if not self.isVisible() or self.direct_images or self.loader is not None:
            return
        visible = ('source','output') if self.mode.currentData()=='compare' else (self.mode.currentData(),)
        for kind in visible:
            key = self.keys.get(kind)
            if key is not None and not self.image_ready(kind) and key not in self.errors:
                pixels = self.cached_preview(key)
                if pixels is not None:
                    self.images[kind] = pixels
                    self.image_edges[kind] = key[2]
        self.update_preview()
        missing = list(dict.fromkeys(self.keys[kind] for kind in visible
            if self.keys.get(kind) is not None and not self.image_ready(kind)
            and self.keys[kind] not in self.errors))
        if missing:
            self.loader = PreviewLoader(missing[:1], max_edge=self.request_edge())
            self.loader.signals.loaded.connect(self.previews_loaded)
            self.loader.signals.finished.connect(self.previews_finished)
            QThreadPool.globalInstance().start(self.loader)

    @Slot(object)
    def previews_loaded(self, images):
        if not self.isVisible() or self.direct_images:
            return
        edge = self.request_edge()
        images = {key:value for key,value in images.items() if key[2]==edge}
        if not images:
            return
        visible = ('source','output') if self.mode.currentData()=='compare' else (self.mode.currentData(),)
        for key,(image,error) in images.items():
            matching = [kind for kind in visible if self.keys.get(kind)==key]
            if matching:
                pixels = QPixmap.fromImage(image) if image is not None else None
                self.store_preview(key,pixels,error)
                if pixels is not None:
                    self.images.update({kind:pixels for kind in matching})
                    self.image_edges.update({kind:edge for kind in matching})
        self.update_preview()

    @Slot()
    def previews_finished(self):
        self.loader = None
        self.load_pending()

    def change_mode(self):
        if self.output or self.direct_images:
            self.choice = self.mode.currentData()
        self.update_preview()
        if self.isVisible():
            self.load_timer.start()

    def update_preview(self):
        mode = self.mode.currentData()
        has_output = bool(self.output) or self.direct_images
        page = 1 if mode == 'compare' and has_output else 0
        if self.pages.currentIndex() != page:
            self.pages.setCurrentIndex(page)
            QTimer.singleShot(0, self.update_preview)
        if not self.isVisible():
            return
        targets = ((self.before, 'source'), (self.after, 'output')) if self.pages.currentIndex() else (
            (self.preview, mode if has_output else 'source'),)
        self.syncing = True
        active = {label for label,kind in targets}
        for view in (self.preview,self.before,self.after):
            if view not in active:
                view.setText('')
        if not self.direct_images:
            visible = {kind for label,kind in targets}
            self.images = {kind:pixels for kind,pixels in self.images.items() if kind in visible}
        for label, kind in targets:
            pixels = self.images.get(kind)
            if pixels is not None:
                empty_canvas = label.photo.pixmap().isNull()
                anchor = self.anchor_for(label,pixels)
                label.setPixmap(pixels, preserve_view=self.direct_images or self.preserve_view)
                if self.native_requested and self.native_zoom=='actual' and (empty_canvas or anchor is not None):
                    # The explicit 100% applies once, when pixels for that read
                    # size arrive; a later same-size update keeps the zoom the
                    # user has set on the canvas since.
                    label.actual_size()
                elif anchor is not None:
                    label.set_zoom(anchor[2])
                if anchor is not None:
                    label.centerOn(anchor[0]*pixels.width(),anchor[1]*pixels.height())
                previous = self.saved_views.pop(label,None)
                if previous is not None and previous[0]==pixels.size():
                    label.set_zoom(previous[1])
                    label.centerOn(previous[2])
            elif not self.source:
                label.setText('选择照片查看预览\n完成后可对比原片与成片')
            elif self.keys.get(kind) in self.errors:
                label.setText(('读取成片失败\n' if kind == 'output' else '读取原片失败\n')+self.errors[self.keys[kind]])
            elif self.keys.get(kind) is not None:
                label.setText('正在读取照片…')
            elif kind == 'output':
                label.setText('成片暂时无法预览\n可打开成片或批次文件夹查看文件')
            elif not Path(self.source).is_file():
                label.setText('原片不在原来的位置\n已保存的成片仍可查看')
            else:
                label.setText('原片暂时无法预览\n可使用“打开原片”查看文件')
        for view in (self.preview,self.before,self.after):
            view.zoom_intent = None
        self.syncing = False
        self.linked.setEnabled(bool(page))
        self.linked.setVisible(bool(page))
        self.update_zoom_label()

    def release_images(self):
        self.load_timer.stop()
        if self.loader is not None:
            self.loader.cancelled.set()
        # The page comes back at fit size: no native pixels and no in-memory
        # editor images are kept while it is away.
        self.native_requested = False
        self.native_zoom = None
        self.image_edges = {}
        self.direct_sources = {}
        if not self.direct_images:
            self.rebuild_keys()
        self.syncing = True
        for view in (self.preview,self.before,self.after):
            if not view.fitting and not view.photo.pixmap().isNull():
                self.saved_views[view] = (view.photo.pixmap().size(),view.transform().m11(),
                    view.mapToScene(view.viewport().rect().center()))
            view.setText('正在读取照片…')
        self.images.clear()
        self.syncing = False

    def hideEvent(self, event):
        self.release_images()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.update_preview()
        self.load_timer.start()

    def synchronize(self, source, *, zoom):
        if self.syncing or not self.pages.currentIndex() or not self.linked.isChecked() or source.photo.pixmap().isNull():
            return
        target = self.after if source is self.before else self.before
        pixels,other = source.photo.pixmap(),target.photo.pixmap()
        if other.isNull():
            return
        center = source.mapToScene(source.viewport().rect().center())
        self.syncing = True
        if zoom and source.fitting:
            target.fit_photo()
        else:
            if zoom:
                target.set_zoom(source.transform().m11()*pixels.width()/other.width())
            target.centerOn(center.x()/pixels.width()*other.width(),center.y()/pixels.height()*other.height())
        target.zoom_intent = None
        self.syncing = False
        self.update_zoom_label()

    def link_views(self, linked):
        if linked:
            self.synchronize(self.after,zoom=True)

    def anchor_for(self, label, incoming):
        """Relative centre and scale to hold while the canvas changes read size."""
        current = label.photo.pixmap()
        if label.fitting or current.isNull() or current.size()==incoming.size():
            return None
        point = label.mapToScene(label.viewport().rect().center())
        # Scene coordinates are image pixels, so a bigger read needs a smaller
        # scale to keep the same picture on screen.
        scale = label.transform().m11()*current.width()/incoming.width()
        return (point.x()/current.width(),point.y()/current.height(),scale)

    def request_native(self, native, *, exact_zoom=False):
        """Read the file's own pixels only for an explicit 100% view."""
        native = bool(native)
        if native == self.native_requested:
            # Pixels for this size are already requested: an explicit 100% still
            # replaces the pending "keep the apparent size" intent.
            if native and exact_zoom:
                self.native_zoom = 'actual'
            return
        self.native_requested = native
        self.native_zoom = ('actual' if exact_zoom else 'keep') if native else None
        if self.loader is not None:
            self.loader.cancelled.set()
        if self.direct_images:
            self.images = self.direct_images_at(native) if self.isVisible() else {}
            self.image_edges = {kind:self.request_edge() for kind in self.images}
            self.update_preview()
            return
        self.rebuild_keys()
        # The pixmap on screen stays until the read for the new size lands; the
        # canvas then keeps its apparent zoom and centre (anchor_for), and only
        # the replacement is held afterwards.
        if self.isVisible():
            self.load_timer.start()

    def projected_long_edge(self, view):
        """Long edge of the shown pixmap on screen, in device pixels."""
        pixels = view.photo.pixmap()
        if pixels.isNull():
            return 0.0
        return max(pixels.width(),pixels.height())*view.transform().m11()*view.devicePixelRatioF()

    def view_mode_changed(self, view):
        """Keyboard, wheel and toolbar zoom decide the read size for both sides."""
        if self.syncing or view.photo.pixmap().isNull():
            return
        intent,view.zoom_intent = view.zoom_intent,None
        if intent=='actual':
            self.request_native(True,exact_zoom=True)
            return
        if intent=='fit':
            self.request_native(False)
            return
        # Compare the size actually shown: a native 9504 px canvas zoomed out and
        # a 2000 px preview zoomed to fit must not flip the read size back and forth.
        shown = self.projected_long_edge(view)
        if shown>=PREVIEW_MAX_EDGE and not self.native_requested:
            self.request_native(True)
        elif shown<=PREVIEW_MAX_EDGE/2 and self.native_requested:
            self.request_native(False)

    def zoom_views(self, action):
        views = (self.before,self.after) if self.pages.currentIndex() else (self.preview,)
        if action=='actual_size':
            self.request_native(True, exact_zoom=True)
        elif action=='fit_photo':
            self.request_native(False)
        self.syncing = True
        for view in views:
            getattr(view,action)()
            view.zoom_intent = None
        self.syncing = False
        self.update_zoom_label()

    def update_zoom_label(self):
        if not hasattr(self,'zoom_label'):
            return
        views = (self.before,self.after) if self.pages.currentIndex() else (self.preview,)
        zooms = [f'{view.transform().m11()*view.devicePixelRatioF():.0%}' for view in views
            if not view.photo.pixmap().isNull()]
        self.zoom_label.setText(' / '.join(zooms)+' · 滚轮缩放' if zooms else '滚轮缩放，拖动查看')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.update_preview)


class SampleComparison(QWidget):
    """Compare one round's three real samples with shared viewing controls."""
    inspect_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.keys = {}
        self.images = {}
        self.saved_views = {}
        self.loader = None
        self.syncing = False
        self.views, self.titles, self.stages, self.inspect_buttons = [], [], [], []
        self.load_timer = QTimer(self)
        self.load_timer.setSingleShot(True)
        self.load_timer.setInterval(80)
        self.load_timer.timeout.connect(self.load_pending)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        row = QHBoxLayout()
        row.setSpacing(8)
        for index in range(3):
            column = QVBoxLayout()
            column.setSpacing(6)
            title = ui_label(f'方案 {index+1}','heading')
            title.setWordWrap(True)
            title.setFixedHeight(40)
            title.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
            column.addWidget(title)
            view = PhotoCanvas()
            view.setMinimumHeight(100)
            column.addWidget(view,1)
            stage = ui_label('尚未生成')
            stage.setFixedHeight(32)
            stage.setWordWrap(True)
            stage.setSizePolicy(title.sizePolicy())
            column.addWidget(stage)
            inspect = QToolButton()
            inspect.setProperty('kind','quiet')
            inspect.setText('单独查看')
            inspect.setEnabled(False)
            inspect.clicked.connect(lambda _checked=False,index=index:self.inspect_requested.emit(index))
            column.addWidget(inspect)
            row.addLayout(column,1)
            self.views.append(view)
            self.titles.append(title)
            self.stages.append(stage)
            self.inspect_buttons.append(inspect)
            view.zoom_changed.connect(lambda view=view:self.synchronize(view,zoom=True))
            view.pan_changed.connect(lambda view=view:self.synchronize(view,zoom=False))
        layout.addLayout(row,1)
        tools = QHBoxLayout()
        self.zoom_label = ui_label('滚轮缩放，拖动查看')
        self.zoom_label.setSizePolicy(self.titles[0].sizePolicy())
        tools.addWidget(self.zoom_label,1)
        self.linked = QToolButton()
        self.linked.setText('联动查看')
        self.linked.setCheckable(True)
        self.linked.setChecked(True)
        self.linked.setToolTip('同步缩放和查看位置；适应窗口时各自显示完整画面')
        self.linked.toggled.connect(self.link_views)
        tools.addWidget(self.linked)
        for text,action in (('适应窗口','fit_photo'),('100%','actual_size')):
            button = QToolButton()
            button.setProperty('kind','quiet')
            button.setText(text)
            button.clicked.connect(lambda _checked=False,action=action:self.zoom_views(action))
            tools.addWidget(button)
        layout.addLayout(tools)

    def set_samples(self, options, plans, stages):
        keys, images = {}, {}
        self.syncing = True
        for index,view in enumerate(self.views):
            option = options.get(index)
            plan = plans[index] if index<len(plans) else None
            description = option or plan
            self.titles[index].setText(f'方案 {index+1}'+(' · '+description['title'] if description else ''))
            self.titles[index].setToolTip(description['summary'] if description else '')
            self.inspect_buttons[index].setEnabled(option is not None)
            status = stages[index]
            self.stages[index].setText(status)
            self.stages[index].setToolTip(status)
            if option is None or option['status']!='complete':
                view.setText('样片失败\n'+option['error'] if option is not None else status)
                continue
            path = option['result']['files']['jpeg']
            try:
                key = (str(path),Path(path).stat().st_mtime_ns,True)
            except OSError as error:
                view.setText('读取样片失败\n'+str(error))
                continue
            keys[index] = key
            pixels = (self.images.get(index) if self.keys.get(index)==key else PhotoPreview.cached_preview(key)) if self.isVisible() else None
            if pixels is not None:
                images[index] = pixels
                view.setPixmap(pixels)
            elif key in PhotoPreview.errors:
                view.setText('读取样片失败\n'+PhotoPreview.errors[key])
            else:
                view.setText('正在读取样片…')
        self.keys, self.images = keys, images
        if self.loader is not None and any(key not in keys.values() for key in self.loader.keys):
            self.loader.cancelled.set()
        self.syncing = False
        self.update_zoom_label()
        if self.isVisible():
            self.load_timer.start()

    def load_pending(self):
        if not self.isVisible() or self.loader is not None:
            return
        for index,key in self.keys.items():
            if index not in self.images:
                pixels = PhotoPreview.cached_preview(key)
                if pixels is not None:
                    self.show_image(index,pixels)
        missing = list(dict.fromkeys(key for index,key in self.keys.items()
            if index not in self.images and key not in PhotoPreview.errors))
        if missing:
            self.loader = PreviewLoader(missing[:1], max_edge=None)
            self.loader.signals.loaded.connect(self.previews_loaded)
            self.loader.signals.finished.connect(self.previews_finished)
            QThreadPool.globalInstance().start(self.loader)

    @Slot(object)
    def previews_loaded(self, images):
        if not self.isVisible():
            return
        self.syncing = True
        for key,(image,error) in images.items():
            if key not in self.keys.values():
                continue
            pixels = QPixmap.fromImage(image) if image is not None else None
            PhotoPreview.store_preview(key,pixels,error)
            for index,current in self.keys.items():
                if current==key:
                    if pixels is not None:
                        self.show_image(index,pixels)
                    else:
                        self.views[index].setText('读取样片失败\n'+error)
        self.syncing = False
        self.update_zoom_label()

    @Slot()
    def previews_finished(self):
        self.loader = None
        self.load_pending()

    def show_image(self, index, pixels):
        self.images[index] = pixels
        view = self.views[index]
        self.syncing = True
        view.setPixmap(pixels)
        previous = self.saved_views.pop(index,None)
        if previous is not None and previous[0]==self.keys[index]:
            view.set_zoom(previous[1])
            view.centerOn(previous[2])
        self.syncing = False
        self.update_zoom_label()

    def hideEvent(self, event):
        self.load_timer.stop()
        if self.loader is not None:
            self.loader.cancelled.set()
        for index,view in enumerate(self.views):
            if not view.fitting and not view.photo.pixmap().isNull():
                self.saved_views[index] = (self.keys[index],view.transform().m11(),
                    view.mapToScene(view.viewport().rect().center()))
            view.setText('正在读取样片…' if index in self.keys else view.empty_message)
        self.images.clear()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.load_timer.start()

    def synchronize(self, source, *, zoom):
        if self.syncing:
            return
        if not self.linked.isChecked() or source.photo.pixmap().isNull():
            self.update_zoom_label()
            return
        center = source.mapToScene(source.viewport().rect().center())
        pixels = source.photo.pixmap()
        x,y = center.x()/pixels.width(),center.y()/pixels.height()
        self.syncing = True
        for view in self.views:
            if view is source or view.photo.pixmap().isNull():
                continue
            if zoom and source.fitting:
                view.fit_photo()
            else:
                if zoom:
                    view.set_zoom(source.transform().m11())
                view.centerOn(x*view.photo.pixmap().width(),y*view.photo.pixmap().height())
        self.syncing = False
        self.update_zoom_label()

    def link_views(self, linked):
        if linked:
            source = next((view for view in self.views if not view.photo.pixmap().isNull()),None)
            if source is not None:
                self.synchronize(source,zoom=True)

    def zoom_views(self, action):
        self.syncing = True
        for view in self.views:
            getattr(view,action)()
        self.syncing = False
        self.update_zoom_label()

    def update_zoom_label(self):
        zooms = [f'{view.transform().m11()*view.devicePixelRatioF():.0%}' for view in self.views
            if not view.photo.pixmap().isNull()]
        self.zoom_label.setText(' / '.join(zooms) if zooms else '滚轮缩放，拖动查看')

