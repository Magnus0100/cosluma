"""Shared desktop theme, conversation display and native-pixel photo canvas."""
from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
import os
import sys
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPalette, QPixmap, QPixmapCache, QTextDocument, QTransform
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QScrollArea, QStackedWidget, QToolButton,
    QSizePolicy, QVBoxLayout, QWidget)
from shiboken6 import isValid

from image_io import open_display


def apply_theme(app):
    """Keep the desktop, creative workspace and native popup colours consistent."""
    if sys.platform == 'win32':
        font_id = QFontDatabase.addApplicationFont(str(Path(os.environ['SystemRoot'])/'Fonts/msyh.ttc'))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0],10))
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
        QPushButton[role="workspaceTab"] { background: transparent; border: none; border-bottom: 2px solid transparent; border-radius: 0; font-size: 14px; padding: 10px 18px; color: #657588; }
        QPushButton[role="workspaceTab"]:hover { color: #243244; background: #e5edf4; }
        QPushButton[role="workspaceTab"]:checked { color: #244f74; border-bottom-color: #315f87; font-weight: 600; }
        QFrame[role="decisionBar"] { background: #ffffff; border: 1px solid #cbd8e3; border-radius: 10px; }
        QLabel[role="chosenPreview"] { background: #202934; color: #c0ccd7; border-radius: 6px; }
        QLabel[role="dropArea"] { background: #f7f9fb; color: #657588; border: 1px dashed #bdcbd7; border-radius: 8px; padding: 18px; }
        QPushButton:disabled, QToolButton:disabled { color: #8d9baa; background: #eaf0f4; border-color: #dbe4eb; }
        QPushButton[kind="primary"] { background: #315f87; color: #ffffff; border: 1px solid #315f87; font-weight: 600; }
        QPushButton[kind="primary"]:hover { background: #3c719e; }
        QPushButton[kind="primary"]:pressed { background: #244a6c; }
        QPushButton[kind="primary"]:disabled { background: #dce5ec; color: #8b9ba9; border-color: #d1dce5; }
        QPushButton[kind="quiet"], QToolButton[kind="quiet"] { background: transparent; border-color: transparent; color: #586e83; }
        QPushButton[kind="quiet"]:hover, QToolButton[kind="quiet"]:hover { color: #234c70; background: #e5edf4; }
        QToolButton[kind="quiet"]:checked { color: #234c70; background: #e5edf4; border-color: #b7c9d8; }
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
        QListWidget#deliverySheet { background: #f7f9fb; padding: 6px; }
        QListWidget#deliverySheet::item { background: #ffffff; border: 1px solid transparent; border-radius: 8px; padding: 6px; }
        QListWidget#deliverySheet::item:hover { border-color: #b7c9d8; }
        QListWidget#deliverySheet::item:selected { background: #e3edf6; border-color: #7c9db9; }
        QListWidget#setupSteps { background: transparent; border: none; border-radius: 0; }
        QListWidget#setupSteps::item { padding: 14px 10px; border-left: 3px solid transparent; }
        QListWidget#setupSteps::item:selected { background: #e3edf6; border-left-color: #315f87; color: #234b6e; }
        QHeaderView::section { background: #f7f9fb; color: #647a8e; border: none; border-bottom: 1px solid #dce5ed; padding: 9px; }
        QTabWidget::pane { border: 1px solid #d9e3ec; border-radius: 10px; padding: 10px; }
        QTabBar::tab { background: #e8eef4; color: #5d7287; padding: 10px 14px; border-radius: 7px; margin: 2px; }
        QTabBar::tab:selected { background: #ffffff; color: #254f73; }
        QProgressBar { background: #e1e9f0; border: none; border-radius: 5px; text-align: center; color: #345573; min-height: 16px; }
        QProgressBar::chunk { background: #88a9c4; border-radius: 5px; }
        QSlider::groove:horizontal { height: 4px; background: #dce5ed; border-radius: 2px; }
        QSlider::sub-page:horizontal { background: #315f87; border-radius: 2px; }
        QSlider::handle:horizontal { background: #ffffff; border: 2px solid #315f87; width: 14px; height: 14px; border-radius: 9px; margin: -7px 0; }
        QSlider::handle:horizontal:disabled { border-color: #bdcbd7; background: #eef1f4; }
        QSlider::sub-page:horizontal:disabled { background: #bdcbd7; }
        QSplitter::handle { background: transparent; width: 10px; }
        QSplitter::handle:hover { background: #cfdfec; }
        QMenu { background: #ffffff; border: 1px solid #d4dfe8; padding: 5px; }
        QMenu::item { padding: 8px 18px; border-radius: 5px; }
        QMenu::item:selected { background: #e6eff6; }
        QToolTip { color: #243244; background: #ffffff; border: 1px solid #c1d1df; padding: 7px; }
        QToolButton[role="samplePreview"] { background: #12151d; border: 1px solid #303849; border-radius: 9px; color: #a7b8c8; padding: 8px; }
        QToolButton[role="samplePreview"]:hover { border-color: #8cb4d4; }
        QToolButton[role="samplePreview"]:checked { border: 2px solid #5e9bca; background: #243b50; }
        QGraphicsView[role="photoCanvas"] { background: #202934; border: 1px solid #202934; border-radius: 10px; }
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
        welcome_layout.setContentsMargins(18,20,18,20)
        welcome_layout.setSpacing(12)
        welcome_layout.addWidget(ui_label('从这组照片聊起','heading'))
        note = ui_label('说说你想保留的细节、角色的气质，或让 Agent 先看看原片。')
        note.setWordWrap(True)
        welcome_layout.addWidget(note)
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
        author = '你' if user else '摄影 Agent'
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
        self.setBackgroundBrush(QColor('#202934'))
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
        self.choice = 'compare'
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
        toolbar.addWidget(QLabel('照片预览'), 1)
        self.mode = QComboBox()
        for label, value in ((output_label, 'output'), (source_label, 'source'), (source_label+' / '+output_label, 'compare')):
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
        self.mode.setItemText(2,source_label+' / '+output_label)

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

