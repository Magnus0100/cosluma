"""Shared desktop theme, conversation display and native-pixel photo canvas."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPalette, QPixmap, QTextDocument
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QScrollArea, QStackedWidget, QToolButton,
    QVBoxLayout, QWidget)

from image_io import open_display


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
        self.fitting = True
        self.empty_message = ''

    def setPixmap(self, pixels):
        if self.photo.pixmap().cacheKey() == pixels.cacheKey():
            return
        self.empty_message = ''
        self.photo.setPixmap(pixels)
        self.setSceneRect(self.photo.boundingRect())
        self.fit_photo()

    def setText(self, text):
        self.photo.setPixmap(QPixmap())
        self.setSceneRect(0,0,0,0)
        self.empty_message = text
        self.viewport().update()

    def fit_photo(self):
        self.fitting = True
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        if not self.photo.pixmap().isNull():
            self.fitInView(self.photo,Qt.KeepAspectRatio)
        self.zoom_changed.emit()

    def set_zoom(self, scale):
        if self.photo.pixmap().isNull():
            return
        self.fitting = False
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        factor = max(.01,min(8.0/self.devicePixelRatioF(),scale))/self.transform().m11()
        self.scale(factor,factor)
        self.zoom_changed.emit()

    def actual_size(self):
        self.set_zoom(1.0/self.devicePixelRatioF())

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



class PreviewLoader(QRunnable):
    """Decode native display pixels in the pool; create pixmaps on the GUI thread."""
    def __init__(self, keys):
        super().__init__()
        self.keys = keys
        self.signals = PreviewSignals()

    def run(self):
        images = {}
        for key in self.keys:
            try:
                images[key] = (open_display(key[0]), '')
            except Exception as error:
                images[key] = (None, str(error))
        self.signals.loaded.emit(images)



class PhotoPreview(QWidget):
    """Read display previews independently of the untouched production images."""
    cache = {}
    errors = {}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.output = None
        self.choice = 'compare'
        self.images = {}
        self.keys = {}
        self.loader = None
        self.load_timer = QTimer(self)
        self.load_timer.setSingleShot(True)
        self.load_timer.setInterval(80)
        self.load_timer.timeout.connect(self.load_pending)
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
        tools = QHBoxLayout()
        self.zoom_label = ui_label('滚轮缩放，拖动查看')
        tools.addWidget(self.zoom_label,1)
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
        self.set_photo()

    @staticmethod
    def image_label():
        return PhotoCanvas()

    def read_preview(self, path, *, full=False):
        if not path:
            return None
        try:
            key = (str(path), Path(path).stat().st_mtime_ns,full)
        except OSError:
            return None
        if key not in self.cache:
            try:
                self.cache[key] = QPixmap.fromImage(open_display(path, max_edge=None if full else 1600))
            except Exception as error:
                self.cache[key] = None
                self.errors[key] = str(error)
            if len(self.cache) > 6:
                removed = next(iter(self.cache))
                self.cache.pop(removed)
                self.errors.pop(removed, None)
        return self.cache[key]

    def set_photo(self, source=None, output=None):
        self.source, self.output = source, output
        self.keys = {}
        for kind,path in (('source',source),('output',output)):
            if path:
                try:
                    self.keys[kind] = (str(path),Path(path).stat().st_mtime_ns,True)
                except OSError:
                    self.keys[kind] = None
        self.images = {kind:self.cache.get(key) for kind,key in self.keys.items()}
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(self.mode.findData(self.choice if output else 'source'))
        self.mode.setEnabled(bool(output))
        self.mode.blockSignals(False)
        self.update_preview()
        self.load_timer.start()

    def load_pending(self):
        if self.loader is not None:
            return
        missing = list(dict.fromkeys(key for key in self.keys.values() if key is not None and key not in self.cache))
        if missing:
            self.loader = PreviewLoader(missing)
            self.loader.signals.loaded.connect(self.previews_loaded)
            QThreadPool.globalInstance().start(self.loader)

    @Slot(object)
    def previews_loaded(self, images):
        self.loader = None
        for key,(image,error) in images.items():
            self.cache[key] = QPixmap.fromImage(image) if image is not None else None
            if error:
                self.errors[key] = error
        while len(self.cache) > 6:
            removed = next(iter(self.cache))
            self.cache.pop(removed)
            self.errors.pop(removed, None)
        self.images = {kind:self.cache.get(key) for kind,key in self.keys.items()}
        self.update_preview()
        self.load_pending()

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
                label.setPixmap(pixels)
            elif not self.source:
                label.setText('选择照片查看预览\n完成后可对比原片与成片')
            elif self.keys.get(kind) is not None and self.keys[kind] not in self.cache:
                label.setText('正在读取照片…')
            elif self.keys.get(kind) in self.errors:
                label.setText(('读取成片失败\n' if kind == 'output' else '读取原片失败\n')+self.errors[self.keys[kind]])
            elif kind == 'output':
                label.setText('成片暂时无法预览\n可打开成片或批次文件夹查看文件')
            elif not Path(self.source).is_file():
                label.setText('原片不在原来的位置\n已保存的成片仍可查看')
            else:
                label.setText('原片暂时无法预览\n可使用“打开原片”查看文件')
        self.update_zoom_label()

    def zoom_views(self, action):
        views = (self.before,self.after) if self.pages.currentIndex() else (self.preview,)
        for view in views:
            getattr(view,action)()

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

