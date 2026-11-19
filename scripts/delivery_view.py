"""A project contact sheet for explicit version choices and delivery exports."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from threading import Event
import uuid

from PySide6.QtCore import QEvent, QObject, QRunnable, QSignalBlocker, QSize, QThread, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QLineEdit, QListView, QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
    QMenu, QPlainTextEdit, QScrollArea, QSplitter, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from delivery_store import (saved_editor_context, beauty_editor_context, adopt_delivery_versions, export_delivery, photo_choices, project_photos,
    retouch_parent, revision_context, saved_tone_batch, selected_version, source_key)
from creative_edits import normalise_edits
from image_io import open_display
from delivery_chat import DETAIL_MAX_EDGE, discuss_delivery, tone_target_summary
from settings import load_settings
from studio_widgets import PhotoPreview, StatusText, chat_shortcuts, chat_composer, ConversationView, ui_label
from tone_edit import ToneParameters, save_tone_batch


class ThumbnailSignals(QObject):
    ready = Signal(object,object,str)
    finished = Signal()


class ThumbnailLoader(QRunnable):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys
        self.signals = ThumbnailSignals()
        self.cancelled = Event()

    def run(self):
        for key in self.keys:
            if self.cancelled.is_set():
                break
            try:
                image = open_display(key[0],max_edge=360)
                error = ''
            except Exception as exception:
                image,error = None,str(exception)
            if not self.cancelled.is_set():
                self.signals.ready.emit(key,image,error)
            del image
        self.signals.finished.emit()


class ExportWorker(QThread):
    progress = Signal(str)
    ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, photos, choices, parent, name, formats, *, adoption=None):
        super().__init__()
        self.arguments = (deepcopy(photos),deepcopy(choices),parent,name,list(formats))
        self.adoption = deepcopy(adoption)

    def run(self):
        try:
            self.ready.emit(export_delivery(*self.arguments,progress=self.progress.emit,adoption=self.adoption))
        except Exception as error:
            self.failed.emit('导出失败：'+str(error))


class ToneBatchWorker(QThread):
    progress = Signal(str)
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, targets, parameters, directory):
        super().__init__()
        self.targets,self.parameters,self.directory = deepcopy(targets),parameters,Path(directory)

    def run(self):
        try:
            self.ready.emit(save_tone_batch(self.targets,self.parameters,self.directory,progress=self.progress.emit))
        except Exception as error:
            self.failed.emit('批量光色未完成：'+str(error)+'。工作目录：'+str(self.directory))


class RevisionWorker(QThread):
    progress = Signal(str)
    ready = Signal(dict)
    failed = Signal(str)
    directory_created = Signal(str)

    def __init__(self, context, root, config):
        super().__init__()
        self.arguments = (context['source'],context['character'],root,deepcopy(config))
        self.options = {key:deepcopy(context[key]) for key in
            ('art_direction','reference','revision_brief','revision_of','brief','size')}

    def run(self):
        from photo_pipeline import revise_photo
        try:
            result = revise_photo(*self.arguments,**self.options,progress=self.progress.emit,
                batch_started=self.directory_created.emit)
            self.ready.emit(result)
        except Exception as error:
            self.failed.emit('改稿未完成：'+str(error))


class RevisionDialog(QDialog):
    draft_changed = Signal(str,dict)

    def __init__(self, photo, current, drafts, parent=None):
        super().__init__(parent)
        self.setWindowTitle('给这张改稿 · '+photo['name'])
        self.resize(560,440)
        self.versions = [version for version in reversed(photo['versions']) if version['kind']=='base']
        self.drafts = deepcopy(drafts)
        self.previous = None
        layout = QVBoxLayout(self)
        layout.addWidget(ui_label(photo['name'],'heading'))
        layout.addWidget(ui_label('改稿依据 · 基础成片'))
        self.basis = QComboBox()
        self.basis.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        initial = (current['id'] if current['kind']=='base' else
            next(version['id'] for version in self.versions if version['work']==current['work']))
        for version in self.versions:
            self.basis.addItem(version['label'],version['id'])
            self.basis.setItemData(self.basis.count()-1,version['description'],Qt.ToolTipRole)
        self.basis.setCurrentIndex(self.basis.findData(initial))
        layout.addWidget(self.basis)
        note = ui_label('沿用所选基础版的摄影方向。下方选项只影响这张新稿，美化可在新稿完成后另做。')
        note.setWordWrap(True)
        layout.addWidget(note)
        options = QHBoxLayout()
        self.hair = QCheckBox('修发与头发生成')
        self.relight = QCheckBox('人物重打光')
        options.addWidget(self.hair)
        options.addWidget(self.relight)
        options.addStretch(1)
        layout.addLayout(options)
        option_note = ui_label('沿用基础版设置。衣物动态在局部修饰中处理。')
        option_note.setWordWrap(True)
        layout.addWidget(option_note)
        self.requirements = QPlainTextEdit()
        self.requirements.setPlaceholderText('说明这张要改什么、保留什么，例如：改为黄昏侧光，保留当前构图与服装。')
        self.requirements.setMinimumHeight(110)
        layout.addWidget(self.requirements,1)
        boundary = ui_label('改稿会重新生成整幅画面，未指定的其他细节也可能变化。新稿另存，交付选择由你决定。')
        boundary.setWordWrap(True)
        layout.addWidget(boundary)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Cancel).setText('返回')
        buttons.rejected.connect(self.reject)
        self.generate = buttons.addButton('生成一张新稿',QDialogButtonBox.AcceptRole)
        self.generate.setProperty('kind','primary')
        self.generate.clicked.connect(self.accept)
        layout.addWidget(buttons)
        self.basis.currentIndexChanged.connect(self.change_basis)
        self.requirements.textChanged.connect(self.change_draft)
        self.hair.toggled.connect(self.change_draft)
        self.relight.toggled.connect(self.change_draft)
        self.change_basis()

    def change_basis(self, _index=None):
        self.previous = self.basis.currentData()
        saved = self.drafts.get(self.previous)
        self.loaded_edits = normalise_edits(saved['edits'] if saved is not None else self.selected()['edits'])
        self.requirements.blockSignals(True)
        self.requirements.setPlainText(saved['brief'] if saved is not None else '')
        self.requirements.blockSignals(False)
        for control,key in ((self.hair,'hair'),(self.relight,'relight')):
            control.blockSignals(True)
            control.setChecked(self.loaded_edits[key])
            control.blockSignals(False)
        self.generate.setEnabled(bool(self.requirements.toPlainText().strip()))

    def change_draft(self, _checked=None):
        text = self.requirements.toPlainText()
        draft = {'brief':text,'edits':self.edits()}
        self.drafts[self.previous] = draft
        self.draft_changed.emit(self.previous,draft)
        self.generate.setEnabled(bool(text.strip()))

    def edits(self):
        edits = dict(self.loaded_edits)
        for control,key,brief in ((self.hair,'hair','hair_brief'),(self.relight,'relight','lighting_brief')):
            if control.isChecked() and not edits[key]:
                edits[brief] = ''
            edits[key] = control.isChecked()
        return edits

    def selected(self):
        return next(version for version in self.versions if version['id']==self.basis.currentData())


class DeliveryChatWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, context, conversation, config_path, directory):
        super().__init__()
        self.context,self.conversation,self.config_path,self.directory = context,conversation,config_path,directory

    def run(self):
        try:
            self.ready.emit(discuss_delivery(self.context,self.conversation,
                load_settings(self.config_path)['llm'],self.directory))
        except Exception as error:
            self.failed.emit(str(error))


class DeliveryView(QWidget):
    beauty_requested = Signal(dict)
    retouch_requested = Signal(dict)
    tone_requested = Signal(object)
    revision_requested = Signal(object)
    batch_created = Signal(str)
    busy_changed = Signal()
    state_changed = Signal()

    def __init__(self, project, parent=None, *, config_path=None):
        super().__init__(parent)
        self.project = project
        self.config_path = config_path
        self.state = deepcopy(project.data.get('delivery',{}))
        self.state.setdefault('choices',{})
        self.state.setdefault('exports',[])
        self.state.setdefault('formats',['JPEG'])
        self.state.setdefault('comparisons',{})
        self.state.setdefault('revision_drafts',{})
        self.state.setdefault('latest_revisions',{})
        self.state.setdefault('discussion',{'messages':[],'input':'','open':False})
        self.state['discussion'].pop('open',None)
        self.photos = []
        self.rows = {}
        self.worker = None
        self.chat_worker = None
        self.pending_chat_operation = None
        self.revision_message = None
        self.revision_comparison = None
        self.tone_batch_message = None
        self.tone_batch_result = None
        self.export_message_row = None
        self.export_prefix = ''
        self.operation = ''
        self.draft_timer = QTimer(self)
        self.draft_timer.setSingleShot(True)
        self.draft_timer.setInterval(400)
        self.draft_timer.timeout.connect(self.save_state)
        self.production_busy = False
        self.thumbnail_loader = None
        self.thumbnail_keys = {}
        self.thumbnail_cache = {}
        self.thumbnail_errors = {}
        self.thumbnail_rows = set()
        self.thumbnail_timer = QTimer(self)
        self.thumbnail_timer.setSingleShot(True)
        self.thumbnail_timer.setInterval(80)
        self.thumbnail_timer.timeout.connect(self.load_thumbnails)
        self.loaded = False
        self.preview_key = None
        self.preview_photo = None
        self.base_path = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(5)
        filters = QWidget()
        tools = QHBoxLayout(filters)
        tools.setContentsMargins(0,0,0,0)
        self.search = QLineEdit()
        self.search.setPlaceholderText('搜索原片文件名')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.apply_filters)
        tools.addWidget(self.search,1)
        self.only_selected = QCheckBox('只看已选交付')
        self.only_selected.toggled.connect(self.apply_filters)
        tools.addWidget(self.only_selected)
        filters.hide()
        layout.addWidget(filters)
        self.filter_button = QToolButton()
        self.filter_button.setText('筛选')
        self.filter_button.setProperty('kind','quiet')
        self.filter_button.setCheckable(True)
        self.filter_button.toggled.connect(filters.setVisible)
        more = QToolButton()
        more.setText('更多')
        more.setProperty('kind','quiet')
        more.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(more)
        self.select_all = menu.addAction('勾选当前显示',lambda:self.mark_visible(True))
        self.select_none = menu.addAction('取消当前显示',lambda:self.mark_visible(False))
        menu.addSeparator()
        self.refresh_button = menu.addAction('刷新作品',self.refresh)
        self.tone_batch_record = menu.addAction('最近批量光色记录',self.show_tone_batch_record)
        self.adopt_tone_batch = menu.addAction('采用最近光调成功版',self.adopt_latest_tone_batch)
        more.setMenu(menu)
        delivery_settings = QWidget()
        settings_column = QVBoxLayout(delivery_settings)
        settings_column.addWidget(ui_label('交付位置','heading'))
        self.delivery_location = QLineEdit(self.state.get('parent',''))
        self.delivery_location.setReadOnly(True)
        self.delivery_location.setPlaceholderText('首次导出时选择文件夹')
        settings_column.addWidget(self.delivery_location)
        self.location_button = QPushButton('设置交付位置')
        self.location_button.clicked.connect(self.choose_delivery_location)
        settings_column.addWidget(self.location_button)
        self.note = ui_label('勾选交付照片，选择版本')
        self.note.setWordWrap(True)
        settings_column.addWidget(self.note)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(12)
        self.listing = QListWidget()
        self.listing.setObjectName('deliverySheet')
        self.listing.setViewMode(QListView.IconMode)
        self.listing.setFlow(QListView.LeftToRight)
        self.listing.setWrapping(False)
        self.listing.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.listing.setResizeMode(QListView.Adjust)
        self.listing.setMovement(QListView.Static)
        self.listing.setSelectionMode(QAbstractItemView.SingleSelection)
        self.listing.setIconSize(QSize(104,57))
        self.listing.setGridSize(QSize(128,87))
        self.listing.setFixedHeight(106)
        self.listing.setSpacing(2)
        self.listing.setTextElideMode(Qt.ElideRight)
        self.listing.currentRowChanged.connect(self.browse_photo)
        self.listing.itemChanged.connect(self.change_inclusion)
        self.listing.horizontalScrollBar().valueChanged.connect(lambda _value:self.thumbnail_timer.start(80))
        self.listing.viewport().installEventFilter(self)
        self.empty = ui_label('暂无成片')
        self.empty.setWordWrap(True)
        chat_host = QWidget()
        discussion_column = QVBoxLayout(chat_host)
        discussion_column.setContentsMargins(0,0,0,0)
        right = QWidget()
        right.setMinimumWidth(300)
        column = QVBoxLayout(right)
        column.setContentsMargins(0,0,0,0)
        self.photo_title = ui_label('选择作品','heading')
        self.photo_title.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        title_row = QHBoxLayout()
        title_row.addWidget(self.photo_title,1)
        title_row.addWidget(self.filter_button)
        title_row.addWidget(more)
        column.addLayout(title_row)
        self.delivery_version = ui_label('')
        self.delivery_version.setTextFormat(Qt.PlainText)
        self.delivery_version.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        version_row = QHBoxLayout()
        version_row.addWidget(ui_label('当前交付：'))
        version_row.addWidget(self.delivery_version,1)
        self.reference = QComboBox()
        self.reference.setMinimumWidth(190)
        self.reference.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        self.reference.setAccessibleName('比较版本')
        self.reference.setToolTip('选择比较版本只改变查看，不改变当前交付')
        self.reference.currentIndexChanged.connect(self.change_reference)
        self.comparison_note = ui_label('')
        self.comparison_note.setWordWrap(True)
        self.comparison_note.hide()
        column.addWidget(self.comparison_note)
        self.preview = PhotoPreview(source_label='比较版本',output_label='交付版',workbench=True)
        mode_tools = self.preview.workbench_tools
        mode_tools.insertWidget(3,self.reference,1)
        self.preview.linked.setChecked(self.state.get('linked_comparison',True))
        self.preview.linked.toggled.connect(self.change_linked_comparison)
        column.addWidget(self.preview,1)
        details = QHBoxLayout()
        details.addLayout(version_row,1)
        edit_button = QToolButton()
        edit_button.setText('调整照片')
        edit_button.setProperty('kind','quiet')
        edit_button.setPopupMode(QToolButton.InstantPopup)
        edit_menu = QMenu(edit_button)
        self.beauty = edit_menu.addAction('人脸美化',self.open_beauty)
        self.beauty.setEnabled(False)
        self.retouch = edit_menu.addAction('局部修饰',self.open_retouch)
        self.clothing = edit_menu.addAction('衣物动态',lambda:self.open_retouch('clothing'))
        self.retouch.setEnabled(False)
        self.tone = edit_menu.addAction('光色调整',self.open_tone)
        self.tone.setEnabled(False)
        edit_button.setMenu(edit_menu)
        edit_menu.addSeparator()
        self.revise = edit_menu.addAction('重新创作这张',self.open_revision)
        self.compare_revision = QPushButton('比较新稿')
        self.compare_revision.setProperty('kind','quiet')
        self.compare_revision.clicked.connect(self.compare_new_revision)
        details.addWidget(self.compare_revision)
        details.addWidget(edit_button)
        column.addLayout(details)
        self.adopt_revision = QPushButton('采用比较版')
        self.adopt_revision.setProperty('kind','primary')
        self.adopt_revision.clicked.connect(self.adopt_comparison)
        self.preview.mode.currentIndexChanged.connect(self.update_revision_controls)
        self.chat_panel = QWidget()
        chat_column = QVBoxLayout(self.chat_panel)
        chat_column.setContentsMargins(0,0,0,0)
        self.chat_history = ConversationView()
        chat_column.addWidget(self.chat_history,1)
        self.chat_input = QPlainTextEdit()
        self.chat_input.setFixedHeight(54)
        self.chat_input.setPlaceholderText('输入要求')
        self.chat_input.setPlainText(self.state['discussion']['input'])
        self.chat_input.textChanged.connect(self.remember_chat_input)
        self.chat_send = QPushButton('发送')
        self.chat_send.clicked.connect(self.send_chat)
        self.send_shortcuts = chat_shortcuts(self.chat_input,self.send_chat)
        chat_column.addWidget(chat_composer(self.chat_input,self.chat_send))
        discussion_column.addWidget(self.chat_panel,1)
        splitter.addWidget(right)
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setMinimumWidth(280)
        self.sidebar_tabs.setMaximumWidth(360)
        self.sidebar_tabs.addTab(chat_host,'修图要求')
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(delivery_settings)
        self.sidebar_tabs.addTab(settings_scroll,'交付设置')
        splitter.addWidget(self.sidebar_tabs)
        splitter.setStretchFactor(0,1)
        splitter.setStretchFactor(1,0)
        splitter.setSizes([820,310])
        self.sidebar_tabs.hide()
        layout.addWidget(splitter,1)
        layout.addWidget(self.listing)
        layout.addWidget(self.empty)
        footer = QHBoxLayout()
        self.summary = ui_label('已选交付 0 张')
        footer.addWidget(self.summary,1)
        self.include_photo = QCheckBox('交付这张')
        self.include_photo.toggled.connect(self.set_current_inclusion)
        footer.addWidget(self.include_photo)
        settings_column.addWidget(ui_label('导出格式','heading'))
        self.formats = {}
        for name,text in (('JPEG','JPEG（8 位）'),('PNG','PNG（16 位）')):
            check = QCheckBox(text)
            check.setChecked(name in self.state['formats'])
            check.toggled.connect(self.change_formats)
            self.formats[name] = check
            settings_column.addWidget(check)
        self.history = QComboBox()
        self.history.setMinimumWidth(170)
        self.history.setMaximumWidth(240)
        self.history.setPlaceholderText('尚未导出')
        self.history.setToolTip('查看本项目已保存的交付记录')
        self.history.currentIndexChanged.connect(self.show_export_record)
        settings_column.addWidget(ui_label('交付记录','heading'))
        settings_column.addWidget(self.history)
        self.open_export = QPushButton('打开交付文件夹')
        self.open_export.clicked.connect(self.open_selected_export)
        settings_column.addWidget(self.open_export)
        settings_column.addStretch(1)
        for index, title in enumerate(('修图要求','导出设置')):
            button = QToolButton()
            button.setText(title)
            button.setProperty('kind','quiet')
            button.clicked.connect(lambda _checked=False,index=index:self.toggle_sidebar(index))
            footer.addWidget(button)
        footer.addWidget(self.adopt_revision)
        self.export = QPushButton('导出所选作品')
        self.export.setProperty('kind','primary')
        self.export.clicked.connect(self.export_selected)
        footer.addWidget(self.export)
        layout.addLayout(footer)
        self.message = StatusText('')
        self.message.setMaximumHeight(48)
        layout.addWidget(self.message)
        self.refresh_history()
        self.refresh_discussion()
        self.refresh()

    def toggle_sidebar(self, index):
        showing = not self.sidebar_tabs.isVisible() or self.sidebar_tabs.currentIndex()!=index
        self.sidebar_tabs.setCurrentIndex(index)
        self.sidebar_tabs.setVisible(showing)

    def is_busy(self):
        return self.worker is not None or self.chat_worker is not None

    def set_production_busy(self, busy):
        self.production_busy = busy
        self.update_controls()

    def save_state(self):
        self.draft_timer.stop()
        try:
            self.project.save_delivery(self.state)
        except OSError as error:
            self.message.setPlainText('选择仍保留在窗口中，保存项目交付选择失败：'+str(error))
            return False
        self.state_changed.emit()
        return True

    def current_photo(self):
        row = self.listing.currentRow()
        return self.photos[row] if row>=0 else None

    def refresh(self):
        if self.is_busy():
            return
        selected = self.current_photo()
        selected_key = selected['key'] if selected else self.state.get('current')
        try:
            photos = project_photos(self.project.data['batches'])
        except ValueError as error:
            self.message.setPlainText(str(error))
            self.loaded = False
            self.update_controls()
            return
        self.photos = photos
        self.state['choices'] = photo_choices(photos,self.state['choices'])
        self.rows = {photo['key']:row for row,photo in enumerate(photos)}
        self.thumbnail_keys = {}
        self.thumbnail_rows.clear()
        self.listing.blockSignals(True)
        self.listing.clear()
        for photo in photos:
            item = QListWidgetItem()
            item.setFlags(item.flags()|Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if self.state['choices'][photo['key']]['included'] else Qt.Unchecked)
            item.setSizeHint(QSize(126,85))
            item.setData(Qt.UserRole,photo['key'])
            self.listing.addItem(item)
            self.update_tile(photo)
        self.listing.setCurrentRow(self.rows.get(selected_key,0) if photos else -1)
        self.listing.blockSignals(False)
        self.loaded = True
        self.apply_filters()
        self.load_thumbnails()
        self.save_state()

    def update_tile(self, photo):
        item = self.listing.item(self.rows[photo['key']])
        try:
            version = selected_version(photo,self.state['choices'][photo['key']])
            item.setText(photo['name'])
            path = version['files']['JPEG']
            key = (path,Path(path).stat().st_mtime_ns)
        except (OSError,ValueError) as error:
            item.setText(photo['name']+'\n所选版本无法读取')
            item.setToolTip(str(error))
            item.setIcon(QIcon())
            self.thumbnail_keys.pop(photo['key'],None)
            return
        self.thumbnail_keys[photo['key']] = key
        item.setIcon(self.thumbnail_cache.get(key,QIcon()) if self.rows[photo['key']] in self.thumbnail_rows else QIcon())
        item.setToolTip(photo['source']+'\n'+version['label']+(
            '\n读取缩略图失败：'+self.thumbnail_errors[key] if key in self.thumbnail_errors else '')+
            '\n'+version['description'])

    def load_thumbnails(self):
        if not self.isVisible():
            return
        viewport = self.listing.viewport().rect()
        area = viewport.adjusted(-128,0,128,0)
        rows = {row for row in range(self.listing.count())
            if not self.listing.item(row).isHidden() and
                self.listing.visualItemRect(self.listing.item(row)).intersects(area)}
        self.listing.blockSignals(True)
        for row in self.thumbnail_rows-rows:
            if row<self.listing.count():
                self.listing.item(row).setIcon(QIcon())
        self.thumbnail_rows = rows
        retained = {self.thumbnail_keys.get(self.photos[row]['key']) for row in rows}
        self.thumbnail_cache = {key:icon for key,icon in self.thumbnail_cache.items() if key in retained}
        self.thumbnail_errors = {key:error for key,error in self.thumbnail_errors.items() if key in retained}
        for row in rows:
            self.update_tile(self.photos[row])
        self.listing.blockSignals(False)
        self.load_next_thumbnail()

    def load_next_thumbnail(self):
        if not self.isVisible() or self.thumbnail_loader is not None:
            return
        keys = list(dict.fromkeys(self.thumbnail_keys[self.photos[row]['key']] for row in sorted(self.thumbnail_rows)
            if self.photos[row]['key'] in self.thumbnail_keys))
        keys = [key for key in keys
            if key not in self.thumbnail_cache and key not in self.thumbnail_errors]
        if keys:
            self.thumbnail_loader = ThumbnailLoader(keys[:1])
            self.thumbnail_loader.signals.ready.connect(self.thumbnail_ready)
            self.thumbnail_loader.signals.finished.connect(self.thumbnails_finished)
            QThreadPool.globalInstance().start(self.thumbnail_loader)

    @Slot(object,object,str)
    def thumbnail_ready(self, key, image, error):
        matching = [row for row in self.thumbnail_rows if
            self.thumbnail_keys.get(self.photos[row]['key'])==key]
        if not self.isVisible() or not matching:
            return
        if image is not None:
            self.thumbnail_cache[key] = QIcon(QPixmap.fromImage(image))
        else:
            self.thumbnail_errors[key] = error
        self.listing.blockSignals(True)
        for row in matching:
            self.update_tile(self.photos[row])
        self.listing.blockSignals(False)

    @Slot()
    def thumbnails_finished(self):
        self.thumbnail_loader = None
        self.load_next_thumbnail()

    def eventFilter(self, watched, event):
        if watched is self.listing.viewport() and event.type()==QEvent.Resize:
            self.thumbnail_timer.start(80)
        return super().eventFilter(watched,event)

    def hideEvent(self, event):
        self.save_state()
        self.thumbnail_timer.stop()
        if self.thumbnail_loader is not None:
            self.thumbnail_loader.cancelled.set()
        self.listing.blockSignals(True)
        for row in self.thumbnail_rows:
            self.listing.item(row).setIcon(QIcon())
        self.listing.blockSignals(False)
        self.thumbnail_rows.clear()
        self.thumbnail_cache.clear()
        self.thumbnail_errors.clear()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.thumbnail_timer.start(80)

    def apply_filters(self):
        query = self.search.text().strip().casefold()
        selected_key = self.current_photo()['key'] if self.current_photo() else None
        visible = []
        self.listing.blockSignals(True)
        for row,photo in enumerate(self.photos):
            matches = query in photo['name'].casefold() and (not self.only_selected.isChecked() or
                self.state['choices'][photo['key']]['included'])
            self.listing.item(row).setHidden(not matches)
            if matches:
                visible.append(row)
        selected_row = self.rows.get(selected_key)
        self.listing.setCurrentRow(selected_row if selected_row in visible else visible[0] if visible else -1)
        self.listing.blockSignals(False)
        self.empty.setText('没有匹配的作品，请调整文件名或“只看已选交付”。' if self.photos else
            '暂无成片')
        self.empty.setVisible(not visible)
        self.listing.setVisible(bool(visible))
        self.show_photo(self.listing.currentRow())
        self.update_controls()
        if self.isVisible():
            self.thumbnail_timer.start(80)

    def mark_visible(self, included):
        self.listing.blockSignals(True)
        for photo in self.photos:
            item = self.listing.item(self.rows[photo['key']])
            if not item.isHidden():
                self.state['choices'][photo['key']]['included'] = included
                item.setCheckState(Qt.Checked if included else Qt.Unchecked)
        self.listing.blockSignals(False)
        self.save_state()
        self.apply_filters()

    def change_inclusion(self, item):
        self.state['choices'][item.data(Qt.UserRole)]['included'] = item.checkState()==Qt.Checked
        self.save_state()
        self.apply_filters()

    def set_current_inclusion(self, included):
        item = self.listing.currentItem()
        if item is not None:
            item.setCheckState(Qt.Checked if included else Qt.Unchecked)

    def browse_photo(self, row):
        self.show_photo(row)
        if row>=0:
            self.state['current'] = self.photos[row]['key']
            self.save_state()

    def focus_saved_photo(self, source):
        row = self.rows.get(source_key(source))
        if row is None:
            return
        item = self.listing.item(row)
        if item.isHidden():
            with QSignalBlocker(self.search), QSignalBlocker(self.only_selected), QSignalBlocker(self.listing):
                if self.search.text().strip().casefold() not in self.photos[row]['name'].casefold():
                    self.search.clear()
                if not self.state['choices'][self.photos[row]['key']]['included']:
                    self.only_selected.setChecked(False)
                self.listing.setCurrentRow(row)
            self.apply_filters()
            self.state['current'] = self.photos[row]['key']
            self.save_state()
        else:
            self.listing.setCurrentRow(row)
        self.listing.scrollToItem(item)

    def remember_chat_input(self):
        self.state['discussion']['input'] = self.chat_input.toPlainText()
        self.draft_timer.start()

    def refresh_discussion(self):
        entries = []
        for row in self.state['discussion']['messages']:
            context = row.get('context')
            label = (context['name']+' · '+context['selected_label']+
                ' · '+context['view_mode']) if context else ''
            entries.append(dict(row,context={'label':label} if label else None))
        self.chat_history.show_messages(entries)

    def edit_target_id(self, photo):
        if self.preview.mode.currentData() == 'source':
            return self.state['comparisons'].get(photo['key'])
        return self.state['choices'][photo['key']]['version']

    def chat_context(self):
        photo = self.current_photo()
        if photo is None:
            raise ValueError('请先选择一张已保存作品')
        version = selected_version(photo,self.state['choices'][photo['key']])
        comparison = self.state['comparisons'].get(photo['key'])
        previous = selected_version(photo,{'version':comparison}) if comparison is not None else None
        mode = self.preview.mode.currentData()
        images = []
        if mode in ('source','compare'):
            images.append({'label':'左侧比较对象：'+(previous['label'] if previous else '原片'),
                'path':previous['files']['JPEG'] if previous else photo['source']})
        if mode in ('output','compare'):
            images.append({'label':'右侧交付版本：'+version['label'],'path':version['files']['JPEG']})
        detail = None
        if mode=='compare' and previous is not None and previous['id']!=version['id'] and previous['work']==version['work']:
            detail = self.preview.comparison_detail(DETAIL_MAX_EDGE)
        if detail is not None:
            box,crops = detail
            images.extend({'label':image['label']+' · 同位置 100% 局部','pixels':crop}
                for image,crop in zip(images[:2],crops))
        bases = {item['work']:item['id'] for item in photo['versions'] if item['kind']=='base'}
        view_mode = {'source':'只看比较版','output':'只看交付版','compare':'拖动分界对比' if self.preview.split_layout else '并排对比'}[mode]
        return {'photo_key':photo['key'],'name':photo['name'],'selected_version':version['id'],
            'edit_target':self.edit_target_id(photo),
            'selected_label':version['label'],'comparison_version':comparison,
            'view_mode':view_mode,**({'detail_box':box} if detail is not None else {}),
            'versions':[{'id':item['id'],'label':item['label'],'description':item['description'],
                'kind':item['kind'],'edits':deepcopy(item['edits']),
                'revision_basis':bases[item['work']]}
                for item in reversed(photo['versions'])],'images':images,'tone_targets':self.tone_targets(),
            'tone_batches':self.tone_batch_context(),
            'export':{'photos':deepcopy(self.photos),'choices':deepcopy(self.state['choices']),
                'parent':self.state.get('parent'),'formats':list(self.state['formats'])}}

    def tone_batch_context(self):
        batches = []
        for path in reversed(self.state.get('tone_batches',[])[-3:]):
            try:
                batches.append(saved_tone_batch(path))
            except (OSError,ValueError,KeyError) as error:
                batches.append({'id':path,'error':str(error),'targets':[]})
        return batches

    def tone_targets(self):
        targets = []
        photos = {photo['key']:photo for photo in self.photos}
        for key,choice in self.state['choices'].items():
            if not choice['included']:
                continue
            target = {'photo_key':key,'name':Path(choice['source']).name}
            try:
                photo = photos.get(key)
                if photo is None:
                    raise ValueError('交付选择的原片不在当前作品记录中：'+choice['source'])
                version = selected_version(photo,choice)
                target.update(base=version['files']['PNG'],directory=version['work'],
                    parent_version=retouch_parent(version))
            except (ValueError,KeyError) as error:
                target['error'] = str(error)
            targets.append(target)
        return targets

    def send_chat(self):
        text = self.chat_input.toPlainText().strip()
        if not text or self.is_busy() or self.production_busy:
            return
        try:
            context = self.chat_context()
            frozen = {key:context[key] for key in ('photo_key','name','selected_version','selected_label',
                'comparison_version','view_mode','edit_target')}
            frozen['attached_images'] = [image['label'] for image in context['images']]
            if 'detail_box' in context:
                frozen['detail_box'] = context['detail_box']
            frozen['tone_targets'] = tone_target_summary(context['tone_targets'])
            frozen['export'] = {'parent':context['export']['parent'],'formats':context['export']['formats'],
                'choices':{key:choice for key,choice in context['export']['choices'].items() if choice['included']}}
            frozen['tone_batches'] = [{key:value for key,value in item.items() if key!='targets'} for item in context['tone_batches']]
            self.state['discussion']['messages'].append({'role':'user','content':text,'context':frozen})
            self.chat_input.clear()
            self.refresh_discussion()
            if not self.save_state():
                return
            directory = self.project.path.parent/'discussion'/('delivery-'+
                datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
            worker = DeliveryChatWorker(context,deepcopy(self.state['discussion']['messages']),self.config_path,directory)
            self.chat_worker = worker
            self.operation = 'chat'
            worker.ready.connect(self.chat_ready)
            worker.failed.connect(self.chat_failed)
            worker.finished.connect(self.chat_finished)
            self.update_controls()
            self.busy_changed.emit()
            self.message.setPlainText('Agent 正在讨论发送时的 '+context['name']+'，可以继续浏览其他作品。')
            worker.start()
        except (OSError,ValueError,KeyError) as error:
            self.message.setPlainText('无法发送作品讨论：'+str(error))

    def chat_ready(self, result):
        row = {'role':'assistant','content':result['reply'],'operation':result['operation']}
        self.state['discussion']['messages'].append(row)
        self.refresh_discussion()
        self.message.setPlainText('讨论已完成：'+self.chat_worker.context['name'])
        if self.save_state() and row['operation'] is not None:
            self.pending_chat_operation = row

    def chat_failed(self, error):
        self.state['discussion']['messages'].append({'role':'assistant','content':'本次讨论失败：'+error})
        self.refresh_discussion()
        self.save_state()
        self.message.setPlainText('作品讨论失败：'+error)

    def chat_finished(self):
        worker,self.chat_worker = self.chat_worker,None
        self.operation = ''
        row,self.pending_chat_operation = self.pending_chat_operation,None
        if row is not None:
            self.execute_chat_operation(row,worker.context)
        self.update_controls()
        self.busy_changed.emit()
        worker.deleteLater()

    def execute_chat_operation(self, row, context):
        try:
            photo = next(photo for photo in self.photos if photo['key']==context['photo_key'])
            operation = row['operation']
            action = operation['action']
            if action in ('adopt_batch','export','adopt_batch_and_export'):
                self.complete_delivery(row,context)
                return
            identifier = operation['version_id']
            version = selected_version(photo,{'version':identifier}) if identifier is not None else None
            if action=='tone_batch':
                if not self.start_tone_batch(context['tone_targets'],operation['parameters'],message=row):
                    raise ValueError(self.message.toPlainText())
            elif action=='tone':
                row['operation_status'] = '光调已请求'
                self.tone_requested.emit({'name':photo['name'],'base':version['files']['PNG'],
                    'directory':version['work'],'batch_directory':version['batch'],'return_to_delivery':True,
                    'delivery_photo_key':photo['key'],
                    'parent_version':retouch_parent(version),'delivery_message':row,
                    'delivery_user_text':self.state['discussion']['messages'][-2]['content']})
            elif action=='local':
                row['operation_status'] = '已打开局部修饰；涂选范围后生成'
                self.retouch_requested.emit({**saved_editor_context(photo,version),
                    'return_to_delivery':True,'local_task':operation['local_task'],
                    'local_brief':operation['local_brief']})
            elif action=='revise':
                revision = revision_context(photo,version)
                revision['art_direction']['edits'] = operation['edits']
                revision.update(revision_brief=operation['revision_brief'],delivery_message=row)
                row['operation_status'] = '单张改稿已请求：'+version['label']
                self.revision_requested.emit(revision)
            else:
                self.listing.setCurrentRow(self.rows[photo['key']])
                if action=='adopt':
                    self.adopt_saved_version(photo,version)
                    row['operation_status'] = '已采用交付版本：'+version['label']
                else:
                    self.compare_saved_version(photo,identifier)
                    row['operation_status'] = '已切换比较对象：'+(version['label'] if version else '原片')
        except (OSError,ValueError,KeyError,StopIteration) as error:
            row['operation_status'] = '操作失败：'+str(error)
            self.message.setPlainText(row['operation_status'])
        self.discussion_changed()

    def discussion_changed(self):
        self.refresh_discussion()
        row = self.state['discussion']['messages'][-1]
        if row.get('operation_status'):
            self.message.setPlainText(row['operation_status'])
        self.save_state()

    def show_photo(self, row):
        self.reference.blockSignals(True)
        self.reference.clear()
        self.comparison_note.hide()
        if row<0:
            self.reference.blockSignals(False)
            self.photo_title.setText('选择作品')
            self.photo_title.setToolTip('')
            self.delivery_version.clear()
            self.delivery_version.setToolTip('')
            self.preview.set_photo()
            self.preview_key = None
            self.preview_photo = None
            self.base_path = None
            self.beauty.setEnabled(False)
            self.retouch.setEnabled(False)
            self.tone.setEnabled(False)
            self.update_current_photo_controls()
            return
        photo = self.photos[row]
        self.photo_title.setText(photo['name'])
        self.photo_title.setToolTip(photo['name'])
        self.reference.addItem('原片',None)
        for version in reversed(photo['versions']):
            self.reference.addItem(version['label'],version['id'])
            self.reference.setItemData(self.reference.count()-1,version['description'],Qt.ToolTipRole)
        saved_reference = self.state['comparisons'].get(photo['key'])
        index = self.reference.findData(saved_reference)
        if index<0:
            self.reference.addItem('此前比较版本已不在记录中',saved_reference)
            index = self.reference.count()-1
        self.reference.setCurrentIndex(index)
        self.reference.blockSignals(False)
        self.reference.setToolTip(self.reference.currentText()+'\n选择比较对象只改变查看。')
        try:
            version = selected_version(photo,self.state['choices'][photo['key']])
        except ValueError as error:
            self.preview.set_photo()
            self.preview_key = None
            self.preview_photo = None
            self.base_path = None
            self.delivery_version.setText('所选交付版本无法读取')
            self.delivery_version.setToolTip(str(error))
            self.comparison_note.setText('当前交付版本无法读取。可以在「比较版本」中选用其他保存版。')
            self.comparison_note.show()
            self.beauty.setEnabled(False)
            self.retouch.setEnabled(False)
            self.tone.setEnabled(False)
            self.update_current_photo_controls()
            return
        width,height = version['dimensions']
        self.delivery_version.setText(version['label'])
        self.delivery_version.setToolTip(version['label']+f'\n{width} × {height}\n'+version['description'])
        self.base_path = version['base']
        self.update_comparison(photo,version)
        self.beauty.setEnabled(not self.is_busy() and not self.production_busy and Path(version['base']).is_file())
        self.retouch.setEnabled(not self.is_busy() and not self.production_busy and Path(version['files']['PNG']).is_file())
        self.tone.setEnabled(self.retouch.isEnabled())
        self.update_current_photo_controls()

    def update_comparison(self, photo, version):
        reference = self.reference.currentData()
        same_frame = False
        if reference is None:
            source = photo['source']
        else:
            try:
                previous = selected_version(photo,{'version':reference})
            except ValueError as error:
                self.preview.set_photo()
                self.preview_key = None
                self.preview_photo = None
                self.comparison_note.setText('比较版本无法读取：'+str(error))
                self.comparison_note.show()
                return
            source = previous['files']['JPEG']
            same_frame = ((version['kind'] in ('tone','relight') and version.get('parent_version',{}).get('id')==previous['id'])
                or (previous['kind'] in ('tone','relight') and previous.get('parent_version',{}).get('id')==version['id']))
        preview_key = (source,version['files']['JPEG'])
        if preview_key!=self.preview_key:
            self.preview.set_split_layout(same_frame)
            self.preview.set_photo(*preview_key,preserve_view=self.preview_photo==photo['key'])
            self.preview_key = preview_key
            self.preview_photo = photo['key']
        self.update_revision_controls()

    def change_reference(self, _index):
        photo = self.current_photo()
        if photo is None or self.reference.currentIndex()<0:
            return
        self.compare_saved_version(photo,self.reference.currentData())

    def compare_saved_version(self, photo, identifier):
        self.state['comparisons'][photo['key']] = identifier
        self.preview.choice = 'compare'
        self.preview.mode.setCurrentIndex(self.preview.mode.findData('compare'))
        self.show_photo(self.rows[photo['key']])
        self.save_state()

    def compare_saved_result(self, photo_key, png):
        photo = self.photos[self.rows[photo_key]]
        version = next(version for version in photo['versions'] if version['files']['PNG']==png)
        self.listing.setCurrentRow(self.rows[photo_key])
        self.compare_saved_version(photo,version['id'])

    def change_linked_comparison(self, linked):
        self.state['linked_comparison'] = linked
        self.save_state()

    def change_formats(self):
        self.state['formats'] = [name for name,check in self.formats.items() if check.isChecked()]
        self.save_state()
        self.update_controls()

    def update_controls(self):
        count = sum(choice['included'] for choice in self.state['choices'].values())
        retained = sum(1 for key,choice in self.state['choices'].items() if choice['included'] and key not in self.rows)
        visible = sum(not self.listing.item(row).isHidden() for row in range(self.listing.count()))
        self.summary.setText(f'已选交付 {count} 张 · 显示 {visible}/{len(self.photos)} 张'+(
            f' · 另有 {retained} 张选择的原片不在当前作品中' if retained else ''))
        self.summary.setToolTip('这些交付选择没有对应的当前作品；选择仍保留，导出时会在交付记录中'
            '列出失败并说明原因。' if retained else '')
        self.filter_button.setText('筛选（已启用）' if self.search.text().strip() or self.only_selected.isChecked() else '筛选')
        busy = self.is_busy()
        self.listing.setEnabled(not busy or self.operation in ('revision','chat','tone_batch'))
        self.reference.setEnabled(self.current_photo() is not None)
        self.select_all.setEnabled(not busy and bool(visible))
        self.select_none.setEnabled(not busy and bool(visible))
        self.refresh_button.setEnabled(not busy)
        self.location_button.setEnabled(not busy and not self.production_busy)
        self.adopt_tone_batch.setEnabled(not busy and not self.production_busy and bool(self.state.get('tone_batches')))
        self.tone_batch_record.setEnabled(not busy and bool(self.state.get('tone_batches')))
        for check in self.formats.values():
            check.setEnabled(not busy)
        self.export.setText(f'导出 {count} 张')
        self.export.setVisible(count>0)
        self.export.setEnabled(not busy and not self.production_busy and self.loaded and count>0 and bool(self.state['formats']))
        self.open_export.setEnabled(bool(self.state['exports']))
        self.history.setEnabled(not busy and bool(self.state['exports']))
        self.beauty.setEnabled(not busy and not self.production_busy and self.base_path is not None and Path(self.base_path).is_file())
        photo = self.current_photo()
        if photo is not None:
            try:
                version = selected_version(photo,self.state['choices'][photo['key']])
                self.retouch.setEnabled(not busy and not self.production_busy and Path(version['files']['PNG']).is_file())
                self.tone.setEnabled(self.retouch.isEnabled())
            except ValueError:
                self.retouch.setEnabled(False)
                self.tone.setEnabled(False)
        else:
            self.retouch.setEnabled(False)
            self.tone.setEnabled(False)
        self.clothing.setEnabled(self.retouch.isEnabled())
        self.update_current_photo_controls()
        self.chat_send.setEnabled(not busy and not self.production_busy and self.current_photo() is not None)
        self.chat_input.setEnabled(self.chat_worker is None)

    def update_revision_controls(self):
        photo = self.current_photo()
        self.revise.setEnabled(not self.is_busy() and not self.production_busy and photo is not None
            and any(version['kind']=='base' for version in photo['versions']))
        self.revise.setToolTip('选择这张的基础成片并填写单张要求，只制作一张新稿；交付选择保持原样')
        self.compare_revision.setVisible(photo is not None and photo['key'] in self.state['latest_revisions'])
        reference = self.state['comparisons'].get(photo['key']) if photo is not None else None
        chosen = self.state['choices'][photo['key']]['version'] if photo is not None else None
        adopt = reference is not None and reference!=chosen and self.preview.choice!='output'
        self.adopt_revision.setVisible(adopt)
        self.adopt_revision.setEnabled(not self.is_busy() and not self.production_busy)
        self.adopt_revision.setToolTip('采用：'+self.reference.currentText()+'\n保留这张照片的交付勾选')
        if hasattr(self,'export'):
            kind = 'quiet' if adopt else 'primary'
            if self.export.property('kind')!=kind:
                self.export.setProperty('kind',kind)
                self.export.style().unpolish(self.export)
                self.export.style().polish(self.export)

    def update_current_photo_controls(self):
        photo = self.current_photo()
        self.include_photo.setVisible(photo is not None)
        self.include_photo.setEnabled(self.listing.isEnabled())
        with QSignalBlocker(self.include_photo):
            self.include_photo.setChecked(bool(photo and self.state['choices'][photo['key']]['included']))
        self.update_revision_controls()

    def adopt_comparison(self):
        if self.is_busy() or self.production_busy:
            return
        photo = self.current_photo()
        if photo is None:
            return
        try:
            reference = selected_version(photo,{'version':self.state['comparisons'][photo['key']]})
            self.adopt_saved_version(photo,reference)
        except (OSError,ValueError,KeyError) as error:
            self.message.setPlainText('无法采用比较版本：'+str(error))

    def adopt_saved_version(self, photo, version):
        result = adopt_delivery_versions(self.photos,self.state['choices'],
            [{'photo_key':photo['key'],'version_id':version['id'],'name':photo['name']}])
        item = result['items'][0]
        if item['status']=='failed':
            raise ValueError(item['error'])
        self.state['comparisons'][photo['key']] = item['previous_version']
        self.preview.choice = 'output'
        with QSignalBlocker(self.listing):
            self.update_tile(photo)
        self.show_photo(self.listing.currentRow())
        self.load_thumbnails()
        if not self.save_state():
            raise OSError(self.message.toPlainText())
        self.message.setPlainText('已采用交付版本：'+version['label'])

    def adopt_tone_results(self, batch):
        if batch.get('error'):
            raise ValueError(batch['error'])
        result = adopt_delivery_versions(self.photos,self.state['choices'],batch['targets'])
        result['batch_id'] = batch['id']
        result['created_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
        for item in result['items']:
            if item['status']=='complete':
                self.state['comparisons'][item['photo_key']] = item['previous_version']
        self.state.setdefault('adoptions',[]).append(result)
        if not self.save_state():
            raise OSError(self.message.toPlainText())
        self.refresh()
        return result

    @staticmethod
    def adoption_summary(result):
        complete = sum(item['status']=='complete' for item in result['items'])
        failed = [item['name']+'：'+item['error'] for item in result['items'] if item['status']=='failed']
        text = f'已采用本轮光调成功版 {complete}/{len(result["items"])} 张；交付勾选保持原样。'
        return text+('\n未采用：'+'\n'.join(failed) if failed else '')

    def adopt_latest_tone_batch(self):
        try:
            result = self.adopt_tone_results(saved_tone_batch(self.state['tone_batches'][-1]))
            self.message.setPlainText(self.adoption_summary(result))
        except (OSError,ValueError,KeyError) as error:
            self.message.setPlainText('采用本轮光调失败：'+str(error))

    def complete_delivery(self, row, context):
        operation = row['operation']
        snapshot = deepcopy(context['export'])
        prefix = ''
        if operation['action']!='export':
            batch = next(item for item in context['tone_batches'] if item['id']==operation['batch_id'])
            result = self.adopt_tone_results(batch)
            snapshot['adoption'] = result
            prefix = self.adoption_summary(result)
            adopted = {item['photo_key']:item['version_id'] for item in result['items'] if item['status']=='complete'}
            for key,choice in snapshot['choices'].items():
                if key in adopted:
                    choice['version'] = adopted[key]
                else:
                    choice['included'] = False
        if operation['action']=='adopt_batch':
            row['operation_status'] = prefix
            self.discussion_changed()
        else:
            if not self.start_export(snapshot['parent'],snapshot=snapshot,formats=operation['formats'],
                    message=row,prefix=prefix):
                raise ValueError(self.message.toPlainText())

    def remember_revision_draft(self, version, draft):
        self.state['revision_drafts'][version] = draft
        self.draft_timer.start()

    def open_revision(self):
        if self.is_busy() or self.production_busy:
            return
        try:
            photo = self.current_photo()
            if photo is None:
                return
            current = selected_version(photo,self.state['choices'][photo['key']])
            dialog = RevisionDialog(photo,current,self.state['revision_drafts'],self)
            dialog.draft_changed.connect(self.remember_revision_draft)
            accepted = dialog.exec()==QDialog.Accepted
            if not self.save_state() or not accepted:
                return
            context = revision_context(photo,dialog.selected())
            context['art_direction']['edits'] = dialog.edits()
            context['revision_brief'] = dialog.requirements.toPlainText().strip()
            self.revision_requested.emit(context)
        except (OSError,ValueError,KeyError) as error:
            self.message.setPlainText('无法开始这张的改稿：'+str(error))

    def start_revision(self, context, root, config):
        self.operation = 'revision'
        self.revision_message = context.get('delivery_message')
        self.revision_comparison = None
        self.worker = RevisionWorker(context,root,config)
        self.worker.progress.connect(self.message.setPlainText)
        self.worker.directory_created.connect(self.batch_created.emit)
        self.worker.ready.connect(self.revision_ready)
        self.worker.failed.connect(self.revision_failed)
        self.worker.finished.connect(self.revision_finished)
        self.message.setPlainText('正在沿所选基础版制作这张的新稿。旧版、整组定稿与交付选择保留。')
        if self.revision_message is not None:
            self.revision_message['operation_status'] = '正在制作单张改稿'
            self.discussion_changed()
        self.update_controls()
        self.busy_changed.emit()
        self.worker.start()

    def revision_ready(self, result):
        if result['status']=='complete':
            self.state['latest_revisions'][source_key(result['source'])] = result['exports']['PNG']
            self.save_state()
            self.message.setPlainText(Path(result['source']).name+' 的新稿已另存。点击“比较新稿”查看，需要交付时再采用。')
            if self.revision_message is not None:
                self.revision_comparison = (source_key(result['source']),result['exports']['PNG'])
                self.revision_message['operation_status'] = '新稿已另存：'+result['exports']['PNG']+'；交付选择保持原样'
                self.discussion_changed()
        else:
            self.revision_failed('这张改稿未完成：'+str(result['error']))

    def revision_failed(self, error):
        self.message.setPlainText(error)
        if self.revision_message is not None:
            self.revision_message['operation_status'] = error
            self.discussion_changed()

    def revision_finished(self):
        worker,self.worker = self.worker,None
        self.operation = ''
        self.refresh()
        if self.revision_comparison is not None:
            try:
                self.compare_saved_result(*self.revision_comparison)
            except (OSError,ValueError,KeyError,StopIteration) as error:
                self.revision_failed('新稿已另存，显示比较失败：'+str(error))
        self.revision_message = None
        self.revision_comparison = None
        self.update_controls()
        self.busy_changed.emit()
        worker.deleteLater()

    def compare_new_revision(self):
        photo = self.current_photo()
        if photo is None:
            return
        reference = self.state['latest_revisions'].get(photo['key'])
        if reference is None:
            return
        try:
            selected_version(photo,{'version':reference})
            self.state['comparisons'][photo['key']] = reference
            self.preview.choice = 'compare'
            self.preview_key = None
            self.show_photo(self.listing.currentRow())
            self.save_state()
        except ValueError as error:
            self.message.setPlainText('新稿无法读取：'+str(error))

    def open_beauty(self):
        self.open_saved_editor(self.beauty_requested,'人脸美化',beauty=True)

    def open_retouch(self, task=None):
        self.open_saved_editor(self.retouch_requested,'局部修饰',local_task=task if isinstance(task,str) else None)

    def open_tone(self):
        self.open_saved_editor(self.tone_requested,'光色调整')

    def start_tone_batch(self, targets, parameters, *, message=None):
        if self.is_busy() or self.production_busy:
            return False
        if not targets:
            self.message.setPlainText('请先勾选要批量另存光色的交付照片。')
            return False
        parameters = ToneParameters(**parameters)
        if not self.save_state():
            return False
        directory = self.project.path.parent/'tone-batches'/('tone-'+
            datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
        self.operation = 'tone_batch'
        self.tone_batch_message = message
        self.tone_batch_result = None
        self.worker = ToneBatchWorker(targets,parameters,directory)
        self.worker.progress.connect(self.message.setPlainText)
        self.worker.ready.connect(self.tone_batch_ready)
        self.worker.failed.connect(self.tone_batch_failed)
        self.worker.finished.connect(self.tone_batch_finished)
        self.message.setPlainText(f'正在按固定参数另存 {len(targets)} 张光调版；交付版本保持原样。')
        if message is not None:
            message['operation_status'] = self.message.toPlainText()
            self.discussion_changed()
        self.update_controls()
        self.busy_changed.emit()
        self.worker.start()
        return True

    @staticmethod
    def tone_batch_summary(result):
        complete = sum(item['status']=='complete' for item in result['items'])
        failed = [item['name']+'：'+item['error'] for item in result['items'] if item['status']=='failed']
        text = f'批量光色已另存 {complete}/{len(result["items"])} 张'
        text += (f'，失败 {len(failed)} 张' if failed else '')+'；交付版本保持原样。\n记录：'+result['record']
        if failed:
            text += '\n'+'\n'.join(failed)
        return text

    def tone_batch_ready(self, result):
        self.tone_batch_result = result
        self.state.setdefault('tone_batches',[]).append(result['record'])
        self.save_state()
        text = self.tone_batch_summary(result)
        self.message.setPlainText(text)
        if self.tone_batch_message is not None:
            self.tone_batch_message['operation_status'] = text
            self.discussion_changed()

    def tone_batch_failed(self, error):
        self.message.setPlainText(error)
        if self.tone_batch_message is not None:
            self.tone_batch_message['operation_status'] = error
            self.discussion_changed()

    def show_tone_batch_record(self):
        try:
            path = Path(self.state['tone_batches'][-1])
            result = json.loads(path.read_text(encoding='utf-8'))
            self.message.setPlainText(self.tone_batch_summary(result))
        except (OSError,ValueError,KeyError) as error:
            self.message.setPlainText('读取批量光色记录失败：'+str(error))

    def tone_batch_finished(self):
        worker,self.worker = self.worker,None
        self.operation = ''
        self.refresh()
        if self.tone_batch_result is not None:
            try:
                for item in self.tone_batch_result['items']:
                    if item['status']=='complete':
                        photo = self.photos[self.rows[item['photo_key']]]
                        version = next(version for version in photo['versions'] if version['files']['PNG']==item['files']['final'])
                        self.state['comparisons'][photo['key']] = version['id']
                self.preview.choice = 'compare'
                self.preview_key = None
                self.show_photo(self.listing.currentRow())
                if self.save_state():
                    self.message.setPlainText(self.tone_batch_summary(self.tone_batch_result))
            except (OSError,ValueError,KeyError,StopIteration) as error:
                self.tone_batch_failed('光调文件已保留，显示比较失败：'+str(error))
        self.tone_batch_message = self.tone_batch_result = None
        self.update_controls()
        self.busy_changed.emit()
        worker.deleteLater()

    def open_saved_editor(self, requested, label, *, beauty=False, local_task=None):
        try:
            photo = self.current_photo()
            identifier = self.edit_target_id(photo)
            if identifier is None:
                raise ValueError('当前查看的是原片，请切换到要编辑的已保存版本')
            version = selected_version(photo,{'version':identifier})
            context = beauty_editor_context(photo,version) if beauty else saved_editor_context(photo,version)
            requested.emit({**context,'return_to_delivery':True,'local_task':local_task})
        except (OSError,ValueError,KeyError) as error:
            self.message.setPlainText('无法打开'+label+'：'+str(error))

    def choose_delivery_location(self):
        parent = QFileDialog.getExistingDirectory(self,'选择交付文件夹的保存位置',
            self.state.get('parent',str(self.project.path.parent)))
        if parent:
            self.state['parent'] = str(parent)
            self.delivery_location.setText(str(parent))
            self.save_state()

    def export_selected(self):
        if not self.state.get('parent'):
            self.choose_delivery_location()
        if self.state.get('parent'):
            self.start_export(self.state['parent'])

    def start_export(self, parent, *, snapshot=None, formats=None, message=None, prefix=''):
        if self.is_busy() or self.production_busy:
            self.message.setPlainText('当前任务尚未结束，无法开始导出。')
            return False
        if not parent:
            self.message.setPlainText('请先设置交付位置。')
            return False
        self.state['parent'] = str(parent)
        self.delivery_location.setText(str(parent))
        if not self.save_state():
            return False
        snapshot = snapshot if snapshot is not None else {'photos':self.photos,'choices':self.state['choices']}
        self.operation = 'export'
        self.export_message_row,self.export_prefix = message,prefix
        self.worker = ExportWorker(snapshot['photos'],snapshot['choices'],str(parent),
            self.project.data['name'],self.state['formats'] if formats is None else formats,
            adoption=snapshot.get('adoption'))
        self.worker.progress.connect(self.message.setPlainText)
        self.worker.ready.connect(self.export_ready)
        self.worker.failed.connect(self.export_failed)
        self.worker.finished.connect(self.export_finished)
        if message is not None:
            message['operation_status'] = (prefix+'\n' if prefix else '')+'正在复制发送时选定的保存版本到交付文件夹。'
            self.discussion_changed()
        self.update_controls()
        self.busy_changed.emit()
        self.worker.start()
        return True

    def export_ready(self, result):
        self.state['exports'].append(result)
        self.refresh_history()
        message = self.export_message(result)
        if self.export_message_row is not None:
            self.export_message_row['operation_status'] = message
            self.refresh_discussion()
        if self.save_state():
            self.message.setPlainText(message)
        else:
            self.message.setPlainText(message+'\n'+self.message.toPlainText())

    def export_failed(self, error):
        text = (self.export_prefix+'\n' if self.export_prefix else '')+error
        self.message.setPlainText(text)
        if self.export_message_row is not None:
            self.export_message_row['operation_status'] = text
            self.refresh_discussion()
            self.save_state()

    @staticmethod
    def export_message(result):
        complete = sum(item['status']=='complete' for item in result['items'])
        failed = [Path(item['source']).name+'：'+item['error'] for item in result['items'] if item['status']=='failed']
        message = f'已导出 {complete} 张'+(f'，失败 {len(failed)} 张' if failed else '')+'。文件夹：'+result['directory']
        if failed:
            message += '\n'+'\n'.join(failed)
        if result.get('manifest'):
            message += '\n逐张清单：'+result['manifest']
        if result.get('adoption'):
            message = DeliveryView.adoption_summary(result['adoption'])+'\n'+message
        return message

    def refresh_history(self):
        self.history.blockSignals(True)
        self.history.clear()
        for index in reversed(range(len(self.state['exports']))):
            result = self.state['exports'][index]
            complete = sum(item['status']=='complete' for item in result['items'])
            failed = len(result['items'])-complete
            adoption_failed = result.get('adoption',{}).get('status','complete')!='complete'
            self.history.addItem(f'交付 {index+1} · {complete} 张'+(f' · 失败 {failed} 张' if failed else '')+
                (' · 采用未全部完成' if adoption_failed else ''),index)
            self.history.setItemData(self.history.count()-1,result['created_at']+'\n'+result['directory'],Qt.ToolTipRole)
        self.history.setCurrentIndex(0 if self.state['exports'] else -1)
        self.history.blockSignals(False)
        if self.state['exports']:
            self.show_export_record()

    def show_export_record(self):
        index = self.history.currentData()
        if index is not None:
            self.message.setPlainText(self.export_message(self.state['exports'][index]))

    def export_finished(self):
        worker,self.worker = self.worker,None
        self.operation = ''
        self.export_message_row,self.export_prefix = None,''
        self.update_controls()
        self.busy_changed.emit()
        worker.deleteLater()

    def open_selected_export(self):
        index = self.history.currentData()
        if index is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.state['exports'][index]['directory']))
