"""A project contact sheet for explicit version choices and delivery exports."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSize, QThread, QThreadPool, Qt, Signal, Slot
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
    QSplitter, QVBoxLayout, QWidget)

from delivery_store import export_delivery, photo_choices, project_photos, selected_version
from image_io import open_display
from studio_widgets import PhotoPreview, StatusText, ui_label


class ThumbnailSignals(QObject):
    ready = Signal(object,object,str)
    finished = Signal()


class ThumbnailLoader(QRunnable):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys
        self.signals = ThumbnailSignals()

    def run(self):
        for key in self.keys:
            try:
                image = open_display(key[0],max_edge=360)
                error = ''
            except Exception as exception:
                image,error = None,str(exception)
            self.signals.ready.emit(key,image,error)
        self.signals.finished.emit()


class ExportWorker(QThread):
    progress = Signal(str)
    ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, photos, choices, parent, name, formats):
        super().__init__()
        self.arguments = (deepcopy(photos),deepcopy(choices),parent,name,list(formats))

    def run(self):
        try:
            self.ready.emit(export_delivery(*self.arguments,progress=self.progress.emit))
        except Exception as error:
            self.failed.emit('导出失败：'+str(error))


class DeliveryView(QWidget):
    beauty_requested = Signal(dict)
    busy_changed = Signal()
    state_changed = Signal()

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.state = deepcopy(project.data.get('delivery',{}))
        self.state.setdefault('choices',{})
        self.state.setdefault('exports',[])
        self.state.setdefault('formats',['JPEG'])
        self.photos = []
        self.rows = {}
        self.worker = None
        self.production_busy = False
        self.thumbnail_loader = None
        self.thumbnail_keys = {}
        self.thumbnail_cache = {}
        self.thumbnail_errors = {}
        self.loaded = False
        self.preview_key = None
        self.base_path = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(8)
        tools = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText('搜索原片文件名')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.apply_filters)
        tools.addWidget(self.search,1)
        self.only_selected = QCheckBox('只看已选交付')
        self.only_selected.toggled.connect(self.apply_filters)
        tools.addWidget(self.only_selected)
        self.select_all = QPushButton('勾选当前显示')
        self.select_all.clicked.connect(lambda:self.mark_visible(True))
        tools.addWidget(self.select_all)
        self.select_none = QPushButton('取消当前显示')
        self.select_none.clicked.connect(lambda:self.mark_visible(False))
        tools.addWidget(self.select_none)
        self.refresh_button = QPushButton('刷新作品')
        self.refresh_button.clicked.connect(self.refresh)
        tools.addWidget(self.refresh_button)
        layout.addLayout(tools)
        self.note = ui_label('勾选决定交付范围；点图检查细节，选择每张作品要交付的版本。')
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(12)
        left = QWidget()
        left.setMinimumWidth(300)
        column = QVBoxLayout(left)
        column.setContentsMargins(0,0,0,0)
        self.listing = QListWidget()
        self.listing.setObjectName('deliverySheet')
        self.listing.setViewMode(QListView.IconMode)
        self.listing.setResizeMode(QListView.Adjust)
        self.listing.setMovement(QListView.Static)
        self.listing.setSelectionMode(QAbstractItemView.SingleSelection)
        self.listing.setIconSize(QSize(160,150))
        self.listing.setGridSize(QSize(190,216))
        self.listing.setSpacing(4)
        self.listing.setWordWrap(True)
        self.listing.setTextElideMode(Qt.ElideRight)
        self.listing.currentRowChanged.connect(self.browse_photo)
        self.listing.itemChanged.connect(self.change_inclusion)
        column.addWidget(self.listing,1)
        self.empty = ui_label('本项目还没有完整成片。完成制作后，可以在这里挑选并导出。')
        self.empty.setWordWrap(True)
        column.addWidget(self.empty)
        splitter.addWidget(left)
        right = QWidget()
        right.setMinimumWidth(380)
        column = QVBoxLayout(right)
        column.setContentsMargins(0,0,0,0)
        self.photo_title = ui_label('选择作品检查细节','heading')
        self.photo_title.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        column.addWidget(self.photo_title)
        self.version = QComboBox()
        self.version.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        self.version.setPlaceholderText('选择已保存的成片版本')
        self.version.currentIndexChanged.connect(self.change_version)
        column.addWidget(self.version)
        self.preview = PhotoPreview()
        column.addWidget(self.preview,1)
        details = QHBoxLayout()
        self.details = ui_label('')
        self.details.setWordWrap(True)
        details.addWidget(self.details,1)
        self.beauty = QPushButton('人脸美化')
        self.beauty.clicked.connect(self.open_beauty)
        self.beauty.setEnabled(False)
        details.addWidget(self.beauty)
        column.addLayout(details)
        splitter.addWidget(right)
        splitter.setSizes([530,650])
        layout.addWidget(splitter,1)
        footer = QHBoxLayout()
        self.summary = ui_label('已选交付 0 张')
        footer.addWidget(self.summary,1)
        self.formats = {}
        for name,text in (('JPEG','JPEG（8 位）'),('PNG','PNG（16 位）')):
            check = QCheckBox(text)
            check.setChecked(name in self.state['formats'])
            check.toggled.connect(self.change_formats)
            self.formats[name] = check
            footer.addWidget(check)
        self.history = QComboBox()
        self.history.setMinimumWidth(170)
        self.history.setMaximumWidth(240)
        self.history.setPlaceholderText('尚未导出')
        self.history.setToolTip('查看本项目已保存的交付记录')
        self.history.currentIndexChanged.connect(self.show_export_record)
        footer.addWidget(self.history)
        self.open_export = QPushButton('打开交付文件夹')
        self.open_export.clicked.connect(self.open_selected_export)
        footer.addWidget(self.open_export)
        self.export = QPushButton('导出所选作品')
        self.export.setProperty('kind','primary')
        self.export.clicked.connect(self.choose_export)
        footer.addWidget(self.export)
        layout.addLayout(footer)
        self.message = StatusText('交付复制所选版本的已有成片，原片和制作文件保留原位置。')
        layout.addWidget(self.message)
        self.refresh_history()
        self.refresh()

    def is_busy(self):
        return self.worker is not None

    def set_production_busy(self, busy):
        self.production_busy = busy
        self.update_controls()

    def save_state(self):
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
        self.listing.blockSignals(True)
        self.listing.clear()
        for photo in photos:
            item = QListWidgetItem()
            item.setFlags(item.flags()|Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if self.state['choices'][photo['key']]['included'] else Qt.Unchecked)
            item.setSizeHint(QSize(180,206))
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
            item.setText(photo['name']+'\n'+version['short_label'])
            path = version['files']['JPEG']
            key = (path,Path(path).stat().st_mtime_ns)
        except (OSError,ValueError) as error:
            item.setText(photo['name']+'\n所选版本无法读取')
            item.setToolTip(str(error))
            item.setIcon(QIcon())
            self.thumbnail_keys.pop(photo['key'],None)
            return
        self.thumbnail_keys[photo['key']] = key
        item.setIcon(self.thumbnail_cache.get(key,QIcon()))
        item.setToolTip(photo['source']+'\n'+version['label']+(
            '\n读取缩略图失败：'+self.thumbnail_errors[key] if key in self.thumbnail_errors else ''))

    def load_thumbnails(self):
        if self.thumbnail_loader is not None:
            return
        keys = list(dict.fromkeys(key for key in self.thumbnail_keys.values()
            if key not in self.thumbnail_cache and key not in self.thumbnail_errors))
        if keys:
            self.thumbnail_loader = ThumbnailLoader(keys)
            self.thumbnail_loader.signals.ready.connect(self.thumbnail_ready)
            self.thumbnail_loader.signals.finished.connect(self.thumbnails_finished)
            QThreadPool.globalInstance().start(self.thumbnail_loader)

    @Slot(object,object,str)
    def thumbnail_ready(self, key, image, error):
        if image is not None:
            self.thumbnail_cache[key] = QIcon(QPixmap.fromImage(image))
        else:
            self.thumbnail_errors[key] = error
        self.listing.blockSignals(True)
        for photo in self.photos:
            if self.thumbnail_keys.get(photo['key'])==key:
                self.update_tile(photo)
        self.listing.blockSignals(False)

    @Slot()
    def thumbnails_finished(self):
        self.thumbnail_loader = None
        self.load_thumbnails()

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
            '本项目还没有完整成片。完成制作后，可以在这里挑选并导出。')
        self.empty.setVisible(not visible)
        self.show_photo(self.listing.currentRow())
        self.update_controls()

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

    def browse_photo(self, row):
        self.show_photo(row)
        if row>=0:
            self.state['current'] = self.photos[row]['key']
            self.save_state()

    def show_photo(self, row):
        self.version.blockSignals(True)
        self.version.clear()
        if row<0:
            self.version.blockSignals(False)
            self.photo_title.setText('选择作品检查细节')
            self.preview.set_photo()
            self.preview_key = None
            self.base_path = None
            self.details.setText('')
            self.beauty.setEnabled(False)
            return
        photo = self.photos[row]
        self.photo_title.setText(photo['name'])
        for version in reversed(photo['versions']):
            self.version.addItem(version['label'],version['id'])
        self.version.setCurrentIndex(self.version.findData(self.state['choices'][photo['key']]['version']))
        self.version.blockSignals(False)
        try:
            version = selected_version(photo,self.state['choices'][photo['key']])
        except ValueError as error:
            self.preview.set_photo()
            self.preview_key = None
            self.base_path = None
            self.details.setText(str(error))
            self.beauty.setEnabled(False)
            return
        preview_key = (photo['source'],version['files']['JPEG'])
        if preview_key!=self.preview_key:
            self.preview.set_photo(*preview_key)
            self.preview_key = preview_key
        width,height = version['dimensions']
        self.base_path = version['base']
        self.details.setText(f'{width} × {height} · '+version['label'])
        self.beauty.setEnabled(not self.is_busy() and not self.production_busy and Path(version['base']).is_file())

    def change_version(self, _index):
        photo = self.current_photo()
        if photo is None or self.version.currentIndex()<0:
            return
        self.state['choices'][photo['key']]['version'] = self.version.currentData()
        self.listing.blockSignals(True)
        self.update_tile(photo)
        self.listing.blockSignals(False)
        self.show_photo(self.listing.currentRow())
        self.load_thumbnails()
        self.save_state()

    def change_formats(self):
        self.state['formats'] = [name for name,check in self.formats.items() if check.isChecked()]
        self.save_state()
        self.update_controls()

    def update_controls(self):
        count = sum(choice['included'] for choice in self.state['choices'].values())
        visible = sum(not self.listing.item(row).isHidden() for row in range(self.listing.count()))
        self.summary.setText(f'已选交付 {count} 张 · 显示 {visible}/{len(self.photos)} 张')
        busy = self.is_busy()
        self.listing.setEnabled(not busy)
        self.version.setEnabled(not busy and self.current_photo() is not None)
        self.select_all.setEnabled(not busy and bool(visible))
        self.select_none.setEnabled(not busy and bool(visible))
        self.refresh_button.setEnabled(not busy)
        for check in self.formats.values():
            check.setEnabled(not busy)
        self.export.setText(f'导出所选 {count} 张' if count else '先勾选交付作品')
        self.export.setEnabled(not busy and not self.production_busy and self.loaded and count>0 and bool(self.state['formats']))
        self.open_export.setEnabled(bool(self.state['exports']))
        self.history.setEnabled(not busy and bool(self.state['exports']))
        self.beauty.setEnabled(not busy and not self.production_busy and self.base_path is not None and Path(self.base_path).is_file())

    def open_beauty(self):
        try:
            photo = self.current_photo()
            version = selected_version(photo,self.state['choices'][photo['key']])
            run = json.loads((Path(version['work'])/'run.json').read_text(encoding='utf-8'))
            self.beauty_requested.emit({'name':photo['name'],'base':version['base'],
                'region':run['head']['region'],'directory':version['work'],'batch_directory':version['batch'],
                'return_to_delivery':True})
        except (OSError,ValueError,KeyError) as error:
            self.message.setPlainText('无法打开人脸美化：'+str(error))

    def choose_export(self):
        parent = QFileDialog.getExistingDirectory(self,'选择交付文件夹的保存位置',
            self.state.get('parent',str(self.project.path.parent)))
        if parent:
            self.start_export(parent)

    def start_export(self, parent):
        if self.is_busy() or self.production_busy:
            return
        self.state['parent'] = str(parent)
        if not self.save_state():
            return
        self.worker = ExportWorker(self.photos,self.state['choices'],str(parent),
            self.project.data['name'],self.state['formats'])
        self.worker.progress.connect(self.message.setPlainText)
        self.worker.ready.connect(self.export_ready)
        self.worker.failed.connect(self.message.setPlainText)
        self.worker.finished.connect(self.export_finished)
        self.update_controls()
        self.busy_changed.emit()
        self.worker.start()

    def export_ready(self, result):
        self.state['exports'].append(result)
        self.refresh_history()
        message = self.export_message(result)
        if self.save_state():
            self.message.setPlainText(message)
        else:
            self.message.setPlainText(message+'\n'+self.message.toPlainText())

    @staticmethod
    def export_message(result):
        complete = sum(item['status']=='complete' for item in result['items'])
        failed = [Path(item['source']).name+'：'+item['error'] for item in result['items'] if item['status']=='failed']
        message = f'已导出 {complete} 张'+(f'，失败 {len(failed)} 张' if failed else '')+'。文件夹：'+result['directory']
        if failed:
            message += '\n'+'\n'.join(failed)
        return message

    def refresh_history(self):
        self.history.blockSignals(True)
        self.history.clear()
        for index in reversed(range(len(self.state['exports']))):
            result = self.state['exports'][index]
            complete = sum(item['status']=='complete' for item in result['items'])
            failed = len(result['items'])-complete
            self.history.addItem(f'交付 {index+1} · {complete} 张'+(f' · 失败 {failed} 张' if failed else ''),index)
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
        self.update_controls()
        self.busy_changed.emit()
        worker.deleteLater()

    def open_selected_export(self):
        index = self.history.currentData()
        if index is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.state['exports'][index]['directory']))
