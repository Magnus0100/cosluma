"""Choose a complete photo project before entering its photo/creative/result pages."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
    QSplitter, QStackedWidget, QFrame, QVBoxLayout, QWidget)

from project_store import Project
from studio_widgets import PhotoPreview, ui_label


def _project_record(project, *, active=False):
    state = project.data['workspace']
    title = project.data['name']
    edited = datetime.fromisoformat(project.data['updated_at']).astimezone().strftime('%Y-%m-%d %H:%M')
    label = title+('  · 当前项目' if active else '')
    label += '\n'+(state.get('character') or '尚未填写角色')+f'  · {len(state["sources"])} 张照片  · {edited}'
    complete = sum(item['status']=='complete' for item in state['results'])
    delivery = project.data.get('delivery',{})
    included = sum(choice['included'] for choice in delivery.get('choices',{}).values())
    delivery_info = f'\n已选交付 {included} 张 · 已导出 {len(delivery.get("exports",[]))} 次' if delivery else ''
    info = (state.get('character') or '角色待填写')+f'\n{len(state["sources"])} 张原片　{complete} 张成片记录　{len(project.data["batches"])} 次制作'+delivery_info
    stage = '已有制作记录' if complete or project.data['batches'] else '已导入照片' if state['sources'] else '等待导入照片'
    source = state.get('representative_source') or next(iter(state['sources']),None)
    result = next((item for item in state['results'] if item['source']==source and item['status']=='complete'),None)
    output = (result.get('exports',{}).get('JPEG') or result.get('files',{}).get('jpeg')) if result else None
    return {'path':str(project.path),'label':label,'title':title,'info':info,'stage':stage,'preview':(source,output)}


class NewProject(QDialog):
    def __init__(self, parent_directory, parent=None):
        super().__init__(parent)
        self.setWindowTitle('新建项目')
        self.resize(650, 330)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.addWidget(ui_label('新建项目', 'title'))
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText('例如：守岸人 · 海边外景')
        form.addRow('项目名称', self.name)
        self.parent_directory = QLineEdit(str(parent_directory))
        row = QHBoxLayout()
        row.addWidget(self.parent_directory, 1)
        choose = QPushButton('选择文件夹')
        choose.clicked.connect(self.choose_directory)
        row.addWidget(choose)
        form.addRow('保存位置', row)
        self.location = ui_label('')
        self.location.setWordWrap(True)
        self.location.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow('项目文件夹', self.location)
        layout.addLayout(form)
        self.message = ui_label('')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Cancel).setText('取消')
        self.create = buttons.addButton('创建项目', QDialogButtonBox.AcceptRole)
        self.create.setProperty('kind', 'primary')
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept_project)
        layout.addWidget(buttons)
        self.name.textChanged.connect(self.update_location)
        self.parent_directory.textChanged.connect(self.update_location)
        self.update_location()
        self.name.setFocus()

    def update_location(self):
        name = self.name.text().strip()
        self.location.setText(str(Path(self.parent_directory.text()).expanduser()/name) if name else '填写名称后显示')
        self.create.setEnabled(bool(name and self.parent_directory.text().strip()))

    def choose_directory(self):
        directory = QFileDialog.getExistingDirectory(self, '选择项目保存位置', self.parent_directory.text())
        if directory:
            self.parent_directory.setText(directory)

    def accept_project(self):
        name = self.name.text().strip()
        if name in ('.', '..') or any(char in name for char in '<>:"/\\|?*') or name.endswith(('.', ' ')):
            self.message.setText('请使用不含路径分隔符或 Windows 文件名保留字符的项目名称')
            self.name.setFocus()
            return
        directory = Path(self.parent_directory.text()).expanduser()/name
        try:
            if (directory/'project.json').exists():
                self.message.setText('此位置已有同名项目。请改用其他名称，或取消后打开已有项目。')
                self.name.setFocus()
                return
            if directory.exists() and any(directory.iterdir()):
                self.message.setText('此位置已有其他文件，请选择新的或空的项目文件夹。')
                self.name.setFocus()
                return
        except OSError as error:
            self.message.setText('无法读取项目保存位置：'+str(error))
            return
        self.directory = directory
        self.accept()


class ProjectBrowser(QWidget):
    create_requested = Signal()
    browse_requested = Signal()
    open_requested = Signal(str)
    rename_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.records = []
        self.active_project = None
        self.preview_key = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.setSpacing(20)
        header = QHBoxLayout()
        intro = QVBoxLayout()
        intro.addWidget(ui_label('最近项目', 'title'))
        header.addLayout(intro, 1)
        self.browse = QPushButton('打开项目')
        self.browse.clicked.connect(self.browse_requested.emit)
        header.addWidget(self.browse)
        self.create = QPushButton('新建项目')
        self.create.setProperty('kind', 'primary')
        self.create.clicked.connect(self.create_requested.emit)
        header.addWidget(self.create)
        layout.addLayout(header)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.sidebar = sidebar = QWidget()
        column = QVBoxLayout(sidebar)
        column.setContentsMargins(0, 0, 12, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText('搜索项目名称或角色')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.filter_projects)
        column.addWidget(self.search)
        self.listing = QListWidget()
        self.listing.setObjectName('projectList')
        self.listing.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.listing.setTextElideMode(Qt.ElideRight)
        self.listing.setSpacing(6)
        self.listing.currentRowChanged.connect(self.show_project)
        self.listing.itemDoubleClicked.connect(lambda _item:self.open_selected())
        column.addWidget(self.listing, 1)
        self.empty = ui_label('暂无项目')
        self.empty.setWordWrap(True)
        column.addWidget(self.empty)
        splitter.addWidget(sidebar)
        self.detail_pages = QStackedWidget()
        detail = QFrame()
        detail.setObjectName('projectDetail')
        column = QVBoxLayout(detail)
        column.setContentsMargins(22, 20, 22, 20)
        column.setSpacing(12)
        self.stage = ui_label('','projectStage')
        column.addWidget(self.stage,0,Qt.AlignLeft)
        self.title = ui_label('选择一个项目', 'title')
        self.title.setWordWrap(True)
        column.addWidget(self.title)
        self.preview = PhotoPreview()
        self.cover_pages = QStackedWidget()
        self.cover_pages.addWidget(self.preview)
        placeholder = ui_label('暂无照片','dropArea')
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setWordWrap(True)
        self.cover_pages.addWidget(placeholder)
        column.addWidget(self.cover_pages, 1)
        self.info = ui_label('')
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        column.addWidget(self.info)
        controls = QHBoxLayout()
        self.rename = QPushButton('重命名')
        self.rename.clicked.connect(self.rename_selected)
        controls.addWidget(self.rename)
        self.folder = QPushButton('打开项目文件夹')
        self.folder.clicked.connect(self.open_folder)
        controls.addWidget(self.folder)
        controls.addStretch(1)
        self.continue_button = QPushButton('继续创作')
        self.continue_button.setProperty('kind', 'primary')
        self.continue_button.clicked.connect(self.open_selected)
        controls.addWidget(self.continue_button)
        column.addLayout(controls)
        self.detail_pages.addWidget(detail)
        welcome = QFrame()
        welcome.setObjectName('projectWelcome')
        welcome_column = QVBoxLayout(welcome)
        welcome_column.setContentsMargins(36,32,36,32)
        welcome_column.setSpacing(18)
        welcome_column.addStretch(1)
        self.welcome_title = ui_label('暂无项目','heading')
        welcome_column.addWidget(self.welcome_title,0,Qt.AlignCenter)
        self.welcome_note = ui_label('')
        self.welcome_note.setWordWrap(True)
        welcome_column.addWidget(self.welcome_note)
        welcome_column.addStretch(1)
        self.detail_pages.addWidget(welcome)
        splitter.addWidget(self.detail_pages)
        splitter.setSizes([320, 760])
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self.message = ui_label('')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.switching_enabled = True

    def refresh(self, paths, active=None):
        selected = self.selected()
        selected_path = selected['path'] if selected else None
        self.records = []
        self.active_project = active
        self.listing.blockSignals(True)
        self.listing.clear()
        for path in paths:
            try:
                project = active if active is not None and Path(path).resolve() == active.path else Project.open(path)
                record = _project_record(project,active=active is not None and project.path==active.path)
            except (OSError, ValueError, KeyError, TypeError) as error:
                label = Path(path).parent.name+'\n无法打开：'+str(error)
                record = {'path':path,'error':str(error),'label':label}
            item = QListWidgetItem(record['label'])
            item.setToolTip(record['path'])
            self.listing.addItem(item)
            self.records.append(record)
        self.listing.blockSignals(False)
        previous = next((row for row, record in enumerate(self.records) if record['path'] == selected_path), -1)
        self.listing.setCurrentRow(previous)
        self.filter_projects()

    def filter_projects(self):
        query = self.search.text().strip().casefold()
        visible = []
        for row in range(self.listing.count()):
            item = self.listing.item(row)
            match = query in item.text().casefold()
            item.setHidden(not match)
            if match:
                visible.append(row)
        self.sidebar.setVisible(bool(self.records))
        self.empty.setVisible(not visible and bool(self.records))
        self.empty.setText('无匹配项目' if self.records else '暂无项目')
        current = self.listing.currentRow()
        if current not in visible:
            self.listing.setCurrentRow(visible[0] if visible else -1)
        self.show_project()

    def selected(self):
        row = self.listing.currentRow()
        return self.records[row] if 0 <= row < len(self.records) else None

    def show_project(self, _row=None):
        record = self.selected()
        valid = record is not None and 'error' not in record
        if valid and self.active_project is not None and record['path']==str(self.active_project.path):
            record = _project_record(self.active_project,active=True)
        enabled = valid and self.switching_enabled
        self.continue_button.setEnabled(enabled)
        self.rename.setEnabled(enabled)
        self.folder.setEnabled(record is not None)
        if not valid:
            self.detail_pages.setCurrentIndex(1)
            self.welcome_title.setText('项目暂时无法打开' if record else '没有找到匹配项目' if self.records else '暂无项目')
            self.welcome_note.setText('读取失败：'+record['error'] if record else
                '')
            self.title.setText('选择一个项目')
            if self.preview_key is not None:
                self.preview.set_photo()
                self.preview_key = None
            self.info.setText(record['error'] if record else '')
            return
        self.detail_pages.setCurrentIndex(0)
        self.stage.setText(record['stage'])
        self.title.setText(record['title'])
        self.info.setText(record['info'])
        self.folder.setToolTip(str(Path(record['path']).parent))
        key = record['preview']
        self.cover_pages.setCurrentIndex(0 if any(key) else 1)
        if key != self.preview_key:
            self.preview.set_photo(*key)
            self.preview_key = key

    def set_busy(self, busy):
        self.switching_enabled = not busy
        self.create.setEnabled(not busy)
        self.browse.setEnabled(not busy)
        self.message.setText('当前项目正在处理中，可查看项目列表；处理结束后可以切换。' if busy else
            '')
        self.show_project()

    def open_selected(self):
        record = self.selected()
        if self.switching_enabled and record and 'error' not in record:
            self.open_requested.emit(record['path'])

    def rename_selected(self):
        record = self.selected()
        if self.switching_enabled and record and 'error' not in record:
            self.rename_requested.emit(record['path'])

    def open_folder(self):
        record = self.selected()
        if record:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(record['path']).parent)))
