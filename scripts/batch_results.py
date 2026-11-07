"""Browse a saved batch in the workbench without submitting photographs."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QHBoxLayout, QListWidget, QPlainTextEdit,
    QPushButton, QSplitter, QVBoxLayout, QWidget)

from studio_widgets import PhotoPreview, ui_label


class BatchResults(QWidget):
    """Keep the selected photograph visible as this batch receives its results."""
    back_requested = Signal()

    def __init__(self, directory, parent=None):
        super().__init__(parent)
        self.directory = Path(directory).expanduser().resolve()
        self.items = {}
        self.sources = []
        self.rows = {}
        self.preview_key = None
        self.output_path = None
        self.work_directory = str(self.directory)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(8)
        header = QHBoxLayout()
        self.heading = ui_label('成片','heading')
        header.addWidget(self.heading)
        self.summary = ui_label('')
        self.summary.setWordWrap(True)
        header.addWidget(self.summary,1)
        layout.addLayout(header)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(14)
        self.listing = QListWidget()
        self.listing.setMinimumWidth(190)
        self.listing.setMaximumWidth(320)
        self.listing.setTextElideMode(Qt.ElideRight)
        splitter.addWidget(self.listing)
        self.preview = PhotoPreview()
        splitter.addWidget(self.preview)
        splitter.setSizes([250,850])
        splitter.setStretchFactor(1,1)
        layout.addWidget(splitter,1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(42)
        self.details.setMaximumHeight(68)
        layout.addWidget(self.details)
        controls = QHBoxLayout()
        self.output = QPushButton('打开成片')
        self.output.clicked.connect(lambda:self.open_path(self.output_path))
        controls.addWidget(self.output)
        self.work = QPushButton('打开此照片文件夹')
        self.work.clicked.connect(lambda:self.open_path(self.work_directory))
        controls.addWidget(self.work)
        folder = QPushButton('打开本批次文件夹')
        folder.clicked.connect(lambda:self.open_path(str(self.directory)))
        controls.addWidget(folder)
        controls.addStretch(1)
        back = QPushButton('返回照片')
        back.setProperty('kind','quiet')
        back.clicked.connect(lambda:self.back_requested.emit())
        controls.addWidget(back)
        layout.addLayout(controls)
        self.listing.currentRowChanged.connect(self.show_photo)
        self.refresh()

    @staticmethod
    def open_path(path):
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def saved_path(self, result, value):
        """Rebase batch-owned files when their complete batch folder has moved."""
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            return str(self.directory/path)
        previous = result.get('batch_directory')
        if previous:
            try:
                return str(self.directory/path.relative_to(Path(previous)))
            except ValueError:
                pass
        return str(path)

    def row_text(self, source):
        result = self.items.get(source,{})
        stage = {'complete':'完成','failed':'失败'}.get(result.get('status'),'未完成')
        return f'{Path(source).name} · {stage}'

    def update_summary(self):
        complete = sum(result['status']=='complete' for result in self.items.values())
        failed = sum(result['status']=='failed' for result in self.items.values())
        pending = len(self.sources)-complete-failed
        status = {'running':'制作中','complete':'已完成','unfinished':'未全部完成','failed':'制作失败'}[self.batch_status]
        self.heading.setText(self.character+' · '+status if self.character else '本批次 · '+status)
        note = '单张结果会陆续显示，可随时切换页面。' if self.batch_status=='running' else '选择照片查看成片或失败原因。'
        self.summary.setText(f'完成 {complete} 张，失败 {failed} 张，未完成 {pending} 张。'+note)

    def update_result(self, result):
        """Apply one result supplied by the workbench for this same batch."""
        source = result['source']
        row = self.rows[source]
        self.items[source] = result
        self.listing.item(row).setText(self.row_text(source))
        self.update_summary()
        if self.listing.currentRow()==row:
            self.show_photo(row)

    def refresh(self):
        """Read once on request, retaining the current photograph and its zoom."""
        state = json.loads((self.directory/'batch.json').read_text(encoding='utf-8'))
        row = self.listing.currentRow()
        selected = self.sources[row] if row>=0 else None
        self.items = state['items']
        self.sources = state['sources'] if 'sources' in state else list(self.items)
        self.rows = {source:row for row,source in enumerate(self.sources)}
        self.error = state.get('error','')
        self.character = state.get('character','')
        self.batch_status = state['status']
        self.update_summary()
        self.listing.blockSignals(True)
        self.listing.clear()
        for source in self.sources:
            self.listing.addItem(self.row_text(source))
            self.listing.item(self.listing.count()-1).setToolTip(source)
        row = self.rows[selected] if selected in self.rows else 0
        self.listing.setCurrentRow(row)
        self.listing.blockSignals(False)
        self.show_photo(self.listing.currentRow())

    def show_photo(self, row):
        if row<0:
            return
        source = self.sources[row]
        result = self.items.get(source,{})
        self.output_path = self.saved_path(result,
            result.get('exports',{}).get('JPEG') or result.get('files',{}).get('jpeg'))
        if result.get('status')!='complete':
            self.output_path = None
        self.work_directory = self.saved_path(result,result.get('directory')) or str(self.directory)
        preview_key = (source,self.output_path)
        if preview_key!=self.preview_key:
            self.preview.set_photo(source,self.output_path)
            self.preview_key = preview_key
        self.output.setEnabled(bool(self.output_path) and Path(self.output_path).is_file())
        self.work.setEnabled(Path(self.work_directory).is_dir())
        self.work.setVisible(bool(result.get('directory')))
        if result.get('status')=='complete':
            width,height = result['output_dimensions']
            message = f'{Path(source).name} · 成片 {width} × {height} · 16 位 PNG 和 JPEG'
            if not self.output.isEnabled():
                message += '\n成片文件已移动或删除，请在照片文件夹中查找。'
        elif result.get('status')=='failed':
            stage = result.get('stage','')
            message = f'{Path(source).name} · 失败'+(f' · {stage}' if stage else '')+'\n'+str(result['error'])
        else:
            message = f'{Path(source).name} · 未完成\n'+str(self.error or (
                '本批次仍在制作，单张结果完成后会显示在这里。' if self.batch_status=='running' else '未保存完整成片。'))
        self.details.setPlainText(message)
