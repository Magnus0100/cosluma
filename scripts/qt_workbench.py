"""Desktop setup, photo queue and previews around the single finishing chain."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import uuid

from PySide6.QtCore import QItemSelectionModel, QSettings, QThread, Signal, Slot, Qt, QUrl, QTimer
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QDialog,
    QFileDialog, QFormLayout, QFrame, QHBoxLayout, QHeaderView, QInputDialog,
    QLabel, QLineEdit, QMainWindow, QMenu, QPushButton, QPlainTextEdit, QProgressBar,
    QScrollArea, QSplitter, QStackedWidget, QStyle, QSystemTrayIcon, QTableWidget, QTableWidgetItem, QToolButton,
    QVBoxLayout, QWidget)

from app_version import VERSION
from studio_widgets import apply_theme, ui_label, PhotoPreview
from creative_workspace import CreativeWorkspace
from batch_results import BatchResults
from delivery_view import DeliveryView
from beauty_editor import BeautyConnection, BeautyEditor
from image_io import IMAGE_EXTENSIONS
from model_clients import select_photos
from photo_pipeline import finish_batch, load_style
from creative_edits import normalise_edits, edits_match
from settings import default_settings_path, load_settings
from service_settings import ServiceSettings, endpoint_url
from project_store import Project, recent_projects, remember_project
from project_browser import NewProject, ProjectBrowser


class ImportWorker(QThread):
    ready = Signal(list, int)
    progress = Signal(str)
    failed = Signal(str)

    def __init__(self, paths, recursive):
        super().__init__()
        self.paths = list(paths)
        self.recursive = recursive

    def run(self):
        def scan_error(error):
            raise error
        try:
            photos, skipped = [], 0
            for value in self.paths:
                path = Path(value).expanduser().resolve()
                if path.is_dir():
                    for parent, directories, files in os.walk(path, onerror=scan_error):
                        if not self.recursive:
                            directories.clear()
                        directories.sort(key=str.casefold)
                        for name in sorted(files, key=str.casefold):
                            candidate = Path(parent)/name
                            if candidate.suffix.lower() in IMAGE_EXTENSIONS:
                                photos.append(str(candidate.resolve()))
                            else:
                                skipped += 1
                        self.progress.emit(f'正在整理导入，已找到 {len(photos)} 张照片…')
                elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    photos.append(str(path))
                else:
                    skipped += 1
            self.ready.emit(photos,skipped)
        except OSError as error:
            self.failed.emit('批量导入失败：'+str(error))


class SelectionWorker(QThread):
    ready = Signal(dict, str)
    progress = Signal(str)
    failed = Signal(str)

    def __init__(self, sources, character, brief, selection_brief, root, config):
        super().__init__()
        self.sources, self.character, self.brief = list(sources), character, brief
        self.selection_brief, self.root, self.config = selection_brief, root, config

    def run(self):
        directory = Path(self.root).expanduser().resolve()/('selection-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
        try:
            directory.mkdir(parents=True,exist_ok=False)
            result = select_photos(self.sources,self.character,self.brief,self.config,directory,
                selection_brief=self.selection_brief,progress=self.progress.emit)
            self.ready.emit(result,str(directory))
        except Exception as error:
            self.failed.emit('Agent 选片失败：'+(getattr(error,'provider_message','') or str(error))+f'。已有勾选保留，文件：{directory}')


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



class Workbench(QMainWindow):
    def __init__(self, root=None, config=None):
        super().__init__()
        self.preferences = QSettings(str(Path.home() / 'CosplayWorkbench/config/desktop.ini'), QSettings.IniFormat)
        self.project = None
        self.project_batches = []
        self.legacy_workspace = None
        self.workspace_restoring = True
        self.workspace_timer = QTimer(self)
        self.workspace_timer.setSingleShot(True)
        self.workspace_timer.setInterval(400)
        self.workspace_timer.timeout.connect(self.save_workspace)
        root = root if root is not None else self.preferences.value('output_root', str(Path.home() / 'CosLuma/output'), type=str)
        config = config or os.environ.get('COSLUMA_CONFIG') or self.preferences.value('config_path', str(default_settings_path()), type=str)
        self.setWindowTitle('CosLuma ' + VERSION)
        self.resize(1280, 820)
        self.setMinimumSize(1080, 680)
        self.setAcceptDrops(True)
        self.worker = None
        self.import_worker = None
        self.selection_worker = None
        self.creative_workspace = None
        self.creative_context = None
        self.creative_direction = ''
        self.creative_edits = normalise_edits()
        self.creative_draft = False
        self.current_batch_view = None
        self.archive_view = None
        self.beauty_editor = None
        self.delivery_view = None
        self.beauty_return_delivery = False
        self.chosen_preview_key = None
        self.restored_page = 0
        self.close_after_finish = False
        self.tray = None
        self.selected_style = None
        self.style_selection_file = None
        self.style_context = None
        self.style_samples_directory = None
        self.style_samples_context = None
        self.style_discussion_directory = None
        self.style_discussion_context = None
        self.current_output = None
        self.current_png = None
        self.batch_directory = None
        self.last_batch_directory = ''
        self.batch_error = ''
        self.photo_errors = {}
        self.stages = {}
        self.preview_key = None
        self.sources = []
        self.excluded_sources = set()
        self.representative_source = None
        self.photo_selection = None
        self.photo_selection_directory = None
        self.photo_selection_error = ''
        self.results = {}
        self.rows = {}
        self.inputs = []

        panel = QWidget()
        self.setCentralWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(10)
        header = QHBoxLayout()
        header.setSpacing(10)
        header.addWidget(ui_label('CosLuma', 'title'))
        self.projects_button = QPushButton('项目')
        self.projects_button.setProperty('role', 'workspaceTab')
        self.projects_button.setCheckable(True)
        self.projects_button.clicked.connect(self.show_projects)
        header.addWidget(self.projects_button)
        self.project_title = QPushButton('新建项目')
        self.project_title.setProperty('kind', 'quiet')
        self.project_title.clicked.connect(self.rename_active_project)
        self.project_title.setMaximumWidth(230)
        header.addWidget(self.project_title)
        self.nav_buttons = []
        for index, text in enumerate(('照片', '创作', '成片')):
            button = QPushButton(text)
            button.setProperty('role', 'workspaceTab')
            button.setCheckable(True)
            button.setMinimumWidth(76)
            button.clicked.connect(lambda _checked=False, index=index: self.navigate(index))
            self.nav_buttons.append(button)
            header.addWidget(button)
        header.addStretch(1)
        self.workspace_context = ui_label('先添加照片与角色')
        self.workspace_context.setMaximumWidth(260)
        header.addWidget(self.workspace_context)
        self.import_button = QPushButton('导入配置')
        self.import_button.setProperty('kind', 'quiet')
        self.import_button.clicked.connect(self.import_config)
        header.addWidget(self.import_button)
        self.setup_button = QPushButton('连接设置')
        self.setup_button.setProperty('kind', 'quiet')
        self.setup_button.clicked.connect(self.configure_services)
        header.addWidget(self.setup_button)
        layout.addLayout(header)
        self.setup_banner = QFrame()
        self.setup_banner.setProperty('role', 'decisionBar')
        setup_row = QHBoxLayout(self.setup_banner)
        setup_row.setContentsMargins(16, 12, 16, 12)
        self.setup_hint = ui_label('首次使用：连接模型服务后，即可与 Agent 讨论你的照片。')
        self.setup_hint.setWordWrap(True)
        setup_row.addWidget(self.setup_hint, 1)
        self.setup_guide_button = QPushButton('开始配置')
        self.setup_guide_button.setProperty('kind', 'primary')
        self.setup_guide_button.clicked.connect(self.configure_services)
        setup_row.addWidget(self.setup_guide_button)
        layout.addWidget(self.setup_banner)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)

        photos = QWidget()
        photos_layout = QVBoxLayout(photos)
        photos_layout.setContentsMargins(0, 0, 0, 0)
        photos_layout.setSpacing(10)
        toolbar = QHBoxLayout()
        import_photos = QToolButton()
        import_photos.setText('批量导入')
        import_photos.setPopupMode(QToolButton.InstantPopup)
        import_menu = QMenu(import_photos)
        import_menu.addAction('选择照片（可多选）',self.select_sources)
        import_menu.addAction('导入文件夹',self.select_folder)
        import_photos.setMenu(import_menu)
        toolbar.addWidget(import_photos)
        self.inputs.append(import_photos)
        self.import_recursive = QCheckBox('包含子文件夹')
        self.import_recursive.setToolTip('导入或拖入文件夹时，连同其中各层子文件夹的照片一起添加')
        self.import_recursive.toggled.connect(self.schedule_workspace_save)
        toolbar.addWidget(self.import_recursive)
        self.inputs.append(self.import_recursive)
        for title, action in (('移除高亮照片', self.remove_selected), ('清空', self.clear_sources),
                ('全选',lambda:self.set_all_included(True)), ('全不选',lambda:self.set_all_included(False))):
            button = QPushButton(title)
            button.clicked.connect(action)
            toolbar.addWidget(button)
            self.inputs.append(button)
        self.count = ui_label('尚未添加照片')
        toolbar.addWidget(self.count, 1)
        photos_layout.addLayout(toolbar)
        selection_row = QHBoxLayout()
        selection_row.addWidget(ui_label('选片要求'))
        self.selection_brief = QLineEdit()
        self.selection_brief.setPlaceholderText('可留空；例如选 8 张，近景和全身都要，连拍只留一张')
        self.selection_brief.textChanged.connect(self.schedule_workspace_save)
        selection_row.addWidget(self.selection_brief,1)
        self.selection_button = QPushButton('Agent 帮我选片')
        self.selection_button.setToolTip('查看全部导入照片，重新给出选片建议；可继续手动调整勾选')
        self.selection_button.clicked.connect(self.request_photo_selection)
        selection_row.addWidget(self.selection_button)
        self.inputs.append(self.selection_brief)
        photos_layout.addLayout(selection_row)
        self.selection_summary = ui_label('勾选用于制作，点击行查看原片；Agent 会查看全部导入照片。')
        self.selection_summary.setWordWrap(True)
        photos_layout.addWidget(self.selection_summary)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(14)
        queue_panel = QFrame()
        queue_panel.setProperty('role', 'panel')
        queue_layout = QVBoxLayout(queue_panel)
        queue_layout.setContentsMargins(14, 14, 14, 14)
        queue_layout.setSpacing(12)
        self.queue_pages = QStackedWidget()
        empty = QLabel('一次拖入多张照片或多个文件夹\n勾选“包含子文件夹”可连同各层照片导入\n\nJPEG、PNG、TIFF、HEIC / HEIF、WebP、BMP 和相机 RAW')
        empty.setAlignment(Qt.AlignCenter)
        empty.setWordWrap(True)
        empty.setProperty('role', 'dropArea')
        self.queue_pages.addWidget(empty)
        self.listing = QTableWidget(0, 2)
        self.listing.setHorizontalHeaderLabels(['勾选制作 · 点击查看', '制作进度'])
        self.listing.verticalHeader().setVisible(True)
        self.listing.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.listing.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.listing.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.listing.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listing.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.listing.setAcceptDrops(False)
        self.listing.viewport().setAcceptDrops(False)
        self.listing.itemSelectionChanged.connect(self.show_selected)
        self.listing.itemChanged.connect(self.photo_check_changed)
        self.queue_pages.addWidget(self.listing)
        queue_layout.addWidget(self.queue_pages, 1)
        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(10)
        self.character = QLineEdit()
        self.character.setPlaceholderText('例如：鸣潮 · 守岸人')
        form.addRow('角色', self.character)
        self.brief = QPlainTextEdit()
        self.brief.setPlaceholderText('可留空；补充想保留的细节或场景想法')
        self.brief.setFixedHeight(64)
        form.addRow('补充要求', self.brief)
        self.root = QLineEdit(str(root))
        output_row = self.path_row(self.root, directory=True)
        form.addRow('保存到', output_row)
        queue_layout.addWidget(form_host)
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
        advanced_form.addRow('配置文件', self.path_row(self.config))
        self.edge = QLineEdit(self.preferences.value('output_edge', '', type=str))
        self.edge.setPlaceholderText('留空使用配置；默认 6000')
        advanced_form.addRow('请求长边', self.edge)
        advanced_layout.addWidget(advanced_panel)
        advanced_panel.hide()
        advanced_toggle.toggled.connect(advanced_panel.setVisible)
        advanced_toggle.toggled.connect(lambda checked: advanced_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow))
        queue_layout.addWidget(advanced)
        queue_scroll = QScrollArea()
        queue_scroll.setWidgetResizable(True)
        queue_scroll.setMinimumWidth(320)
        queue_scroll.setWidget(queue_panel)
        splitter.addWidget(queue_scroll)
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.photo_preview = PhotoPreview()
        preview_layout.addWidget(self.photo_preview, 1)
        self.output_info = ui_label('')
        self.output_info.setWordWrap(True)
        self.output_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details = QScrollArea()
        details.setWidgetResizable(True)
        details.setMinimumHeight(40)
        details.setMaximumHeight(70)
        details.setWidget(self.output_info)
        preview_layout.addWidget(details)
        self.selection_reason = ui_label('')
        self.selection_reason.setWordWrap(True)
        self.selection_reason.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.selection_reason.hide()
        preview_layout.addWidget(self.selection_reason)
        photo_controls = QHBoxLayout()
        for name, text, action in (('open_source_button', '打开原片', self.open_source),
                ('open_work_button', '打开此照片文件夹', self.open_work_directory),
                ('open_photo_button', '打开 JPEG', self.open_selected_output),
                ('open_png_button', '打开 16 位 PNG', self.open_selected_png),
                ('representative_button', '设为试样原片', self.set_representative)):
            button = QPushButton(text)
            button.setProperty('kind', 'quiet')
            button.setEnabled(False)
            button.clicked.connect(action)
            setattr(self, name, button)
            photo_controls.addWidget(button)
        photo_controls.addStretch(1)
        preview_layout.addLayout(photo_controls)
        splitter.addWidget(preview_panel)
        splitter.setSizes([350, 880])
        photos_layout.addWidget(splitter, 1)
        next_row = QHBoxLayout()
        self.style_hint = ui_label('先添加照片、填写角色，再一起讨论摄影方向。')
        self.style_hint.setWordWrap(True)
        next_row.addWidget(self.style_hint, 1)
        self.style_button = QPushButton('与 Agent 讨论风格')
        self.style_button.setProperty('kind', 'primary')
        self.style_button.clicked.connect(self.choose_style)
        next_row.addWidget(self.style_button)
        photos_layout.addLayout(next_row)
        self.pages.addWidget(photos)

        self.creative_page = QWidget()
        self.creative_layout = QVBoxLayout(self.creative_page)
        self.creative_layout.setContentsMargins(0, 0, 0, 0)
        self.creative_empty = QFrame()
        empty_layout = QVBoxLayout(self.creative_empty)
        empty_layout.addStretch(1)
        empty_layout.addWidget(ui_label('从你的原片开始', 'title'), 0, Qt.AlignCenter)
        self.creative_next = ui_label('添加照片和角色后，在这里与摄影 Agent 一起确定方向。')
        self.creative_next.setWordWrap(True)
        self.creative_next.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(self.creative_next)
        preparation = QHBoxLayout()
        preparation.addStretch(1)
        self.creative_prepare = QPushButton('添加照片')
        self.creative_prepare.setProperty('kind','primary')
        self.creative_prepare.clicked.connect(self.prepare_creative)
        preparation.addWidget(self.creative_prepare)
        self.creative_import = QPushButton('导入配置')
        self.creative_import.clicked.connect(self.import_config)
        preparation.addWidget(self.creative_import)
        preparation.addStretch(1)
        empty_layout.addLayout(preparation)
        empty_layout.addStretch(1)
        self.creative_layout.addWidget(self.creative_empty)
        self.pages.addWidget(self.creative_page)

        results_page = QWidget()
        results_layout = QVBoxLayout(results_page)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(10)
        results_header = QHBoxLayout()
        self.results_caption = ui_label('本次制作', 'heading')
        results_header.addWidget(self.results_caption, 1)
        self.delivery_button = QPushButton('项目作品 / 交付')
        self.delivery_button.setProperty('kind','quiet')
        self.delivery_button.clicked.connect(self.show_delivery)
        results_header.addWidget(self.delivery_button)
        self.return_current = QPushButton('回到本次制作')
        self.return_current.setProperty('kind', 'quiet')
        self.return_current.clicked.connect(self.show_current_batch)
        self.return_current.hide()
        results_header.addWidget(self.return_current)
        self.last_batch_button = QToolButton()
        self.last_batch_button.setText('查看已有结果')
        self.last_batch_button.setPopupMode(QToolButton.MenuButtonPopup)
        saved_menu = QMenu(self)
        self.project_batches_menu = saved_menu.addMenu('本项目的制作批次')
        self.last_batch_action = saved_menu.addAction('查看上次制作', self.open_last_batch)
        saved_menu.addAction('选择其他批次文件夹', self.choose_saved_batch)
        self.last_batch_button.setMenu(saved_menu)
        self.last_batch_button.clicked.connect(self.open_last_batch)
        results_header.addWidget(self.last_batch_button)
        self.open_folder_button = QPushButton('打开当前批次文件夹')
        self.open_folder_button.setProperty('kind', 'quiet')
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self.open_results)
        results_header.addWidget(self.open_folder_button)
        results_layout.addLayout(results_header)
        self.result_views = QStackedWidget()
        self.results_empty = QFrame()
        waiting = QVBoxLayout(self.results_empty)
        waiting.addStretch(1)
        self.results_next = ui_label('定稿后，准备制作这一组照片', 'heading')
        self.results_next.setAlignment(Qt.AlignCenter)
        self.results_next.setWordWrap(True)
        waiting.addWidget(self.results_next)
        self.results_note = ui_label('先在创作页比较三个真实样片，选择喜欢的方向。')
        self.results_note.setAlignment(Qt.AlignCenter)
        self.results_note.setWordWrap(True)
        waiting.addWidget(self.results_note)
        explore = QPushButton('去创作页')
        explore.clicked.connect(self.choose_style)
        waiting.addWidget(explore, 0, Qt.AlignCenter)
        waiting.addStretch(1)
        self.result_views.addWidget(self.results_empty)
        results_layout.addWidget(self.result_views, 1)
        decision = QFrame()
        self.decision_bar = decision
        decision.setProperty('role', 'decisionBar')
        decision_layout = QHBoxLayout(decision)
        decision_layout.setContentsMargins(14, 12, 14, 12)
        decision_layout.setSpacing(14)
        self.chosen_thumbnail = QLabel('尚未定稿')
        self.chosen_thumbnail.setProperty('role', 'chosenPreview')
        self.chosen_thumbnail.setFixedSize(112, 80)
        self.chosen_thumbnail.setAlignment(Qt.AlignCenter)
        decision_layout.addWidget(self.chosen_thumbnail)
        chosen_text = QVBoxLayout()
        chosen_text.setSpacing(4)
        self.chosen_title = ui_label('先选择真实样片', 'heading')
        self.chosen_title.setWordWrap(True)
        chosen_text.addWidget(self.chosen_title)
        self.chosen_source = ui_label('定稿只保存方向，制作由你另行开始。')
        chosen_text.addWidget(self.chosen_source)
        self.chosen_summary = ui_label('')
        self.chosen_summary.setWordWrap(True)
        self.chosen_summary.setMaximumHeight(36)
        chosen_text.addWidget(self.chosen_summary)
        decision_layout.addLayout(chosen_text, 1)
        self.view_chosen_button = QPushButton('查看定稿')
        self.view_chosen_button.setProperty('kind', 'quiet')
        self.view_chosen_button.clicked.connect(self.view_chosen)
        decision_layout.addWidget(self.view_chosen_button)
        controls = QVBoxLayout()
        self.start = QPushButton('按选定风格批量制作')
        self.start.setProperty('kind', 'primary')
        self.start.setMinimumHeight(40)
        self.start.clicked.connect(lambda: self.produce(only_pending=True))
        controls.addWidget(self.start)
        self.remaining_button = QPushButton('勾选照片另做一批')
        self.remaining_button.setProperty('kind', 'quiet')
        self.remaining_button.clicked.connect(self.produce)
        self.remaining_button.hide()
        controls.addWidget(self.remaining_button)
        decision_layout.addLayout(controls)
        results_layout.addWidget(decision)
        self.readiness = ui_label('先比较样片并定稿，再制作当前照片队列。')
        self.readiness.setWordWrap(True)
        results_layout.addWidget(self.readiness)
        self.pages.addWidget(results_page)

        self.welcome_page = QWidget()
        welcome_layout = QVBoxLayout(self.welcome_page)
        welcome_layout.setContentsMargins(36, 24, 36, 24)
        welcome_layout.addStretch(1)
        welcome_body = QHBoxLayout()
        welcome_body.setSpacing(44)
        introduction = QVBoxLayout()
        introduction.setSpacing(16)
        introduction.addWidget(ui_label('从你的原片，开始一组创作', 'title'))
        note = ui_label('先与摄影 Agent 聊想法，比较三个真实样片；\n选定喜欢的方向，再制作整组照片。')
        note.setWordWrap(True)
        introduction.addWidget(note)
        for title, description in (('先讨论', '提供照片和角色，一起确定摄影方向。'),
                ('看样片', '同一张原片做三个方向，由你决定定稿。'),
                ('整组出片', '沿用定稿制作，原片和基础成片分别保留。')):
            introduction.addWidget(ui_label(title, 'heading'))
            text = ui_label(description)
            text.setWordWrap(True)
            introduction.addWidget(text)
        welcome_body.addLayout(introduction, 1)
        connection = QFrame()
        connection.setProperty('role', 'panel')
        connection.setMaximumWidth(410)
        connection_layout = QVBoxLayout(connection)
        connection_layout.setContentsMargins(24, 24, 24, 24)
        connection_layout.setSpacing(16)
        connection_layout.addWidget(ui_label('先连接你的模型服务', 'heading'))
        note = ui_label('你需要图文模型、图像编辑服务，以及已部署分割与超分模型的 ComfyUI。软件包不含服务、账号或模型权重。')
        note.setWordWrap(True)
        connection_layout.addWidget(note)
        begin = QPushButton('开始配置')
        begin.setProperty('kind', 'primary')
        begin.clicked.connect(self.configure_services)
        connection_layout.addWidget(begin)
        imported = QPushButton('导入已有配置')
        imported.clicked.connect(self.import_config)
        connection_layout.addWidget(imported)
        later = QPushButton('先整理照片，稍后配置')
        later.setProperty('kind', 'quiet')
        later.clicked.connect(self.defer_setup)
        connection_layout.addWidget(later)
        note = ui_label('保存设置不调用模型。照片只会在你主动选片、发送讨论或制作时交给所配服务。')
        note.setWordWrap(True)
        connection_layout.addWidget(note)
        welcome_body.addWidget(connection, 1)
        welcome_layout.addLayout(welcome_body)
        welcome_layout.addStretch(1)
        self.welcome_index = self.pages.addWidget(self.welcome_page)

        self.project_browser = ProjectBrowser(self)
        self.projects_index = self.pages.addWidget(self.project_browser)
        self.project_browser.create_requested.connect(self.new_project)
        self.project_browser.browse_requested.connect(self.browse_project)
        self.project_browser.legacy_requested.connect(lambda:self.new_project(from_legacy=True))
        self.project_browser.open_requested.connect(self.open_project)
        self.project_browser.rename_requested.connect(self.rename_project)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('等待开始')
        layout.addWidget(self.progress_bar)
        self.status = ui_label('新建或打开项目，继续照片创作。')
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status.setMaximumHeight(42)
        layout.addWidget(self.status)
        self.inputs.extend([self.character, self.brief, output_row, advanced, self.setup_button,
            self.import_button, self.setup_guide_button])
        for edit in (self.character, self.brief, self.root, self.config, self.edge):
            edit.textChanged.connect(self.update_actions)
            edit.textChanged.connect(self.schedule_workspace_save)
        self.config.textChanged.connect(self.update_setup_hint)
        self.update_setup_hint()
        self.workspace_restoring = False
        self.show_projects()
        self.load_initial_project()
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
        return os.path.normcase(str(source))

    def refresh_projects(self):
        self.project_browser.refresh(recent_projects(self.preferences), self.project)

    def show_projects(self):
        if self.project is not None and not self.persist_project():
            return
        try:
            self.refresh_projects()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.status.setText('读取项目列表失败：'+str(error))
        self.navigate(self.projects_index)

    def load_initial_project(self):
        try:
            saved = self.preferences.value('workspace', '', type=str)
            if saved and not self.preferences.value('projects/migrated', '', type=str):
                state = json.loads(saved)
                creative = {name:self.preferences.value('style/'+name, '', type=str)
                    for name in ('discussion', 'samples', 'selected')}
                if state.get('sources') or state.get('character') or state.get('brief') or state.get('batch_directory') or any(creative.values()):
                    state.setdefault('output_root', self.preferences.value('output_root', self.root.text(), type=str))
                    state.setdefault('output_edge', self.preferences.value('output_edge', '', type=str))
                    self.legacy_workspace = (state, creative)
                    self.project_browser.set_legacy_available(True)
        except (ValueError, KeyError, TypeError) as error:
            self.status.setText('读取上次工作区失败：'+str(error))

    def new_project(self, *, from_legacy=False):
        if self.work_in_progress():
            return
        parent = self.preferences.value('projects/directory', str(Path.home()/'CosLuma/projects'), type=str)
        dialog = NewProject(parent, self)
        if from_legacy:
            dialog.setWindowTitle('将上次工作区收为项目')
            dialog.name.setText((self.legacy_workspace[0].get('character') or '上次创作')+' · 原工作区')
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            if not self.persist_project():
                return
            legacy = self.legacy_workspace if from_legacy else None
            if legacy is not None:
                workspace, creative = deepcopy(legacy)
                last = self.preferences.value('last_batch_directory', '', type=str)
                recorded = [result.get('batch_directory') for result in workspace['results']]
                batches = list(dict.fromkeys(path for path in [*recorded, workspace.get('batch_directory'), last] if path))
            else:
                workspace, creative, batches = None, None, None
            project = Project.create(dialog.directory, dialog.name.text(), workspace=workspace,
                creative=creative, batches=batches)
            self.preferences.setValue('projects/directory', str(dialog.directory.parent))
            self.activate_project(project)
            if legacy is not None:
                self.preferences.setValue('projects/migrated', str(project.path))
                self.legacy_workspace = None
                self.project_browser.set_legacy_available(False)
                self.preferences.sync()
            self.status.setText('项目已创建：'+project.data['name']+'。'+
                ('已收纳上次的照片、讨论、样片与定稿，已有文件保持原位置。' if legacy else '可以导入照片，开始这组创作。'))
            return project
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.status.setText('创建项目失败：'+str(error))

    def browse_project(self):
        if self.work_in_progress():
            return
        path, _ = QFileDialog.getOpenFileName(self, '打开 CosLuma 项目',
            self.preferences.value('projects/directory', str(Path.home()/'CosLuma/projects'), type=str),
            'CosLuma 项目 (project.json)')
        if path:
            self.open_project(path)

    def open_project(self, path):
        if self.work_in_progress():
            self.status.setText('当前项目仍在处理中，结束后可切换项目。')
            return
        try:
            project = Project.open(path)
            if self.project is not None and project.path == self.project.path:
                self.navigate(self.restored_page if self.pages.currentIndex() == self.projects_index else 0)
                return
            if self.persist_project():
                self.activate_project(project)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.status.setText('打开项目失败：'+str(error))

    def persist_project(self):
        if self.project is None:
            return True
        if self.creative_workspace is not None and not self.creative_workspace.flush_draft():
            self.status.setText(self.creative_workspace.message.toPlainText())
            return False
        if self.beauty_editor is not None:
            self.beauty_editor.draft_timer.stop()
            if not self.beauty_editor.save_draft():
                self.status.setText(self.beauty_editor.message.text())
                return False
        if self.delivery_view is not None and not self.delivery_view.save_state():
            self.status.setText(self.delivery_view.message.toPlainText())
            return False
        return self.save_workspace()

    def activate_project(self, project):
        self.workspace_restoring = True
        self.workspace_timer.stop()
        if self.creative_workspace is not None:
            self.creative_workspace.draft_timer.stop()
            self.creative_layout.removeWidget(self.creative_workspace)
            self.creative_workspace.deleteLater()
            self.creative_workspace = None
        for view in (self.current_batch_view, self.archive_view, self.beauty_editor,self.delivery_view):
            if view is not None:
                self.result_views.removeWidget(view)
                view.deleteLater()
        self.current_batch_view = self.archive_view = self.beauty_editor = None
        self.delivery_view = None
        self.beauty_return_delivery = False
        self.result_views.setCurrentWidget(self.results_empty)
        self.creative_context = self.style_context = None
        self.style_samples_context = self.style_discussion_context = None
        self.selected_style = None
        self.style_selection_file = project.data['creative'].get('selected') or None
        self.style_samples_directory = self.style_discussion_directory = None
        self.creative_direction = ''
        self.creative_edits = normalise_edits()
        self.creative_draft = False
        self.sources, self.results, self.stages, self.photo_errors = [], {}, {}, {}
        self.excluded_sources = set()
        self.representative_source = self.photo_selection = self.photo_selection_directory = None
        self.photo_selection_error = self.batch_error = ''
        self.preview_key = self.chosen_preview_key = self.current_output = self.current_png = None
        self.photo_preview.set_photo()
        self.batch_directory = None
        self.project = project
        self.status.setText('已打开项目：'+project.data['name'])
        self.project_batches = list(project.data['batches'])
        self.last_batch_directory = self.project_batches[-1] if self.project_batches else ''
        self.root.setText(project.data['workspace']['output_root'])
        self.edge.setText(project.data['workspace'].get('output_edge', ''))
        self.restore_workspace()
        self.restore_style_selection()
        self.workspace_restoring = False
        self.refresh_batch_menu()
        remember_project(self.preferences, project)
        self.refresh_projects()
        if project.data['workspace'].get('results_view')=='delivery':
            self.ensure_delivery()
            self.result_views.setCurrentWidget(self.delivery_view)
        self.navigate(self.restored_page)
        if (not Path(self.config.text()).expanduser().is_file() and not self.sources and not self.last_batch_directory
                and not self.preferences.value('setup_deferred', False, type=bool)):
            self.navigate(self.welcome_index)
        self.update_actions()
        if not self.save_workspace():
            self.status.setText('项目已打开，保存失败。请检查项目文件夹的写入权限。')

    def rename_active_project(self):
        if self.project is None:
            self.new_project()
        else:
            self.rename_project(str(self.project.path))

    def rename_project(self, path):
        if self.work_in_progress():
            return
        try:
            project = self.project if self.project is not None and self.project.path == Path(path).resolve() else Project.open(path)
            name, accepted = QInputDialog.getText(self, '重命名项目', '项目名称', text=project.data['name'])
            if not accepted:
                return
            if project is self.project and not self.persist_project():
                return
            project.rename(name)
            self.refresh_projects()
            self.update_actions()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.status.setText('重命名项目失败：'+str(error))

    def refresh_batch_menu(self):
        self.project_batches_menu.clear()
        for directory in reversed(self.project_batches):
            self.project_batches_menu.addAction(Path(directory).name,
                lambda _checked=False, directory=directory:self.show_saved_batch(directory))
        self.project_batches_menu.setEnabled(bool(self.project_batches))

    def schedule_workspace_save(self):
        if not self.workspace_restoring and self.project is not None:
            self.workspace_timer.start()

    def save_workspace(self):
        if self.workspace_restoring or self.project is None:
            return True
        self.workspace_timer.stop()
        fields = ('source','status','directory','batch_directory','files','exports',
            'output_dimensions','output_adjustment','stage','error')
        row = self.listing.currentRow()
        state = {'sources':self.sources,'character':self.character.text(),'brief':self.brief.toPlainText(),
            'excluded_sources':sorted(self.excluded_sources),'representative_source':self.representative_source,
            'import_recursive':self.import_recursive.isChecked(),
            'selection_brief':self.selection_brief.text(),'photo_selection':self.photo_selection,
            'photo_selection_directory':self.photo_selection_directory,'photo_selection_error':self.photo_selection_error,
            'page':self.pages.currentIndex() if self.pages.currentIndex() in (0, 1, 2) else self.restored_page,
            'results_view':'delivery' if self.result_views.currentWidget() is self.delivery_view or
                (self.result_views.currentWidget() is self.beauty_editor and self.beauty_return_delivery) else 'batch',
            'selected_source':self.sources[row] if 0<=row<len(self.sources) else None,
            'results':[{name:result[name] for name in fields if name in result} for result in self.results.values()],
            'stages':self.stages,'photo_errors':self.photo_errors,
            'batch_directory':self.batch_directory,'batch_error':self.batch_error,
            'output_root':self.root.text(), 'output_edge':self.edge.text()}
        creative = {'discussion':self.style_discussion_directory or '',
            'samples':self.style_samples_directory or '',
            'selected':self.style_selection_file or ''}
        try:
            self.project.save(state, creative, self.project_batches)
        except OSError as error:
            self.status.setText('项目保存失败：'+str(error)+'。当前窗口内容与已有成片保留。')
            return False
        return True

    def restore_workspace(self):
        try:
            state = deepcopy(self.project.data['workspace'])
            self.restored_page = state.get('page', 0)
            self.sources = state['sources']
            known = {self.source_key(source) for source in self.sources}
            self.excluded_sources = set(state.get('excluded_sources',[])) & known
            self.import_recursive.setChecked(state.get('import_recursive',False))
            self.representative_source = state.get('representative_source')
            self.selection_brief.setText(state.get('selection_brief',''))
            self.photo_selection = state.get('photo_selection')
            self.photo_selection_directory = state.get('photo_selection_directory')
            self.photo_selection_error = state.get('photo_selection_error','')
            self.character.setText(state.get('character',self.character.text()))
            self.brief.setPlainText(state.get('brief',self.brief.toPlainText()))
            self.batch_directory = state.get('batch_directory')
            self.batch_error = state.get('batch_error','')
            self.photo_errors = {key:str(error) for key,error in state.get('photo_errors',{}).items() if key in known}
            self.results = {self.source_key(result['source']):result for result in state['results']
                if self.source_key(result['source']) in known}
            for key,stage in state.get('stages',{}).items():
                if key in known and key not in self.results:
                    self.stages[key] = '未完成'
                    self.photo_errors.setdefault(key,'上次停在“'+stage+'”。未自动继续制作；如需重做，请明确选择“勾选照片另做一批”。')
            if self.batch_directory and self.stages:
                batch = json.loads((Path(self.batch_directory)/'batch.json').read_text(encoding='utf-8'))
                for result in batch['items'].values():
                    key = self.source_key(result['source'])
                    if key in self.stages and result['status'] in ('complete','failed'):
                        self.results[key] = result
                        self.stages.pop(key)
                        self.photo_errors.pop(key,None)
                if batch['status']=='failed':
                    self.batch_error = batch['error']
            self.refresh_queue()
            selected = state.get('selected_source')
            if selected and self.source_key(selected) in self.rows:
                self.listing.selectRow(self.rows[self.source_key(selected)])
            self.open_folder_button.setEnabled(bool(self.batch_directory))
            interrupted = len(self.stages)
            missing = sum(not Path(source).is_file() for source in self.sources)
            if self.sources:
                self.status.setText(f'已恢复 {len(self.sources)} 张照片和已有制作状态。'+
                    (f'{interrupted} 张上次未完成，未自动重做。' if interrupted else '')+
                    (f'{missing} 张原片已移动或缺失，请移除后重新添加。' if missing else ''))
        except (OSError,ValueError,KeyError) as error:
            self.refresh_queue()
            self.status.setText('上次工作区记录无法完整恢复：'+str(error)+'。已有成片可从“查看已有结果”打开。')

    def current_style_context(self):
        return (self.character.text().strip(),self.brief.toPlainText())

    def style_is_current(self):
        if self.selected_style is None or self.style_context != self.current_style_context():
            return False
        if self.creative_workspace is not None and self.creative_workspace.is_busy():
            return False
        return self.creative_context != self.style_context or (
            not self.creative_draft and
            self.creative_direction == self.selected_style['art_direction'].get('creative_direction', '') and
            edits_match(self.creative_edits,self.selected_style['art_direction'].get('edits')))

    def restore_style_selection(self):
        creative = self.project.data['creative']
        self.style_discussion_directory = creative.get('discussion') or None
        self.style_samples_directory = creative.get('samples') or None
        try:
            discussion = creative.get('discussion')
            if discussion:
                state = json.loads((Path(discussion) / 'conversation.json').read_text(encoding='utf-8'))
                self.style_discussion_directory = discussion
                self.style_discussion_context = (state['character'], state['brief'])
                self.creative_context = self.style_discussion_context
                self.creative_direction = state.get('direction', '')
                self.creative_edits = normalise_edits(state.get('edits'))
                self.creative_draft = bool(state.get('draft', '').strip())
            directory = creative.get('samples')
            if directory:
                state = json.loads((Path(directory) / 'styles.json').read_text(encoding='utf-8'))
                self.style_samples_directory = directory
                self.style_samples_context = (state['character'], state['brief'])
            chosen = creative.get('selected')
            if chosen:
                self.selected_style = load_style(chosen)
                self.style_context = (self.selected_style['result']['character'], self.selected_style['result']['brief'])
        except (OSError, ValueError, KeyError) as error:
            self.status.setText('读取上次创作内容失败：' + str(error))

    def desktop_config(self):
        config = load_settings(self.config.text())
        if self.edge.text().strip():
            edge = int(self.edge.text())
            if edge <= 0:
                raise ValueError('请求长边应为正整数')
            config['output_long_edge'] = edge
        return config

    def choose_style(self):
        self.navigate(1)

    def defer_setup(self):
        self.preferences.setValue('setup_deferred', True)
        self.preferences.sync()
        self.navigate(0)

    def update_setup_hint(self):
        exists = Path(self.config.text()).expanduser().is_file()
        self.setup_banner.setVisible(not exists and self.pages.currentIndex()!=self.welcome_index)

    def import_config(self):
        filename, _ = QFileDialog.getOpenFileName(self, '导入已有服务配置', self.config.text(), 'JSON (*.json)')
        if not filename:
            return
        try:
            load_settings(filename)
            self.config.setText(filename)
            self.remember_preferences()
            self.status.setText('已导入配置。可以导入照片，与 Agent 讨论创作方向。')
            self.refresh_creative_preparation()
        except Exception as error:
            self.status.setText('无法导入配置：' + str(error))

    def open_last_batch(self):
        if self.last_batch_directory:
            self.show_saved_batch(self.last_batch_directory)
        else:
            self.choose_saved_batch()

    def choose_saved_batch(self):
        directory = QFileDialog.getExistingDirectory(self, '选择一次制作的结果文件夹', self.root.text())
        if directory:
            self.show_saved_batch(directory)

    def show_saved_batch(self, directory):
        try:
            path = Path(directory).expanduser().resolve()
            if self.current_batch_view is not None and path == self.current_batch_view.directory:
                view = self.current_batch_view
                view.refresh()
            elif self.archive_view is not None and path == self.archive_view.directory:
                view = self.archive_view
                view.refresh()
            else:
                view = self.make_results_view(path)
                if self.archive_view is not None:
                    self.result_views.removeWidget(self.archive_view)
                    self.archive_view.deleteLater()
                self.archive_view = view
            self.result_views.setCurrentWidget(view)
            self.navigate(2)
        except (OSError, ValueError, KeyError) as error:
            self.status.setText('读取已有结果失败：' + str(error))

    def configure_services(self):
        try:
            dialog = ServiceSettings(self.config.text(), self, output_root=self.root.text(),
                project_output=self.project is not None)
            if dialog.exec() == QDialog.Accepted:
                self.config.setText(str(dialog.path))
                if self.project is not None:
                    self.root.setText(str(Path(dialog.output_root.text()).expanduser().resolve()))
                self.remember_preferences()
                self.status.setText('设置已保存。'+('可以继续与 Agent 讨论。' if self.sources else
                    '下一步：批量导入照片，填写角色。' if self.project is not None else '新建或打开项目后开始创作。'))
                self.refresh_creative_preparation()
            elif self.pages.currentIndex()==self.welcome_index:
                self.defer_setup()
        except Exception as error:
            self.status.setText('无法打开连接设置：' + str(error))

    def refresh_creative_preparation(self):
        self.update_setup_hint()
        if self.pages.currentIndex()==self.welcome_index:
            self.navigate(0)
        elif self.pages.currentIndex()==1:
            self.navigate(1)
        else:
            self.update_actions()

    def prepare_creative(self):
        context = self.current_style_context()
        chosen = self.selected_style if self.style_context==context else None
        if not self.included_sources() and chosen is None:
            if self.sources:
                self.navigate(0)
                self.status.setText('请先勾选要制作的照片，或让 Agent 选片。')
            else:
                self.select_sources()
        elif not context[0]:
            self.navigate(0)
            self.character.setFocus()
        else:
            self.configure_services()

    def remember_preferences(self):
        self.preferences.setValue('config_path', self.config.text().strip())
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
        if self.work_in_progress() or not paths:
            return
        if self.project is None or self.pages.currentIndex() == self.projects_index:
            if self.new_project() is None:
                return
        self.navigate(0)
        self.import_worker = ImportWorker(paths,self.import_recursive.isChecked())
        self.import_worker.ready.connect(self.import_ready)
        self.import_worker.progress.connect(self.status.setText)
        self.import_worker.failed.connect(self.status.setText)
        self.import_worker.finished.connect(self.import_finished)
        self.status.setText('正在批量导入照片'+('，包含各层子文件夹…' if self.import_recursive.isChecked() else '…'))
        self.update_actions()
        self.import_worker.start()

    def import_ready(self, candidates, skipped):
        known = {self.source_key(source) for source in self.sources}
        added = duplicates = 0
        for source in candidates:
            key = self.source_key(source)
            if key in known:
                duplicates += 1
            else:
                self.sources.append(source)
                known.add(key)
                added += 1
        self.refresh_queue()
        notes = [f'添加 {added} 张照片']
        if duplicates:
            notes.append(f'{duplicates} 张已在队列中')
        if skipped:
            notes.append(f'{skipped} 项不是支持的照片，已跳过')
        self.status.setText('；'.join(notes) + '。')

    def import_finished(self):
        worker,self.import_worker = self.import_worker,None
        if self.pages.currentIndex()==1:
            self.navigate(1)
        self.update_actions()
        self.finish_background_work()
        worker.deleteLater()

    def select_sources(self):
        paths, _ = QFileDialog.getOpenFileNames(self, '批量导入原片（可多选）', '',
            '照片 (' + ' '.join('*' + suffix for suffix in sorted(IMAGE_EXTENSIONS)) + ')')
        self.add_sources(paths)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self,
            '导入文件夹及其子文件夹中的照片' if self.import_recursive.isChecked() else '导入文件夹中的照片')
        if folder:
            self.add_sources([folder])

    def remove_selected(self):
        if self.work_in_progress():
            return
        rows = {item.row() for item in self.listing.selectedItems()}
        self.sources = [source for row, source in enumerate(self.sources) if row not in rows]
        self.refresh_queue()

    def clear_sources(self):
        if self.work_in_progress():
            return
        self.sources.clear()
        self.results.clear()
        self.photo_selection = None
        self.photo_selection_directory = None
        self.photo_selection_error = ''
        self.refresh_queue()

    def included_sources(self):
        return [source for source in self.sources if self.source_key(source) not in self.excluded_sources]

    def sync_creative_sources(self):
        sources = self.included_sources()
        if self.representative_source not in sources:
            self.representative_source = sources[0] if sources else None
        representative = self.representative_source
        workspace = self.creative_workspace
        if workspace is not None and (workspace.character,workspace.brief)==self.current_style_context():
            workspace.sync_sources(sources)
            self.representative_source = representative
            if representative:
                workspace.sample_source.setCurrentIndex(workspace.sample_source.findData(representative))

    def photo_check_changed(self, item):
        if item.column()!=0 or self.work_in_progress():
            return
        key = self.source_key(item.data(Qt.UserRole))
        included = item.checkState()==Qt.Checked
        if included==(key not in self.excluded_sources):
            return
        self.excluded_sources.discard(key) if included else self.excluded_sources.add(key)
        self.refresh_queue()

    def set_all_included(self, included):
        if self.work_in_progress():
            return
        self.excluded_sources = set() if included else {self.source_key(source) for source in self.sources}
        self.refresh_queue()

    def set_representative(self):
        if self.work_in_progress():
            return
        row = self.listing.currentRow()
        if row<0 or row>=len(self.sources):
            return
        source = self.sources[row]
        if self.source_key(source) in self.excluded_sources:
            return
        self.representative_source = source
        self.sync_creative_sources()
        self.refresh_queue()
        self.status.setText('试样原片已设为 '+Path(source).name+'，查看其他照片不会改变它。')

    def request_photo_selection(self):
        if self.work_in_progress() or not self.sources:
            return
        try:
            if not Path(self.config.text()).expanduser().is_file():
                self.configure_services()
                if not Path(self.config.text()).expanduser().is_file():
                    return
            config = load_settings(self.config.text())
            if not self.root.text().strip():
                raise ValueError('请选择选片记录的保存目录')
            self.selection_worker = SelectionWorker(self.sources,self.character.text().strip(),self.brief.toPlainText(),
                self.selection_brief.text().strip(),self.root.text(),config['llm'])
            self.selection_worker.ready.connect(self.photo_selection_ready)
            self.selection_worker.progress.connect(self.status.setText)
            self.selection_worker.failed.connect(self.photo_selection_failed)
            self.selection_worker.finished.connect(self.photo_selection_finished)
            self.photo_selection_error = ''
            self.status.setText('Agent 正在查看全部导入照片；选片后可继续调整勾选。')
            self.update_actions()
            self.selection_worker.start()
        except Exception as error:
            self.selection_worker = None
            self.photo_selection_failed('无法开始选片：'+str(error))
            self.update_actions()

    def photo_selection_ready(self, result, directory):
        self.photo_selection = result
        self.photo_selection_directory = directory
        self.photo_selection_error = ''
        self.excluded_sources = {self.source_key(photo['source']) for photo in result['photos'] if not photo['selected']}
        self.representative_source = result['representative']
        self.refresh_queue()
        if self.representative_source:
            self.listing.selectRow(self.rows[self.source_key(self.representative_source)])
        self.status.setText('Agent 选片建议已应用，可逐张调整勾选，或进入创作讨论。')
        self.save_workspace()

    def photo_selection_failed(self, error):
        self.photo_selection_error = error
        self.status.setText(error)
        self.update_selection_summary()
        self.schedule_workspace_save()

    def photo_selection_finished(self):
        worker,self.selection_worker = self.selection_worker,None
        self.update_actions()
        self.finish_background_work()
        worker.deleteLater()

    def update_selection_summary(self):
        if self.photo_selection_error:
            text = self.photo_selection_error
        elif self.photo_selection is not None:
            result = self.photo_selection
            selected = sum(photo['selected'] for photo in result['photos'])
            text = f'Agent 建议 {selected}/{len(result["photos"])} 张。'+result['summary']
            if self.representative_source:
                text += ' 试样原片：'+Path(self.representative_source).name+'。'
        else:
            text = '勾选用于制作，点击行查看原片；Agent 会查看全部导入照片。'
            if self.representative_source:
                text += ' 试样原片：'+Path(self.representative_source).name+'。'
        self.selection_summary.setText(text)
        self.selection_summary.setToolTip(text)

    def refresh_queue(self):
        selected = [self.listing.item(row, 0).data(Qt.UserRole) for row in sorted(
            {item.row() for item in self.listing.selectedItems()}) if self.listing.item(row, 0)]
        current = self.listing.item(self.listing.currentRow(), 0)
        current = current.data(Qt.UserRole) if current else None
        self.listing.blockSignals(True)
        self.rows.clear()
        keys = {self.source_key(source) for source in self.sources}
        self.excluded_sources.intersection_update(keys)
        included = self.included_sources()
        if self.representative_source not in included:
            self.representative_source = included[0] if included else None
        self.results = {key: result for key, result in self.results.items() if key in keys}
        self.stages = {key: stage for key, stage in self.stages.items() if key in keys}
        self.photo_errors = {key: error for key, error in self.photo_errors.items() if key in keys}
        self.listing.clearSelection()
        self.listing.setRowCount(len(self.sources))
        selection_numbers = ({self.source_key(photo['source']):index for index,photo in enumerate(self.photo_selection['photos'],1)}
            if self.photo_selection is not None else None)
        for row, source in enumerate(self.sources):
            key = self.source_key(source)
            self.rows[key] = row
            number = str(selection_numbers[key]) if selection_numbers is not None and key in selection_numbers else (
                '新' if selection_numbers is not None else str(row+1))
            header = QTableWidgetItem(number)
            header.setToolTip('上次 Agent 选片总览中的编号' if selection_numbers is not None and key in selection_numbers else '尚未经过 Agent 选片')
            self.listing.setVerticalHeaderItem(row,header)
            name = QTableWidgetItem(('★ ' if source==self.representative_source else '')+Path(source).name)
            name.setData(Qt.UserRole, source)
            name.setToolTip(source)
            name.setCheckState(Qt.Unchecked if key in self.excluded_sources else Qt.Checked)
            self.listing.setItem(row, 0, name)
            result = self.results.get(key)
            stage = ('完成' if result['status'] == 'complete' else '失败') if result else self.stages.get(key,
                '未勾选' if key in self.excluded_sources else '等待制作')
            cell = QTableWidgetItem(stage)
            if result and result.get('error'):
                cell.setToolTip(str(result['error']))
            self.listing.setItem(row, 1, cell)
        self.queue_pages.setCurrentIndex(1 if self.sources else 0)
        self.count.setText(f'已勾选 {len(included)} / 导入 {len(self.sources)} 张' if self.sources else '尚未添加照片')
        if self.worker is None:
            self.progress_bar.setRange(0, max(1, len(included)))
            self.progress_bar.setValue(sum(self.source_key(source) in self.results for source in included))
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
        self.sync_creative_sources()
        self.update_actions()
        self.schedule_workspace_save()

    def update_actions(self):
        if self.workspace_restoring:
            return
        running = self.work_in_progress()
        has_project = self.project is not None
        for button in self.nav_buttons:
            button.setEnabled(has_project)
        self.project_title.setEnabled(not running)
        self.project_title.setVisible(has_project)
        self.project_title.setText(self.project.data['name'] if has_project else '新建项目')
        self.project_title.setToolTip(str(self.project.path) if has_project else '开始一组新的创作')
        self.setWindowTitle('CosLuma '+VERSION+(' · '+self.project.data['name'] if has_project else ''))
        self.project_browser.set_busy(running)
        choosing = self.import_worker is not None or self.selection_worker is not None
        self.progress_bar.setVisible(self.worker is not None or choosing)
        if choosing:
            self.progress_bar.setRange(0,0)
        for widget in self.inputs:
            widget.setEnabled(not running)
        for row in range(self.listing.rowCount()):
            item = self.listing.item(row,0)
            flags = item.flags()
            item.setFlags(flags & ~Qt.ItemIsUserCheckable if running else flags | Qt.ItemIsUserCheckable)
        self.selection_button.setEnabled(not running and bool(self.sources))
        row = self.listing.currentRow()
        source = self.sources[row] if 0<=row<len(self.sources) else None
        self.representative_button.setEnabled(not running and source is not None and
            self.source_key(source) not in self.excluded_sources and source!=self.representative_source)
        self.update_selection_summary()
        if self.creative_workspace is not None:
            self.creative_workspace.set_production_busy(self.worker is not None or choosing or
                (self.delivery_view is not None and self.delivery_view.is_busy()) or
                (self.beauty_editor is not None and self.beauty_editor.is_busy()))
        if self.delivery_view is not None:
            self.delivery_view.set_production_busy(self.worker is not None or choosing or
                (self.creative_workspace is not None and self.creative_workspace.is_busy()) or
                (self.beauty_editor is not None and self.beauty_editor.is_busy()))
        included = self.included_sources()
        has_photos = bool(included)
        has_character = bool(self.character.text().strip())
        has_config = Path(self.config.text()).expanduser().is_file()
        context = self.current_style_context()
        has_previous = bool(self.results or any(stage == '未完成' for stage in self.stages.values()))
        pending = sum(self.source_key(source) not in self.results and self.source_key(source) not in self.stages
            for source in included)
        chosen = self.selected_style if self.style_context == context else None
        ready = self.style_is_current()
        self.start.setEnabled(not running and pending > 0 and has_character and ready)
        self.start.setText(f'按选定风格制作新增 {pending} 张' if has_previous and pending else
            '勾选照片已处理' if has_photos and has_previous and not pending else f'按选定风格制作 {pending} 张')
        self.remaining_button.setVisible(has_previous)
        self.remaining_button.setEnabled(not running and has_photos and has_character and ready)
        self.style_button.setEnabled((has_photos or chosen is not None) and has_character)
        self.style_button.setText('继续创作' if self.style_discussion_context == context or chosen else '与 Agent 讨论风格')
        self.view_chosen_button.setEnabled(chosen is not None)
        self.workspace_context.setText((context[0] + ' · ' if has_character else '') + f'已选 {len(included)}/{len(self.sources)} 张')
        if self.worker is not None:
            next_step = '正在按定稿制作，可切页查看原片、样片和已有成片。'
        elif self.import_worker is not None:
            next_step = '正在整理导入的照片和子文件夹，可切页查看已有内容。'
        elif self.selection_worker is not None:
            next_step = 'Agent 正在查看全部导入照片，完成后可继续调整勾选。'
        elif self.beauty_editor is not None and self.beauty_editor.is_busy():
            next_step = '正在处理可选人脸美化，可切页查看已有照片。'
        elif self.delivery_view is not None and self.delivery_view.is_busy():
            next_step = '正在复制所选版本到交付文件夹，可切页查看已有照片。'
        elif not has_photos:
            next_step = '先勾选要制作的照片，或让 Agent 从已导入照片中选片。' if self.sources else '先在照片页批量导入原片。'
        elif not has_character:
            next_step = '先填写角色，补充要求可以留空。'
        elif running:
            next_step = 'Agent 或样片正在处理中，可切页浏览，完成后再决定下一步。'
        elif not has_config:
            next_step = '先连接模型服务：打开“连接设置”或导入已有配置，保存后即可开始讨论。'
        elif self.creative_context == context and self.creative_draft:
            next_step = '创作页还有未发送的输入；发送或清空后，再按当前讨论制作。'
        elif chosen is not None and not ready:
            next_step = '创作要求或修发、光照设置已改变；原定稿仍可查看，按当前设置主动试样后再定稿。'
        elif chosen is None:
            next_step = '到创作页讨论并比较三个真实样片，选定一张后再批量制作。'
        elif not pending:
            next_step = '勾选照片已处理。可添加新照片，或明确选择“勾选照片另做一批”。'
        else:
            next_step = f'已定稿。勾选照片中有 {pending} 张尚未制作，点击制作后才会开始。'
        self.readiness.setText(next_step)
        self.style_hint.setText(next_step)
        self.start.setToolTip(next_step)
        if chosen is not None:
            self.chosen_title.setText('已定稿：' + chosen['title'])
            self.chosen_source.setText('代表原片 ' + Path(chosen['sample_source']).name)
            self.chosen_summary.setText(chosen['summary'])
            self.chosen_summary.setToolTip(chosen['summary'])
            path = chosen['result']['files']['jpeg']
            if path != self.chosen_preview_key:
                pixels = self.photo_preview.read_preview(path)
                if pixels is not None:
                    self.chosen_thumbnail.setPixmap(pixels.scaled(112, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                else:
                    self.chosen_thumbnail.setText('样片无法读取')
                self.chosen_preview_key = path
        else:
            self.chosen_title.setText('先选择真实样片')
            self.chosen_source.setText('定稿只保存方向，制作由你另行开始。')
            self.chosen_summary.clear()
            self.chosen_thumbnail.setText('尚未定稿')
            self.chosen_preview_key = None
        self.results_next.setText('准备制作这一组照片' if ready else '从一张真实样片开始')
        self.results_note.setText(next_step)
        self.last_batch_button.setText('查看已有批次')
        self.last_batch_button.setEnabled(True)
        self.last_batch_action.setEnabled(bool(self.last_batch_directory))
        view = self.result_views.currentWidget()
        archived = self.archive_view is not None and view is self.archive_view
        beautifying = view is self.beauty_editor and self.beauty_editor is not None
        delivering = self.delivery_view is not None and view is self.delivery_view
        self.return_current.setVisible(archived or delivering)
        self.decision_bar.setVisible(not (archived or beautifying or delivering))
        self.readiness.setVisible(not (archived or beautifying or delivering))
        self.delivery_button.setEnabled(has_project)
        self.delivery_button.setVisible(not delivering)
        self.open_folder_button.setVisible(not delivering)
        self.open_folder_button.setEnabled(isinstance(view, BatchResults))
        self.results_caption.setText('项目作品 · 选择版本与交付' if delivering else '成片后的可选人脸美化' if beautifying else
            '已有批次：' + self.archive_view.directory.name if archived else '本次制作')

    def dragEnterEvent(self, event):
        if not self.work_in_progress() and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        try:
            self.add_sources([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
            event.acceptProposedAction()
        except OSError as error:
            self.status.setText('无法添加照片：' + str(error))

    def show_selected(self):
        self.schedule_workspace_save()
        row = self.listing.currentRow()
        if row < 0 or row >= len(self.sources):
            self.preview_key = None
            self.current_output = None
            self.current_png = None
            self.photo_preview.set_photo()
            self.output_info.clear()
            self.selection_reason.clear()
            self.selection_reason.hide()
            self.representative_button.setEnabled(False)
            self.open_source_button.setEnabled(False)
            self.open_work_button.setEnabled(False)
            self.open_photo_button.setEnabled(False)
            self.open_png_button.setEnabled(False)
            return
        source = self.sources[row]
        key = self.source_key(source)
        self.representative_button.setEnabled(not self.work_in_progress() and key not in self.excluded_sources
            and source!=self.representative_source)
        self.representative_button.setText('当前试样原片' if source==self.representative_source else '设为试样原片')
        advice = (next((photo for photo in self.photo_selection['photos'] if self.source_key(photo['source'])==key),None)
            if self.photo_selection is not None else None)
        self.selection_reason.setText(('Agent 建议保留：' if advice['selected'] else 'Agent 本轮未选：')+advice['reason'] if advice else '')
        self.selection_reason.setVisible(advice is not None)
        result = self.results.get(self.source_key(source))
        self.current_output = None
        self.current_png = None
        self.open_photo_button.setEnabled(False)
        self.open_png_button.setEnabled(False)
        self.open_source_button.setEnabled(Path(source).is_file())
        if result and result['status'] == 'complete':
            path = result.get('exports', {}).get('JPEG', result['files']['jpeg'])
            self.current_output = path
            self.current_png = result.get('exports', {}).get('PNG', result['files']['final'])
            width, height = result['output_dimensions']
            self.output_info.setText(f'{Path(source).name} · 成片 {width} × {height}\n16 位 PNG 和 JPEG 已保存')
            if result.get('output_adjustment'):
                self.output_info.setText(self.output_info.text() + '\n已按模型实际覆盖范围输出。')
            self.open_photo_button.setEnabled(Path(path).is_file())
            self.open_png_button.setEnabled(Path(self.current_png).is_file())
            if not self.open_photo_button.isEnabled():
                self.output_info.setText(self.output_info.text() + '\nJPEG 文件已移动或删除，请在照片文件夹中查找。')
        else:
            path = source
            error = result.get('error') if result else self.photo_errors.get(self.source_key(source))
            self.output_info.setText(f'{Path(source).name} · 制作失败\n{error}' if error else
                f'{Path(source).name} · {self.stages.get(self.source_key(source), "等待制作")}')
        self.output_info.setText(self.output_info.text()+'\n'+('本次已勾选' if key not in self.excluded_sources else '本次未勾选')+
            (' · 当前试样原片' if source==self.representative_source else ''))
        work_directory,folder_label = self.source_work_directory(source)
        self.open_work_button.setText(folder_label)
        self.open_work_button.setEnabled(bool(work_directory))
        preview_key = (source, self.current_output)
        if preview_key != self.preview_key:
            self.photo_preview.set_photo(source, self.current_output)
            self.preview_key = preview_key

    def produce(self, _checked=False, *, only_pending=False):
        if self.work_in_progress():
            return
        try:
            included = self.included_sources()
            if not included or not self.character.text().strip():
                raise ValueError('请勾选要制作的照片并填写角色')
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
            sources = [source for source in included if not only_pending or (
                self.source_key(source) not in self.results and self.source_key(source) not in self.stages)]
            if not sources:
                return
            if not only_pending:
                for source in sources:
                    key = self.source_key(source)
                    self.results.pop(key,None)
                    self.stages.pop(key,None)
                    self.photo_errors.pop(key,None)
            self.batch_error = ''
            self.batch_directory = None
            if self.current_batch_view is not None:
                self.result_views.removeWidget(self.current_batch_view)
                self.current_batch_view.deleteLater()
                self.current_batch_view = None
            self.result_views.setCurrentWidget(self.results_empty)
            for source in sources:
                self.stages[self.source_key(source)] = '等待制作'
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
            self.navigate(2)
            self.status.setText('沿用选中的摄影方向，开始制作本批次')
            self.save_workspace()
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
            self.schedule_workspace_save()

    @Slot(str)
    def batch_started(self, directory):
        self.batch_directory = directory
        self.last_batch_directory = directory
        if directory not in self.project_batches:
            self.project_batches.append(directory)
        self.refresh_batch_menu()
        self.open_folder_button.setEnabled(True)
        self.current_batch_view = self.make_results_view(directory)
        if self.result_views.currentWidget() is self.results_empty:
            self.result_views.setCurrentWidget(self.current_batch_view)
        self.show_selected()
        self.update_actions()
        self.save_workspace()

    @Slot(dict)
    def show_result(self, result):
        key = self.source_key(result['source'])
        self.results[key] = result
        self.stages.pop(key,None)
        self.photo_errors.pop(key,None)
        self.batch_directory = result.get('batch_directory', self.batch_directory)
        self.listing.item(self.rows[key], 1).setText('完成' if result['status'] == 'complete' else '失败')
        self.listing.item(self.rows[key], 1).setToolTip('' if result['status'] == 'complete' else str(result['error']))
        self.active_processed += 1
        self.progress_bar.setValue(self.active_processed)
        self.show_selected()
        if self.current_batch_view is not None:
            self.current_batch_view.update_result(result)
        self.save_workspace()

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
        if self.current_batch_view is not None:
            self.current_batch_view.refresh()
        if self.delivery_view is not None:
            self.delivery_view.refresh()
        self.update_actions()
        self.show_selected()
        self.save_workspace()
        worker.deleteLater()
        self.finish_background_work()

    def open_selected_output(self):
        if self.current_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.current_output))

    def open_selected_png(self):
        if self.current_png:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.current_png))

    def open_source(self):
        row = self.listing.currentRow()
        if 0 <= row < len(self.sources):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.sources[row]))

    def open_work_directory(self):
        row = self.listing.currentRow()
        if 0 <= row < len(self.sources):
            directory,_label = self.source_work_directory(self.sources[row])
            if directory:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def source_work_directory(self, source):
        result = self.results.get(self.source_key(source),{})
        if result.get('directory'):
            return result['directory'],'打开此照片文件夹'
        directory = result.get('batch_directory') or self.batch_directory
        if directory:
            return directory,'打开本批次文件夹'
        return self.photo_selection_directory,'打开选片记录'

    def open_results(self):
        view = self.result_views.currentWidget()
        if isinstance(view, BatchResults):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(view.directory)))

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
        if not self.persist_project():
            event.ignore()
            return
        self.remember_preferences()
        if self.work_in_progress():
            if self.tray is not None:
                self.close_after_finish = True
                QApplication.instance().setQuitOnLastWindowClosed(False)
                self.tray.show()
                self.hide()
                self.tray.showMessage('CosLuma 继续处理', '导入、选片、制作、美化与交付导出会继续处理并保存。点击托盘返回；保持托盘时处理结束后退出。')
            else:
                self.status.setText('仍在处理中，可以最小化等待；完成后可关闭。')
                self.showMinimized()
            event.ignore()
        else:
            if self.tray is not None:
                self.tray.hide()
            super().closeEvent(event)


    def navigate(self, index):
        if index in (0, 1, 2) and self.project is None:
            index = self.projects_index
        if self.pages.currentIndex() == 1 and self.creative_workspace is not None:
            if not self.creative_workspace.flush_draft():
                return
        if index == 1:
            try:
                self.ensure_creative()
            except (OSError, ValueError, KeyError) as error:
                self.status.setText('打开创作页失败：' + str(error))
                return
        elif index == 0:
            self.refresh_queue()
        elif index == 2 and self.current_batch_view is None and self.batch_directory:
            try:
                self.current_batch_view = self.make_results_view(self.batch_directory)
                if self.result_views.currentWidget() is self.results_empty:
                    self.result_views.setCurrentWidget(self.current_batch_view)
            except (OSError, ValueError, KeyError) as error:
                self.status.setText('读取本次结果失败：' + str(error))
        self.pages.setCurrentIndex(index)
        if index in (0, 1, 2):
            self.restored_page = index
        self.update_setup_hint()
        self.projects_button.setChecked(index == self.projects_index)
        for number, button in enumerate(self.nav_buttons):
            button.setChecked(number == index)
        self.update_actions()
        self.schedule_workspace_save()


    def ensure_creative(self):
        context = self.current_style_context()
        chosen = self.selected_style if self.style_context == context else None
        selected_sources = self.included_sources()
        missing_photos = not selected_sources and chosen is None
        missing_character = not context[0]
        missing_config = not Path(self.config.text()).expanduser().is_file()
        if missing_photos or missing_character or missing_config:
            self.creative_empty.show()
            self.creative_next.setText(('请先在照片页勾选要制作的照片，或让 Agent 选片。' if self.sources else '先批量导入这组照片。') if missing_photos else
                '请填写角色，补充要求可以留空。' if missing_character else
                '照片与角色已准备好。连接模型服务或导入已有配置后，即可开始讨论。')
            self.creative_prepare.setText(('返回照片选片' if self.sources else '批量导入照片') if missing_photos else '填写角色' if missing_character else '连接设置')
            self.creative_import.setVisible(not missing_photos and not missing_character and missing_config)
            if self.creative_workspace is not None:
                self.creative_workspace.hide()
            return
        config = self.desktop_config()
        workspace = self.creative_workspace
        if workspace is None or (workspace.character, workspace.brief) != context:
            if workspace is not None:
                if not workspace.flush_draft():
                    return
                self.creative_layout.removeWidget(workspace)
                workspace.deleteLater()
            sources = selected_sources if selected_sources else [chosen['sample_source']]
            source = self.representative_source if self.representative_source in sources else sources[0]
            previous = self.style_samples_directory if self.style_samples_context == context else None
            discussion = self.style_discussion_directory if self.style_discussion_context == context else None
            workspace = CreativeWorkspace(sources, source, context[0], context[1],
                self.root.text(), config, self.creative_page, previous=previous, discussion=discussion)
            self.creative_workspace = workspace
            if chosen is not None and workspace.chosen_file != chosen['selection_file']:
                workspace.show_selected_style(chosen)
            workspace.discussion_saved.connect(self.remember_discussion)
            workspace.style_selected.connect(self.select_style)
            workspace.state_changed.connect(self.creative_state_changed)
            workspace.back_requested.connect(lambda: self.navigate(0))
            self.creative_layout.addWidget(workspace)
            self.sync_creative_sources()
        else:
            self.sync_creative_sources()
            workspace.config, workspace.root = config, self.root.text()
        self.creative_empty.hide()
        workspace.show()
        self.creative_state_changed()


    def remember_discussion(self, directory):
        workspace = self.creative_workspace
        context = (workspace.character, workspace.brief)
        changed = (self.style_discussion_directory, self.style_discussion_context) != (directory, context)
        self.style_discussion_directory, self.style_discussion_context = directory, context
        if changed:
            self.save_workspace()


    def creative_state_changed(self):
        workspace = self.creative_workspace
        self.creative_context = (workspace.character, workspace.brief)
        self.creative_direction = workspace.creative_direction
        self.creative_edits = workspace.current_edits()
        self.creative_draft = bool(workspace.chat_input.toPlainText().strip() or workspace.pending_message)
        source = workspace.sample_source.currentData()
        if source in self.included_sources():
            self.representative_source = source
        if workspace.directory is not None and workspace.options:
            self.style_samples_directory = workspace.directory
            self.style_samples_context = self.creative_context
        self.schedule_workspace_save()
        self.update_actions()
        self.finish_background_work()


    def select_style(self, option):
        self.selected_style = option
        self.style_selection_file = option['selection_file']
        workspace = self.creative_workspace
        self.style_context = (workspace.character, workspace.brief)
        self.save_workspace()
        self.status.setText('已定稿“' + option['title'] + '”。确认这张真实样片后，单独点击批量制作。')
        self.show_current_batch()


    def view_chosen(self):
        self.navigate(1)
        if self.pages.currentIndex() == 1 and self.creative_workspace is not None:
            self.creative_workspace.show_selected_style(self.selected_style)


    def make_results_view(self, directory):
        view = BatchResults(directory, self.result_views)
        view.back_requested.connect(lambda: self.navigate(0))
        view.beauty_requested.connect(self.open_beauty)
        self.result_views.addWidget(view)
        return view

    def open_beauty(self, context):
        same_editor = self.beauty_editor is not None and self.beauty_editor.context['directory']==context['directory']
        if self.work_in_progress() and not (same_editor and self.beauty_editor.is_busy()):
            self.status.setText('当前项目仍在处理中，结束后可打开另一张照片的人脸美化。')
            return
        if same_editor:
            self.beauty_return_delivery = context.get('return_to_delivery',False)
            self.result_views.setCurrentWidget(self.beauty_editor)
            self.navigate(2)
            return
        try:
            config = load_settings(self.config.text())
            if 'beauty' not in config:
                dialog = BeautyConnection(self.config.text(),self)
                if dialog.exec()!=QDialog.Accepted:
                    return
                config = load_settings(self.config.text())
            editor = BeautyEditor(context,config['beauty'],self.result_views)
            editor.back_requested.connect(self.return_from_beauty)
            editor.saved.connect(self.beauty_saved)
            editor.busy_changed.connect(self.beauty_busy_changed)
            editor.connection_requested.connect(self.configure_beauty)
            if self.beauty_editor is not None:
                self.result_views.removeWidget(self.beauty_editor)
                self.beauty_editor.deleteLater()
            self.beauty_editor = editor
            self.beauty_return_delivery = context.get('return_to_delivery',False)
            self.result_views.addWidget(editor)
            self.result_views.setCurrentWidget(editor)
            self.navigate(2)
            self.status.setText('基础成片已完成。可分别调整美肤与瘦脸，满意后另存美化版。')
        except (OSError,ValueError,KeyError) as error:
            self.status.setText('打开人脸美化失败：'+str(error))

    def configure_beauty(self):
        try:
            dialog = BeautyConnection(self.config.text(),self)
            if dialog.exec()==QDialog.Accepted:
                self.beauty_editor.config = load_settings(self.config.text())['beauty']
                self.beauty_editor.message.setText('美化连接已保存，下次主动分析或生成美肤基础时使用。')
        except (OSError,ValueError,KeyError) as error:
            self.beauty_editor.message.setText('保存美化连接失败：'+str(error))

    def beauty_busy_changed(self):
        self.update_actions()
        self.finish_background_work()

    def return_from_beauty(self):
        if self.beauty_return_delivery:
            self.show_delivery()
            return
        context = self.beauty_editor.context
        self.show_saved_batch(context['batch_directory'])

    def beauty_saved(self, result):
        self.return_from_beauty()
        self.status.setText('已另存美化版：'+result['files']['final'])


    def show_current_batch(self):
        self.result_views.setCurrentWidget(self.current_batch_view or self.results_empty)
        self.navigate(2)

    def ensure_delivery(self):
        if self.delivery_view is None:
            self.delivery_view = DeliveryView(self.project,self.result_views)
            self.delivery_view.beauty_requested.connect(self.open_beauty)
            self.delivery_view.revision_requested.connect(self.start_photo_revision)
            self.delivery_view.batch_created.connect(self.remember_revision_batch)
            self.delivery_view.state_changed.connect(self.update_actions)
            self.delivery_view.busy_changed.connect(self.delivery_busy_changed)
            self.result_views.addWidget(self.delivery_view)

    def show_delivery(self):
        if self.project is None:
            self.show_projects()
            return
        if self.delivery_view is None:
            self.ensure_delivery()
        else:
            self.delivery_view.refresh()
        self.result_views.setCurrentWidget(self.delivery_view)
        self.navigate(2)

    def delivery_busy_changed(self):
        self.update_actions()
        self.finish_background_work()

    def start_photo_revision(self, context):
        if self.work_in_progress():
            return
        try:
            config = load_settings(self.config.text())
            self.delivery_view.start_revision(context,self.root.text(),config)
        except (OSError,ValueError,KeyError) as error:
            self.delivery_view.message.setPlainText('无法开始改稿：'+str(error))

    def remember_revision_batch(self, directory):
        if directory not in self.project_batches:
            self.project_batches.append(directory)
        self.last_batch_directory = directory
        self.refresh_batch_menu()
        self.save_workspace()


    def work_in_progress(self):
        return (self.worker is not None or self.import_worker is not None or self.selection_worker is not None or
            (self.delivery_view is not None and self.delivery_view.is_busy()) or
            (self.beauty_editor is not None and self.beauty_editor.is_busy()) or
            (self.creative_workspace is not None and self.creative_workspace.is_busy()))


    def finish_background_work(self):
        if self.work_in_progress():
            return
        if self.tray is not None:
            self.tray.hide()
        if self.close_after_finish:
            self.close()
            QApplication.instance().quit()

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
    window = Workbench(args.root, args.config)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
