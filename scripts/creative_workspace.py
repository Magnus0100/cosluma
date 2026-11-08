"""A persistent creative workspace for discussion, real samples and user direction."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import uuid

from PySide6.QtCore import QThread, Signal, Qt, QUrl, QTimer, QSize
from PySide6.QtGui import QDesktopServices, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QSizePolicy, QSplitter, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from image_io import save_json
from model_clients import _representatives, discuss_style
from photo_pipeline import load_style, make_style_samples
from creative_edits import normalise_edits, edits_match, edits_label, edits_description
from studio_widgets import ui_label, StatusText, SampleCard, ConversationView, PhotoPreview, SampleComparison


class StyleWorker(QThread):
    progress = Signal(str)
    plans_ready = Signal(list, str)
    sample_stage = Signal(int, str)
    sample_ready = Signal(dict)
    directory_created = Signal(str)
    failed = Signal(str)

    def __init__(self, sources, sample_source, character, brief, root, config, *, creative_direction='', discussion=None, edits=None, base_style=None):
        super().__init__()
        self.sources, self.sample_source = sources, sample_source
        self.character, self.brief = character, brief
        self.root, self.config = root, config
        self.creative_direction = creative_direction
        self.discussion = discussion
        self.edits = normalise_edits(edits)
        self.base_style = deepcopy(base_style)

    def run(self):
        try:
            for option in make_style_samples(self.sources,self.sample_source,self.character,self.root,self.config,
                    brief=self.brief,progress=self.progress.emit,planned=self.plans_ready.emit,
                    sample_progress=self.sample_stage.emit,started=self.directory_created.emit,
                    creative_direction=self.creative_direction,discussion=self.discussion,edits=self.edits,
                    base_style=self.base_style):
                self.sample_ready.emit(option)
        except Exception as error:
            self.failed.emit(str(error))


class StyleChatWorker(QThread):
    reply_ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, sources, character, brief, config, directory, messages, direction, context):
        super().__init__()
        self.arguments = (sources,character,brief,config['prompt_library'],config['llm'],Path(directory),messages)
        self.direction, self.context = direction, context

    def run(self):
        try:
            result = discuss_style(*self.arguments,direction=self.direction,context=self.context,edits=self.context.get('edits'))
            self.reply_ready.emit(result)
        except Exception as error:
            self.failed.emit(getattr(error,'provider_message','') or str(error))


class CreativeWorkspace(QWidget):
    """Discuss photographs, compare real samples, then explicitly settle the batch direction."""
    discussion_saved = Signal(str)
    style_selected = Signal(dict)
    state_changed = Signal()
    back_requested = Signal()

    def __init__(self, sources, sample_source, character, brief, root, config, parent=None, *, previous=None, discussion=None):
        super().__init__(parent)
        self.sources, self.character, self.brief = sources, character, brief
        self.config, self.root = config, root
        self.production_busy = False
        self.worker = None
        self.chat_worker = None
        self.messages = []
        self.creative_direction = ''
        self.samples_direction = ''
        self.samples_edits = normalise_edits()
        self.discussion_directory = None
        self.sample_directories = []
        self.round_sources = {}
        self.round_kinds = {}
        self.viewed_count = 3
        self.viewed_directory = None
        self.viewed_source = sample_source
        self.viewed_plans = []
        self.viewing_original = True
        self.active_directory = None
        self.active_plans = []
        self.active_options = {}
        self.active_stages = {}
        self.pending_message = None
        self.frozen_direction = ''
        self.chosen_file = ''
        self.options = {}
        self.cards = {}
        self.current_option = 0
        self.selected_style = None
        self.viewing_selected = False
        self.directory = None
        self.failure_message = ''
        self.comparison_sample = None
        self.comparison_option = None
        self.comparison_original_choice = 'output'
        self.draft_timer = QTimer(self)
        self.draft_timer.setSingleShot(True)
        self.draft_timer.setInterval(400)
        self.draft_timer.timeout.connect(self.save_draft)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(8)
        header = QHBoxLayout()
        self.chosen_label = ui_label('尚未定稿，先讨论或试三个真实样片。')
        self.chosen_label.setTextFormat(Qt.PlainText)
        header.addWidget(self.chosen_label,1)
        header.addWidget(ui_label(character,'badge'))
        layout.addLayout(header)
        self.workspace = QSplitter(Qt.Horizontal)
        self.workspace.setHandleWidth(12)
        self.workspace.setChildrenCollapsible(False)
        sample_panel = QFrame()
        sample_panel.setProperty('role','panel')
        sample_layout = QVBoxLayout(sample_panel)
        sample_layout.setContentsMargins(10,10,10,10)
        sample_layout.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(ui_label('下次试样原片'))
        self.sample_source = QComboBox()
        for source in sources:
            self.sample_source.addItem(Path(source).name, source)
            self.sample_source.setItemData(self.sample_source.count()-1,source,Qt.ToolTipRole)
        self.sample_source.setCurrentIndex(max(0,self.sample_source.findData(sample_source)))
        row.addWidget(self.sample_source,1)
        self.focus_button = QToolButton()
        self.focus_button.setProperty('kind','quiet')
        self.focus_button.setText('专注看图')
        self.focus_button.setCheckable(True)
        self.focus_button.toggled.connect(self.focus_photo)
        row.addWidget(self.focus_button)
        self.generate = QPushButton('探索三个方向')
        self.generate.setProperty('kind','primary')
        self.generate.setMinimumHeight(32)
        self.generate.clicked.connect(self.generate_samples)
        self.change_source = QPushButton('查看原片')
        self.change_source.setProperty('kind','quiet')
        self.change_source.clicked.connect(self.change_sample_source)
        row.addWidget(self.change_source)
        sample_layout.addLayout(row)
        self.canvas = PhotoPreview()
        self.canvas.choice = 'output'
        for view in (self.canvas.preview,self.canvas.before,self.canvas.after):
            view.setMinimumHeight(100)
        self.canvas.set_photo(self.sample_source.currentData())
        self.photo_pages = QStackedWidget()
        self.photo_pages.addWidget(self.canvas)
        self.comparison = SampleComparison()
        self.comparison.inspect_requested.connect(self.inspect)
        self.photo_pages.addWidget(self.comparison)
        sample_layout.addWidget(self.photo_pages,1)
        self.active_caption = ui_label('原片')
        self.active_caption.setWordWrap(True)
        self.active_caption.setMaximumHeight(42)
        sample_layout.addWidget(self.active_caption)
        self.sample_settings = ui_label('')
        self.sample_settings.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        sample_layout.addWidget(self.sample_settings)
        self.sample_actions = QWidget()
        sample_actions = QVBoxLayout(self.sample_actions)
        sample_actions.setContentsMargins(0,0,0,0)
        sample_actions.setSpacing(5)
        round_row = QHBoxLayout()
        self.sample_round = QComboBox()
        self.sample_round.setPlaceholderText('样片轮次 · 生成后可回看')
        self.sample_round.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        self.sample_round.setToolTip('切换仅查看已有样片；再试样使用上方的下次试样原片')
        self.sample_round.currentIndexChanged.connect(self.view_round)
        round_row.addWidget(self.sample_round,1)
        self.compare_button = QToolButton()
        self.compare_button.setText('三个样片并排')
        self.compare_button.setCheckable(True)
        self.compare_button.setEnabled(False)
        self.compare_button.toggled.connect(self.compare_samples)
        round_row.addWidget(self.compare_button)
        sample_actions.addLayout(round_row)
        generation_row = QHBoxLayout()
        generation_row.addWidget(self.generate,1)
        self.revise = QPushButton('沿这张改一张')
        self.revise.setMinimumHeight(32)
        self.revise.clicked.connect(self.revise_sample)
        generation_row.addWidget(self.revise,1)
        sample_actions.addLayout(generation_row)
        self.compare_parent = QCheckBox('与改稿前样片并排')
        self.compare_parent.setToolTip('左侧查看本稿所依据的样片，右侧查看当前稿；原片与定稿保持原样')
        self.compare_parent.toggled.connect(self.toggle_parent_comparison)
        self.compare_parent.hide()
        sample_actions.addWidget(self.compare_parent)
        self.revision_note = ui_label('单独查看一张完整样片后，可沿它改稿。')
        self.revision_note.setWordWrap(True)
        self.revision_note.setMaximumHeight(44)
        self.revision_note.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        sample_actions.addWidget(self.revision_note)
        sample_layout.addWidget(self.sample_actions)
        self.sample_strip = QWidget()
        self.sample_strip.setFixedHeight(132)
        choices = QHBoxLayout(self.sample_strip)
        choices.setContentsMargins(0,0,0,0)
        choices.setSpacing(10)
        for index in range(3):
            panel = SampleCard(index+1)
            panel.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
            panel.title.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
            card = panel.column
            card.setContentsMargins(6,6,6,6)
            card.setSpacing(4)
            card_body = QHBoxLayout()
            card_body.setSpacing(6)
            thumbnail = QToolButton()
            thumbnail.setProperty('role','samplePreview')
            thumbnail.setText('等待样片')
            thumbnail.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            thumbnail.setFixedSize(88,64)
            thumbnail.setIconSize(QSize(70,46))
            thumbnail.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
            thumbnail.setCheckable(True)
            thumbnail.clicked.connect(lambda _checked=False,index=index:self.show_option(index))
            card_body.addWidget(thumbnail)
            description = QVBoxLayout()
            description.setSpacing(4)
            summary = QLabel('分析当前角色与照片后，再确定此方向')
            summary.setWordWrap(True)
            summary.setTextFormat(Qt.PlainText)
            summary.setFixedHeight(34)
            summary.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
            stage = ui_label('尚未分析')
            stage.setWordWrap(True)
            stage.setTextFormat(Qt.PlainText)
            stage.setFixedHeight(16)
            stage.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
            description.addWidget(summary)
            description.addWidget(stage)
            description.addStretch(1)
            card_body.addLayout(description,1)
            card.addLayout(card_body)
            button = QPushButton('专注看图')
            button.setProperty('kind','quiet')
            button.setFixedHeight(24)
            button.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
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
        self.workspace.addWidget(sample_panel)
        self.chat_panel = QFrame()
        self.chat_panel.setProperty('role','panel')
        chat_layout = QVBoxLayout(self.chat_panel)
        chat_layout.setContentsMargins(12,12,12,12)
        chat_layout.setSpacing(8)
        chat_layout.addWidget(ui_label('一起构思','heading'))
        self.discussion_target = ui_label('')
        self.discussion_target.setTextFormat(Qt.PlainText)
        self.discussion_target.setWordWrap(True)
        self.discussion_target.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        chat_layout.addWidget(self.discussion_target)
        self.chat_history = ConversationView()
        chat_layout.addWidget(self.chat_history,1)
        direction_tools = QHBoxLayout()
        direction_tools.addWidget(ui_label('当前创作要求'),1)
        self.edit_direction = QToolButton()
        self.edit_direction.setProperty('kind','quiet')
        self.edit_direction.setText('编辑要求')
        self.edit_direction.setToolTip('直接编辑完整创作要求，不发送给 Agent；已有对话、样片和草稿保留')
        self.edit_direction.clicked.connect(self.edit_creative_direction)
        direction_tools.addWidget(self.edit_direction)
        self.undo_direction = QToolButton()
        self.undo_direction.setProperty('kind','quiet')
        self.undo_direction.setText('撤回更新')
        self.undo_direction.setToolTip('恢复最近一次要求更新之前的完整内容；不删除对话、样片或草稿')
        self.undo_direction.clicked.connect(self.undo_creative_direction)
        direction_tools.addWidget(self.undo_direction)
        chat_layout.addLayout(direction_tools)
        self.discussion_summary = QPlainTextEdit()
        self.discussion_summary.setProperty('role','summary')
        self.discussion_summary.setReadOnly(True)
        self.discussion_summary.setPlaceholderText('讨论形成的创作要求会显示在这里')
        self.discussion_summary.setFixedHeight(58)
        chat_layout.addWidget(self.discussion_summary)
        self.edits_button = QToolButton()
        self.edits_button.setProperty('kind','quiet')
        self.edits_button.setCheckable(True)
        self.edits_button.setText('下次试样设置 · 均未开启')
        self.edits_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.edits_button.setArrowType(Qt.RightArrow)
        chat_layout.addWidget(self.edits_button)
        self.edits_panel = QWidget()
        edits_layout = QVBoxLayout(self.edits_panel)
        edits_layout.setContentsMargins(0,0,0,0)
        edits_layout.setSpacing(6)
        self.hair = QCheckBox('修发与头发生成')
        self.hair.setToolTip('生成修整后的头发，保留原片脸部与头饰；具体范围与飘动方式可填写或在对话中说明')
        edits_layout.addWidget(self.hair)
        self.hair_brief = QLineEdit()
        self.hair_brief.setPlaceholderText('如：补齐右侧缺发，保留刘海；长发向身后扬')
        self.hair_brief.setAccessibleName('修发要求')
        edits_layout.addWidget(self.hair_brief)
        self.relight = QCheckBox('人物重打光')
        self.relight.setToolTip('让面部、身体、服装与背景统一受光，保留脸型和相机细节；不会自动开启人脸美化')
        edits_layout.addWidget(self.relight)
        self.lighting_brief = QLineEdit()
        self.lighting_brief.setPlaceholderText('如：左侧暖主光，右侧承接蓝色环境反光')
        self.lighting_brief.setAccessibleName('人物光照要求')
        edits_layout.addWidget(self.lighting_brief)
        self.edits_note = ui_label('两项独立。要求可留空或继续与 Agent 讨论；改变后试样再定稿。')
        self.edits_note.setWordWrap(True)
        edits_layout.addWidget(self.edits_note)
        self.edits_panel.hide()
        chat_layout.addWidget(self.edits_panel)
        self.edits_button.toggled.connect(self.toggle_edits_panel)
        for checkbox in (self.hair,self.relight):
            checkbox.toggled.connect(self.edits_changed)
        for field in (self.hair_brief,self.lighting_brief):
            field.textChanged.connect(self.edits_changed)
        composer = QFrame()
        composer.setProperty('role','composer')
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(12,12,12,10)
        composer_layout.setSpacing(8)
        self.chat_input = QPlainTextEdit()
        self.chat_input.setProperty('role','composer')
        self.chat_input.setPlaceholderText('聊聊你想要的感觉，或指出样片里想改的地方…')
        self.chat_input.setFixedHeight(58)
        self.chat_input.textChanged.connect(self.draft_changed)
        composer_layout.addWidget(self.chat_input)
        send_row = QHBoxLayout()
        self.input_hint = ui_label('Ctrl + Enter 发送')
        self.input_hint.setWordWrap(True)
        send_row.addWidget(self.input_hint,1)
        self.send = QPushButton('发送')
        self.send.clicked.connect(self.send_message)
        send_row.addWidget(self.send)
        composer_layout.addLayout(send_row)
        chat_layout.addWidget(composer)
        self.send_shortcut = QShortcut(QKeySequence('Ctrl+Return'),self.chat_input)
        self.send_shortcut.activated.connect(self.send_message)
        self.keypad_send_shortcut = QShortcut(QKeySequence('Ctrl+Enter'),self.chat_input)
        self.keypad_send_shortcut.activated.connect(self.send_message)
        self.chat_panel.setMinimumWidth(300)
        sample_panel.setMinimumWidth(480)
        self.workspace.addWidget(self.chat_panel)
        self.workspace.setStretchFactor(0,2)
        self.workspace.setStretchFactor(1,1)
        self.workspace.setSizes([760,360])
        layout.addWidget(self.workspace,1)
        footer = QHBoxLayout()
        self.message = StatusText('先讨论和试样，满意后再定稿。')
        self.message.setMaximumHeight(42)
        footer.addWidget(self.message,1)
        self.open_folder = QPushButton('打开样片文件夹')
        self.open_folder.setProperty('kind','quiet')
        self.open_folder.setEnabled(False)
        self.open_folder.clicked.connect(self.open_sample_folder)
        footer.addWidget(self.open_folder)
        self.cancel = QPushButton('返回照片队列')
        self.cancel.clicked.connect(self.go_back)
        footer.addWidget(self.cancel)
        self.finalize = QPushButton('选择完整样片后定稿')
        self.finalize.setProperty('kind','primary')
        self.finalize.setMinimumHeight(32)
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
                self.directory = str(previous)
                self.restore_samples(previous)
                self.creative_direction = self.samples_direction
                self.set_edit_controls(self.samples_edits)
                self.render_conversation()
                self.show_round_note()
            except (OSError,ValueError,KeyError,TypeError) as error:
                self.change_sample_source()
                self.message.setPlainText('已有样片记录无法读取，请重新选择原片后手动生成：'+str(error))
        self.update_creative_actions()
        self.sample_source.currentIndexChanged.connect(self.change_representative)
        self.canvas.mode.currentIndexChanged.connect(self.refresh_discussion_target)
        self.canvas.linked.toggled.connect(lambda:self.draft_timer.start())

    def sync_sources(self, sources):
        if list(sources)==self.sources:
            return
        representative = self.sample_source.currentData()
        self.sources = list(sources)
        choices = list(self.sources)
        selected = representative if representative in choices else choices[0] if choices else None
        self.sample_source.blockSignals(True)
        self.sample_source.clear()
        for source in choices:
            self.sample_source.addItem(Path(source).name,source)
            self.sample_source.setItemData(self.sample_source.count()-1,source,Qt.ToolTipRole)
        self.sample_source.setCurrentIndex(self.sample_source.findData(selected))
        self.sample_source.blockSignals(False)
        if selected!=representative and self.viewing_original:
            self.show_original()
        self.update_creative_actions()
        self.state_changed.emit()

    def set_production_busy(self, busy):
        self.production_busy = busy
        self.update_creative_actions()

    def current_edits(self):
        return normalise_edits({'hair':self.hair.isChecked(),'hair_brief':self.hair_brief.text(),
            'relight':self.relight.isChecked(),'lighting_brief':self.lighting_brief.text()})

    def set_edit_controls(self, value):
        value = normalise_edits(value)
        for control,key in ((self.hair,'hair'),(self.hair_brief,'hair_brief'),
                (self.relight,'relight'),(self.lighting_brief,'lighting_brief')):
            control.blockSignals(True)
            control.setChecked(value[key]) if isinstance(control,QCheckBox) else control.setText(value[key])
            control.blockSignals(False)
        if value['hair'] or value['relight']:
            self.edits_button.setChecked(True)

    def toggle_edits_panel(self, expanded):
        self.edits_panel.setVisible(expanded)
        self.edits_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def edits_changed(self):
        self.update_creative_actions()
        if self.viewed_directory is not None:
            self.show_round_note()
        else:
            self.message.setPlainText('可选设置已更新。可以继续讨论，按当前设置试样后再定稿。')
        self.state_changed.emit()
        self.draft_timer.start()

    def samples_are_current(self):
        return self.samples_direction==self.creative_direction and edits_match(self.samples_edits,self.current_edits())

    def draft_changed(self):
        self.draft_timer.start()
        self.update_creative_actions()
        self.state_changed.emit()

    def flush_draft(self):
        self.draft_timer.stop()
        try:
            if self.discussion_directory is None:
                if (not self.chat_input.toPlainText().strip() and self.pending_message is None
                        and self.current_edits()==normalise_edits()):
                    return True
                self.ensure_discussion()
            self.save_discussion()
        except OSError as error:
            self.message.setPlainText('输入仍保留在工作台中，保存对话失败：'+str(error))
            return False
        return True

    def go_back(self):
        self.back_requested.emit()

    def focus_photo(self, focused):
        self.focus_button.setText('返回讨论' if focused else '专注看图')
        self.chat_panel.setVisible(not focused)
        self.sample_actions.setVisible(not focused)
        self.refresh_sample_view()

    def refresh_sample_view(self):
        if self.viewed_count<2 and self.compare_button.isChecked():
            self.leave_comparison()
        has_samples = self.viewed_directory is not None or bool(self.options)
        focused = self.focus_button.isChecked()
        comparing = self.compare_button.isChecked()
        self.photo_pages.setCurrentIndex(1 if comparing else 0)
        self.sample_strip.setVisible(has_samples and not focused and not comparing)
        self.empty_hint.setVisible(not has_samples and not focused and not self.viewing_selected)
        self.empty_hint.setText('正在查看下次试样原片。可从“样片轮次”回看已有样片。' if self.sample_directories else
            '还没有样片。先聊想法，或让 Agent 根据角色和原片探索三个方向。')
        self.update_sample_settings()
        if comparing:
            self.comparison.set_samples(self.options,self.viewed_plans,
                {index:card[1].text() for index,card in self.cards.items()})
            label = self.sample_round.currentText()+' · 三个真实样片并排'
            self.active_caption.setText(label+'\n单独查看满意的样片后，可将它定稿；也可以继续和 Agent 讨论。')
            self.active_caption.setToolTip(self.active_caption.text())

    def update_sample_settings(self):
        visible = not self.viewing_original and (self.viewing_selected or self.viewed_directory is not None or bool(self.options))
        self.sample_settings.setVisible(visible)
        if not visible:
            return
        option = self.selected_style if self.viewing_selected else self.options.get(self.current_option)
        edits = option['art_direction'].get('edits') if option is not None else self.samples_edits
        label = '本轮样片：' if self.compare_button.isChecked() else '眼前样片：'
        self.sample_settings.setText(label+edits_label(edits))
        self.sample_settings.setToolTip(edits_description(edits))

    def compare_samples(self, comparing):
        self.compare_button.setText('返回单张查看' if comparing else '三个样片并排')
        if comparing:
            self.clear_parent_comparison()
            self.refresh_sample_view()
            self.update_creative_actions()
            self.state_changed.emit()
        else:
            self.show_option(self.current_option)
        self.flush_draft()

    def leave_comparison(self):
        self.compare_button.blockSignals(True)
        self.compare_button.setChecked(False)
        self.compare_button.setText('三个样片并排')
        self.compare_button.blockSignals(False)

    def change_representative(self):
        if self.viewing_original:
            self.show_original()
        else:
            self.message.setPlainText('下次试样将使用所选原片；正在回看的样片与定稿保持原样。')
        self.update_creative_actions()
        self.state_changed.emit()
        try:
            self.ensure_discussion()
            self.save_discussion()
        except OSError as error:
            self.message.setPlainText('保存代表原片选择失败：'+str(error))

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
                'sample_source':self.sample_source.currentData(),
                'messages':self.messages,'direction':self.creative_direction,
                'edits':self.current_edits(),
                'samples_directory':self.directory,'sample_directories':self.sample_directories,
                'viewed_samples_directory':self.viewed_directory,
                'viewing_selected':self.viewing_selected,
                'comparing_samples':self.compare_button.isChecked(),
                'viewed_option':self.current_option,'canvas_mode':self.canvas.choice,
                'linked_comparison':self.comparison.linked.isChecked(),
                'linked_photo_comparison':self.canvas.linked.isChecked(),
                'comparison_sample':self.comparison_sample,
                'comparison_original_mode':self.comparison_original_choice,
                'selected_file':self.chosen_file,'frozen_direction':self.frozen_direction,
                'draft':self.pending_message['content'] if self.pending_message else self.chat_input.toPlainText()})
            self.discussion_saved.emit(self.discussion_directory)

    def save_draft(self):
        return self.flush_draft()

    def restore_discussion(self, directory):
        state = json.loads((Path(directory)/'conversation.json').read_text(encoding='utf-8'))
        if (state['character'],state['brief']) != (self.character,self.brief):
            raise ValueError('此对话的角色或补充要求与当前输入不同')
        self.discussion_directory = str(directory)
        self.messages = state['messages']
        self.creative_direction = state.get('direction','')
        self.set_edit_controls(state.get('edits'))
        self.sample_directories = state.get('sample_directories',[])
        self.directory = state.get('samples_directory')
        for sample_directory in self.sample_directories:
            sample_state = json.loads((Path(sample_directory)/'styles.json').read_text(encoding='utf-8'))
            self.round_sources[sample_directory] = sample_state['sample_source']
            self.round_kinds[sample_directory] = sample_state.get('kind','exploration')
        self.chosen_file = state.get('selected_file','')
        self.frozen_direction = state.get('frozen_direction','')
        if self.chosen_file:
            self.selected_style = load_style(self.chosen_file)
        self.chat_input.setPlainText(state.get('draft',''))
        self.current_option = state.get('viewed_option',0)
        self.canvas.choice = state.get('canvas_mode','output')
        self.comparison.linked.setChecked(state.get('linked_comparison',True))
        self.canvas.linked.setChecked(state.get('linked_photo_comparison',True))
        if state.get('sample_source') in self.sources:
            self.sample_source.setCurrentIndex(self.sample_source.findData(state['sample_source']))
            self.canvas.set_photo(self.sample_source.currentData())
        viewed = state.get('viewed_samples_directory',self.directory)
        if viewed:
            self.restore_samples(viewed)
        else:
            self.refresh_round_choices()
            self.show_original()
        if state.get('viewing_selected') and self.selected_style is not None:
            self.show_selected_style(self.selected_style)
        elif state.get('comparing_samples') and self.viewed_directory is not None and self.viewed_count>1:
            self.compare_button.blockSignals(True)
            self.compare_button.setChecked(True)
            self.compare_button.setText('返回单张查看')
            self.compare_button.blockSignals(False)
            self.refresh_sample_view()
        comparison = state.get('comparison_sample')
        option = self.revision_source()
        if comparison is not None and option is not None and option.get('revision_of')==comparison:
            self.compare_parent.blockSignals(True)
            self.compare_parent.setChecked(True)
            self.compare_parent.blockSignals(False)
            self.toggle_parent_comparison(True,mode=state['canvas_mode'],
                original_mode=state.get('comparison_original_mode','output'))
        self.render_conversation()

    def render_conversation(self):
        self.chat_history.show_messages(self.messages)
        self.discussion_summary.setPlainText(self.creative_direction)

    def direction_change_to_undo(self):
        for index in range(len(self.messages)-1,-1,-1):
            row = self.messages[index]
            if row.get('direction_changed') and not row.get('direction_reverted') and row.get('direction_action')!='undo':
                return index if 'direction_before' in row and row['direction_update']==self.creative_direction else None
        return None

    def edit_creative_direction(self):
        if self.is_busy() or self.production_busy:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('编辑当前创作要求')
        dialog.resize(620,420)
        layout = QVBoxLayout(dialog)
        note = ui_label('直接保存完整要求，也可清空。已有对话、样片与输入草稿保留；修改后按当前要求试样再定稿。')
        note.setWordWrap(True)
        layout.addWidget(note)
        text = QPlainTextEdit(self.creative_direction)
        text.setAccessibleName('完整创作要求')
        layout.addWidget(text,1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText('保存要求')
        buttons.button(QDialogButtonBox.Cancel).setText('取消')
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec()==QDialog.Accepted:
            direction = text.toPlainText().strip()
            content = '我已直接编辑当前创作要求，完整内容如下：\n'+direction if direction else '我已直接清空当前创作要求。'
            self.save_direction_action(direction,content,'edit')

    def undo_creative_direction(self):
        if self.is_busy() or self.production_busy:
            return
        index = self.direction_change_to_undo()
        if index is not None:
            direction = self.messages[index]['direction_before']
            content = ('撤回最近一次创作要求更新，恢复为：\n'+direction if direction else
                '撤回最近一次创作要求更新，恢复为尚无创作要求。')
            self.save_direction_action(direction,content,'undo',reverted=index)

    def save_direction_action(self, direction, content, action, *, reverted=None):
        if direction==self.creative_direction:
            return
        try:
            self.ensure_discussion()
        except OSError as error:
            self.message.setPlainText('无法保存创作要求：'+str(error))
            return
        row = {'role':'user','content':content,'context':deepcopy(self.discussion_context()),
            'direction_before':self.creative_direction,'direction_update':direction,
            'direction_action':action,'direction_changed':True}
        if reverted is not None:
            self.messages[reverted]['direction_reverted'] = True
            row['reverted_message'] = reverted
        self.messages.append(row)
        self.creative_direction = direction
        self.render_conversation()
        self.chat_history.scroll_to_latest()
        self.update_creative_actions()
        self.state_changed.emit()
        if self.flush_draft():
            self.message.setPlainText('已恢复更新前的创作要求，已有对话与样片保留。' if action=='undo' else
                '已直接保存创作要求。按当前要求试样后再定稿；已有对话、样片与草稿保留。')

    def discussion_sample(self, option):
        directory = option['styles_directory']
        round_label = (f'第 {self.sample_directories.index(directory)+1} 轮'
            if directory in self.sample_directories else '已有样片')
        sample = {'id':option['selection_file'],
            'label':f'{round_label} · 方案 {option["index"]+1} · '+option['title'],
            'index':option['index'],'title':option['title'],'summary':option['summary'],
            'direction':option['art_direction']['direction'],'source':option['sample_source'],
            'status':option['status'],'selected':option['selection_file']==self.chosen_file,
            'image':option['result']['files']['jpeg'] if option['status']=='complete' else None,
            'edits':normalise_edits(option['art_direction'].get('edits')),
            'revision_of':option.get('revision_of')}
        if option['status']=='failed':
            sample['error'] = option['error']
        return sample

    def discussion_context(self):
        """Describe the actual canvas and related samples without changing creative decisions."""
        options = [self.options[index] for index in sorted(self.options)]
        if self.selected_style is not None and not any(
                option['selection_file']==self.selected_style['selection_file'] for option in options):
            options.append(self.selected_style)
        if self.comparison_option is not None and not any(
                option['selection_file']==self.comparison_option['selection_file'] for option in options):
            options.append(self.comparison_option)
        samples = [self.discussion_sample(option) for option in options]
        option = (self.selected_style if self.viewing_selected else
            self.options.get(self.current_option) if not self.viewing_original else None)
        comparing = self.compare_button.isChecked()
        viewed_sample = option['selection_file'] if option is not None and not comparing else None
        viewed_samples = [self.options[index]['selection_file'] for index in sorted(self.options)
            if self.options[index]['status']=='complete'] if comparing else ([viewed_sample] if viewed_sample else [])
        viewed_source = option['sample_source'] if option is not None else self.viewed_source if comparing else self.canvas.source
        next_source = self.sample_source.currentData()
        selected_source = self.selected_style['sample_source'] if self.selected_style is not None else None
        required = list(dict.fromkeys(source for source in (viewed_source,next_source,selected_source) if source))
        sources = _representatives(list(dict.fromkeys([*required,*self.sources])),required=required)
        mode = 'samples' if comparing else self.canvas.mode.currentData()
        if comparing:
            round_number = self.sample_directories.index(self.viewed_directory)+1
            label = f'第 {round_number} 轮 · 三个真实样片并排 · 原片 '+Path(viewed_source).name
        elif option is not None:
            mode_label = ({'source':'改稿前样片','output':'成片','compare':'改稿前样片与当前稿并排'} if self.comparison_sample is not None else
                {'source':'原片','output':'成片','compare':'原片与成片并排'})[mode]
            sample = next(sample for sample in samples if sample['id']==viewed_sample)
            label = ('已定稿 · ' if sample['selected'] else '')+sample['label']+' · '+mode_label
            label += ' · 原片 '+Path(viewed_source).name
        else:
            label = '原片 · '+Path(viewed_source).name if viewed_source else '尚未选择原片'
        return {'label':label,'viewed_source':viewed_source,'viewed_sample':viewed_sample,'view_mode':mode,
            'viewed_samples':viewed_samples,
            'comparison_sample':self.comparison_sample,
            'next_sample_source':next_source,'selected_sample':self.chosen_file or None,
            'sources':sources,'samples':samples,'edits':self.current_edits()}

    def refresh_discussion_target(self):
        label = '当前讨论：'+self.discussion_context()['label']
        self.discussion_target.setText(label)
        self.discussion_target.setToolTip(label)

    def update_creative_actions(self):
        busy = self.is_busy() or self.production_busy
        draft = bool(self.chat_input.toPlainText().strip())
        self.send.setEnabled(not busy and draft)
        self.chat_input.setEnabled(not busy)
        self.edit_direction.setEnabled(not busy)
        self.undo_direction.setEnabled(not busy and self.direction_change_to_undo() is not None)
        self.hair.setEnabled(not busy)
        self.relight.setEnabled(not busy)
        self.hair_brief.setEnabled(not busy and self.hair.isChecked())
        self.lighting_brief.setEnabled(not busy and self.relight.isChecked())
        enabled = [label for control,label in ((self.hair,'修发'),(self.relight,'重打光')) if control.isChecked()]
        self.edits_button.setText('下次试样设置 · '+('、'.join(enabled)+'已开启' if enabled else '均未开启'))
        self.edits_button.setToolTip(edits_description(self.current_edits()))
        if self.relight.isChecked():
            self.edits_note.setText('两项独立。重打光保留原片脸部细节，原有锐利硬阴影可能残留；改变设置后试样再定稿。')
        elif self.hair.isChecked():
            self.edits_note.setText('修发会采用生成的头发，未指定修改的发束也可能变化；改变设置后试样再定稿。')
        else:
            self.edits_note.setText('两项默认关闭；展开后分别开启，要求可留空或继续与 Agent 讨论。')
        self.update_sample_settings()
        revision_view = self.revision_source()
        self.compare_parent.setVisible(revision_view is not None and bool(revision_view.get('revision_of')))
        self.generate.setEnabled(not busy and not draft and bool(self.sample_source.currentData()))
        self.input_hint.setText('尚未发送 · Ctrl + Enter 发送' if draft and not busy else 'Ctrl + Enter 发送')
        draft_note = '这段输入尚未发送。请先发给 Agent，或清空输入，再按已经讨论的要求试样或定稿。'
        source = self.sample_source.currentData()
        self.generate.setToolTip(draft_note if draft else '按当前创作要求，用上方所选原片制作新一轮三个真实样片')
        revision = self.revision_source()
        self.revise.setEnabled(not busy and not draft and revision is not None)
        self.revise.setToolTip(draft_note if draft else
            '沿眼前完整样片的方向和原片，按当前讨论制作一张新稿；不会自动定稿')
        note = ('改稿依据：'+revision['title']+' · '+Path(revision['sample_source']).name+
            '\n沿用方向制作一张；整幅重新生成，其他细节也可能变化。' if revision is not None else
            '单独查看一张完整样片后，可沿它改稿。')
        self.revision_note.setText(note)
        self.revision_note.setToolTip(note)
        self.finalize.setToolTip(draft_note if draft else
            '这一轮使用此前的创作要求或可选设置，可回看；按当前设置试样后再定稿' if not self.samples_are_current() else
            '将当前完整样片保存为这组照片的最终方向')
        self.change_source.setEnabled(bool(source))
        self.sample_source.setEnabled(not busy)
        self.sample_round.setEnabled(bool(self.sample_directories))
        self.compare_button.setEnabled(self.viewed_directory is not None and not self.viewing_original
            and not self.viewing_selected and self.viewed_count>1 and bool(self.viewed_plans or self.options))
        for index,card in self.cards.items():
            card[2].setEnabled(index in self.options)
        option = self.options.get(self.current_option,{})
        chosen = bool(self.chosen_file and option.get('selection_file')==self.chosen_file)
        self.finalize.setText('单独查看满意样片后定稿' if self.compare_button.isChecked() else '选择完整样片后定稿' if self.viewing_original else
            '正在查看已定稿样片' if self.viewing_selected else '已定稿此样片' if chosen else
            '将“'+option['title']+'”定稿' if option else '选择完整样片后定稿')
        self.finalize.setEnabled(not busy and not draft and not self.compare_button.isChecked()
            and not self.viewing_original and not self.viewing_selected and self.samples_are_current()
            and option.get('status')=='complete')
        if self.selected_style is not None:
            current = (self.selected_style['art_direction'].get('creative_direction','')==self.creative_direction
                and edits_match(self.selected_style['art_direction'].get('edits'),self.current_edits()))
            if current:
                self.chosen_label.setText('已定稿：'+self.selected_style['title']+
                    ('；有未发送草稿，请先发送或清空。' if draft else '；查看其他方案不会改变定稿。'))
            else:
                self.chosen_label.setText('上次定稿：'+self.selected_style['title']+'；创作要求或可选设置已更新，试样后再定稿。')
        elif self.chosen_file:
            chosen_option = next((item for item in self.options.values() if item.get('selection_file')==self.chosen_file),None)
            self.chosen_label.setText('已定稿：'+chosen_option['title']+'；查看其他方案不会改变定稿。' if chosen_option else '已有定稿，当前正在讨论新的样片。')
        self.refresh_discussion_target()

    def send_message(self):
        text = self.chat_input.toPlainText().strip()
        if self.is_busy() or self.production_busy or not text:
            return
        try:
            self.ensure_discussion()
            context = deepcopy(self.discussion_context())
            self.pending_message = {'role':'user','content':text,'context':context}
            self.save_discussion()
            conversation = [row for row in self.messages if row.get('status')!='failed']
            conversation.append(self.pending_message)
            self.chat_worker = StyleChatWorker(context['sources'],self.character,self.brief,self.config,
                self.discussion_directory,conversation,self.creative_direction,context)
            self.chat_worker.reply_ready.connect(self.receive_reply)
            self.chat_worker.failed.connect(self.chat_failed)
            self.chat_worker.finished.connect(self.chat_finished)
            self.chat_history.show_messages([*self.messages,self.pending_message],
                {'role':'assistant','content':'正在查看原片与样片…'},scroll_to_bottom=True)
            self.chat_input.clear()
            self.message.setPlainText('Agent 正在结合角色、原片与已有样片回复；这一步只讨论。')
            self.update_creative_actions()
            self.state_changed.emit()
            self.chat_worker.start()
        except Exception as error:
            self.chat_worker = None
            self.message.setPlainText('无法开始对话：'+str(error))
            self.update_creative_actions()
            self.state_changed.emit()

    def receive_reply(self, reply):
        update = reply['direction_update']
        changed = update is not None and update!=self.creative_direction
        self.messages.extend([self.pending_message,{'role':'assistant','content':reply['reply'],
            'direction_update':update,'direction_evidence':reply['direction_evidence'],'direction_changed':changed,
            'direction_before':self.creative_direction}])
        self.pending_message = None
        if update is not None:
            self.creative_direction = update
        self.render_conversation()
        self.update_creative_actions()
        self.state_changed.emit()
        try:
            self.save_discussion()
        except OSError as error:
            self.message.setPlainText('回复已收到，保存对话失败：'+str(error))
            return
        self.message.setPlainText('创作要求已更新。可继续讨论；按当前讨论生成新样片后，再定稿用于批量。' if
            changed and self.options and self.samples_direction!=self.creative_direction else
            '可以继续讨论、查看已有样片，或按当前讨论主动生成样片。由你决定何时定稿。')

    def chat_failed(self, error):
        if self.pending_message is not None:
            self.messages.append({**self.pending_message,'status':'failed','error':error})
            self.chat_input.setPlainText(self.pending_message['content'])
        self.pending_message = None
        self.render_conversation()
        self.state_changed.emit()
        self.message.setPlainText('Agent 回复失败：'+error+'。已保留输入与原有样片，可修改后手动发送。')
        try:
            self.save_discussion()
        except OSError as saved_error:
            self.message.setPlainText(self.message.toPlainText()+'\n保存对话失败：'+str(saved_error))

    def chat_finished(self):
        worker,self.chat_worker = self.chat_worker,None
        self.update_creative_actions()
        self.state_changed.emit()
        worker.deleteLater()

    def inspect(self, index):
        self.show_option(index)
        self.focus_button.setChecked(True)

    def show_selected_style(self, option):
        """Preview the host's saved direction without choosing a new candidate."""
        self.clear_parent_comparison()
        self.selected_style = option
        self.chosen_file = option['selection_file']
        self.frozen_direction = option['art_direction']['direction']
        self.leave_comparison()
        for index,candidate in self.options.items():
            if candidate.get('selection_file')==self.chosen_file:
                self.show_option(index)
                return
        self.viewing_original = False
        self.viewing_selected = True
        self.refresh_sample_view()
        for card in self.cards.values():
            card[3].setChecked(False)
        self.canvas.set_photo(option['sample_source'],option['result']['files']['jpeg'])
        self.active_caption.setText('已定稿：'+option['title']+' · 原片 '+Path(option['sample_source']).name+'\n'+option['summary'])
        self.active_caption.setToolTip(self.active_caption.text())
        self.message.setPlainText('正在查看先前定稿的真实样片。本轮三个候选保留，点击下方方案可继续比较。')
        self.open_folder.setEnabled(True)
        self.update_creative_actions()
        self.state_changed.emit()

    def change_sample_source(self):
        self.show_original()
        self.message.setPlainText('正在查看下次试样原片。已有样片可从“样片轮次”回看，定稿保持原样。')
        self.update_creative_actions()
        self.state_changed.emit()
        self.flush_draft()

    def show_original(self):
        self.clear_parent_comparison()
        self.leave_comparison()
        self.options = {}
        self.viewed_plans = []
        self.viewed_directory = None
        self.viewed_source = self.sample_source.currentData()
        self.viewing_original = True
        self.viewing_selected = False
        self.sample_round.blockSignals(True)
        self.sample_round.setCurrentIndex(-1)
        self.sample_round.blockSignals(False)
        self.refresh_sample_view()
        self.open_folder.setEnabled(False)
        self.generate.setText('探索三个方向')
        self.canvas.set_photo(self.viewed_source)
        self.active_caption.setText('下次试样原片'+(' · '+Path(self.viewed_source).name if self.viewed_source else ' · 请在照片页添加原片'))
        self.active_caption.setToolTip(self.active_caption.text())
        self.refresh_discussion_target()

    def open_sample_folder(self):
        directory = (self.selected_style['styles_directory'] if self.viewing_selected else self.viewed_directory)
        if directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(directory))

    def refresh_round_choices(self):
        self.sample_round.blockSignals(True)
        self.sample_round.clear()
        for number,directory in enumerate(self.sample_directories,1):
            source = self.round_sources[directory]
            label = f'第 {number} 轮'+(' · 改稿' if self.round_kinds.get(directory)=='revision' else '')+' · '+Path(source).name
            if self.worker is not None and directory==self.active_directory:
                label += ' · 制作中'
            self.sample_round.addItem(label,directory)
            self.sample_round.setItemData(number-1,directory,Qt.ToolTipRole)
        self.sample_round.setCurrentIndex(self.sample_round.findData(self.viewed_directory))
        self.sample_round.blockSignals(False)
        self.sample_round.setEnabled(bool(self.sample_directories))

    def view_round(self, index):
        if index<0:
            return
        directory = self.sample_round.itemData(index)
        try:
            self.restore_samples(directory)
            self.flush_draft()
        except (OSError,ValueError,KeyError,TypeError) as error:
            self.message.setPlainText('无法读取这一轮样片：'+str(error))

    def reset_cards(self, stage_text, count=3):
        self.viewed_count = count
        for index,(summary,stage,button,thumbnail,panel) in self.cards.items():
            panel.setVisible(index<count)
            panel.setTitle(f'方案 {index+1}')
            summary.setText('分析当前角色与照片后，再确定此方向')
            summary.setToolTip('')
            stage.setText(stage_text)
            stage.setToolTip(stage_text)
            button.setEnabled(False)
            thumbnail.setIcon(QIcon())
            thumbnail.setText('等待样片')
            thumbnail.setChecked(False)

    def display_plans(self, plans):
        self.viewed_plans = plans
        if plans:
            self.viewed_count = len(plans)
            for index,card in self.cards.items():
                card[4].setVisible(index<len(plans))
        for index,plan in enumerate(plans):
            self.cards[index][4].setTitle(f'方案 {index+1} · '+plan['title'])
            self.cards[index][0].setText(plan['summary'])
            self.cards[index][0].setToolTip(plan['summary'])

    def display_sample(self, option):
        summary,stage,button,thumbnail,_panel = self.cards[option['index']]
        summary.setText(option['summary'])
        stage.setText('样片完成' if option['status']=='complete' else '样片失败：'+option['error'])
        stage.setToolTip(stage.text())
        button.setEnabled(True)
        if option['status']=='complete':
            pixels = self.canvas.read_preview(option['result']['files']['jpeg'])
            thumbnail.setIcon(QIcon(pixels) if pixels is not None else QIcon())
            thumbnail.setText('' if pixels is not None else '无法预览，点击查看')
            thumbnail.setToolTip('点击在画布中查看；“专注看图”收起聊天与方案，查看原始像素')
        else:
            thumbnail.setText('查看错误')
            thumbnail.setToolTip(option['error'])

    def restore_samples(self, directory):
        directory = str(Path(directory))
        active = self.worker is not None and directory==self.active_directory
        if active:
            source = self.worker.sample_source
            direction = self.worker.creative_direction
            edits = self.worker.edits
            options = self.active_options
            plans = self.active_plans
            error = self.failure_message
            count = 1 if self.worker.base_style is not None else 3
            kind = 'revision' if self.worker.base_style is not None else 'exploration'
        else:
            path = Path(directory)
            state = json.loads((path/'styles.json').read_text(encoding='utf-8'))
            source = state['sample_source']
            direction = state.get('creative_direction','')
            edits = normalise_edits(state.get('edits'))
            error = state.get('error','')
            status = state['status']
            count = state.get('option_count',3)
            kind = state.get('kind','exploration')
            plans = (json.loads((path/'style-options.json').read_text(encoding='utf-8'))
                if status in ('rendering','complete','unfinished') else [])
            options = {}
            for saved in state['options'].values():
                index = saved['index']
                try:
                    option = load_style(path/f'style-{index+1}.json') if saved['status']=='complete' else saved
                except (OSError, ValueError, KeyError) as saved_error:
                    option = {**saved,'status':'failed','error':str(saved_error)}
                options[index] = option
        if directory not in self.sample_directories:
            self.sample_directories.append(directory)
        self.round_sources[directory] = source
        self.round_kinds[directory] = kind
        self.viewed_directory = directory
        self.viewed_source = source
        self.viewing_original = False
        self.viewing_selected = False
        self.samples_direction = direction
        self.samples_edits = dict(edits)
        self.options = options
        self.reset_cards('准备改稿' if kind=='revision' and not plans else '设计方向中' if active and not plans else '未完成',count)
        self.display_plans(plans)
        if self.current_option>=self.viewed_count:
            self.current_option = 0
        for option in options.values():
            self.display_sample(option)
        if active:
            for index,stage in self.active_stages.items():
                if index not in options:
                    self.cards[index][1].setText(stage)
                    self.cards[index][1].setToolTip(stage)
        self.refresh_round_choices()
        self.open_folder.setEnabled(True)
        self.generate.setText('重新探索三个方向')
        if self.compare_button.isChecked():
            self.refresh_sample_view()
            self.update_creative_actions()
            self.state_changed.emit()
        else:
            self.show_option(self.current_option)
        if error or not self.options:
            self.show_round_note(error)

    def show_round_note(self, error=''):
        if error:
            self.message.setPlainText('本轮样片未完成：'+error)
        elif not self.samples_are_current():
            self.message.setPlainText('这一轮使用此前的创作要求或修发、光照设置，仅供回看；按当前设置试样后再定稿。')
        elif self.worker is not None and self.viewed_directory==self.active_directory:
            self.message.setPlainText('这一轮正在制作，可切换其他轮次回看；新结果不会改变定稿。')
        else:
            self.message.setPlainText('正在回看已有真实样片，可选择完整样片明确定稿；再试样使用上方所选原片与当前讨论。')

    def revision_source(self):
        if self.viewing_original or self.compare_button.isChecked():
            return None
        option = self.selected_style if self.viewing_selected else self.options.get(self.current_option)
        return option if option is not None and option.get('status')=='complete' else None

    def clear_parent_comparison(self):
        self.compare_parent.blockSignals(True)
        self.compare_parent.setChecked(False)
        self.compare_parent.blockSignals(False)
        if self.comparison_sample is not None:
            self.canvas.choice = self.comparison_original_choice
        self.comparison_sample = None
        self.comparison_option = None
        self.canvas.set_labels()

    def toggle_parent_comparison(self, checked, *, mode='compare', original_mode=None):
        option = self.revision_source()
        if option is None:
            return
        if checked:
            try:
                parent = load_style(option['revision_of'])
            except (OSError,ValueError,KeyError) as error:
                self.message.setPlainText('改稿前样片无法读取：'+str(error))
                self.clear_parent_comparison()
                self.update_creative_actions()
                return
            self.comparison_sample = parent['selection_file']
            self.comparison_option = parent
            self.comparison_original_choice = self.canvas.choice if original_mode is None else original_mode
            self.canvas.set_labels('改稿前样片','当前稿')
            self.canvas.choice = mode
            self.canvas.set_photo(parent['result']['files']['jpeg'],option['result']['files']['jpeg'])
        else:
            self.clear_parent_comparison()
            self.canvas.set_photo(option['sample_source'],option['result']['files']['jpeg'])
        self.update_creative_actions()
        self.state_changed.emit()
        self.flush_draft()

    def generate_samples(self):
        self.start_samples()

    def revise_sample(self):
        option = self.revision_source()
        if option is not None:
            self.start_samples(base_style=deepcopy(option))

    def start_samples(self, *, base_style=None):
        if self.is_busy() or self.production_busy:
            return
        if self.chat_input.toPlainText().strip():
            self.message.setPlainText('输入框中还有尚未发送的内容。先发送给 Agent，或清空输入，再按已经讨论的要求生成样片。')
            self.chat_input.setFocus()
            return
        try:
            self.ensure_discussion()
        except OSError as error:
            self.message.setPlainText('无法创建创作文件夹：'+str(error))
            return
        discussion = deepcopy({'messages':[row for row in self.messages if row.get('status')!='failed'],
            'context':self.discussion_context()})
        source = base_style['sample_source'] if base_style is not None else self.sample_source.currentData()
        if not source:
            return
        self.clear_parent_comparison()
        self.leave_comparison()
        self.active_directory = None
        self.active_plans = []
        self.active_options = {}
        self.active_stages = {}
        self.options = self.active_options
        self.viewed_plans = []
        self.viewed_directory = None
        self.viewed_source = source
        self.current_option = 0
        self.viewing_original = False
        self.viewing_selected = False
        self.failure_message = ''
        self.samples_direction = self.creative_direction
        self.samples_edits = self.current_edits()
        self.open_folder.setEnabled(False)
        self.reset_cards('准备改稿' if base_style is not None else '设计方向中',1 if base_style is not None else 3)
        self.sample_round.blockSignals(True)
        self.sample_round.setCurrentIndex(-1)
        self.sample_round.blockSignals(False)
        self.canvas.set_photo(source,base_style['result']['files']['jpeg'] if base_style is not None else None)
        self.active_caption.setText('沿“'+base_style['title']+'”制作一张修改稿 · 原片 '+Path(source).name if base_style is not None else
            '原片 · 正在设计三个方向')
        self.generate.setEnabled(False)
        self.change_source.setEnabled(False)
        self.sample_source.setEnabled(False)
        represented = list(dict.fromkeys([source,*self.sources]))
        self.worker = StyleWorker(represented,source,self.character,
            self.brief,self.root,self.config,creative_direction=self.creative_direction,discussion=discussion,
            edits=self.samples_edits,base_style=base_style)
        self.worker.progress.connect(self.sample_progress)
        self.worker.plans_ready.connect(self.show_plans)
        self.worker.sample_stage.connect(self.show_stage)
        self.worker.sample_ready.connect(self.show_sample)
        self.worker.directory_created.connect(self.directory_ready)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.samples_finished)
        self.message.setPlainText('沿“'+base_style['title']+'”的摄影方向，按当前讨论制作一张新稿。旧样片与定稿保留。' if base_style is not None else
            '正在结合角色“'+self.character+'”、代表原片与补充要求设计三个方向。其余照片尚未提交。')
        self.refresh_sample_view()
        self.update_creative_actions()
        self.state_changed.emit()
        self.worker.start()

    def directory_ready(self, directory):
        self.directory = directory
        self.active_directory = directory
        if directory not in self.sample_directories:
            self.sample_directories.append(directory)
        self.round_sources[directory] = self.worker.sample_source
        self.round_kinds[directory] = 'revision' if self.worker.base_style is not None else 'exploration'
        if self.viewed_directory is None and not self.viewing_original and not self.viewing_selected:
            self.viewed_directory = directory
            self.open_folder.setEnabled(True)
        self.refresh_round_choices()
        self.refresh_sample_view()
        try:
            self.save_discussion()
        except OSError as error:
            self.message.setPlainText('样片已开始制作，保存对话目录失败：'+str(error))
        self.state_changed.emit()

    def show_plans(self, plans, directory):
        self.active_plans = plans
        if self.viewed_directory==directory and not self.viewing_original:
            self.display_plans(plans)
            self.refresh_sample_view()
            self.update_creative_actions()
        self.state_changed.emit()

    def sample_progress(self, message):
        if self.viewed_directory==self.active_directory and not self.viewing_original and not self.viewing_selected:
            self.message.setPlainText(message)

    def show_stage(self, index, stage):
        self.active_stages[index] = stage
        if self.viewed_directory==self.active_directory and not self.viewing_original:
            self.cards[index][1].setText(stage)
            self.cards[index][1].setToolTip(stage)
            if self.compare_button.isChecked():
                self.refresh_sample_view()

    def show_sample(self, option):
        self.active_options[option['index']] = option
        if self.viewed_directory==self.active_directory and not self.viewing_original:
            self.options = self.active_options
            self.display_sample(option)
            if self.compare_button.isChecked():
                self.refresh_sample_view()
            elif self.current_option == option['index'] and not self.viewing_selected:
                self.show_option(option['index'])
                return
            self.update_creative_actions()
        self.state_changed.emit()

    def show_option(self, index):
        self.clear_parent_comparison()
        self.leave_comparison()
        self.viewing_original = False
        self.viewing_selected = False
        self.current_option = index
        for number,card in self.cards.items():
            card[3].setChecked(number==index)
        option = self.options.get(index)
        output = option['result']['files']['jpeg'] if option and option['status']=='complete' else None
        source = option['sample_source'] if option else self.viewed_source
        self.canvas.set_photo(source,output)
        plan = self.viewed_plans[index] if index<len(self.viewed_plans) else None
        description = option or plan
        self.active_caption.setText((f'方案 {index+1} · '+description['title']+' · 原片 '+Path(source).name+'\n'+description['summary'])
            if description else '原片'+(' · '+Path(source).name if source else ''))
        self.active_caption.setToolTip(self.active_caption.text())
        self.refresh_sample_view()
        if option and option['status']!='complete':
            self.message.setPlainText(option['error'])
        elif self.viewed_directory is not None:
            self.show_round_note()
        self.update_creative_actions()
        self.state_changed.emit()

    def failed(self, error):
        self.failure_message = error
        if self.viewed_directory==self.active_directory and not self.viewing_original and not self.viewing_selected:
            self.message.setPlainText('样片未完成：'+error)

    def samples_finished(self):
        worker,self.worker = self.worker,None
        self.generate.setText('重新探索三个方向')
        if self.viewed_directory==self.active_directory and not self.viewing_original:
            complete = sum(option['status']=='complete' for option in self.active_options.values())
            for index in range(1 if worker.base_style is not None else 3):
                if index not in self.active_options:
                    self.cards[index][1].setText('未完成')
                    self.cards[index][1].setToolTip('未完成')
            if not self.viewing_selected:
                if self.active_options:
                    self.message.setPlainText(('修改稿已完成。可勾选“与改稿前样片并排”比较、继续讨论或明确定稿；此前版本保留。' if worker.base_style is not None else
                        f'已完成 {complete} 个真实样片。可比较、继续讨论或明确定稿；再次试样将保留这一轮。') if complete else
                        '本轮样片未完成，点击方案查看真实错误；已有文件保留，由你决定是否再试样。')
                if self.failure_message:
                    self.message.setPlainText('样片未全部完成：'+self.failure_message)
        self.refresh_round_choices()
        self.refresh_sample_view()
        self.update_creative_actions()
        self.state_changed.emit()
        worker.deleteLater()

    def choose(self, index):
        if self.is_busy() or self.production_busy or self.compare_button.isChecked() or self.viewing_original or self.viewing_selected or self.options.get(index,{}).get('status')!='complete':
            return
        if self.chat_input.toPlainText().strip():
            self.message.setPlainText('这段输入尚未发送，未纳入当前创作要求。先发送或清空输入，再决定是否定稿。')
            self.chat_input.setFocus()
            return
        if not self.samples_are_current():
            self.message.setPlainText('创作要求或修发、光照设置已改变，请先按当前设置生成样片，再定稿。已有样片仍可对比查看。')
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
                'edits':self.current_edits(),
                'sample_source':option['sample_source']})
            self.save_discussion()
        except OSError as error:
            self.message.setPlainText('无法保存定稿：'+str(error))
            return
        self.selected_style = option
        self.message.setPlainText('已定稿“'+option['title']+'”。可继续查看样片，批量制作由你另行开始。')
        self.update_creative_actions()
        self.style_selected.emit(option)
        self.state_changed.emit()
