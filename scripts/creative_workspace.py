"""A persistent creative workspace for discussion, real samples and user direction."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import uuid

from PySide6.QtCore import QThread, Signal, Qt, QUrl, QTimer, QSize
from PySide6.QtGui import QDesktopServices, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QSizePolicy, QSplitter, QToolButton, QVBoxLayout, QWidget)

from image_io import save_json
from model_clients import discuss_style
from photo_pipeline import load_style, make_style_samples
from studio_widgets import ui_label, StatusText, SampleCard, ConversationView, PhotoPreview


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
        self.discussion_directory = None
        self.sample_directories = []
        self.round_sources = {}
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
        self.generate = QPushButton('按当前讨论生成三个样片')
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
        sample_layout.addWidget(self.canvas,1)
        self.active_caption = ui_label('原片')
        self.active_caption.setWordWrap(True)
        self.active_caption.setMaximumHeight(42)
        sample_layout.addWidget(self.active_caption)
        self.sample_actions = QWidget()
        sample_actions = QHBoxLayout(self.sample_actions)
        sample_actions.setContentsMargins(0,0,0,0)
        self.sample_round = QComboBox()
        self.sample_round.setPlaceholderText('样片轮次 · 生成后可回看')
        self.sample_round.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        self.sample_round.setToolTip('切换仅查看已有样片；再试样使用上方的下次试样原片')
        self.sample_round.currentIndexChanged.connect(self.view_round)
        sample_actions.addWidget(self.sample_round,1)
        sample_actions.addWidget(self.generate)
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
        chat_layout.addWidget(ui_label('摄影 Agent 会结合原片与样片回应。'))
        self.chat_history = ConversationView()
        chat_layout.addWidget(self.chat_history,1)
        chat_layout.addWidget(ui_label('当前创作要求'))
        self.discussion_summary = QPlainTextEdit()
        self.discussion_summary.setProperty('role','summary')
        self.discussion_summary.setReadOnly(True)
        self.discussion_summary.setPlaceholderText('讨论形成的创作要求会显示在这里')
        self.discussion_summary.setFixedHeight(58)
        chat_layout.addWidget(self.discussion_summary)
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
                self.render_conversation()
                self.show_round_note()
            except (OSError,ValueError,KeyError,TypeError) as error:
                self.change_sample_source()
                self.message.setPlainText('已有样片记录无法读取，请重新选择原片后手动生成：'+str(error))
        self.update_creative_actions()
        self.sample_source.currentIndexChanged.connect(self.change_representative)

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

    def draft_changed(self):
        self.draft_timer.start()
        self.update_creative_actions()
        self.state_changed.emit()

    def flush_draft(self):
        self.draft_timer.stop()
        try:
            if self.discussion_directory is None:
                if not self.chat_input.toPlainText().strip() and self.pending_message is None:
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
        has_samples = self.viewed_directory is not None or bool(self.options)
        focused = self.focus_button.isChecked()
        self.sample_strip.setVisible(has_samples and not focused)
        self.empty_hint.setVisible(not has_samples and not focused and not self.viewing_selected)
        self.empty_hint.setText('正在查看下次试样原片。可从“样片轮次”回看已有样片。' if self.sample_directories else
            '还没有样片。先聊想法，或让 Agent 根据角色和原片探索三个方向。')

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
                'samples_directory':self.directory,'sample_directories':self.sample_directories,
                'viewed_samples_directory':self.viewed_directory,
                'viewing_selected':self.viewing_selected,
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
        self.sample_directories = state.get('sample_directories',[])
        self.directory = state.get('samples_directory')
        for sample_directory in self.sample_directories:
            sample_state = json.loads((Path(sample_directory)/'styles.json').read_text(encoding='utf-8'))
            self.round_sources[sample_directory] = sample_state['sample_source']
        self.chosen_file = state.get('selected_file','')
        self.frozen_direction = state.get('frozen_direction','')
        if self.chosen_file:
            self.selected_style = load_style(self.chosen_file)
        self.chat_input.setPlainText(state.get('draft',''))
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
        self.render_conversation()

    def render_conversation(self):
        self.chat_history.show_messages(self.messages)
        self.discussion_summary.setPlainText(self.creative_direction)

    def update_creative_actions(self):
        busy = self.is_busy() or self.production_busy
        draft = bool(self.chat_input.toPlainText().strip())
        self.send.setEnabled(not busy and draft)
        self.chat_input.setEnabled(not busy)
        self.generate.setEnabled(not busy and not draft and bool(self.sample_source.currentData()))
        self.input_hint.setText('尚未发送 · Ctrl + Enter 发送' if draft and not busy else 'Ctrl + Enter 发送')
        draft_note = '这段输入尚未发送。请先发给 Agent，或清空输入，再按已经讨论的要求试样或定稿。'
        source = self.sample_source.currentData()
        self.generate.setToolTip(draft_note if draft else '按当前创作要求，用上方所选原片制作新一轮三个真实样片')
        self.finalize.setToolTip(draft_note if draft else
            '这一轮使用此前的创作要求，可回看；按当前讨论试样后再定稿' if self.samples_direction!=self.creative_direction else
            '将当前完整样片保存为这组照片的最终方向')
        self.change_source.setEnabled(bool(source))
        self.sample_source.setEnabled(not busy)
        self.sample_round.setEnabled(bool(self.sample_directories))
        for index,card in self.cards.items():
            card[2].setEnabled(index in self.options)
        option = self.options.get(self.current_option,{})
        chosen = bool(self.chosen_file and option.get('selection_file')==self.chosen_file)
        self.finalize.setText('选择完整样片后定稿' if self.viewing_original else
            '正在查看已定稿样片' if self.viewing_selected else '已定稿此样片' if chosen else
            '将“'+option['title']+'”定稿' if option else '选择完整样片后定稿')
        self.finalize.setEnabled(not busy and not draft and not self.viewing_original and not self.viewing_selected and self.samples_direction==self.creative_direction
            and option.get('status')=='complete')
        if self.selected_style is not None:
            current = self.selected_style['art_direction'].get('creative_direction','')==self.creative_direction
            if current:
                self.chosen_label.setText('已定稿：'+self.selected_style['title']+
                    ('；有未发送草稿，请先发送或清空。' if draft else '；查看其他方案不会改变定稿。'))
            else:
                self.chosen_label.setText('上次定稿：'+self.selected_style['title']+'；创作要求已更新，试样后再定稿。')
        elif self.chosen_file:
            chosen_option = next((item for item in self.options.values() if item.get('selection_file')==self.chosen_file),None)
            self.chosen_label.setText('已定稿：'+chosen_option['title']+'；查看其他方案不会改变定稿。' if chosen_option else '已有定稿，当前正在讨论新的样片。')

    def send_message(self):
        text = self.chat_input.toPlainText().strip()
        if self.is_busy() or self.production_busy or not text:
            return
        try:
            self.ensure_discussion()
            self.pending_message = {'role':'user','content':text}
            self.save_discussion()
            conversation = [*self.messages,self.pending_message]
            viewed_source = self.selected_style['sample_source'] if self.viewing_selected else self.viewed_source
            represented = list(dict.fromkeys(source for source in [viewed_source,self.sample_source.currentData(),*self.sources] if source))
            focus = (('原片：'+Path(viewed_source).name if viewed_source else '尚未选择下次试样原片') if self.viewing_original else
                '已定稿样片：'+self.selected_style['title'] if self.viewing_selected else
                self.options.get(self.current_option,{}).get('title'))
            samples = [dict(option,selected=option.get('selection_file')==self.chosen_file) for option in self.options.values()]
            if self.selected_style is not None and not any(option['selected'] for option in samples):
                samples.append(dict(self.selected_style,selected=True))
            self.chat_worker = StyleChatWorker(represented,self.character,self.brief,self.config,
                self.discussion_directory,conversation,self.creative_direction,samples,focus)
            self.chat_worker.reply_ready.connect(self.receive_reply)
            self.chat_worker.failed.connect(self.chat_failed)
            self.chat_worker.finished.connect(self.chat_finished)
            self.chat_history.show_messages(conversation,{'role':'assistant','content':'正在查看原片与样片…'})
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
        self.messages.extend([self.pending_message,{'role':'assistant','content':reply['reply']}])
        self.pending_message = None
        self.creative_direction = reply['direction']
        self.render_conversation()
        self.update_creative_actions()
        self.state_changed.emit()
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
        self.selected_style = option
        self.chosen_file = option['selection_file']
        self.frozen_direction = option['art_direction']['direction']
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
        self.generate.setText('按当前讨论生成三个样片')
        self.canvas.set_photo(self.viewed_source)
        self.active_caption.setText('下次试样原片'+(' · '+Path(self.viewed_source).name if self.viewed_source else ' · 请在照片页添加原片'))
        self.active_caption.setToolTip(self.active_caption.text())

    def open_sample_folder(self):
        directory = (self.selected_style['styles_directory'] if self.viewing_selected else self.viewed_directory)
        if directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(directory))

    def refresh_round_choices(self):
        self.sample_round.blockSignals(True)
        self.sample_round.clear()
        for number,directory in enumerate(self.sample_directories,1):
            source = self.round_sources[directory]
            label = f'第 {number} 轮 · '+Path(source).name
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

    def reset_cards(self, stage_text):
        for index,(summary,stage,button,thumbnail,panel) in self.cards.items():
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
            options = self.active_options
            plans = self.active_plans
            error = self.failure_message
        else:
            path = Path(directory)
            state = json.loads((path/'styles.json').read_text(encoding='utf-8'))
            source = state['sample_source']
            direction = state.get('creative_direction','')
            error = state.get('error','')
            status = state['status']
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
        self.viewed_directory = directory
        self.viewed_source = source
        self.viewing_original = False
        self.viewing_selected = False
        self.samples_direction = direction
        self.options = options
        self.reset_cards('设计方向中' if active and not plans else '未完成')
        self.display_plans(plans)
        for option in options.values():
            self.display_sample(option)
        if active:
            for index,stage in self.active_stages.items():
                if index not in options:
                    self.cards[index][1].setText(stage)
                    self.cards[index][1].setToolTip(stage)
        self.refresh_round_choices()
        self.refresh_sample_view()
        self.open_folder.setEnabled(True)
        self.generate.setText('按当前讨论再做三个样片')
        self.show_option(self.current_option)
        if error or not self.options:
            self.show_round_note(error)

    def show_round_note(self, error=''):
        if error:
            self.message.setPlainText('本轮样片未完成：'+error)
        elif self.samples_direction!=self.creative_direction:
            self.message.setPlainText('这一轮使用此前的创作要求，仅供回看；按当前讨论试样后再定稿。')
        elif self.worker is not None and self.viewed_directory==self.active_directory:
            self.message.setPlainText('这一轮正在制作，可切换其他轮次回看；新结果不会改变定稿。')
        else:
            self.message.setPlainText('正在回看已有真实样片，可选择完整样片明确定稿；再试样使用上方所选原片与当前讨论。')

    def generate_samples(self):
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
        self.active_directory = None
        self.active_plans = []
        self.active_options = {}
        self.active_stages = {}
        self.options = self.active_options
        self.viewed_plans = []
        self.viewed_directory = None
        self.viewed_source = self.sample_source.currentData()
        self.viewing_original = False
        self.viewing_selected = False
        self.failure_message = ''
        self.samples_direction = self.creative_direction
        self.open_folder.setEnabled(False)
        self.reset_cards('设计方向中')
        self.sample_round.blockSignals(True)
        self.sample_round.setCurrentIndex(-1)
        self.sample_round.blockSignals(False)
        self.canvas.set_photo(self.sample_source.currentData())
        self.active_caption.setText('原片 · 正在设计三个方向')
        self.generate.setEnabled(False)
        self.change_source.setEnabled(False)
        self.sample_source.setEnabled(False)
        represented = list(dict.fromkeys([self.sample_source.currentData(),*self.sources]))
        self.worker = StyleWorker(represented,self.sample_source.currentData(),self.character,
            self.brief,self.root,self.config,creative_direction=self.creative_direction)
        self.worker.progress.connect(self.sample_progress)
        self.worker.plans_ready.connect(self.show_plans)
        self.worker.sample_stage.connect(self.show_stage)
        self.worker.sample_ready.connect(self.show_sample)
        self.worker.directory_created.connect(self.directory_ready)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.samples_finished)
        self.message.setPlainText('正在结合角色“'+self.character+'”、代表原片与补充要求设计三个方向。其余照片尚未提交。')
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
        self.state_changed.emit()

    def sample_progress(self, message):
        if self.viewed_directory==self.active_directory and not self.viewing_original and not self.viewing_selected:
            self.message.setPlainText(message)

    def show_stage(self, index, stage):
        self.active_stages[index] = stage
        if self.viewed_directory==self.active_directory and not self.viewing_original:
            self.cards[index][1].setText(stage)
            self.cards[index][1].setToolTip(stage)

    def show_sample(self, option):
        self.active_options[option['index']] = option
        if self.viewed_directory==self.active_directory and not self.viewing_original:
            self.options = self.active_options
            self.display_sample(option)
            if self.current_option == option['index'] and not self.viewing_selected:
                self.show_option(option['index'])
                return
            self.update_creative_actions()
        self.state_changed.emit()

    def show_option(self, index):
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
        self.generate.setText('按当前讨论再做三个样片')
        if self.viewed_directory==self.active_directory and not self.viewing_original:
            complete = sum(option['status']=='complete' for option in self.active_options.values())
            for index in self.cards:
                if index not in self.active_options:
                    self.cards[index][1].setText('未完成')
                    self.cards[index][1].setToolTip('未完成')
            if not self.viewing_selected:
                if self.active_options:
                    self.message.setPlainText(f'已完成 {complete} 个真实样片。可比较、继续讨论或明确定稿；再次试样将保留这一轮。' if complete else
                        '本轮样片未完成，点击方案查看真实错误；已有文件保留，由你决定是否再试样。')
                if self.failure_message:
                    self.message.setPlainText('样片未全部完成：'+self.failure_message)
        self.refresh_round_choices()
        self.update_creative_actions()
        self.state_changed.emit()
        worker.deleteLater()

    def choose(self, index):
        if self.is_busy() or self.production_busy or self.viewing_original or self.viewing_selected or self.options.get(index,{}).get('status')!='complete':
            return
        if self.chat_input.toPlainText().strip():
            self.message.setPlainText('这段输入尚未发送，未纳入当前创作要求。先发送或清空输入，再决定是否定稿。')
            self.chat_input.setFocus()
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
        self.message.setPlainText('已定稿“'+option['title']+'”。可继续查看样片，批量制作由你另行开始。')
        self.update_creative_actions()
        self.style_selected.emit(option)
        self.state_changed.emit()
