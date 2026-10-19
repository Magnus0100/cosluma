"""A photo-first Cos workspace; all production actions belong to the host workbench."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, QTimer, QSize, QRectF, QPointF
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter
from PySide6.QtWidgets import (QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QSlider, QSplitter, QSizePolicy, QProgressBar, QLineEdit, QComboBox,
                              QCheckBox, QFormLayout, QSpinBox, QGroupBox, QToolButton, QScrollArea)

_RAW = frozenset(('.arw', '.cr2', '.cr3', '.nef', '.dng', '.raf', '.rw2', '.orf', '.srw', '.pef'))
_PHOTO = _RAW | frozenset(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp'))


def same_photo_finish_checkbox():
    control = QCheckBox('对照同一张照片的精修成片')
    control.setAccessibleName('对照同一张照片的精修成片')
    control.setToolTip('仅用于已有参考＋原景增强：参考作为光色与修饰标尺，执行时实际验证同照配准；'
                       '原片拥有几何和原生纹理，不复制参考人像几何或直接合成参考 RGB。新增生图 0 次，不使用局部生图。')
    return control


class _ImagePanel(QWidget):
    """Fit a prepared display image without loading the original RAW or native master."""

    viewChanged = Signal(float, float, float)
    regionSelected = Signal(list)
    detailRequested = Signal(list)

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.path = None
        self.image = QImage()
        self.empty_text = '导入照片开始制作' if title == '原片' else '制作完成后在这里查看效果'
        self._identity = None
        self.zoom = 1.; self.offset = QPointF(); self._drag = None
        self.selecting_region = False; self.selection = None; self._selection_start = None
        self.native_detail_enabled = False
        self.setToolTip('滚轮放大，拖动查看，双击还原；两边同步。')
        self.setMinimumSize(180, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(80)
        self._resize_timer.timeout.connect(self._read_image)

    def set_path(self, value):
        path = Path(value) if value else None
        try:
            stat = path.stat() if path else None
            identity = (str(path), stat.st_mtime_ns, stat.st_size) if stat else None
        except OSError:
            identity = (str(path), None, None)
        if identity == self._identity and path == self.path:
            return
        self.path = path
        self._identity = identity
        self._read_image()

    def _read_image(self):
        self.image = QImage()
        if self.path is None:
            self.empty_text = '导入照片开始制作' if self.title == '原片' else '制作完成后在这里查看效果'
        elif self.path.suffix.lower() in _RAW:
            self.empty_text = '原片预览准备中…'
        else:
            reader = QImageReader(str(self.path))
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid():
                scale = self.devicePixelRatioF()
                target = QSize(max(1, round(self.width() * scale * self.zoom)), max(1, round(self.height() * scale * self.zoom)))
                reader.setScaledSize(size.scaled(target, Qt.KeepAspectRatio))
                self.image = reader.read()
            if self.image.isNull():
                self.empty_text = '预览暂时无法显示'
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.path is not None:
            self._resize_timer.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#101115'))
        if not self.image.isNull():
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            rect = self.image_rect()
            painter.drawImage(rect, self.image)
            if self.selection:
                left, top, right, bottom = self.selection
                selected = QRectF(rect.left()+left*rect.width(), rect.top()+top*rect.height(),
                                  (right-left)*rect.width(), (bottom-top)*rect.height())
                painter.setPen(QColor('#b8a5ff')); painter.setBrush(QColor(135, 105, 255, 45))
                painter.drawRect(selected)
        else:
            painter.setPen(QColor('#b8bece'))
            font = self.font(); font.setPointSizeF(12); painter.setFont(font)
            center = self.rect().adjusted(20, 0, -20, -20)
            painter.drawText(center, Qt.AlignCenter, self.title)
            font.setPointSizeF(9); painter.setFont(font); painter.setPen(QColor('#777f91'))
            painter.drawText(center.adjusted(0, 55, 0, 55), Qt.AlignCenter | Qt.TextWordWrap, self.empty_text)

    def image_rect(self):
        size = self.image.size().scaled(self.size(), Qt.KeepAspectRatio)
        width, height = size.width()*self.zoom, size.height()*self.zoom
        return QRectF((self.width()-width)/2+self.offset.x()*self.width(),
                      (self.height()-height)/2+self.offset.y()*self.height(), width, height)

    def image_point(self, point, *, clip=False):
        if self.image.isNull(): return None
        rect = self.image_rect()
        if not clip and not rect.contains(point): return None
        return [min(1., max(0., (point.x()-rect.left())/max(1., rect.width()))),
                min(1., max(0., (point.y()-rect.top())/max(1., rect.height())))]

    def set_view(self, zoom, x, y):
        reload = self.zoom != zoom
        self.zoom = zoom; self.offset = QPointF(x, y)
        if reload: self._resize_timer.start()
        self.update()

    def wheelEvent(self, event):
        self.set_view(min(6., max(1., self.zoom*(1.2 if event.angleDelta().y()>0 else 1/1.2))), self.offset.x(), self.offset.y())
        if self.zoom == 1.: self.offset = QPointF()
        self.viewChanged.emit(self.zoom, self.offset.x(), self.offset.y()); event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.selecting_region:
                self._selection_start = self.image_point(event.position())
                return
            self._drag = event.position(); self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.selecting_region and self._selection_start is not None:
            point = self.image_point(event.position(), clip=True)
            left, top = self._selection_start
            self.selection = [min(left, point[0]), min(top, point[1]), max(left, point[0]), max(top, point[1])]
            self.update(); return
        if self._drag is not None and self.zoom > 1:
            delta = event.position()-self._drag; self._drag = event.position()
            self.offset += QPointF(delta.x()/max(1,self.width()), delta.y()/max(1,self.height()))
            self.viewChanged.emit(self.zoom, self.offset.x(), self.offset.y()); self.update()

    def mouseReleaseEvent(self, event):
        if self._selection_start is not None:
            point = self.image_point(event.position(), clip=True)
            left, top = self._selection_start; self._selection_start = None
            box = [min(left, point[0]), min(top, point[1]), max(left, point[0]), max(top, point[1])]
            if box[2]-box[0] > .001 and box[3]-box[1] > .001:
                self.selection = box; self.regionSelected.emit(box)
            self.update()
        self._drag = None; self.unsetCursor()

    def mouseDoubleClickEvent(self, event):
        if self.selecting_region: return
        point = self.image_point(event.position())
        if self.native_detail_enabled and point is not None:
            self.detailRequested.emit(point); return
        self.set_view(1.,0.,0.); self.viewChanged.emit(1.,0.,0.)


class _SegmentedProgressBar(QProgressBar):
    """Task milestone bar; plan items are not fabricated time partitions."""

    SEGMENTS = ((0, 100, '制作'),)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(0, 1, -1, -1)
        radius = min(6, rect.height() // 2)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#202631'))
        painter.drawRoundedRect(rect, radius, radius)
        minimum, maximum = self.minimum(), self.maximum()
        if maximum == minimum:
            # Indeterminate means the backend has not emitted a fraction yet.
            # Keep the whole route visible and mark the state honestly.
            painter.setPen(QColor('#aeb8ca'))
            painter.setFont(self.font())
            painter.drawText(rect, Qt.AlignCenter, '处理中…')
            painter.end()
            return
        value = min(maximum, max(minimum, self.value()))
        percent = 0.0 if maximum <= minimum else (value - minimum) / (maximum - minimum)
        for start, end, _label in self.SEGMENTS:
            segment_left = rect.left() + rect.width() * start / 100.0
            segment_right = rect.left() + rect.width() * end / 100.0
            filled_right = rect.left() + rect.width() * min(percent, end / 100.0)
            if filled_right > segment_left:
                fill = QRectF(segment_left, rect.top(), max(0.0, min(segment_right, filled_right) - segment_left), rect.height())
                painter.setBrush(QColor('#6d8cff' if start < 40 else '#58b7a8' if start < 55 else '#d5a85c' if start < 75 else '#bc78d2'))
                painter.drawRect(fill)
            if end < 100:
                painter.setPen(QColor('#0e1117'))
                painter.drawLine(round(segment_right), rect.top() + 2, round(segment_right), rect.bottom() - 2)
        painter.setPen(QColor('#f4f6fb'))
        painter.setFont(self.font())
        painter.drawText(rect, Qt.AlignCenter, f'{round(percent * 100)}%')
        painter.end()


class _ModelSelector(QComboBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        # The dark stylesheet suppresses the platform's dropdown arrow.
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor('#cbd0de' if self.isEnabled() else '#6e7485'))
        x, y = self.width()-16, self.height()//2
        painter.drawLine(x-4, y-2, x, y+2); painter.drawLine(x, y+2, x+4, y-2)
        painter.end()


class RegionalRefinementControls(QWidget):
    """One explicit native rectangle; the worker decides background safety."""
    enabledChanged = Signal(bool)
    boxChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox('局部 AI 精修（最多 1 区）')
        self.enabled.setAccessibleName('局部 AI 精修最多一个区域')
        self.enabled.setToolTip('原景增强任务可主动授权一次新的本地局部图像推理；已有参考默认零，重新制作不会自动增加额度。')
        self.enabled.toggled.connect(self.enabledChanged); layout.addWidget(self.enabled)
        self.note = QLabel('先载入原片尺寸，再框选需要调整的背景；人物安全范围由后台核验。')
        self.note.setWordWrap(True); self.note.setObjectName('caption'); layout.addWidget(self.note)
        toggle = QToolButton(); toggle.setText('位置与尺寸（原片像素）'); toggle.setCheckable(True)
        toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon); toggle.setArrowType(Qt.RightArrow)
        layout.addWidget(toggle)
        self.geometry = QWidget(); form = QFormLayout(self.geometry); form.setContentsMargins(0, 0, 0, 0)
        self.edits = {}
        for key, label in (('left', '左侧位置'), ('top', '顶部位置'), ('width', '区域宽度'), ('height', '区域高度')):
            edit = QSpinBox(); edit.setRange(0 if key in {'left', 'top'} else 1, 100000)
            edit.setAccessibleName(label)
            if key in {'width','height'}: edit.setMaximum(768)
            edit.setSuffix(' px'); form.addRow(label, edit); self.edits[key] = edit
        self.apply = QPushButton('使用此背景区域'); self.apply.clicked.connect(self.apply_box); form.addRow(self.apply)
        self.apply.setAccessibleName('应用背景区域位置尺寸')
        toggle.setAccessibleName('展开背景区域位置尺寸')
        layout.addWidget(self.geometry); self.geometry.hide()
        toggle.toggled.connect(self.geometry.setVisible)
        toggle.toggled.connect(lambda checked: toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow))

    def apply_box(self):
        left, top, width, height = (self.edits[key].value() for key in ('left', 'top', 'width', 'height'))
        self.boxChanged.emit([left, top, left+width, top+height])

    def sync(self, size, enabled, box, *, available=True, cpu_reuse=False):
        known = isinstance(size, (list, tuple)) and len(size) == 2 and min(size) > 0
        self.enabled.blockSignals(True); self.enabled.setChecked(enabled); self.enabled.blockSignals(False)
        self.enabled.setEnabled(bool(known and available))
        self.geometry.setEnabled(bool(known and available and (enabled or cpu_reuse)))
        if known:
            width, height = map(int, size)
            for key, maximum in (('left', width-1), ('top', height-1), ('width', width), ('height', height)):
                self.edits[key].setMaximum(min(768, maximum) if key in {'width','height'} else maximum)
            values = (box[0], box[1], box[2]-box[0], box[3]-box[1]) if box else (0, 0, min(768,width), min(768,height))
            for key, value in zip(('left', 'top', 'width', 'height'), values):
                self.edits[key].setValue(int(value))
        self.note.setText('原片尺寸尚未载入；不会提交区域生图。' if not known else
            '已有局部候选可复用；此处调整CPU合成区域，执行恢复不新增生图。新区域须位于原模型画布内。' if cpu_reuse else
            '此路线不新增局部生图；已有选区仅供查看。' if not available else
            ('已选区域：' + '，'.join(map(str, box)) + '；后台仍会检查人物安全范围。') if box else
            '请在原片上框选背景，或展开位置与尺寸并应用；未选区不会开始局部生图。')


class SimpleCosWorkspace(QWidget):
    sourceChosen = Signal()
    sourcePathChanged = Signal(str)
    actionRequested = Signal(str)
    strengthChanged = Signal(int)
    briefChanged = Signal(str)
    generationModelChanged = Signal(str)
    characterChanged = Signal(str)
    characterNotesChanged = Signal(str)
    backgroundModeChanged = Signal(str)
    noEffectsChanged = Signal(bool)
    samePhotoFinishChanged = Signal(bool)
    regionalEnabledChanged = Signal(bool)
    regionalBoxChanged = Signal(list)
    normalizedRegionSelected = Signal(list)
    nativeDetailRequested = Signal(list)
    taskModeChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._busy = False
        self._source_path = ''
        self._comparing = False
        self._generation_models = []
        self._generation_model_locked = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(16)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel('Cos 后期'); title.setObjectName('pageTitle'); titles.addWidget(title)
        subtitle = QLabel('导入原片，选择制作方式，确认效果后保存。')
        subtitle.setObjectName('mutedLabel'); titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        self.professional_button = self._action_button('专业模式', 'professional')
        self.professional_button.setObjectName('secondaryButton')
        self.professional_button.setToolTip('打开图层、精修工具和工作流画布。')
        header.addWidget(self.professional_button, 0, Qt.AlignTop)
        layout.addLayout(header)
        self.version_summary = QLabel('尚未打开保存版本'); self.version_summary.setObjectName('caption')
        self.version_summary.setWordWrap(True); layout.addWidget(self.version_summary)
        self.direction_summary = QLabel('本次计划尚未冻结；先填写角色与效果要求。')
        self.direction_summary.setWordWrap(True); self.direction_summary.setObjectName('caption')
        self.direction_summary.setMaximumHeight(90)
        layout.addWidget(self.direction_summary)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel('生成模型'))
        self.generation_model = _ModelSelector()
        self.generation_model.setMinimumWidth(260)
        self.generation_model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.generation_model.setMinimumContentsLength(28)
        self.generation_model.setPlaceholderText('正在读取模型列表…')
        self.generation_model.currentIndexChanged.connect(self._generation_model_changed)
        model_row.addWidget(self.generation_model, 1)
        self.generation_model_note = QLabel('选择模型后再开始制作。')
        self.generation_model_note.setObjectName('caption')
        model_row.addWidget(self.generation_model_note)
        self.generation_settings_button = self._action_button('配置模型', 'generation_settings')
        self.generation_settings_button.setObjectName('secondaryButton')
        model_row.addWidget(self.generation_settings_button)
        layout.addLayout(model_row)

        content = QHBoxLayout(); content.setSpacing(20); layout.addLayout(content, 1)
        controls = QFrame(); controls.setObjectName('card')
        form = QVBoxLayout(controls); form.setContentsMargins(18, 18, 18, 18); form.setSpacing(12)
        self.source_button = QPushButton('导入原片')
        self.source_button.setObjectName('primaryButton')
        self.source_button.clicked.connect(lambda _checked=False: self.sourceChosen.emit())
        form.addWidget(self.source_button)
        self.source_label = QLabel('或将照片拖到这里')
        self.source_label.setObjectName('caption'); self.source_label.setWordWrap(True)
        self.source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addWidget(self.source_label)
        self.preparation_note = QLabel(''); self.preparation_note.setWordWrap(True)
        self.preparation_note.setObjectName('caption'); self.preparation_note.setVisible(False)
        form.addWidget(self.preparation_note)
        form.addWidget(QLabel('角色名（由你确认）'))
        self.character_name = QLineEdit(); self.character_name.setPlaceholderText('角色名，可留空；不会自动编造角色能力')
        self.character_name.setAccessibleName('角色名')
        self.character_name.textChanged.connect(self.characterChanged); form.addWidget(self.character_name)
        self.character_notes = QLineEdit(); self.character_notes.setPlaceholderText('作品、版本与已知角色线索（可选）')
        self.character_notes.setAccessibleName('角色作品版本说明')
        self.character_notes.textChanged.connect(self.characterNotesChanged); form.addWidget(self.character_notes)
        self.background_mode = QComboBox()
        self.background_mode.setAccessibleName('场景调整范围')
        for label, value in (('原景增强', 'enhance_source'), ('严格保留原背景', 'preserve'), ('允许换景', 'replace')):
            self.background_mode.addItem(label, value)
        self.background_mode.currentIndexChanged.connect(lambda: self.backgroundModeChanged.emit(self.background_mode.currentData()))
        form.addWidget(QLabel('场景调整范围')); form.addWidget(self.background_mode)
        self.brief = QLineEdit(); self.brief.setPlaceholderText('角色效果与氛围要求，例如：主效：……')
        self.brief.setAccessibleName('角色效果与氛围要求')
        self.brief.setToolTip('写明需要的角色意象和效果；不要特效也可直接写。不会自动添加未声明的能力。')
        self.brief.textChanged.connect(self.briefChanged); form.addWidget(self.brief)
        self.no_effects = QCheckBox('不加特效'); self.no_effects.toggled.connect(self.noEffectsChanged); form.addWidget(self.no_effects)
        locked = QLabel('固定保留身份、五官、妆造、姿态、手势与服装结构。')
        locked.setWordWrap(True); locked.setObjectName('caption'); form.addWidget(locked)
        self.task_mode = QComboBox()
        for label, mode in (('创作参考并完成后期', 'create_reference'), ('已有参考完整后期（默认零生图）', 'existing_reference'), ('自然原片后期（零生图）', 'local_portrait')):
            self.task_mode.addItem(label, mode)
        self.task_mode.setAccessibleName('本次制作输入方式')
        self.task_mode.currentIndexChanged.connect(lambda: self.taskModeChanged.emit(self.task_mode.currentData()))
        form.addWidget(self.task_mode)
        self.same_photo_finish = same_photo_finish_checkbox()
        self.same_photo_finish.toggled.connect(self.samePhotoFinishChanged)
        form.addWidget(self.same_photo_finish)
        form.addSpacing(8)
        heading = QLabel('选择制作方式'); heading.setObjectName('sectionTitle'); form.addWidget(heading)
        self.beauty_button = self._action_button('仅人像美化', 'beauty')
        self.beauty_button.setMinimumHeight(34); form.addWidget(self.beauty_button)
        beauty_note = QLabel('调整肤质与面部观感。'); beauty_note.setObjectName('caption'); form.addWidget(beauty_note)
        self.create_cos_button = self._action_button('完整 Cos 后期', 'create_cos')
        self.create_cos_button.setObjectName('primaryButton'); self.create_cos_button.setMinimumHeight(34)
        form.addWidget(self.create_cos_button)
        cos_note = QLabel('美化人物，搭配场景与光效。'); cos_note.setObjectName('caption'); cos_note.setWordWrap(True); form.addWidget(cos_note)
        self.candidate_panel = QFrame(); self.candidate_panel.setObjectName('candidatePanel')
        candidate_layout = QVBoxLayout(self.candidate_panel); candidate_layout.setContentsMargins(10, 8, 10, 8); candidate_layout.setSpacing(5)
        self.candidate_label = QLabel(''); self.candidate_label.setObjectName('caption'); self.candidate_label.setWordWrap(True); candidate_layout.addWidget(self.candidate_label)
        self.candidate_button = self._action_button('查看并选择候选设计', 'select_candidate')
        self.candidate_button.setObjectName('secondaryButton'); candidate_layout.addWidget(self.candidate_button)
        form.addWidget(self.candidate_panel); self.candidate_panel.setVisible(False)
        self.existing_reference_button = self._action_button('已有参考图（复用参考）', 'existing_reference')
        self.existing_reference_button.setObjectName('secondaryButton')
        self.existing_reference_button.setToolTip('复用参考，不重新生成全局参考。默认零生图；主动授权局部精修时，以检查计划中的预算为准。')
        form.addWidget(self.existing_reference_button)
        self.regional = RegionalRefinementControls()
        self.regional.setAccessibleName('快速制作局部精修设置')
        self.regional.enabledChanged.connect(self.regionalEnabledChanged)
        self.regional.boxChanged.connect(self.regionalBoxChanged); form.addWidget(self.regional)
        self.select_region = QPushButton('在原片上框选背景区'); self.select_region.setCheckable(True)
        self.select_region.setAccessibleName('在原片上框选背景区域')
        self.select_region.toggled.connect(self.set_region_selection); form.addWidget(self.select_region)
        self.check_plan_button = self._action_button('检查本次计划', 'check_plan'); form.addWidget(self.check_plan_button)
        self.check_plan_button.setAccessibleName('检查本次制作计划')
        self.start_plan_button = self._action_button('按已检查计划开始', 'start_plan')
        self.start_plan_button.setAccessibleName('按已检查计划开始制作'); self.start_plan_button.setEnabled(False)
        form.addWidget(self.start_plan_button)
        self.strength_group = QWidget(); strength_layout = QVBoxLayout(self.strength_group)
        strength_layout.setContentsMargins(0, 0, 0, 0); strength_layout.setSpacing(7)
        strength_row = QHBoxLayout(); strength_row.addWidget(QLabel('美化强度')); strength_row.addStretch()
        self.strength_label = QLabel('100%'); strength_row.addWidget(self.strength_label); strength_layout.addLayout(strength_row)
        self.strength_slider = QSlider(Qt.Horizontal); self.strength_slider.setRange(0, 100); self.strength_slider.setValue(100)
        self.strength_slider.valueChanged.connect(self._strength_changed); strength_layout.addWidget(self.strength_slider)
        form.addWidget(self.strength_group)
        self.compare_button = self._action_button('对比原片', 'compare')
        self.compare_button.setCheckable(True); form.addWidget(self.compare_button)
        self.native_review_button = self._action_button('同坐标原尺寸审查', 'native_review')
        self.native_review_button.setToolTip('双击左右照片任意位置，可核对已保存版本同一坐标的原尺寸像素。')
        form.addWidget(self.native_review_button)
        form.addStretch()
        # The quick route is intentionally coarse, but it must still expose
        # the percentage to an ordinary user.  A five-pixel line with hidden
        # text made a real 40% candidate wait look like a stalled request.
        self.progress = _SegmentedProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.setTextVisible(True)
        self.progress.setToolTip('检查本次制作计划后显示路线；当前模块以实际任务事件为准。百分比为任务里程碑，不表示耗时比例。')
        self._progress_stages = []
        self.progress.setFormat('%p%'); self.progress.setMaximumHeight(22); form.addWidget(self.progress)
        self.progress_detail = QLabel(''); self.progress_detail.setObjectName('caption'); self.progress_detail.setWordWrap(True)
        form.addWidget(self.progress_detail)
        self.state_label = QLabel('先导入一张原片。'); self.state_label.setObjectName('caption'); self.state_label.setWordWrap(True)
        form.addWidget(self.state_label)
        self.cancel_button = self._action_button('取消制作', 'cancel'); form.addWidget(self.cancel_button)
        self.save_button = self._action_button('保存高清版本', 'save')
        self.save_button.setObjectName('primaryButton'); form.addWidget(self.save_button)
        self.post_button = self._action_button('按当前算法重新制作', 'continue_post')
        self.post_button.setObjectName('secondaryButton')
        self.post_button.setToolTip('复用已有参考与美化，按当前算法重新制作并保存新版本，不新增生图。')
        self.post_button.setVisible(False); form.addWidget(self.post_button)
        self.export_button = self._action_button('导出成片', 'export'); form.addWidget(self.export_button)
        controls_scroll = QScrollArea(); controls_scroll.setWidgetResizable(True); controls_scroll.setWidget(controls)
        controls_scroll.setMinimumWidth(260); controls_scroll.setMaximumWidth(310)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); content.addWidget(controls_scroll)

        self.image_splitter = QSplitter(Qt.Horizontal)
        self.image_splitter.setChildrenCollapsible(False)
        self.source_image = _ImagePanel('原片')
        self.source_image.setAccessibleName('原片预览与背景区域框选')
        self.result_image = _ImagePanel('当前效果')
        self.source_image.viewChanged.connect(self.result_image.set_view)
        self.result_image.viewChanged.connect(self.source_image.set_view)
        self.source_image.regionSelected.connect(self.source_region_selected)
        self.source_image.detailRequested.connect(self.nativeDetailRequested)
        self.result_image.detailRequested.connect(self.nativeDetailRequested)
        self.source_card = self._image_card('原片', self.source_image)
        self.result_card = self._image_card('当前效果', self.result_image)
        self.image_splitter.addWidget(self.source_card); self.image_splitter.addWidget(self.result_card)
        self.image_splitter.setStretchFactor(0, 1); self.image_splitter.setStretchFactor(1, 1)
        content.addWidget(self.image_splitter, 1)
        self.set_state('', False, False, False, False)

    def set_region_selection(self, selecting):
        self.source_image.selecting_region = bool(selecting)
        self.source_image.setCursor(Qt.CrossCursor if selecting else Qt.ArrowCursor)
        self.source_image.setToolTip('拖动选择一个背景区域；不移动人物，不保证此区安全，后台仍会核验。' if selecting else
                                    '滚轮放大、拖动查看；双击可查看已保存版本的同坐标原尺寸细节。')

    def source_region_selected(self, normalized_box):
        self.normalizedRegionSelected.emit(normalized_box)
        self.select_region.setChecked(False)

    def _action_button(self, text, action):
        button = QPushButton(text)
        button.clicked.connect(lambda _checked=False, key=action: self.actionRequested.emit(key))
        return button

    @staticmethod
    def _image_card(title, image):
        frame = QFrame(); frame.setObjectName('card')
        layout = QVBoxLayout(frame); layout.setContentsMargins(1, 10, 1, 1); layout.setSpacing(8)
        label = QLabel(title); label.setObjectName('caption'); label.setContentsMargins(12, 0, 12, 0)
        layout.addWidget(label); layout.addWidget(image, 1)
        return frame

    def set_images(self, source_path=None, result_path=None):
        """The host supplies prepared display/proxy paths, never a source RAW decode."""
        self.source_image.set_path(source_path)
        self.result_image.set_path(result_path)

    def set_state(self, source_path, has_run, busy, has_result, beauty_ready, progress_active=None,
                  readiness='ready'):
        self._source_path = str(source_path or '')
        self._busy = bool(busy)
        source_ready = bool(self._source_path)
        self.source_label.setText(Path(self._source_path).name if source_ready else '或将照片拖到这里')
        self.source_label.setToolTip(self._source_path)
        self.source_button.setText('更换原片' if source_ready else '导入原片')
        self.source_button.setEnabled(not busy)
        startup_ready = readiness in {'ready', 'degraded'}
        self.beauty_button.setEnabled(source_ready and not busy and startup_ready)
        self.create_cos_button.setEnabled(source_ready and not busy and startup_ready)
        self.existing_reference_button.setEnabled(source_ready and not busy and startup_ready)
        self.brief.setEnabled(not busy)
        for control in (self.character_name, self.character_notes, self.background_mode, self.no_effects, self.check_plan_button, self.task_mode):
            control.setEnabled(not busy)
        self.generation_model.setEnabled(bool(self._generation_models) and not busy and not self._generation_model_locked)
        self.generation_settings_button.setEnabled(not busy and not self._generation_model_locked)
        self.cancel_button.setVisible(bool(busy))
        result_ready = bool(has_run and has_result)
        for button in (self.compare_button, self.save_button, self.export_button):
            button.setEnabled(result_ready and not busy)
        self.native_review_button.setEnabled(result_ready and not busy)
        self.source_image.native_detail_enabled = self.result_image.native_detail_enabled = result_ready and not busy
        self.strength_group.setVisible(bool(beauty_ready))
        self.strength_slider.setEnabled(bool(has_run and beauty_ready and not busy))
        # ``progress_active`` also covers the deliberate candidate-selection
        # gate: the worker is idle so the candidate button remains usable, but
        # the route is still visibly parked at its 40% milestone.
        progress_visible = bool(busy if progress_active is None else progress_active)
        self.progress.setVisible(progress_visible)
        self.progress_detail.setVisible(progress_visible)
        if not result_ready:
            self.set_compare(False)
        self.state_label.setText('正在初始化后台与模型…' if readiness == 'initializing' else
            '模型配置需要处理。' if readiness == 'degraded' and source_ready else
            '正在制作中…' if busy else
            '美化已就绪，可调整强度后查看效果。' if beauty_ready else
            '确认效果后，保存高清版本。' if result_ready else
            '选择一种制作方式开始。' if source_ready else '先导入一张原片。')

    def set_generation_models(self, choices, selected, *, locked=False):
        """Display the host's public catalog; synchronization never emits a choice."""
        self._generation_model_locked = bool(locked)
        self.generation_model.blockSignals(True)
        if choices != self._generation_models:
            self._generation_models = list(choices)
            self.generation_model.clear()
            for row in choices:
                self.generation_model.addItem(row['label'], row['id'])
                self.generation_model.setItemData(self.generation_model.count()-1, row['detail'], Qt.ToolTipRole)
        self.generation_model.setCurrentIndex(self.generation_model.findData(selected))
        self.generation_model.blockSignals(False)
        choice = next((row for row in choices if row['id'] == selected), None)
        self.generation_model.setPlaceholderText('当前模式不生成参考图' if choices else '正在读取模型列表…')
        self.generation_model_note.setText(choice['detail'] if choice else '可选择模型或导入已有参考图。')
        self.generation_model.setToolTip((choice['label'] + '\n' if choice else '') + '点击选择生成模型；切换不会立即生图。')
        self.generation_model.setEnabled(bool(choices) and not self._busy and not locked)
        self.generation_settings_button.setEnabled(not self._busy and not locked)

    def _generation_model_changed(self, _index):
        provider = self.generation_model.currentData()
        if provider and self.generation_model.isEnabled():
            self.generationModelChanged.emit(provider)

    def set_candidate_pending(self, pending, count=0):
        """Expose the mandatory candidate-selection step in quick mode."""
        pending = bool(pending)
        self.candidate_panel.setVisible(pending)
        if pending:
            self.candidate_label.setText(f'已生成 {int(count or 0)} 张角色风格候选；先选择一张，再进入仿色、融合和后期。')

    def set_post_continuation(self, available, *, dirty=False):
        """Expose the explicit hand-off from local fusion to effect completion."""
        available = bool(available)
        self.post_button.setVisible(available)
        self.post_button.setEnabled(available and not self._busy)
        self.post_button.setText('保存并按当前算法重制' if dirty else '按当前算法重新制作')

    def set_preparation_state(self, text):
        self.preparation_note.setText(text)
        self.preparation_note.setVisible(bool(text))

    def set_progress_stages(self, stages):
        """Use the checked route verbatim, including conditional/manual items."""
        self._progress_stages = [str(row.get('label') or row.get('id') or '')
            if isinstance(row, dict) else str(row) for row in (stages or [])]
        self._progress_stages = [label for label in self._progress_stages if label]
        route = ('本次路线共 ' + str(len(self._progress_stages)) + ' 项：'
            + ' → '.join(self._progress_stages) if self._progress_stages else '本次制作路线尚未确认。')
        self.progress.setToolTip(route + '\n包含条件步骤及人工审片／导出，以实际任务模块为准；里程碑百分比不表示耗时比例。')

    def set_job_progress(self, progress=None, stage='', message='', local_progress=None,
                         stage_index=None, stage_count=None):
        """Show real coarse stage progress; None keeps an indeterminate bar."""
        if progress is None:
            self.progress.setRange(0, 0)
            overall = '整体处理中'
        else:
            value = min(100, max(0, round(float(progress) * 100)))
            self.progress.setRange(0, 100); self.progress.setValue(value)
            self.progress.setFormat(f'{value}%')
            overall = f'任务里程碑 {value}%'
        names = {'base': '原片准备', 'selecting_variant': '所选候选准备', 'awaiting_reference': '等待选择候选', 'workflow': '制作流程',
                 'portrait_beauty_preparation': '人像美化', 'beauty_source_preparation': '人像美化',
                 'reference_preparation': '仿色融合',
                 'beauty': '人像美化', 'colour': '仿色融合',
                 'generating': '创作参考', 'rendering': '原尺寸渲染',
                 'imported': '原片导入', 'candidate_ready': '参考图已绑定',
                 'source-retouch': '原片精修', 'review': '待审片',
                 'completion': '后期处理', 'material_finish': '材质协调',
                 'reference_scene': '场景协调', 'independent_effects': '已授权效果处理',
                 'final_local_repair': '局部修复',
                 'post_save_wait': '高清保存', 'post': '成片处理 / 审片'}
        label = names.get(str(stage), str(stage or '当前阶段'))
        if self._progress_stages:
            detail = f'{overall} · 当前：{label} · 本次路线 {len(self._progress_stages)} 项'
            if local_progress is not None:
                detail += f' · 本步骤 {min(100, max(0, round(float(local_progress) * 100)))}%'
        elif local_progress is None:
            detail = f'{overall} · {label} · 阶段处理中'
        else:
            local = min(100, max(0, round(float(local_progress) * 100)))
            detail = f'{overall} · {label} {local}%'
            if stage_index is not None and stage_count:
                detail += f' · 第 {int(stage_index)+1}/{int(stage_count)} 阶段'
        self.progress_detail.setText(detail)
        if message:
            self.state_label.setText(str(message))

    def set_message(self, message):
        self.state_label.setText(str(message or ''))

    def set_compare(self, before):
        self._comparing = bool(before)
        self.compare_button.blockSignals(True)
        self.compare_button.setChecked(self._comparing)
        self.compare_button.blockSignals(False)
        self.compare_button.setText('结束对比' if self._comparing else '对比原片')

    def set_strength(self, value):
        value = min(100, max(0, int(value)))
        self.strength_slider.blockSignals(True); self.strength_slider.setValue(value); self.strength_slider.blockSignals(False)
        self.strength_label.setText(str(value) + '%')

    def _strength_changed(self, value):
        self.strength_label.setText(str(value) + '%')
        if self.strength_slider.isEnabled():
            self.strengthChanged.emit(value)

    @staticmethod
    def _drop_path(event):
        urls = event.mimeData().urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = Path(urls[0].toLocalFile())
        return path if path.is_file() and path.suffix.lower() in _PHOTO else None

    def dragEnterEvent(self, event):
        if not self._busy and self._drop_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = self._drop_path(event)
        if self._busy or path is None:
            event.ignore(); return
        self.sourcePathChanged.emit(str(path.resolve()))
        event.acceptProposedAction()
