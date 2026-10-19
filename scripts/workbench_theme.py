"""Quiet photo-editor styling and a few small PySide6 presentation helpers."""
from __future__ import annotations


COLORS = {
    "window": "#16171c", "surface": "#1d1f26", "raised": "#252731",
    "canvas": "#101115", "border": "#30333e", "text": "#eceef5",
    "muted": "#a2a7b7", "accent": "#9380ff", "accent_hover": "#a491ff",
}


def configure_font(app, point_size: float = 9.5) -> None:
    """Use readable Windows UI type, including offscreen screenshot sessions."""
    import os
    from pathlib import Path
    from PySide6.QtGui import QFont, QFontDatabase

    families = set(QFontDatabase.families())
    if "Microsoft YaHei UI" not in families:
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("msyh.ttc", "msyhbd.ttc"):
            path = fonts / name
            if path.is_file():
                QFontDatabase.addApplicationFont(str(path))
        families = set(QFontDatabase.families())
    family = next((name for name in ("Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI")
                   if name in families), app.font().family())
    font = QFont(family)
    font.setPointSizeF(point_size)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)


def theme_stylesheet() -> str:
    return """
QWidget { color: #eceef5; selection-background-color: #6655ba; selection-color: white; }
QMainWindow, QDialog { background: #16171c; }
QLabel { background: transparent; }
QLabel#brandName { font-size: 15pt; font-weight: 700; color: #f7f7fc; }
QLabel#projectLabel, QLabel#mutedLabel { color: #a2a7b7; }
QLabel#sectionTitle { font-size: 11pt; font-weight: 600; color: #e9eaf2; padding: 2px 0; }
QLabel#pageTitle { font-size: 18pt; font-weight: 600; color: #f4f4fa; }
QLabel#caption { color: #a2a7b7; font-size: 8.5pt; }
QWidget#brandBar { background: #191a20; border-bottom: 1px solid #2c2e38; }
QWidget#inspector, QFrame#card { background: #1d1f26; border: 1px solid #2d303b; border-radius: 10px; }
QWidget#canvas, QScrollArea#canvas { background: #101115; border: 1px solid #272a34; border-radius: 10px; }
QLabel#canvas { background: #101115; border: none; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QSplitter::handle { background: #16171c; }
QSplitter::handle:horizontal { width: 7px; }
QSplitter::handle:vertical { height: 7px; }
QSplitter::handle:hover { background: #383246; }

QPushButton, QToolButton { background: #282b35; border: 1px solid #373b48; border-radius: 7px;
    padding: 7px 11px; color: #e7e9f2; min-height: 18px; }
QPushButton:hover, QToolButton:hover { background: #343745; border-color: #555967; }
QPushButton:pressed, QToolButton:pressed { background: #22242e; }
QPushButton:focus, QToolButton:focus { border-color: #9380ff; }
QPushButton:disabled, QToolButton:disabled { background: #22242b; border-color: #2d3039; color: #666c7d; }
QPushButton#primaryButton { background: #8974ef; border-color: #a08aff; color: #ffffff; font-weight: 600; padding: 9px 16px; }
QPushButton#primaryButton:hover { background: #9a85fc; border-color: #b3a2ff; }
QPushButton#primaryButton:pressed { background: #7862dc; }
QPushButton#primaryButton:disabled { background: #3d3654; border-color: #4c4266; color: #8c809f; }
QPushButton#secondaryButton { background: transparent; border-color: #3b3f4c; }
QPushButton#secondaryButton:hover { background: #2d303b; border-color: #626675; }
QToolButton#accordion { background: transparent; border: none; border-bottom: 1px solid #30333e;
    border-radius: 0; padding: 10px 2px; text-align: left; font-weight: 600; color: #cbd0de; }
QToolButton#accordion:hover { color: #ffffff; background: #252731; }
QToolButton#accordion:checked { color: #eceef5; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background: #22242d; border: 1px solid #353946; border-radius: 7px;
    padding: 7px 9px; color: #e5e8f1; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #9380ff; background: #262832; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background: #1e2027; border-color: #2c2f39; color: #6e7485; }
QLineEdit[readOnly="true"] { color: #b5bbca; background: #1c1e25; }
QComboBox { padding-right: 28px; }
QComboBox::drop-down { width: 24px; border: none; }
QComboBox QAbstractItemView { background: #252731; border: 1px solid #454957;
    padding: 5px; selection-background-color: #393348; outline: none; }
QSpinBox::up-button, QDoubleSpinBox::up-button { width: 20px; border: none; border-top-right-radius: 6px; }
QSpinBox::down-button, QDoubleSpinBox::down-button { width: 20px; border: none; border-bottom-right-radius: 6px; }
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: #383b48; }

QGroupBox { background: #1d1f26; border: 1px solid #30333e; border-radius: 9px;
    margin-top: 15px; padding: 16px 12px 12px 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 6px; color: #cdd2e0; }
QCheckBox, QRadioButton { spacing: 8px; color: #cbd0de; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #515666; border-radius: 4px; background: #23262f; }
QCheckBox::indicator:hover { border-color: #ad9cff; }
QCheckBox::indicator:checked { background: #9580fc; border-color: #b7a9ff; }
QCheckBox::indicator:disabled { background: #252730; border-color: #373b46; }
QRadioButton::indicator { width: 14px; height: 14px; border: 1px solid #515666; border-radius: 8px; background: #23262f; }
QRadioButton::indicator:checked { background: #9580fc; border: 3px solid #383040; }

QSlider:horizontal { min-height: 22px; }
QSlider::groove:horizontal { height: 4px; background: #393d4a; border: none; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #9380ff; border-radius: 2px; }
QSlider::add-page:horizontal { background: #393d4a; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; height: 12px; margin: -5px 0;
    border-radius: 6px; background: #e5dfff; border: 1px solid #a391ff; }
QSlider::handle:horizontal:hover { background: #ffffff; border: 2px solid #b8aaff; }
QSlider::handle:horizontal:pressed { background: #b5a4ff; }
QSlider::sub-page:horizontal:disabled { background: #555060; }
QSlider::handle:horizontal:disabled { background: #697080; border-color: #697080; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 3px 1px; }
QScrollBar::handle:vertical { background: #454a58; min-height: 32px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #686c7b; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 1px 3px; }
QScrollBar::handle:horizontal { background: #454a58; min-width: 32px; border-radius: 4px; }
QScrollBar::handle:horizontal:hover { background: #686c7b; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QListWidget, QTreeWidget { background: #1b1d24; border: 1px solid #2e323d; border-radius: 8px;
    outline: none; padding: 5px; alternate-background-color: #20222b; }
QListWidget::item, QTreeWidget::item { min-height: 24px; padding: 6px 8px; border-radius: 5px; }
QListWidget::item:hover, QTreeWidget::item:hover { background: #2a2d39; }
QListWidget::item:selected, QTreeWidget::item:selected { background: #39324e; color: #f1ecff; }
QListWidget#navRail { background: #191a21; border: none; border-right: 1px solid #2c2f39;
    border-radius: 0; padding: 12px 8px; font-size: 10pt; }
QListWidget#navRail::item { padding: 14px 12px; margin: 3px 0; min-height: 20px;
    border: 1px solid transparent; border-radius: 8px; color: #a8adbc; }
QListWidget#navRail::item:hover { background: #262832; color: #e8eaf3; }
QListWidget#navRail::item:selected { background: #332e47; border-color: #4b4266; color: #d9cfff; }
QHeaderView::section { background: #252731; color: #bfc5d4; border: none; border-bottom: 1px solid #3a3e4b; padding: 8px; }

QTabWidget::pane { border: 1px solid #30333e; border-radius: 8px; background: #1d1f26; top: -1px; }
QTabBar::tab { background: transparent; color: #969eaf; border: none; border-bottom: 2px solid transparent;
    padding: 10px 13px; margin-right: 3px; }
QTabBar::tab:hover { background: #282b35; color: #e1e5ef; }
QTabBar::tab:selected { color: #d6caff; border-bottom-color: #9380ff; background: #252530; }
QTabBar::tab:disabled { color: #626979; }
QTabBar::tab:left { border-bottom: none; border-left: 2px solid transparent; padding: 14px 12px; margin: 2px 0; }
QTabBar::tab:left:selected { border-left-color: #9380ff; background: #332e47; }
QLabel#statusPill { background: #292d39; color: #bac2d4; border: 1px solid #3d4455;
    border-radius: 9px; padding: 4px 9px; font-size: 8.5pt; }
QLabel#statusPill[tone="success"] { color: #a3dabb; background: #21372d; border-color: #365b47; }
QLabel#statusPill[tone="active"] { color: #d0c2ff; background: #342d49; border-color: #564672; }
QLabel#statusPill[tone="warning"] { color: #e4c193; background: #3a3025; border-color: #5e4b34; }
QToolTip { background: #303440; border: 1px solid #5a6071; border-radius: 6px; color: #f0f2f8; padding: 7px; }
QMenu { background: #252832; border: 1px solid #454b5b; padding: 5px; }
QMenu::item { padding: 7px 20px; border-radius: 5px; }
QMenu::item:selected { background: #39324e; }
QStatusBar { background: #181a20; border-top: 1px solid #2c303b; color: #a2a7b7; }
"""


def apply_theme(app) -> None:
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in ((QPalette.Window, COLORS["window"]), (QPalette.WindowText, COLORS["text"]),
                        (QPalette.Base, COLORS["surface"]), (QPalette.AlternateBase, COLORS["raised"]),
                        (QPalette.Text, COLORS["text"]), (QPalette.Button, COLORS["raised"]),
                        (QPalette.ButtonText, COLORS["text"]), (QPalette.Highlight, "#39324e"),
                        (QPalette.HighlightedText, "#f3edff"), (QPalette.ToolTipBase, "#303440"),
                        (QPalette.ToolTipText, COLORS["text"]), (QPalette.PlaceholderText, "#80889b")):
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#6e7485"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#6e7485"))
    app.setPalette(palette)
    configure_font(app)
    app.setStyleSheet(theme_stylesheet())






def card():
    from PySide6.QtWidgets import QFrame
    frame = QFrame()
    frame.setObjectName("card")
    return frame


def accordion(title: str, content, *, expanded: bool = False):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSizePolicy, QToolButton, QVBoxLayout, QWidget
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    header = QToolButton()
    header.setObjectName("accordion")
    header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    header.setText(title)
    header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    header.setCheckable(True)
    header.setChecked(expanded)
    header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
    content.setVisible(expanded)
    def toggle(checked):
        content.setVisible(checked)
        header.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
    header.toggled.connect(toggle)
    layout.addWidget(header)
    layout.addWidget(content)
    return container
