"""Explicit upstream downloads and installation instructions for model services."""
from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QHBoxLayout, QLineEdit, QListWidget, QPlainTextEdit, QPushButton,
    QScrollArea, QVBoxLayout, QWidget)

from settings import resource_path
from studio_widgets import ui_label


KONTEXT_REVISION = '24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d'
BEAUTY_REVISION = '241d57903834ba55cb544b665686eb5d494b3dcf'
MODELS = {
    'head': {
        'title': '整头分割 · SAM3.1',
        'use': '基础制作需要。使用 Comfy Org 为 ComfyUI 发布的 SAM3.1 FP16 权重，约 1.75 GB。',
        'page': 'https://huggingface.co/Comfy-Org/sam3.1',
        'download': 'https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors',
        'target': 'models/checkpoints/sam3.1_multiplex_fp16.safetensors',
        'note': '分割模型名称填写 sam3.1_multiplex_fp16.safetensors。服务须有 SAM3_Detect 节点；现有服务使用的其他兼容文件名可继续保留。'},
    'sr': {
        'title': '超分 · Nomos2',
        'use': '基础制作需要。作者 Phhofm 发布的 4xNomos2_hq_atd，约 82 MB。',
        'page': 'https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_atd',
        'download': 'https://github.com/Phhofm/models/releases/download/4xNomos2_hq_atd/4xNomos2_hq_atd.safetensors',
        'target': 'models/upscale_models/4xNomos2_hq_atd.safetensors',
        'note': '超分模型名称填写 4xNomos2_hq_atd.safetensors。下载 safetensors 文件即可。'},
    'kontext': {
        'title': '可选美肤 · FLUX Kontext',
        'use': '仅美肤需要。BFL 官方完整 BF16 Diffusers 组件，约 33.75 GB；另需 BeautyGRPO LoRA。',
        'page': 'https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev',
        'target': 'models/diffusers/FLUX.1-Kontext-dev',
        'note': '先在官方页登录并接受模型许可，再在服务环境运行 hf auth login。命令下载同一官方版本的完整 transformer、两个文本编码器、VAE、tokenizer 与 scheduler；不下载重复的单文件整合版。美化模型目录填写 FLUX.1-Kontext-dev。'},
    'beauty': {
        'title': '可选美肤 · BeautyGRPO',
        'use': '仅美肤需要。BeautyGRPO 官方完整 FLUX Kontext LoRA，约 179 MB。',
        'page': 'https://huggingface.co/Sehnsucht24/BeautyGRPO',
        'download': 'https://huggingface.co/Sehnsucht24/BeautyGRPO/resolve/'+BEAUTY_REVISION+'/fluxkontext/checkpoint/pytorch_lora_weights.safetensors',
        'target': 'models/loras/beautygrpo.safetensors',
        'note': '浏览器下载后将 pytorch_lora_weights.safetensors 重命名为 beautygrpo.safetensors。复制的命令会放到此名称；LoRA 字段填写 beautygrpo.safetensors。'},
    'face': {
        'title': '可选美化 · FaceXFormer',
        'use': '美肤和瘦脸的人脸分析需要。官方下载 ckpts/model.pt，约 1.01 GB，再无损提取完整推理权重。',
        'page': 'https://huggingface.co/kartiknarayan/facexformer',
        'download': 'https://huggingface.co/kartiknarayan/facexformer/resolve/main/ckpts/model.pt',
        'target': 'models/cosluma_face/facexformer.safetensors',
        'note': '在已有 ComfyUI Python 中转换，不能只改文件后缀。命令引用随软件的 convert_facexformer.py；远端部署时改为服务机器上这份脚本的路径。FaceXFormer 字段填写 facexformer.safetensors。'},
    'landmark': {
        'title': '可选美化 · FaceLandmarker',
        'use': '美肤和瘦脸的人脸分析需要。Google 官方 FaceLandmarker 模型包，约 3.8 MB。',
        'page': 'https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker',
        'download': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        'target': 'models/cosluma_face/face_landmarker.task',
        'note': '面部定位模型字段填写 face_landmarker.task。美化还需安装随软件提供的 CosLuma 服务节点与依赖，详见服务部署说明。'}
}


def powershell_quote(value):
    return "'"+str(value).replace("'", "''")+"'"


def download_command(key, comfy_root):
    """Return commands for the user to run in their existing service environment."""
    root = str(comfy_root).strip().replace('\\', '/').rstrip('/')
    model = MODELS[key]
    target = root+'/'+model['target']
    directory = target.rsplit('/', 1)[0]
    if key == 'head':
        return 'hf download Comfy-Org/sam3.1 checkpoints/sam3.1_multiplex_fp16.safetensors --local-dir '+powershell_quote(root+'/models')
    if key == 'kontext':
        components = ' '.join('--include '+powershell_quote(name) for name in ('model_index.json', 'transformer/*',
            'text_encoder/*', 'text_encoder_2/*', 'vae/*', 'scheduler/*', 'tokenizer/*', 'tokenizer_2/*', 'LICENSE.md'))
        return ('hf auth login\n'
            'hf download black-forest-labs/FLUX.1-Kontext-dev --revision '+KONTEXT_REVISION+
            ' '+components+' --local-dir '+powershell_quote(target))
    if key in ('sr', 'landmark'):
        return ('New-Item -ItemType Directory -Force -Path '+powershell_quote(directory)+' | Out-Null\n'
            'Invoke-WebRequest -Uri '+powershell_quote(model['download'])+' -OutFile '+powershell_quote(target))
    if key == 'beauty':
        # Download and copy are one Python call, so an unsuccessful download
        # cannot continue to install a different or unfinished file.
        code = ("from huggingface_hub import hf_hub_download; from pathlib import Path; import shutil; "
            "source=hf_hub_download('Sehnsucht24/BeautyGRPO', filename='fluxkontext/checkpoint/pytorch_lora_weights.safetensors', "
            f"revision={BEAUTY_REVISION!r}); target=Path({target!r}); "
            "target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)")
        return 'python -c '+powershell_quote(code)
    script = str(resource_path('providers/convert_facexformer.py')).replace('\\', '/')
    code = ("from huggingface_hub import hf_hub_download; import subprocess, sys; "
        f"source=hf_hub_download('kartiknarayan/facexformer', filename='ckpts/model.pt', local_dir={root+'/models/cosluma_face'!r}); "
        f"subprocess.run([sys.executable, {script!r}, source, {target!r}], check=True)")
    return 'python -c '+powershell_quote(code)


class ModelDownloads(QDialog):
    def __init__(self, parent=None, *, model='head'):
        super().__init__(parent)
        self.setWindowTitle('CosLuma 模型下载')
        self.resize(1040, 720)
        self.setMinimumSize(820, 600)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)
        layout.addWidget(ui_label('准备模型', 'title'))
        note = ui_label('模型放在提供服务的机器上。本机或远端已有这些模型时，直接填写服务地址和实际文件名即可。')
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.models = QListWidget()
        self.models.setFixedWidth(235)
        self.models.addItems(item['title'] for item in MODELS.values())
        row.addWidget(self.models)
        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(6, 0, 12, 12)
        body.setSpacing(12)
        self.title = ui_label('', 'heading')
        body.addWidget(self.title)
        self.use = ui_label('')
        self.use.setWordWrap(True)
        body.addWidget(self.use)
        actions = QHBoxLayout()
        self.download = QPushButton('浏览器下载')
        self.download.setProperty('kind', 'primary')
        self.download.clicked.connect(self.open_download)
        actions.addWidget(self.download)
        page = QPushButton('打开上游模型页')
        page.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.current_model()['page'])))
        actions.addWidget(page)
        actions.addStretch(1)
        body.addLayout(actions)
        self.note = ui_label('')
        self.note.setWordWrap(True)
        body.addWidget(self.note)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        root_row = QHBoxLayout()
        self.comfy_root = QLineEdit('D:/ComfyUI')
        self.comfy_root.setPlaceholderText('服务机器上实际的 ComfyUI 目录')
        self.comfy_root.textChanged.connect(self.update_command)
        root_row.addWidget(self.comfy_root, 1)
        browse = QPushButton('选择本机目录')
        browse.clicked.connect(self.choose_root)
        root_row.addWidget(browse)
        form.addRow('ComfyUI 目录', root_row)
        self.target = ui_label('')
        self.target.setWordWrap(True)
        self.target.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow('安装到', self.target)
        body.addLayout(form)
        self.path_note = ui_label('D:/ComfyUI 是路径示例，请改成服务机器上的实际目录。')
        self.path_note.setWordWrap(True)
        body.addWidget(self.path_note)
        body.addWidget(ui_label('PowerShell 下载命令', 'heading'))
        note = ui_label('在服务机器的已有 ComfyUI Python 环境执行。下载工具安装命令：python -m pip install "huggingface_hub>=1.0"。')
        note.setWordWrap(True)
        body.addWidget(note)
        self.command = QPlainTextEdit()
        self.command.setReadOnly(True)
        self.command.setFixedHeight(155)
        body.addWidget(self.command)
        self.copy_button = QPushButton('复制下载命令')
        self.copy_button.clicked.connect(self.copy_command)
        body.addWidget(self.copy_button, 0, Qt.AlignLeft)
        body.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        row.addWidget(scroll, 1)
        layout.addLayout(row, 1)
        footer = QHBoxLayout()
        docs = QPushButton('查看服务部署说明')
        docs.setProperty('kind', 'quiet')
        docs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(resource_path('README.md')))))
        footer.addWidget(docs)
        footer.addStretch(1)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.button(QDialogButtonBox.Close).setText('关闭')
        close.rejected.connect(self.reject)
        footer.addWidget(close)
        layout.addLayout(footer)
        self.models.currentRowChanged.connect(self.update_model)
        self.models.setCurrentRow(list(MODELS).index(model))

    def current_key(self):
        return list(MODELS)[self.models.currentRow()]

    def current_model(self):
        return MODELS[self.current_key()]

    def update_model(self):
        model = self.current_model()
        self.title.setText(model['title'])
        self.use.setText(model['use'])
        self.note.setText(model['note'])
        self.download.setText('前往官方页取得下载权限' if self.current_key() == 'kontext' else '浏览器下载')
        self.update_command()

    def update_command(self):
        root = self.comfy_root.text().strip().replace('\\', '/').rstrip('/')
        self.target.setText(root+'/'+self.current_model()['target'] if root else '请填写服务机器上的 ComfyUI 目录')
        self.command.setPlainText(download_command(self.current_key(), root) if root else '')
        self.copy_button.setEnabled(bool(root))
        self.copy_button.setText('复制下载命令')
        self.path_note.setVisible(root == 'D:/ComfyUI')

    def choose_root(self):
        directory = QFileDialog.getExistingDirectory(self, '选择已有 ComfyUI 目录', self.comfy_root.text())
        if directory:
            self.comfy_root.setText(directory)

    def open_download(self):
        model = self.current_model()
        QDesktopServices.openUrl(QUrl(model.get('download', model['page'])))

    def copy_command(self):
        QApplication.clipboard().setText(self.command.toPlainText())
        self.copy_button.setText('已复制')
