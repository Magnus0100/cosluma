# 第三方许可

项目代码许可见根目录 [LICENSE](../LICENSE)。本目录中的上游许可证与版权声明
保持原文，各组件版权归其作者所有。

当前代码直接使用 PySide6/Qt、NumPy、Pillow、pillow-heif/libheif、OpenCV、SciPy、rawpy/LibRaw、
color-matcher。相关依赖的许可文本保存在各组件目录中。
Windows 发布包同时包含 Python 与 PyInstaller 引导程序及其许可；发布源码包含构建配置。
当前 OpenCV 依赖为 `opencv-python-headless`，桌面界面由 PySide6 提供。
Pillow 当前为 13.0.0。构建时另从实际安装版本复制依赖的许可证、版权声明与随包第三方声明，
包括由 PyInstaller 收集的辅助模块；目录中的额外声明不表示软件提供对应的旧模型或功能。

本项目代码按 GPL-3.0-or-later 发布；每个 Release 随附同一提交的源码与构建配置。
Qt/PySide、LibRaw 等组件保留各自许可。Windows 包使用独立的动态库与 PySide 模块，
未阻止用户替换库；重新打包可使用仓库中的 `CosLuma.spec` 与构建依赖。
对应上游源码：

| 组件 | 源码 |
| --- | --- |
| Qt 6.11.2 | https://download.qt.io/official_releases/qt/6.11/6.11.2/single/ |
| PySide 6.11.2 | https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.2-src/ |
| rawpy 0.27.1 与其 LibRaw 子模块 | https://github.com/letmaik/rawpy/tree/v0.27.1 |
| color-matcher 0.6.0 | https://pypi.org/project/color-matcher/0.6.0/#files |
| pillow-heif 1.8.1 | https://github.com/bigcat88/pillow_heif/tree/v1.8.1 |
| libheif 1.23.6、libde265 1.1.3、x265 4.3 | https://github.com/strukturag/libheif/tree/v1.23.6 · https://github.com/strukturag/libde265/tree/v1.1.3 · https://github.com/Multicorewareinc/x265/tree/4.3 |

整头分割和超分通过仓库外安装的 ComfyUI 服务调用。SAM3.1 权重使用随权重提供的 SAM License；
转换为服务可读取的权重格式后仍保留原许可。ComfyUI、SAM3.1、PyTorch 和超分模型的许可
随各自安装保留，不随本项目主许可证重新授权。

可选美化节点随附 [FaceXFormer](https://github.com/Kartik-3004/facexformer/tree/10fe8291f8a64e2ca1daf938e3e0007bd860303b)
的网络与 Transformer 实现，使用 MIT 许可，原文保存在 `FaceXFormer/LICENSE` 和节点目录。
适配仅取消额外的 ImageNet 初始化下载，随后严格加载完整 FaceXFormer 权重；权重本身位于仓库外，
按 [官方权重页](https://huggingface.co/kartiknarayan/facexformer)的许可使用，转换为 safetensors 不改变张量或许可。

[BeautyGRPO](https://github.com/vivoCameraResearch/BeautyGRPO) 的公开推理调用用于独立编写服务适配，直接调用官方 Diffusers FluxKontextPipeline，未复制其推理实现。
上游代码声明 MIT，同时明确预训练权重与 FLUX.1-Kontext-dev 底座有各自许可；
不据代码许可推定 BeautyGRPO 权重已重新授权。BeautyGRPO、FLUX、文本编码器、VAE 与 MediaPipe FaceLandmarker 权重
由服务部署者从上游取得，未随桌面或源码包分发。FLUX.1-dev 底座适用其非商业许可；
其他权重按各自模型页与随权重条款使用，不受本项目 GPL 许可替代。
MediaPipe 运行库在服务环境安装，其代码适用 [Apache-2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)；
FaceLandmarker 使用 Google 发布的模型包，未将运行库或模型包随桌面分发。
Diffusers、Transformers、Accelerate 与 PEFT 仅在模型服务环境安装，各自的 Apache-2.0 许可随安装保留。
随附的 `providers/prompts.example.json` 是本项目原创通用提示词，使用项目的 GPL-3.0-or-later 许可。
个人提示词库位于仓库外，外部引用条目的来源、作者与许可随该库保留。
