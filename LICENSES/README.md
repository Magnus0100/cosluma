# 第三方许可

项目代码许可见根目录 [LICENSE](../LICENSE)。本目录中的上游许可证与版权声明
保持原文，各组件版权归其作者所有。

当前代码直接使用 PySide6/Qt、NumPy、Pillow、OpenCV、SciPy、rawpy/LibRaw、
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

整头分割和超分通过仓库外安装的 ComfyUI 服务调用。SAM3.1 权重使用随权重提供的 SAM License；
转换为服务可读取的权重格式后仍保留原许可。ComfyUI、SAM3.1、PyTorch 和超分模型的许可
随各自安装保留，不随本项目主许可证重新授权。
提示词库位于仓库外，引用条目的来源、作者与许可随该库保留。
