# 第三方许可

项目代码许可见根目录 [LICENSE](../LICENSE)。本目录中的上游许可证与版权声明
保持原文，各组件版权归其作者所有。

当前代码直接使用 PySide6/Qt、NumPy、Pillow、OpenCV、SciPy、rawpy/LibRaw、
color-matcher。相关依赖的许可文本保存在各组件目录中。
Windows 发布包同时包含 Python 与 PyInstaller 引导程序及其许可；发布源码包含构建配置。
当前 OpenCV 依赖为 `opencv-python-headless`，桌面界面由 PySide6 提供。

整头分割和超分通过仓库外安装的 ComfyUI 服务调用。SAM3.1 权重使用随权重提供的 SAM License；
转换为服务可读取的权重格式后仍保留原许可。ComfyUI、SAM3.1、PyTorch 和超分模型的许可
随各自安装保留，不随本项目主许可证重新授权。
提示词库位于仓库外，引用条目的来源、作者与许可随该库保留。
