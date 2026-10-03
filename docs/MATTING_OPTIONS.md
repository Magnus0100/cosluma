# 抠图 / Matting 依赖调查

更新时间：2026-10-04

## 调查结果

当前工作区可用的 bundled Python：

```text
C:\Users\80959\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
Python 3.12.14
```

该环境已有 `numpy 2.3.5`、`Pillow 12.3.0`，但没有可直接使用的分割或 matting 后端：

| 包 | 结果 |
|---|---|
| `torch` / `torchvision` | 未安装 |
| `opencv-python` (`cv2`) | 未安装 |
| `onnxruntime` / `onnxruntime-gpu` | 未安装 |
| `rembg` | 未安装 |
| `scipy` / `scikit-image` | 未安装 |
| `transformers` / `segment_anything` | 未安装 |
| `kornia` / `transparent_background` | 未安装 |

本机 PATH 也没有可用的独立 Python、Conda、Micromamba、FFmpeg 或 ImageMagick；只有 WindowsApps 的 Python 占位入口。因此现在的 Demo 依赖 Pillow + NumPy 是当前环境中最稳的可复现基线。

## 对十小时 Demo 的建议

暂时不下载大模型。两组数据已经可以用一次人工 alpha 编辑验证主流程，当前主要可见问题是轮廓范围和羽化，而不是缺少神经网络能力。继续使用 `scripts/edit_mask.py` 生成灰度 alpha，并通过 `run_demo.py --alpha ... --fit-alpha` 复跑。

如果人工 mask 仍无法处理发丝、抬手或半透明边缘，再单独增加 matting 实验环境。优先级建议：

1. `onnxruntime` + 一个轻量 MODNet/BRIA RMBG ONNX 模型：部署小、CPU 可跑，便于不引入 PyTorch。
2. `rembg`：调用简单，但仍需下载模型，且模型输出需要针对本项目的逆光和发丝重新调 alpha。
3. `torch` + `torchvision` / SAM：安装和模型体积最大，不适合作为当前十小时 Demo 的依赖。

上述包当前均未下载；本调查没有修改 Python 环境或联网安装任何内容。

