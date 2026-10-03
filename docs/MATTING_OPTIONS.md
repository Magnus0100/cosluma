# 抠图 / Matting 依赖调查

更新时间：2026-10-04

## 调查结果

当前工作区可用的 bundled Python：

```text
C:\Users\80959\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
Python 3.12.14
```

默认 bundled 环境已有 `numpy 2.3.5`、`Pillow 12.3.0`。生产目录另外隔离安装了可选的 `onnxruntime 1.30.0`，并下载了 `u2net_human_seg.onnx`；默认路径仍不加载它：

| 包 | 结果 |
|---|---|
| `torch` / `torchvision` | 未安装 |
| `opencv-python` (`cv2`) | 未安装 |
| `onnxruntime` / `onnxruntime-gpu` | `onnxruntime` 已隔离安装到 `.cache/onnxruntime`；GPU 未安装 |
| `rembg` | 未安装 |
| `scipy` / `scikit-image` | 未安装 |
| `transformers` / `segment_anything` | 未安装 |
| `kornia` / `transparent_background` | 未安装 |

本机 PATH 也没有可用的独立 Python、Conda、Micromamba、FFmpeg 或 ImageMagick；只有 WindowsApps 的 Python 占位入口。因此现在的 Demo 依赖 Pillow + NumPy 是当前环境中最稳的可复现基线。

## 生产版本使用建议

默认生产路径不强制加载大模型。两组数据已经可以用人工 alpha 交付，当前主要可见问题仍是轮廓范围和羽化；需要批量生成初始 silhouette 时使用 U2Net，再通过 `scripts/edit_mask.py` 修边。

如果人工 mask 仍无法处理发丝、抬手或半透明边缘，继续使用 U2Net 作为初始轮廓，不把它当作最终 matting。其他后端优先级：

1. `onnxruntime` + 一个轻量 MODNet/BRIA RMBG ONNX 模型：部署小、CPU 可跑，便于不引入 PyTorch。
2. `rembg`：调用简单，但仍需下载模型，且模型输出需要针对本项目的逆光和发丝重新调 alpha。
3. `torch` + `torchvision` / SAM：安装和模型体积最大，不适合作为当前十小时 Demo 的依赖。

当前 U2Net 模型约 176 MB，CPU 推理约 2 秒/张。它只改善人物 silhouette，手指、飞发和包仍需人工修边；完整流程继续叠加 confidence profile 和 structure-safe。
