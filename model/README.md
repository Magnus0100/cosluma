# 本地模型

把模型和各自的运行环境放在软件主目录的 `model` 文件夹。软件优先识别这里的安装。

```text
cosplay-workbench.exe
_internal/
model/
  sapiens2/       # repo、checkpoints、runtime
  vitmatte/      # ViTMatte Hugging Face 模型目录
  dinov3/        # DINOv3 Hugging Face 模型目录
  sam3.1/        # SAM3.1 安装目录，含 code、权重、runtime
  birefnet/      # birefnet_lite_fp16.onnx
  lama/          # big-lama.pt
  realesrgan/    # realesr-general-x4v3.pth
  face/          # face_landmarker.task、runtime
  onnxruntime/   # 独立 ONNX Runtime 安装目录
  beautygrpo/    # base、lora、evidence、runtime、installation.json
  segface/       # repo、checkpoints、runtime
```

在“模型管理”中点击“自动查找已有模型”。标准目录里的分析模型会自动填写；完整安装会保存配置，缺少权重或运行环境时逐项提示。仅有权重文件还需要相应运行环境。

`local-models.json` 和 `beautygrpo.json` 可放在这个文件夹，路径相对于配置文件所在目录解析。移动完整安装后，无需逐项修改路径。大模型也可以通过 Windows 目录联接接入这个文件夹，避免复制；移动安装时须保留联接目标。

程序包只包含这份说明，不包含模型、运行环境或私人配置。模型遵循各自许可。
