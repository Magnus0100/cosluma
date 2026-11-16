# CosLuma

**为 Cosplay 摄影设计的 AI 后期工作台。** 先讨论方向、比较真实样片，再批量制作；用原片人物细节参与合成，保留身份与摄影质感。

简体中文 · [English](README.en.md)

[下载 Windows 版](https://github.com/Magnus0100/cosluma/releases/tag/v0.27.5) · [第三方许可](LICENSES/README.md) · [开发约定](AGENTS.md)

Windows x64 · 中文界面 · 源码 0.27.6（尚未发布）· 公开版 0.27.5

## 能做什么

- **从想法到样片**：结合角色、原片和讨论，为同一张代表原片制作三个完整方向；选中一张后可继续改稿。
- **定稿后制作整组**：沿用选定方向和样片，默认四张并行；制作范围包含样片原片时直接复用成片。
- **保留人物身份**：使用原片身份区域，进行整体等比对齐、光色匹配与浮点融合。修发、人物重打光独立可选，默认关闭。
- **成片后精修**：按需美肤、瘦脸、补发、调整发束或修饰颈部，保留基础成片与历史版本。
- **项目与交付管理**：独立保存照片、讨论、样片和版本；逐张选择交付版本，直接复制已有 PNG / JPEG。

## 开始使用

下载 `CosLuma-0.27.5-windows-x64.zip`，**完整解压**，运行 `CosLuma/CosLuma.exe`。桌面包已包含 Python 和图像处理依赖。

**模型服务、模型权重和账号需自行配置，不随桌面包提供。** 基础制作需要：

| 服务 | 要求 |
| --- | --- |
| 摄影 Agent | 兼容 Chat Completions，支持系统消息、图片输入、工具调用与 JSON 内容输出 |
| 整图生图 | 图像编辑接口，支持 multipart `image[]`；原片加风格参考需要同时接收两张图，返回图片 URL 或 `b64_json` |
| 分割与超分 | 部署 SAM3.1 和超分模型的 ComfyUI，可使用同一服务 |

服务可以在本机或远端。使用远端服务时，照片和提示词会发送到所配置的服务。

### 使用流程

1. **新建项目、配置连接**：按引导填写服务，或导入已有配置。连接页的“模型下载”提供上游地址、安装位置和命令。
2. **导入照片、填写角色**：支持常见图片、HEIC / HEIF 和相机 RAW。勾选决定制作范围，高亮行只改变预览；代表原片单独选择。
3. **讨论并试样**：说明想保留和改变的内容，按需开启修发或重打光，再点击“探索三个方向”。可沿选中样片只改一张，也可重新探索。
4. **明确定稿、批量制作**：选定满意的完整样片，制作勾选照片。某张失败时，其他照片继续；错误显示在对应步骤。
5. **选择版本、导出**：在“项目作品 / 交付”逐张采用版本，导出已有文件；需要时再主动进行美化、局部修饰或单张改稿。

项目恢复、切页和浏览读取已有记录；制作、改稿和可选精修由用户发起。

## 输出与处理方式

默认请求**长边 6000 像素**，保存带 sRGB 配置的 **16 位 PNG** 和同结果的 **8 位 JPEG**。原片只读，成片另存。

```text
原片 + 角色与方向 → 单张提示词 → 整图生图 → 必要时一次模型超分
→ SAM3.1 原片身份区域 → SIFT 等比对齐 → MKL 仿色 / 可选重打光
→ float32 蒙版融合 → 可选明度曲线 → PNG + JPEG
```

- 关闭修发时保留原片整头；开启时保留脸、耳与具体实体头饰，采用生成头发。原片服装与道具默认保留其设计。
- 基础链不以局部形变塑脸。人物重打光使用预测光照场调制原片像素，替代普通 MKL 仿色。
- 整图生图会重新生成画面；身体、背景等区域可能变化。发丝、半透明材质、硬阴影和大幅几何差异仍可能产生融合接缝。
- 模型环节包含 8 位输入输出，16 位保存不代表全链具有原片精度。只超分一次，覆盖不足时按实际尺寸保存。
- 失败保留已有文件与错误记录；重新制作由用户发起。模型和设备按连接配置使用。

<details>
<summary><strong>模型服务部署与配置</strong></summary>

### 配置位置

默认配置为 `~/CosplayWorkbench/config/cosluma.json`，可用 `--config` 或 `COSLUMA_CONFIG` 指定。项目默认保存在 `~/CosLuma/projects/`，每个项目有独立的 `project.json` 和工作目录。

配置模板见 [settings.example.json](providers/settings.example.json)。密钥可由环境变量或独立文件提供；界面保存密钥时，JSON 只记录文件路径。

| 配置项 | 用途 |
| --- | --- |
| `llm` / `image` | 摄影 Agent / 整图编辑的地址、模型和凭据 |
| `head` / `sr` | 分割 / 超分的 ComfyUI 根地址与模型名 |
| `relight` / `beauty` / `local_edit` | 可选重打光 / 美化 / 局部修饰 |
| `workers` / `output_long_edge` | 并行张数 / 输出长边，默认 `4` / `6000` |
| `prompt_library` | 留空或 `null` 使用内置分类参考库；自定义库为 JSON 数组，条目含 `title`、`text`、`source` |

### 基础 ComfyUI

服务需提供 `SAM3_Detect` 节点及标准图片上传、任务提交和结果接口。模型名称填写服务实际加载的文件名：

| 目录 | 默认文件 |
| --- | --- |
| `models/checkpoints/` | `sam3.1_multiplex_fp16.safetensors` |
| `models/upscale_models/` | `4xNomos2_hq_atd.safetensors` |

若服务尚未包含 SAM3 文本编码器 512 维投影修正，可在 ComfyUI 目录应用 [投影补丁](providers/patches/comfyui-sam3-clip-projection.patch)：`git apply --ignore-space-change <CosLuma>/providers/patches/comfyui-sam3-clip-projection.patch`，然后重启。

### 可选模型

在服务端把所需的 `providers/comfyui_cosluma_*` 文件夹复制到 `ComfyUI/custom_nodes/`，用该服务已有 Python 安装缺少的依赖并重启。桌面更新后，节点代码需在服务端单独更新。下载入口提供具体文件与命令。

| 功能 / 节点目录 | 权重与默认推理参数 |
| --- | --- |
| 重打光 · `comfyui_cosluma_relight` | 完整 PIXLRelight + Marigold-IID-Lighting v1.1；512 分析画布，网络 BF16，Marigold 单步 FP16，输出 float32 光照场；`cuda:0`、seed 42 |
| 美化 · `comfyui_cosluma_beauty` | 完整 BeautyGRPO LoRA + 官方 `FluxKontextPipeline`；1024 输入、28 步、guidance 2.5、seed 42；FGTBT WFLW 98 点定位，FaceXFormer 皮肤解析 |
| 局部修饰 · `comfyui_cosluma_local` | 默认 Qwen-Image-2.1 INT8 ConvRot，Qwen3-VL 8B 与 VAE 为 BF16；1024 面积分辨率、40 步、cfg 1、Euler/simple、seed 42 |
| 局部修饰 · `comfyui_cosluma_h3` | 可选 MiniMax-H3-Image INT8 ConvRot，Qwen3-VL 32B NVFP4 AWQ，VAE FP16；2048 面积分辨率、6 步、cfg 1、Euler/simple、seed 42 |

“面积分辨率”按输入宽高比换算，分别约 1 MP / 4 MP；模型文件位于服务机器，不随桌面分发。

**重打光**：默认目录为 `models/cosluma_relight/pixlrelight`（上游源码、`model.safetensors`、`config.yaml`）与 `models/cosluma_relight/marigold-iid-lighting-v1-1`。可通过 `relight.model_directory`、`intrinsics_directory`、`device`、`seed` 指定；设备也可明确选 `cpu`。节点另需 `addict==2.4.0`、OmegaConf。

**美化**：`models/diffusers/FLUX.1-Kontext-dev/` 放完整官方 Diffusers 目录；`models/loras/beautygrpo.safetensors` 放完整 LoRA；`models/cosluma_face/` 放 `facexformer.safetensors` 与 `fgtbt.safetensors`。原始分析权重用 [FaceXFormer 转换器](providers/convert_facexformer.py) 和 [FGTBT 转换器](providers/convert_fgtbt.py) 提取为推理文件。

`beauty.memory=bnb_nf4` 为 16 GB 显存的默认布局：Transformer 使用 NF4 / double quant / BF16 计算，完整 BF16 T5 由 Accelerate CPU offload，CLIP 与 VAE 保持 BF16；完整 LoRA 保留。`cuda` 布局将整套模型放入显存。分析设备用 `beauty.analysis_device` 指定，生成设备为 `beauty.inference_device=cuda:0`。Windows 服务使用 `PYTORCH_ALLOC_CONF=backend:native`、`PYTORCH_CUDA_ALLOC_CONF=backend:native`，启动参数为 `--disable-dynamic-vram --disable-cuda-malloc`。全流程桌面与参与服务合计运行内存峰值要求不超过 30 GB。

**局部修饰文件**：生成权重放 `models/diffusion_models/`，文本编码器放 `models/text_encoders/`，VAE 放 `models/vae/`。

| 引擎 | 生成权重 / 文本编码器 / VAE |
| --- | --- |
| `local_edit.engine=qwen` | `qwen_image_2.1_int8_convrot.safetensors` / `qwen3vl_8b_bf16.safetensors` / `qwen_image_2.1_vae_bf16.safetensors` |
| `local_edit.engine=h3` | `minimax_h3_image_pruned_int8_convrot.safetensors` / `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` / `minimax_h3_image_vae_fp16.safetensors` |

Qwen 使用 CUDA 与原生低显存卸载，VAE 分块 512、重叠 64。H3 参数在 `local_edit.h3`，需要原生 H3 支持；颈部输入整图，发丝输入细节图与长边 1536 的整图上下文。两种引擎仅将涂选区域写回原尺寸版本；H3 使用梯度域写回。局部修饰与美化完成后释放本次模型；已保存的结果可继续比较、调节与另存。

</details>

## 源码与接口

使用 Python 3.12，模型服务独立部署：

```powershell
python -m pip install -r scripts/requirements-desktop.txt
python -B scripts/qt_workbench.py
```

CLI 提供 `styles`（三个样片）、`stylize`（批量）、`compose`（复用生成图）和 `beautify`（成片美化）。`--root`、`--config` 放在子命令前：

```powershell
python -B scripts/project_cli.py --root D:/output --config D:/config/cosluma.json styles --source D:/photos --sample D:/photos/a.jpg --character "角色名"
python -B scripts/backend_server.py --root D:/output --config D:/config/cosluma.json
```

HTTP 默认监听 `127.0.0.1:8765`：`POST /api/styles`、`/api/stylize`、`/api/compose`、`/api/beautify`；用 `GET /api/jobs/<id>` 查询任务，`GET /api/status` 查看服务。桌面包分别用 `CosLuma.exe --cli`、`CosLuma.exe --server` 进入这两个入口。

- [photo_pipeline.py](scripts/photo_pipeline.py)：唯一单张入口 `finish_photo` 与批量入口 `finish_batch`。
- [model_clients.py](scripts/model_clients.py)：摄影 Agent、整图编辑与模型服务调用。
- [image_io.py](scripts/image_io.py) / [head_composite.py](scripts/head_composite.py)：图像读写、身份区域对齐与融合。
- [providers/](providers/)：模型服务节点与连接模板。

桌面构建：安装 [构建依赖](scripts/requirements-build.txt)，执行 `python -m PyInstaller --noconfirm CosLuma.spec`。开发边界见 [AGENTS.md](AGENTS.md)。

## 许可

代码采用 [GPL-3.0-or-later](LICENSE)。本项目不商用、不盈利；模型权重和第三方组件各自遵循上游许可，部分权重存在非商业或地区限制，详见 [第三方许可](LICENSES/README.md)。
