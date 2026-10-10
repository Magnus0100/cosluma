# Qwen Image 2.1 本地服务

Qwen是可选的参考图与独立效果生成服务。图像服务与角色研究、文字规划和视觉审片连接分别配置；选择本地图像生成不会自动配置文本/视觉服务。

| Provider | 图像模型 | 取向 |
|---|---|---|
| `qwen_image_2_1` | Comfy-Org适配的官方BF16权重 | 完整精度 |
| `qwen_image_2_1_uncensored` | `abenzerps/Qwen-Image-2.1-Uncensored-GGUF` BF16 | 指定社区版本 |
| `qwen_image_2_1_fast` | 官方INT8 ConvRot | 速度选项 |
| `qwen_image_2_1_uncensored_fast` | 社区Q6_K GGUF | 速度选项 |

BF16选项使用BF16 Qwen3-VL编码器，速度选项使用INT8编码器，均使用BF16 VAE。Uncensored是发布者的版本名称，不能保证任意请求的输出效果。精度选项和可运行性不代替作品质量比较。

## 独立安装

先阅读上游代码和权重许可。Qwen Image 2.1使用Qwen Research License，仅允许非商业研究/评估；商业使用需要另行取得许可。ComfyUI和GGUF节点分别有自己的条款。

程序使用 [qwen-image-2.1-models.example.json](../providers/qwen-image-2.1-models.example.json) 中的公共清单，并通过 `COSPLAY_QWEN_DEPLOY_ROOT`选择独立部署目录。源码用户也可复制为本地 `providers/qwen-image-2.1-models.json`。清单记录每个权重的来源、大小、SHA-256和Provider所需角色；社区GGUF固定到提交 `6b34e59458d3eb7ba6a6f86a116aed5253dc02c3`，自行维护清单时保留这些身份和必需角色。

在自己的模型目录安装ComfyUI、兼容的PyTorch/CUDA环境及 `leejet/ComfyUI-GGUF`。自动启动器要求下列布局：

```text
模型目录/
├─ runtime/Scripts/python.exe
└─ ComfyUI/
   ├─ main.py
   └─ models/
```

桌面程序包不包含这些权重或运行时。显存、系统内存和磁盘需求随精度及卸载策略变化，安装桌面依赖不会自动完成模型服务部署。

设置部署目录并下载所选角色，例如官方BF16组合：

```powershell
$qwenRoot = Join-Path $env:LOCALAPPDATA 'CosplayWorkbench\models\qwen-image-2.1'
$env:COSPLAY_QWEN_DEPLOY_ROOT = $qwenRoot
.\.venv\Scripts\python.exe .\scripts\download_qwen_models.py --root "$qwenRoot" --roles official_transformer text_encoder vae
```

下载命令支持 `--workers`、`--proxy` 和 `--chunk-root`。下载器按大小与SHA-256确认文件，支持带Content-Range检查的分段续传；它不安装ComfyUI或CUDA运行时。

当前仓库提供 [GGUF兼容补丁](../providers/patches/comfyui-gguf-linear-compat.patch)，用于ComfyUI 0.39的Linear激活/残差接口。升级节点时核对上游是否已有对应修复，避免重复应用。

## 检查与启动

```powershell
.\.venv\Scripts\python.exe .\scripts\start_qwen_image.py --check --provider qwen_image_2_1
.\.venv\Scripts\python.exe .\scripts\start_qwen_image.py --provider qwen_image_2_1
```

`--check`列出所需模型文件及本地服务状态；完整文件的哈希验证由下载校验完成。默认服务监听 `http://127.0.0.1:8188`，日志位于部署目录的 `server.log`。工作台选择对应Provider后也会按需启动已安装的服务，启动锁避免重复冷启动。

在工作台连接设置中配置实际Provider。`COSPLAY_QWEN_API_BASE`可指定图像服务，`COSPLAY_VISION_PROVIDER`、`COSPLAY_VISION_API_BASE`和 `COSPLAY_VISION_MODEL`配置文本/视觉服务。工作流位于 `providers/workflows`；实际工作流、随机种子、模型、源图与输出身份随生成记录保存。

默认编辑工作流使用25步、Euler/simple、CFG 1，并按输入比例准备照片。生成结果仍是候选，须经过当前准备、最终合成与人工审片。参考纠错和零生图本地恢复的区别见 [GENERATION_PROMPT.md](GENERATION_PROMPT.md)。

## 上游来源

- [Qwen Image 2.1](https://github.com/QwenLM/Qwen-Image-2.1)
- [Comfy-Org适配权重](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)
- [社区GGUF权重](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF)
- [ComfyUI-GGUF](https://github.com/leejet/ComfyUI-GGUF)
- [Qwen Research License](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE)

其他生产组件的许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
