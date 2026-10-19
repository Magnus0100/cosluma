# 安装与启动

桌面程序面向Windows x64。使用打包的EXE无需另装桌面Python；从源码运行或构建需要Python 3.12 x64。图像分析模型、GPU运行时和生成服务均独立安装，桌面包和源码仓库都不携带它们。模型由用户从对应的开源社区来源自行下载；仓库只保留公开模板、来源、revision、哈希和许可说明，详见[仓库内容与本地文件边界](REPOSITORY_LAYOUT.md)。

## Windows程序包

解压完整的 `cosplay-workbench` 目录，保留 `cosplay-workbench.exe` 和 `_internal` 的相对位置。移动或分享程序时复制整个目录，单独复制EXE会缺少Qt、原生库和资源。

双击 `cosplay-workbench.exe` 即可启动。配置默认位于用户主目录的 `CosplayWorkbench/config` 下，避免从打包宿主启动时 AppData 被 Windows 重定向成另一份文件视图。未配置时，程序在 `%USERPROFILE%\CosplayWorkbench\workspace` 创建可写工作区，并自动启动内嵌后台。

工程数据可独立放在容量充足的磁盘。在配置目录保存 `workspace-location.json`，内容为 `{"schema":"cosplay_workspace_location.v1","path":"D:/CosplayStudio/workspace-current"}`，其中路径必须是绝对目录。桌面入口和直接打开EXE都读取该选择；显式 `--root` 或 `COSPLAY_WORKSPACE` 优先。坏配置或不可写目录会报错，不静默打开另一工作区。此设置不会搬移或删除已有工程，也不改变安装目录。

需要指定自己的工作区时，在该目录打开PowerShell：

```powershell
$workspace = Join-Path $env:USERPROFILE 'CosplayWorkbench\workspace'
.\cosplay-workbench.exe --root "$workspace" --embedded-backend --auto-start-backend
```

`--root`指定素材、版本和输出所在的工程目录。EXE自动启动内嵌后台；源码运行时用 `--embedded-backend`和 `--auto-start-backend`启用同样行为。默认请求 `http://127.0.0.1:8787`，端口占用时自动分配可用端口；实际地址显示在“连接设置”。内嵌后台使用独占监听，每个窗口绑定自己的工程，不能通过修改地址连接其它工程。用 `--root`打开另一工程的新窗口。`--backend-url`可指定其它本机端口或端口 `0`（系统分配）。

打开界面后配置自己的模型及服务连接，导入原片，选择本地原片、已有参考或生成参考的输入模式并填写要求。三种模式使用同一生产管线，检查计划会列出本次启用节点的能力缺口、服务与预算，再开始制作。已有参考默认不新增图像生成；显式选择原景增强、本地Qwen及单区精修的新独立任务可获局部最多1次额度，全局参考仍为0。普通重试与最终修复不新增额度，见[原景增强](SOURCE_SCENE_ENHANCEMENT.md)。文本/视觉规划与外部审片默认关闭，启用时才需要对应服务。未启用外部服务的本地原片制作可离线处理。保存当前调整后才能导出可编辑工程；人工批准与自动执行结果分别记录。

当前副本没有公开Release下载地址。需要程序包时，可按 [开发指南](DEVELOPMENT.md) 从源码构建。

## 从源码运行

在仓库根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\scripts\requirements-desktop.txt
$workspace = Join-Path $env:USERPROFILE 'CosplayWorkbench\workspace'
.\.venv\Scripts\python.exe .\scripts\project_cli.py init "$workspace"
.\.venv\Scripts\python.exe .\scripts\qt_workbench.py --root "$workspace" --embedded-backend --auto-start-backend
```

固定依赖包括PySide6、NumPy/SciPy、Pillow、tifffile、rawpy、ONNX Runtime和PyInstaller。使用明确的Python路径，无需激活虚拟环境。桌面依赖安装成功只表示界面和基础原生处理环境可用。

也可双击桌面唯一的 `Cosplay Studio.lnk`。它调用安装目录自己的 `scripts/start_studio.ps1`，启动桌面下唯一的 `CosplayStudio/cosplay-workbench.exe`，先绑定内嵌后台，再在后台预热已验证的 Beauty 服务。启动器会校验 `install-manifest.json`、EXE SHA 和必需目录，并在工作区写入 `installation-health.json`。安装配置支持 `${PACKAGE_ROOT}` 解析包内 runtime/registry；外部模型根目录仍由健康回执明确列出。预热没有可用模型时不会阻止界面打开，首次任务会按需加载；缺失外部模型/runtime 会显示为 degraded，不会回退源码或旧程序。可通过 `COSPLAY_DESKTOP_EXE` 显式指定构建好的程序，通过 `COSPLAY_WORKSPACE` 指定工作区。源码与发行包共用同一生产管线，源码启动器不会自动选择仓库中的旧 `dist` 程序。

启动界面的“人像美化预热完成”只表示 Beauty 返回了实际加载完成回执；已有任务占用、失败或未知回执都不能视为预热成功。Qwen 的“配置已检查”不代表服务可连接或模型已加载。每个制作请求仍检查自身 task plan，预热失败不会授权缺失能力，也不阻止检查其他可用路线。

在仓库根目录运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_studio.ps1 -Describe` 可查看实际选择的程序、参数和工作区，无需启动界面。检查前先确认 `.env` 与 `.env.local` 中没有旧的 `COSPLAY_DESKTOP_EXE` 覆盖。

## 本地配置与模型

推荐把模型和运行环境放在软件主目录的 [`model` 文件夹](../model/README.md)，在主界面“模型管理”中点击“自动查找已有模型”。放在其他位置时，选择一次模型总目录即可查找安装配置。分析组件和 BeautyGRPO 按用途显示状态；缺权重、源码或 Python 时列出具体缺口。此检查不加载模型、不执行推理，实际处理仍核对固定身份与运行能力。处理任务未结束时不能切换配置。

主目录 `model/local-models.json` 与 `model/beautygrpo.json` 支持相对于配置文件的路径，完整安装移动后继续解析。界面导入的配置另保存用户副本，位于 `%USERPROFILE%/CosplayWorkbench/config`，之后双击 EXE 继续使用。显式任务配置及环境变量覆盖优先；用户手动选择的配置跨重启保留，未手动选择时优先使用主目录模型配置，再使用自动恢复的选择。启动器 `.env.local` 中已有覆盖会在下次启动时生效。公开模板作为最后默认，私人 JSON 不进入源码或发行包。

BeautyGRPO 的大文件完整核验和固定官方提示词预编码都属于安装操作。使用已配置的 BeautyGRPO Python 和 CUDA 运行时运行唯一安装入口：

```powershell
<BeautyGRPO Python> scripts/beautygrpo_prepare.py `
  --verify-installation <path-to-installation.json>
```

命令会在配置旁生成 `installation.verification.json`，记录 base 25 个文件、LoRA、配置和运行时身份，然后调用同一 prompt prepare 函数准备文本资源。对于 `installation.json`，缓存回执为同目录的 `installation.prompt-embeddings.json`，BF16 张量保存在 `installation.prompt-embeddings-<SHA256>.safetensors`。回执绑定当前配置及核验回执、官方 prompt、CLIP/T5 和 tokenizer 文件身份、运行库与原生 `encode_prompt` 源码、CUDA 设备及数学配置。预编码只执行官方文本编码器，不生成图片，不下载模型；这笔一次性成本不计入日常出图耗时。

已有完整核验回执的安装可用同一 prepare 流程补齐该资源，无需再次扫描模型大文件：

```powershell
<BeautyGRPO Python> scripts/beautygrpo_prepare.py `
  --prepare-prompt-embeddings <path-to-installation.json>
```

有效且匹配当前设备的缓存会复用。正式出图读取已核验的 `prompt_embeds` 和 `pooled_prompt_embeds`，省略 CLIP/T5 与 tokenizer 构造，继续使用官方 Kontext pipeline、scheduler 和 VAE，Transformer 与 LoRA 绑定由唯一 Nunchaku 后端执行。官方 prompt、BF16、LoRA scale 1、1024、28 步、guidance 2.5 与 seed 42 保持固定。日常运行只检查小回执/缓存内容及安装文件的存在性、大小和 `mtime_ns`，不重新逐模型计算 SHA。缺失或过期缓存明确要求修复安装资源，推理路径不隐式构造旧编码器；模型或配置变化需重新执行安装核验。

当前 Beauty 只有 `nunchaku_cuda` 执行后端。固定 Nunchaku 1.2.1、Torch 2.10 / CUDA 12.8 / Python 3.12 Windows ABI，使用官方 Kontext SVDQuant INT4 r32 权重和完整 Beauty LoRA。官方转换器已实证消费全部 684 个源张量，没有未处理目标；原提示、1024 / 28 步 / guidance 2.5 / seed 42 / LoRA 1 保持固定。旧 BF16 常驻块、CPU backing、暂停／恢复代码和专用测试已经移除。安装记录里的 sequential offload 仅为旧权重核验声明，运行参数只接受 nunchaku_cuda。空闲且本请求拥有的 Beauty 服务在下一 GPU 阶段前退出；需要时重新加载，忙碌或外部服务保持。实际保存、画质与速度见 [当前性能记录](WORKFLOW_PERFORMANCE.md)。

安装同 ABI 的[官方 Nunchaku wheel](https://github.com/nunchux-ai/nunchaku/releases/tag/v1.2.1)，并将固定[Kontext INT4 r32 权重](https://huggingface.co/nunchaku-ai/nunchaku-flux.1-kontext-dev/blob/main/README.md)放在安装配置同层的 `nunchaku/svdq-int4_r32-flux.1-kontext-dev.safetensors`。安装验证生成独立的量化权重 SHA／大小／mtime 回执；正式出图在参考提交前检查该回执，缺失或变化需要显式修复安装。运行时不自动下载或大文件重哈希。GPU 环境的 `pip check` 必须通过。


首次配置时从公共模板建立本地文件：

| 模板 | 本地配置 | 用途 |
|---|---|---|
| `providers/local-models.example.json` | 自己的模型注册表JSON | 分析模型、权重、源码与独立运行时路径 |
| `providers/qwen-image-2.1-models.example.json` | 公共权重清单；源码用户可建立本地副本 | 可选Qwen的来源、必需角色、大小和哈希 |

EXE的模板位于 `_internal/providers`。将模型注册表模板复制到安装目录之外，再填写自己的路径，通过 `COSPLAY_MODEL_REGISTRY`指定该JSON的绝对路径；不要直接编辑 `_internal`中的资源。源码用户也可使用未提交的 `providers/local-models.current.json`。

已有本地配置保留自己的设置。配置文件支持绝对路径和相对于该文件的路径，当前 registry 只接受受支持的 profile：Sapiens2 0.4B、ViTMatte、DINOv3 ViT-L/16 和 SAM3.1 静态 detector；其余当前能力路径是 BiRefNet/ONNX、LaMa、背景 SR（RealESRGAN checkpoint）和五官定位运行时。权重和源码身份须实际验证，填写路径或修改 `ready` 标记不能代替当前路线的安装预检。旧 profile、旧 registry、旧计划或旧 receipt 不会被升级为当前合同；仅接受当前合同，缺失字段不自动补齐。

### 按能力裁剪的模型清单

`scripts/task_model_preflight.py` 只检查当前任务实际启用节点的能力。未请求的组会显示 `not_requested`，缺失不会阻塞该任务。计划只读取安装结构、小文件与记录身份，不加载模型或重复完整权重 SHA 扫描。

| 输入模式/能力 | 当前生产预检模型与运行时 | 启用范围 |
|---|---|---|
| `local_portrait` 基础节点 → `source_silhouette` | BiRefNet ONNX、ONNX Runtime | 不消费参考、场景或边界，不隐式准备新的 Beauty 结果 |
| `existing_reference` / `create_reference` | Sapiens2 0.4B、ViTMatte、BiRefNet/ONNX、DINOv3 ViT-L/16、SAM3.1、LaMa、背景 SR、五官定位 Python/模型 | 当前任务计划固定 `complete_post=true`，执行同一完整后期 |
| 上述有参考任务的 `portrait_beauty`；或显式提交冻结美化层 | BeautyGRPO 独立安装回执、专用 Python 及配置的脸区解析/定位依赖 | 独立于分析模型 registry；已有绑定正确的冻结层可复用 |

唯一后期策略为 `layered_reference_transfer`；背景 SR 属于当前完整场景重建能力。有参考的当前任务自动纳入 BeautyGRPO 安装预检，公共请求不使用 `use_beautygrpo` 开关。Qwen/ComfyUI、GPT Image 只在生成参考时提供上游图像；文本/视觉规划与审片按任务服务策略启用，由各自 provider 健康检查管理。

安装仅使用当前 registry 中的模型。废弃模型和实验方案不属于本程序的数据格式、预检或回退路径。

例如，在PowerShell中设置外置模型注册表后启动程序：

```powershell
$modelRegistry = Join-Path $env:USERPROFILE 'CosplayWorkbench\config\local-models.current.json'
$env:COSPLAY_MODEL_REGISTRY = (Resolve-Path "$modelRegistry").Path
```

`providers/registry.json`是可公开的Provider定义，仅记录端点、模型及密钥环境变量名称。模型路径与工程目录分开。保存的冻结资产可用于重开和导出；重新分析准备需要本机实际安装的当前模型。下载和使用权重前，阅读 [第三方许可](THIRD_PARTY_NOTICES.md)。可选Qwen部署使用 `COSPLAY_QWEN_DEPLOY_ROOT`，安装与权重清单说明见 [QWEN_IMAGE_LOCAL.md](QWEN_IMAGE_LOCAL.md)。

## 服务与环境变量

在工作台连接设置中配置端点、模型和凭据来源。密钥由进程环境或已有凭据存储提供，公开配置只记录密钥环境变量名称。

| 变量 | 用途 |
|---|---|
| `COSPLAY_WORKSPACE` | 工作区；EXE和启动器默认 `%USERPROFILE%\CosplayWorkbench\workspace` |
| `COSPLAY_DESKTOP_EXE` | 显式覆盖启动器程序选择的EXE路径；路径缺失时停止 |
| `COSPLAY_BACKEND_URL` | 启动器使用的后台地址，默认 `http://127.0.0.1:8787` |
| `COSPLAY_MODEL_REGISTRY` | 外置模型注册表JSON的绝对路径 |
| `COSPLAY_PROVIDER_SETTINGS` | 自己的公开连接设置文件路径 |
| `COSPLAY_REFERENCE_PROVIDER` / `COSPLAY_REFERENCE_API_BASE` / `COSPLAY_REFERENCE_MODEL` | 参考图服务 |
| `COSPLAY_REFERENCE_API_KEY` | 参考图服务凭据 |
| `COSPLAY_VISION_PROVIDER` / `COSPLAY_VISION_API_BASE` / `COSPLAY_VISION_MODEL` | 研究、规划和文本/视觉审片服务 |
| `COSPLAY_VISION_API_KEY_ENV` | 文本/视觉服务使用的密钥环境变量名称 |
| `COSPLAY_QWEN_DEPLOY_ROOT` | 可选Qwen独立部署目录 |

可从根目录 `.env.example`复制为 `.env`或 `.env.local`。CMD启动器依次读取相邻的 `.env`和 `.env.local`，后者覆盖前者。文件使用字面量 `KEY=VALUE`；支持 `COSPLAY_`前缀变量及 `OPENAI_API_KEY`，不执行命令、不展开Shell变量。

直接运行Python或EXE时，先在进程环境中设置变量；它们不会自行读取这些环境文件。直接启动的后台地址使用 `--backend-url`指定。自己的环境文件、密钥、机器路径、照片和工程输出留在本地；公共模板不携带凭据。服务调用可能发送图片并产生费用，提交前查看实际连接、任务和预算。

## 常见问题

- 提示缺少Qt或DLL：重新解压完整程序目录，或按固定依赖重新建立源码虚拟环境。
- 提示模型不可用或身份不匹配：检查注册表的实际文件、运行时及受支持profile，保留具体缺口。
- 界面有原片但尚无高清草稿：先完成本地准备或参考准备，再保存；原片显示不等于原生母版已建立。
- 任务成功但作品仍待修复或待证据：查看开放问题和对应范围，完成必要的审片或有资格的恢复后再决定批准。

工作区不需要复制运行源码或模型权重。工程格式和导出精度见 [PORTABLE_EDITABLE_PROJECT.md](PORTABLE_EDITABLE_PROJECT.md)。
