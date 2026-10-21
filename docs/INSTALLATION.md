# 安装与启动

桌面程序面向Windows x64。使用打包的EXE无需另装桌面Python；从源码运行或构建需要Python 3.12 x64。图像分析模型、GPU运行时和生成服务均独立安装，桌面包和源码仓库都不携带它们。模型由用户从对应的开源社区来源自行下载；仓库只保留公开模板、来源、revision、哈希和许可说明，详见[仓库内容与本地文件边界](REPOSITORY_LAYOUT.md)。

## Windows程序包

解压完整的 `cosplay-workbench` 目录，保留 `cosplay-workbench.exe` 和 `_internal` 的相对位置。移动或分享程序时复制整个目录，单独复制EXE会缺少Qt、原生库和资源。

双击 `cosplay-workbench.exe` 即可启动。配置默认位于用户主目录的 `CosplayWorkbench/config` 下，避免从打包宿主启动时 AppData 被 Windows 重定向成另一份文件视图。未配置时，程序在 `%USERPROFILE%\CosplayWorkbench\workspace` 创建可写工作区，并自动启动内嵌后台。

工程数据可独立放在容量充足的磁盘。在配置目录保存 `workspace-location.json`，内容为 `{"schema":"cosplay_workspace_location.v1","path":"D:/CosLumaProjects/MyProject"}`，其中路径必须是绝对目录。桌面入口和直接打开EXE都读取该选择；显式 `--root` 或 `COSPLAY_WORKSPACE` 优先。坏配置或不可写目录会报错，不静默打开另一工作区。此设置不会搬移或删除已有工程，也不改变安装目录。

需要指定自己的工作区时，在该目录打开PowerShell：

```powershell
$workspace = Join-Path $env:USERPROFILE 'CosplayWorkbench\workspace'
.\cosplay-workbench.exe --root "$workspace" --embedded-backend --auto-start-backend
```

`--root`指定素材、版本和输出所在的工程目录。EXE自动启动内嵌后台；源码运行时用 `--embedded-backend`和 `--auto-start-backend`启用同样行为。默认请求 `http://127.0.0.1:8787`，端口占用时自动分配可用端口；实际地址显示在“连接设置”。内嵌后台使用独占监听，每个窗口绑定自己的工程，不能通过修改地址连接其它工程。用 `--root`打开另一工程的新窗口。`--backend-url`可指定其它本机端口或端口 `0`（系统分配）。

打开界面后配置模型与服务，导入原片并填写角色和创作要求。默认本地 Qwen 半合成、鲜明风格，保留场地室内外关系；精修、大合成和风格分别选择。所有入口都先检查任务计划再开始。新建半合成／大合成必须观察原图，默认独立审片与最多两步通常改进及一轮零生图收尾开启；单候选从原图原生基线出发，预算上限为 3 图像、10 视觉；模型选择首次局部或整体编辑，实际用量以回执为准。用户可关闭条件纠偏或独立审片，计划如实列出外发内容和预算。`local_portrait` 本地原片路线不外发；使用模型编辑的精修范围按检查计划确认外发内容。普通已有参考默认新增图像为 0，其显式局部任务和冻结工具见[原景编辑](SOURCE_SCENE_ENHANCEMENT.md)。保存后才能导出工程；自动执行、自动评价和人工批准分别记录。

独立“研究角色并准备提示词”和旧自动数值光色规划入口已删除。当前原图观察与设计在已检查任务内执行；显式角色识别仍须当前计划与外发许可。保存的实际提示词和设计只读展示，不注入下一任务，也不因此额外研究或生图。已有参考可显式启用成对观察，仍不授予新图像额度。

Windows 预览版程序和对应源码见 [GitHub Release](https://github.com/Magnus0100/cosluma/releases/tag/v0.7.0)。普通用户下载 `windows-x64.zip`，完整解压后双击 EXE；开发者可克隆源码并按[开发指南](DEVELOPMENT.md)构建。程序包不包含 AI 模型和模型专用运行环境。

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

也可通过安装时创建的桌面快捷方式启动。它调用安装目录自己的 `scripts/start_studio.ps1`，校验安装后只启动对应 `cosplay-workbench.exe`。Qt 绑定内嵌后台、恢复保存的制作方式和 provider，再统一初始化实际选择的受管服务；启动器不另起固定 Qwen 版本。`install-manifest.json`、EXE SHA 和必需目录会被检查，工作区写入 `installation-health.json`。安装配置支持 `${PACKAGE_ROOT}` 解析包内 runtime/registry；外部模型根目录由健康回执明确列出。缺失外部模型/runtime 显示为 degraded，不回退源码或旧程序。`COSPLAY_DESKTOP_EXE` 可显式选择已构建程序，`COSPLAY_WORKSPACE` 可选择工程；源码启动器不自动选择仓库旧 `dist`。

默认 scoped Qwen 不预热未使用的 Beauty；明确人像操作仍按需加载和复用 Beauty。Qwen 的配置检查、服务连接与权重加载分开：现有无图初始化只准备服务，不证明模型已温热，首次生成可能仍需加载。每个制作请求继续核验 task plan，初始化失败不授权缺失能力，也不阻止检查其他可用路线。

在仓库根目录运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_studio.ps1 -Describe` 可查看实际选择的程序、参数和工作区，无需启动界面。检查前先确认 `.env` 与 `.env.local` 中没有旧的 `COSPLAY_DESKTOP_EXE` 覆盖。

## 本地配置与模型

推荐把模型和运行环境放在软件主目录的 [`model` 文件夹](../model/README.md)，在主界面“模型管理”中点击“自动查找已有模型”。放在其他位置时，选择一次模型总目录即可查找安装配置。分析组件和 BeautyGRPO 按用途显示状态；缺权重、源码或 Python 时列出具体缺口。此检查不加载模型、不执行推理，实际处理仍核对固定身份与运行能力。处理任务未结束时不能切换配置。

主目录 `model/local-models.json` 与 `model/beautygrpo.json` 支持相对于配置文件的路径，完整安装移动后继续解析。界面导入的配置另保存用户副本，位于 `%USERPROFILE%/CosplayWorkbench/config`，之后双击 EXE 继续使用。显式任务配置及环境变量覆盖优先；用户手动选择的配置跨重启保留，未手动选择时优先使用主目录模型配置，再使用自动恢复的选择。启动器 `.env.local` 中已有覆盖会在下次启动时生效。公开模板作为最后默认，私人 JSON 不进入源码或发行包。

BeautyGRPO 的大文件完整核验和固定官方提示词预编码都属于安装操作。使用已配置的 BeautyGRPO Python 和 CUDA 运行时运行唯一安装入口：

```powershell
<BeautyGRPO Python> scripts/beautygrpo_prepare.py `
  --verify-installation <path-to-installation.json>
```

命令会在配置旁生成 `installation.verification.json`，记录实际需要的 base 组件、LoRA、配置和运行时身份，然后调用同一 prompt prepare 函数准备文本资源。原始 25 文件下载清单保留为来源证据；唯一 Nunchaku 后端替代的三个 BF16 transformer 分片不再要求保留，也不进入当前组件核验清单。新版 `beautygrpo_verification.v2` 必须由此命令重新生成，不接受修改旧回执来宣称通过。对于 `installation.json`，缓存回执为同目录的 `installation.prompt-embeddings.json`，BF16 张量保存在 `installation.prompt-embeddings-<SHA256>.safetensors`。回执绑定当前配置及核验回执、官方 prompt、CLIP/T5 和 tokenizer 文件身份、运行库与原生 `encode_prompt` 源码、CUDA 设备及数学配置。预编码只执行官方文本编码器，不生成图片，不下载模型；这笔一次性成本不计入日常出图耗时。

已有完整核验回执的安装可用同一 prepare 流程补齐该资源，无需再次扫描模型大文件：

```powershell
<BeautyGRPO Python> scripts/beautygrpo_prepare.py `
  --prepare-prompt-embeddings <path-to-installation.json>
```

有效且匹配当前设备的缓存会复用。正式出图读取已核验的 `prompt_embeds` 和 `pooled_prompt_embeds`，省略 CLIP/T5 与 tokenizer 构造，继续使用官方 Kontext pipeline、scheduler 和 VAE，Transformer 与 LoRA 绑定由唯一 Nunchaku 后端执行。官方 prompt、BF16、LoRA scale 1、1024、28 步、guidance 2.5 与 seed 42 保持固定。日常运行只检查小回执/缓存内容及安装文件的存在性、大小和 `mtime_ns`，不重新逐模型计算 SHA。缺失或过期缓存明确要求修复安装资源，推理路径不隐式构造旧编码器；模型或配置变化需重新执行安装核验。

当前 Beauty 只有 `nunchaku_cuda` 执行后端。固定 Nunchaku 1.2.1、Torch 2.10 / CUDA 12.8 / Python 3.12 Windows ABI，使用官方 Kontext SVDQuant INT4 r32 权重和完整 Beauty LoRA。安装核验须确认量化权重和完整 LoRA；原提示、1024 / 28 步 / guidance 2.5 / seed 42 / LoRA 1 保持固定。旧 BF16 常驻块、CPU backing、暂停／恢复代码和专用测试已经移除。安装记录里的 sequential offload 仅为旧权重核验声明，运行参数只接受 nunchaku_cuda。空闲且本请求拥有的 Beauty 服务在下一 GPU 阶段前退出；需要时重新加载，忙碌或外部服务保持。计时口径见[流程性能](WORKFLOW_PERFORMANCE.md)，画质和保存按当前任务分别验收。

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
| 普通 `existing_reference` 原生工具 | Sapiens2 0.4B、ViTMatte、BiRefNet/ONNX、DINOv3 ViT-L/16、SAM3.1、LaMa、背景 SR、五官定位 Python/模型 | 当前任务计划固定 `complete_post=true`，执行同一完整后期 |
| 普通已有参考的 `portrait_beauty`；或显式提交冻结美化层 | BeautyGRPO 独立安装回执、专用 Python 及配置的脸区解析/定位依赖 | 独立于分析模型 registry；已有绑定正确的冻结层可复用 |
| `create_reference` 半合成／大合成；已有成片显式改进 | 所选 Qwen/GPT 图像 provider、文本视觉服务、CPU 图像环境 | 不要求未使用的 Sapiens/SAM/Beauty 原生工具；执行使用获准的生成外观 |

独立塑形按已检查任务请求能力：脸部为 CPU 几何操作，需当前人脸定位环境，图像预算为零、视觉至多三次；身体为单区生成，图像至多一次、视觉至多三次。脸部与身体组合目前拒绝。关闭恢复只使用已拥有的母版，图像、视觉、定位和 RAW 准备全零；完整合同见[可撤销形体调整](REVERSIBLE_SHAPING.md)。

`layered_reference_transfer` 是现有协调策略标识，节点由检查计划的真实制作范围决定。普通已有参考的完整原生工具才检查分析与 Beauty 能力；新建范围任务不先做固定 Beauty、不以旧抠图盖回生成受光。Qwen/GPT 与文本视觉配置和连接状态分别显示。

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
