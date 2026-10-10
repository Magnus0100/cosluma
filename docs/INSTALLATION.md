# 安装与启动

桌面程序面向Windows x64。使用打包的EXE无需另装桌面Python；从源码运行或构建需要Python 3.12 x64。图像分析模型、GPU运行时和生成服务均独立安装，桌面包和源码仓库都不携带它们。模型由用户从对应的开源社区来源自行下载；仓库只保留公开模板、来源、revision、哈希和许可说明，详见[仓库内容与本地文件边界](REPOSITORY_LAYOUT.md)。

## Windows程序包

解压完整的 `cosplay-workbench` 目录，保留 `cosplay-workbench.exe` 和 `_internal` 的相对位置。移动或分享程序时复制整个目录，单独复制EXE会缺少Qt、原生库和资源。

双击 `cosplay-workbench.exe` 即可启动。程序默认在 `%LOCALAPPDATA%\CosplayWorkbench\workspace` 创建可写工作区，并自动启动内嵌后台。

需要指定自己的工作区时，在该目录打开PowerShell：

```powershell
$workspace = Join-Path $env:LOCALAPPDATA 'CosplayWorkbench\workspace'
.\cosplay-workbench.exe --root "$workspace" --embedded-backend --auto-start-backend
```

`--root`指定素材、版本和输出所在的工程目录。EXE自动启动内嵌后台；源码运行时用 `--embedded-backend`和 `--auto-start-backend`启用同样行为。默认请求 `http://127.0.0.1:8787`，端口占用时自动分配可用端口；实际地址显示在“连接设置”。内嵌后台使用独占监听，每个窗口绑定自己的工程，不能通过修改地址连接其它工程。用 `--root`打开另一工程的新窗口。`--backend-url`可指定其它本机端口或端口 `0`（系统分配）。

打开界面后配置自己的模型及服务连接，导入原片，选择三种制作路线之一并填写要求。检查计划会列出具体缺口、服务与预算，再开始制作。已有参考直接执行完整后期，图像生成预算为零；仍需独立配置文本/视觉规划与审片服务。纯离线使用本地原片精修路线。保存当前调整后才能导出可编辑工程；人工批准与自动执行结果分别记录。

当前副本没有公开Release下载地址。需要程序包时，可按 [开发指南](DEVELOPMENT.md) 从源码构建。

## 从源码运行

在仓库根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\scripts\requirements-desktop.txt
$workspace = Join-Path $env:LOCALAPPDATA 'CosplayWorkbench\workspace'
.\.venv\Scripts\python.exe .\scripts\project_cli.py init "$workspace"
.\.venv\Scripts\python.exe .\scripts\qt_workbench.py --root "$workspace" --embedded-backend --auto-start-backend
```

固定依赖包括PySide6、NumPy/SciPy、Pillow、tifffile、rawpy、ONNX Runtime和PyInstaller。使用明确的Python路径，无需激活虚拟环境。桌面依赖安装成功只表示界面和基础原生处理环境可用。

也可双击仓库根目录的 `Start-Cosplay-Studio.cmd`。它调用 `scripts/start_studio.ps1`，优先使用根目录或 `dist/cosplay-workbench` 下的EXE，然后寻找 `.venv`或 `.cache/desktop-runtime` 中的桌面Python。可通过 `COSPLAY_DESKTOP_EXE`指定程序，通过 `COSPLAY_WORKSPACE`指定工作区。

## 本地配置与模型

推荐在“连接设置 → 配置分析模型”中导入已有模型配置，或选择每项已安装路径后点击“保存并使用”。配置保存在 `%LOCALAPPDATA%/CosplayWorkbench/config`，之后双击EXE即可继续使用，无需设置环境变量。软件逐项显示路径是否存在；缺权重、源码或Python时列出具体缺口。此检查不加载模型、不执行推理，实际处理仍核对固定身份与运行能力。处理任务未结束时不能切换配置。

界面选择会在当前进程生效，并记录为用户默认配置。显式任务配置及 `COSPLAY_MODEL_REGISTRY`优先于记住的默认选择；启动器 `.env.local`中已有覆盖仍会在下次启动时生效。公开模板作为最后默认，私人JSON不进入源码或发行包。

首次配置时从公共模板建立本地文件：

| 模板 | 本地配置 | 用途 |
|---|---|---|
| `providers/local-models.example.json` | 自己的模型注册表JSON | 分析模型、权重、源码与独立运行时路径 |
| `providers/qwen-image-2.1-models.example.json` | 公共权重清单；源码用户可建立本地副本 | 可选Qwen的来源、必需角色、大小和哈希 |

EXE的模板位于 `_internal/providers`。将模型注册表模板复制到安装目录之外，再填写自己的路径，通过 `COSPLAY_MODEL_REGISTRY`指定该JSON的绝对路径；不要直接编辑 `_internal`中的资源。源码用户也可使用未提交的 `providers/local-models.current.json`。

已有本地配置保留自己的设置。模型注册表需要绝对路径，并使用当前受支持的profile：Sapiens2 0.4B、DINOv3 ViT-L/16、SAM3.1静态detector和ViTMatte。其余路径包含BiRefNet、LaMa、背景超分、五官定位及相应运行时。权重和源码身份须实际验证，填写路径或修改ready标记不能代替安装验证。

例如，在PowerShell中设置外置模型注册表后启动程序：

```powershell
$modelRegistry = Join-Path $env:LOCALAPPDATA 'CosplayWorkbench\config\local-models.current.json'
$env:COSPLAY_MODEL_REGISTRY = (Resolve-Path "$modelRegistry").Path
```

`providers/registry.json`是可公开的Provider定义，仅记录端点、模型及密钥环境变量名称。模型路径与工程目录分开。保存的冻结资产可用于重开和导出；重新分析准备需要本机实际安装的当前模型。下载和使用权重前，阅读 [第三方许可](THIRD_PARTY_NOTICES.md)。可选Qwen部署使用 `COSPLAY_QWEN_DEPLOY_ROOT`，安装与权重清单说明见 [QWEN_IMAGE_LOCAL.md](QWEN_IMAGE_LOCAL.md)。

## 服务与环境变量

在工作台连接设置中配置端点、模型和凭据来源。密钥由进程环境或已有凭据存储提供，公开配置只记录密钥环境变量名称。

| 变量 | 用途 |
|---|---|
| `COSPLAY_WORKSPACE` | 工作区；EXE和启动器默认 `%LOCALAPPDATA%\CosplayWorkbench\workspace` |
| `COSPLAY_DESKTOP_EXE` | 启动器使用的EXE路径 |
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
