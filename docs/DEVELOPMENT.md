# 开发与Windows构建

使用Windows x64和Python 3.12。先按 [安装指南](INSTALLATION.md) 建立 `.venv` 并配置本地文件。桌面依赖位于 `scripts/requirements-desktop.txt`；GPU模型使用模型注册表指定的独立运行时。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `scripts/` | 桌面、后台、CLI、生产worker及开发工具 |
| `tests/` | 当前离线行为检查及临时工程fixture；不进入Windows运行包 |
| `scripts/runtime_manifest.json` | 发行包生产源码白名单 |
| `providers/*.example.json` | 可公开的配置模板 |
| `providers/workflows/`、`providers/patches/` | 服务工作流及适配补丁 |
| `presets/` | 风格预设 |
| `docs/` | 公共使用和开发文档 |
| `dist/`、`build/` | 本地构建产物 |

工程工作区、个人照片、模型、虚拟环境、运行输出和机器配置与源码分开，不进入公开提交。现有本机验证记录可以留在本地。

本地Git历史含私人工程及机器记录。公开仓库应从清洁源码导出建立新历史，并检查导出的文件、配置和许可；不要直接推送旧本地分支或标签。仅更新 `.gitignore`不能移除历史中已提交的内容。

导出使用 `prepare_source_release.py`，选择仓库之外尚不存在的目录：

```powershell
$sourceExport = Join-Path $env:TEMP 'cosplay-source-0.4.8'
.\.venv\Scripts\python.exe .\scripts\prepare_source_release.py --output "$sourceExport"
```

导出包含源码、公共模板、构建与依赖许可，并生成逐文件大小与SHA-256清单 `SOURCE_MANIFEST.json`。检查完成后再在导出目录建立公开Git历史；不要对含私人历史的本地仓库使用 `git archive`作为公开源码交付。二进制分发与对应源码要求见 [BINARY_LICENSES.md](BINARY_LICENSES.md)。

## 运行与CLI

```powershell
.\.venv\Scripts\python.exe .\scripts\project_cli.py --help
.\.venv\Scripts\python.exe .\scripts\project_cli.py one-click --help
.\.venv\Scripts\python.exe .\scripts\qt_workbench.py --help
```

用CLI建立新工作区，并读取状态：

```powershell
$workspace = Join-Path $env:LOCALAPPDATA 'CosplayWorkbench\workspace'
.\.venv\Scripts\python.exe .\scripts\project_cli.py init "$workspace"
.\.venv\Scripts\python.exe .\scripts\project_cli.py --root "$workspace" status
.\.venv\Scripts\python.exe .\scripts\project_cli.py --root "$workspace" recovery
```

产品任务使用 `task-plan`与 `task-start`。请求JSON包括原片、可选参考、当前模型注册表和 `creative_task`：

```json
{
  "source": "data/raw/portrait.jpg",
  "reference": "data/reference/design.png",
  "creative_task": {
    "schema_version": "cosplay_task_intent.v1",
    "mode": "existing_reference",
    "brief": "暗调氛围，保留本人妆造",
    "constraints": {"preserve_background": false, "no_effects": false},
    "services": {"text_vision": true}
  }
}
```

运行 `project_cli.py --root <工作区> task-plan --request-file <请求JSON>`，将返回的 `request`和 `plan.plan_id`写成执行请求，再用 `task-start --request-file <执行请求JSON>`同步执行。计划不联网、不创建任务；执行重新验证内容身份。三路线为 `local_portrait`、`existing_reference`、`create_reference`。已有参考路线图像生成预算全零；文本/视觉服务单独配置。

底层 `one-click`仍用于生产协调与开发：要求 `--source`，支持 `--reference`、`--model-registry`和 `--options-file`。`--local-only`是本地原片路线，`--complete-post`协调完整后期。`--final-repair-only`同时要求已有参考及 `--complete-post`，禁止生图，并要求完整已保存设计/资产。查看保存版本使用 `source-status <run-id>`；`source-accept`要求明确reviewer和人工checks。

桌面内嵌后台可使用安装指南命令。开发时也可分别启动后台与界面：

```powershell
.\.venv\Scripts\python.exe .\scripts\backend_server.py --root "$workspace" --host 127.0.0.1 --port 8787 --max-workers 1
```

另一个PowerShell窗口运行 `qt_workbench.py --root <工作区> --backend-url http://127.0.0.1:8787`。后台默认限制在本机，接口说明见 [REVISION_API.md](REVISION_API.md)。

## 构建EXE

从仓库根目录运行：

```powershell
.\scripts\package_windows.ps1 -Clean -PythonExecutable ".\.venv\Scripts\python.exe"
```

构建脚本先检查桌面依赖，再构建Debug与Release，并运行冻结Qt/native检查。环境检查要求Python 3.12 x64、依赖文件中全部精确版本，并核对许可证清单及已安装wheel中的许可字节。版本或许可不符时在构建前失败。默认只选择项目 `.venv`，其次项目 `.cache/desktop-runtime`；其它解释器须通过 `-PythonExecutable`指定。这保证依赖版本及许可一致性，不声明二进制逐字节可复现。输出分别为：

```text
dist/
├─ cosplay-workbench-debug/cosplay-workbench-debug.exe
└─ cosplay-workbench/cosplay-workbench.exe
```

每个目录同时包含 `_internal`；分发时保留整个Release目录。模型权重、模型运行时、凭据和个人工程另行配置。PyInstaller规格以 `scripts/runtime_manifest.json`选择生产源码，开发验证工具不应进入发行闭包。

| 构建参数 | 用途 |
|---|---|
| `-PythonExecutable` | 明确选择具有完整依赖的Python |
| `-Clean` | 清理PyInstaller缓存后构建 |
| `-DebugOnly` | 只构建和检查Debug |
| `-DistPath` / `-WorkPath` | 指定发行与构建目录 |
| `-SmokeRoot` | 指定冻结检查所用工作区 |
| `-SkipSmoke` | 跳过冻结检查；结果仍须单独验证后再交付 |

Debug带控制台，适合收集脱敏后的启动错误。Release不显示控制台。当前仓库没有自动发布远程Release的步骤。

用已核验的清洁源码目录和Release目录生成完整Windows包、源码包及校验清单：

```powershell
.\.venv\Scripts\python.exe .\scripts\assemble_windows_release.py --source "$sourceExport" --dist ".\dist\cosplay-workbench" --output "..\release-0.4.8"
```

输出目录须不存在。工具复用公开源码检查，核对运行源码、许可清单和源码快照，保留完整EXE依赖目录；源码ZIP排除 `.git`历史与Python缓存。打包后逐一核对ZIP成员、目录文件及大小／SHA-256，拒绝未登记文件，并生成发布说明和不含本机路径的公开验证摘要。发布资产与独立核验见 [RELEASING.md](RELEASING.md)。

`.github/workflows/ci.yml`在Windows上安装固定桌面依赖、检查公开资源并运行当前基础行为检查。手动运行 `prepare-release.yml` 会从清洁源码快照构建和检查Debug/Release，上传完整Windows包、对应源码和四份发布说明／清单。工作流产物不等于已经发布远程Release。

行为检查统一从 `tests/`运行，不再散落于生产脚本目录。常规CI执行 `python tests/run.py core`，按模块检查可使用 `python tests/run.py reference_appearance_stack boundary_light_coordination_worker`；修改共享导入或测试目录时可执行 `python tests/run.py all --model-python <已安装分析Python>`。桌面依赖和模型依赖不混装；检查使用临时工程，避免写入开发者的真实工作区。

## 当前开发工具

`full_cosplay_review.py`是source-only保存子版的审片CLI，支持`prepare/review/all`，调用现行`cosplay_native_review`与`cosplay_post_review`。它保护源图、父/子母版及工程选择的哈希，保存并验证API证据；不生图、不做GPU推理，也不自动批准作品。它属于开发工具，不进入发行runtime。使用前先查看`--help`，审片模式会使用配置的文本/视觉服务。

`check_raw_api_input.py --raw <原片RAW>`是手工RAW上传边界检查，通过拦截上传确认实际预处理输入；不使用网络或生成额度。它需要明确真实RAW，不属于自动fixture检查。

`install_local_dinov3.py`核验并安装用户提供的当前DINOv3下载；`install_sam3.py`核验当前SAM3.1安装。两者使用模型运行时并保留固定身份。`major_release_evaluation.py`汇总明确提供的多素材保存证据，不能代替人工成片批准。`verify_saved_native_delivery.py`核验保存工程的原生回放与导出；使用前查看 `--help`。

## 验证与修改边界

基础检查：

```powershell
git diff --check
.\.venv\Scripts\python.exe .\scripts\check_repository.py
```

`check_repository.py`已经对全部Python源做AST语法检查，并核验公开资源、runtime及许可；无需再重复`compileall`。

按改动选择行为检查：文档改动检查本地链接和实际入口；界面/后台改动检查状态、取消与恢复；生产者或合成改动检查当前冻结工程的原生回放、保存及相关导出，不以所有旧smoke累计数量作为能力成绩。

发布仍须保留桌面与内嵌后台实际启动、当前已保存冻结工程的原生回放/母版导出及portable重开检查，以及清洁源码身份和source/license gate。结果须记录具体输入、观察范围和限制；启动、数值一致性和迁移成功不授予图像质量通过。网络审片、生图和GPU推理分别记账。

新增生产模块时维护运行白名单，保持安装源码与工程数据分开。失败任务保留原片和父版本；新像素进入独立版本并绑定新证据。贡献说明见 [CONTRIBUTING.md](../CONTRIBUTING.md)，安全边界见 [SECURITY.md](../SECURITY.md)。
