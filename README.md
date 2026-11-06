# CosLuma

当前版本：**0.9.2**，Windows x64；源码使用 Python 3.12。

给原片和角色，自动设计 COS 后期，并将原片整头融合回生成画面，保留本人五官、脸型和相机细节。

## 首次使用

Windows 发布包包含 Python、桌面与图像处理依赖，解压后打开 `CosLuma/CosLuma.exe` 即可启动。
**首次出片需要配置模型服务；软件不附带模型服务、权重、账号或密钥。**
通用提示词库随包提供，保存首次配置时复制到用户配置目录；无需先寻找其他提示词库。

1. 打开桌面顶部的“服务设置”。填写提示词服务和图像编辑服务的完整接口地址、模型名，
   以及整头分割、超分的 ComfyUI 地址和权重名。示例地址必须换成服务方实际提供的地址。
2. 凭据选择“环境变量名”或“密钥文件路径”，填写名称或已有文件路径；不要把密钥粘贴到名称栏。
   可在 Windows 的“环境变量”设置中创建 `COSLUMA_API_KEY`，值填写服务密钥，再重新打开软件；
   或将密钥保存为仓库外的 UTF-8 文本文件并选择该路径。服务不需要密钥时选择“无需密钥”。
3. 点击“保存配置”。默认保存到 `~/CosplayWorkbench/config/cosluma.json`，
   同目录的 `prompts.json` 是随附通用库的个人副本；已有库不会被覆盖。保存配置不发起模型调用。
4. 选择照片、填写角色、点击“开始制作”。补充要求可留空，完成结果按原片名整理到批次内的 `PNG` 和 `JPEG` 目录。

已配置过服务后，日常只需第 4 步。顶部“服务设置”可修改服务；“高级设置”可选择另一份配置或调整请求尺寸。
仅安装和启动桌面不代表远端模型已准备好；请求错误在实际失败步骤显示，并保留已经生成的文件。
这是非实时的照片批处理；使用远端服务时需要网络，照片与提示词会发送到所配置的服务。

## 制作流程

```text
原片 + 角色 + 提示词库
  → LLM 看代表原片与提示词库目录，确定整组场景、光色并选择参考
  → LLM 按单张构图写编辑提示词，同时标出原片一个主体头部范围
  → 图像模型生成完整画面
  → 必要时模型超分一次
  → SAM3.1 提取原片头部、头发和头饰
  → SIFT 对应点与整体等比对齐
  → 一次 MKL 光色匹配、浮点蒙版融合
  → 可选整图明度曲线
  → 保存 16 位 PNG 和便于查看的 JPEG
```

唯一单张入口是 `scripts/photo_pipeline.py` 的 `finish_photo`。
桌面与 CLI 的批量制作统一使用 `finish_batch`，默认四张并行，每张调用同一个单张入口一次。
HTTP 单张请求调用 `finish_photo`，多张请求调用 `finish_batch`。
生图是基于完整原片的整图编辑，身体、服装和环境使用生成结果，保留该画面的空间受光；
原片回贴区域包含脸、头发和头饰。提示词要求保持实物服装结构，生成模型仍可能改变衣物细节。
提示词要求保留人物位置、大小、姿态、原片头部几何与服装结构，使场景光向与原片头部相容。
清理干扰景物，特效依据单张照片的动作、视线、留白和情绪决定，允许不加。

这是离线高质量照片后期，没有轻量或实时要求。模块职责固定，实现可以替换，问题在对应模块内解决。
没有评分放行、审批、预算、自动重试、自动换模型或 fallback。真实错误显示在失败步骤，
保留已有生成文件，不把未完成结果标为完成。原片只读，每次制作另建目录。
照片、模型、运行环境、私人配置和凭据保存在仓库外。

## 配置与使用

日常只需选择照片、填写角色、点击开始制作。补充要求可留空；软件先依据角色、原片受光与实物妆造
确定一个整组摄影方向，再处理每张照片。场景、材质、光质和配色保持一致，具体光向、景别与特效按单张决定。
服务地址、模型与凭据来源可在顶部“服务设置”中编辑，配置文件路径与请求尺寸位于折叠的“高级设置”。
桌面会记住角色、保存目录、配置路径、请求长边和每个角色各自的补充要求；切换角色时分别恢复。
这些偏好保存在 `~/CosplayWorkbench/config/desktop.ini`，不保存凭据。

Windows 发布包解压后双击 `CosLuma.exe`，已包含桌面和图像处理依赖。
同一可执行文件可用 `CosLuma.exe --cli ...` 运行下方 CLI 参数，或用 `CosLuma.exe --server ...` 启动本地 HTTP。
模型服务与权重仍需另行准备；发布包不含个人服务地址或密钥，随附的提示词库可直接作为起点。
源码包使用下方 Python 启动方式，或双击 `Start-Cosplay-Studio.cmd` 使用配置中指定的已有 Python。

手动配置可将 [配置示例](providers/settings.example.json) 和
[通用提示词库](providers/prompts.example.json) 分别另存为
`~/CosplayWorkbench/config/cosluma.json` 与同目录的 `prompts.json`。
`COSLUMA_CONFIG` 或桌面 `--config` 可指定其他配置文件，并优先于记住的路径。
资源相对路径以配置文件目录为准；私有配置与提示词库保存在仓库外。

| 配置项 | 填写内容 |
| --- | --- |
| `prompt_library` | 提示词库 JSON 路径；相对配置文件解析 |
| `llm.url`、`llm.model` | 支持图片输入与 JSON 文本输出的 Chat Completions 兼容接口完整地址，以及服务提供的模型名 |
| `image.url`、`image.model` | 接受 multipart `image[]` 的图像编辑接口完整地址，以及服务提供的模型名 |
| `llm/image.api_key_env` | 保存密钥的环境变量名，不是密钥本身 |
| `llm/image.api_key_file` | 保存密钥的已有文本文件路径；与环境变量二选一 |
| `image.parameters` | 服务支持的编辑参数 JSON；`model`、`prompt`、`n` 由程序提供 |
| `head.url`、`head.model` | ComfyUI 根地址、`models/checkpoints` 中的 SAM3.1 权重名 |
| `sr.url`、`sr.model` | ComfyUI 根地址、`models/upscale_models` 中的超分权重名 |
| `workers` | 同时制作的照片数，默认 4 |
| `output_long_edge` | 请求长边，默认 6000；`null` 请求原片尺寸 |
| `python` | 仅源码启动器使用的已有 Python 完整路径；Windows 可执行包不使用此项 |

LLM 接口返回 `choices[0].message.content`，内容为 JSON；图像编辑接口返回 `data[0].b64_json` 或 `data[0].url`。
参数和模型名必须与实际服务匹配，程序不自动探测、切换模型或替换接口协议。

### 分割与超分服务

服务部署者在独立的 ComfyUI 安装中准备 GPU 与权重。客户端使用 ComfyUI 核心节点：
分割为 `LoadImage → CheckpointLoaderSimple → CLIPTextEncode → SAM3_Detect → MaskToImage → SaveImage`，
超分为 `LoadImage → UpscaleModelLoader → ImageUpscaleWithModel → SaveImage`。
需要包含 `SAM3_Detect` 的 ComfyUI 版本，不需要另装同名第三方分割节点。
ComfyUI 的安装入口与运行命令见[上游说明](https://github.com/Comfy-Org/ComfyUI#installing)。
分割节点实现见[上游源码](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_sam3.py)，
部署者需按上游要求准备权重格式；本项目补丁只修正文中所述投影适配。

1. 将 ComfyUI 能读取的 SAM3.1 图像分割权重放入服务的 `models/checkpoints`，
   超分权重放入 `models/upscale_models`。配置中的文件名必须与服务中的实际文件名一致。
2. 当前适配需要 SAM3 文本编码器输出 512 维投影。对尚未包含该修正的服务，在 ComfyUI 目录中运行
   `git apply <本项目目录>/providers/patches/comfyui-sam3-clip-projection.patch`，然后重启服务。
   如果上游已包含等价修正，不重复应用；补丁不适用时应对照当前服务源码处理，不自动替换实现。
3. 启动已有 ComfyUI 服务，将根地址填写到 `head.url` 和 `sr.url`；同一服务可以承担两个阶段。
   客户端使用 `/upload/image`、`/prompt`、`/history/<id>` 和 `/view`，不要求用户手工搭建工作流。

当前示例权重名表示服务约定，不会触发下载；权重格式需与服务加载器匹配。
设备、驱动和内存由服务部署者配置，本机 SIFT 与光色合成使用 OpenCV、NumPy 和 color-matcher。
服务准备属于首次安装，不是每张照片的检查、评分或放行步骤。

### 源码运行

使用已有 Python 3.12，不新建运行环境。另一台机器安装本机依赖时：

```powershell
python -m pip install -r scripts/requirements-desktop.txt
```

源码桌面也有同一个服务设置界面，首次保存会记录当前 Python 路径。
双击 `Start-Cosplay-Studio.cmd` 前，可在已保存的配置中填写 `python`，或设置 `COSLUMA_PYTHON`；
也可直接使用以下 Python 命令启动。凭据只在调用对应服务时读取，桌面设置不会读取或测试密钥。

```powershell
# 桌面
python -B scripts/qt_workbench.py --root <保存目录>

# 单张或文件夹
python -B scripts/project_cli.py --root <保存目录> stylize --source <原片或目录> --character <角色>

# 显式设置并行照片数；也可在配置中设置 workers
python -B scripts/project_cli.py --root <保存目录> stylize --source <原片目录> --character <角色> --workers 4

# 使用已有生成图，完成主体头部定位、必要超分与整头融合
python -B scripts/project_cli.py --root <保存目录> compose --source <原片> --generated <生成图>

# 复用已有模型超分图
python -B scripts/project_cli.py --root <保存目录> compose --source <原片> --generated <原生生成图> --superresolved <模型超分图>

# 本地 HTTP
python -B scripts/backend_server.py --root <保存目录>
```

`compose` 仍调用一次 LLM 确定主体头部，跳过生图；提供 `--superresolved` 时也跳过超分。
可同时提供 `--character` 和 `--brief`。这些是显式复用阶段结果，后续仍走同一制作链。
HTTP 提供 `POST /api/stylize`、`POST /api/compose`、`GET /api/jobs/<id>`，默认监听 `127.0.0.1:8765`。
两种 POST 都接受 `source`、`character`、可选的 `brief`、`size`；`stylize` 可用 `sources` 路径数组
代替 `source` 批量制作，共享整组方向。`compose` 另外接受必填的 `generated` 和可选的 `superresolved`，
与 CLI 的复用行为一致。

例如，启动 HTTP 后从另一个 PowerShell 窗口提交任务并读取进度：

```powershell
$job = Invoke-RestMethod http://127.0.0.1:8765/api/stylize -Method Post -ContentType 'application/json' -Body (@{
  source = 'D:/photos/original.jpg'
  character = '角色名'
} | ConvertTo-Json)
Invoke-RestMethod "http://127.0.0.1:8765/api/jobs/$($job.id)"
```

任务状态保存在内存，制作结果保存在磁盘。批量制作另建一个批次目录，`PNG` 与 `JPEG` 子目录
按原片文件名整理完成的图片，桌面“打开保存目录”直接打开该批次；每张工作文件位于各自子目录。
不同目录的原片重名时，成片使用各自工作目录名区分，不覆盖已经保存的文件。
整理成片先写临时文件，写入完成后才改为正式文件名；写入失败会清理本次临时文件，工作目录内的完整成片仍保留。
`batch.json` 记录当前批次的方向、结果与失败原因。整组方向只设计一次；这一步失败时显示原始错误并停止该批次。
单张失败保留该张错误，其他照片继续处理；
批次存在失败时，桌面显示“批次未全部完成”，CLI 返回非零退出码，不自动重试或补入替代图。

## 当前行为与限制

原片通过 Qt 或 rawpy/LibRaw 解码为浮点 sRGB，校正朝向。生图发送完整画面的 sRGB JPEG，
默认输入长边最多 1536 像素，`image.input_max_edge` 可调整。只做一次整图编辑，不追加独立背景生成。
最终回贴素材从完整分辨率原片直接采样。

超分只运行一次配置模型，之后按目标比例裁切或下采样。默认请求长边 6000；`output_long_edge` 可改长边，
设为 `null` 时请求原片尺寸。
CLI `--size WIDTH HEIGHT` 可指定画布。一次返回不足时，使用实际覆盖范围并记录尺寸，
不循环超分或通过 RGB 插值扩大交付尺寸。

LLM 只选一个主体头部，不选择水面、屏幕或照片里的次要人像；镜中正脸是作品主体时可以选择镜像。
这仍是模型判断，无法保证所有照片都选对。SAM3.1 在选定的原片裁片中分别识别头部、头发和头饰，
使用同次执行返回蒙版的并集；不是精细透明度抠图。分析输入默认长边最多 2048 像素，
远端 ComfyUI 的 SAM3 节点内部使用 1008×1008 输入。当前服务的 SAM3 CLIP 投影层需要匹配
权重的 512 维输出，适配修正见 [补丁](providers/patches/comfyui-sam3-clip-projection.patch)。

SIFT 在原片头部和生成图对应范围寻找纹理对应点，OpenCV 拟合等比缩放、旋转和平移。
仅将这一个变换用于完整分辨率原片，不通过局部形变或非均匀缩放塑脸，分析预览不作为回贴素材。
纹理不足或头部姿态明显变化时，对齐可能失败或偏移；模型蒙版与纹理对应不保证每张都正确。

原片 RGB 与蒙版以预乘透明度方式使用同一个 Lanczos 变换采样，避免裁片边界的黑色或背景颜色渗入头部边缘。
color-matcher 的 MKL 在蒙版内部拟合一次光色变换，随后进行 float32 RGB 蒙版合成。
不模糊或生成式重绘脸部。MKL 不能重新创造方向不同的空间光照，
源头受光相容仍依赖生图模块。头发边缘和半透明头纱可能留下接缝或原背景颜色。

`finish_curve` 接受归一化的 `[输入明度, 输出明度]` 控制点，用单调三次插值整理整图明度，默认不调整。
纯黑像素按曲线的黑点映射为中性灰，其余像素保持 RGB 比例后裁切到可保存范围。
它不提供空间重打光；通透依靠明暗层次，不等于暗背景中的人物异常明亮。
原片与 MKL 计算保留浮点，合成结果保存为带 sRGB 配置的 16 位 PNG。
模型图片和分割蒙版在服务传输中为 8 位，16 位保存不会恢复上游缺失的精度，模型超分纹理也不等于相机原生细节。

每次制作保存摄影方向、提示词、模型返回、原片头部裁片、匹配头部、蒙版、`final.png`、`final.jpg` 和 `run.json`。
PNG 是 16 位成片，JPEG 是同一结果的 8 位查看副本。
提示词库是包含 `title`、`text`、`source` 的 JSON 数组，通用条目可标 `overview: true`，
整组设计时向 LLM 提供非通用条目的标题目录和最多六张均匀抽取的代表原片，由它选择具体参考。
单张提示词使用这些参考的正文和整组方向，不依赖角色名与条目的字面匹配。
库中其他角色的妆造、换脸、改衣、固定画幅和版式要求不应用于原片。
摄影方向与逐张判断仍依赖模型，不保证每张都符合用户审美，也不通过后期评分、重试或替代图掩盖失败。

| 职责 | 文件 |
| --- | --- |
| 制作顺序与保存 | `photo_pipeline.py` |
| LLM、生图、远端超分与分割 | `model_clients.py` |
| 整头对齐、MKL 与浮点融合 | `head_composite.py` |
| 图片读取与保存 | `image_io.py` |
| 提示词库目录与配置 | `prompt_library.py`、`settings.py` |
| 桌面、CLI、HTTP | `qt_workbench.py`、`project_cli.py`、`backend_server.py` |

照片与文本会发送到配置的 LLM 和图像编辑服务；生成图发送到超分服务，原片头部裁片发送到分割服务。
项目不商用、不盈利，依赖和权重仍遵循各自许可。项目许可见 [LICENSE](LICENSE)，
第三方许可见 [LICENSES](LICENSES/README.md)。

## 开发与发布

先阅读当前 README 与 [AGENTS.md](AGENTS.md)。`scripts/` 是业务代码，`providers/` 是公开配置、
通用提示词与当前适配补丁，`CosLuma.spec` 负责 Windows 打包，`LICENSES/` 只存许可和版权声明。
修改制作顺序从 `photo_pipeline.py` 入手，服务协议修改 `model_clients.py`，
身份区域和融合修改 `head_composite.py`，读取、位深与保存修改 `image_io.py`；
桌面首次配置界面位于 `qt_workbench.py`，配置路径与保存位于 `settings.py`。
生产工作文件中的 `run.json`、蒙版与模型返回用于实际结果追溯，保存在输出目录，不是仓库中的历史报告。

使用已有 Python 安装构建依赖并把临时产物放在仓库外：

```powershell
python -m pip install -r scripts/requirements-build.txt
python -m PyInstaller --noconfirm --distpath "$env:TEMP/CosLuma/dist" --workpath "$env:TEMP/CosLuma/build" CosLuma.spec
```

版本统一定义在 `scripts/app_version.py`，README 的当前版本随发布同步；
桌面、CLI 与 HTTP 启动命令均支持 `--version`。发布包需随附同版本源码与第三方许可。

GitHub Actions 在 Windows 上安装公开依赖、检查源码和浮点图像保存、构建发布目录，
并启动桌面与 HTTP 接口；成功后提供 Windows 包、同一提交的源码包和 SHA256 校验文件。
这些检查不调用付费模型，不验证审美或远端服务的可用性。仓库启用了 CodeQL、Dependabot 漏洞提醒、
密钥扫描和推送保护；制作过程仍按上面的单次链执行。

改动提交后使用该提交已有的 Actions 结果；成功产物为 Windows ZIP、同一提交源码 ZIP 与 `SHA256SUMS.txt`。
发布时确认 tag 指向该提交，上传这三个文件后再公开草稿，不使用旧包替换新提交的构建结果。
Actions 中的检查代码是当前发布流程的一部分，不随应用进入照片制作；不需要添加独立测试框架或历史检查报告。
GitHub 的 `run failed` 邮件对应邮件所链接的那次运行，查看其提交、失败步骤和时间；
修复后新提交成功不会改变旧运行的失败状态。Python 在 Actions 中固定主次版本 `3.12`，
使用 runner 实际提供的补丁版本；源码运行不要求同一个补丁版本。

欢迎通过 GitHub Issues 报告问题，通过 Pull Requests 提交修改。说明实际问题、修改后的行为和必要验证即可；
请勿上传私人照片、配置、模型权重或服务凭据。安全漏洞使用仓库 Security 页的私密报告入口。
