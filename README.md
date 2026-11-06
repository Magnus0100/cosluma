# CosLuma

**0.10.0 · Windows x64 · 源码使用 Python 3.12**

提供照片和角色，按这组素材设计三个摄影方向，先看同一张原片的三个真实样片，选定后再制作整组 COS 摄影后期；原片整头融合回生成画面，保留本人五官、脸型和相机细节。

## 下载与首次使用

[下载 Windows 包](https://github.com/Magnus0100/cosluma/releases/download/v0.10.0/CosLuma-0.10.0-windows-x64.zip) · [Release 与校验文件](https://github.com/Magnus0100/cosluma/releases/tag/v0.10.0)

Windows 用户下载 Release 中的 `CosLuma-0.10.0-windows-x64.zip`，完整解压，打开 `CosLuma/CosLuma.exe`。
包内已有 Python、桌面和图像处理依赖，源码 ZIP 用于开发，不是可直接双击运行的 Windows 包。

**软件是模型服务的桌面客户端。首次出片需要你自己的图文提示词服务、图像编辑服务，以及已部署分割和超分模型的 ComfyUI。**
包内不含这些服务、权重或账号；使用远端服务时需要网络，照片与提示词会发送到配置的服务。

1. 已有服务配置时，点击顶部“导入已有配置”选择 JSON；软件使用原文件，不复制或覆盖它。
2. 没有配置时，点击“连接设置”。按顺序填写提示词服务、图像编辑服务、分割与超分服务。
   地址可填写到 `/v1`，保存时补全兼容接口路径；也可填写完整接口地址。模型名以实际服务为准。
   用“上一步／下一步”切换；两个接口由同一服务提供时，可直接复用提示词服务的地址和凭据，再填写图像模型名。
3. 服务密钥可直接粘贴到密码框。保存时写入用户配置目录内的独立密钥文件，配置 JSON 只记录路径。
   已有的环境变量和密钥文件可继续使用，编辑设置不会读取或显示原密钥。
4. 提示词库默认使用随附通用库，无需另找文件。保存设置后添加照片、填写角色，点击“按角色与原片设计样片”。
   在弹窗选择一张代表原片，点击“设计方向并生成三个样片”。对比后选择一个方向，返回主界面点击“按选定风格批量制作”。

设置只保存本机文件，不连接服务。模型调用在实际制作步骤中进行，错误会显示在对应步骤，不自动重试或换模型。
默认配置位于 `~/CosplayWorkbench/config/cosluma.json`；通用库个人副本是同目录的 `prompts.json`，已有库不会被覆盖。
**配置目录可能包含密钥文本，请勿公开或上传。** 凭据文件只在调用对应服务时读取，不写入制作记录。

## 日常制作

照片可多次添加、选择文件夹，或把照片与文件夹拖入窗口；重复路径只保留一张。
文件夹添加其直属照片，不递归导入其他子目录。选错照片可以移除选中或清空。
添加后会说明新增、重复和不支持的项目数量，并保留当前选择。

角色为必填，补充要求可以留空。候选方向在收到角色和照片后设计，没有固定风格菜单。
LLM 结合角色的可靠设定、原片妆造与道具、构图和受光决定方向；不了解的角色设定不编造，依据用户输入与可见素材设计。
提示词库提供摄影参考，不把库条目直接变成预设候选。桌面先把摄影方向交给用户选择：

1. 在照片列表选中一张代表原片，点击“按角色与原片设计样片”；弹窗显示当前角色，仍可更换这张原片。
2. 点击“设计方向并生成三个样片”。LLM 一次观察代表原片、角色、补充要求和提示词库，设计三个适合当前素材的不同方向；
   三张样片使用同一张原片，各自完整执行整图编辑、必要时一次超分、整头回贴和保存，不是滤镜示意或临时背景。
3. 三张缩略图同时展示进度；点击任一缩略图，可放大查看并与原片对比。方向名称与说明来自本次设计，说明画面、光感、氛围与适合这组素材的理由。
   样片会调用实际配置的模型；失败的方案显示真实错误并保留文件，不自动重试，也不能被当作完整样片选择。
4. 选择一个已完成方向。主界面显示当前选择，点击“按选定风格批量制作”后才提交其余原片。
   选中的完整样片直接复制收进本批次，不再生成一次；其余照片沿用该方向和具体参考，不再次规划风格。

后续 LLM 和生图均收到选中的真实样片作为风格参考；原片是第一张生图输入，样片是第二张输入。
本张的头部定位、人物身份、构图和姿态仍只依据本张原片，样片不参与身份回贴。
场景、材质、配色和光质保持一致，各张按实际受光、动作和情绪细化，特效按单张决定，允许不加。
样片三张并行；选定后的批量默认四张并行，其余每张只执行一次，单张失败不影响其他照片。
返回队列不会自动开始批量。修改角色或补充要求后重新生成并选择；同一要求下追加照片可沿用当前选择。
已选样片保留其制作时的实际尺寸，后续尺寸设置应用于新制作照片；如需改变样片本身尺寸，可明确重新生成。
返回后再打开会优先载入已有样片；重开软件也记住当前选择，查看或换选已有方向不调用模型。

左侧列表逐张显示实际制作阶段；下方进度是已处理张数，不是估算完成时间。
点击任一行可查看原片预览、成片或失败原因。完成后默认并排对比原片与成片，也可切换为单独查看；
可打开原片、选中成片或此照片的工作文件夹。分割前或整组方向设计失败时，也能打开本批次文件夹查看已保留文件。
RAW 不支持桌面预览时仍可进入制作，最终读取由 rawpy/LibRaw 完成；分析预览不参与整头回贴。

完成后添加新照片，主按钮变为“按选定风格制作新增 N 张”，已有成功与失败照片都不自动重做；同一要求下沿用当前选定方向。
“全部另做一批”会重新制作当前队列中的其他照片，代表原片仍复用选中的完整样片；原有结果另存保留。
“查看上次制作”直接读取上次批次文件夹中的记录，可在重开软件后浏览成片、失败原因与未完成状态；
该窗口不替换当前照片队列、不发起模型请求。完整移动批次文件夹后仍按其中的相对位置查找成片；
缺失的原片或成片会明确显示，不把未完成结果标为完成。
旁边的下拉菜单也可选择其他包含 `batch.json` 的批次文件夹，浏览更早或另存的结果。

“打开本次结果”定位当前批次。`PNG` 与 `JPEG` 子目录按原片名保存成片：
PNG 是带 sRGB 配置的 16 位结果，JPEG 是同一结果的 8 位查看副本。
不同目录原片重名时使用各自工作目录名区分，不覆盖已有文件；整理失败保留工作目录内的完整成片。
每张工作目录还保留提示词、模型返回、头部裁片与蒙版、`final.png`、`final.jpg`、`run.json`，批次概况位于 `batch.json`。
样片保存在独立的 `styles-*` 文件夹，包含 `styles.json`、三个 `style-N.json` 及各自完整工作目录；未选样片也保留。

桌面记住角色、每个角色的补充要求、保存目录、配置路径和请求尺寸，偏好保存于
`~/CosplayWorkbench/config/desktop.ini`。“更多设置”可选择其他配置或调整请求长边。
制作中不修改队列或输入；关闭窗口时继续在系统托盘制作，点击托盘图标或菜单可返回窗口。
保持托盘状态时完成后退出；返回窗口后保留完成界面。没有系统托盘时仅最小化窗口，制作继续。

## 配置与服务部署

桌面支持直接粘贴密钥、已有文件、环境变量和无需密钥四种方式；编辑已有配置默认保留其凭据。
新密钥写入配置目录中的独立 `.key` 文件；替换时仅清理由软件创建、且新配置不再引用的旧密钥文件。
凭据文本由当前 Windows 用户配置目录保护，软件不宣称提供额外的加密保险库。

手动配置可将 [配置示例](providers/settings.example.json) 和 [通用提示词库](providers/prompts.example.json)
另存为上述 `cosluma.json` 与 `prompts.json`，替换服务地址、模型名和凭据引用。
`COSLUMA_CONFIG` 或命令行 `--config` 优先于记住的路径；相对资源路径以配置文件目录为准。

| 配置 | 内容 |
| --- | --- |
| `llm.url`、`llm.model` | 支持图片输入的 Chat Completions 兼容接口完整地址、实际模型名 |
| `image.url`、`image.model` | 接受 multipart `image[]` 的图像编辑接口完整地址、实际模型名 |
| `llm/image.api_key_file` | 密钥文本文件路径；与 `api_key_env` 二选一 |
| `llm/image.api_key_env` | 保存密钥的环境变量名；进程启动后改变变量需要重新启动软件 |
| `image.parameters` | 服务支持的额外图像参数；界面提供质量、尺寸、格式，其他项可在 JSON 中设置 |
| `head.url`、`head.model` | ComfyUI 根地址、`models/checkpoints` 中的 SAM3.1 图像分割权重名 |
| `sr.url`、`sr.model` | ComfyUI 根地址、`models/upscale_models` 中的超分权重名 |
| `prompt_library` | 提示词 JSON 文件路径 |
| `workers` | 并行照片数，默认 4 |
| `output_long_edge` | 请求长边，默认 6000；`null` 请求原片尺寸 |
| `finish_curve` | 可选的归一化 `[输入明度, 输出明度]` 数组，默认不调整 |
| `python` | 源码启动器使用的已有 Python 路径；Windows 包不使用 |

### 接口约定

`llm.url` 保存完整请求地址，客户端发送非流式 `messages`，其中图片为 `image_url` 数据 URL。
服务返回 `choices[0].message.content`，文本须为模型按提示要求生成的 JSON。
`image.url` 接受 multipart `image[]` 和文本字段 `model`、`prompt`、`n=1` 及配置的额外参数，
返回 `data[0].b64_json` 或可下载的 `data[0].url`。样片之后的批量请求包含两个 `image[]`：原片在前、用户选定样片在后，服务须支持多图参考。
接口名称相似不代表兼容；程序不自动转换协议或更换模型。

### ComfyUI

在仓库外使用已有的 ComfyUI 环境准备 GPU 与权重，安装入口见[上游说明](https://github.com/Comfy-Org/ComfyUI#installing)。
服务须包含核心 `SAM3_Detect` 节点，客户端直接提交以下工作流，无需用户手工搭图：

| 阶段 | 节点链 |
| --- | --- |
| 整头分割 | LoadImage → CheckpointLoaderSimple → CLIPTextEncode → SAM3_Detect → MaskToImage → SaveImage |
| 超分 | LoadImage → UpscaleModelLoader → ImageUpscaleWithModel → SaveImage |

将服务加载器支持的 SAM3.1 图像权重放入 `models/checkpoints`，超分权重放入 `models/upscale_models`。
配置中的模型名须与服务文件名一致；示例名不会触发下载。分割节点实现见[上游源码](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_sam3.py)。
原始权重与转换格式的许可均随权重保留，不能只改后缀冒充转换。

当前分割适配需要 SAM3 文本编码器输出 512 维投影。对尚未包含等价修正的服务，在 ComfyUI 目录执行
`git apply <本项目目录>/providers/patches/comfyui-sam3-clip-projection.patch` 后重启；
上游已修正时不重复应用，补丁与服务源码不匹配时由部署者处理，不自动覆盖。
GPU、驱动与内存由服务部署者明确配置，客户端不自动切换设备。

将 ComfyUI 根地址填入 `head.url` 和 `sr.url`，同一服务可承担两个阶段。
客户端使用 `/upload/image`、`/prompt`、`/history/<id>` 和 `/view`；这些端点需从运行 CosLuma 的机器可达。
这里是服务提供方的安装约定，不是每张照片的评分或放行步骤。

### 自定义提示词库

库是非空 JSON 数组，每条包含 `title`、`text`、`source`。例如：

```json
[{"title":"自然窗光","text":"使用与原片相容的窗光和真实室内材质，保持人物与环境曝光自然连续。","source":"作者与来源说明"}]
```

`overview: true` 可标记通用条目。整组设计向 LLM 提供非通用标题目录和最多六张均匀抽取的代表原片，
由它选择具体参考；目录不限制可设计的场景范围，也不把标题直接凑成三个候选。
单张使用选中正文和整组方向，不依赖角色名检索。外部库的来源与许可随库保留。

## 源码、CLI 与 HTTP

使用已有 Python 3.12，在仓库根目录安装本机依赖，不新建运行环境：

```powershell
python -m pip install -r scripts/requirements-desktop.txt
python -B scripts/qt_workbench.py
```

源码桌面提供相同的连接设置。首次保存新配置时记录当前 Python 路径；
以后可用 `Start-Cosplay-Studio.cmd` 启动，也可用 `COSLUMA_PYTHON` 或启动器 `-Python` 指定已有解释器。
Windows 包无需这些步骤，同一 `CosLuma.exe` 使用 `--cli` 或 `--server` 进入下列模式。

```powershell
# 先为同一张代表原片生成三个真实风格样片
python -B scripts/project_cli.py --root D:/CosLuma-output styles --source D:/photos --sample D:/photos/a.jpg --character "角色名"

# 选中一个已完成的 style-N.json，再制作整组；完整样片直接复用
python -B scripts/project_cli.py --root D:/CosLuma-output stylize --source D:/photos --character "角色名" --style D:/CosLuma-output/styles-时间与编号/style-2.json

# 显式直接调用：不传 --style 时自动设计一个方向；默认四张并行
python -B scripts/project_cli.py --root D:/CosLuma-output stylize --source D:/photos --character "角色名"

# 调整并行数、补充要求与请求尺寸
python -B scripts/project_cli.py --root D:/CosLuma-output stylize --source D:/photos --character "角色名" --workers 4 --brief "保留原有道具" --size 4000 6000

# 复用已有生成图；仍调用一次 LLM 定位主体头部
python -B scripts/project_cli.py --root D:/CosLuma-output compose --source D:/photos/a.jpg --generated D:/renders/a.png

# 另有已保存的模型超分图时，显式复用该结果
python -B scripts/project_cli.py --root D:/CosLuma-output compose --source D:/photos/a.jpg --generated D:/renders/a.png --superresolved D:/renders/a-sr.png

# 本地 HTTP，默认仅监听 127.0.0.1:8765
python -B scripts/backend_server.py --root D:/CosLuma-output
```

CLI 的 `--config <路径>` 放在子命令前；桌面、CLI、HTTP 均支持 `--version`。
HTTP 提供 `POST /api/styles`、`POST /api/stylize`、`POST /api/compose`、`GET /api/jobs/<id>` 和 `GET /api/status`。
`styles` 接受 `sources`、`sample_source`、`character`，可选 `brief` 和 `size`；完成后返回三个方案及各自 `selection_file`。
`stylize` 接受 `source` 或批量 `sources`、`character`，可选 `brief`、`size` 和选中的 `style` 文件路径；
使用选中方案时保持相同角色与补充要求，复用样片保留其原有实际尺寸。不传 `style` 的直接调用仍自动设计一个方向。
`compose` 另需 `generated`，可选 `superresolved`，角色可以留空。

```powershell
$job = Invoke-RestMethod http://127.0.0.1:8765/api/stylize -Method Post -ContentType 'application/json' -Body (@{
  sources = @('D:/photos/a.jpg', 'D:/photos/b.jpg')
  character = '角色名'
} | ConvertTo-Json)
Invoke-RestMethod "http://127.0.0.1:8765/api/jobs/$($job.id)"
```

提交返回 HTTP 202 和 `id`；查询的状态为 `running`、`complete`、`unfinished` 或 `failed`，失败附 `error`。
状态只保存在进程内，重启后不可查询，磁盘成片不受影响。
单张失败不自动重试；批次未全部完成时桌面明确提示，CLI 返回非零退出码。

## 制作链与限制

```text
原片 + 角色 + 提示词库 → 一次设计三个方向 → 同一原片的三个真实样片 → 用户选择
→ 复用选中样片；其余按选中方向逐张写提示词与主体整头范围
→ 整图编辑 → 必要时一次模型超分 → SAM3.1 原片整头/头发/头饰并集
→ SIFT 整体等比对齐 → 一次 MKL 浮点光色匹配与蒙版融合
→ 可选整图明度曲线 → 16 位 PNG 与 JPEG
```

唯一单张入口是 `scripts/photo_pipeline.py` 的 `finish_photo`。桌面与 CLI 批量使用 `finish_batch`，
HTTP 单张调用前者，批量调用后者；没有并列的贴脸、全身覆盖、单独背景或人工替代路径。
样片仍逐张使用同一个 `finish_photo`，风格选择属于用户创作决定，不是成片质量审批。
选中方向不再次规划，单张据构图和原片受光细化。生成身体、实物服装和环境保留整体空间受光，
只将原片脸、头发、头饰融合回去。提示词要求保持服装结构，模型仍可能改变衣物细节。

| 环节 | 当前行为与实际限制 |
| --- | --- |
| 原片 | Qt 或 rawpy/LibRaw 解码为浮点 sRGB 并校正朝向；原片只读，成片另存 |
| 生图 | 发送完整画面的 sRGB JPEG，默认输入长边最多 1536，`image.input_max_edge` 可调整；只整图编辑一次 |
| 超分与尺寸 | 仅一次配置模型，之后只裁切或下采样；覆盖不足使用实际范围并记录，不循环超分或插值扩大交付尺寸 |
| 头部定位与分割 | LLM 选择一个主体头部；SAM3.1 在原片裁片识别头部、头发、头饰并取同次返回并集；分析默认长边最多 2048，节点内部 1008×1008；不是精细透明度抠图 |
| 几何 | SIFT 纹理匹配后只拟合一个等比缩放、旋转和平移，不局部变形或非均匀塑脸；纹理不足、头部姿态改变时可能失败或偏移 |
| 采样与融合 | 原片 RGB 与蒙版预乘后使用同一 Lanczos 变换采样一次；蒙版内部一次 MKL 匹配，float32 RGB 合成；不模糊或生成式重绘脸部 |
| 光与边界 | MKL 不能重造方向不同的空间光，源头受光相容依赖生图；头发和半透明头纱可能留下接缝或背景色 |
| 明度曲线 | 单调三次插值调整整图明度，默认关闭；纯黑按黑点映射中性灰，其余保持 RGB 比例；不提供空间重打光 |
| 精度 | 原片与 MKL 保持浮点，保存带 sRGB 配置的 16 位 PNG；服务图片和分割蒙版传输为 8 位，16 位保存不能恢复丢失精度，超分纹理不等于相机原生细节 |

LLM 不保证每张都定位正确或符合审美；镜中清晰正脸是主体时可以选择镜像，其他倒影与次要人像不应被选中。
参考库中的其他角色妆造、换脸、改衣和固定版式不应用于原片。
通透来自合理的明暗层次，不能靠暗背景中异常明亮的人物或整图染色冒充光感。
没有评分、审批、预算、能力门禁、自动重试、换模型或失败 fallback；真实错误在失败步骤显示，已生成文件保留，未完成结果不标成完成。

## 开发与发布

先读当前 README 和 [AGENTS.md](AGENTS.md)，再按问题修改对应模块：

| 职责 | 文件 |
| --- | --- |
| 唯一制作顺序、批量与保存目录 | `scripts/photo_pipeline.py` |
| 提示词、生图、ComfyUI 请求 | `scripts/model_clients.py` |
| 原片身份提取、几何与局部融合 | `scripts/head_composite.py` |
| 图片解码、浮点数据、位深与色彩保存 | `scripts/image_io.py` |
| 提示词目录、配置与凭据引用 | `scripts/prompt_library.py`、`scripts/settings.py` |
| 桌面、CLI、HTTP 入口 | `scripts/qt_workbench.py`、`scripts/project_cli.py`、`scripts/backend_server.py` |

照片、模型、运行环境、个人配置、凭据和输出都在仓库外。使用现有 Python，不提交调试脚本、测试照片、
运行记录或交接报告；项目说明只维护 README、AGENTS 和 LICENSES/README，上游许可原文另行保留。
只做与改动直接相关的基本验证，不增加测试框架或照片制作放行阶段。

```powershell
python -m pip install -r scripts/requirements-build.txt
python -m PyInstaller --noconfirm --distpath "$env:TEMP/CosLuma/dist" --workpath "$env:TEMP/CosLuma/build" CosLuma.spec
```

版本在 `scripts/app_version.py` 定义，README 同步更新。GitHub Actions 为新提交检查源码、浮点保存、
Windows 构建、桌面/CLI/HTTP 启动，并提供 Windows ZIP、同提交源码 ZIP 和 `SHA256SUMS.txt`。
这些检查不调用模型，不证明所有照片都成功；已经通过的同一提交不重复运行。
发布时用该提交产物，核对三个文件与 SHA256、tag 指向后公开草稿。
Python 在 GitHub Windows runner 中指定 `3.12`，不假设其提供任意补丁版本；
Actions 旧运行的失败状态不会因新提交成功而改变，失败邮件应按所链接的提交与运行查看。

欢迎通过 Issues 报告问题、Pull Requests 提交修改，说明具体问题、修改行为和必要验证。
安全漏洞使用仓库 Security 页的私密报告入口，勿公开照片、凭据或私有配置。
项目不商用、不盈利，许可证为 [GPL-3.0-or-later](LICENSE)，第三方与权重遵循各自许可，见 [LICENSES](LICENSES/README.md)。
