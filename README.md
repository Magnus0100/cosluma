# CosLuma

当前版本：**0.9.0**，Windows x64。

给原片和角色，自动设计 COS 后期，并将原片整头融合回生成画面，保留本人五官、脸型和相机细节。

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

## 使用

日常只需选择照片、填写角色、点击开始制作。补充要求可留空；软件先依据角色、原片受光与实物妆造
确定一个整组摄影方向，再处理每张照片。场景、材质、光质和配色保持一致，具体光向、景别与特效按单张决定。
配置文件和请求尺寸放在桌面的折叠高级设置中，服务配置完成后不必每次调整。
桌面会记住角色、保存目录、配置路径、请求长边和每个角色各自的补充要求；切换角色时分别恢复。
这些偏好保存在 `~/CosplayWorkbench/config/desktop.ini`，不保存凭据。

Windows 发布包解压后双击 `CosLuma.exe`，已包含桌面和图像处理依赖。
同一可执行文件可用 `CosLuma.exe --cli ...` 运行下方 CLI 参数，或用 `CosLuma.exe --server ...` 启动本地 HTTP。
模型服务、权重、提示词库与服务配置仍需另行准备；发布包不含个人服务地址或密钥。
源码包使用下方 Python 启动方式，或双击 `Start-Cosplay-Studio.cmd` 使用配置中指定的已有 Python。

使用已有 Python，不新建运行环境。另一台机器安装本机依赖时：

```powershell
python -m pip install -r scripts/requirements-desktop.txt
```

将 [配置示例](providers/settings.example.json) 另存到 `~/CosplayWorkbench/config/cosluma.json`；
`COSLUMA_CONFIG` 或桌面 `--config` 可指定其他配置文件，并优先于记住的路径。
资源相对路径以配置文件目录为准。
配置提示词库、LLM、图像编辑服务，以及 ComfyUI 的整头提取和超分服务。
`head.model` 是远端 `models/checkpoints` 中的 SAM3.1 权重名，`sr.model` 是超分权重名。
GPU 设备由远端服务明确配置，本机 SIFT 和光色合成使用 OpenCV、NumPy 和 color-matcher。
凭据通过 `api_key_env` 或 `api_key_file` 指定，只在调用对应服务时读取。

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

构建 Windows 发布目录使用已有、安装了 PyInstaller 的 Python 运行 `python -m PyInstaller CosLuma.spec`。
可用 `--distpath`、`--workpath` 将产物放在仓库外。版本统一定义在 `scripts/app_version.py`；
桌面、CLI 与 HTTP 启动命令均支持 `--version`。发布包需随附同版本源码与第三方许可。
