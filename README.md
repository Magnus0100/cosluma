# Cosplay Studio

[![CI](https://github.com/Magnus0100/cosplay/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Magnus0100/cosplay/actions/workflows/ci.yml) [![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/) [![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)

**English:** Cosplay Studio is a Windows desktop workbench for cosplay photo finishing, with a simple photo workspace and a professional layer/workflow view. It prepares editable portrait beauty, appearance, lighting, background and effect layers. See [English overview](README.en.md).

面向Windows的Cos后期工作台：从原片出发，完成人像美化、参考仿色、人物与背景融合及独立光效，保存可继续编辑的高清作品。默认使用简洁的“快速制作”，专业模式提供精修工具与 Cos 节点画布，两种模式共用同一工程。

当前源码版本为 `0.5.46`。项目提供桌面界面、CLI和本机HTTP接口；本地模型和生成服务单独配置。

完整 Cos 后期使用一条管线：原片 → 参考 → Beauty → 场景与原生 T/B/S → 边界受光 → 保存。参考可导入，也可由本地 ComfyUI/Qwen Image、GPT Image 或兼容 Provider 生成；默认生成 1 张，可显式增加候选数量。

原景增强将原尺寸人物与可见背景作为细节基底，角色与主效要求进入检查计划；需要局部生图时显式选择一个背景区域并检查额度，局部之外保持原生像素。快速制作与专业画布共用同一管线及工程，输出仍需整图和原尺寸审片。

## 快速开始

拿到Windows程序包后，完整解压并保留EXE旁边的 `_internal`，使用安装目录的 `Start-Cosplay-Studio.cmd` 或桌面唯一快捷方式启动。程序默认在 `%USERPROFILE%\CosplayWorkbench\workspace` 创建可写工作区，启动内嵌后台，并尝试预热人像美化。实际加载完成才显示 Beauty 预热成功；Qwen 配置检查不证明服务已连接或模型已加载，开始前仍检查本次任务计划。

需要指定工作区时，在程序目录打开PowerShell：

```powershell
$workspace = Join-Path $env:USERPROFILE 'CosplayWorkbench\workspace'
.\cosplay-workbench.exe --root "$workspace" --embedded-backend --auto-start-backend
```

默认进入“快速制作”：导入原片，选择“仅人像美化”或“完整 Cos 后期”，可补充效果说明，对比画面，再保存高清版本和导出成片。美化完成后可直接调整强度；原片和当前效果并排显示，缩放和拖动同步，也可双击查看。处理中可取消任务。处理、保存与导出沿用同一工程和任务状态。

首次使用点击主界面的“模型管理”。软件优先使用主目录下的 [`model` 文件夹](model/README.md)，自动识别标准目录和已有安装；也可以选择一次模型总目录。分析模型与 BeautyGRPO 状态按用途显示，保存配置后重启继续使用；相对路径支持移动完整模型目录。逐项路径和 JSON 导入保留在高级设置。生成服务在设置中配置。专业创作入口可选择“本地原片精修”“已有参考完整后期”或“创作参考并完成后期”，填写创作要求和限制，再检查制作计划中的缺失配置、步骤、外部服务和生图预算。检查计划只读取文件和配置。

节点架构的当前范围与后续路线见 [ComfyUI 架构借鉴](docs/COMFYUI_ARCHITECTURE.md)。

源码与发行包的桌面、CLI 和 HTTP 都使用 `creative_task → production_coordinator → single_source_workflow`，唯一执行策略是 `layered_reference_transfer`。现行 worker 完成 Beauty、场景/人物图层、边界及保存；画布合同与真实执行事实分别记录。runtime 选择、节点职责、失败恢复和活动 job 互斥见[产品数据流](docs/PRODUCT_ARCHITECTURE.md)。

所有制作入口先检查当前 task plan、源/参考身份、模型/runtime、控制与预算，再开始有归属的 job。无参考原片制作关闭参考及场景节点。预检失败不启动 worker；失败、取消或过期保留父版本及首错。旧冻结输出继续只读查看，缺现行任务合同的版本须创建新任务后继续制作。

当前保留按显存预算选择设备/host staging 的共享背景 SR 和有界并行 TIFF 压缩。性能以当前请求的真实账本为准；当前仍未达到稳定发布。

已有参考制作默认不新增图像生成；显式选择原景增强、使用本地Qwen并开启单区精修的新独立任务，可获全局参考0次、局部最多1次的检查预算。普通重试与最终修复不新增额度，详见[原景增强](docs/SOURCE_SCENE_ENHANCEMENT.md)。外部文本/视觉规划与审片显式可选，默认关闭。批量每张独立建立计划、job、版本与质量证据，失败项可独立重试。保存成功与人工艺术批准分别记录。

当前请求与每模块实测见[流程与性能](docs/WORKFLOW_PERFORMANCE.md)。制作中显示阶段及累计用时；预计剩余时间只使用至少三次兼容成功记录，样本不足显示“估算中”。AI 审阅、执行完成和文件导出不能代替用户最终人工批准。

每个桌面窗口绑定自己的工程和内嵌后台。默认端口占用时自动选择可用端口；界面显示实际地址。使用 `--root`打开另一工程的新窗口。

审片遇到明确的上游服务失败，或服务已完成但返回损坏的 JSON，会只重试失败审片板一次。成功板保留，两次请求和错误独立记录；缺少结论的区域继续显示未知。

源码运行使用Windows x64和Python 3.12。完整配置、源码运行及打包命令见 [安装指南](docs/INSTALLATION.md)。当前 GitHub 仓库为私有开发仓库，尚未发布公开 Release；可从源码构建 Windows 包。

## 工作流程

当前生产数据流为：原片解码与输入冻结 → 导入或生成参考 → Beauty → 场景与原生人物仿色、明暗及背景图层 → 边界受光与合成 → 保存。保存后按显式服务策略取得整图与原生窗口审片，执行有资格的本地修复或补证据，人工批准、重开与导出分别记录。`local_portrait` 使用同一管线的无参考节点子集，关闭参考、场景及边界；该任务不隐式准备新的 BeautyGRPO 结果，可消费此前显式准备的冻结美化层。

- BeautyGRPO 默认 100% 使用模型生成的脸部外观，保留生成肤质、清晰度和色调；只在 ROI 接缝和真实遮挡处融合。前额区域沿可见皮肤向发际线扩展，避免脸部椭圆造成上下额断层；不强制恢复原片纹理或锁定眼口颜色。可降低强度或在专业页按人脸恢复原片。范围与限制见[人像美化实跑与接入](docs/PORTRAIT_BEAUTY.md)。
- 原片承担未编辑区域的结构与纹理；参考提供目标外观与场景。生成美化纹理单独记录，不声称是相机原生细节，美化和光效均不改变人物 alpha。
- 专业 Cos 画布提供“仅人像美化”“完整 Cos 场景”“保留原片人物外观”预设。可调整已有可编辑节点的开关、强度、位置并保存/加载模板；参数对应真实工程图层。画布与任务计划共用节点契约，明确输入输出、颜色域及人物 alpha 职责；后台检查依赖、断连、重复输入和循环。状态刷新保留节点选择、布局及观察视图。当前连线固定，未实现自由接线或通用 ComfyUI 插件运行器。模板不携带另一张照片的图像、遮罩或私人配置。
- 颜色和明暗可分别调整，已有冻结版本可重开；支持PNG/JPEG、真实TIFF16/32和portable v2工程。
- “重新审片并本地修复”会取得新文本/视觉审片，按资格处理一次并保存独立版本，不新增参考或独立效果生图。
- 对已审查鞋部窗口的地面细节损失，自动核对地面支持和原生透明度，从原片同坐标恢复亮度细节；保留当前背景的低频色彩基础，另存新版本并重新审片。
- 人物细节先核对实际频带损失和独立结构支持；符合资格才恢复原片亮度细节。仿色、曝光和未确定的噪声不会被当作补纹理的依据，未测得损失时保留原像素与开放审片结论。方法见 [人物细节自动核验](docs/AUTOMATIC_SUBJECT_DETAIL.md)。
- 审片确认人物缺少局部环境响应时，自动比较当前人物和参考的低频光色残差，协调有可靠对应的头发与衣料，并保存独立子版本。方法见 [人物环境光色协调](docs/AUTOMATIC_SUBJECT_ENVIRONMENT.md)。
- 图层页可自动准备“参考发梢受光”和“协调肩袖暖光”两个独立层，已有层可重新准备并替换。“同时更新边界受光”依次准备两层，成功后一起应用，预览后另存一个高清版本；失败或取消保留当前编辑。已接受效果包含发梢颗粒减轻及右肩／上臂的柔和暖色布面；完整强边缘光、前臂收益与跨照片效果仍待验证。方法及限制见 [参考边界受光](docs/BOUNDARY_LIGHT.md)。
- 执行成功、自动诊断和人工批准分别呈现；未知、未支持或失败的问题继续显示。

## 当前边界

本地分析需要另行安装模型权重及兼容运行时。模型从各自的开源社区来源下载到用户自己的目录，仓库只提供来源、版本、哈希和许可说明，不上传模型本体。生成、研究和审片可能使用配置的外部服务；程序包不包含权重、凭据或个人工程。仓库中的文件边界见 [仓库内容与本地文件边界](docs/REPOSITORY_LAYOUT.md)。

局部原尺寸通过只覆盖实际检查的窗口。零对应支持、接触/反射细节及低支持区域仍可能需要补证据或人工修复。系统没有完整物理重光照或自动接收面投影/镜面反射，跨人物、妆造和场次的艺术质量泛化尚未完成验收。

## 文档与参与

使用 Agent 参与开发：先读英文 [AGENTS.md](AGENTS.md) 与 [Agent 开发指南](docs/AGENT_GUIDE.md)，按隔离工作区、模块导航和验证流程开始。

[安装](docs/INSTALLATION.md) · [开发与构建](docs/DEVELOPMENT.md) · [Release发布](docs/RELEASING.md) · [文档索引](docs/README.md) · [参与贡献](CONTRIBUTING.md) · [安全与隐私](SECURITY.md) · [更新记录](CHANGELOG.md)

当前实现及未解决问题见 [算法选择与范围](docs/ALGORITHM_DECISIONS.md) 和[人像美化实跑与接入](docs/PORTRAIT_BEAUTY.md)。0.5.0 引入可编辑学习型人像美化与普通/专业共享工作区，沿用现有联合受光流程；跨人物与场次的艺术质量仍需验证。

## 仓库结构

```text
.github/       CI、发布工作流、Issue 模板和代码所有者
docs/          使用、开发、算法范围、发布和许可说明
presets/       可公开的创作预设
providers/     provider 定义、公开模型清单和工作流模板
scripts/       桌面、后台、CLI、worker 和构建工具
tests/         离线行为检查及临时工程 fixture
LICENSES/      依赖和第三方组件的许可证原文
```

更详细的提交边界见 [仓库内容与本地文件边界](docs/REPOSITORY_LAYOUT.md)。公开交付流程见 [开源准备](docs/OPEN_SOURCE_READINESS.md)，使用不含旧 Git 历史的清洁源码目录建立公开历史。

## 开源许可

项目代码采用 `GPL-3.0-or-later`，详见 [LICENSE](LICENSE)。这表示基于项目代码发布的衍生版本也应在 GPL 条件下提供对应源码；第三方依赖、provider、模型代码和模型权重继续使用各自许可。分发桌面包前请阅读 [第三方声明](docs/THIRD_PARTY_NOTICES.md) 和 [Windows 依赖许可](docs/BINARY_LICENSES.md)。
