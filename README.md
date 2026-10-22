# CosLuma

[![CI](https://github.com/Magnus0100/cosluma/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Magnus0100/cosluma/actions/workflows/ci.yml) [![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/) [![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)

**English:** CosLuma is a Windows desktop workbench for cosplay photo finishing, with a simple photo workspace and a professional layer/workflow view. It prepares editable portrait beauty, appearance, lighting, background and effect layers. See [English overview](README.en.md).

面向 Windows 的 Cos 后期工作台：从原图出发，按精修、半合成或大合成目标设计照片，处理人物、场景与角色氛围。快速制作和专业画布共用同一任务入口、资产与工程。

**下载 0.7.2 正式版：** [Windows x64 完整程序 ZIP](https://github.com/Magnus0100/cosluma/releases/download/v0.7.2/cosplay-workbench-0.7.2-windows-x64.zip) · [对应源码 ZIP](https://github.com/Magnus0100/cosluma/releases/download/v0.7.2/cosplay-workbench-0.7.2-source.zip) · [SHA-256 校验](https://github.com/Magnus0100/cosluma/releases/download/v0.7.2/SHA256SUMS.txt)。

普通用户完整解压后双击 `cosplay-workbench.exe`；模型权重和推理运行环境需按安装指南另行配置。开发者可克隆仓库或下载对应源码。本次正式分发面向独立桌面程序。可选形体调整仍为实验功能；跨图自然度、高清和同场一致性尚未完成产品验收，AI 成片须逐张检查。

当前源码版本为 `0.7.7`。项目提供桌面界面、CLI 和本机 HTTP 接口，已接通原片浮点基线、摄影显影与模型工具选择。已安装程序的代码身份、实际模型效果和跨图片质量须分别验收；本地模型和生成服务单独配置。

当前 AI 制作流程：原片观察与设计 → 原生浮点基线 → 模型选择摄影/局部/整体编辑工具 → 独立审片与有界改进 → 保存与导出。制作范围与风格分别选择，默认在原场地内半合成。普通原片与导入参考的有效图层工具按任务启用。

Qwen 和 GPT 使用相同创作目标、各自的实际编辑指令。生成结果按实际可见且获准的覆盖合成，其他位置保留当前浮点像素；整图编辑允许同时调整人物与场景外观。生成分辨率与原尺寸画布分别记录，不将放大的生成纹理称为相机原生细节。执行成功、自动评价和人工批准分别记录，不兼容已删除的旧资产格式或流程。

## 快速开始

拿到Windows程序包后，完整解压并保留EXE旁边的 `_internal`，使用安装目录的 `Start-Cosplay-Studio.cmd` 或桌面唯一快捷方式启动。程序默认在 `%USERPROFILE%\CosplayWorkbench\workspace` 创建可写工作区，启动内嵌后台，并在恢复用户设置后初始化选定服务。服务就绪与权重加载分别记录，不无条件加载未选中的 Beauty；开始前仍检查本次任务计划。

需要指定工作区时，在程序目录打开PowerShell：

```powershell
$workspace = Join-Path $env:USERPROFILE 'CosplayWorkbench\workspace'
.\cosplay-workbench.exe --root "$workspace" --embedded-backend --auto-start-backend
```

默认进入“快速制作”：导入原片，选择“仅人像美化”或“完整 Cos 后期”，可补充效果说明，对比画面，再保存高清版本和导出成片。美化完成后可直接调整强度；原片和当前效果并排显示，缩放和拖动同步，也可双击查看。处理中可取消任务。处理、保存与导出沿用同一工程和任务状态。

首次使用点击主界面的“模型管理”。软件优先使用主目录下的 [`model` 文件夹](model/README.md)，自动识别标准目录和已有安装；也可以选择一次模型总目录。分析模型与 BeautyGRPO 状态按用途显示，保存配置后重启继续使用；相对路径支持移动完整模型目录。逐项路径和 JSON 导入保留在高级设置。生成服务在设置中配置。专业创作入口可选择“本地原片精修”“已有参考完整后期”或“创作参考并完成后期”，填写创作要求和限制，再检查制作计划中的缺失配置、步骤、外部服务和生图预算。检查计划只读取文件和配置。

节点架构的当前范围见 [产品架构](docs/PRODUCT_ARCHITECTURE.md)。

源码与发行包的桌面、CLI 和 HTTP 都使用 `creative_task → production_coordinator → single_source_workflow`，唯一执行策略是 `layered_reference_transfer`。现行 worker 完成 Beauty、场景/人物图层、边界及保存；画布合同与真实执行事实分别记录。runtime 选择、节点职责、失败恢复和活动 job 互斥见[产品数据流](docs/PRODUCT_ARCHITECTURE.md)。

所有制作入口先检查当前 task plan、源/参考身份、模型/runtime、控制与预算，再开始有归属的 job。无参考原片制作关闭参考及场景节点。预检失败不启动 worker；失败、取消或过期保留父版本及首错。旧冻结输出继续只读查看，缺现行任务合同的版本须创建新任务后继续制作。

当前保留按显存预算选择设备/host staging 的共享背景 SR 和有界并行 TIFF 压缩。性能以当前请求的真实账本为准；当前仍未达到稳定发布。

已有参考制作默认不新增图像生成；显式选择原景增强、使用本地Qwen并开启单区精修的新独立任务，可获全局参考0次、局部最多1次的检查预算。普通重试与最终修复不新增额度，详见[原景增强](docs/SOURCE_SCENE_ENHANCEMENT.md)。普通原片／导入参考的文本视觉服务默认关闭；scoped 创建及塑形 apply 按计划要求视觉服务。批量每张独立建立计划、job、版本与质量证据，失败项可独立重试。保存成功与人工艺术批准分别记录。

阶段计时口径见[流程与性能](docs/WORKFLOW_PERFORMANCE.md)。制作中显示阶段及累计用时；预计剩余时间只使用至少三次兼容成功记录，样本不足显示“估算中”。AI 审阅、执行完成和文件导出不能代替用户最终人工批准。

每个桌面窗口绑定自己的工程和内嵌后台。默认端口占用时自动选择可用端口；界面显示实际地址。使用 `--root`打开另一工程的新窗口。

未决请求和缺失审片结论明确保留为待核查，不自动补发生图；已有完整草稿、实际请求和错误记录保留。

源码运行使用Windows x64和Python 3.12。完整配置、源码运行及打包命令见 [安装指南](docs/INSTALLATION.md)。可从源码构建 Windows 包。

## 工作流程

桌面、CLI 与本机 HTTP 均通过 `task-plan / task-start → production_coordinator → single_source_workflow` 检查和执行。模型看实际原图形成设计，选择必要工具；代码负责权限、像素资产、预算、取消与保存。

- 精修、半合成和大合成分别表达制作范围；风格另选。原景半合成为默认，不自动把室内改成室外。
- 当前浮点母版与查看预览分离。局部编辑保留覆盖外像素，整图生成不再被原人物 RGB 盖回。
- 首次操作与后续改进分别记账，默认最多两轮条件改进。独立判断自然可信度与创作收益，保留较好完整版本，未知请求不续发额度。
- 当前工程可复制、重开、导出 PNG/JPEG 和真实 TIFF16/32，并通过 v3 便携工程交换；只支持当前格式。自动评价不授予人工批准。

普通原片与导入参考按任务启用有效的人像美化、原生图层与光色工具。详见[人像美化](docs/PORTRAIT_BEAUTY.md)、[源域边界受光](docs/BOUNDARY_LIGHT.md)与[原景增强](docs/SOURCE_SCENE_ENHANCEMENT.md)；它们不构成 AI 创建的固定必经链。

## 当前边界

本地分析需要另行安装模型权重及兼容运行时。模型从各自的开源社区来源下载到用户自己的目录，仓库只提供来源、版本、哈希和许可说明，不上传模型本体。生成、研究和审片可能使用配置的外部服务；程序包不包含权重、凭据或个人工程。仓库中的文件边界见 [仓库内容与本地文件边界](docs/REPOSITORY_LAYOUT.md)。

局部原尺寸通过只覆盖实际检查的窗口。零对应支持、接触/反射细节及低支持区域仍可能需要补证据或人工修复。系统没有完整物理重光照或自动接收面投影/镜面反射，跨人物、妆造和场次的艺术质量泛化尚未完成验收。

## 文档与参与

使用 Agent 参与开发：先读英文 [AGENTS.md](AGENTS.md) 与 [Agent 开发指南](docs/AGENT_GUIDE.md)，按隔离工作区、模块导航和验证流程开始。

[安装](docs/INSTALLATION.md) · [开发与构建](docs/DEVELOPMENT.md) · [Release发布](docs/RELEASING.md) · [文档索引](docs/README.md) · [参与贡献](CONTRIBUTING.md) · [安全与隐私](SECURITY.md) · [更新记录](CHANGELOG.md)

当前目标、验证事实和未完成能力见 [产品目标](docs/PRODUCT_GOAL.md) 与 [工程状态](docs/ENGINEERING_STATE.json)。

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
