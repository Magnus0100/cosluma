# Cosplay Studio

[![CI](https://github.com/Magnus0100/cosplay/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Magnus0100/cosplay/actions/workflows/ci.yml) [![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/) [![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)

**English:** Cosplay Studio is a Windows desktop workbench for source-faithful cosplay photo finishing. It keeps source structure and texture while preparing editable appearance, lighting, background and effect layers. See [English overview](README.en.md).

面向Windows的Cos后期工作台：从原片和设计参考出发，保留原人物的结构与纹理，准备材质仿色、明暗、背景和效果，保存可继续编辑的高清草稿。

当前源码版本为 `0.4.8`。项目提供桌面界面、CLI和本机HTTP接口；本地模型和生成服务单独配置。

## 快速开始

拿到Windows程序包后，解压完整的 `cosplay-workbench` 文件夹，保留EXE旁边的 `_internal`，双击 `cosplay-workbench.exe`。程序默认在 `%LOCALAPPDATA%\CosplayWorkbench\workspace` 创建可写工作区，并自动启动内嵌后台。

需要指定工作区时，在程序目录打开PowerShell：

```powershell
$workspace = Join-Path $env:LOCALAPPDATA 'CosplayWorkbench\workspace'
.\cosplay-workbench.exe --root "$workspace" --embedded-backend --auto-start-backend
```

打开“连接设置 → 配置分析模型”，可导入已有JSON或选择已安装的模型与Python路径，保存后软件会记住选择并显示缺失项。导入原片，选择“本地原片精修”“已有参考完整后期”或“创作参考并完成后期”，填写创作要求及保留背景、不要特效等限制。先检查制作计划，查看缺失配置、处理步骤、外部服务和生图预算，再开始制作。检查计划只读取文件和配置，不执行推理或网络请求。

已有参考会先对原片与参考做成对设计，再进入完整后期，所有图像生成预算为零；文本/视觉规划和审片有独立服务策略。完整参考后期目前需要文本/视觉服务，本地原片路线可离线处理。批量作品共享创作要求，每张分别建立设计、几何、遮罩和质量证据，失败项可独立重试。保存版本后查看整图、眼口及原尺寸证据，按缺项修复或补证据，人工审片后导出图片或可编辑工程。

每个桌面窗口绑定自己的工程和内嵌后台。默认端口占用时自动选择可用端口；界面显示实际地址。使用 `--root`打开另一工程的新窗口。

审片遇到明确的上游服务失败，或服务已完成但返回损坏的 JSON，会只重试失败审片板一次。成功板保留，两次请求和错误独立记录；缺少结论的区域继续显示未知。

源码运行使用Windows x64和Python 3.12。完整配置、源码运行及打包命令见 [安装指南](docs/INSTALLATION.md)。当前 GitHub 仓库为私有开发仓库，尚未发布公开 Release；可从源码构建 Windows 包。

## 工作流程

原片基线 → 原图与参考仿色 → 材质和明暗协调 → 人物、背景、接触与明确独立效果合成 → 整图与原生窗口审片 → 有资格的本地修复或补证据 → 保存、人工批准、重开与导出。

- 原片承担人物身份、几何和高频纹理；参考提供目标外观与场景。
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

[安装](docs/INSTALLATION.md) · [开发与构建](docs/DEVELOPMENT.md) · [Release发布](docs/RELEASING.md) · [文档索引](docs/README.md) · [参与贡献](CONTRIBUTING.md) · [安全与隐私](SECURITY.md) · [更新记录](CHANGELOG.md)

当前实现及未解决问题见 [算法选择与范围](docs/ALGORITHM_DECISIONS.md)。0.4.8清理过时源码和文档，沿用现有联合受光流程与已接受艺术范围。

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

更详细的提交边界见 [仓库内容与本地文件边界](docs/REPOSITORY_LAYOUT.md)。

## 开源许可

项目代码采用 `GPL-3.0-or-later`，详见 [LICENSE](LICENSE)。这表示基于项目代码发布的衍生版本也应在 GPL 条件下提供对应源码；第三方依赖、provider、模型代码和模型权重继续使用各自许可。分发桌面包前请阅读 [第三方声明](docs/THIRD_PARTY_NOTICES.md) 和 [Windows 依赖许可](docs/BINARY_LICENSES.md)。
