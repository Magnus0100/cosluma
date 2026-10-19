# 公共文档

从 [安装指南](INSTALLATION.md) 开始；源码运行、构建与验证见 [开发指南](DEVELOPMENT.md)。当前文档对应 0.5.45，功能范围以当前生产实现和真实验收记录为准；跨人物、跨场景的艺术泛化仍需单独验证。

## 当前入口与生产管线

桌面、CLI与后台统一使用 `creative_task → production_coordinator → single_source_workflow`。本地单候选完整制作按 `原片 → 源Beauty并冻结 → 原RAW生成设计参考 → 场景与原生T/B/S → 边界受光 → 原尺寸光色统一 → 保存` 执行，避免卸载已预热的Beauty后再次冷加载。远程生成可与源Beauty重叠，已有参考从同一编排复用已拥有的步骤。唯一策略为 `layered_reference_transfer`；无参考原片制作关闭参考与场景节点。

这条链路按任务编译，不能从某一张人工精修图反推固定参数。新任务默认从“按角色与本图自动设计”开始，参考图只提供本次用户意图约束下的光影、氛围和局部色彩设计；原片拥有几何、人物身份、服装、道具、纹理和最终画布。用户主动选择的角色或风格会写入该任务合同，当前任务只回放自己保存的合同，不会把上一张图的建议带到下一张图。

本地模型注册表随任务传给每一个原生后处理 worker。桌面入口、CLI 和后台因此使用同一份模型配置，即使任务显式选择的注册表与启动程序的环境变量不同，场景、人物、材质和导出阶段仍不会悄悄切换到另一套模型。端到端完成只表示源图绑定、参考生成（如启用）、后处理和高清文件已保存；艺术接受仍由人工审片状态确认。

创建参考默认 1 张并直接完成制作；显式请求多候选时才暂停选择。桌面加载完整任务的保存结果后结束制作，不再次启动 Beauty 和后期。已有参考默认零新增图像；显式原景增强＋本地Qwen单区新独立任务可获局部最多1次额度，全局参考仍为0，普通重试与最终修复不授新额度，见[原景增强](SOURCE_SCENE_ENHANCEMENT.md)。外部文本/视觉规划与审片显式可选，默认关闭。预检只检查启用能力：本地原片使用 BiRefNet/ONNX，参考制作增加当前 Sapiens2、ViTMatte、DINOv3、SAM3.1；完整后期增加 LaMa、背景 SR 和五官定位。BeautyGRPO 使用独立安装回执。

`cos_node_registry` / `cos_workflow_graph` / `cos_workflow_cache` 编译现行固定依赖、模板参数和 UI 失效提示；真实执行由当前 worker 及冻结资产回执证明。仅支持当前任务与资产合同，缺失合同的记录不可载入画布或继续执行。详细职责见 [产品数据流](PRODUCT_ARCHITECTURE.md) 与 [ComfyUI 架构借鉴](COMFYUI_ARCHITECTURE.md)。

## 使用与配置

- [Agent 开发指南（英文）](AGENT_GUIDE.md) 与 [Agent 规则](../AGENTS.md)：让编码 Agent 快速理解代码、建立环境并完成验证。
- [开源交付准备](OPEN_SOURCE_READINESS.md)：清洁源码、隐私边界和公开历史建立流程。
- [安装指南](INSTALLATION.md)：Windows程序包、源码运行、模型和服务配置。
- [Windows包使用说明](WINDOWS_README.md)：解压、启动、制作和保存。
- [配置与预设](PRESETS.md)：风格、创作要求与用户限制。
- [参考服务](REFERENCE_PROVIDER_API.md)：服务连接、凭据及生成记录。
- [Qwen本地服务](QWEN_IMAGE_LOCAL.md)：可选ComfyUI部署与许可。
- [人物边界受光](BOUNDARY_LIGHT.md)：发梢、肩袖暖光及联合更新，保存规则与实际限制。
- [可编辑工程与导出](PORTABLE_EDITABLE_PROJECT.md)：portable v2、真实母版与迁移。
- [Cos成片标准](COSPLAY_POST_STANDARD.md)：正常整图、原生细节、证据范围及人工批准。

## 架构、算法与接口

- [产品数据流](PRODUCT_ARCHITECTURE.md)：任务入口、资产职责、图层、保存及恢复。
- [算法选择与范围](ALGORITHM_DECISIONS.md)：当前实现、资格边界和未解决问题。
- [生图与设计合同](GENERATION_PROMPT.md)：要求、预算、独立效果及纠错。
- [人物细节核验](AUTOMATIC_SUBJECT_DETAIL.md)：结构损失资格与冻结恢复。
- [人物环境光色](AUTOMATIC_SUBJECT_ENVIRONMENT.md)：低频残差协调与局部留出。
- [人物透明度](SOURCE_ALPHA.md)：源前景、覆盖率与职责。
- [本机API](REVISION_API.md)：制作、准备、编辑、审片与工程接口。
- [开发与构建](DEVELOPMENT.md)：CLI、后台、源码交付和行为验证。
- [GitHub Release发布](RELEASING.md)：成套下载资产、核验和发布流程。

## 许可与参与

- [第三方声明](THIRD_PARTY_NOTICES.md)：组件与模型许可。
- [Windows依赖许可](BINARY_LICENSES.md)：对应源码与重建说明。
- [参与贡献](../CONTRIBUTING.md) · [安全与隐私](../SECURITY.md) · [更新记录](../CHANGELOG.md) · [项目许可](../LICENSE)

## 后续研究

- [人像美化实跑与接入](PORTRAIT_BEAUTY.md)：BeautyGRPO 生产配置、冻结回放、用户选择及真实验收范围。

个人素材、工程状态、会议和运行证据由用户工程保存；它们不是公共安装要求或发布成绩。历史版本变更只在更新记录保留。
