# 文档入口

用户从[安装与启动](INSTALLATION.md)开始；开发者从[AGENTS.md](../AGENTS.md)和英文[开发指南](AGENT_GUIDE.md)开始。

## 当前产品

- [产品目标与验收边界](PRODUCT_GOAL.md)
- [统一生产路径与资产职责](PRODUCT_ARCHITECTURE.md)
- [工程状态与下一项验收](ENGINEERING_STATE.json)
- [AI harness 工作方法](AI_HARNESS.md)
- [Cos 成片标准](COSPLAY_POST_STANDARD.md)

精修、半合成、大合成共用 `creative_task → production_coordinator → single_source_workflow`。执行先观察原图并冻结浮点基线，再由模型选择摄影调整、局部生成或联合整图编辑，独立审片后按预算改进并保存较好完整版本。详细合同由对应专题维护。

## 使用与配置

- [Windows 程序使用](WINDOWS_README.md) · [配置与预设](PRESETS.md)
- [本地 Qwen](QWEN_IMAGE_LOCAL.md) · [Qwen 提示词](QWEN_PROMPTING.md) · [参考服务](REFERENCE_PROVIDER_API.md)
- [可编辑工程与导出](PORTABLE_EDITABLE_PROJECT.md) · [可撤销形体调整](REVERSIBLE_SHAPING.md)
- [已有参考与原景工具](SOURCE_SCENE_ENHANCEMENT.md) · [人像美化](PORTRAIT_BEAUTY.md)
- [人物边界受光](BOUNDARY_LIGHT.md) · [人物覆盖率](SOURCE_ALPHA.md)

## 开发与接口

- [开发、CLI 与 Windows 构建](DEVELOPMENT.md) · [本机 API](REVISION_API.md)
- [生图与设计合同](GENERATION_PROMPT.md) · [模型提示词适配](MODEL_PROMPTING.md)
- [源域算法范围](ALGORITHM_DECISIONS.md) · [人物细节恢复](AUTOMATIC_SUBJECT_DETAIL.md) · [人物环境光色](AUTOMATIC_SUBJECT_ENVIRONMENT.md)
- [性能计时](WORKFLOW_PERFORMANCE.md) · [存储布局](DISK_STORAGE.md)
- [开源交付准备](OPEN_SOURCE_READINESS.md) · [发布](RELEASING.md) · [仓库边界](REPOSITORY_LAYOUT.md)

## 许可与参与

[第三方声明](THIRD_PARTY_NOTICES.md) · [二进制依赖许可](BINARY_LICENSES.md) · [项目许可](../LICENSE) · [贡献](../CONTRIBUTING.md) · [安全](../SECURITY.md) · [更新记录](../CHANGELOG.md)

公共指南描述当前合同及限制。历史版本变更由更新记录保留；个人照片、费用、运行日志、旧检查点和单次艺术判断保存在私有工程证据中。
