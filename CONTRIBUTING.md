# 参与开发

欢迎改进 Cosplay Studio 的界面、文档、图像处理和可复现验证。先阅读 [开发指南](docs/DEVELOPMENT.md) 和 [产品数据流](docs/PRODUCT_ARCHITECTURE.md)，了解原片、参考、冻结资产及人工批准的职责。

## 提交前

使用 Windows x64、Python 3.12 和仓库固定的桌面依赖。模型权重及GPU运行时单独配置；纯文档或界面改动通常不需要下载模型。

一次修改围绕一个明确问题。说明触发条件、修改后的行为、验证方法和仍存在的限制。修复处理链路时，保留输入和父版本，验证新版本的来源绑定、保存及重开行为。

提交前至少运行：

```powershell
git diff --check
.\.venv\Scripts\python.exe .\scripts\check_repository.py
.\.venv\Scripts\python.exe .\tests\run.py core
```

图像算法修改需要与变化相符的验证：检查实际原生像素、参考资格、身份和妆造，并记录失败或缺证据的区域。网络审片、生图及GPU推理的成本要在验证说明中单独列出。无需为简单文案修改重新生成图像。

## 仓库内容

提交源码、公共文档、配置模板和合法的最小测试素材。自己的照片、工程选择、输出、模型权重、运行时、API密钥及机器配置留在本地。复现材料使用取得授权并去除个人信息的素材。

增加生产模块或外部worker时，同步维护 `scripts/runtime_manifest.json`；涉及打包时按开发指南检查Debug和Release。成功启动、数值一致、自动诊断和人工成片批准分别报告。

当前仓库位于 GitHub 私有仓库；维护者会在开放协作前启用实际的 Issue 和 Pull Request 流程。贡献的授权条件以根目录 [LICENSE](LICENSE) 为准，第三方代码和模型须保留各自许可及来源。

## 公开源码导出

本地开发历史包含私人工程及机器记录。公开仓库使用清洁源码导出建立新历史，不直接推送旧本地分支或标签。导出时检查实际内容、配置模板、第三方许可和依赖；`.gitignore`不能隐藏已经进入历史的内容。

使用 `scripts/prepare_source_release.py`生成清洁目录及文件身份清单，具体命令见 [开发指南](docs/DEVELOPMENT.md)。
