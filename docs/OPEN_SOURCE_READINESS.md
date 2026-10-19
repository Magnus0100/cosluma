# 开源交付准备

公开内容包括源码、测试、公共配置模板、使用和开发文档、构建入口与许可证。现有本地 Git 历史包含私人记录，应使用清洁源码导出建立公开历史。源码整理不会改变远端可见性，也不会自动发布 Release。

## 生成候选源码

在已安装桌面依赖的 Python 3.12 环境中，从仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -B .\scripts\check_repository.py
.\.venv\Scripts\python.exe -B .\tests\run.py core
$sourceExport = Join-Path (Split-Path (Get-Location) -Parent) 'cosplay-open-source'
.\.venv\Scripts\python.exe -B .\scripts\prepare_source_release.py --for-repository --output "$sourceExport"
```

目标目录必须尚不存在且位于当前仓库之外。导出保留工作区中的当前源码，包括尚未提交的修改；Git 忽略的未跟踪文件不会进入候选目录。已跟踪文件仍接受检查，不能通过新增忽略规则隐藏。没有 Git 的源码目录也可再次导出。

工具检查版本一致性、源码语法、运行模块清单、依赖许可身份、常见凭据格式、私钥、个人主目录路径和受限文件类型。已跟踪但不在公开源码白名单中的文件也会导致检查失败。`--for-repository` 只导出可继续编辑的公开文件，不携带 `.git`、本机提交标识或生成的快照清单。这些检查不能识别所有自定义凭据和个人文本，仍应审阅实际交付内容。

制作 Release 的源码资产时省略 `--for-repository`。默认模式另生成 `SOURCE_MANIFEST.json`，记录逐文件大小、SHA-256 和本地基线提交；它只用于冻结发布输入，不提交到公开仓库。复核快照会拒绝文件变更或未列出的额外文件；改动后应重新导出。对应源码包的交付流程见 [发布指南](RELEASING.md)。

导出后，用同一个 Python 环境在候选目录执行 `scripts/check_repository.py` 和 `tests/run.py core`，确认缺少本地配置时仍可完成离线检查。第三方许可原文及清单保持完整，模型权重和独立模型运行时由使用者另行安装。

可选的真实素材回归通过 `COSPLAY_REGIONAL_REGRESSION_DIRS` 指定素材目录列表，Windows 使用分号分隔。未配置时跳过该项，合成素材回归继续执行；显式配置但缺失素材时检查失败。不得将私人素材加入公开源码。

## 建立公开历史与发布

审阅候选目录后，在该目录单独执行 `git init` 并创建首次提交，连接公开目标仓库。不要复制旧 `.git`，也不要从开发仓库推送旧分支、标签或使用镜像推送。

维护者在 GitHub 中确认目标仓库、分支、可见性和私密漏洞报告设置，运行 CI 后再开放协作。源码可运行的验证与桌面二进制验收分别完成；Windows 包需要按 [发布指南](RELEASING.md) 构建 Debug/Release、验证资产并同时提供对应源码。未构建的候选源码不代表已发布程序包。

文档入口：[安装](INSTALLATION.md)、[开发](DEVELOPMENT.md)、[仓库边界](REPOSITORY_LAYOUT.md)、[第三方声明](THIRD_PARTY_NOTICES.md)、[安全与隐私](../SECURITY.md)。
