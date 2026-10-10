# 发布 GitHub Release

公开发布使用清洁源码快照。开发工作区包含私人历史、照片和工程记录，不直接推送它的历史、分支或标签。源码导出、Windows 包和发布清单须来自同一个版本。

## 准备交付

按 [开发指南](DEVELOPMENT.md) 导出清洁源码，从该目录构建 Debug 和 Release，再用 `assemble_windows_release.py` 生成发布资产。工具自动核验源码、许可、全部当前运行模块、逐文件身份、ZIP 成员和 SHA-256；失败时不交付。

GitHub 中可手动运行 `Prepare release assets` 工作流。它在 Windows 上安装固定依赖、导出清洁源码，执行当前基础行为检查，构建并启动检查 Debug/Release，然后上传成套资产。工作流仅生成下载产物；发布由维护者在 GitHub 中完成。

| 发布资产 | 内容 |
|---|---|
| `cosplay-workbench-<版本>-windows-x64.zip` | 完整 EXE、桌面依赖、用户说明和许可 |
| `cosplay-workbench-<版本>-source.zip` | 对应源码、当前测试、配置模板、构建文件和第三方许可 |
| `SHA256SUMS.txt` | 两份 ZIP 的 SHA-256 |
| `release-manifest.json` | 版本、资产大小、哈希和分发范围 |
| `validation.json` | 无本机路径的公开发布核验摘要 |
| `RELEASE_NOTES.md` | 从当前版本更新记录生成的发布说明 |

独立复核已有交付目录：

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_windows_release.py --source "<清洁源码目录>" --release "<发布资产目录>"
```

程序包必须完整解压并保留 `_internal`。用户运行模型需要单独安装；发布包不得夹带模型权重、个人配置、照片、工程或服务凭据。包内开发文档中的源码命令使用对应 `source.zip`。

## 发布到 GitHub

从清洁源码快照建立公开仓库历史，并完成 CI。版本以 `scripts/single_source_workflow.py` 的 `SOFTWARE_VERSION` 为准，标签使用 `v<版本>`，标签所指源码与源码 ZIP 相同。未发布的清理可以继续完善同一个版本；公开后禁止替换已有标签和资产，需要新补丁版本。

创建 Release 时上传上述六个资产，把 `RELEASE_NOTES.md` 内容作为说明，并同时提供程序与对应源码。调试包、构建缓存、私有验证日志和本机操作记录不上传。

发布前在干净工作区实际启动程序，检查基本制作、保存、重开和导出。图像算法有变化时另行验证真实素材与原尺寸细节；本次清理复用已有运行字节和已有冻结回放证据，不授予新的艺术效果或跨照片验收。

启用仓库的私密漏洞报告入口，并按 [安全说明](../SECURITY.md) 接收报告。当前仓库仍是私有开发仓库，尚未发布公开 Release；发布前不要把本地验证记录或未审查资产上传到 GitHub。
