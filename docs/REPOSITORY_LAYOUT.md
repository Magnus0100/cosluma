# 仓库内容与本地文件边界

这个仓库只保存可以复现源码、测试、公开配置模板、文档、许可证和 GitHub 工作流。模型权重、模型运行时、个人照片、工程文件和生成结果都留在用户自己的电脑上。

## 会提交到 GitHub 的内容

- `scripts/`、`tests/`和桌面启动器：源码、行为检查和构建入口。
- `providers/*.example.json`、`providers/registry.json`、`presets/`：公开配置、模型来源、版本、哈希和路径模板。它们不包含权重和凭据。
- `docs/`、`README.md`、`CHANGELOG.md`、`CONTRIBUTING.md`、`SECURITY.md`：使用、开发、许可和安全说明。
- `LICENSES/`和`.gitattributes`：源码及随包依赖的许可原文和字节保持规则。
- `.github/`、`pyproject.toml`和依赖清单：CI、发布检查和源码安装所需的项目文件。

## 永远留在本地的内容

- 模型权重和本地运行时：`.safetensors`、`.onnx`、`.gguf`、`.pt`、`.pth`、`.ckpt`、`.bin`、`.tflite`等文件，以及`models/`、`weights/`、`checkpoints/`和`.cache/`目录。
- 私人素材和运行结果：`data/`、`output/`、`outputs/`中的内容（目录说明文件除外）。
- 本机环境和凭据：`.env.local`、本地模型注册表、`.venv/`、缓存、构建目录、发行包和工程工作区。
- 私有实跑与临时材料：`evidence/`、`reports/`、`artifacts/`、`scratch/`、`logs/`、数据库和压缩包；公共文档只保留当前方法与合同。

建立公开仓库时使用 `prepare_source_release.py --for-repository`，只导出公共源码文件；生成的 `SOURCE_MANIFEST.json` 属于 Release 快照，不进入公开仓库。已跟踪但位于公开白名单之外的文件会使 `check_repository.py` 失败。

模型由用户从各自的开源社区来源自行下载，并放到自己的模型目录；源码通过 `COSPLAY_MODEL_REGISTRY` 或工作台配置读取这些绝对路径。仓库只保留来源地址、固定 revision、SHA-256 和许可说明，方便用户下载后核对身份，不上传模型本体。

提交前可以用下面的命令确认某个本地文件被忽略：

```powershell
git check-ignore -v .cache/local-models/sapiens2/checkpoints/model.safetensors
git status --short --ignored
```
