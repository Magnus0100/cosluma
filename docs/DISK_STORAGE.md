# 存储布局

源码目录、程序安装目录、模型目录和工程工作区分别管理。实际位置由用户选择。

| 内容 | 位置与配置 |
|---|---|
| 源码 | 任意开发目录；保存代码、公共模板、文档与许可 |
| Windows 程序 | 独立安装目录，保留 EXE 和匹配的 `_internal` |
| 模型与模型运行时 | 安装目录的 `model/`、`runtimes/`，或用户选择的独立磁盘目录 |
| 工程与照片 | 默认 `%USERPROFILE%\CosplayWorkbench\workspace`；通过 `COSPLAY_WORKSPACE` 或 `--root` 指定 |
| 可重建缓存 | 通过 `COSPLAY_CACHE_ROOT` 指定缓存总目录 |

当前开发缓存策略在存在 D 盘的 Git 工作区中使用 `D:/CosplayStudio/outputs/cache`，其余工程通常使用工作区内的 `outputs/cache`。需要隔离验证或使用其他磁盘时，显式设置 `COSPLAY_CACHE_ROOT`；独立 worker 的身份和 RAW 元数据缓存还可分别设置 `COSPLAY_ASSET_IDENTITY_CACHE`、`COSPLAY_SOURCE_METADATA_CACHE`。照片、工程、缓存、模型和运行时均不进入公开源码导出。

当前程序版本来自源码 `scripts/single_source_workflow.py` 的 `SOFTWARE_VERSION`，构建脚本使用已验证的 Python 3.12 x64 desktop runtime，并执行冻结版 smoke。源码、程序包和桌面入口使用同一生产流程：

`production_coordinator → single_source_workflow`

工作区、模型和运行时不写入源码仓库的历史结果目录。保存资产由当前工程合同、输入 SHA、代码身份和运行时身份绑定；缓存失效时重新准备，不解释旧 receipt 为当前任务。

工作数组继续使用固定无损编码策略：native 数组使用 `ZIP_STORED`，稀疏 proxy 成员使用低级别 DEFLATE，scene dense baseline 使用 `ZIP_STORED`。读取保持 `allow_pickle=False`、dtype/shape/owner 校验和原子发布。
