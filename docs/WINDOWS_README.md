# Cosplay Studio — Windows 程序包

完整解压文件夹后，双击 `cosplay-workbench.exe`。保留旁边的 `_internal`；默认工作区为 `%LOCALAPPDATA%\CosplayWorkbench\workspace`。

1. 在“连接设置 → 配置分析模型”中选择已安装的模型和运行时，并配置需要的外部服务。
2. 导入原片，选择本地原片精修、已有参考完整后期或创作参考完整后期，填写创作要求。
3. 检查计划、缺失项和预算，再开始制作。已有参考不消耗图像生成预算，完整参考后期仍需要文本/视觉服务；本地原片路线可离线处理。
4. 查看预览、整图和原尺寸细节，保存独立版本，再导出图片或可编辑工程。自动执行成功与人工成片批准分别记录。

“同时更新边界受光”可准备发梢和肩袖两层，预览后保存一个新版本。当前已接受范围是发梢颗粒减轻和右肩／上臂柔和暖色布面；强窄轮廓光、前臂收益及跨照片效果仍待验证。详见 [边界受光](BOUNDARY_LIGHT.md)。

指定工作区可在程序目录运行：

```powershell
.\cosplay-workbench.exe --root "D:\My Cosplay Project" --embedded-backend --auto-start-backend
```

双击 `Start-Cosplay-Studio.cmd` 可读取相邻 `.env`、`.env.local` 中的字面量配置。直接运行 EXE 时，在连接设置中配置，或预先设置进程环境变量。

程序包包含桌面依赖和许可，不包含模型权重、模型运行时、私人照片、工程或凭据。开发脚本、自动测试、构建工具及 Qwen 安装启动器位于同版本 `source.zip`；文档中“从源码运行”和构建命令需先解压源码包。

[安装与模型配置](INSTALLATION.md) · [可编辑工程与导出](PORTABLE_EDITABLE_PROJECT.md) · [第三方许可](THIRD_PARTY_NOTICES.md) · [项目许可](../LICENSE)
