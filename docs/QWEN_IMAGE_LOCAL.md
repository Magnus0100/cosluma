# 本地 Qwen 入口

本地 Qwen 通过当前 `task-plan` / `task-start` 入口执行明确授权的图像生成。新建创作任务使用当前 `editing_scope`：观察原图、冻结浮点基线与设计，再由模型选择摄影调整、局部或整图编辑，按预算独立审片和改进、保存较好完整版本。有效的已有参考导入继续使用其原生后期节点，不因导入而重新生图；两者由同一 coordinator 和 `single_source_workflow` 按检查后的合同执行。

提示词只描述当前用户要求、动作、位置、遮挡和期望结果。入口逐字发送用户指令，不自动加入固定色调、蓝雾、光束、角色或场景描述。默认负向提示词为空；CFG 1 不启用负向分支引导，当前编码适配器同时省略无用的空负向编码。

新工作区默认选择官方 BF16 扩散模型与 BF16 文本编码器。全局编辑保留 `resolution=2048` 的像素预算、40 步、Euler/simple、CFG 3；竖幅 2:3 实际生成约 1664×2496，不把原片高清回写称为模型原生生成。RAW 上传长边上限为4096，为2K画布保留足够条件细节。用户显式选择的低显存模型和自定义提示词继续保留。

当前内置 base 编辑工作流保留 40 步、CFG 3 和默认空负向提示；显式自定义工作流值由请求冻结，适配器不擅自重写。当前配置不代表所有图片或模型变体的最佳参数，修改策略需做受控对照。

继续编辑只组合本轮修改指令与必要保留项，避免重复旧计划或保留正要修改的受光。效果以实际候选和独立审片为准。

模型存储位置与桌面程序位置分别配置。Windows 的模型锁页缓冲区可能额外预提交大量内存、触发系统分页文件扩容，因此当前 Windows 服务使用 `--disable-pinned-memory`，存储由上游按实际磁盘检测，不强制 RAM 或磁盘流式模式；其他平台沿用上游锁页默认。把高频权重放到 SSD 后，映射加载仍可复用系统文件缓存，不需要用额外大锁页缓冲占满系统盘。该策略不改变模型精度或采样设置。原尺寸缓存和结果仍会写入配置的工作盘，不据此声称全程没有磁盘等待。

用户配置目录的 `qwen-model-storage.json` 可以单独配置权重位置：`schema` 为 `cosplay_qwen_model_storage.v1`，`models` 将当前模型清单中的 role 映射到绝对文件路径。迁移须保留原文件名、核验原大小与 SHA。预检和 ComfyUI 启动共同消费该配置，启动时生成独立模型搜索路径；位置改变后重启已验证且空闲的自有服务。未指定的可选模型沿用其部署位置，已指定但缺失的模型不会静默回退。

默认 VAE 为 [Texture-Fix VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1)，固定版本 `d0a90bba8885da7f445ea31646ab646d56086483` 和权重 SHA-256。它是非官方解码器微调，沿用 Qwen Research License，用于处理原版 VAE 的周期网格。图像直接从解码输出保存，不叠加自动去网格滤镜；现有网格检查仍拒绝有伪影的候选，不能降低门槛冒充修复。模型清单登记实际文件与来源，不在推理中自动下载。

显式原生局部编辑按 `source_canvas_roi` 传递坐标，编码器使用 `resolution=0` 保持已声明的1024画布；全局2K默认不会改变局部回写坐标。独立效果画布由效果请求自己的尺寸合同控制。

新 scope 的完整生成外观不再套用旧默认 Beauty、TBS 和补边链；有效的已有参考任务仍按其合同准备和冻结原生后期资产。两条输入路线都保留源图绑定、预算检查、原子保存和独立的质量状态。失败、取消、超时和未知结果保留当前父版本；未知外部调用不隐式重发，也不因文件保存成功授予人工批准。

显式使用 Beauty 的任务只有 `nunchaku_cuda` 后端。模型、runtime、provider 和输出路径由当前配置绑定；缓存命中仍验证代码、模型、输入和执行 profile。当前完整流程的冷/暖耗时见 [当前性能记录](WORKFLOW_PERFORMANCE.md)。

## 上游来源

- [Qwen Image 2.1](https://github.com/QwenLM/Qwen-Image-2.1)
- [Comfy-Org 适配权重](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)
- [Comfy-Org 官方编辑模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json)：CFG 1 与 base Euler 采样说明。
- [Diffusers Qwen 2.1](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21)：统一编辑/文生图的40步、无引导默认；负向提示与 CFG 的关系。
- [GGUF 发布者用法](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized)：复用官方工作流并替换加载器。
- [ComfyUI-GGUF](https://github.com/leejet/ComfyUI-GGUF)
- [Qwen Research License](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE)
