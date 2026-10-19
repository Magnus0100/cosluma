# 本地 Qwen 入口

本地 Qwen 只作为明确授权的参考候选生成服务。生产任务必须先通过当前 task plan、输入身份、模型能力和预算检查；生成结果随后进入唯一的 `layered_reference_transfer` 后期流程。

提示词只描述当前用户要求、动作、位置、遮挡和期望结果。入口逐字发送用户指令，不自动加入固定色调、蓝雾、光束、角色或场景描述。CFG 1 时不发送负向提示词。

新工作区默认选择官方 BF16 扩散模型与 BF16 文本编码器。全局编辑使用 `resolution=2048` 的像素预算、40 步、Euler/simple、CFG 3；竖幅 2:3 实际生成约 1664×2496，不把原片高清回写称为模型原生生成。RAW 上传长边上限为4096，为2K画布保留足够条件细节。用户显式选择的低显存模型和自定义提示词继续保留。

模型存储位置与桌面程序位置分别配置。部署盘可能是机械盘，服务使用 `--disable-fast-disk`，权重换出优先复用主内存，并保留 ComfyUI 上游的限额锁页机制；不再一概关闭 Windows 锁页，也不使用 `--high-ram` 绕过上游限额。该策略不改变模型精度或采样设置。加载阶段仍需要读取模型，原尺寸缓存和渲染结果也仍会写入配置的工作盘；不能据此声称全程没有磁盘等待。

默认 VAE 为 [Texture-Fix VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1)，固定版本 `d0a90bba8885da7f445ea31646ab646d56086483` 和权重 SHA-256。它是非官方解码器微调，沿用 Qwen Research License，用于处理原版 VAE 的周期网格。图像直接从解码输出保存，不叠加自动去网格滤镜；现有网格检查仍拒绝有伪影的候选，不能降低门槛冒充修复。模型清单登记实际文件与来源，不在推理中自动下载。

显式原生局部编辑按 `source_canvas_roi` 传递坐标，编码器使用 `resolution=0` 保持已声明的1024画布；全局2K默认不会改变局部回写坐标。独立效果画布由效果请求自己的尺寸合同控制。

生成候选不会跳过源图绑定、Beauty、场景准备、fit、colour fusion、native baseline、scene balance、freeze、边界协调或人工审查。失败、取消、超时和未知结果保留当前父版本，并按任务合同决定是否允许重试。

当前 Beauty 只有 `nunchaku_cuda` 后端。模型、runtime、provider 和输出路径由当前配置绑定；缓存命中仍验证代码、模型、输入和执行 profile。当前完整流程的冷/暖耗时见 [当前性能记录](WORKFLOW_PERFORMANCE.md)。

## 上游来源

- [Qwen Image 2.1](https://github.com/QwenLM/Qwen-Image-2.1)
- [Comfy-Org 适配权重](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)
- [ComfyUI-GGUF](https://github.com/leejet/ComfyUI-GGUF)
- [Qwen Research License](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE)
