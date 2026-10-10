# 当前参考图服务

Provider 位于 `providers/registry.json`，公开连接由 `provider_settings.py` 管理。密钥来自本机凭据，不写工程、来源文件或日志。

`reference_provider.py` 执行图像请求，`generation_design.py/generation_prompt.py` 形成完整设计与文本。已选参考可直接准备，无须重复生成。当前参考由工程选择和哈希决定，没有某张样图作为全局默认。

Qwen Image 2.1 提供官方 BF16、指定社区 BF16 GGUF、INT8/Q6_K。文字规划与图像服务独立，见 [QWEN_IMAGE_LOCAL.md](QWEN_IMAGE_LOCAL.md)。

生成记录 Provider、模型、发送文本、源图、工作流、请求和输出。成功仅登记候选，不能跳过当前准备、源基线或质量门；失败保留输入版本，不以旧 API 升级、旧计划或调色替代新参考。

参考生成纠错走完整当前QA绑定的多参考masked请求，原生staging与实际成片复审通过后才继续融合。生成、仿色、背景、独立效果和视觉接受分别记录。

已保存成片的本地恢复由 `POST /api/source-runs/<id>/repair-final` 执行：取得新的当前绑定QA，审查材质或背景资格，有资格时保存独立修复版本并复审。该任务的参考与独立效果生图预算为零；未支持或资格不足的问题保留。请求与纠错合同见 [GENERATION_PROMPT.md](GENERATION_PROMPT.md)，当前入口见 [REVISION_API.md](REVISION_API.md)。
