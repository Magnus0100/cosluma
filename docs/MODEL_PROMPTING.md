# 图像模型提示词适配

共享创作目标、角色信息和用户限制；分别为实际图像模型编译编辑指令。接口、参考图和采样设置同样影响结果，不能把历史不同请求的差异都归因于模型能力。

## 当前实现与缺口

带 `editing_scope` 的新任务由 `photo_art_direction` 观察实际原图并保存结构化设计，再生成当前模型的编辑指令；`reference_provider.qwen_prompt_recipe` 记录并原样转发该指令，没有调用 Qwen 官方提示词增强器。普通原片及导入参考工具按各自已检查用途运行，不借用其它任务的视觉观察证据。

主提示词中重复保留项、双语模板、抽象审美词和内部合同说明需要分清职责。用户原文及精确重试不能被静默删改。适配应针对系统管理的创作计划，冻结请求目标与实际发送文本，并保留两个身份。

## GPT Image 2.5 Sunburst

当前 provider 注册表的远程图像目标为 `gpt-image-2.5-sunburst`；经过兼容网关时，请求模型名不单独证明网关真实执行的模型。

按场景、主体、具体修改、保留项组织，复杂任务可以分节。明确要改变的光影、材质、氛围与限制；给每张参考图一个明确职责。继续编辑时一次针对一个主要问题，并重申重要保留项。

`size`、输出格式及接口支持的质量设置属于请求参数。当前适配器未显式发送 `quality`；对照时记录实际参数和端点行为，不能把提示词中的“高清”当成 API 的高质量设置。

## Qwen Image 2.1 编辑

使用 I2I 编辑指导，不把 T2I 的成片描述模板直接当成编辑模板。官方提供独立的 `Qwen-Image-2.1-PE-I2I` 和 `PE-T2I`；编辑增强器同时读取用户要求与输入图像。

编辑指令以操作开始，把目标变化写为可观察的光线、对象、材质和空间关系。保留项锁定身份和内容，不削弱已经要求的变化。对于未要求改变的部分，用简洁的保留声明，避免不断重新描述其外观而引起重生成。也不要因为保留要求多就把全部改动缩成轻微调色。

官方 PE 编辑输出为一段连续文字。中文要求输出中文描述，英文要求输出英文描述，其他语言按其官方策略处理；这不构成“中文必然比英文画质好”的实验结论。单图自然指代；多图使用 `<image1>`、`<image2>` 等标记，并逐一说明画布、身份和样式来源。让真实输入图确定未明确要求改变的画幅。

官方“单段文字”是其增强器的输出约定，不是证明换行会让基础模型失效，也不是提示词越短越好的规则。仅压缩字符不能替代观察图片和做明确设计决定。

当前内置 base 编辑工作流保留 40 步、CFG 3 和默认空负向提示；显式自定义工作流值由请求冻结，适配器不擅自重写。当前配置不代表所有图片或模型变体的最佳参数，修改策略需做受控对照。

继续编辑只组合本轮修改指令与必要保留项，避免重复旧计划或保留正要修改的受光。效果以实际候选和独立审片为准。

## 当前 AI harness 与验证

`用户创作意图 + 实际原图观察 → 现有 generation_design 结构化设计 → 模型专属编辑指令 → provider 执行 → 原图/生成参考/最终图反馈`

复用现有设计、工具、任务预算和资产身份，不另建生产入口。提示词规划器、图像生成器和独立评价分别承担职责。官方 PE 是提示词工具，本身不等于完整 agent runtime。新增模型或外部调用须具有明确运行配置与任务授权，不能以纯离线预检为名隐式调用。

接入前先保存当前默认请求作基线。第一轮在同一模型上只改变系统提示词策略，固定输入字节、模型权重、VAE、分辨率、采样参数和种子；再验证不同图片。GPT 与 Qwen 的比较共享创作目标和输入，并记录各自实际发送的文本与 API 设置。生成参考和最终后期成片分别评价，以定位生成能力和后期损失。

## 官方来源

- [OpenAI 图像提示词指南](https://developers.openai.com/api/docs/guides/image-prompting)：当前 2.5 参数、编辑、参考职责和比较方法。
- [OpenAI 图像生成提示词 Cookbook](https://developers.openai.com/cookbook/examples/multimodal/image-gen-models-prompting-guide)：结构、具体视觉描述和保留要求；其中模型迁移建议须结合当前指南，不把早期 GPT Image 2 的默认结论用于替代当前 2.5。
- [Qwen 2.1 官方项目](https://github.com/QwenLM/Qwen-Image-2.1)：模型与增强器入口。
- [Comfy-Org 2.1 编辑模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json)及[Diffusers 2.1 采样说明](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21)：base CFG 1、40步与负向提示的采样依据。
- [Qwen 官方 prompt rewrite](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite)：I2I/T2I 模型、调用及输出字段。
- [Qwen 官方编辑改写 system prompt](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_edit.txt)：编辑强度、保留、多图标记和语言策略。

以上是官方建议与当前实现审查；新策略尚需实际对照，不据文档直接宣称质量提高。
