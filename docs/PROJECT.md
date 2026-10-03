# 项目约束与决策

## 问题定义

本项目不是重新生成 cosplay 图，也不是单纯超分辨率。它要把 GPT 已确认的低分辨率美术结果，迁移到同一张高清原片的可信真实纹理上。

最终结构固定为两层和一个 alpha：

```text
Final = alpha * HighResOriginalStyled + (1 - alpha) * UpscaledGPT
```

其中 alpha 表示“高清原片信息允许写回最终图的权重”，不是传统人物分割的同义词。

## 不可推翻的约束

- GPT 图中的妆容、皮肤、光影、背景、特效和材质设计优先保留。
- 原片主要提供高频纹理与真实细节，不用生成式方法重画人物主干。
- GPT 下层第一版使用 bicubic 或 Lanczos 放大即可。
- 一次只处理一张照片，不引入队列、批处理调度或分布式推理。
- 模块通过 backend 接口可替换：alignment、matting、appearance transfer、upscale、compositor、color manager。
- 主工作空间使用 linear float，入口和出口明确处理 ICC，优先输出 16 bit。
- 先集成成熟算法；只有真实样例证明传统方法不足时才增加神经网络或更复杂模型。

## 处理策略

### 适合 operator transfer 的内容

调色、局部明暗、环境色、对比度、色温、肤色影调等 appearance modification，优先验证 BGU / bilateral guided upsampling，并保留便宜的传统 baseline 作为对照。

### 应交给 GPT 下层的内容

新增火焰、烟雾、魔法阵、物件、重画服装花纹、全新头发或被前景遮挡的区域，属于 new content / structural modification。不要让颜色映射模型强行复现这些内容，应通过降低 alpha 保留 GPT 结果。

### 对齐

默认认为两张图同构图。优先 phase correlation、ECC，以及必要时基于边缘或梯度的 SIFT/ORB + RANSAC fallback；不把 dense neural correspondence 作为正常路径。

### alpha

稳定服装纹理、饰品和身体等可信区域趋近 1；GPT 新背景、新特效和结构变化区域趋近 0；发丝、薄纱、脸部美化区域使用中间值并通过 refinement 处理边缘 halo。

## 成功标准

不要求与人工成片像素一致。需要回答：最终是否仍是原来的 coser，GPT 美术感觉是否保留，原片纹理是否回来，人物与背景是否自然，发丝和边缘是否可接受，以及人工是否只需少量 mask 修改。
