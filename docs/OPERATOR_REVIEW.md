# Appearance operator review

100% 局部检查确认了两个不同的问题：

1. `DSC02634` 的旧 polygon 轮廓把抬手与肩部之间的背景填进了主体，也覆盖了头发上方的部分背景。
2. GPT 参考图与原片不能保证逐像素重合。即使做了平移，对脸、发丝、手部和新光效的生成式结构改动仍然存在；把两张图在这些区域按高权重直接平均，会产生双影。

当前流程的修正是：

- proxy 上用边缘相关搜索轻量平移，并把偏移记录到 metadata；
- `make_mask.py` 使用更紧的 DSC02634 轮廓；
- mask 默认采用结构安全权重：头发、脸和上半身区域 alpha 接近 0，稳定服装与身体纹理区域逐步升到约 0.72；
- 外部 alpha 用于最终融合，appearance operator 默认用对齐后的完整画面拟合（`--fit-full`）；
- `--fit-alpha` 仅保留给需要严格限制拟合区域的实验。

实测 `outputs/final_demo/DSC02634/comparison.jpg` 的脸部和轮廓光已回到 GPT 参考图的外观，身体与服装仍能写回高清纹理；比 subject-only fit 或粗 polygon mask 更接近人工成片。

当前仍不能自动恢复 GPT 对人物的不可控重绘。对于这类区域，正确策略是把 alpha 降低，让 GPT 下层完整保留，再用一次人工 mask 修正边界。
