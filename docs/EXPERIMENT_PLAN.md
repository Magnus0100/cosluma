# 最小实验计划

## 阶段 0：素材确认

1. 逐组并排查看 JPG、无后缀 PNG 和“参考” PNG。
2. 确认三类角色和是否存在裁切、旋转、缩放或明显导出色差。
3. 建立低分辨率 proxy，并记录从 proxy 到原图的坐标映射。

产物：角色确认记录、proxy、几何对齐报告。

## 阶段 1：只验证 appearance transfer

先不引入 matting。使用人工给定的可信区域或简单中心区域，比较：

- BGU / bilateral guided upsampling；
- Reinhard；
- 线性 histogram matching；
- 其它低成本 baseline。

输入为 `Original_low → GPT_reference`，再将估计出的 operator 应用到 `Original_high`。与原片、GPT 参考图和人工成片并排查看肤色、服装颜色、局部明暗、环境色、纹理保持、halo 与异常映射。

产物：每种 backend 的高分辨率结果、参数和对比图。

## 阶段 1.5：可解释几何与结构分歧

在进入 matting 前，先把整体缩放和平移与 GPT 局部重绘分开处理：

- proxy 上搜索小范围 scale + translation，并把获胜变换应用到参考图；
- 输出 `structure_mismatch.png`，用归一化亮度边缘分歧标出脸、发丝、手部和服装边缘等不宜强行写回的位置；
- 用 `alpha` 决定高清纹理回写，保留高分歧区的 GPT 结构和新光效。

产物：两组样例的联合对齐 metadata、结构诊断图、四联对比图。当前结果已完成；后续只在诊断图显示大片稳定误差时扩大搜索范围。

## 阶段 2：alpha 与边缘

在 transfer 基本成立后，使用可编辑 polygon 作为零依赖后端，并提供可选 U2Net human segmentation 作为轮廓后端；再用 confidence profile、structure-safe 调制和一次人工修边做 refinement。重点检查发丝、手指、脸部、衣服边缘和 GPT 前景特效。U2Net 只负责 silhouette，不能解决 GPT 局部重绘造成的身份/结构不重合。

产物：初始 alpha、U2Net A/B、refined alpha、少量人工刷 mask 的验证结果。当前两组样例均已完成。

## 阶段 3：两层合成与交付

实现：

```text
HighResStyledOriginal
+ UpscaledGPT
+ alpha
→ Final
```

生产入口 `scripts/build_final.py` 已完成全分辨率两层合成、ICC 保留、manifest 和 metadata。当前工作空间仍以 sRGB 8-bit 计算；16-bit/linear-light 处理列为下一次质量升级，不能把当前结果标记为 16-bit 成片。

## 暂不实现

暂不做训练模型、diffusion 主干、dense neural correspondence、多模型 ensemble、自动 QC 拒收、复杂插件系统和自研 LUT 系统。GUI 只承担人工 mask 修边和复核。

## 单次升级原则

如果失败集中在发丝，换 matting/refinement；集中在追色，换 appearance-transfer backend；集中在结构变化，改 alpha/trusted region；集中在 GPT 下层模糊，再评估非生成式 upscale。一次只升级一个实际造成返工的模块。
