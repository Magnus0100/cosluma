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

## 阶段 2：alpha 与边缘

在 transfer 基本成立后，只加入一个 matting backend，优先 BiRefNet HR-matting 或同等级单模型；再用 erosion/dilation、guided filter、distance transform 和 feather 做传统 refinement。重点检查发丝、薄纱、脸部、衣服边缘和 GPT 前景特效。

产物：初始 alpha、refined alpha、少量人工刷 mask 的验证结果。

## 阶段 3：两层合成

实现：

```text
HighResStyledOriginal
+ UpscaledGPT
+ alpha
→ Final
```

全流程明确 ICC、linear float 和 16 bit 输出，并和人工成片比较实际返工量。

## 暂不实现

暂不做完整 GUI、训练模型、diffusion 主干、dense neural correspondence、多模型 ensemble、自动 QC 拒收、批处理调度、复杂插件系统和自研 LUT 系统。

## 单次升级原则

如果失败集中在发丝，换 matting/refinement；集中在追色，换 appearance-transfer backend；集中在结构变化，改 alpha/trusted region；集中在 GPT 下层模糊，再评估非生成式 upscale。一次只升级一个实际造成返工的模块。
