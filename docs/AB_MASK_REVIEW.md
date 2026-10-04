# DSC02634 真实原片 Mask A/B

更新时间：2026-10-04

本次 A/B 固定真实原片目录、scale/translation 搜索、appearance operator、residual 参数和 `structure-safe`，只替换 alpha 来源：

- A：`outputs/mask_review/DSC02634/alpha.png`（人工修正 polygon）；
- B：U2Net human silhouette + 同一 confidence profile，输出到 `outputs/ab_u2net/DSC02634/_u2net_alpha.png`。

运行命令记录在 [ALPHA_ACCEPTANCE.md](ALPHA_ACCEPTANCE.md)。模型文件 `.cache/models/u2net_human_seg.onnx`（约 176 MB）和隔离运行时 `.cache/onnxruntime` 均可用，两组输出均完成全分辨率渲染。

## 可复现统计

两次运行得到相同对齐：scale `1.01`、shift `(-2, 24)`，结构分歧 p95 `0.329`，结构分歧大于 0.5 的 proxy 像素比例 `2.05%`。

| 指标 | Polygon A | U2Net B |
|---|---:|---:|
| alpha 平均值 | 0.2108 | 0.1481 |
| alpha > 0.5 面积 | 22.16% | 19.66% |
| alpha > 0.9 面积 | 20.37% | 0% |
| A/B `final.png` 平均绝对差 | — | 1.90/255（95 分位 10/255） |

U2Net 经过 confidence profile 后，头部和发丝区域保持低写回权重，身体上限约 0.72；因此它不是“白色人物分割图”，而是更保守的 GPT 保留策略。

## 人工局部检查

- 两个版本的人物身份、服装几何、手臂位置和背景直线都保持一致；没有发现新增肢体或大面积错位。
- 头顶飞发、右侧抬手和肩部逆光均没有连续彩色 halo 或硬白边。U2Net 的 GPT 发丝轮廓光略完整，Polygon 的脸部和衣服高频纹理略多。
- U2Net 的脸部更接近 GPT 参考图的平滑美化；Polygon 更接近高清原片的真实皮肤/衣服纹理。两者差异集中在可信权重，不是结构错误。
- 右上太阳光、地面反射、粒子和雾化背景在两个版本中都保留，说明 structure-safe 没有吞掉 GPT 下层效果。

## 决策

这次 A/B 没有显示 U2Net 在成片质量上明确胜出：它降低了原片写回面积，也没有消除 Polygon 已经通过的主要边缘问题。当前默认继续使用人工修正 Polygon；U2Net 保留为没有现成 mask 时的 silhouette 候选，或在未来批量素材上以“减少人工描轮廓时间”为唯一理由启用。

下一步应冻结这组视觉路径，转入 16 bit/linear-light/ICC 交付整理。若未来需要更细的飞发或半透明饰物，先在同一 A/B 框架中测单一 matting refinement，不串联多个模型，也不绕过一次人工 alpha 确认。
