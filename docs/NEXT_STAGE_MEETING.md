# 下一阶段专家会议纪要

更新时间：2026-10-04

## 会议结论

真实原片已经接入生产流程。当前 `alpha-trust` appearance fit 比 `fit-full` 更稳定，但四联图仍显示人物局部比 GPT/人工成片暗，尤其是 `DSC02634` 的头发轮廓光、肩部暖光和服装局部高光。

当前最值得投入的升级是：

```text
真实原片
  → alpha-trust affine operator
  → 低频 residual light transfer
  → structure-safe alpha
  → GPT reference 下层合成
```

残差 transfer 只迁移低频光影，不复制 GPT 的高频人物结构。它应在浮点 proxy 上计算，避免先转 8-bit 造成色阶和 halo；高分歧区域仍由 structure-safe alpha 降低高清原片写回。

## 各专家意见

### 色彩 / operator

- 真实原片上的 alpha-trust 主体 MAE 约为：`DSC02631 8.0/255`、`DSC02634 16.4/255`。
- `fit-full` 更差，约为 `10.7/255` 和 `20.0/255`，会把真实原片背景色偏带入人物。
- affine-only 无法生成 GPT 新增的低频暖光和轮廓光。
- 建议增加 `styled = affine(source) + beta * lowpass(reference - affine(source))`，默认 `sigma_proxy=2–4`、`beta=0.8–1.0`，并保留 affine-only 开关。

### 几何 / 配准

- 全局结果：`DSC02631 scale=1.00, shift=(0,0)`；`DSC02634 scale=1.01, shift=(-2,16)`。
- DSC02634 躯干稳定网格的局部位移约 1px proxy，头脸/背景位移更大且相关性低。
- 局部 warp 只允许作为躯干小范围实验，默认关闭；不能对脸、头发、手指和逆光轮廓做全图 TPS/mesh warp。
- 当前收益优先级低于 alpha 边缘和 residual light transfer。

### 遮罩 / 生产

- polygon 继续作为默认 alpha backend；U2Net 只提供 silhouette 候选。
- structure-safe 必须默认开启，且只能在已有 alpha 内部降权。
- 人工只负责脸、头发、手指、轮廓光和稳定服装边缘的最后确认。
- 交付硬门槛见 [`ALPHA_ACCEPTANCE.md`](ALPHA_ACCEPTANCE.md)。

## 下一阶段任务顺序

1. 实现低频 residual light transfer，并在两组真实原片上与 alpha-trust affine 做 A/B。
2. 用主体 MAE、白色马甲色偏、头发轮廓光保留和 100% halo 检查选择默认 beta/sigma。
3. 若 residual transfer 改善 DSC02634 且不破坏 DSC02631，接入 `build_final.py`，metadata 记录参数。
4. 再做真实原片 U2Net vs polygon 的人工修边时间 A/B；不以 mask 面积或 IoU 单独决定 backend。
5. 只有 residual + alpha 稳定后，才做躯干受限 local warp；脸、头发和手部 warp 暂不进入默认生产路径。

## 暂不做

- 不做全脸 landmark/TPS warp；
- 不继续叠加多个 matting 模型；
- 不把 `fit-full` 设为真实原片默认；
- 不把当前 sRGB 8-bit 输出宣称为最终 16-bit 色彩管理版本。
