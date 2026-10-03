# 当前状态

更新时间：2026-10-04

## 已完成

- 完成两组素材分析和最小两层合成 Demo。
- 实现可编辑 polygon mask 与 `--fit-alpha` 主体可信区域拟合。
- 新增 `scripts/edit_mask.py`：Tkinter 低交互画笔编辑器，支持左键增加 alpha、右键擦除、滚轮调整笔刷、S 保存。
- 新增无窗口 CLI 修补入口，支持 `--add`、`--erase` 或 `--points-file`。
- 新增 [MASK_WORKFLOW.md](MASK_WORKFLOW.md)，记录生成、编辑、运行和检查步骤。
- 完成 [MASK_QUALITY_REVIEW.md](MASK_QUALITY_REVIEW.md)：比较软椭圆、polygon 和编辑后 mask 的覆盖率与成片表现。
- 完成 [MATTING_OPTIONS.md](MATTING_OPTIONS.md)：确认当前环境没有现成 matting 依赖，也没有下载模型。
- 已使用编辑后的 DSC02634 mask 跑通完整 Demo；metadata 记录 `alpha.path` 和 `fit_trust`。

## 当前判断

polygon/editor mask 明显优于固定软椭圆：主体有效区域从约 38% 收敛到约 24%，背景粒子和太阳光仍由 GPT 下层保留，人物衣服和身体纹理保持清楚。两组样例的剩余问题集中在发丝、手指尖和逆光轮廓的少量半透明像素。

当前不需要下载或接入 matting backend。现有一次人工 mask 已经完成单张样例的交接；模型会增加环境、显存和调参成本，当前数据尚未显示必须依赖模型才能修复的大片错误。

## 下一步

如果进入批量生产，再以 DSC02634 为 A/B 样本评估单一 ONNX matting backend；触发条件是 100% 检查中人工修边持续超过约 30 分钟，或必须保留飞发、半透明饰物。当前继续使用 `scripts/edit_mask.py` 和 `--alpha ... --fit-alpha ...`。
