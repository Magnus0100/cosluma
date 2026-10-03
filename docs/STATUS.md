# 当前状态

更新时间：2026-10-04

## 已完成

- 完成两组素材分析和最小两层合成 Demo。
- 实现可编辑 polygon mask、结构安全 confidence profile 与 `--fit-full`。
- proxy 上增加轻量边缘平移对齐，并把偏移记录到 metadata。
- proxy 上增加受限 scale + translation 联合搜索；获胜尺度、平移范围和分数写入 metadata。
- 每次 Demo 额外输出 `structure_mismatch.png`，并记录归一化亮度边缘分歧统计，用于区分可解释几何误差和 GPT 结构变化。
- 新增独立 `scripts/structure_diagnostic.py`，可只生成诊断图和 JSON，不重复渲染高清结果。
- 新增 `scripts/edit_mask.py`：Tkinter 低交互画笔编辑器，支持左键增加 alpha、右键擦除、滚轮调整笔刷、S 保存。
- 新增无窗口 CLI 修补入口，支持 `--add`、`--erase` 或 `--points-file`。
- 新增 [MASK_WORKFLOW.md](MASK_WORKFLOW.md)，记录生成、编辑、运行和检查步骤。
- 完成 [MASK_QUALITY_REVIEW.md](MASK_QUALITY_REVIEW.md)：比较软椭圆、polygon 和编辑后 mask 的覆盖率与成片表现。
- 完成 [MATTING_OPTIONS.md](MATTING_OPTIONS.md)：确认当前环境没有现成 matting 依赖，也没有下载模型。
- 已使用编辑后的 DSC02634 mask 跑通完整 Demo；metadata 记录 `alpha.path`、对齐偏移和 `fit_trust`。

## 当前判断

GPT 参考图存在两类差异：可由轻量平移处理的几何偏移，以及生成模型造成的脸、发丝、手部、服装边缘和新光效结构变化。当前最稳定的组合是：对齐后的完整画面拟合外观，结构安全 alpha 控制高清纹理写回。两组样例的剩余问题集中在发丝、手指尖和逆光轮廓的少量半透明像素。

当前不需要下载或接入 matting backend。现有一次人工 mask 已经完成单张样例的交接；模型会增加环境、显存和调参成本，当前数据尚未显示必须依赖模型才能修复的大片错误。

联合对齐的结果也支持这一判断：`DSC02631` 保持 1.00 倍、零平移；`DSC02634` 仍以约 1.00 倍和明显的垂直平移为主。说明整体尺度/位置差异可以单独处理，剩余高分歧区域主要来自 GPT 的局部重绘和逆光结构，应该通过较低 alpha 保留 GPT 下层。

## 下一步

如果进入批量生产，再以 DSC02634 为 A/B 样本评估单一 ONNX matting backend；触发条件是 100% 检查中人工修边持续超过约 30 分钟，或必须保留飞发、半透明饰物。当前继续使用 `scripts/edit_mask.py`、`--alpha ... --fit-full`，先看 `structure_mismatch.png` 再修边，只有实验时才使用 `--fit-alpha`。
