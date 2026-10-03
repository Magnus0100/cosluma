# 当前状态

更新时间：2026-10-04

## 已完成

- 完成两组素材分析和最小两层合成 Demo。
- 实现可编辑 polygon mask 与 `--fit-alpha` 主体可信区域拟合。
- 新增 `scripts/edit_mask.py`：Tkinter 低交互画笔编辑器，支持左键增加 alpha、右键擦除、滚轮调整笔刷、S 保存。
- 新增无窗口 CLI 修补入口，支持 `--add`、`--erase` 或 `--points-file`。
- 新增 [MASK_WORKFLOW.md](MASK_WORKFLOW.md)，记录生成、编辑、运行和检查步骤。
- 已使用 CLI 编辑后的 DSC02634 mask 跑通完整 Demo；metadata 记录 `alpha.path` 和 `fit_trust`。

## 当前判断

项目已经具备一次人工 mask 确认的最小操作闭环：生成起始 mask → 画几笔修边 → 使用同一 mask 做合成和 operator 拟合。这个闭环比固定软椭圆更接近实际后期工作流。

## 已知限制

- 当前编辑器是低交互工具，不提供图层、撤销历史或发丝级 matting。
- polygon mask 仍需人工确认头发、手部、轮廓光和半透明饰物。
- 输出当前为 PNG RGB；正式高质量流程仍需补齐 linear/ICC/16 bit 管理。
- 暂未下载大型模型；当前数据还没有证明需要它。

## 下一步

用真实人工修正的 DSC02634 mask 做一次视觉验收；如果边缘仍需大量刷改，再评估单一 matting backend。继续保持单张、单次人工确认和可替换 backend 的范围。
