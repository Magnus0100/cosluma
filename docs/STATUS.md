# 当前状态

更新时间：2026-10-04

## 已完成

- 完成两组素材分析和最小两层合成 Demo。
- 实现可编辑 polygon mask、结构安全 confidence profile 与 `--fit-full`。
- proxy 上增加轻量边缘平移对齐，并把偏移记录到 metadata。
- proxy 上增加受限 scale + translation 联合搜索；获胜尺度、平移范围和分数写入 metadata。
- 每次 Demo 额外输出 `structure_mismatch.png`，并记录归一化亮度边缘分歧统计，用于区分可解释几何误差和 GPT 结构变化。
- 新增独立 `scripts/structure_diagnostic.py`，可只生成诊断图和 JSON，不重复渲染高清结果。
- 新增 `scripts/build_final.py` 生产入口：每组样例独立 mask、对齐、结构安全合成、ICC 保留、manifest 和 metadata。
- 新增可选 `scripts/u2net_mask.py` 人体轮廓后端；U2Net 只负责 silhouette，仍由 confidence profile、structure-safe 和人工修边决定最终写回范围。
- 已确认 `data/*.jpg` 是像素蛋糕追色后的旧实验输入；生产流程改用外部目录 `C:\Users\80959\Desktop\新建文件夹\新建文件夹 (7)` 中的真正原片，并通过 `--source-dir` 记录来源。
- 新增 `scripts/edit_mask.py`：Tkinter 低交互画笔编辑器，支持左键增加 alpha、右键擦除、滚轮调整笔刷、S 保存。
- 新增无窗口 CLI 修补入口，支持 `--add`、`--erase` 或 `--points-file`。
- 新增 [MASK_WORKFLOW.md](MASK_WORKFLOW.md)，记录生成、编辑、运行和检查步骤。
- 完成 [MASK_QUALITY_REVIEW.md](MASK_QUALITY_REVIEW.md)：比较软椭圆、polygon 和编辑后 mask 的覆盖率与成片表现。
- 完成 [MATTING_OPTIONS.md](MATTING_OPTIONS.md)：确认当前环境没有现成 matting 依赖，也没有下载模型。
- 已使用编辑后的 DSC02634 mask 跑通完整 Demo；metadata 记录 `alpha.path`、对齐偏移和 `fit_trust`。
- 新增 `--structure-safe`：在 polygon/confidence alpha 内部叠加 proxy 结构分歧调制。高分歧边缘自动降低高清原片写回权重，保持 GPT 下层结构；不扩大遮罩、不依赖模型。压力样本 DSC02634 已生成 A/B 对比：`outputs/structure_safe_dsc02634/DSC02634/structure_safe_ab.jpg`。

## 当前判断

GPT 参考图存在两类差异：可由轻量平移处理的几何偏移，以及生成模型造成的脸、发丝、手部、服装边缘和新光效结构变化。对真实原片实测后，当前最稳定的组合是：人物可信 alpha 拟合外观，结构安全 alpha 控制高清纹理写回。两组样例的剩余问题集中在发丝、手指尖和逆光轮廓的少量半透明像素。

完整版本默认仍不强制下载模型：已有人工 mask 或 polygon fallback 可以直接生产。对批量素材可显式启用 U2Net，它改善粗轮廓和抬手外缘，但不会解决 GPT 重绘造成的人脸、手指和飞发不重合；这些区域仍需结构安全 alpha 和人工修边。真实原片 A/B 已证明，人物可信区域拟合比整幅拟合更能避免青色原片被映射成红偏。

联合对齐的结果也支持这一判断：`DSC02631` 保持 1.00 倍、零平移；`DSC02634` 仍以约 1.00 倍和明显的垂直平移为主。说明整体尺度/位置差异可以单独处理，剩余高分歧区域主要来自 GPT 的局部重绘和逆光结构，应该通过较低 alpha 保留 GPT 下层。

## 下一步

当前完整版本的验收入口是 `scripts/build_final.py --source-dir "C:\Users\80959\Desktop\新建文件夹\新建文件夹 (7)"`。先使用已有 mask 生成真实原片的交付 manifest；需要更自动的轮廓时，再用 `--segmentation-backend u2net --force-segmentation` 做 A/B。无论使用哪种 backend，都要先看 `structure_mismatch.png` 再修边；`--fit-full` 只用于兼容旧实验。
