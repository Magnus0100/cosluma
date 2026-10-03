# Cosplay 高分辨率后期自动化

这个仓库用于把已经确认的 GPT 后期参考图，落回同一张高清原片的真实纹理上。

目标输出满足：

- 保持原 coser、姿态、动作和主要几何关系；
- 保留 GPT 参考图确认过的调色、光影、背景和特效设计；
- 在可信区域补回高清原片的真实纹理；
- 最多通过一次简单的 alpha mask 确认或涂刷完成交接。

当前仓库处于**素材整理与最小实验准备阶段**，还没有实现完整应用或 GUI。

## 当前目录

```text
.
├─ data/       原始测试素材（大文件，不在代码中复制）
├─ docs/       项目约束、素材台账、实验计划和状态
├─ outputs/    实验输出（默认不纳入版本控制）
├─ scripts/    后续最小实验脚本
└─ .cache/     本地缓存（不纳入版本控制）
```

## 输入约定

每组样例按相同编号组织。当前文件名的角色是根据尺寸和交接信息作出的**暂定判断**，正式实验前需要用人工最终合成图再次确认：

- `*.jpg`：高清原片候选；
- 同编号、无“参考”后缀的 `*.png`：人工最终合成图候选；
- `*参考.png`：GPT 后期参考图候选。

详细清单见 [`docs/DATASET.md`](docs/DATASET.md)。

## 推荐实验顺序

1. 在 proxy 尺寸完成几何一致性检查；
2. 先验证 BGU / operator upsampling 与传统追色 baseline；
3. 再加入单一 matting backend 和传统 alpha refinement；
4. 最后实现两层融合、16 bit 输出和一次 mask 确认。

约束与决策记录见 [`docs/PROJECT.md`](docs/PROJECT.md)，可执行计划见 [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md)。

## 本地开发

当前阶段只需要 Python、图像查看工具和 Git。依赖尚未锁定，避免在尚未完成模块验证前引入大型模型或复杂运行时。

后续加入脚本后，应优先使用 proxy 做分析，只有最终 operator 应用与合成才处理全分辨率图片。

## 当前 Demo

最小 Demo 已实现于 [`scripts/run_demo.py`](scripts/run_demo.py)，使用 Pillow/NumPy 完成局部 affine appearance transfer、参考图放大、软 alpha 和两层合成。运行方式与输出说明见 [`scripts/README.md`](scripts/README.md)。

两组样例的当前结果位于 `outputs/final_demo/`，其中四联对比图和 `structure_mismatch.png` 最适合快速判断流程是否成立。使用外部 alpha 时，按样例分别传入对应的 `outputs/mask_backend/<sample>/alpha.png`。

需要替换软椭圆时，按 [`docs/MASK_WORKFLOW.md`](docs/MASK_WORKFLOW.md) 生成并人工修正一次灰度 mask，再传给 `--alpha` 和 `--fit-alpha`。质量结论见 [`docs/MASK_QUALITY_REVIEW.md`](docs/MASK_QUALITY_REVIEW.md)，matting 依赖调查见 [`docs/MATTING_OPTIONS.md`](docs/MATTING_OPTIONS.md)。

