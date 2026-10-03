# Cosplay 高分辨率后期自动化

这个仓库用于把已经确认的 GPT 后期参考图，落回同一张高清原片的真实纹理上。

目标输出满足：

- 保持原 coser、姿态、动作和主要几何关系；
- 保留 GPT 参考图确认过的调色、光影、背景和特效设计；
- 在可信区域补回高清原片的真实纹理；
- 最多通过一次简单的 alpha mask 确认或涂刷完成交接。

当前仓库已经有可交接的完整批处理版本：自动配准、独立 alpha、结构安全融合、全分辨率输出和 manifest 都已实现。GUI 仍用于人工修边，不把人工修边假设成全自动。

## 当前目录

```text
.
├─ data/       原始测试素材（大文件，不在代码中复制）
├─ docs/       项目约束、素材台账、实验计划和状态
├─ outputs/    实验输出（默认不纳入版本控制）
├─ scripts/    生产入口、mask 后端和诊断脚本
└─ .cache/     本地缓存（不纳入版本控制）
```

## 输入约定

每组样例按相同编号组织。当前文件名的角色是根据尺寸和交接信息作出的**暂定判断**，正式实验前需要用人工最终合成图再次确认：

- `*.jpg`：高清原片候选；
- 同编号、无“参考”后缀的 `*.png`：人工最终合成图候选；
- `*参考.png`：GPT 后期参考图候选。

详细清单见 [`docs/DATASET.md`](docs/DATASET.md)。

## 完整版本入口

使用生产入口生成每组样例的全分辨率结果：

```powershell
$py = "C:\Users\80959\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py scripts/build_final.py --alpha-dir outputs/mask_review --out-dir outputs/final
```

入口会独立搜索 scale + translation，默认开启结构安全 alpha，并写出每组 `metadata.json` 和总 `manifest.json`。已有 mask 优先；没有 mask 时使用 polygon fallback。需要更自动的人物轮廓时，显式传 `--segmentation-backend u2net --force-segmentation`，详见 [`scripts/README.md`](scripts/README.md)。

当前正式交付候选位于 `outputs/final_delivery/`；U2Net A/B 候选位于 `outputs/final_u2net/`。

约束与决策记录见 [`docs/PROJECT.md`](docs/PROJECT.md)，可执行计划见 [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md)。

## 本地开发

默认路径只需要 Python、Pillow、NumPy 和 Git。U2Net 是可选依赖，需要隔离的 onnxruntime 和约 176 MB 模型；不启用时不会加载它。

后续加入脚本后，应优先使用 proxy 做分析，只有最终 operator 应用与合成才处理全分辨率图片。

## 低层 Demo 与质量检查

最小 Demo 已实现于 [`scripts/run_demo.py`](scripts/run_demo.py)，使用 Pillow/NumPy 完成局部 affine appearance transfer、参考图放大、软 alpha 和两层合成。运行方式与输出说明见 [`scripts/README.md`](scripts/README.md)。

两组样例的生产结果位于 `outputs/final_delivery/`，其中四联对比图和 `structure_mismatch.png` 用于质量检查。`outputs/final_demo/` 保留低层实验结果。使用外部 alpha 时，按样例分别传入对应的 mask 路径。

需要替换软椭圆时，按 [`docs/MASK_WORKFLOW.md`](docs/MASK_WORKFLOW.md) 生成并人工修正一次灰度 mask，再传给 `--alpha` 和 `--fit-alpha`。质量结论见 [`docs/MASK_QUALITY_REVIEW.md`](docs/MASK_QUALITY_REVIEW.md)，matting 依赖调查见 [`docs/MATTING_OPTIONS.md`](docs/MATTING_OPTIONS.md)。

