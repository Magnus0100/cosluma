# 当前状态

更新时间：2026-10-04

## 已完成

- 初始化本地 Git 仓库。
- 保留并盘点 `data/` 中两组共六张图片。
- 记录每个素材的尺寸、文件大小、ICC 元数据检测结果和 SHA-256。
- 建立项目约束、素材台账、分析文档和十小时 Demo 计划。
- 完成两组素材的并排视觉检查和结构/色彩诊断。
- 实现 `scripts/run_demo.py`：局部 affine appearance transfer、参考图放大、软 alpha 和两层合成。
- 实现 `scripts/make_mask.py`：两组样例的可编辑归一化多边形 mask。
- 为 operator 增加 `--fit-alpha`，可只用主体可信区域拟合 appearance operator，减少背景粒子和太阳光对人物增益的影响。
- 两组样例均已生成默认 Demo 和 polygon mask Demo，输出在 `outputs/demo/`、`outputs/demo_polygon/`。

## 当前判断

数据已经足够支撑第一版 Demo。DSC02631 的 polygon mask 交接自然；DSC02634 使用扩大头发外缘并增加羽化后的 mask 后，头顶双轮廓明显减轻。主体纹理与 GPT 背景光效的两层交接已经可以继续迭代。

## 已知限制

- polygon mask 是可编辑的传统轮廓，不是发丝级 matting；半透明发丝、轮廓光和薄纱仍需要人工修正。
- appearance operator 是可解释的局部 RGB affine baseline，不负责生成原片中不存在的光效和粒子。
- 输出当前为 PNG RGB；正式高质量流程仍需补齐 linear/ICC/16 bit 管理。
- 未下载大型模型，避免让 Demo 被环境安装阻塞。

## 下一步

优先把 polygon mask 做成一次人工刷选入口，并在实际修正后的 mask 上使用 `--fit-alpha`。只有 mask 仍明显影响成片，才评估单一 matting backend；暂不扩大测试框架或下载大模型。
