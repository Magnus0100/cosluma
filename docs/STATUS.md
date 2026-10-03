# 当前状态

更新时间：2026-10-04

## 已完成

- 初始化本地 Git 仓库。
- 保留并盘点 `data/` 中两组共六张图片。
- 记录每个素材的尺寸、文件大小、ICC 元数据检测结果和 SHA-256。
- 建立项目约束、素材台账、分析文档和十小时 Demo 计划。
- 完成两组素材的并排视觉检查和结构/色彩诊断。
- 实现 `scripts/run_demo.py`：局部 affine appearance transfer、参考图放大、软 alpha 和两层合成。
- 实现 `scripts/validate_outputs.py`：轻量检查输入与生成文件是否可读。
- 两组样例均已成功生成 `outputs/demo/` 下的 Demo 结果和四联对比图。

## 当前判断

数据已经足够支撑第一版 Demo。DSC02631 适合验证温和的 appearance 变化；DSC02634 作为强逆光、粒子和局部亮度变化的压力样本。当前输出已经能保留原片人物纹理，同时让 GPT 背景光效进入最终图，但 alpha 仍是软椭圆，不能视为最终 matting。

## 已知限制

- 默认 alpha 是可替换的软主体信任图，不是发丝级分割。
- appearance operator 是可解释的局部 RGB affine baseline，不负责生成原片中不存在的光效和粒子。
- 输出当前为 PNG RGB；正式高质量流程仍需补齐 linear/ICC/16 bit 管理。
- 未下载大型模型，避免让 Demo 被环境安装阻塞。

## 下一步

优先把 `alpha.png` 替换为一次人工刷选或单一 matting backend，比较是否减少 halo 和背景污染；只有该问题真实影响成片后，再升级 appearance backend 或色彩管理。
