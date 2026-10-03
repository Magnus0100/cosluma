# 十小时 Demo 计划

目标截止：从本计划开始计时的 10 小时内，完成两组样例的可运行、可复现 Demo，并留下中间产物和判断依据。

## 交付定义

- 一个不依赖 GUI 的命令行脚本，可按样例编号运行；
- 读取高清原片、GPT 参考图和人工成片；
- 生成 proxy、局部 appearance-styled 原片、放大参考图、alpha 和最终合成；
- 输出对比图和 JSON/文本参数记录；
- 两组样例都成功运行；
- 文档明确已验证内容、已知局限和下一步。

## 时间分配

| 时间段 | 工作 | 必须产物 |
|---|---|---|
| 0–1 小时 | 固化素材角色、读取环境、建立 CLI 参数与输出目录 | `ANALYSIS.md`、运行配置 |
| 1–2.5 小时 | 实现图片读取、统一尺寸、proxy、少量平移搜索与色彩统计 | proxy、对齐诊断 |
| 2.5–5 小时 | 实现局部网格 affine appearance operator，先在 DSC02631 跑通 | styled 原片、参数 JSON |
| 5–6.5 小时 | 加入 DSC02634，比较全局 affine 与局部 operator，固定默认参数 | 两组 styled 结果 |
| 6.5–8 小时 | 实现可替换软 alpha、两层融合、Lanczos 放大和 16 bit PNG 输出 | mask、final 合成 |
| 8–9 小时 | 生成四联对比图，检查 halo、颜色溢出、身份和结构保持 | comparison sheets |
| 9–10 小时 | 清理文档、补运行说明、验证从干净命令重复运行 | README、状态更新、Git 提交 |

## 技术取舍

- 使用工作区自带 Python、Pillow 和 NumPy，先不下载模型。
- 先做确定性传统 baseline，避免安装 CUDA、PyTorch 或 matting 模型消耗 Demo 时间。
- 默认保持原片与参考图的归一化坐标；平移搜索只作为诊断和可选修正。
- appearance operator 只负责可映射的色调/亮度变化；新增粒子、太阳光和雾由 GPT 层保留。
- alpha 默认是软主体信任图，写出独立 PNG；后续可替换为人工 mask 或 matting backend。

## 下载策略

当前环境已经有可用的 Python、Pillow 和 NumPy，第一版不需要外部下载。如果后续确实需要 OpenCV 或 matting 模型，先记录具体缺口，再使用可配置代理下载，并把版本和来源写入文档；不为 Demo 预先下载大型模型。

## 风险与降级

- 局部 affine 出现异常色彩：降级到全局稳健 affine，并保存结果供比较。
- 两组样例有轻微位置偏差：启用小范围平移搜索，不引入 dense registration。
- 软 mask 边缘不理想：保留 mask 作为中间产物，优先完成流程闭环。
- 16 bit 编码或 ICC 处理受 Pillow 限制：输出 PNG 同时记录工作空间，后续接入 pyvips；不阻塞 Demo。

## 完成标准

在 10 小时结束时，用户可以运行一条命令对任一组样例生成完整输出，并能从对比图判断 BGU 风格的 operator transfer 是否值得进入下一阶段。
