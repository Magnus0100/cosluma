# 人像美化：独立实跑与回放

本轮为下一大版本的 M0 实验，生产版本仍为 0.4.8。新增源码工具已实际运行，不在现有 EXE、桌面制作入口或参考融合主流程内。算法调用成功、保存成功与画面效果通过分别记录。

## 已实现的数据流

`RAW → 解码 / 一次显像 → 自动多人定位 → SegFace 解析 / 区域资格 → 学习型修复与 blend 场 → 冻结原生 ROI → 浮点原片回放 → TIFF16 / float32 TIFF / A/B`

原片是浮点纹理载体，8 位 PNG 只用于模型分析。ModelScope 专用美肤的缺陷检测、局部 inpainting 和 blend 网络实际运行；美白网络完全不加载。blend 曲线作用于原生源像素，瑕疵残差在准备时完成区域门控。回放分别控制肤质/明暗与瑕疵剂量，避免重复乘修复遮罩。美化没有 alpha 输出。

解析概率不能直接作为编辑区域。实测中 softmax 皮肤概率几乎处处非零，遮挡符纸也曾被高置信误识别。当前资格使用硬标签材料归属、向内羽化、原坐标脸部范围及可见性；眼/唇/眉的纹理范围独立排除。嘴唇和口腔均不可见时，只保留可靠上半脸。这是保守几何资格，尚不是学习型遮挡识别；脸颊可能漏处理，耳颈及身体美肤尚未完成。此纹理范围不冻结整脸的环境光色。

## 首轮实测范围

同一人物、同一场次的三个 RAW 样本分别覆盖全身小脸、符纸/手遮挡、倾斜近景。不能据此声称跨人物或跨场次泛化。

- SegFace 和专用美肤均完成三图真实 GPU 推理，并用修正资格重新跑美肤。前两张没有检测到瑕疵；第三张仅有 100 个原生像素的候选修补，不能宣传为明显祛痘收益。
- 一张 RAW 完成独立 `prepare` 自动闭环，自动定位包含通用重叠 tile 扫描，没有照片坐标配置。
- 冻结字段在原始浮点母版回放；支持外与零剂量精确不变。独立剂量、重叠多人 ROI、资产身份和亚 8 位精度的行为检查通过。
- BeautyGRPO 为近期质量对照，底模授权已取得；真实运行配置和质量结论按本轮实测记录更新，尚未选择默认美化后端。

当前美肤基线的变化较轻，保留本人和妆造不等于取得充分美化收益。尚未完成脸颈手臂协调、三路线共用、场景 finishing、桌面保存重开和跨场次验收；不能把独立实验工具称为下一大版本已交付。

## 源码使用

使用已固定桌面依赖的 Python；模型使用单独配置的分析运行时。先根据 [配置模板](../providers/portrait-beauty.example.json) 创建私人 JSON，填入以下绝对路径并设置 `ready: true`：

| 字段 | 必要内容 |
|---|---|
| `regions_python` | 可运行 Torch / torchvision / SegFace 的 Python |
| `regions_model_root` | 含官方代码、固定 checkpoint 和 manifest 的 SegFace 安装目录 |
| `beauty_python` | 可运行专用美肤 Torch 网络的 Python |
| `beauty_model_root` | 含两个固定权重、官方网络源码和 manifest 的美肤目录 |
| `face_registry` | 含独立 MediaPipe runtime 和 face landmarker 配置的现有模型注册表 |

工具严格核对已安装文件，不自动改动生产运行时或下载模型。SegFace 官方代码 commit 为 `9416628e7b5a3e4b9a1068b8467bdc0cf0be7a7d`，512 CelebAMask-HQ checkpoint SHA-256 为 `3cd535bfcfab4e5c67b7df7015370c2b18cb9a4044ecb00c7c1fa8555558c507`。美肤网络固定 ModelScope 1.40.2 源码；`pytorch_model.pt` / `joint_20210926.pth` 身份由 worker 与安装 manifest 核对。模型代码、权重和底模的许可见 [研究路线](PORTRAIT_BEAUTIFICATION_ROADMAP.md)。

```powershell
python scripts/portrait_beauty.py prepare --source "<原片路径>" --model-config "<私人配置路径>" --output "<新的准备目录>"
python scripts/portrait_beauty.py replay --result "<准备目录>/portrait-beauty.json" --skin-strength 0.7 --blemish-strength 1 --output "<新的导出目录>"
```

`prepare` 运行模型并保存字段；`replay` 只需要桌面 Python、原片及冻结资产，重新打开不加载模型。两个剂量范围为 0–1；均为 0 时精确保源。输出目录须为空，结果 JSON 内的路径属于本地实验资产，尚不是 portable 工程合同。

每次导出包含真正的 sRGB TIFF16、display-linear float32 TIFF、整图 A/B、原生 ROI A/B 和 `export.json`。高位母版不代表模型生成了原生高位细节；生成式修补来源独立记录。此源码入口只验证本地原片美化，完整 Cos 流水线继续按 [M1–M4 里程碑](PORTRAIT_BEAUTIFICATION_ROADMAP.md)推进。
