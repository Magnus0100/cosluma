# 本地脚本

脚本使用工作区自带的 Python、Pillow 和 NumPy，不需要先安装 CUDA 或下载模型。

## 运行 Demo

在仓库根目录运行：

```powershell
$py = "C:\Users\80959\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py scripts/run_demo.py --out-dir outputs/demo
```

默认处理 `DSC02631` 和 `DSC02634` 两组样例。每组输出：

- `styled_original.png`：appearance operator 应用到高清原片；
- `upscaled_reference.png`：Lanczos 放大的 GPT 参考图；
- `alpha.png`：可替换的软主体信任图；
- `final.png`：两层合成结果；
- `comparison.jpg`：原片、参考图、人工成片、Demo 成片四联图；
- `structure_mismatch.png`：对齐后的亮度边缘分歧图；红色越强表示 GPT 参考图的局部结构越不可控，适合决定把 alpha 压低的位置；
- `metadata.json`：参数和输入记录。

Demo 会在 proxy 尺寸对参考图做受限的 scale + translation 搜索（默认缩放 0.97–1.03、步长 0.01、平移范围 ±24 像素），再把获胜变换应用到全分辨率参考图。`metadata.json` 的 `alignment` 记录尺度、平移和相关分数；`structure_diagnostic` 记录分歧统计。尺度搜索只处理整体构图差异，不会试图修正 GPT 新生成的脸、发丝、手指或光斑。

可用 `--alpha path/to/mask.png` 替换默认软 mask。对于 GPT 可能改动结构的区域，推荐用 `--fit-full` 让 operator 从对齐后的完整画面估计外观，再由 alpha 决定哪里写回原片；只有想严格限制拟合区域时才传 `--fit-alpha path/to/mask.png`。默认 operator 是低分辨率网格 affine 映射，目的是快速验证流程，不是最终 matting 或生成式修复方案。

## 传统轮廓 mask backend

两组样例也提供了一个无模型依赖的可编辑轮廓后端。它在 proxy 分辨率栅格化样例对应的人物多边形，并做约 32 像素羽化；默认把脸、头发和上半身的写回权重压低，把稳定服装/身体纹理保留到约 0.72：

```powershell
& $py scripts/make_mask.py --out-dir outputs/mask_backend
& $py scripts/run_demo.py --out-dir outputs/demo_polygon `
  --sample DSC02631 --alpha outputs/mask_backend/DSC02631/alpha.png `
  --fit-full
& $py scripts/run_demo.py --out-dir outputs/demo_polygon `
  --sample DSC02634 --alpha outputs/mask_backend/DSC02634/alpha.png `
  --fit-full
```

控制点位于 `scripts/make_mask.py` 顶部的 `POLYGONS` 表中，坐标是 0–1 归一化的 x,y；需要刷改时也可以用 `--points-file points.json` 覆盖。该后端比软椭圆更贴近人物外轮廓，但仍不是发丝级 matting，轮廓光和半透明饰物应在后续人工 mask 中修正。

## 一次人工编辑

推荐把 `outputs/mask_backend/<sample>/alpha.png` 复制到一个新的目录，用仓库自带的 painter 修正人物边缘：

```powershell
& $py scripts/edit_mask.py `
  --image data/DSC02634.jpg `
  --alpha outputs/mask_backend/DSC02634/alpha.png `
  --out outputs/mask_review/DSC02634/alpha.png
```

窗口中左键增加 alpha、右键擦除、滚轮调整笔刷、按 `S` 保存；也可以用 Krita、GIMP 或 Photoshop 编辑同一张灰度 PNG。没有桌面窗口时，`edit_mask.py --cli --points-file edits.json` 支持用归一化点坐标做少量加/擦修补。然后把保存的 mask 传给 `--alpha`，并默认使用 `--fit-full`：

```powershell
& $py scripts/run_demo.py --sample DSC02634 `
  --alpha outputs/mask_review/DSC02634/alpha.png `
  --fit-full `
  --out-dir outputs/mask_demo
```

mask 需要保持原片画布尺寸和普通灰度 PNG 格式；白色是稳定纹理写回区，黑色是保留 GPT 结构和光效的区域，发丝和轮廓光可以用灰色过渡。完整的生成、编辑和检查步骤见 [`docs/MASK_WORKFLOW.md`](../docs/MASK_WORKFLOW.md)。

`run_demo.py` 会把实际传入的 `--alpha` 路径写入该样例的 `metadata.json`（`alpha.path`），并把 `--fit-alpha` 写入 `fit_trust`，方便比较不同 mask 版本。

也可以单独生成结构诊断，不写高清成片：

```powershell
& $py scripts/structure_diagnostic.py `
  --source data/DSC02634.jpg `
  --reference 'data/DSC02634参考.png' `
  --out-dir outputs/structure_diagnostic
```

该脚本输出 proxy 级 `structure_mismatch.png` 和 `structure_mismatch.json`。它只作为修 mask 的依据；不会自动把分歧区强行改回原片。

## 轻量验证

```powershell
& $py scripts/validate_outputs.py --outputs outputs/demo --strict
```

验证只检查素材可读性、输出存在性和基本像素范围，不承担大型测试框架的职责。

