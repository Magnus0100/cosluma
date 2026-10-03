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
- `metadata.json`：参数和输入记录。

可用 `--alpha path/to/mask.png` 替换默认软 mask。若希望 operator 只从主体可信区域估计，可同时传入 `--fit-alpha path/to/mask.png`；这会排除大部分背景粒子和太阳光对人物色调拟合的影响。默认 operator 是低分辨率网格 affine 映射，目的是快速验证流程，不是最终 matting 或生成式修复方案。

## 传统轮廓 mask backend

两组样例也提供了一个无模型依赖的可编辑轮廓后端。它在 proxy 分辨率栅格化样例对应的人物多边形，并做约 32 像素羽化：

```powershell
& $py scripts/make_mask.py --out-dir outputs/mask_backend
& $py scripts/run_demo.py --out-dir outputs/demo_polygon `
  --sample DSC02631 --alpha outputs/mask_backend/DSC02631/alpha.png `
  --fit-alpha outputs/mask_backend/DSC02631/alpha.png
& $py scripts/run_demo.py --out-dir outputs/demo_polygon `
  --sample DSC02634 --alpha outputs/mask_backend/DSC02634/alpha.png `
  --fit-alpha outputs/mask_backend/DSC02634/alpha.png
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

窗口中左键增加 alpha、右键擦除、滚轮调整笔刷、按 `S` 保存；也可以用 Krita、GIMP 或 Photoshop 编辑同一张灰度 PNG。没有桌面窗口时，`edit_mask.py --cli --points-file edits.json` 支持用归一化点坐标做少量加/擦修补。然后把保存的 mask 传给两个参数：

```powershell
& $py scripts/run_demo.py --sample DSC02634 `
  --alpha outputs/mask_review/DSC02634/alpha.png `
  --fit-alpha outputs/mask_review/DSC02634/alpha.png `
  --out-dir outputs/mask_demo
```

mask 需要保持原片画布尺寸和普通灰度 PNG 格式；白色是主体信任区，黑色是参考图区，发丝和轮廓光可以用灰色过渡。完整的生成、编辑和检查步骤见 [`docs/MASK_WORKFLOW.md`](../docs/MASK_WORKFLOW.md)。

`run_demo.py` 会把实际传入的 `--alpha` 路径写入该样例的 `metadata.json`（`alpha.path`），并把 `--fit-alpha` 写入 `fit_trust`，方便比较不同 mask 版本。

## 轻量验证

```powershell
& $py scripts/validate_outputs.py --outputs outputs/demo --strict
```

验证只检查素材可读性、输出存在性和基本像素范围，不承担大型测试框架的职责。

