# Mask 编辑工作流

更新时间：2026-10-04

当前 Demo 用一张灰度 PNG 同时承担两件事：`--alpha` 决定人物与参考图的合成范围，`--fit-alpha` 决定 appearance operator 从哪些像素估计颜色变化。先用多边形生成可运行的起点，再在图像编辑器里做一次轮廓修正即可；不需要安装模型或运行 GUI 工具。

## 1. 生成起始 mask

在仓库根目录打开 PowerShell。`$py` 指向工作区自带的 Python；如果路径不同，只需替换这一行：

```powershell
$py = "C:\Users\80959\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py scripts/make_mask.py --sample DSC02634 --out-dir outputs/mask_backend --feather 32
```

生成文件为 `outputs/mask_backend/DSC02634/alpha.png`。要同时生成两组样例，可以省略 `--sample`；默认轮廓和控制点位于 `scripts/make_mask.py` 顶部的 `POLYGONS` 表中。

## 2. 在编辑器中修正轮廓

可以用任意图像编辑器打开 `alpha.png`，也可以直接使用仓库自带的轻量 painter。painter 只把预览缩小，保存时仍按原片全分辨率写出 L 通道 PNG：

```powershell
& $py scripts/edit_mask.py `
  --image data/DSC02634.jpg `
  --alpha outputs/mask_backend/DSC02634/alpha.png `
  --out outputs/mask_review/DSC02634/alpha.png
```

窗口中左键增加白色 alpha，右键擦除，滚轮调整笔刷，按 `S` 保存。没有可用桌面窗口时，可以用 `--cli` 和归一化点坐标做少量修补：

```powershell
& $py scripts/edit_mask.py --cli `
  --image data/DSC02634.jpg `
  --alpha outputs/mask_backend/DSC02634/alpha.png `
  --points-file mask_edits.json `
  --out outputs/mask_review/DSC02634/alpha.png
```

`mask_edits.json` 可以是 `{"add": [[0.5, 0.2]], "erase": [[0.7, 0.4]]}`；坐标默认是 0–1 的归一化 `x,y`。也可以用 Krita、GIMP、Photoshop 等编辑器完成同样的操作。无论使用哪种方式，都保持以下约定：

- 画布尺寸保持原片尺寸（当前为 6336×9504）；
- 灰度通道为 `L`：白色（接近 255）表示可信人物区域，黑色（0）表示交给参考图的背景；
- 头发、薄纱、轮廓光和手指边缘使用灰色过渡，不要用硬锯齿边界；
- 不要裁剪、旋转或改变画布比例；导出为普通非索引 PNG。

对 `DSC02634` 优先检查抬起的手、头顶发丝和肩部；这些区域最容易出现人物与参考图的双边缘。外轮廓可以比原片主体略向外留出余量，再用 30–50 px 的羽化收边。编辑后建议另存为 `outputs/mask_review/DSC02634/alpha.png`，保留 `mask_backend` 作为可回退的起点。

如果需要修改多边形而不是直接刷 PNG，可复制一份 JSON（坐标为 0–1 的归一化 `x,y`）：

```json
{
  "DSC02634": [[0.35, 0.25], [0.45, 0.22], [0.67, 0.39], [0.64, 0.64], [0.665, 1.0], [0.29, 1.0]]
}
```

然后重新栅格化：

```powershell
& $py scripts/make_mask.py --sample DSC02634 `
  --points-file points.json --out-dir outputs/mask_review --feather 32
```

## 3. 用 mask 运行 Demo

同一张人工修正后的 mask 同时传给两个参数：

```powershell
& $py scripts/run_demo.py --sample DSC02634 `
  --alpha outputs/mask_review/DSC02634/alpha.png `
  --fit-alpha outputs/mask_review/DSC02634/alpha.png `
  --out-dir outputs/mask_demo
```

如果只想先看合成边缘，可以省略 `--fit-alpha`；如果想让 operator 只从主体颜色估计，则保留两个参数。`--alpha` 和 `--fit-alpha` 目前按单张样例传入，另一组样例需单独运行：

```powershell
& $py scripts/run_demo.py --sample DSC02631 `
  --alpha outputs/mask_review/DSC02631/alpha.png `
  --fit-alpha outputs/mask_review/DSC02631/alpha.png `
  --out-dir outputs/mask_demo
```

每组结果在 `outputs/mask_demo/<sample>/`：

- `comparison.jpg`：快速检查原片、参考图、人工成片和 Demo 成片；
- `final.png`：当前两层合成结果；
- `alpha.png`：实际被读取并调整到原片尺寸后的 mask；
- `metadata.json`：记录 `alpha.mode`、`alpha.path` 和 `fit_trust`，便于回溯本次运行。

## 4. 快速判断与迭代

先看 `comparison.jpg` 的人物边缘，再看衣服纹理是否仍来自高清原片。若出现双轮廓，扩大该处白色主体区或把边缘刷成窄灰带；若背景光效被人物轮廓带入，收窄白色区域并重新运行。每轮只修改 mask，再复用相同命令，避免同时改变 operator 参数。

该流程是一次人工 mask 确认的最小入口，不是发丝级 matting。半透明饰物和极细发丝仍需在成片检查中手工取舍；暂不引入大型模型或批处理参数。
