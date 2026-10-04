# 色彩与 appearance operator 专家结论

更新时间：2026-10-04

真实原片路径为 `C:\Users\80959\Desktop\新建文件夹\新建文件夹 (7)`。在 341×512 proxy 上对两组真实原片运行当前 alpha-trust operator，并以 `data/DSC02631.png`、`data/DSC02634.png` 作为人工成片的颜色参考，得到以下 A/B（MAE 仅用于比较算子，不是最终验收标准）：

| 样例 | 当前 affine，主体 alpha>0.5 | fit-full | affine + 低频 residual |
| --- | ---: | ---: | ---: |
| DSC02631 | 约 8.0/255 | 约 10.7/255 | 约 8.1/255 |
| DSC02634 | 约 16.1/255 | 约 20.0/255 | **约 10.2/255** |

`DSC02634` 的主要误差来自 GPT 参考图新生成的头发轮廓光、衣服高光和局部曝光。单纯 RGB 通道 scale/bias 只能改变已有像素，无法从原片的暗发丝合成出参考图中的宽光源。直接 `fit-full` 会让背景太阳光和粒子参与统计，人物误差反而升高。

下一阶段采用 detail-preserving residual transfer：

```text
affine = local_affine(source)
residual = GaussianBlur(proxy_reference - proxy_affine, sigma)
styled = affine + strength * resize(residual, source_size)
final = alpha * styled + (1 - alpha) * reference
```

残差只在 proxy 上估计并以浮点数上采样，因此不会把 GPT 的新边缘或粒子以锐利细节写进高清层；原片的高频纹理仍来自 `affine`。当前推荐起点为 `sigma=2` proxy 像素、`strength=1.0`，并保留开关便于逐组比较：

```powershell
& $py scripts/build_final.py `
  --source-dir "C:\Users\80959\Desktop\新建文件夹\新建文件夹 (7)" `
  --alpha-dir outputs/mask_review `
  --residual-transfer --residual-sigma 2 --residual-strength 1 `
  --out-dir outputs/final_delivery_residual
```

DSC02631 的数值变化很小，但头发和肩部暖光交接更接近 GPT 参考；DSC02634 的亮度与人工成片明显更接近。若某组出现 halo，应先降低该组 `residual-strength`（建议 0.6–0.8），再修 alpha；不要改用 `fit-full`。
