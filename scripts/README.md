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

可用 `--alpha path/to/mask.png` 替换默认软 mask。默认 operator 是低分辨率网格 affine 映射，目的是快速验证流程，不是最终 matting 或生成式修复方案。

## 轻量验证

```powershell
& $py scripts/validate_outputs.py --outputs outputs/demo --strict
```

验证只检查素材可读性、输出存在性和基本像素范围，不承担大型测试框架的职责。
