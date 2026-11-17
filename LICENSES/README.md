# 第三方许可

项目代码许可见根目录 [LICENSE](../LICENSE)。本目录中的上游许可证与版权声明
保持原文，各组件版权归其作者所有。

README 中的例图出镜为抖音「奶糕Naige」。照片已获授权用于本项目 README 展示；
`docs/images/readme/` 中的照片、衍生展示图及截图中的照片不适用项目代码的 GPL 许可。

当前代码直接使用 PySide6/Qt、NumPy、Pillow、pillow-heif/libheif、OpenCV、SciPy、rawpy/LibRaw、
color-matcher。相关依赖的许可文本保存在各组件目录中。
Windows 发布包同时包含 Python 与 PyInstaller 引导程序及其许可；发布源码包含构建配置。
当前 OpenCV 依赖为 `opencv-python-headless`，桌面界面由 PySide6 提供。
Pillow 当前为 13.0.0。构建时另从实际安装版本复制依赖的许可证、版权声明与随包第三方声明，
包括由 PyInstaller 收集的辅助模块；目录中的额外声明不表示软件提供对应的旧模型或功能。

本项目代码按 GPL-3.0-or-later 发布；每个 Release 随附同一提交的源码与构建配置。
Qt/PySide、LibRaw 等组件保留各自许可。Windows 包使用独立的动态库与 PySide 模块，
未阻止用户替换库；重新打包可使用仓库中的 `CosLuma.spec` 与构建依赖。
对应上游源码：

| 组件 | 源码 |
| --- | --- |
| Qt 6.11.2 | https://download.qt.io/official_releases/qt/6.11/6.11.2/single/ |
| PySide 6.11.2 | https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.2-src/ |
| rawpy 0.27.1 与其 LibRaw 子模块 | https://github.com/letmaik/rawpy/tree/v0.27.1 |
| color-matcher 0.6.0 | https://pypi.org/project/color-matcher/0.6.0/#files |
| pillow-heif 1.8.1 | https://github.com/bigcat88/pillow_heif/tree/v1.8.1 |
| libheif 1.23.6、libde265 1.1.3、x265 4.3 | https://github.com/strukturag/libheif/tree/v1.23.6 · https://github.com/strukturag/libde265/tree/v1.1.3 · https://github.com/Multicorewareinc/x265/tree/4.3 |

整头分割和超分通过仓库外安装的 ComfyUI 服务调用。SAM3.1 权重使用随权重提供的 SAM License；
转换为服务可读取的权重格式后仍保留原许可。ComfyUI、SAM3.1、PyTorch 和超分模型的许可
随各自安装保留，不随本项目主许可证重新授权。

可选美化节点随附 [FaceXFormer](https://github.com/Kartik-3004/facexformer/tree/10fe8291f8a64e2ca1daf938e3e0007bd860303b)
的网络与 Transformer 实现，使用 MIT 许可，原文保存在 `FaceXFormer/LICENSE` 和节点目录。
适配仅取消额外的 ImageNet 初始化下载，随后严格加载完整 FaceXFormer 权重；权重本身位于仓库外，
按 [官方权重页](https://huggingface.co/kartiknarayan/facexformer)的许可使用，转换为 safetensors 不改变张量或许可。

[BeautyGRPO](https://github.com/vivoCameraResearch/BeautyGRPO) 的公开推理调用用于独立编写服务适配，直接调用官方 Diffusers FluxKontextPipeline，未复制其推理实现。
上游代码声明 MIT，同时明确预训练权重与 FLUX.1-Kontext-dev 底座有各自许可；
不据代码许可推定 BeautyGRPO 权重已重新授权。BeautyGRPO、FLUX、文本编码器与 VAE 权重
由服务部署者从上游取得，未随桌面或源码包分发。FLUX.1-dev 底座适用其非商业许可；
其他权重按各自模型页与随权重条款使用，不受本项目 GPL 许可替代。
当前美化定位随附 [FGTBT](https://github.com/Xi0ngxinyu/FGTBT/tree/32627a248c66b5a5defec347fa197860347d080d) 的推理网络适配，取消训练注册器和旧 MMCV 依赖，保留完整推理层和严格权重加载。FGTBT 新增代码为 MIT，声明见节点目录 `FGTBT-LICENSE`；其中 SegFormer 原有部分保留 NVIDIA Source Code License，原文见节点目录 `SegFormer-LICENSE`。该部分仅供非商业研究或评估，不由主项目 GPL 重新授权。作者检查点放在服务端，不随桌面分发。

[PIXLRelight](https://github.com/mlfarinha/pixlrelight/tree/f8cb2dba4d08b3dfcd392cf9b2946a11f1f48c23) 上游源码、DINOv3 层及相关声明由部署者原样保留在仓库外模型目录；本仓库提供独立编写的节点与光照场适配，不打包这些源码和权重。上游代码声明 MIT，模型为 CC-BY-NC-4.0；其第三方 DINOv3 部分遵循随文件提供的 DINOv3 条款。[Marigold-IID-Lighting](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1) 按模型页及随权重的 OpenRAIL++ 条款使用。
[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) 的权重和 ComfyUI 格式文件只在服务端读取，不随软件分发，按模型页的 Qwen Research License Agreement 与各组件随附声明使用，格式转换或量化不改变模型许可。局部修饰的独立进程调用已有 ComfyUI 编辑接口，明确指定检查点、BF16 文本编码的 CUDA/CPU 卸载与 VAE 分块布局，不复制 Qwen 推理源码。DynamicVRAM 使用服务已安装的 `comfy-aimdo`，其随包许可为 GPL-3.0；该原生库只在服务端使用，不随桌面包分发。
[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) 的检查点和 Comfy-Org 量化版本仅在模型服务端安装，不随本软件分发；采用其 [2026-11-10 社区许可](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/492fd8c4eca666f1085f57e30738c1e0d86b3439/LICENSE)，不是 Apache-2.0，也不由本项目 GPL 重新授权。许可排除欧盟、英国、韩国和美国；年化营收超过 100 万美元或向第三方提供托管服务需取得 MiniMax 另行书面授权。部署者应自行核对完整许可与适用地域，本适配仅调用独立安装的 ComfyUI 原生推理接口，DynamicVRAM 使用服务已有的 GPL-3.0 `comfy-aimdo`，不复制模型实现或分发提示词增强权重。
Diffusers、Transformers、Accelerate 与 PEFT 仅在模型服务环境安装，各自的 Apache-2.0 许可随安装保留。[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 用于用户明确选择的 NF4 底座加载，其 MIT 许可随服务环境安装保留；量化不改变底座或 LoRA 的许可。
README 表情包 `docs/images/readme/honest-retouch-meme.png` 改写「老实赚钱，很枯燥吧」字幕，底图来自 [MemeBako 的 FX 战士久留美素材页](https://memebako.com/meme/fxkurumi)（[底图](https://img.memebako.com/memes/fxkurumi/memes/kurumi-ai-anime-screenshot.webp)）。素材页将该图标注为 AI 改图；本项目仅裁切并替换字幕，不作为官方动画截图引用。原角色与底图权利归各自权利人，未随代码按 GPL 重新授权。

内置 `providers/prompts.json` 含 260 条分类摄影参考，每条保留来源；项目原创 4 条使用 GPL-3.0-or-later。
其余条目来自以下明确授权的上游版本，随附许可证原文，不包含原始图片：

| 来源 | 条数 | 版本 | 许可及本地原文 |
| --- | --- | --- | --- |
| [ComfyUI-Prompt-Vault](https://github.com/jeremieLouvaert/ComfyUI-Prompt-Vault) | 57 | `59969e74569cb05bcdfcd0b3109aaeab52755510` | MIT；`ComfyUI-Prompt-Vault/LICENSE` |
| [cos-effect-prompt](https://github.com/Couer869/cos-effect-prompt) | 84 | `b2970e1299eb6e28739fd02ff5f2dd36102ae61e` | MIT；`cos-effect-prompt/LICENSE` |
| [zsky-prompt-library](https://github.com/zsky-ai/zsky-prompt-library) | 93 | `151b8aa1f10d9638752063cb5491df22967ba359` | MIT；`zsky-prompt-library/LICENSE` |
| [awesome-gpt-image-2-API-Prompts](https://github.com/Evolink-AI/awesome-gpt-image-2-API-Prompts) | 8 | `e2a269ad1a055a0b4f6c1e170341c6c1aba30faa` | CC0-1.0；`awesome-gpt-image-2-API-Prompts/LICENSE` |
| [nuyoah-xiezhen-prompt](https://github.com/nuyoah-ai-works/nuyoah-xiezhen-prompt) | 12 | `7482a14074bcffb9fed8eb8fe2ddcdc4ac2e980b` | MIT；`nuyoah-xiezhen-prompt/LICENSE` |
| [nuyoah-image-reverse-prompt](https://github.com/nuyoah-ai-works/nuyoah-image-reverse-prompt) | 2 | `f56616e2e0db10f5344645d8f6c09f3915ce0d25` | MIT；`nuyoah-image-reverse-prompt/LICENSE` |

条目按摄影任务选取短文本或改编，改编条目标注 section、适用范围和来源文件散列；具体来源版本见上表及条目元数据。
库中条目用于摄影参考，不作为执行规则或预设风格菜单。自定义库位于仓库外，由其提供者保留来源、作者与许可。
