# 当前保留组件与许可来源

本文件登记当前生产组件。分发或升级以安装包、代码和模型目录中保留的许可原文及固定身份为准，模型权重与代码分别核对。

| 组件 | 当前用途 | 许可依据 |
|---|---|---|
| PySide6 / Qt | 桌面界面 | 安装包中的 LGPL/GPL/商业条款 |
| NumPy、SciPy、Pillow、tifffile | 数组、颜色及编码 | 各安装包 LICENSE |
| rawpy / LibRaw | RAW 解码 | 各安装包 LICENSE |
| ONNX Runtime | 轮廓推理 | MIT，安装包 LICENSE |
| PyTorch、Transformers | 本地模型运行时 | 各安装包 LICENSE |
| Sapiens2 segmentation 0.4B | 两图语义 | 官方 Sapiens2 自定义许可及模型来源记录 |
| BiRefNet | 轮廓种子 | 保留代码/权重来源和条款 |
| ViTMatte / PyMatting | 原生覆盖率及成对前景背景 | 对应模型目录/安装包 LICENSE |
| DINOv3 ViT-L/16 | 两图局部特征对应 | DINOv3 License，详情如下 |
| SAM3.1 静态 detector | 场景概念实例 | SAM License，详情如下 |
| LaMa | 参考人物区域背景补全 | 模型目录及上游许可 |
| Real-ESRGAN compact | 背景细节 | 模型目录及上游许可 |
| MediaPipe face landmarker | 五官结构观察 | 模型来源及安装包 LICENSE |
| Qwen Image 2.1 / ComfyUI / GGUF 节点 | 参考及独立效果生成 | 权重和节点分别登记，见 [QWEN_IMAGE_LOCAL.md](QWEN_IMAGE_LOCAL.md) |

本地权重和运行时不打入桌面程序包或 portable 工程。外部服务按其条款使用。旧模型与旧算法不作为当前生产入口；外部原始权重、许可证和历史安装回执保留，不为更新生产配置而改写。

## DINOv3 固定身份

- 仓库：[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)。
- HF revision：`ea8dc2863c51be0a264bab82070e3e8836b02d51`。
- 权重 SHA-256：`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`。
- 当前profile：`dinov3_vitl16`；本地目录由模型注册表的 `dinov3_model_root` 指定。
- 许可原文：模型目录中的 `LICENSE.md`，标题DINOv3 License，更新日期2025-08-19；不是Apache-2.0。

安装文件须与固定上游revision及内容身份核对。镜像名称不替代官方授权或文件一致性验证；准备阶段核对实际文件、配置和适配器身份。

## SAM3.1 固定身份

- 权重仓库：[facebook/sam3.1](https://huggingface.co/facebook/sam3.1)。
- HF revision：`daa63191845a41281374e725f4c9e51c7a824460`。
- 权重 SHA-256：`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`。
- 官方代码：[facebookresearch/sam3](https://github.com/facebookresearch/sam3)，commit `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`。
- 当前profile：`sam31_static_detector_v1`；本地目录由模型注册表的 `scene_model_root` 指定。
- 许可原文：`code/LICENSE`，SAM License，更新日期 2025-11-19。
- 许可 SHA-256：`67f67e15efa517e0f67f59ff2d2f603a624da5d2f5458b0c5b366ccd66bb3eea`。

当前适配器构造SAM3.1 Tri静态detector，并严格加载全部1166个detector tensors。模型、代码、预处理和输出身份分别绑定；版本号或成功加载不证明静态Cos分割质量更优。

## Sapiens2 当前语义模型

仓库 `facebook/sapiens2-seg-0.4b`，revision `449b3c5335e6722bb94990abdd1aa6e612432f22`，权重 SHA-256 `b85fdb50b7d6123a967d5ee4a505e222baff8d2f7ad6bbf353578c1a61dfbac9`。当前唯一语义 profile 为 `sapiens2_seg_0.4b`；原始许可与来源记录保留在本机模型目录。

使用 [local-models.example.json](../providers/local-models.example.json) 创建自己的模型注册表，填写模型和运行时的绝对路径，并通过 `COSPLAY_MODEL_REGISTRY`选择它；源码用户也可使用本地 `providers/local-models.current.json`。模型身份只证明生产输入，不授予完整透明度、人物保真、成片质量或跨场次泛化通过。

项目代码采用 `GPL-3.0-or-later`，详见根目录 [LICENSE](../LICENSE)。桌面依赖的许可清单、对应源码及重建说明见 [BINARY_LICENSES.md](BINARY_LICENSES.md)。本文件不是第三方许可原文的替代品；分发桌面程序、依赖或模型前，须分别保留并遵守相应许可。
