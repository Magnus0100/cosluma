# 人物边界受光

本页仅适用于有原生可编辑合同的原片／导入参考源域工具；这些层不是 scoped 生成外观的必经后处理。

当前提供“参考发梢受光”“协调肩袖暖光”两个独立层，以及“同时更新边界受光”入口。两层在原片坐标内补充有依据的受光，保留人物几何和alpha。衣料层仅在有效对应和不透明材料支持内转移宽尺度暖光，不证明完整轮廓光或物理重光照。

## 使用与保存

打开已有原片和参考的高清版本，在图层页先完成仿色融合。需要两层时使用“同时更新边界受光”：先准备发梢，再以新发梢端点准备肩袖，两层都成功后一次应用到当前编辑。预览正常整图和原尺寸细节后另存高清版本；准备成功不自动授予艺术批准。

也可单独补充或重新准备一层。成功替换唯一旧层，保留开关和强度；取消、失败或冻结像素相同不会新增子版本。单独重新准备发梢后，旧肩袖层会因端点过期停止回放，需要再协调肩袖。联合任务可在任务中心取消，不应用中间结果。

改变上游仿色、人物细节、环境层、背景、alpha或相关控制会使旧层停止回放。有生效的降噪、手工人物边缘或后续人物精修时，当前准备会拒绝，因为尚未独立观察这些操作后的最终像素。参考无可用受光或对应不足时保留当前版本。

源码联合准备命令为（只写隔离的准备回执，不发布 revision）：

```powershell
.\.venv\Scripts\python.exe .\scripts\boundary_light_coordination_worker.py --root "工程目录" --run-id "版本ID" --output "新的准备目录"
```

单层使用同样参数执行 `boundary_light_worker.py` 或 `cloth_boundary_worker.py`。也可用 `--run-json` 读取当前编辑快照；分析Python须具备配置的完整模型依赖。准备结果必须返回协调器，由 `adjust_source_run()` 在当前 checked task plan 下统一发布 revision；worker 本身不能保存 child。后台使用 `POST /api/source-runs/<id>/reconstruct-design`，请求 `preparation_kind: "boundary_light_coordination"`；接口详见 [本机API](REVISION_API.md)。

## 两层方法与职责

发梢层从真实参考RGB、独立语义及ViTMatte/PyMatting前景观察中取得边缘相对同类内部的正响应。不透明参考可直接观察；半透明参考只使用有前景/背景分离支持且RGB有界的前景。内部对应锚点做局部初定位，双方独立轮廓、法向和语义另行核验，不能把内部对应有效或拟合误差当作精确发丝注册证明。

发梢使用 `reference_hair_boundary_continuous_illumination_v3`：参考资格与RGB联合归一采样，低通取得照明包络，再扣除当前人物已消费响应。参考最近邻owner不能二次硬裁切原图；源语义不确定带只向头发内部渐隐，衣料增量保持零，保护与采样有效性连续衰减。保留源发丝结构，不注入参考细发纹理或亮度细节倍增。局部亮度增量受1.5EV上界与共享RGB色域余量约束，不逐通道截断。原alpha不变，写入非负预乘RGB增量；这是受光外观转移，不是物理重光照。

发梢和衣料准备缓存均核验真实producer代码闭包；缺失或过期证据只能触发重新准备，不能冒充当前缓存命中。冻结回放仍须满足当前资产、producer 与绑定合同。流程节点签名使用同一闭包，边界算法升级会连同合成和导出失效，不会把旧边界结果标为当前算法。

衣料层从原照片的成对基线确定主要源材料，配对参考/当前人物差值先扣除同材料内部偏置，再稳健拟合粗光色场。当前人物原尺寸明暗承载该场，避免复制参考褶皱和金边。仅独立几何、语义及有效支持合格的不透明外衣可写入，支持在资格域内渐隐，alpha不变。当前限定源owner 23、alpha≥0.95，不扩展到裤子、配饰、手或白色腰部。宽尺度褶皱与照明仍可能混淆，不声明恢复法线、反照率或入射光。

| 职责 | 当前生产模块 |
|---|---|
| 发梢算子 / 准备 / 回放 | `reference_boundary_light_transfer.py` / `boundary_light_worker.py` / `boundary_light_runtime.py` |
| 衣料算子 / 准备 / 回放 | `reference_cloth_boundary_transfer.py` / `cloth_boundary_worker.py` / `cloth_boundary_runtime.py` |
| 联合准备 | `boundary_light_coordination_worker.py` |

联合合同为 `source_boundary_light_coordination.v1`，层kind分别为 `source_boundary_light` 与 `source_cloth_boundary_light`。衣料在发梢之后、手工人物精修之前回放。原尺寸增量、支持和端点身份冻结后，全图、ROI、预览及portable重开使用同一资产，不重新拟合或推理。

## 验收范围与限制

当前发梢节点处理有资格的连续低频受光；衣料节点处理有对应支持的不透明外衣宽尺度光色。强窄轮廓光、其它部位收益和跨照片效果须分别验证。单张照片或局部窗口的观察不代表整作或其它照片通过。

算法验证须覆盖无边缘响应、已消费光、统一变亮/变暖、背景柱面、错注册/语义、保护区及半透明资格。回放验证须覆盖错误端点、损坏资产、full/ROI/preview与迁移。实际艺术评审先看正常整图，再看相同原生窗口，核对纹理、亮带、暗缝和退步；能量增量、执行成功或技术一致性不能代替这些判断。最终结论绑定实际保存文件，并与技术报告分开。

更完整的能力边界见 [算法选择与范围](ALGORITHM_DECISIONS.md)，作品批准标准见 [Cos成片标准](COSPLAY_POST_STANDARD.md)。

### 发梢对应的严格备用观察

当当前 DINO receipt 对发梢 owner 没有有效 token 时，主流程可以调用独立的 `reference_hair_boundary_registration.v1`。它只把已绑定的 MediaPipe 面部 landmark 作为初始仿射候选，然后重新验证 source/reference alpha、owner 4 轮廓、face q90、affine condition、发梢轮廓覆盖和 q90；失败就保留原计划。它不修改 DINO receipt、不把脸部纹理当发梢纹理，也不放宽 `0.7` 覆盖和 `8 px` 误差门禁。肩袖没有同等备用对应时仍安全拒绝，局部发梢成功不会单独提交联合 pair。
