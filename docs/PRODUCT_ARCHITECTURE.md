# 当前产品数据流

桌面、CLI 和本机后台使用同一生产入口：`creative_task → production_coordinator → single_source_workflow`。当前 0.7 开发源码默认以原场景内半合成、鲜明风格工作：原图视觉观察、原片浮点基线、基线审片、模型选择首次局部或整图编辑、独立复审、条件改进和保存。局部许可外保持原片浮点数据；整图生成接管人物与环境外观，不能再用源人物 RGB 整体覆盖。

`editing_request` 将精修、半合成、大合成与自然、鲜明、强烈风格分开。原片精修、半合成与大合成在同一 harness 中选择获准工具；精修不广告整图重绘。默认保持原地点，只有大合成明确授权才改变场景目标；重光和风格不授权瘦脸瘦身。普通导入参考的源域图层工具仍保留，显式原景增强与单区编辑见[原景增强](SOURCE_SCENE_ENHANCEMENT.md)，不充当新默认链。

## 任务入口与运行时

`creative_task` 检查任务并冻结计划，`production_coordinator` 核验执行合同并进入 `single_source_workflow`，后者调用当前准备与渲染 worker。新生成任务通过 `photo_edit_finish` 共用已有 session、工具和评价；原片/普通导入工具保留其有界完成阶段。`pipeline_stage_report` 归约保存事实。制作入口为 `/api/tasks/plan`、`/api/tasks/start` 及其批量形式，CLI 使用 `task-plan` / `task-start`；内部 `one-click` 仅承接同一 checked task 的子进程执行。

`local_portrait`、`existing_reference` 和 `create_reference` 是同一管线的输入与节点开关。原片本地任务不消费参考，也不隐式准备新 Beauty，可消费明确准备的冻结美化层；普通导入参考默认零新增图像。新 scoped 创建默认 **1** 个候选，必须配置执行时视觉观察；独立审片默认开启，`editing_autofinish={"maximum_improvements":2,"maximum_zero_image_finishes":1}` 预留通常两轮及最后一次仅允许零生图工具或停止的收尾，默认最多图像 **3** 次、视觉 **10** 次。首次编辑两项通过实际只用图像 **1** 次、视觉 **4** 次；原片审片通过不等于完成编辑。关闭审片仍实际执行设计、首次决策和工具，保存未审草稿；关闭自动改进取消后续轮次及收尾。显式一轮偏好不升级。多候选显式选择，来源额度共享，未知与停止决策占用原轮次，选择或重开不重置。

模型注册表选择顺序是：任务显式 `COSPLAY_MODEL_REGISTRY` → 用户保存的选择（`prefer_user=true`，或工程旁没有 portable 配置）→ 工程旁 `model/local-models.json` → 公共 `providers/local-models.current.json` → 公开模板。整项任务固定使用选定模型/runtime 身份；缺失或身份不符时停止并报告缺口。未请求的能力标为 `not_requested`。

| 能力 | 当前模型/运行时 | 启用条件 |
|---|---|---|
| scoped 生成外观 | CPU、所选 Qwen/GPT 图像 provider、已配置视觉服务 | 当前 scoped 创建或明确纠偏；不要求未用的原生分析模型 |
| `source_silhouette` | BiRefNet ONNX、ONNX Runtime | 本地原片节点子集 |
| `source_geometry` | Sapiens2 0.4B、ViTMatte、BiRefNet ONNX、ONNX Runtime | 普通导入参考的原生准备 |
| `reference_correspondence` | DINOv3 ViT-L/16 | 普通导入参考的对应准备 |
| `scene_analysis` | SAM3.1 静态 detector 与隔离 Python | 普通导入参考的场景准备 |
| `portrait_regions` | 五官定位模型与独立 Python | 原生工具的 `complete_post=true` |
| `background_completion` | LaMa | 原生工具的 `complete_post=true` |
| `background_superresolution` | 当前 RealESRGAN checkpoint 与 SR Python | 原生工具启用的背景重建 |
| `portrait_beauty` | BeautyGRPO 独立安装与专用 Python | 实际启用 Beauty 的原片/普通导入工具；安装预检读取回执，worker 核验执行身份 |

`cos_node_registry`、`cos_workflow_graph` 和 `cos_workflow_cache` 为同一管线声明端口、参数、依赖及 UI 失效提示。实际执行仍由当前 Python 协调器与 worker 完成；画布计划的 `reused` 不证明真实缓存命中。未接入生产的占位 Node/callback 执行器已删除。具体范围见 [产品架构](PRODUCT_ARCHITECTURE.md)。

## 计划、执行与保存合同

输入是 `task_plan/plan_id`、mode、唯一 execution strategy、源/参考快照、模型/runtime 身份、控制参数与预算；输出是有归属的 job/run、真实阶段回执、`stage_report`、独立 revision 和导出回执。批量每张照片独立建立这些事实。

1. 计划只读取文件、配置与已记录身份，不联网、不推理；执行前重新核验绑定。
2. scoped 执行观察原图，冻结本图设计，分别编译 Qwen/GPT 提示词；RAW 的源显示变换只做一次。
3. 冻结 `photo_appearance` 原片浮点基线并评价，再由模型选择首次摄影显影、局部生成或联合整图编辑。工具返回有实际覆盖归属、可独立审查的完整候选；记录实际生成分辨率，不将大画布宣称为相机原生细节。首次操作不要求生图。
4. 独立评价自然可信度与创作收益。首次编辑通过跳过条件节点；未达标且有额度时，模型在同一 session 选择当前图的摄影显影、生成编辑、合格局部增量的 CPU 融合或停止，再独立复审。未获选为最佳的 working 仅供资格成立的工具处理，不自动取代最佳图；CPU 工具消耗一次改进或零生图收尾轮次，但新增生图为零。
5. 保留最佳已评价完整版本。保存时核对被审像素、ICC 与实际 PNG，失败或取消保留已有版本；结果未知先查原请求，不自动重发。
6. 像素、资产和合同完整后原子发布。自动审片、软件完成和人工批准分别记录；未达艺术标准如实保留问题。

明确点击改进建立新的 `editing_iteration` checked task，绑定已提交父版本、真实审片和原范围，不是默认流程必须再点的一步。自动两项通过后，仍可用非空 `user_feedback` 提出新目标；旧通过是历史基线，不能令新目标自动通过。新候选独立评价，不佳则保留完整基线。它与自动纠偏共用 `photo_edit_finish` 和工具合同，不对任意导入图开放。

### 原片与普通导入参考工具

以下源域能力仅在对应工具启用时使用。源语义以 `source_semantic_binding.v1` 绑定源 SHA、尺寸与模型身份；Beauty 冻结用户选用的生成脸部 RGB，不能称为相机细节。`reference_appearance_stack.v2` 的颜色/明暗控制、T/B/S、场景和边界层消费对应冻结资产，不是 scoped 生成外观的附加步骤。

无参考本地编辑使用 `source_only_display_plan.v1`，不制造参考资产。保留背景时关闭参考背景、接触、阴影及其效果转移，人物仍可仿色；禁止特效与参考已烘焙效果冲突时报告冲突。

T 局部参数、色彩融合 B 与明暗平衡 S 共用源域边界合同。参考对应置信度描述观测可靠性，不直接决定应用边界；源材质、真实遮挡与强光色边缘在分析降采样前建立连通拓扑，回采样仍受该拓扑约束。同一可靠材质内可延拓有界的低频校正；没有观测的独立区域保持零校正，分析网格丢失的小区域明确报告未覆盖。新 T 参数保存 anchor 所属源连通域位图，在此域内延拓有界双线性节点，anchor hull 只描述观测；缺少当前应用域的参数拒绝执行。B/S 的置信度仅权重数据项，可靠度衰减按 fine component 连续计算并统一作用于观测目标，零与趋近零的置信度不会切出剂量边。柔和保护只影响连续剂量，不能阈值化成硬 owner。标量与向量约束分别验证实际解与收敛残差，不平滑源图 RGB 或扩张参考几何对应。

保存版本重开采用 `replay_frozen`，明确新准备采用 `prepare_current` 并绑定 checked plan 与执行身份；缓存须匹配实际输入、代码、参数与模型。源域版本保留适用图层控制；scoped 生成外观可查看、比较、审片和导出，不启用旧 T/B/S、Beauty 与手工像素编辑器。新制作保存独立版本，不覆盖父像素或继承人工批准。画布 producer 签名不单独证明像素缓存命中。

最新保存结果、显式选择版本与已批准交付分别记录。受协调器管理的任务在落盘时保留未提交标记，只有像素与生产合同均保存后才发布；失败草稿可保留诊断，但不能进入最近结果扫描或抢占默认打开版本。桌面失败回调解锁当前作品，不自动选择失败任务的子版本。独立调整、复制与导入成功后正常登记。重启默认打开最新保存结果；显式选择历史版本则保留该选择。修改审片记录及文件 mtime 不改变保存顺序，迟到的启动恢复响应不会覆盖新导入或正在编辑的内容。

相同输入、控制与 runtime binding 的活动 job 独占；重复开始返回活动 job 或 `duplicate_active_run`。计划/能力预检失败返回 `blocked`，不启动 worker。失败、取消、过期或后台重启保留父版本、当前编辑、首错与事件日志，不应用半成品，也不隐式扩大生成预算。

画布与重试仅接受当前 task intent/plan；缺少合同的历史结果仍可查看已有像素，不补造可执行图、升级签名或授予重试。缓存缺少当前字段时重建，不在旧记录上补齐。

## 原片与普通导入参考的图层及边界受光

主效由参考背景拥有。独立RGBA须声明未烘焙增量、`front|behind`及`over|emission_add`，不能重复消费参考主效。独立效果不自动建立材质受光、投影或反射。

发梢和肩袖分别使用`source_boundary_light`与`source_cloth_boundary_light`。它们是源坐标的冻结预乘RGB增量，不改alpha或人物几何；衣料在发梢之后、手工人物精修之前回放。单层可补充或重新准备，已有层只替换唯一同kind资产并保留开关、强度。

`boundary_light_coordination_worker`按`source_boundary_light_coordination.v1`先准备发梢，再以新发梢端点准备肩袖；完整资格通过才向桌面或后台返回一起应用的计划，不应用中间状态。取消、失败保持当前编辑；相同像素复用冻结资产，无变化不创建子版本。改变上游像素、控制或身份会使旧层停用，必须重新准备。界面入口为“同时更新边界受光”，方法及实际范围见 [人物边界受光](BOUNDARY_LIGHT.md)。

## 作品状态与本地恢复

阶段报告读取已保存事实和资产存在性，不重新证明大资产哈希，也不执行网络审片或像素修复。scoped 状态由 `scoped_product_state` 区分执行中、待发布、请求待核实、软件恢复与艺术待修；原生工具保留 `native_pipeline_stage_report.v1` 的适用状态。审片 fail 不等于软件执行失败。进度显示实际阶段及耗时；条件节点未触发时为 `not_requested`，不虚报原生细节准备完成或固定百分比。

| 状态 | 动作 |
|---|---|
| `awaiting_input` | 补原片或参考 |
| `needs_local_recovery` | 当前准备未完成；复用参考重试或重新准备 |
| `pending_save` | 保存当前编辑 |
| `needs_repair` | 查看当前失败证据并执行有资格修复 |
| `needs_evidence` | 补未知或缺失证据 |
| `needs_visual_review` | 人工审查已保存版本 |
| `accepted` | 当前保存版本已有明确人工批准 |

动作同时声明是否支持、原因、关联问题和是否需生成/QA。Qt、CLI及`GET /api/source-runs/<id>`共享完整报告；异步/批量执行状态与作品质量分开。阶段跳过不兑现必需主效。合同及接口见 [本机API](REVISION_API.md)。

以下本地恢复动作适用于原片/普通导入参考工具，不替代 scoped 的生成纠偏。`repair-final`只接受原片、已有参考、完整当前设计及已保存草稿。它取得当前像素绑定的新QA，按职责尝试一次有资格的材质、背景、人物环境或人物/接收面细节处理，另存子版本并复审；不新增参考或独立效果生图。父版本和工程选择保留，未知先补证据，未支持能力保持开放。

背景恢复先审计实际清理、注册与超分资产，资产已一致或资格不足时不制造修复。人物细节须证明匹配窗口内的实际结构损失，人物环境只处理可靠对应的低频残差，分别见 [人物细节核验](AUTOMATIC_SUBJECT_DETAIL.md) 和 [人物环境光色](AUTOMATIC_SUBJECT_ENVIRONMENT.md)。

接收面恢复仅接受已审片匹配鞋部窗口的`native_detail fail/native_retouch/retouch_missing`，同窗身份与材质须通过。共同地面、源鞋语义、原始/当前alpha及有效采样限定源同坐标亮度细节；保留当前背景低频色彩，不重画人物或制造反射。共同地面要求双alpha低于0.5，近阈值连续衰减。鞋下短缺口仅在有共同地面终点、长度受鞋宽和参考采样限制且路径不穿不透明前景时支持。资格不足或无改善保持父版；新像素另存并取得新审片。

## 审片与证据

scoped 独立审片由 `photo_session_review` 针对绑定的原图、候选和设计分别评价自然可信度、创作收益，结果进入同一 session 的最佳版本选择。原生工具的 `cosplay_post_review.v2` 使用 `cosplay_review_scope.v1`，分开记录整图、上下文、匹配原生窗口、眼口及覆盖缺口；不可靠时保留 unknown，不能用整脸替代五官。

诊断先持久化独占尝试并绑定输入、渲染及设计，长准备和请求前后重验身份。失败、取消、中断保留证据，重开不会覆盖首错。原生诊断失败板仅在其合同明确许可时重试；这不授予 scoped session 或生图请求额外重试。超时、断连、未完成、未知和语义 schema 错误不授权隐式重发。文本/视觉请求与生图预算分别记录。

同像素继承QA只能显示历史出处，不能授权当前修复。局部原生pass只覆盖列出的匹配窗口，整图艺术pass不认证全图原生质量；任务成功不自动批准作品。AI 审阅即使各项通过也不构成用户最终人工批准，用户须针对当前已保存版本单独确认。批准标准见 [Cos成片标准](COSPLAY_POST_STANDARD.md)。

## 安装、冻结与交付

运行源码来自`scripts/runtime_manifest.json`白名单，路径由`runtime_resources.code_path`定位。模型位于用户配置的安装目录，工程只承载素材、版本、缓存和回执；迁移工程不改变模型选择。准备阶段验证实际模型、代码及输入身份。

子版本保存自己的像素及JSON资产。portable v3 只接受当前 schema、当前 registry 身份和当前资产依赖；不把代码、模型、凭据或历史缓存作为可编辑工程资产。不迁移、升级或加载旧计划、旧 receipt 和淘汰模型。完整输入包可以在空目录重开，但必须重新生成当前计划并通过当前路线预检；冻结回放不推理。

TIFF16/32来自真实保存母版，PNG/JPEG从当前保存成片派生。RAW显示变换、source alpha、接触衰减和冻结增量不得重复消费。更多规则见 [可编辑工程与导出](PORTABLE_EDITABLE_PROJECT.md)，安装和验证见 [安装指南](INSTALLATION.md) 与 [开发指南](DEVELOPMENT.md)。

源码实现、离线合同验证、真实模型出图与独立安装验收分别成立；跨图艺术质量仍需实际审查。未解决能力集中记录在 [算法选择与范围](ALGORITHM_DECISIONS.md)，当前迭代事实见 [产品目标](PRODUCT_GOAL.md)。
