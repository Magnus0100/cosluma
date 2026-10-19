# 当前产品数据流

桌面、CLI 和本机后台使用同一生产管线：`原片 → 参考（导入，或由 GPT / 本地 provider 生成）→ Beauty → 场景与原生 T/B/S → 边界受光 → 保存`。原片承担人物身份、几何及原生 alpha；BeautyGRPO 提供用户选择的生成脸部外观，参考提供仿色目标与场景。无参考制作沿用相同编排，关闭参考、场景及边界节点。

显式 `background_mode=enhance_source` 使用原生可见背景而非参考全景作为背景细节基底。纯离线 `creative_direction` 将用户角色、原景结构保持、主效与禁止项冻结进检查计划、实际提示词和生产合同；不伪称执行了角色研究。可选 `regional_refinement` 首版允许新创作任务，或显式使用本地Qwen的已有参考新独立任务，处理一个明确原生背景区域、至多一次提交；已有参考默认零新增，普通回放、重试及最终修复不续发额度。局部结果核验权限后冻结进同一背景资产，来源与预算证据可追踪，详见[原景增强](SOURCE_SCENE_ENHANCEMENT.md)。

## 任务入口与运行时

`creative_task` 检查任务并冻结计划，`production_coordinator` 核验执行合同并进入 `single_source_workflow`，后者调用当前准备与渲染 worker。`system_post_completion` 执行有界完成阶段，`pipeline_stage_report` 归约保存事实。制作入口为 `/api/tasks/plan`、`/api/tasks/start` 及其批量形式，CLI 使用 `task-plan` / `task-start`；内部 `one-click` 仅承接同一 checked task 的子进程执行。

唯一执行策略是 `layered_reference_transfer`。`local_portrait`、`existing_reference` 和 `create_reference` 是同一管线的输入与节点开关。有参考的任务固定完整后期并纳入 Beauty 安装预检；本地原片任务不消费参考，也不隐式准备新的 Beauty 结果，可消费此前显式准备的冻结美化层。GPT Image、本地 ComfyUI/Qwen 及已配置的兼容 provider 提供上游参考；本地Qwen还可执行显式获准的单区编辑，不切换后期实现。创建参考默认生成 **1** 张，可显式配置多个候选；已有参考的初始、纠错及独立效果生图预算均为零，主动开启原景增强单区新任务时另有独立区域最多1次预算。文本/视觉规划与外部审片由服务策略显式选择，默认关闭。

模型注册表选择顺序是：任务显式 `COSPLAY_MODEL_REGISTRY` → 用户保存的选择（`prefer_user=true`，或工程旁没有 portable 配置）→ 工程旁 `model/local-models.json` → 公共 `providers/local-models.current.json` → 公开模板。整项任务固定使用选定模型/runtime 身份；缺失或身份不符时停止并报告缺口。未请求的能力标为 `not_requested`。

| 能力 | 当前模型/运行时 | 启用条件 |
|---|---|---|
| `source_silhouette` | BiRefNet ONNX、ONNX Runtime | 本地原片节点子集 |
| `source_geometry` | Sapiens2 0.4B、ViTMatte、BiRefNet ONNX、ONNX Runtime | 有参考的基础准备 |
| `reference_correspondence` | DINOv3 ViT-L/16 | 有参考的对应准备 |
| `scene_analysis` | SAM3.1 静态 detector 与隔离 Python | 有参考的场景准备 |
| `portrait_regions` | 五官定位模型与独立 Python | `complete_post=true` |
| `background_completion` | LaMa | `complete_post=true` |
| `background_superresolution` | 当前 RealESRGAN checkpoint 与 SR Python | `complete_post=true` 的背景重建 |
| `portrait_beauty` | BeautyGRPO 独立安装与专用 Python | 有参考的当前任务，或请求提交冻结美化层；安装预检读取回执，Beauty worker 核验执行身份 |

`cos_node_registry`、`cos_workflow_graph` 和 `cos_workflow_cache` 为同一管线声明端口、参数、依赖及 UI 失效提示。实际执行仍由当前 Python 协调器与 worker 完成；画布计划的 `reused` 不证明真实缓存命中。未接入生产的占位 Node/callback 执行器已删除。具体范围见 [ComfyUI 架构借鉴](COMFYUI_ARCHITECTURE.md)。

## 计划、执行与保存合同

输入是 `task_plan/plan_id`、mode、唯一 execution strategy、源/参考快照、模型/runtime 身份、控制参数与预算；输出是有归属的 job/run、真实阶段回执、`stage_report`、独立 revision 和导出回执。批量每张照片独立建立这些事实。

1. 计划只读取文件、配置与已记录身份，不联网、不推理；执行前重新核验绑定。
2. 解码与批准精修建立冻结源基线。RAW 仅进行一次源显示变换进入 `display_linear_sRGB`；普通显示图使用恒等/裁切准备。保存编码不能重复施加 RAW 变换。
3. 参考由导入或选定 provider 提供；按启用节点准备语义、alpha、场景实例及对应。源语义以 `source_semantic_binding.v1` 绑定源 SHA、尺寸与模型身份。
4. Beauty 冻结实际生成脸部 RGB，随后准备人物仿色/明暗、背景及场景。`reference_appearance_stack.v2` 独立颜色/明暗控制消费完整四角点银行及 support 回执。
5. 发梢后肩袖准备边界受光，再读取候选人物低频光照证据与背景环境光；人物、背景、接触与明确独立效果在同一工作域合成。预览、原尺寸 ROI、保存及重开消费同一冻结资产。
6. 保存 revision 与冻结母版，按显式审片策略取得当前像素证据。人工批准与软件执行、保存成功分别记录。

无参考本地编辑使用 `source_only_display_plan.v1`，不制造参考资产。保留背景时关闭参考背景、接触、阴影及其效果转移，人物仍可仿色；禁止特效与参考已烘焙效果冲突时报告冲突。

T 局部参数、色彩融合 B 与明暗平衡 S 共用源域边界合同。参考对应置信度描述观测可靠性，不直接决定应用边界；源材质、真实遮挡与强光色边缘在分析降采样前建立连通拓扑，回采样仍受该拓扑约束。同一可靠材质内可延拓有界的低频校正；没有观测的独立区域保持零校正，分析网格丢失的小区域明确报告未覆盖。新 T 参数保存 anchor 所属源连通域位图，在此域内延拓有界双线性节点，anchor hull 只描述观测；缺少当前应用域的参数拒绝执行。B/S 的置信度仅权重数据项，可靠度衰减按 fine component 连续计算并统一作用于观测目标，零与趋近零的置信度不会切出剂量边。柔和保护只影响连续剂量，不能阈值化成硬 owner。标量与向量约束分别验证实际解与收敛残差，不平滑源图 RGB 或扩张参考几何对应。

当前保存版本的重开、普通微调及最终局部修复采用 `replay_frozen`；明确重新制作、本地重试和新制作采用 `prepare_current`，并绑定 checked task plan 与执行身份。后者必须进入 producer 身份核验与准备，可复用当前输入、代码及模型完全一致的缓存；保存独立子版本并重新准备依赖新 T/B/S 的边界层。四项图层控制、Beauty 与手工图层保留，父像素、旧快照和人工批准不会被覆盖或继承。轻量 currentness 报告区分冻结代码与当前安装版本；画布计划签名包括真实 producer 代码，但签名可复用不等于实际像素缓存命中。

最新保存结果、显式选择版本与已批准交付分别记录。受协调器管理的任务在落盘时保留未提交标记，只有像素与生产合同均保存后才发布；失败草稿可保留诊断，但不能进入最近结果扫描或抢占默认打开版本。桌面失败回调解锁当前作品，不自动选择失败任务的子版本。独立调整、复制与导入成功后正常登记。重启默认打开最新保存结果；显式选择历史版本则保留该选择。修改审片记录及文件 mtime 不改变保存顺序，迟到的启动恢复响应不会覆盖新导入或正在编辑的内容。

相同输入、控制与 runtime binding 的活动 job 独占；重复开始返回活动 job 或 `duplicate_active_run`。计划/能力预检失败返回 `blocked`，不启动 worker。失败、取消、过期或后台重启保留父版本、当前编辑、首错与事件日志，不应用半成品，也不隐式扩大生成预算。

画布与重试仅接受当前 task intent/plan；缺少当前节点图、发布凭据或资产协议的记录不进行只读降级、自动升级或重新解释。缓存缺少当前字段时重建，不在旧记录上补齐。

## 图层与边界受光

主效由参考背景拥有。独立RGBA须声明未烘焙增量、`front|behind`及`over|emission_add`，不能重复消费参考主效。独立效果不自动建立材质受光、投影或反射。

发梢和肩袖分别使用`source_boundary_light`与`source_cloth_boundary_light`。它们是源坐标的冻结预乘RGB增量，不改alpha或人物几何；衣料在发梢之后、手工人物精修之前回放。单层可补充或重新准备，已有层只替换唯一同kind资产并保留开关、强度。

`boundary_light_coordination_worker`按`source_boundary_light_coordination.v1`先准备发梢，再以新发梢端点准备肩袖；完整资格通过才向桌面或后台返回一起应用的计划，不应用中间状态。取消、失败保持当前编辑；相同像素复用冻结资产，无变化不创建子版本。改变上游像素、控制或身份会使旧层停用，必须重新准备。界面入口为“同时更新边界受光”，方法及实际范围见 [人物边界受光](BOUNDARY_LIGHT.md)。

## 作品状态与本地恢复

`native_pipeline_stage_report.v1`的`whole_work`记录状态、开放问题、必需效果及恢复动作。报告读取已保存事实和资产存在性，不重新证明大资产哈希，也不执行网络审片或像素修复。

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

`repair-final`只接受原片、已有参考、完整当前设计及已保存草稿。它取得当前像素绑定的新QA，按职责尝试一次有资格的材质、背景、人物环境或人物/接收面细节处理，另存子版本并复审；不新增参考或独立效果生图。父版本和工程选择保留，未知先补证据，未支持能力保持开放。

背景恢复先审计实际清理、注册与超分资产，资产已一致或资格不足时不制造修复。人物细节须证明匹配窗口内的实际结构损失，人物环境只处理可靠对应的低频残差，分别见 [人物细节核验](AUTOMATIC_SUBJECT_DETAIL.md) 和 [人物环境光色](AUTOMATIC_SUBJECT_ENVIRONMENT.md)。

接收面恢复仅接受已审片匹配鞋部窗口的`native_detail fail/native_retouch/retouch_missing`，同窗身份与材质须通过。共同地面、源鞋语义、原始/当前alpha及有效采样限定源同坐标亮度细节；保留当前背景低频色彩，不重画人物或制造反射。共同地面要求双alpha低于0.5，近阈值连续衰减。鞋下短缺口仅在有共同地面终点、长度受鞋宽和参考采样限制且路径不穿不透明前景时支持。资格不足或无改善保持父版；新像素另存并取得新审片。

## 审片与证据

`cosplay_post_review.v2`使用`cosplay_review_scope.v1`分开记录整图、上下文、匹配原生窗口、眼口及覆盖缺口。源特征观察复用独立语义缓存和CPU定位，不依赖材质finish；不可靠时保留unknown，不能用整脸替代五官。

诊断先持久化独占尝试并绑定源、参考、渲染及设计，长准备和请求前后重验身份。失败、取消、中断保留证据，重开不会覆盖首错。仅明确429/5xx、已识别上游服务错误或完成响应中的损坏JSON允许失败板重试一次；成功板不重发，超时、断连、未完成、未知和语义schema错误不授权隐式重试。文本/视觉请求与生图预算分别记录。

同像素继承QA只能显示历史出处，不能授权当前修复。局部原生pass只覆盖列出的匹配窗口，整图艺术pass不认证全图原生质量；任务成功不自动批准作品。AI 审阅即使各项通过也不构成用户最终人工批准，用户须针对当前已保存版本单独确认。批准标准见 [Cos成片标准](COSPLAY_POST_STANDARD.md)。

## 安装、冻结与交付

运行源码来自`scripts/runtime_manifest.json`白名单，路径由`runtime_resources.code_path`定位。模型位于用户配置的安装目录，工程只承载素材、版本、缓存和回执；迁移工程不改变模型选择。准备阶段验证实际模型、代码及输入身份。

子版本保存自己的像素及JSON资产。portable v2 只接受当前 schema、当前 registry 身份和当前资产依赖；不把代码、模型、凭据或历史缓存作为可编辑工程资产。不迁移、升级或加载旧计划、旧 receipt 和淘汰模型。完整输入包可以在空目录重开，但必须重新生成当前计划并通过当前路线预检；冻结回放不推理。

TIFF16/32来自真实保存母版，PNG/JPEG从当前保存成片派生。RAW显示变换、source alpha、接触衰减和冻结增量不得重复消费。更多规则见 [可编辑工程与导出](PORTABLE_EDITABLE_PROJECT.md)，安装和验证见 [安装指南](INSTALLATION.md) 与 [开发指南](DEVELOPMENT.md)。

未解决能力集中记录在 [算法选择与范围](ALGORITHM_DECISIONS.md)。
