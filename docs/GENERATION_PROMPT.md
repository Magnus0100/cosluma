# 当前生图提示词与设计合同

默认目标是在原场景内完成可信的 Cos 半合成：人物光影、适度场景变化、角色氛围和材质协调有可见收益。自然可信度与创作收益分别评价；统一染色、压暗或堆雾不能代替完成度。验收见 [Cos成片标准](COSPLAY_POST_STANDARD.md)，数据流见 [产品架构](PRODUCT_ARCHITECTURE.md)。

设计应说明位置、形态、材质、尺度、遮挡和焦点；发光效果须指明实际受光面。

## 当前 scoped 创建

`editing_request` 以 `cosluma_editing_scope.v1` 冻结制作范围与风格。`work_type` 为 `retouch|partial_composite|full_composite`，`style` 为独立的 `natural|expressive|dramatic`；桌面默认半合成、鲜明、原景。精修允许模型选择原片上的获准局部编辑，整图生成不属于精修能力。只有大合成明确授权才允许 `scene_target/target_setting` 改变场景目标，不能静默室内变室外；风格与重光不自动授权瘦脸瘦身。

| 来源 | 职责 |
|---|---|
| `photo_editing.py` | 范围校验、`cosluma_observed_design.v1`、模型专属提示词编译 |
| `photo_art_direction.py` | 执行时真实观察原图，冻结 `generation_design.observed_design` 与观察身份 |
| 当前角色、brief、范围和限制 | 约束本图设计，不由预设或上一张图推定授权 |
| `photo_edit_tools.py`、`photo_edit_finish.py` | 共用 session 中的有界改进、独立评价及最佳完整版本保存 |

计划阶段不联网、不推理。开始后实际观察原图，形成本图设计，再分别编译 Qwen 与 GPT 指令；提示词版本为 `cosluma_model_edit_prompt.v1`，编译记录为 `cosluma_compiled_edit_prompt.v1`。实际发送文本、模型身份、输入和输出须可追溯。模型用法依据见 [模型专属提示词](MODEL_PROMPTING.md)。

原图约束人物身份、妆造、姿态、重要服装道具和场景事实。0.7 开发流程先解码原片并只执行一次显示准备，冻结 `photo_appearance` 的 float32 基线；模型选择摄影显影、局部编辑或联合整图编辑。实际覆盖之外直接保留上一版浮点像素；整图编辑允许共同改变人物与环境受光，不再把原人物 RGB 整体盖回。生成结果的真实透明度也属于覆盖证据，不能丢弃后把隐藏 RGB 当作可见内容。源像素责任与累积生成覆盖分别保存，生成细节不描述为原生相机纹理。

当前照片工具注册结果时保留真实 alpha，将线性预乘 RGB 与 alpha 分别做相同坐标的双线性注册，再与当前母版合成。局部实际覆盖为模型透明度乘许可区域羽化，透明区域不带入隐藏颜色。整图结果完全不透明时使用全生成资产；含透明或半透明时保存真实混合覆盖与父版贡献，不能伪称全生成。不对结果套用原人物抠图。相应整画幅混合权限只用于半合成或大合成；精修的权限不随文件透明度扩大。无实际覆盖的返回图不作为有效编辑发布。

scoped 创建需要执行时视觉设计，独立审片默认开启；`editing_autofinish={"maximum_improvements":2,"maximum_zero_image_finishes":1}` 默认预留两轮条件改进及一次仅允许零生图工具或停止的收尾。单候选预算上界为图像 **3** 次、视觉 **10** 次：设计、共享原片基线审片、首次工具决策和复审，以及至多三轮后续决策和复审；最后收尾不授权生图。首次编辑通过实际只用图像 **1** 次、视觉 **4** 次。原片基线通过不算完成编辑。用户可以关闭；显式只设置一次通常改进不增加收尾轮次。关闭独立审片也关闭自动改进和收尾，但仍执行设计、首次模型决策和实际工具，保存未审草稿。未知与停止决策占用原轮次，重开不刷新额度。

初审未达标时，模型依据原图、当前最佳与真实审片，选择生成编辑、摄影融合或停止；工具执行与复审共用现有 session。摄影融合只消费有真实合格局部增量的 working，新增生图为零，仍消耗一轮工具决定与独立复审。working 不自动成为最佳；修复无收益或变差时保留最佳完整版本。`photo_request_budget` 将额度绑定任务来源，候选选择、子版本、重开和重试不续发；结果未知先查原请求，不自动重提或退还额度。实际外部调用取来源账本；缓存复审不按会话尝试条数重复计费。

`develop_photograph` 对当前完整 float 母版执行曝光、相对暖度/色偏与可选区域软化，不新增生图、不重新解码 RAW。每条 `adjustments` 包含当前画布矩形 `region`、`exposure_ev`、`warmth_ev`、`tint_ev` 和按区域短边计的 `feather`，可选 `focus_sigma` 为原生画布短边的 0–0.02，省略或零严格不模糊。各区域从同一份不可变当前母版计算 Gaussian，按有序柔和覆盖合并，再执行曝光与色平衡；不是重复叠加模糊。区域包含的原片与生成像素都会处理，不作人物分割，也不只处理生成增量。模型须根据实际图像选择背景区域和羽化，避免影响皮肤与道具。实现用带完整 Gaussian 邻域的分带计算、画布边缘反射，区域覆盖之外逐 float 精确保留，累积生成覆盖继承父版。receipt 记录真实参数、顺序、方法及输出，冻结时从父版重算核对；这只是摄影近似，不证明物理光学或艺术收益。该零生图工具仍占一次首次操作或已检查的改进步骤。

同一次开始包含上述条件闭环。另行点击改进才创建新的 `editing_iteration` checked task，绑定已提交父版本、真实审片和原范围，走相同工具。即使自动两项通过，用户仍可填写明确的 `user_feedback` 申请一次新的有界改进；旧通过仅作历史基线，不证明新反馈已满足。不对任意导入图冒充支持。

改进阶段由决策模型看当前图、设计及真实审片，写出本轮完整编辑指令。执行器只编译该指令与当前制作范围、身份和几何保留要求；不再重复拼接旧设计、旧审片或旧外观保留清单。需要修改的光照与景深不得同时被要求保持现状。Qwen 使用直接的连续编辑文字，GPT 保留清楚的分节；局部工具仅描述实际裁图及其许可边界。

## 原片与普通导入参考的设计字段

以下为仍有效的源域工具合同，不是 scoped 生成外观的必经步骤。普通导入参考默认零新增图像；明确检查的原景单区新任务另有有界额度，见 [原景增强](SOURCE_SCENE_ENHANCEMENT.md)。其 `generation_design.py`、T/B/S、Beauty、alpha 与边界层按对应工具启用。

必需非空 `summary`、`scene_analysis`、`character_basis`、`subject_retouch_plan`、`spatial_integration_plan`、`lighting_plan`、`palette_material_plan`、`tradeoffs`、`reference_application`、`english_art_direction`，并校验：

- `scene_transformation_plan`：`refine|transform`、描述、非空 `visible_changes`。
- `primary_effect_plan`：`none|scene_effect`、描述与目的。
- `user_constraints`：引用当前用户原文，模板、预设和旧设计不能冒充限制。
- `foreground_effect_plan`：明确的 `enabled` 与 `layers`；未提供时保存为关闭的空计划。主效已烘焙于参考或无需独立增量时使用 `{"enabled":false,"layers":[]}`。

所有风格均允许 refine/none。风格首先体现为人物精修、妆造与服装色彩、主光塑形和画面主次；风格名称不授予换景或强制加特效。默认沿用原地点与空间布局，允许局部整理、替换干扰细节和克制氛围；明确要求完整换景时才设计 transform。暗调与恐怖题材仍保留皮肤高光、清晰眼神和暗部色彩，不能以全图压暗、染蓝、堆雾替代完成度。

场景、效果、空间、精修、材质和取舍逐项进入发送文本。`composition_plan/environment_effects_plan` 从当前结构派生供场景消费者使用，不接收旧数字计划或前图别名。文字不能代替 alpha、RGBA、投影资格或三维光照；缺字段/约束冲突/旧数字计划不能 ready。

## 源域工具的类型化独立效果

主效归参考背景所有。额外 RGBA 必须明确为 `increment=true`、`baked_in_reference=false`，禁止重复添加已烘焙或归属未知的烟雾。当前规划提示词要求每个任务最多一个额外独立层；前/后位置是艺术遮挡声明，不建立物理深度。

| 字段 | 当前合同 |
|---|---|
| `id` / `description` | 唯一安全标识与实际效果意图 |
| `slot` | `front` 或 `behind` |
| `operation` | `over` 或 `emission_add`，与材质匹配 |
| `occlusion` | 前层保护脸手或显式置前；后层使用原片人物遮挡 |
| `receiver_response` | 仅 `none`；不生成表面光照、阴影或反射 |
| `effect_design` | 仅 `material`、`colour_rgb8`、`extent_2d`、`density` |

`over` 材质为 `scattering_smoke|atmospheric_haze|dust_particles`；`emission_add` 为 `luminous_particles|light_beam`。颜色是三个 0–255 整数，范围是 [0,1] 内有效 `[left,top,right,bottom]` 矩形，密度固定 `sparse`。效果生产者从真实可见背景测量环境颜色与亮度，设计文本不猜测环境光。参考生成提示词明确排除单独增量；实际效果模型条件来自类型化白名单，描述用于意图与 QA。

## 源域工具的质量与纠错

实际实施由审片判断，眼白/虹膜/唇妆/裸肤/衣料/金属应有区别和连续过渡。T/B/S 迁移低频光色，纹理精修与跨人物特效各自独立。

QA 以 `cosplay_review_scope.v1` 保存判断范围。整图与上下文承担艺术/场景判断；原生窗口只判断对应的原片/成片 pair，并记录眼口妆造、材质和细节。局部不得借另一窗口证据，也不能凭原片/成片 pair 断言参考原生缺陷。实际 API 缩小输入时不授予 1:1 原生证据。`native_detail` 通过只覆盖请求且实际观察的窗口；未覆盖区域、眼口未知继续开放，不授予全图原生通过。

参考纠错仅消费完整当前 QA 中明确归因于参考的失败，使用原片、参考和受限 mask 的多参考输入，核对实际请求/输出。先做原生 staging 子版本，再复核当前成片 QA；失败、unknown、unavailable 保留草稿。预算不递归，不自动批准或选择。

源域工具的最终成片问题由 `repair-final` 处理，不替代 scoped 的生成纠偏：不新增参考或独立效果生图，会请求新的文本/视觉 QA；先取得完整当前绑定诊断，再执行一次有资格的本地材质、背景、人物环境或人物/接收面细节处理并复审。背景先核对实际资产，未证明需要修复时保留问题；未知补证据，原生精修资格不足及未支持职责不转成随机生图。同像素继承 QA 只展示出处与历史结论，不能授权这次修复。

共享作品结果保留 `needs_repair`、`needs_evidence` 或 `needs_visual_review` 等状态及对应动作。独立效果跳过不能兑现必需主效；当前最终证据和人工批准分别记录。工程执行结束不代表完整自动成片已获视觉通过。

测试证明合同被消费，真实生成证明服务运行，都不授予视觉通过。作品状态以用户工程中所选版本与绑定审片为准；跨图质量和当前独立安装仍需分别验收。
