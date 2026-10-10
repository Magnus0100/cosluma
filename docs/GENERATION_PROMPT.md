# 当前生图提示词与设计合同

当前流程要求完整 Cos 后期：本人妆造、精修、角色场景、主要效果、空间融合与材质光色。最终人物结构/高清纹理由原片提供，生成图是参考。验收见 [COSPLAY_POST_STANDARD.md](COSPLAY_POST_STANDARD.md)，数据流见 [PRODUCT_ARCHITECTURE.md](PRODUCT_ARCHITECTURE.md)。

brief 为 `cosplay-post-v4`，风格为 `cosplay-complete-v2`，设计为 `cosplay-researched-design-v6-cosplay-post`。

| 来源 | 职责 |
|---|---|
| `generation_prompt.py` | 成片目标、人物相机保持、场景改造和材质区别 |
| `presets/presets.json` | 场景主题、效果、光色和完成度 |
| `generation_design.py` | 可见造型及有来源研究形成完整计划，逐项编译 |
| 当前用户 brief/备注 | 效果与限制，优先于预设 |

识别研究 → 完整计划 → `effective_prompt` → 参考生成 → 原片仿色/独立合成 → 全面审片。文本版本、风格快照、源图、工作流、模型和输出均保存来源。

已有参考路线为：只读任务预检 → 一次原片/参考成对观察 → 绑定设计 → 原片仿色、背景与现有合格效果层合成 → 全面审片与有资格的本地修复。该路线不生成参考、纠错图或独立效果图；不自动扩张角色研究。设计声明 `reference_background_policy`与 `reference_effect_observation`，以当前任务限制决定真实合成消费，不能只修改提示词。

切换风格保留 brief，替换 `ART DIRECTION` 并清除旧自动角色/设计段。取消风格只清受管理段，“恢复默认”才重置文字。精确重试保留实际文本与快照；重新规划须获得完整当前计划。失败保留原片与版本，不用旧计划或自动调色替代成片。

## 必要字段

必需非空 `summary`、`scene_analysis`、`character_basis`、`subject_retouch_plan`、`spatial_integration_plan`、`lighting_plan`、`palette_material_plan`、`tradeoffs`、`reference_application`、`english_art_direction`，并校验：

- `scene_transformation_plan`：`refine|transform`、描述、非空 `visible_changes`。
- `primary_effect_plan`：`none|scene_effect`、描述与目的。
- `user_constraints`：引用当前用户原文，模板、预设和旧设计不能冒充限制。
- `foreground_effect_plan`：明确的 `enabled` 与 `layers`；未提供时保存为关闭的空计划。主效已烘焙于参考或无需独立增量时使用 `{"enabled":false,"layers":[]}`。

普通人像允许 refine/none，仍须完成精修层次和光色。舞台/夜景/幻想/仪式须可见场景改造和有目的主效，用户限制优先。魔法不是所有风格必需。

场景、效果、空间、精修、材质和取舍逐项进入发送文本。`composition_plan/environment_effects_plan` 从当前结构派生供场景消费者使用，不接收旧数字计划或前图别名。文字不能代替 alpha、RGBA、投影资格或三维光照；缺字段/约束冲突/旧数字计划不能 ready。

## 类型化独立效果

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

## 质量与纠错

实际实施由审片判断，眼白/虹膜/唇妆/裸肤/衣料/金属应有区别和连续过渡。T/B/S 迁移低频光色，纹理精修与跨人物特效各自独立。

QA 以 `cosplay_review_scope.v1` 保存判断范围。整图与上下文承担艺术/场景判断；原生窗口只判断对应的原片/成片 pair，并记录眼口妆造、材质和细节。局部不得借另一窗口证据，也不能凭原片/成片 pair 断言参考原生缺陷。实际 API 缩小输入时不授予 1:1 原生证据。`native_detail` 通过只覆盖请求且实际观察的窗口；未覆盖区域、眼口未知继续开放，不授予全图原生通过。

参考纠错仅消费完整当前 QA 中明确归因于参考的失败，使用原片、参考和受限 mask 的多参考输入，核对实际请求/输出。先做原生 staging 子版本，再复核当前成片 QA；失败、unknown、unavailable 保留草稿。预算不递归，不自动批准或选择。

当前最终成片的问题另由 `repair-final` 处理：不新增参考或独立效果生图，会请求新的文本/视觉 QA；先取得完整当前绑定诊断，再执行一次有资格的本地材质、背景、人物环境或人物/接收面细节处理并复审。背景先核对实际资产，未证明需要修复时保留问题；未知补证据，原生精修资格不足及未支持职责不转成随机生图。同像素继承 QA 只展示出处与历史结论，不能授权这次修复。

共享作品结果保留 `needs_repair`、`needs_evidence` 或 `needs_visual_review` 等状态及对应动作。独立效果跳过不能兑现必需主效；当前最终证据和人工批准分别记录。工程执行结束不代表完整自动成片已获视觉通过。

测试证明合同被消费，真实生成证明服务运行，都不授予视觉通过。作品状态以用户工程中所选版本与绑定审片为准。
