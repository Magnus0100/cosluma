# 当前本地编辑接口

入口为 `scripts/backend_server.py` 和 `scripts/qt_workbench.py`。Qt、后台、CLI 共享当前请求及组件替换规则，以下以源码路由为准。

| 接口 | 行为 |
|---|---|
| `POST /api/tasks/plan` | 只读检查三路线任务，返回绑定输入和要求的 `plan_id`与安全执行 `request` |
| `POST /api/tasks/start` | 消费上述 `request`＋`plan_id`，重新检查后创建制作任务 |
| `POST /api/tasks/batch/plan` | 检查共享要求与逐图原片/参考，返回独立计划和批次 `plan_id` |
| `POST /api/tasks/batch` | 全部输入再次核验后逐图入队，几何和设计不共享 |
| `POST /api/tasks/batch/<id>/retry-item` | 以 `index`选择失败项；保留已有参考，未知生图结果不自动再提交 |
| `GET /api/source-runs/<id>` | 读取保存工程与共享 `stage_report` |
| `GET /api/source-runs/<id>/history` | 读取版本 |
| `POST /api/source-runs/<id>/preview` | 当前编辑预览，不改母版 |
| `POST /api/source-runs/<id>/native-detail` | 保存版本原尺寸局部 |
| `POST /api/source-runs/<id>/native-detail-preview` | 未保存编辑原尺寸局部 |
| `POST /api/source-runs/<id>/matte-preview` | 覆盖率预览 |
| `POST /api/source-runs/<id>/matte-adjust` | 保存覆盖率并更新真实依赖 |
| `POST /api/source-runs/<id>/adjust` | 保存新版本 |
| `POST /api/source-runs/<id>/copy` | 复制工程 |
| `POST /api/source-runs/<id>/reconstruct-design` | 完整当前准备，先进入未保存编辑，不改父版本 |
| `POST /api/source-runs/<id>/complete-post` | 从已保存原片/参考和完整设计创建独立后期任务，保留父版本与工程选择 |
| `POST /api/source-runs/<id>/repair-final` | 取得当前绑定新QA，执行一次有资格的本地修复并复审；参考/独立效果生图预算为零 |
| `POST /api/source-runs/<id>/accept` | 明确提交当前已保存版本的人工审片、reviewer与checks |
| `POST /api/source-runs/<id>/export` | 真实母版导出 |
| `POST /api/source-runs/<id>/export-project` | portable v2 打包 |
| `POST /api/projects/import` | portable v2 验证导入 |

`options.appearance_revision` 使用 `params`、`regions`、当前显示变换及显式源/重建计划；笔画位于 `options.manual_layers`。省略组件继承保存值，提供组件整体替换，空对象/关闭也明确处理。`parameters` 等别名、旧 trust/alpha/alignment 升级及 `controls_only` 不受支持。

人物层为 `reference_appearance_stack.v2`，顶层 `strength/colour_strength/lightness_strength` 均 0..1，消费真实四角点银行与回执。`params.reference_support` 仅是支持证据，不代表非零效果或视觉接受。会话草稿是编辑参数，不能替代冻结资产。

`stage_report.whole_work` 提供作品状态、开放问题、主效履约与恢复动作。报告和已有证据读取不执行修复或新QA。`complete-post` 按完整设计和有界预算协调后期，`repair-final` 则限制为已有参考上的本地恢复；后者会新增文本/视觉QA请求。任务执行成功、自动诊断与当前版本人工批准分别记录，未知或未支持问题继续开放。生产职责和提交条件见 [PRODUCT_ARCHITECTURE.md](PRODUCT_ARCHITECTURE.md)。

## 边界受光准备

`reconstruct-design`请求支持`preparation_kind`：

| 值 | 当前行为 |
|---|---|
| `reference_appearance` | 准备人物仿色融合，默认值 |
| `hair_boundary_light` | 单独准备或替换参考发梢受光 |
| `cloth_boundary_light` | 对当前发梢/外观端点准备或替换肩袖暖光 |
| `boundary_light_coordination` | 先发梢、后肩袖，完整资格通过才一起应用 |

联合合同为`source_boundary_light_coordination.v1`。两层仍可独立禁用或调强度；失败、取消或无变化不新增子版本。准备返回当前编辑计划，须预览并另存高清版本；上游端点变化时旧层停用，不偷偷重算。接口成功不代表视觉接受，方法和限制见 [人物边界受光](BOUNDARY_LIGHT.md)。

## 批量与后台

- 完整制作使用 `/api/tasks/batch/plan`与 `/api/tasks/batch`：`sources`用于生成或本地路线，`items:[{source,reference}]`用于已有参考；顶层共享 `creative_task`、服务与当前模型配置。
- 已有参考任务三项生图预算均为零；外部文本/视觉规划和审片由 `creative_task.services.text_vision`独立声明。完整参考路线未允许该服务时明确阻断，本地原片路线可离线执行。
- `GET /api/batch`、`GET /api/batch/<id>`：批次和逐项状态。
- `POST /api/tasks/batch/<id>/retry-item`：失败/取消新尝试，保留旧 attempt，禁止重复提交同一活动项。
- `POST /api/source-runs/export-batch`：逐项指定保存 run id/artifact，单项失败独立。
- `/api/jobs`、`/api/jobs/<id>`：后台任务与恢复依据。

保存 run id、revision 和渲染记录须一致。输入、模型、快照或资产改变时拒绝过期结果，取消/失败/重试/输出分别记录。成功不自动 accepted。见 [PORTABLE_EDITABLE_PROJECT.md](PORTABLE_EDITABLE_PROJECT.md)。
