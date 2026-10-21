# AI harness 驱动的研发

本项目使用现有编码 agent 的模型、文件与命令工具、会话和上下文管理。仓库提供持久目标、明确的模块边界、可运行的软件和反馈，帮助 agent 连续推进真实产品。开发工具不另建照片生产流程。

## 每轮工作循环

`读取当前目标与事实 → 选择主要问题 → 按需读取代码和资产 → 调用工具执行 → 观察真实软件和作品 → 独立评价 → 保存事实与下一行动 → 继续`

开始时读取 [PRODUCT_GOAL.md](PRODUCT_GOAL.md)，运行：

```powershell
<desktop-python> -B scripts/engineering_context.py
```

它汇总当前任务、相关代码位置和修改、后续行动。需要接续私有运行证据时，指定 `--observations <directory>`；目录保存在工程或研发工作目录，不能放进公开源码。完整日志和图像保留为文件，给模型短摘要和定位信息，按需查看，避免重复装入整段历史。

观察正在执行的照片任务时可加 `--run-manifest <manifest.json>`。工具读取真实阶段、输出位置、计时快照和已提交请求，核对源文件字节；生成响应尚未确定时明确提示先查现有 provider 任务。观察不启动任务、不补发请求，不把运行中的计时快照当作最终耗时。

## 模型的判断与工具的责任

模型根据实际观察决定要查什么、修改哪里、调用什么工具、继续还是结束。现有 checked task、CLI、HTTP 后台、桌面程序、图像 worker、计时和审片工具负责真正执行。确定性代码管理输入身份、进程、资源、取消、资产归属和原子保存。

从 `docs/ENGINEERING_STATE.json` 的当前任务进入相关模块；无需每轮通读所有算法。状态表达待完成工作，不因填入状态就证明通过。完成情况由当前绑定的实际证据支撑。

一次通用代码修改应解决已经观察到的问题。新增抽象、节点、模型和提示词都要能解释对成片或用户流程的实际收益。修改图像算法前先看原图、参考与最终图，定位问题首先出现在哪一层。

## 反馈与独立审查

分别记录技术执行、实际图像观察和人工接受。执行者的成功摘要不代替审片。评价要直接观察当前资产，指出缺陷、范围和未知；正常观看的整图优先，再看确有必要的原生细节。

独立审片模型的目标来自原始用户要求、声明的角色事实、明确反馈及 checked 制作范围；实际图像角色保持真实。规划者的设计叙述保留在执行和证据绑定中，不作为用户硬要求或成功证据发送给审片模型。审片提示改变后，旧观察仍是历史事实，不能重签身份或改写账本使它成为当前评价。

审片与工具决策板只在图像 SHA 和 RAW/raster 解码分支都相同时合并物理面板；所有逻辑角色、原路径和 SHA 仍分别保存、逐文件核验。共享面板标明全部角色，映射保留其共同矩形；不同图像维持每格 512×768 的既有尺寸。合并减少重复视觉输入，不增加调用次数，也不证明艺术质量提高；提示和 producer 变化仍按当前身份校验缓存。

真实区域编辑的独立审片从已核验的 typed candidate core 与实际父版派生同坐标裁图，在同一次请求中先看整图、再看局部及周边融合。仅这种双行板选择 `detailed_board`（最长边2048）；规划、工具决策及非区域审片保持 `standard`（1024）。裁图事实与档位绑定提示、请求和审片证据，缺失或不符的档位不能接受，也不触发自动重试。局部板是重采样预览，不证明原生1:1细节、额外相机信息或人工接受。

下一轮工具决策优先观察已准入的 working；没有它时，可用 previous_attempt 只读观察最近一张同画布、当前目标下已审且不同于原片和最佳图的非最佳尝试。反馈以实际 observed_images 的 candidate/previous_best SHA 绑定，不能把历史判词归给当前最佳图。previous_attempt 不成为 working、不进入任何工具输入，也不改变最佳版本、批准或请求预算；执行仍只消费各工具原先获准的像素。

私有 observation JSON 的字段是 `task`、`code_identity`、`kind`、`result`、`finding`、`next_action` 和 `artifacts`。`kind` 为 `execution`、`visual` 或 `human`；`result` 为 `pass`、`fail` 或 `unknown`；每个 artifact 记录 `path` 与 `sha256`。使用 context 命令返回的 `code_identity`，证据文件由实际执行或审查产生。文件身份核对只证明绑定的字节，不能授予艺术通过或人工批准。

换代码或改变资产后，旧观察仍可用于理解原因，不能升级为当前实现已通过。不改旧 receipt 的身份字段冒充重新执行。不确定远程请求结果时先核对已有任务，避免重复生图。

视觉 HTTP 的每次实际尝试独立保存请求绑定、响应头及已收到的响应 ID，完整响应在解析设计 JSON 前落盘。部分流、断线与进程消失仍是未知结果，不能推断服务端失败或退款。记录不包含密钥或请求图像的 base64 正文；它们证明已观察到的响应，不授予重试、额外预算或人工批准。

重新进入同一 checked task 的观察准备时，只有当前源码、原片、实际 API JPEG、完整提示词、模型和连接选择均匹配，且完整终态响应已可靠保存，才可纯本地解析并完成尚未保存的设计。该恢复不查询连接或凭据、不发 HTTP，原观察仍计一次模型请求。无完整响应、设计不合法或旧版本记录均保留原状态；不迁移旧 producer，也不为未知请求创建替代任务。
若中断任务仍标为 running，现有任务去重仍会拒绝直接重入；先由任务列表的公开 `GET /api/jobs` 观察核对实际 worker 状态，再重新提交同一 checked request。不得手改任务、预算或删除锁来绕过这一边界。这不表示未知响应可取回，也不等同于已经验证所有桌面重启场景。

## 版本迭代与架构收敛

保持当前生产入口。先按职责缩小 UI、任务控制、运行环境、图像处理和结果状态之间的耦合，每次迁出一项完整职责并删除被替代代码。泛化依靠多图的同一默认实现验证，而非针对照片增加参数分支。

每轮结束前保存：当前版本/代码身份、实际执行和审片的文件位置、主要结论、未解决问题、下一项具体行动。只有运行检查时，只能报告运行证据；没有新图时不能宣称艺术质量提高。

## 参考

- [LangChain: The Anatomy of an Agent Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness)：模型之外的上下文、工具、运行环境与反馈。
- [Anthropic: Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)：持久进展和跨会话接续。
- [Anthropic: Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)：独立评价与生成的反馈循环。
- [OpenAI: Harness engineering](https://openai.com/index/harness-engineering/)：可理解的代码、实际可观察的软件和持续架构维护。
