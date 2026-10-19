# 当前本地生成架构

ComfyUI 是可选的本地参考生成 provider，不是 Cos 后期执行器。生产执行由当前 task plan、production coordinator、native workers 和冻结资产回执证明。

当前桌面程序使用唯一入口和当前 provider 配置。请求前验证 provider、模型、源图与参考图身份、任务预算和运行时能力；未通过时不提交请求。

生成完成后只登记候选。人物外观、场景、边界受光和保存仍由唯一 `layered_reference_transfer` 流程完成。缓存、取消、失败和重开读取当前 receipt，不从旧 options 或旧输出推断新任务。

Beauty 只接受 `nunchaku_cuda`。空闲且本请求拥有的 Beauty 服务在下一个 GPU 阶段前按内存合同释放；外部或忙碌服务保持原状。当前流程的节点耗时见 [当前性能记录](WORKFLOW_PERFORMANCE.md)。
