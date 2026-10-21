# 当前配置与预设

当前 brief `cosplay-post-v6`，风格 `cosplay-character-portrait-v3`。`preset_store.py/provider_settings.py` 保存 `presets/presets.json`、`providers/registry.json` 和公开连接设置。模型由当前本地注册表管理。

预设决定方向，本图计划决定布局，预设不是渲染结果或质量承诺。

| 风格 | 方向 |
|---|---|
| 原景角色精修 | 人物妆造、主光塑形、原景局部整理与适合角色的克制氛围 |
| 清透角色写真 | 明亮通透、眼妆与干净色彩、轻盈背景 |
| 舞台光角色写真 | 人物主光与局部色光；不默认搭建新舞台或添加光束 |
| 暗调角色写真 | 清晰眼神、皮肤高光、服装色彩和丰富暗部 |
| 柔和人像精修 | 皮肤妆造、深度、自然衔接，可无特效 |
| 逆光氛围写真 | 面部塑形和选择性轮廓光，背景有呼吸感 |
| 幻想角色写真 | 原场景内局部角色意象，有需要时才使用合适特效 |
| 民俗仪式写真 | 表情、妆造、材质与含蓄叙事，不默认符号墙或浓雾 |

这些是本软件的审美方向，不能代表所有 Cos 用户的统一偏好。所有风格默认延续原地点和布局、允许局部场景修饰，不强制换景或加特效；“画质高”和“特效多”都不等于风格好。画面先验收人物表现、光影和主次，再检查技术细节。

方向依据包括 [Cosplayer マヒオ的修图访谈](https://www.asukanet.co.jp/main/photo/column/column_093.html)（局部肤色、眼神与对比）、[小神的 Phantom 拍摄与后期记录](https://godlingstudio.wordpress.com/2012/11/26/phantom-cos%E6%8B%8D%E6%91%84%E5%9C%BA%E8%AE%B0%E5%90%8E%E6%9C%9F-%E5%85%A8%E6%B5%81%E7%A8%8B%E8%A7%A3%E8%AF%BB/)（角色叙事、布光、背景主次与效果布局）及 [Frekki 的角色后期实例](https://frekki.wordpress.com/2017/03/04/cosplay-photo-edit-walkthrough/)（角色背景、妆容、景深）。这些是不同作者的实践实例，本产品据此归纳目标，不据此声称统计意义上的流行趋势或要求复刻其作品。

五官、妆造、美瞳、姿态、服装和相机保持，颜色可适应场景。用户禁特效/保背景优先，研究事实与创意分记。切换风格保留 brief，只改受管理段及旧自动设计；精确重试保留文本、版本和快照。“恢复默认”才重置文字。

原片和普通导入参考的源域整体/颜色/明暗控制消费 `reference_appearance_stack.v2`，与生图风格分别记录。背景效果随合格背景消费，跨人物效果走独立 RGBA 并审核。轮廓响应、投影及发光要求需真实资产和作品证明。

见 [GENERATION_PROMPT.md](GENERATION_PROMPT.md) 与 [COSPLAY_POST_STANDARD.md](COSPLAY_POST_STANDARD.md)。
