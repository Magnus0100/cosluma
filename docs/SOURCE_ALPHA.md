# 当前原片前景与覆盖率

本页适用于原片和普通导入参考的源域前景。scoped 生成外观的实际生成覆盖 G 与编辑覆盖 E 不等于人物 alpha，整图生成不套回此源人物蒙版。

`source_alpha.py` 用 BiRefNet 生成轮廓种子，Sapiens2 segmentation 0.4B 提供语义。原生分块 ViTMatte 细化透明度，PyMatting 求解成对前景/旧背景。语义和场景实例不是发丝 alpha，缺模型/身份不符直接失败，不用软椭圆。

`source_foreground.py` 与包装器保存预乘前景、覆盖率、旧背景及诊断。RAW 完成批准源修补与明确显像，最终进入 `display_linear_sRGB`，保持源基线和实际绑定。

工程拥有成对前景/覆盖率/基线。`native_matte.py` 保存笔画；改覆盖率同步更新前景背景和真实接触依赖。seed、标签或覆盖率不能重复当作人物透明度。

缓存只复用当前模型/算法/源/编辑身份，不升级旧 profile/trust/别名。仿色作用人物前景，接触先作用合格背景，最终人物 alpha 一次。

发丝、半透明边、原背景残留和鞋底仍需原尺寸检查，覆盖率存在不授予视觉通过。当前数据流见 [PRODUCT_ARCHITECTURE.md](PRODUCT_ARCHITECTURE.md)，范围与视觉门见 [COSPLAY_POST_STANDARD.md](COSPLAY_POST_STANDARD.md)。
