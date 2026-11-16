"""The user's two optional production edits, frozen with each real sample."""
from __future__ import annotations


def normalise_edits(value=None):
    value = value or {}
    result = {'hair':value.get('hair',False),'hair_brief':value.get('hair_brief',''),
        'relight':value.get('relight',False),'lighting_brief':value.get('lighting_brief','')}
    if any(type(result[key]) is not bool for key in ('hair','relight')):
        raise ValueError('修发与人物重打光的开关须为布尔值')
    for key in ('hair_brief','lighting_brief'):
        if not isinstance(result[key],str):
            raise ValueError('修发与光照要求须为文本')
        result[key] = result[key].strip()
    return result


def edits_match(first, second):
    first,second = normalise_edits(first),normalise_edits(second)
    return all(first[enabled]==second[enabled] and (not first[enabled] or first[brief]==second[brief])
        for enabled,brief in (('hair','hair_brief'),('relight','lighting_brief')))


def edits_label(value=None):
    edits = normalise_edits(value)
    enabled = [label for key,label in (('hair','修发'),('relight','重打光')) if edits[key]]
    return ' + '.join(enabled) if enabled else '保留原发与光向'


def edits_description(value=None):
    edits = normalise_edits(value)
    hair = edits['hair_brief'] or '修整瑕疵，保留原有发型'
    lighting = edits['lighting_brief'] or '按摄影方向统一人物与环境受光'
    return '\n'.join(('修发：'+hair if edits['hair'] else '修发：关闭，保留原片发型',
        '重打光：'+lighting if edits['relight'] else '重打光：关闭，沿用原片光向'))


def edit_instructions(value=None):
    edits = normalise_edits(value)
    hair = (
        '修发开启：按 edits.hair_brief 处理，留空时修补露网、缺口、断层与杂发，保留原有发型和发束形态。'
        '保留发色、发际线、刘海特征及头饰，头发采用生成结果，脸与实体头饰保留原片。'
        '用户要求飘发、延长或造型时，依据动作、风向和留白设计连续发根、重量与自然遮挡，保持眼睛和头饰可见，补全露出的背景与衣物。'
        if edits['hair'] else
        '修发关闭：保留原片整头，包括发型、发色、发束位置与朝向。')
    lighting = (
        '人物重打光开启：按 edits.lighting_brief 处理，留空时按摄影方向设计；可重新设计主光方向。'
        '统一脸、耳、颈部、身体、服装、头发、道具与环境的光向、软硬、色温、衰减、明暗面、反光与接触阴影。'
        '保留原片脸型、五官、妆容、表情、位置与朝向，为原片细节的空间光照迁移提供同几何参考。'
        if edits['relight'] else
        '人物重打光关闭：场景主光与原片头部受光相容，保留原片光向和脸部体积，可协调曝光与色温。')
    return '\n本次制作设置：\n'+hair+'\n'+lighting
