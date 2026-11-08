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
        '修发与头发生成已开启。结合角色原有发型、本张假发、动作、视线和留白处理头发；'
        '修补露网、缺口、断层与不自然的发束，清理干扰杂发，顺着发根和现有发束补出自然细丝与高光。'
        '只有用户要求飘发、延长或改变发束形态时才生成相应变化；区分清理乱飞的杂发与主动设计飘动发束。'
        '飘发要有连续发根、合理的重量与风向、疏密和遮挡，不把所有发丝画成平行线或光带。'
        '保留角色发色、发际线、刘海识别特征及头饰结构；脸部与头饰是保留区，不让新增头发遮住眼睛或吞掉头饰。'
        '移走旧发束后补全露出的背景或衣物。头发由完整生成画面提供，脸与头饰仍来自原片。'
        f'用户的修发要求：{edits["hair_brief"] or "按本张原片修整瑕疵，保留原有发型和发束形态。"}'
        if edits['hair'] else
        '修发与头发生成未开启：保留原片发型、发束位置与朝向；不要设计新增飘发、换发型或发色。')
    lighting = (
        '人物重打光已开启。可按已选场景重新设计光向，不受原片主光方向约束；'
        '必须把脸、耳、脖颈、裸露皮肤、身体、实物服装、头发、道具与环境作为同一场景受光。'
        '依据可解释的主光源确定方向、软硬、色温、衰减、明暗面、鼻影与下颌阴影；'
        '环境反光、衣料与金属反射、接触阴影和近处地面必须呼应，不能给所有边缘加一圈同亮度轮廓光。'
        '保留脸型、五官位置、妆容、表情、衣物结构与材质，不把光照调整变成美肤、漂白或改脸。'
        '生成的人脸须与原片同位置、同朝向、同几何，为保留原片细节的空间光照迁移提供依据。'
        f'用户的光照要求：{edits["lighting_brief"] or "依据本次摄影方向统一人物与环境的受光，保持自然明暗层次。"}'
        if edits['relight'] else
        '人物重打光未开启：场景主光与原片头部受光相容，保留原片光向和脸部体积；'
        '允许曝光与色温协调，不设计相反的光向，不用整图曲线冒充重打光。')
    return '\n可选制作设置（用户明确开关，优先于旧方向中冲突的描述）：\n'+hair+'\n'+lighting
