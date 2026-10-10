"""Persisted status for original/reference colour matching and composition."""
from __future__ import annotations
import hashlib
import json
import math

SCHEMA = 'source_colour_match_status.v1'
REFERENCE_STACK_KIND = 'reference_appearance_stack'
REFERENCE_STACK_STAGES = (
    'reference_colour_match', 'reference_colour_blend',
    'reference_scene_balance', 'edge_donor_extension',
)


def _record(value):
    return value if isinstance(value, dict) else {}


def reference_stack_state(run, plan):
    """Inspect declared frozen-layer bindings, never verify pixels or execution."""
    plan = _record(plan)
    layer = next((row for row in plan.get('layers', [])
                  if isinstance(row, dict) and row.get('kind') == REFERENCE_STACK_KIND), {})
    params = _record(layer.get('params'))
    asset = _record(_record(layer.get('assets')).get('automatic_delta'))
    background = next((_record(row.get('assets')) for row in plan.get('layers', [])
                       if isinstance(row, dict) and row.get('kind') == 'background_design'), {})
    baseline = _record(_record(plan.get('source_baseline')).get('assets'))
    fp = _record(baseline.get('foreground_premultiplied'))
    current_alpha = _record(_record(plan.get('mask_roles')).get('foreground_alpha'))
    alpha = _record(baseline.get('foreground_alpha')) or current_alpha
    expected = {
        'source_sha256': _record(run.get('source')).get('sha256'),
        'reference_sha256': _record(run.get('reference')).get('sha256'),
        'alpha_sha256': alpha.get('sha256'),
        'source_fp_sha256': fp.get('sha256'),
        'canvas_size': _record(plan.get('canvas')).get('source_oriented_size'),
    }
    errors = []
    stages = params.get('prepared_stages')
    stages = stages if isinstance(stages, list) and all(isinstance(x, str) for x in stages) else []
    try:
        strength = float(layer.get('strength', 1))
    except (TypeError, ValueError, OverflowError):
        strength = -1.
    independent_controls = params.get('schema') == 'reference_appearance_stack.v2'
    controls = {}
    for key in ('colour_strength', 'lightness_strength'):
        try:
            controls[key] = float(layer.get(key, 1.))
        except (ValueError, TypeError, OverflowError):
            controls[key] = -1.
    if layer:
        if (_record(plan.get('source_baseline')).get('schema') != 'appearance_source_baseline.v1'
                or _record(plan.get('source_baseline')).get('working_domain') != 'display_linear_sRGB'):
            errors.append('reference_stack_current_baseline_required')
        for key, value in expected.items():
            if not value or params.get(key) != value:
                errors.append('reference_stack_' + key + '_binding_mismatch')
        size = expected['canvas_size']
        if (not isinstance(size, list) or len(size) != 2 or
                not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in size)):
            errors.append('reference_stack_invalid_canvas_size')
        if (expected['source_sha256'] != plan.get('source_sha256') or
                expected['reference_sha256'] != plan.get('candidate_sha256')):
            errors.append('reference_stack_plan_input_binding_mismatch')
        if params.get('schema') != 'reference_appearance_stack.v2':
            errors.append('reference_stack_schema_mismatch')
        if params.get('working_domain') != 'display_linear_sRGB':
            errors.append('reference_stack_working_domain_mismatch')
        if params.get('freeze_mode') != 'native_premultiplied_delta':
            errors.append('reference_stack_freeze_mode_mismatch')
        if not asset.get('path') or not asset.get('sha256'):
            errors.append('reference_stack_delta_not_declared')
        if asset.get('size') is not None and asset['size'] != size:
            errors.append('reference_stack_delta_size_mismatch')
        for stage in REFERENCE_STACK_STAGES:
            if stage not in stages:
                errors.append('reference_stack_stage_not_prepared:' + stage)
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            errors.append('reference_stack_invalid_strength')
        if current_alpha.get('sha256') != alpha.get('sha256') and not _record(
                _record(plan.get('mask_roles')).get('edited_edge_light_applicability')).get('sha256'):
            errors.append('reference_stack_changed_alpha_without_retained_applicability')
        if independent_controls:
            declared_assets = _record(layer.get('assets'))
            if (params.get('control_recipe') != 'premultiplied_oklab_four_corner.v1'
                    or params.get('control_source_delta_sha256') != asset.get('sha256')):
                errors.append('reference_stack_control_recipe_binding_mismatch')
            bound_controls = _record(params.get('control_assets_sha256'))
            for role in ('colour_delta', 'lightness_delta'):
                entry = _record(declared_assets.get(role))
                if (not entry.get('path') or not entry.get('sha256') or
                        bound_controls.get(role) != entry.get('sha256')):
                    errors.append('reference_stack_control_asset_binding_mismatch:' + role)
            receipt = _record(declared_assets.get('control_receipt'))
            if not receipt.get('path') or not receipt.get('sha256'):
                errors.append('reference_stack_control_receipt_not_declared')
            semantic_identity = params.get('control_receipt_identity_sha256')
            if (not isinstance(semantic_identity, str) or len(semantic_identity) != 64
                    or any(char not in '0123456789abcdefABCDEF' for char in semantic_identity)):
                errors.append('reference_stack_invalid_control_receipt_semantic_identity')
            for key, value in controls.items():
                if not math.isfinite(value) or not 0 <= value <= 1:
                    errors.append('reference_stack_invalid_' + key)
    from reference_support import report_state
    support = report_state(params.get('reference_support'),
        {**expected, 'automatic_delta_sha256': asset.get('sha256')})
    errors.extend(support['errors'])
    invalid_strength = not math.isfinite(strength) or not 0 <= strength <= 1
    enabled = bool(plan.get('enabled') and layer.get('enabled') and (strength > 0 or invalid_strength)
                   and (not independent_controls or any(value != 0 for value in controls.values())))
    return {
        'declared': bool(layer), 'prepared': bool(layer) and not errors,
        'binding_valid': bool(layer) and not errors, 'enabled': enabled,
        'strength': strength, 'required_for_execution': enabled,
        **controls, 'independent_controls': independent_controls,
        'control_receipt_identity_mode': 'canonical_json' if independent_controls else None,
        'control_receipt_semantics_verified': False,
        'reconstruction_required': enabled and bool(errors),
        'prepared_stages': stages, 'errors': errors,
        'asset_sha256': asset.get('sha256'),
        'provenance_identity_sha256': params.get('provenance_identity_sha256'),
        'working_domain': params.get('working_domain'),
        'freeze_mode': params.get('freeze_mode'),
        'control_scope': 'independent_colour_lightness_and_overall_strength' if independent_controls else 'unsupported_contract',
        'reference_support': support['report'], 'support_status': support['state'],
        'support_reason': support['reason'], 'support_report_valid': support['valid'],
        'support_stage_statuses': {stage: _record(_record((support['report'] or {}).get('stages')).get(stage)).get(
            'status', 'partial') for stage in REFERENCE_STACK_STAGES},
        'arrays_or_file_hashes_verified': False,
        'automatic_completion_claim': False,
    }


def _prepare_reference_stack(run, plan, state):
    ready = state['prepared']
    message = ('已保存人物融合层；各区域依据与降级状态见兼容性报告。'
               '缺少可靠对应或同材质供体时保留原片，背景和编辑可继续；实际应用与视觉验收分别记录。'
               if ready else '人物与场景融合的来源或准备记录不匹配；保留已保存画面，重新准备后再应用。')
    descriptor = {
        'schema': SCHEMA, 'source_identity': source_identity(run, plan),
        'match_sha256': state['asset_sha256'], 'status': state['support_status'] if ready else 'pending',
        'reference_appearance_stack': state,
        'user_status': {
            'title': '人物与场景融合', 'message': message,
            'available_actions': (['对照原片、参考与结果', '调整整体融合或局部精修', '保存当前版本']
                                  if ready else ['查看已保存画面', '重新准备人物与场景融合']),
            'automatic_completion_claim': False, 'requires_user_full_body_recolour': False,
        },
        'responsibilities': {
            'original_person': {'status': 'ready', 'user_description': '人物结构、妆容轮廓与原生纹理来自原片。'},
            **{stage: {'status': state['support_stage_statuses'][stage] if ready and stage in state['prepared_stages'] else 'pending',
                       'user_description': description} for stage, description in (
                ('reference_colour_match', '原图与参考图仿色。'),
                ('reference_colour_blend', '同材质参考颜色融合，眼口颜色允许适应。'),
                ('reference_scene_balance', '人物与环境的局部明暗、光色协调，不重画人物。'),
                ('edge_donor_extension', '最终人物颜色延展至半透明边缘，保留原片覆盖率。'))},
            'composition': {'status': 'partial', 'user_description': '人物与背景合成的实际效果仍需审片。'},
        },
    }
    descriptor['identity_sha256'] = hashlib.sha256(json.dumps(descriptor, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return descriptor

def source_identity(run, plan):
    background = next((row.get('assets', {}) for row in plan.get('layers', [])
                       if row.get('kind') == 'background_design'), {})
    if any(row.get('kind') == REFERENCE_STACK_KIND for row in plan.get('layers', [])):
        baseline = _record(_record(plan.get('source_baseline')).get('assets'))
        background = {**background, 'foreground_premultiplied':
                      baseline.get('foreground_premultiplied') or background.get('foreground_premultiplied', {})}
    return {'source_sha256': (run.get('source') or {}).get('sha256'),
        'candidate_sha256': (run.get('reference') or {}).get('sha256'),
        'source_fp_sha256': background.get('foreground_premultiplied', {}).get('sha256'),
        'source_alpha_sha256': plan.get('mask_roles', {}).get('foreground_alpha', {}).get('sha256'),
        'source_canvas_size': plan.get('canvas', {}).get('source_oriented_size')}

def prepare(run, plan):
    stack = reference_stack_state(run, plan)
    if stack['declared']:
        return _prepare_reference_stack(run, plan, stack)
    identity = source_identity(run, plan)
    if any(row.get('kind') == 'source_reference_colour_match' for row in plan.get('layers', [])):
        raise ValueError('Retired automatic colour matcher cannot supply current appearance status')
    ready = False
    descriptor = {'schema': SCHEMA, 'source_identity': identity,
        'match_sha256': None, 'status': 'partial' if ready else 'pending',
        'user_status': {'title': '原图与参考图仿色，再溶图',
            'message': '保留原片人物结构与纹理，按参考图匹配颜色后合成背景；当前结果仍需审片。' if ready else '尚未保存完整仿色参数。',
            'available_actions': ['查看原片与参考', '检查仿色及溶图结果', '保存已有编辑'],
            'automatic_completion_claim': False, 'requires_user_full_body_recolour': False},
        'responsibilities': {
            'original_person': {'status': 'ready', 'user_description': '保留原图人物结构和纹理。'},
            'reference_colour_match': {'status': 'partial' if ready else 'pending', 'user_description': '原图与参考图颜色统计匹配。'},
            'composition': {'status': 'partial', 'user_description': '使用原片覆盖率与设计背景合成，融合质量待审片。'}}}
    descriptor['identity_sha256'] = hashlib.sha256(json.dumps(descriptor, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return descriptor

def validate_plan(run, plan):
    descriptor = plan.get('source_appearance_responsibility')
    current = prepare(run, plan)
    valid = descriptor == current
    return {'valid': valid, 'errors': [] if valid else ['colour_match_status_stale']}

def stage_summary(run, plan):
    descriptor = prepare(run, plan)
    result = {'status': descriptor['status'], 'identity_sha256': descriptor['identity_sha256'],
        **descriptor['user_status'], 'responsibilities': [
            {'id': key, 'status': value['status'], 'description': value['user_description']}
            for key, value in descriptor['responsibilities'].items()]}
    if 'reference_appearance_stack' in descriptor:
        result.update(reference_appearance_stack=descriptor['reference_appearance_stack'],
                      binding_valid=descriptor['reference_appearance_stack']['binding_valid'])
    return result
