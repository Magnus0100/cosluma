"""Read-only product progress for the native source-design workflow.

The reducer consumes persisted facts and file-presence observations.  It does
not run workers, validate image quality, hash large assets, or update a run.
Successful model execution and stored exports never imply a visual Go.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


STAGE_STATUS_LABELS = {
    "complete": "已完成", "partial": "部分完成", "unsupported": "尚未支持",
    "failed": "需恢复", "pending": "待处理", "not_requested": "本次无需",
}
VISUAL_CHECK_KEYS = (
    "identity", "face_eyes", "hands_fingers", "hair_halo", "outline_light",
    "garment_texture", "background_light_effects", "overall_impression",
)
COSPLAY_CHECK_LABELS = {
    'identity_makeup':'本人和妆造', 'concept_hierarchy':'角色与画面重点',
    'effects_space':'特效空间', 'skin_materials':'肤色与材质',
    'native_detail':'原生细节', 'scene_integration':'人物与场景融合',
}
RESULT_STATE_LABELS = {
    'awaiting_input': '等待输入', 'needs_local_recovery': '需要本地恢复',
    'pending_save': '当前修改尚未保存', 'needs_repair': '成片有待修项',
    'needs_evidence': '成片证据待补', 'needs_visual_review': '等待人工审片批准',
    'accepted': '已人工批准',
}
SUBJECT_DIAGNOSTIC_LABELS = {
    'support_unmeasurable': '独立归属支持不足，尚不能测量人物纹理损失。',
    'detail_preserved_diagnostic_conflict': '所选可测频带的结构响应仍被保留，与原审片观感存在冲突；其它频带和未测区域仍未知。',
    'structured_loss_confirmed': '原片可支持的结构纹理损失已由数值诊断确认。',
    'measurement_inconclusive': '当前测量不足以判断结构纹理是否损失。',
    'recovery_rejected': '本次恢复未获准保存，父版本和原审片失败项仍保留。',
}
SUBJECT_DIAGNOSTIC_ACTIONS = {
    'support_unmeasurable': '扩大独立匹配的原尺寸证据，并改善原片材质归属支持。',
    'detail_preserved_diagnostic_conflict': '复核同尺寸局部诊断，区分纹理损失与缺少修饰；原失败项保留，不重复已占用的自动尝试。',
    'structured_loss_confirmed': '符合恢复资格时保存独立子版本，并为新像素取得整体、上下文和原尺寸新审片。',
    'measurement_inconclusive': '补充原片结构、噪声与频带诊断，确认损失后再恢复。',
    'recovery_rejected': '查看拒绝原因，补齐独立支持或修正造成损失的模块后再评估新版本。',
}


def cosplay_review_state(run, *, current_render):
    """Display current or proven-copy findings without granting repair entry.

    Copy receipts explain historical findings; only the coordinator can verify
    all current pixel bindings and grant a new repair or network budget.
    """
    review = _record(run.get('cosplay_post_review'))
    binding = _record(review.get('run_binding'))
    inputs_match = bool(binding.get('source_sha256')
        and binding.get('source_sha256') == _record(run.get('source')).get('sha256')
        and binding.get('reference_sha256')
        and binding.get('reference_sha256') == _record(run.get('reference')).get('sha256'))
    matches = bool(current_render and review.get('status') == 'reviewed'
        and review.get('actual_api_evidence_valid') is True and inputs_match
        and binding.get('run_id') == run.get('run_id')
        and binding.get('revision') == run.get('revision'))
    copy = _record(run.get('saved_render_copy'))
    render_copy = _record(_record(run.get('render')).get('saved_render_copy'))
    copied_hashes = _record(copy.get('copied_master_sha256'))
    reviewed_hashes = _record(binding.get('render_artifacts'))
    inherited = bool(current_render and not matches and inputs_match
        and review.get('status') == 'reviewed' and review.get('actual_api_evidence_valid') is True
        and copy.get('schema') == 'saved_render_copy.v1' and copy == render_copy
        and copy.get('copied_from_manifest_sha256')
        and copy.get('copied_from_run_id') == binding.get('run_id')
        and copy.get('rendered_run_id') == binding.get('run_id')
        and copy.get('render_executed') is False
        and all(copied_hashes.get(key) and copied_hashes[key] == reviewed_hashes.get(key)
                for key in ('auto_draft', 'render_master', 'render_scene_master'))
        and binding.get('final_sha256') == copied_hashes.get('auto_draft'))
    binding_state = ('current' if matches else 'inherited_same_pixels' if inherited else
                     'pending' if review.get('status') == 'pending' else 'stale' if review else 'unavailable')
    rows = []
    if matches or inherited:
        for key, label in COSPLAY_CHECK_LABELS.items():
            check = _record(_record(_record(review.get('checks')).get(key)).get('final'))
            rows.append({'id':key,'label':label,'status':check.get('status','unknown'),
                'reason':check.get('reason',''),'action':check.get('action',''),
                'correction_stage':check.get('correction_stage'), 'cause':check.get('cause','unresolved'),
                'target':'final', 'evidence_scope':'whole_and_cited_native_windows',
                'binding_state':binding_state})
    return {'status':'reviewed' if matches else 'inherited' if inherited else 'pending' if review.get('status')=='pending'
            else 'unavailable' if review.get('status')=='unavailable' else 'stale' if review else 'not_requested',
            'current_render_binding_matches':matches,
            'checks':rows,'diagnosis_grants_visual_acceptance':False,
            'binding_state':binding_state, 'producer_run_id':binding.get('run_id'),
            'current_repair_authorized':False, 'repair_requires_current_full_qa':True,
            'binding_basis':'saved_receipt_identity_only; report_does_not_rehash_pixels',
            'native_sampling_coverage':_record(review.get('native_sampling_coverage')),
            'error':review.get('error') if review.get('status')=='unavailable' else None}


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _export_receipt_state(run: Mapping[str, Any], observations) -> dict[str, Any]:
    """Pure reduction of current-workspace receipts observed by the reader."""
    from artifact_receipts import binding_for_run
    rows = _record(observations.get('artifact_receipts')).get('exports') or []
    valid: list[dict[str, Any]] = []
    formats: dict[str, dict[str, Any]] = {}
    depths = {'PNG': 8, 'JPEG': 8, 'TIFF': 8, 'TIFF16': 16, 'TIFF32_LINEAR': 32}
    for row in rows:
        if (not isinstance(row, dict) or row.get('schema_version') != 'source_export_receipt.v2'
                or row.get('status') != 'succeeded' or row.get('binding') != binding_for_run(run)
                or row.get('files_unchanged') is not True or row.get('metadata_matches') is not True):
            continue
        fmt = row.get('format')
        hashes = [run.get('source', {}).get('sha256'), _record(row.get('saved_render')).get('sha256'),
                  _record(row.get('export')).get('sha256')]
        if (fmt not in depths or row.get('actual_bit_depth') != depths[fmt]
                or any(not isinstance(value, str) or len(value) != 64
                       or any(c not in '0123456789abcdef' for c in value) for value in hashes)):
            continue
        receipt = dict(row)
        valid.append(receipt)
        formats[fmt] = receipt
    return {
        'valid': valid,
        'formats': sorted(formats),
        'complete': bool(valid),
        'verification_basis': 'exporter_checksums_and_unchanged_file_stats; no_refresh_pixel_hashing',
    }


def _layers(value: Any) -> dict[str, dict[str, Any]]:
    return {str(item.get("kind")): item for item in value
            if isinstance(item, dict) and item.get("kind")} if isinstance(value, list) else {}


def portrait_beauty_reference_fit_state(run, observation=None):
    """Compare saved fit inputs without granting or refusing frozen replay.

    The identity observation reads only the small Beauty result JSON. Pixel
    integrity remains the renderer's responsibility; this report does not
    hash assets, run models, or automatically refit after a control edit.
    """
    observation = _record(observation)
    appearance = _record(_record(run.get('options')).get('appearance_revision'))
    control = _record(appearance.get('portrait_beauty'))
    plan = _record(appearance.get('design_reconstruction_plan'))
    layer = _layers(plan.get('layers')).get('reference_appearance_stack', {})
    params = _record(layer.get('params'))
    recorded = bool(layer) and 'fit_beauty_identity' in params
    observed = not control or observation.get('identity_observed') is True
    current = observation.get('current_identity') if control else None
    if control and not isinstance(current, dict):
        observed = False
    fit = params.get('fit_beauty_identity')
    matches = (fit == current) if layer and recorded and observed else None
    state = ('not_requested' if not layer else 'not_recorded' if not recorded else
             'unavailable' if not observed else 'matched' if matches else 'changed')
    reprepare = (None if state == 'unavailable' else
                 state in {'changed', 'not_recorded'})
    detail = {
        'not_requested': '当前没有参考仿色融合层，无需重新准备参考匹配。',
        'not_recorded': '参考融合层未记录美化拟合来源；建议用当前美化重新准备参考匹配。',
        'unavailable': '当前美化身份暂不可读取；恢复美化资产后再核对参考匹配来源。',
        'matched': '参考仿色融合使用的美化来源与当前设置一致。',
        'changed': '当前美化设置已改变；可继续重放已有融合层，建议重新准备参考匹配以适应新外观。',
    }[state]
    strength = (current.get('strength') if observed and control else
                control.get('strength', 1.) if control else None)
    disabled = (current.get('disabled_faces', []) if observed and control else
                control.get('disabled_faces') or [])
    return {
        'state': state, 'reference_layer_declared': bool(layer),
        'fit_identity_recorded': recorded, 'current_identity_observed': observed,
        'fit_dependency_matches': matches,
        'current_strength': strength,
        'disabled_faces': list(disabled) if isinstance(disabled, (list, tuple)) else [],
        'fit_beauty_identity': fit, 'current_beauty_identity': current,
        'reference_reprepare_recommended': reprepare, 'detail': detail,
        'error': observation.get('error'),
        'frozen_replay_blocked_by_fit_dependency': False,
        'automatically_refits': False, 'model_inferences': 0,
        'assets_rehashed': False, 'quality_acceptance_inferred': False,
        'basis': 'saved_fit_identity_and_current_beauty_json_metadata',
    }


def _required_effect_state(run, diagnosis, *, source_only, visual_approved, fx_status):
    design = _record(run.get('generation_design'))
    brief = _record(design.get('design_brief')) or design
    primary = _record(brief.get('primary_effect_plan'))
    constraints = _record(brief.get('user_constraints'))
    required = (False if source_only or constraints.get('no_effects') or constraints.get('no_effects_quote')
                or primary.get('mode') == 'none' else True if primary.get('mode') == 'scene_effect' else None)
    direction = _record(run.get('creative_direction') or _record(run.get('options')).get('creative_direction'))
    main_effect = _record(direction.get('main_effect'))
    if direction.get('schema_version') == 'cosplay_creative_direction.v1':
        required = (False if source_only or main_effect.get('mode') == 'none' else
                    True if main_effect.get('mode') == 'user_described' else None)
        primary = {'description':main_effect.get('description','')}
    final = next((row for row in diagnosis['checks'] if row['id'] == 'effects_space'), {})
    final_status = final.get('status', 'unknown')
    current = diagnosis['current_render_binding_matches']
    fulfillment = ('not_required' if required is False else 'fulfilled' if visual_approved
        or (required is True and current and final_status == 'pass') else 'failed'
        if required is True and current and final_status == 'fail' else 'unknown')
    return {'required':required, 'description':primary.get('description',''),
        'owner':'scene_or_declared_increment' if required else 'not_required' if required is False else 'unresolved',
        'fulfillment':fulfillment, 'final_visual_status':final_status,
        'binding_state':diagnosis['binding_state'], 'independent_effect_status':fx_status,
        'reason':('本次无需场景主效。' if required is False else
                  '已保存版本的人工批准或当前最终审片已确认主效。' if fulfillment == 'fulfilled' else
                  final.get('reason') or '主效仍需最终成片证据；独立效果跳过不能证明主效完成。'),
        'skipped_grants_fulfillment':False}


def native_receiver_repair_state(run, diagnosis):
    """Describe a candidate and prior outcome; never grant pixel authority.

    The shared planner classifies stored QA. Actual saved boards, pixel hashes,
    joint floor support and native alphas are validated by the producer later.
    """
    from cosplay_repair_plan import plan_final_repairs

    classified = plan_final_repairs(run,
        current_review=diagnosis['current_render_binding_matches'])
    tasks = [row for row in classified['tasks'] if row.get('correction_stage') == 'native_retouch']
    candidates = [row for row in tasks if row.get('producer') == 'native_receiver_detail']
    relevant = [row for row in tasks if row.get('criterion') in {'native_detail', 'scene_integration'}]
    state = ('requires_producer_validation' if candidates else
             'needs_current_pixel_bound_diagnosis' if relevant and not classified['current_review'] else
             'needs_scoped_evidence' if any(row.get('status') == 'unknown' for row in relevant) else
             'needs_source_owned_retouch_qualification' if relevant else 'not_requested')
    reasons = {
        'requires_producer_validation': '鞋部匹配审片记录了细节损失，局部身份和材质观察通过；可进入接收面原尺寸恢复资格检查。生产者仍须验证当前像素、共同地面支持、双份覆盖率和采样有效性。',
        'needs_current_pixel_bound_diagnosis': '旧审片或未绑定证据不能授予接收面修复；先取得当前已保存版本的完整匹配审片。',
        'needs_scoped_evidence': '原生细节仍未知，先补充明确的匹配窗口证据；未知不能授权接收面像素修复。',
        'needs_source_owned_retouch_qualification': '尚无合格的鞋部细节失败证据；该恢复只处理明确的鞋部匹配窗口，并要求局部身份和材质观察通过。其它原生精修问题继续保留。',
        'not_requested': '当前没有请求接收面原尺寸恢复。',
    }
    attempt = _record(run.get('automatic_receiver_detail_completion'))
    attempt_matches = bool(attempt.get('schema') == 'automatic_receiver_detail_completion.v1'
        and run.get('run_id') in {attempt.get('initial_run_id'), attempt.get('result_run_id')})
    outcome = ({key: attempt.get(key) for key in ('status', 'reason', 'result_run_id', 'qualification')}
               if attempt.get('schema') == 'automatic_receiver_detail_completion.v1' else {})
    outcome_source = 'automatic_receiver_detail_completion' if outcome else None
    orchestration = _record(run.get('system_post_completion'))
    for stage in reversed(orchestration.get('stages') or []):
        if not isinstance(stage, dict) or stage.get('id') != 'final_local_repair':
            continue
        latest = next((row for row in reversed(stage.get('outcomes') or [])
                       if isinstance(row, dict) and row.get('producer') == 'native_receiver_detail'), None)
        if latest is not None:
            outcome = {key: latest.get(key) for key in ('status', 'reason', 'input_run_id', 'result_run_id')}
            if attempt_matches:
                outcome['qualification'] = attempt.get('qualification')
            attempt_matches = run.get('run_id') in {latest.get('input_run_id'), latest.get('result_run_id')}
            outcome_source = 'system_post_completion.final_local_repair.outcomes'
            break
    if outcome:
        outcome['applies_to_run'] = attempt_matches
    outcome_labels = {
        'receiver_detail_child_saved': '接收面恢复已保存独立子版本；实际细节与融合须由新审片确认。',
        'receiver_evidence_unsupported': '接收面证据或支持不足，保留原草稿和失败项。',
        'receiver_detail_rejected': '本次接收面恢复未获准保存，保留原草稿和失败项。',
        'already_attempted': '已有接收面恢复尝试，不自动重复。',
    }
    outcome_detail = (outcome_labels.get(outcome.get('status'), '已有接收面恢复记录，查看实际结果。')
                      if outcome else '')
    if outcome and not attempt_matches:
        outcome_detail = '其它版本的接收面恢复记录不能证明当前版本已执行。'
    plan = _record(_record(_record(run.get('options')).get('appearance_revision')).get('design_reconstruction_plan'))
    prepared = _record(_record(plan.get('provenance')).get('native_receiver_detail'))
    background = next((_record(row.get('assets')) for row in plan.get('layers', [])
                       if isinstance(row, dict) and row.get('kind') == 'background_design'), {})
    asset = _record(background.get('native_receiver_detail'))
    declared = bool(prepared.get('state') == 'prepared' and prepared.get('receipt_sha256')
                    and prepared['receipt_sha256'] == asset.get('sha256') and asset.get('path'))
    return {'producer': 'native_receiver_detail' if candidates else None,
        'capability': 'native_receiver_detail', 'qualification_state': state,
        'reason': reasons[state], 'finding_ids': [row['criterion'] for row in relevant],
        'candidate_finding_ids': [row['criterion'] for row in candidates],
        'qualification_pair_ids': sorted({pair for row in candidates
            for pair in row.get('qualification_pair_ids', [])}),
        'candidate_from_saved_review': bool(candidates), 'last_outcome': outcome,
        'last_outcome_source': outcome_source,
        'last_outcome_detail': outcome_detail, 'prepared_asset_declared': declared,
        'prepared_asset_state': prepared.get('state'),
        'basis': 'saved_review_classification_and_producer_receipt_only',
        'requires_actual_receiver_asset_validation': True, 'pixel_repair_authorized': False,
        'assets_rehashed': False, 'visual_acceptance': False}


def native_subject_repair_state(run, diagnosis):
    """Show stored subject evidence and outcomes without authorizing a repair."""
    from cosplay_repair_plan import plan_final_repairs
    classified = plan_final_repairs(run,
        current_review=diagnosis['current_render_binding_matches'])
    relevant = [row for row in classified['tasks'] if row.get('criterion') == 'native_detail'
                and row.get('correction_stage') == 'native_retouch']
    candidates = [row for row in relevant if row.get('producer') == 'native_subject_detail']
    state = ('requires_producer_validation' if candidates else
             'needs_current_pixel_bound_diagnosis' if relevant and not classified['current_review'] else
             'needs_scoped_evidence' if any(row.get('status') == 'unknown' for row in relevant) else
             'needs_source_owned_retouch_qualification' if relevant else 'not_requested')
    reasons = {
        'requires_producer_validation': '人物匹配窗口记录了细节损失，可检查原片细节恢复资格；须独立验证材质归属、批准后的原片纹理、噪声支持和实际损失。',
        'needs_current_pixel_bound_diagnosis': '人物细节修复须取得当前已保存版本的完整匹配审片。',
        'needs_scoped_evidence': '人物原生细节仍未知；未知不能授权像素修复。',
        'needs_source_owned_retouch_qualification': '尚无合格的人物细节窗口，已有失败项继续保留。',
        'not_requested': '当前没有请求人物原片细节恢复。',
    }
    attempt = _record(run.get('automatic_subject_detail_completion'))
    valid_record = attempt.get('schema') == 'automatic_subject_detail_completion.v1'
    matches = valid_record and run.get('run_id') in {attempt.get('initial_run_id'), attempt.get('result_run_id')}
    outcome = ({key: attempt.get(key) for key in ('status', 'reason', 'result_run_id', 'qualification',
                                                'diagnostic_summary', 'input_binding', 'diagnosis')}
               if valid_record else {})
    origin = 'automatic_subject_detail_completion' if outcome else None
    for stage in reversed(_record(run.get('system_post_completion')).get('stages') or []):
        if not isinstance(stage, dict) or stage.get('id') != 'final_local_repair':
            continue
        latest = next((row for row in reversed(stage.get('outcomes') or [])
                       if isinstance(row, dict) and row.get('producer') == 'native_subject_detail'), None)
        if latest is None:
            continue
        outcome = {key: latest.get(key) for key in ('status', 'reason', 'input_run_id', 'result_run_id',
                                                  'diagnostic_summary', 'input_binding', 'diagnosis')}
        latest_matches = run.get('run_id') in {latest.get('input_run_id'), latest.get('result_run_id')}
        if matches and latest_matches:
            outcome.update({key: attempt.get(key) for key in
                ('qualification', 'diagnostic_summary', 'input_binding', 'diagnosis')})
        matches = latest_matches
        origin = 'system_post_completion.final_local_repair.outcomes'
        break
    labels = {
        'subject_detail_child_saved': '人物原片细节已保存独立子版本；新像素须重新审片。',
        'subject_evidence_unsupported': '人物证据或归属支持不足，保留父版本和失败项。',
        'subject_detail_rejected': '未确认可安全改善的纹理损失，保留父版本和失败项。',
        'already_attempted': '已有人物细节恢复尝试，不自动重复。',
    }
    detail = labels.get(outcome.get('status'), '已有人物细节恢复记录。') if outcome else ''
    saved_child_declared = bool(valid_record and attempt.get('status') == 'subject_detail_child_saved'
        and attempt.get('child_native_artifacts_owned') is True
        and attempt.get('result_run_id') == run.get('run_id')
        and attempt.get('candidate_run_id') == run.get('run_id')
        and attempt.get('initial_run_id') == run.get('parent_run_id')
        and run.get('run_id') != attempt.get('initial_run_id')
        and _record(run.get('continuation_edit')).get('render_state') == 'saved'
        and _record(run.get('continuation_edit')).get('rendered_run_id') == run.get('run_id')
        and _record(run.get('continuation_edit')).get('rendered_revision') == run.get('revision'))
    if outcome.get('status') == 'subject_detail_child_saved' and not saved_child_declared:
        detail = '已有子版本保存记录；当前版本尚未声明完整的恢复像素归属，不能据此认为恢复已应用。'
    if outcome:
        outcome['applies_to_run'] = bool(matches)
        if not matches:
            detail = '其它版本的人物细节记录不能证明当前版本已执行。'
    summary = _record(outcome.get('diagnostic_summary'))
    input_binding = _record(outcome.get('input_binding'))
    measured_diagnosis = _record(outcome.get('diagnosis'))
    current_binding = _record(_record(run.get('cosplay_post_review')).get('run_binding'))
    # These are persisted identities only. Foreign, old-pixel or incomplete
    # records remain historical and cannot describe the current endpoint.
    summary_current = bool(matches and diagnosis['current_render_binding_matches']
        and summary.get('status') in SUBJECT_DIAGNOSTIC_LABELS
        and summary.get('visual_qa_unchanged') is True
        and summary.get('visual_acceptance', False) is False
        and measured_diagnosis.get('actual_saved_evidence_verified') is True
        and _record(measured_diagnosis.get('evidence')).get('path')
        and _record(measured_diagnosis.get('evidence')).get('sha256')
        and measured_diagnosis.get('diagnostic_producer_run_id') == input_binding.get('run_id')
        and measured_diagnosis.get('consumer_run_id') == input_binding.get('run_id')
        and measured_diagnosis.get('final_sha256') == input_binding.get('final_sha256')
        and input_binding.get('run_id') == run.get('run_id')
        and input_binding.get('revision') == run.get('revision')
        and input_binding.get('source_sha256') == _record(run.get('source')).get('sha256')
        and input_binding.get('reference_sha256') == _record(run.get('reference')).get('sha256')
        and input_binding.get('final_sha256')
        and input_binding.get('final_sha256') == current_binding.get('final_sha256'))
    summary_label = SUBJECT_DIAGNOSTIC_LABELS.get(summary.get('status'), '')
    summary_action = SUBJECT_DIAGNOSTIC_ACTIONS.get(summary.get('status'), '')
    if summary:
        if not summary_current:
            summary_label = '历史人物细节诊断仅供查看；其绑定未证明当前版本的数值状态。'
            summary_action = '核对当前版本与原诊断出处，再补充当前原尺寸诊断；原失败项保持。'
        outcome.update(diagnostic_summary_current_binding_matches=summary_current,
                       diagnostic_summary_label=summary_label,
                       diagnostic_next_action=summary_action,
                       numerical_gate_grants_visual_acceptance=False)
    plan = _record(_record(_record(run.get('options')).get('appearance_revision')).get('design_reconstruction_plan'))
    prepared = _record(_record(plan.get('provenance')).get('native_subject_detail'))
    asset = next((_record(_record(row.get('assets')).get('qualification_receipt'))
                  for row in plan.get('layers') or [] if isinstance(row, dict)
                  and row.get('kind') == 'source_subject_detail'), {})
    declared = bool(prepared.get('state') == 'prepared' and asset.get('sha256') and asset.get('path'))
    return {'producer': 'native_subject_detail' if candidates else None,
        'capability': 'native_subject_detail', 'qualification_state': state,
        'reason': reasons[state], 'finding_ids': [row['criterion'] for row in relevant],
        'candidate_finding_ids': [row['criterion'] for row in candidates],
        'qualification_pair_ids': sorted({pair for row in candidates for pair in row.get('qualification_pair_ids', [])}),
        'candidate_from_saved_review': bool(candidates), 'last_outcome': outcome,
        'last_outcome_source': origin, 'last_outcome_detail': detail,
        'diagnostic_summary': summary, 'diagnostic_summary_current_binding_matches': summary_current,
        'diagnostic_summary_label': summary_label, 'diagnostic_next_action': summary_action,
        'numerical_gate_grants_visual_acceptance': False,
        'saved_child_declared': saved_child_declared,
        'prepared_asset_declared': declared, 'prepared_asset_state': prepared.get('state'),
        'prepared_asset_declaration_basis': 'layer_qualification_receipt_path_and_sha256_only',
        'basis': 'saved_review_classification_and_producer_receipt_only',
        'requires_actual_subject_asset_validation': True, 'pixel_repair_authorized': False,
        'assets_rehashed': False, 'visual_acceptance': False}


def subject_environment_repair_state(run, diagnosis):
    """Report a supported subject response candidate without granting pixels."""
    from cosplay_repair_plan import plan_final_repairs
    classified = plan_final_repairs(run, current_review=diagnosis['current_render_binding_matches'])
    relevant = [row for row in classified['tasks'] if row.get('criterion') == 'scene_integration'
        and row.get('correction_stage') == 'scene_composition' and row.get('cause') == 'retouch_missing']
    candidates = [row for row in relevant if row.get('producer') == 'subject_environment_response']
    state = ('requires_producer_validation' if candidates else
        'needs_current_pixel_bound_diagnosis' if relevant and not classified['current_review'] else
        'needs_scoped_evidence' if relevant else 'not_requested')
    reasons = {
        'requires_producer_validation': '人物与环境的局部明暗或光色响应尚未衔接；可自动检查头发和衣料的独立匹配支持，再准备保留原片细节的局部光色修正。',
        'needs_current_pixel_bound_diagnosis': '旧审片不能授权人物环境响应修复；需完整绑定当前已保存像素的实际审片证据。',
        'needs_scoped_evidence': '人物环境响应证据尚不足，先核对匹配参考、源材质和当前原尺寸画面。',
        'not_requested': '当前没有请求人物与环境光色协调。',
    }
    attempt = _record(run.get('automatic_subject_environment_completion'))
    valid = attempt.get('schema') == 'automatic_subject_environment_completion.v1'
    matches = bool(valid and run.get('run_id') in {attempt.get('initial_run_id'), attempt.get('result_run_id')})
    outcome = ({key: attempt.get(key) for key in ('status', 'reason', 'input_binding', 'repair_receipt',
        'result_run_id', 'changed_pixels', 'requires_new_visual_QA')} if valid else {})
    origin = 'automatic_subject_environment_completion' if outcome else None
    for stage in reversed(_record(run.get('system_post_completion')).get('stages') or []):
        if not isinstance(stage, dict) or stage.get('id') != 'final_local_repair':
            continue
        latest = next((row for row in reversed(stage.get('outcomes') or []) if isinstance(row, dict)
            and row.get('producer') == 'subject_environment_response'), None)
        if latest is None:
            continue
        outcome = {key: latest.get(key) for key in ('status', 'reason', 'input_run_id', 'result_run_id')}
        latest_matches = run.get('run_id') in {latest.get('input_run_id'), latest.get('result_run_id')}
        if matches and latest_matches:
            outcome.update({key: attempt.get(key) for key in ('input_binding', 'repair_receipt',
                'changed_pixels', 'requires_new_visual_QA')})
        matches = latest_matches; origin = 'system_post_completion.final_local_repair.outcomes'
        break
    labels = {
        'subject_environment_child_saved': '人物与环境光色协调已保存独立子版本；数值改善须由新整体和原尺寸审片确认。',
        'subject_environment_evidence_unsupported': '人物局部对应或归属支持不足，保留原版本与审片失败项。',
        'subject_environment_rejected': '本次人物环境响应未达到安全改善要求，保留父版本与失败项。',
        'subject_environment_cancelled': '人物与环境光色协调已取消，原版本与证据保留。',
        'already_attempted': '已有人物环境响应尝试，不自动重复。',
    }
    detail = labels.get(outcome.get('status'), '已有人物环境响应记录。') if outcome else ''
    continuation = _record(run.get('continuation_edit'))
    child_declared = bool(valid and attempt.get('status') == 'subject_environment_child_saved'
        and attempt.get('child_native_artifacts_owned') is True
        and attempt.get('candidate_run_id') == run.get('run_id') == attempt.get('result_run_id')
        and attempt.get('initial_run_id') == run.get('parent_run_id')
        and run.get('run_id') != attempt.get('initial_run_id')
        and continuation.get('render_state') == 'saved'
        and continuation.get('rendered_run_id') == run.get('run_id')
        and continuation.get('rendered_revision') == run.get('revision'))
    if outcome:
        outcome['applies_to_run'] = matches
        if not matches:
            detail = '其它版本的人物环境响应记录不能证明当前版本已应用。'
        elif outcome.get('status') == 'subject_environment_child_saved' and not child_declared:
            detail = '已有协调子版本记录；当前版本尚未声明完整的恢复像素归属，不能据此认为已应用。'
    plan = _record(_record(_record(run.get('options')).get('appearance_revision')).get('design_reconstruction_plan'))
    layer = next((row for row in plan.get('layers') or [] if isinstance(row, dict)
        and row.get('kind') == 'source_environment_response'), {})
    asset = _record(_record(layer.get('assets')).get('qualification_receipt'))
    return {'producer': 'subject_environment_response' if candidates else None,
        'capability': 'subject_environment_response', 'qualification_state': state, 'reason': reasons[state],
        'finding_ids': [row['criterion'] for row in relevant],
        'candidate_finding_ids': [row['criterion'] for row in candidates],
        'candidate_from_saved_review': bool(candidates), 'last_outcome': outcome,
        'last_outcome_source': origin, 'last_outcome_detail': detail, 'saved_child_declared': child_declared,
        'prepared_asset_declared': bool(asset.get('path') and asset.get('sha256')),
        'scope': 'independently_supported_source_hair_and_clothing_only',
        'requires_actual_subject_asset_validation': True, 'pixel_repair_authorized': False,
        'new_pixels_require_new_visual_QA': True, 'automatic_image_budget_added': 0,
        'assets_rehashed': False, 'visual_acceptance': False}


def _product_result(run, diagnosis, *, draft_state, current_render, source_only,
                    visual_approved, assets, reconstruction_required, required_effects):
    """Reduce display-only findings and whitelisted existing recovery actions."""
    findings = []
    for row in diagnosis['checks']:
        if row['status'] not in {'fail','unknown','unavailable'}:
            continue
        cause = row.get('cause') or 'unresolved'
        stage = row.get('correction_stage')
        # A current diagnosis owns its explicit supported stage; broad cause
        # hints cannot redirect source retouch or material failures to scene.
        owner = (stage if row.get('binding_state') == 'current' and stage in (
            'reference_generation','subject_appearance','scene_composition',
            'native_retouch','unsupported_layering') else
            'scene_composition' if stage == 'scene_composition' or cause in
            {'transfer_incomplete','composition_defect'} else stage or 'evidence_review')
        findings.append({'criterion':row['id'], 'label':row['label'], 'target':'final',
            'status':row['status'], 'cause':cause, 'owner':owner, 'reason':row['reason'],
            'action':row['action'], 'evidence_scope':row['evidence_scope'],
            'binding_state':row['binding_state']})
    receiver_repair = native_receiver_repair_state(run, diagnosis)
    subject_repair = native_subject_repair_state(run, diagnosis)
    environment_repair = subject_environment_repair_state(run, diagnosis)
    for finding in findings:
        if finding['criterion'] in receiver_repair['finding_ids']:
            finding['local_repair'] = receiver_repair
        if (finding['criterion'] in subject_repair['finding_ids']
                and (subject_repair['candidate_from_saved_review'] or subject_repair['last_outcome'])):
            finding['local_repair'] = subject_repair
            if subject_repair['diagnostic_summary_current_binding_matches']:
                finding['original_review_action'] = finding['action']
                finding['action'] = subject_repair['diagnostic_next_action']
        if finding['criterion'] in environment_repair['finding_ids']:
            finding['local_repair'] = environment_repair
    coverage = diagnosis['native_sampling_coverage']
    missing = coverage.get('unobserved_pairs') or []
    unsupported = coverage.get('unsupported_regions') or []
    if missing or unsupported:
        findings.append({'criterion':'sampling_coverage','label':'原尺寸证据覆盖','target':'native',
            'status':'unknown','cause':'unobserved_source_regions','owner':'evidence_review',
            'reason':'部分原片区域尚无可判定证据。','action':'查看原尺寸证据与未覆盖区域。',
            'evidence_scope':{'unobserved_pairs':missing,'unsupported_regions':unsupported},
            'binding_state':diagnosis['binding_state']})
    if current_render and not source_only and not diagnosis['current_render_binding_matches']:
        findings.append({'criterion':'current_review','label':'当前版本审片','target':'final',
            'status':'unknown','cause':'current_full_qa_required','owner':'evidence_review',
            'reason':('此处展示同像素旧审片；执行修复前仍需当前完整绑定的新审片。'
                if diagnosis['binding_state']=='inherited_same_pixels' else '当前版本没有有效的成片审片。'),
            'action':'先查看证据并核对当前版本。','evidence_scope':'current_full_pixel_binding',
            'binding_state':diagnosis['binding_state']})
    if required_effects['required'] is not False and required_effects['fulfillment']=='unknown' and not any(
            row['criterion']=='effects_space' for row in findings):
        findings.append({'criterion':'effects_space','label':'特效空间','target':'final','status':'unknown',
            'cause':'effect_fulfillment_unverified','owner':'evidence_review',
            'reason':required_effects['reason'],'action':'查看主效对照并确认最终空间效果。',
            'evidence_scope':'final_primary_effect','binding_state':diagnosis['binding_state']})
    if draft_state in {'awaiting_source','awaiting_reference'}:
        state='awaiting_input'
    elif draft_state in {'unsupported_route','needs_recovery','awaiting_reconstruction','needs_local_rebuild'}:
        state='needs_local_recovery'
    elif draft_state in {'pending_save','awaiting_fusion_save'}:
        state='pending_save'
    elif visual_approved:
        state='accepted'
    elif any(row['status']=='fail' and row['binding_state']=='current' for row in findings):
        state='needs_repair'
    elif findings:
        state='needs_evidence'
    else:
        state='needs_visual_review'
    actions=[]
    def action(action_id,label,supported,reason='',ids=(),qa=False,generation=False):
        actions.append({'action_id':action_id,'label':label,'supported':bool(supported),
            'reason':reason,'generation_required':generation,'QA_required':qa,'finding_ids':list(ids)})
    if state=='needs_local_recovery' and assets.get('source') and assets.get('reference'):
        regional_budget = _record(run.get('generation_budget'))
        resume_region = (regional_budget.get('regional_refinement_outcome') == 'failed_no_submission'
            and regional_budget.get('regional_refinement_image_budget') == 1
            and regional_budget.get('regional_refinement_image_submissions') == 0)
        reuse_candidate = (regional_budget.get('regional_refinement_outcome') == 'generated_rejected'
            and regional_budget.get('regional_refinement_image_submissions') == 1)
        action('retry_local_processing',
               '复用已生成局部结果继续合成（不生图）' if reuse_candidate else
               '复用参考并恢复未提交局部步骤（上限1次）' if resume_region else '复用参考重试本地处理',
               bool(run.get('run_id')),
               '核验已有局部结果及单图输入证据后，仅本地对齐与合成；可调整原模型上下文内的CPU区域，旧提交记录保留。' if reuse_candidate else
               '严格核验零提交失败证明后，沿用已有参考并恢复原区域，至多一次局部生图。' if resume_region
               else '沿用已有参考，不新增生图。',generation=resume_region)
        if reuse_candidate:
            actions[-1]['regional_candidate_reuse'] = True
    if reconstruction_required and assets.get('source') and assets.get('reference'):
        action('prepare_reconstruction','重新准备仿色融合',bool(run.get('run_id')),
               '准备当前本地图层，不新增生图。')
    if state=='pending_save':
        action('save_native_revision','保存高清新版本',not reconstruction_required)
    if findings:
        effect_ids=[row['criterion'] for row in findings if row['criterion']=='effects_space']
        if effect_ids:
            action('inspect_scene_producer','查看主效对照',assets.get('reference') and assets.get('draft'),
                   '先核对效果可读性及空间，不能据旧审片直接补效果。',effect_ids)
        detail_ids=[row['criterion'] for row in findings if row['criterion'] in
                    {'identity_makeup','native_detail','skin_materials','sampling_coverage','current_review'}]
        if detail_ids:
            action('inspect_native_evidence','查看眼口与原尺寸证据',assets.get('draft'),
                   '只查看已有证据与当前原尺寸画面，不发起生图或自动批准。',detail_ids)
        if not source_only:
            repair_reason = '先取得当前完整绑定的新QA，再执行有资格的本地修复；会新增文本审片请求，不新增生图。'
            if subject_repair['finding_ids']:
                repair_reason += subject_repair['reason']
                if subject_repair['last_outcome_detail']:
                    repair_reason += subject_repair['last_outcome_detail']
                if subject_repair['diagnostic_next_action']:
                    repair_reason += subject_repair['diagnostic_next_action']
            if receiver_repair['finding_ids']:
                repair_reason += receiver_repair['reason']
                if receiver_repair['last_outcome_detail']:
                    repair_reason += receiver_repair['last_outcome_detail']
            if environment_repair['finding_ids']:
                repair_reason += environment_repair['reason']
                if environment_repair['last_outcome_detail']:
                    repair_reason += environment_repair['last_outcome_detail']
            action('repair_final_locally','重新审片并本地修复',
                   current_render and run.get('status')=='needs_visual_review'
                   and assets.get('source') and assets.get('reference'), repair_reason,
                   [row['criterion'] for row in findings],qa=True)
    if current_render and state!='accepted':
        action('review_and_accept','进入人工审片',True,'按证据逐项复核，批准须单独确认。')
    return {'result_state':state,'result_label':RESULT_STATE_LABELS[state],
        'open_findings':findings,'recovery_actions':actions,'required_effects':required_effects,
        'native_receiver_detail_repair':receiver_repair,
        'native_subject_detail_repair':subject_repair,
        'subject_environment_repair':environment_repair,
        'task_success_is_visual_acceptance':False}


def paired_subject_observation_state(plan, *, source_sha256=None, candidate_sha256=None, source_only_plan=None):
    """Read saved colour-match binding metadata without loading large assets."""
    plan = _record(plan)
    if not plan and not candidate_sha256 and isinstance(source_only_plan, dict):
        from raw_support import RGB_DECODE_VERSION
        local = source_only_plan
        size = local.get('native_size')
        valid = bool(local.get('schema') == 'source_only_display_plan.v1'
            and local.get('working_domain') == 'display_linear_sRGB'
            and source_sha256 and local.get('source_sha256') == source_sha256
            and local.get('decode_version') == RGB_DECODE_VERSION
            and local.get('source_display_applied_once') is True
            and local.get('reference_assets_used') is False
            and isinstance(size, list) and len(size) == 2
            and all(isinstance(n, int) and n > 0 for n in size))
        return {'state':'source_only_prepared' if valid else 'not_prepared',
            'observation_kind':'explicit_source_only_display', 'required_for_execution':True,
            'persisted_binding_valid':valid, 'reconstruction_required':not valid,
            'editable':valid, 'execution_supported':valid, 'read_only':False,
            'candidate_person_pair_required':False, 'arrays_or_file_hashes_verified':False,
            'errors':[] if valid else ['current_source_only_plan_required']}
    from source_appearance_responsibility import reference_stack_state
    stack = reference_stack_state({'source': {'sha256': source_sha256 or plan.get('source_sha256')},
                                  'reference': {'sha256': candidate_sha256 or plan.get('candidate_sha256')}}, plan)
    if stack['declared']:
        valid = stack['prepared']
        return {'state': 'reference_appearance_stack_saved' if valid else 'not_prepared',
            'observation_kind': 'frozen_original_reference_appearance',
            'required_for_execution': stack['required_for_execution'],
            'persisted_binding_valid': stack['binding_valid'],
            'reconstruction_required': stack['reconstruction_required'],
            'editable': valid, 'execution_supported': valid, 'read_only': False,
            'candidate_person_pair_required': False, 'arrays_or_file_hashes_verified': False,
            'errors': stack['errors'], 'reference_appearance_stack': stack}
    return {'state':'not_prepared', 'observation_kind':'frozen_original_reference_appearance',
        'required_for_execution':True, 'persisted_binding_valid':False, 'reconstruction_required':True,
        'editable':False, 'execution_supported':False, 'read_only':False,
        'candidate_person_pair_required':False, 'arrays_or_file_hashes_verified':False,
        'errors':['current_reference_appearance_stack_required']}


def _stage(stage_id: str, label: str, status: str, detail: str, action: str,
           **evidence: Any) -> dict[str, Any]:
    return {"id": stage_id, "label": label, "status": status,
            "status_label": STAGE_STATUS_LABELS[status], "detail": detail,
            "next_action": action, "evidence": evidence}


def source_responsibility_state(run: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Declare concrete source responsibilities without executing a field."""
    from source_appearance_responsibility import stage_summary
    try:
        summary = stage_summary(run, plan)
        return {**summary, 'binding_valid': summary.get('binding_valid', True),
                'scope': 'source_bound_preparation_declaration_only; no execution or visual acceptance'}
    except (ValueError, TypeError, KeyError) as exc:
        return {'status': 'unsupported', 'binding_valid': False,
                'title': '人物自动职责尚未齐全',
                'message': '当前来源或职责声明未齐；已保存画面仍可查看，新的自动人物效果需先完成本地准备。',
                'available_actions': ['查看已保存画面', '保存或导出已有结果'],
                'automatic_completion_claim': False, 'requires_user_full_body_recolour': False,
                'errors': [str(exc)], 'scope': 'source_bound_declaration_only; no effect execution'}


def _reference_stack_consumed(planned, consumed, stack, *, current_render, asset_present):
    """A saved recipe and an existing master are not evidence of consumption."""
    wanted = planned.get('reference_appearance_stack', {})
    used = consumed.get('reference_appearance_stack', {})
    try:
        strength_matches = float(used.get('strength', -1)) == float(wanted.get('strength', 1))
    except (TypeError, ValueError):
        strength_matches = False
    control_matches = True
    if stack.get('independent_controls'):
        try:
            control_matches = all(float(used.get(key, 1)) == float(wanted.get(key, 1))
                for key in ('colour_strength', 'lightness_strength'))
        except (TypeError, ValueError):
            control_matches = False
        declared_controls = _record(_record(wanted.get('params')).get('control_assets_sha256'))
        used_controls = _record(used.get('consumed_control_assets_sha256'))
        wanted_assets = _record(wanted.get('assets'))
        control_matches = bool(control_matches and used.get('control_recipe') ==
            _record(wanted.get('params')).get('control_recipe') and
            all(used_controls.get(role) and used_controls[role] == declared_controls.get(role)
                == _record(wanted_assets.get(role)).get('sha256')
                for role in ('colour_delta', 'lightness_delta')) and
            used_controls.get('control_receipt') and used_controls['control_receipt'] ==
            _record(wanted_assets.get('control_receipt')).get('sha256'))
    asset = _record(_record(used.get('assets')).get('automatic_delta'))
    binding_matches = bool(
        asset.get('sha256') == stack['asset_sha256'] and stack['asset_sha256']
        and used.get('consumed_asset_sha256') == stack['asset_sha256']
        and _record(used.get('params')) == _record(wanted.get('params'))
        and strength_matches and control_matches)
    applied = bool(current_render and asset_present and stack['prepared'] and stack['enabled']
                   and used.get('enabled') is True and used.get('applied') is True
                   and used.get('execution_stage') == 'native_frozen_reference_appearance_before_composition'
                   and binding_matches)
    return applied, binding_matches


def build_stage_report(run: Mapping[str, Any], observations: Mapping[str, Any]) -> dict[str, Any]:
    """Pure reduction of a native manifest plus read-only observations.

    ``observations.assets`` contains booleans for named files, not worker
    availability.  Review facts must refer to this run and revision.  Stored
    render facts must also match this revision before any effect is reported
    as applied; inherited masters cannot make pending edits look saved.
    """
    assets = _record(observations.get("assets"))
    source = _record(run.get("source"))
    reference = _record(run.get("reference"))
    options = _record(run.get("options"))
    plan = _record(_record(options.get("appearance_revision")).get("design_reconstruction_plan"))
    render = _record(run.get("render"))
    applied_plan = _record(_record(render.get("operator")).get("design_reconstruction_plan"))
    planned = _layers(plan.get("layers"))
    consumed = _layers(applied_plan.get("layers"))
    beauty_fit = portrait_beauty_reference_fit_state(run, observations.get('portrait_beauty_identity'))
    provenance = _record(plan.get("provenance"))
    continuation = _record(run.get("continuation_edit"))
    run_status = str(run.get("status") or "pending")
    native = run.get("input_mode") == "single_source" and options.get("render_mode") == "native_source_design"
    stopped = run_status in {"failed", "interrupted", "cancelled", "blocked"}
    continuation_pending = bool(continuation) and continuation.get("render_state") != "saved"
    revision = run.get("revision")
    rendered_revision = continuation.get("rendered_revision") if continuation else render.get("revision")
    current_render = (native and bool(run.get("run_id")) and render.get("run_id") == run.get("run_id")
                      and isinstance(revision, int) and not isinstance(revision, bool)
                      and revision > 0 and rendered_revision == revision
                      and bool(assets.get("draft")) and not stopped and not continuation_pending)
    if continuation:
        current_render = (current_render and continuation.get("rendered_run_id") == run.get("run_id")
                          and continuation.get("rendered_revision") == run.get("revision"))
    review = _record(observations.get("review"))
    checks = _record(review.get("checks"))
    review_matches = (bool(review) and review.get("run_id") == run.get("run_id")
                      and review.get("revision") == run.get("revision"))
    visual_approved = (current_render and review_matches
                       and run_status in {"accepted", "delivered"}
                       and review.get("status") in {"accepted", "delivered"}
                       and all(checks.get(key) == "pass" for key in VISUAL_CHECK_KEYS))
    subject_observation = paired_subject_observation_state(plan,
        source_sha256=source.get('sha256'), candidate_sha256=reference.get('sha256'),
        source_only_plan=None if reference or _record(run.get('artifacts')).get('reference_candidate') else options.get('source_only_plan'))
    source_only = subject_observation['state'] == 'source_only_prepared'
    source_responsibility = source_responsibility_state(run, plan)
    stack = subject_observation.get('reference_appearance_stack')
    stack_declared = bool(stack)
    control_file_keys = ('reference_appearance_colour_delta', 'reference_appearance_lightness_delta',
                         'reference_appearance_control_receipt')
    missing_controls = [key for key in control_file_keys if not assets.get(key)] if (
        stack and stack.get('independent_controls')) else []
    stack_assets_present = bool(assets.get('reference_appearance_delta') and not missing_controls)
    stack_prepared = bool(stack and stack['prepared'] and stack_assets_present)
    stack_used, stack_binding_matches = (False, False)
    if stack_declared:
        stack_used, stack_binding_matches = _reference_stack_consumed(
            planned, consumed, stack, current_render=current_render,
            asset_present=stack_assets_present)
        if stack['required_for_execution'] and not assets.get('reference_appearance_delta'):
            subject_observation = {**subject_observation, 'reconstruction_required': True,
                'errors': [*subject_observation['errors'], 'reference_stack_delta_missing']}
        if missing_controls:
            subject_observation = {**subject_observation, 'reconstruction_required': True,
                'errors': [*subject_observation['errors'], *['reference_stack_control_file_missing:' + key
                                                            for key in missing_controls]]}
        if stack['enabled'] and not stack_used:
            visual_approved = False
    reconstruction_required = subject_observation['reconstruction_required']
    stages = []

    size = source.get("oriented_size")
    source_complete = (assets.get("source") is True and bool(source.get("sha256"))
                       and isinstance(size, list) and len(size) == 2
                       and all(isinstance(value, int) and value > 0 for value in size))
    source_status = "complete" if source_complete else "failed" if source and not assets.get("source") else "partial" if assets.get("source") else "pending"
    stages.append(_stage("source_import", "RAW / 原片导入", source_status,
                         f"原片已记录，{size[0]} × {size[1]}；人物构图以原片为准。" if source_complete else "原片或尺寸、来源记录尚未齐全。",
                         "保留原片，继续使用当前设计。" if source_complete else "定位原片文件后重新打开工程。",
                         source_present=bool(assets.get("source")), source_identity_declared=bool(source.get("sha256"))))

    retouch = _record(options.get('native_source_retouch'))
    source_attempt = _record(run.get('automatic_native_retouch'))
    baseline_retouch = _record(_record(plan.get('source_baseline')).get('native_source_retouch'))
    retouch_owned = bool(retouch and retouch.get('owner_run_id') == run.get('run_id')
        and retouch.get('source_sha256') == source.get('sha256') and retouch == baseline_retouch)
    retouch_used = bool(current_render and retouch_owned and assets.get('native_retouch_layer'))
    patches = int(retouch.get('approved_patches', 0))
    retouch_detail = (f'已冻结 {patches} 处有依据的原片修复，并随原片基线进入当前版本。' if patches else
        '本次没有获准修复增量，保留原片细节；是否无需修补仍需原尺寸审片确认。') if retouch_used else (
        '自动原片修复未完成，已保留可编辑草稿。' if source_attempt.get('status') in
        ('unavailable_draft_retained', 'cancelled_draft_retained') else '当前版本未执行自动原片瑕疵检查。')
    stages.append(_stage('automatic_native_retouch', '原片自动精修',
        'partial' if retouch_used else 'failed' if source_attempt.get('status') == 'unavailable_draft_retained' else 'pending',
        retouch_detail, '检查原生细节，必要时使用自然修复笔。',
        applied_to_source_baseline=retouch_used, approved_patches=patches,
        identity_binding_matches=retouch_owned, attempt_status=source_attempt.get('status'),
        quality_acceptance_inferred=False))

    beauty = _record(_record(options.get('appearance_revision')).get('portrait_beauty'))
    beauty_receipt = _record(_record(render.get('operator')).get('portrait_beauty')) or _record(applied_plan.get('portrait_beauty'))
    beauty_prepared = bool(beauty.get('result') and assets.get('portrait_beauty_layer'))
    beauty_applied = bool(current_render and beauty_prepared and beauty_receipt.get('applied')
        and beauty_receipt.get('strength') == beauty.get('strength',1.))
    stages.append(_stage('portrait_beauty', 'BeautyGRPO 人像美化',
        'complete' if beauty_applied else 'partial' if beauty_prepared else 'pending',
        '模型生成外观已进入当前保存图；没有强制恢复原片纹理。' if beauty_applied else
        '已准备生成脸部图层，可调整强度后保存。' if beauty_prepared else '当前版本尚未准备生成式美化图层。',
        '比较本人、妆造与遮挡，必要时按人脸恢复原片。',
        prepared=beauty_prepared,applied=beauty_applied,alpha_changed=False,
        current_strength=beauty_fit['current_strength'],disabled_faces=beauty_fit['disabled_faces'],
        reference_fit=beauty_fit,
        source_texture_restoration=False,quality_acceptance_inferred=False))

    reference_complete = assets.get("reference") is True and bool(reference.get("sha256"))
    reference_status = "complete" if reference_complete else "failed" if reference and not assets.get("reference") else "partial" if assets.get("reference") else "pending"
    stages.append(_stage("design_reference", "设计参考", reference_status,
                         "已保存设计参考；本地重建复用它，人物细节来自原片。" if reference_complete else "当前版本尚无完整的设计参考。",
                         "复用已保存设计，进行本地重建。" if reference_complete else "选择已有设计参考，再运行本地重建。",
                         reference_present=bool(assets.get("reference")), reference_identity_declared=bool(reference.get("sha256"))))

    local_status = "unsupported" if not native else "failed" if stopped else "pending"
    background_used = current_render and consumed.get("background_design", {}).get("applied") is True
    background_present = assets.get("background") is True and assets.get("background_validity") is True
    floor = _record(provenance.get("candidate_floor_contract"))
    floor_detail = ("地面仅有局部清理，完整洁净状态未建立。" if floor.get("automatic_cleanup_applied") is True
                    else "地面完整清理状态未建立。")
    background_status = ("complete" if background_used and background_present and visual_approved
                          and floor.get("clean_established") is True else "partial" if background_used and background_present else local_status)
    if background_used and not background_present:
        background_status = "failed"
    stages.append(_stage("background_reconstruction", "背景高清重建", background_status,
                         ("环境设计已进入已保存草稿；生成的高清细节仍需审片。"
                          + (floor_detail if floor.get("clean_established") is not True else ""))
                         if background_used and background_present else "当前版本尚未保存可用的原片坐标背景重建。",
                         "检查木格、雾层和地面是否自然；用当前设计继续本地处理。",
                         applied=bool(background_used), required_assets_present=bool(background_present),
                         floor_cleanup_status=floor.get("analysis_status"), clean_floor_established=floor.get("clean_established") is True))

    support_statuses = stack.get('support_stage_statuses', {}) if stack_declared else {}
    def supported_stage(stage):
        return support_statuses.get(stage) != 'unsupported'
    def stage_effect_used(stage):
        if not stack_used or not supported_stage(stage):
            return False
        support = _record(_record((stack.get('reference_support') or {}).get('stages')).get(stage))
        if not support:
            return True  # saved numeric target consumption, support still unverified
        return bool((support.get('colour_effect_nonzero') and stack.get('colour_strength', 1) > 0)
            or (support.get('lightness_effect_nonzero') and stack.get('lightness_strength', 1) > 0))
    match_used = stage_effect_used('reference_colour_match') if stack_declared else False
    match_detail = ('人物仿色已随整体融合层用于当前高清版本；结构与纹理保留，效果需审片。' if match_used else
                    '整体融合层已准备，尚未用于当前已保存画面。' if stack_prepared else '尚未保存仿色结果。')
    if stack_prepared and not stack['enabled']:
        match_detail = '整体融合层已准备，当前关闭或强度为零；画面使用原人物外观。'
    if stack_declared and not supported_stage('reference_colour_match'):
        match_detail = '缺少可靠同材质颜色锚点；该自动仿色步骤不产生效果，保留原人物颜色并可继续编辑。'
    if stack_declared and beauty_fit['state'] != 'matched':
        match_detail += beauty_fit['detail']
    stages.append(_stage('reference_colour_match', '原图与参考图仿色',
        'unsupported' if stack_declared and not supported_stage('reference_colour_match') else
        'complete' if match_used and visual_approved else 'partial' if match_used or stack_prepared else local_status,
        match_detail if stack_declared else
        '按参考图匹配原片人物颜色，保留结构与纹理；已保存结果待审片。' if match_used else '尚未保存仿色结果。',
        '用当前美化重新准备参考匹配；已有冻结融合层仍可保存和导出。' if beauty_fit['reference_reprepare_recommended'] else
        '检查原图、参考图与仿色结果。', applied=bool(match_used),
        prepared=stack_prepared if stack_declared else subject_observation['persisted_binding_valid'],
        beauty_fit_dependency_matches=beauty_fit['fit_dependency_matches'],
        reference_reprepare_recommended=beauty_fit['reference_reprepare_recommended'],
        beauty_fit_dependency_state=beauty_fit['state']))
    if stack_declared:
        for stage_id, label, used_detail in (
                ('reference_colour_blend', '参考颜色融合', '眼口与同材质参考颜色已随整体融合层应用；轮廓与纹理保留。'),
                ('reference_scene_balance', '人物与环境明暗协调', '已有发丝高光、衣服暗部及局部光色协调已随整体融合层应用，不重画人物。')):
            stage_used = stage_effect_used(stage_id)
            detail = (used_detail + ('本版本已记录视觉批准。' if visual_approved else '仍需整体审片。')
                      if stage_used else '已准备为整体融合层；当前未应用。' if stack_prepared else
                      '来源或准备资产不匹配，尚未有可用的整体融合层。')
            if not supported_stage(stage_id):
                detail = '缺少该步骤所需的可靠区域依据；当前准备降低该区效果。背景和编辑可继续。'
            stages.append(_stage(stage_id, label,
                'unsupported' if not supported_stage(stage_id) else
                'complete' if stage_used and visual_approved else 'partial' if stack_prepared else
                'failed' if reconstruction_required else local_status,
                detail, '对照原片、参考与结果，调整整体融合或局部精修。' if stack_prepared else '保留已保存画面，重新准备人物与场景融合。',
                prepared=bool(stack_prepared and stage_id in stack['prepared_stages']),
                applied=stage_used, current_render_saved=bool(current_render),
                consumed_binding_matches=stack_binding_matches,
                freeze_mode=stack['freeze_mode'], working_domain=stack['working_domain'],
                independent_stage_controls=False, independent_colour_lightness_controls=stack.get('independent_controls', False),
                stage_parameter_changes_require_reprepare=True,
                reference_support=stack.get('reference_support'), support_status=support_statuses.get(stage_id),
                visual_approved=bool(visual_approved)))
    contact = consumed.get('contact_shadow', {})
    contact_used = current_render and contact.get('applied') is True and contact.get('receiver_support_consumed') is True
    authored_used = current_render and consumed.get('authored_receiver_shadow', {}).get('applied') is True
    stages.append(_stage('contact_and_edges', '溶图、接触与边缘',
        'partial' if current_render and assets.get('alpha') else local_status,
        '原片人物覆盖率与设计背景合成；边缘与接触效果需审片。',
        '查看整图、发丝、衣缘与鞋底。', contact_attenuation_applied=bool(contact_used),
        authored_receiver_projection_applied=bool(authored_used),
        final_colour_edge_extension_applied=bool(stage_effect_used('edge_donor_extension')
            and 'edge_donor_extension' in stack['prepared_stages']),
        single_alpha_composition_declared=bool(current_render and assets.get('alpha') and
            applied_plan.get('background_coverage') == 'derived 1-alpha; exactly one final composition')))

    effects = _record(run.get('automatic_foreground_effects'))
    effect_layers = [row for row in plan.get('layers', []) if row.get('kind') == 'independent_effect']
    effect_records = [row for row in applied_plan.get('layers', []) if row.get('kind') == 'independent_effect']
    effects_used = bool(current_render and effect_layers and all(any(
        saved.get('id') == row.get('id') and saved.get('applied') is True for saved in effect_records)
        for row in effect_layers if row.get('enabled')))
    orchestration = _record(run.get('system_post_completion'))
    fx_stage = next((row for row in orchestration.get('stages', []) if row.get('id') == 'independent_effects'), {})
    fx_skipped = fx_stage.get('status') == 'skipped' and not effect_layers
    stages.append(_stage('independent_effects', '空间特效',
        'partial' if effects_used or fx_skipped else 'failed' if effects.get('status') in
        ('effect_unavailable', 'effect_rejected', 'effect_review_unavailable') else 'pending',
        '独立 RGBA 特效已进入当前版本；遮挡和画面协调仍以成片审片为准。' if effects_used else
        '本次未新增独立效果；背景主效是否满足设计见最终审片。' if fx_skipped else
        '独立特效尚未通过生成与视觉检查，当前草稿保留。',
        '按角色设计检查特效位置、遮挡和主次。', applied=effects_used,
        independent_effect_skipped=fx_skipped, skipped_grants_primary_effect_fulfillment=False,
        actual_visual_gate=effects.get('actual_visual_gate', 'unknown'),
        image_submissions=effects.get('independent_effect_image_submissions', 0),
        receiver_light_shadow_reflection_established=False))

    manual = _record(options.get("manual_layers"))
    saved_manual = _record(render.get("manual_layers"))
    brush = _record(manual.get("local_colour_brush"))
    strokes = brush.get("strokes") if isinstance(brush.get("strokes"), list) else []
    manual_used = current_render and bool(manual) and manual == saved_manual
    manual_status = "complete" if manual_used and visual_approved else "partial" if manual_used else "failed" if stopped and manual else "pending" if native else "unsupported"
    stages.append(_stage("manual_refinement", "手动精修", manual_status,
                         f"已保存精修参数与 {len(strokes)} 笔局部光色笔画；效果仍以当前版本审片为准。" if manual_used else "当前精修尚未保存到本版本高清结果。",
                         "在原生细节视图检查脸、发、手、衣、鞋；精修后保存一个新版本。",
                         parameters_saved_in_render=bool(manual_used), local_colour_stroke_count=len(strokes)))

    diagnostic = cosplay_review_state(run, current_render=current_render)
    native_selection = _record(_record(run.get('cosplay_post_review')).get('native_selection'))
    stages.append(_stage('complete_cosplay_review', '整体与原尺寸审片',
        'partial' if diagnostic['status'] in {'reviewed','inherited'} else 'failed' if diagnostic['status'] == 'unavailable' else 'pending',
        '已完成整体和原尺寸抽样审片；通过、失败与证据不足分别记录。' if diagnostic['status'] == 'reviewed' else
        '展示同像素旧审片与出处；不是当前版本新审片，不授予自动修复资格。' if diagnostic['status']=='inherited' else
        '当前版本尚无有效的完整审片。', '先处理审片失败项，再确认成片。',
        current_review_binding_matches=diagnostic['current_render_binding_matches'],
        selection=native_selection, automatic_visual_acceptance=False))

    masters = (current_render and source_complete and (reference_complete or source_only) and assets.get("manifest")
               and assets.get("master") and assets.get("scene_master"))
    export_state = _export_receipt_state(run, observations) if current_render else {'valid': [], 'formats': [], 'complete': False}
    from artifact_receipts import binding_for_run
    reopen_verified = bool(current_render and any(
        row.get('schema_version') == 'portable_import_receipt.v1'
        and row.get('binding') == binding_for_run(run)
        and row.get('asset_checksums_verified') is True and row.get('files_unchanged') is True
        for row in _record(observations.get('artifact_receipts')).get('imports') or []
        if isinstance(row, dict)))
    journey_complete = bool(masters and reopen_verified and export_state['complete'])
    save_status = "complete" if journey_complete else "partial" if masters else "failed" if stopped else "partial" if current_render else "pending" if native else "unsupported"
    export_formats = ["PNG"] if current_render else []
    if current_render and assets.get("master"):
        export_formats.append("TIFF16")
    if current_render and assets.get("scene_master"):
        export_formats.append("TIFF32_LINEAR")
    export_status = ('complete' if export_state['complete'] else
                     'partial' if export_state['formats'] else 'pending')
    if journey_complete:
        save_detail = "工程与高清母版已保存；portable 导入已核验资产校验和，当前版本已导出所选格式。"
        save_action = "继续编辑会创建新版本；当前版本已完成保存、重开与导出验收。"
    elif masters:
        save_detail = "本版本工程与高清母版已保存；保存、重开和导出分别记录，尚未形成完整回执链。"
        save_action = "导出所需格式；如需迁移工程，可保存 portable 工程并在新工作区重开。"
    else:
        save_detail = "需先保存当前版本高清结果；待处理或中断编辑不能导出旧母版。"
        save_action = "恢复本地处理并保存当前版本，再导出。"
    stages.append(_stage("save_and_export", "工程保存、重开与导出", save_status,
                         save_detail, save_action,
                         current_render_saved=bool(current_render), native_masters_present=bool(masters),
                         exportable_formats=export_formats,
                         export_completion="verified" if export_state['complete'] else
                         "partial" if export_state['formats'] else "not_verified",
                         save_status="complete" if masters else "pending",
                         reopen_status="complete" if reopen_verified else "pending",
                         export_status=export_status,
                         export_receipts=export_state['valid'],
                         verified_export_formats=export_state['formats'],
                         portable_asset_checksums_verified=reopen_verified,
                         process_resume_verified=False,
                         refresh_rehashes_pixels=False))

    if source_only:
        for row in stages:
            if row['id'] in {'design_reference','background_reconstruction','reference_colour_match','contact_and_edges','independent_effects'}:
                row.update(status='not_requested',status_label=STAGE_STATUS_LABELS['not_requested'],
                    detail='当前使用无参考的本地原片编辑，本阶段未请求。',
                    next_action='继续精修原片，预览并保存当前版本。',required_for_this_run=False)
        source_responsibility = {'status':'partial','binding_valid':True,
            'title':'本地原片编辑','message':'使用明确绑定的原片显示基线，不使用参考资产。',
            'available_actions':['预览当前编辑','保存高清版本','导出保存结果'],
            'automatic_completion_claim':False,'requires_user_full_body_recolour':False}
    if not native:
        draft_state, headline, next_action = "unsupported_route", "该版本不属于当前生产路径", "选择 native 原片重建版本；历史资产仅供查看。"
    elif stopped or (source and not assets.get("source")):
        draft_state, headline, next_action = "needs_recovery", "本版本需要恢复", "检查失败记录，定位缺失资产；复用已保存设计恢复本地处理。"
    elif continuation_pending:
        draft_state, headline, next_action = "pending_save", "当前编辑尚未保存", "先完成本版本的本地处理与高清保存，再导出。"
    elif source_status == "pending":
        draft_state, headline, next_action = "awaiting_source", "等待导入原片", "导入一张 RAW 或原片，再选择设计参考。"
    elif source_only and current_render:
        draft_state, headline, next_action = 'local_source_draft', '原片高清编辑已保存', '继续本地精修，或导出当前已保存结果。'
    elif not reference_complete:
        draft_state, headline, next_action = "awaiting_reference", "原片已导入，等待设计参考", "选择已有设计参考，继续本地重建。"
    elif not current_render:
        draft_state, headline, next_action = "awaiting_reconstruction", "设计已保存，等待本地重建", "复用已保存设计，完成本地高清重建。"
    elif reconstruction_required:
        draft_state = "needs_local_rebuild"
        headline = "已保存草稿可查看；人物与场景融合需重新准备" if stack_declared else "已保存草稿可查看；仿色参数尚未齐全"
        next_action = "保留当前原图与参考，重新准备人物与场景融合。" if stack_declared else "复用当前原图与参考图，准备仿色参数。"
    elif stack_declared and stack['enabled'] and not stack_used:
        draft_state, headline, next_action = "awaiting_fusion_save", "人物与场景融合已准备，当前画面尚未应用", "预览整体融合，保存当前高清版本后再审片。"
    elif visual_approved:
        draft_state, headline, next_action = "visually_accepted", "本版本已记录视觉批准", "从已保存母版导出所需格式；效果模块的未支持项仍保留。"
    else:
        draft_state, headline, next_action = "usable_draft", "已保存高清草稿，整体效果仍需完善", "检查仿色、人物边缘与背景融合。"
    cosplay_diagnosis = cosplay_review_state(run, current_render=current_render)
    failed_checks = [row for row in cosplay_diagnosis['checks'] if row['status']=='fail']
    if failed_checks and draft_state == 'usable_draft':
        headline = '已保存高清草稿，Cos 成片诊断发现待修项'
        next_action = str(failed_checks[0].get('action') or '先修正诊断中的失败阶段，再比较原片、设计参考和成片。')
    whole_work={
        'status':'partial','current_revision_visual_approved':bool(visual_approved),
        'complete_automatic_pipeline_claim':False,'saved_current_revision':bool(current_render),
        'reference_colour_match_applied':bool(match_used),
        'reference_appearance_stack_prepared':stack_prepared,
        'reference_appearance_stack_applied':stack_used,
        'portrait_beauty_reference_fit':beauty_fit,
        'reference_colour_blend_applied':stage_effect_used('reference_colour_blend') if stack_declared else False,
        'reference_scene_balance_applied':stage_effect_used('reference_scene_balance') if stack_declared else False,
        'current_recipe_render_saved':bool(current_render and (not stack_declared or not stack['enabled'] or stack_used)),
        'output_strategy':'source_native_layers',
        'composition_applied':bool(background_used),
        'save_reopen_export_journey_complete':journey_complete,
        'system_completion_attempted':bool(orchestration),
        'system_completion_status':orchestration.get('status'),
        'automatic_native_retouch_baseline_consumed':retouch_used,
        'independent_effect_applied':effects_used,
        'basis':'saved_manifest_execution_and_current_revision_review'}
    required_effects = _required_effect_state(run, cosplay_diagnosis,
        source_only=source_only, visual_approved=visual_approved,
        fx_status=fx_stage.get('status', effects.get('status','not_requested')))
    whole_work.update(_product_result(run, cosplay_diagnosis, draft_state=draft_state,
        current_render=current_render, source_only=source_only, visual_approved=visual_approved,
        assets=assets, reconstruction_required=reconstruction_required, required_effects=required_effects))
    if draft_state in {'usable_draft','visually_accepted','local_source_draft'}:
        headline = whole_work['result_label'] + ' · 已保存高清版本'
        if whole_work['open_findings'] and not visual_approved:
            next_action = whole_work['open_findings'][0]['action'] or '查看具体问题与证据后继续处理。'
    return {"schema_version": "native_pipeline_stage_report.v1", "run_id": run.get("run_id"),
            "creative_task": _task_summary(run),
            "revision": run.get("revision"), "basis": "persisted_manifest_and_asset_presence",
            "scope": "saved_revision", "native_route": native, "draft_state": draft_state,
            "headline": headline, "next_action": next_action, "visual_approved": bool(visual_approved),
            'cosplay_post_completion': (_record(run.get('cosplay_post_completion'))
                if run.get('run_id') in {_record(run.get('cosplay_post_completion')).get('initial_run_id'),
                                        _record(run.get('cosplay_post_completion')).get('result_run_id')} else {}),
            "reconstruction_required": reconstruction_required,
            "paired_subject_observation": subject_observation,
            "source_appearance_responsibility": source_responsibility,
            "portrait_beauty_reference_fit": beauty_fit,
            "reference_appearance_stack": ({**stack, 'asset_present': bool(assets.get('reference_appearance_delta')),
                'control_assets_present': not missing_controls, 'missing_control_assets': missing_controls,
                'prepared': stack_prepared, 'applied': stack_used,
                'consumed_binding_matches': stack_binding_matches} if stack_declared else None),
            "whole_work": whole_work, "cosplay_post_diagnosis":cosplay_diagnosis,
            'system_post_completion':orchestration,
            "assets_rehashed": False, "quality_inferred_from_worker_success": False,
            "stages": stages}


def read_stage_report(root: Path, run: Mapping[str, Any], manifest_path: Path | None = None) -> dict[str, Any]:
    """Read presence, review and small Beauty identity metadata; preserve assets."""
    root = Path(root)
    artifacts = _record(run.get("artifacts"))
    plan = _record(_record(_record(run.get("options")).get("appearance_revision")).get("design_reconstruction_plan"))
    planned = _layers(plan.get("layers"))
    applied = _record(_record(_record(run.get("render")).get("operator")).get("design_reconstruction_plan"))

    def path(value: Any) -> Path | None:
        value = _record(value).get("path") if isinstance(value, dict) else value
        if not isinstance(value, str) or not value:
            return None
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    background = _record(planned.get("background_design", {}).get("assets"))
    stack = _record(planned.get('reference_appearance_stack', {}).get('assets'))
    paths = {"manifest": manifest_path or path(run.get("manifest")),
             "source": path(_record(run.get("source"))), "reference": path(_record(run.get("reference"))),
             "draft": path(artifacts.get("auto_draft")), "master": path(artifacts.get("render_master")),
             "scene_master": path(artifacts.get("render_scene_master")), "alpha": path(artifacts.get("alpha")),
             "background": path(background.get("background")), "background_validity": path(background.get("validity")),
             "reference_appearance_delta": path(stack.get('automatic_delta')),
             "reference_appearance_colour_delta": path(stack.get('colour_delta')),
             "reference_appearance_lightness_delta": path(stack.get('lightness_delta')),
             "reference_appearance_control_receipt": path(stack.get('control_receipt')),
             "edge_support": path(applied.get("edge_applicability"))}
    paths['native_retouch_layer'] = path(_record(_record(run.get('options')).get('native_source_retouch')).get('layer'))
    paths['portrait_beauty_layer'] = path(_record(_record(_record(run.get('options')).get('appearance_revision')).get('portrait_beauty')).get('result'))
    from artifact_receipts import observe_receipts
    observations: dict[str, Any] = {"assets": {key: value is not None and value.is_file() for key, value in paths.items()},
                                    "review": {}, "artifact_receipts": observe_receipts(root, run)}
    from portrait_beauty_layer import identity_for_run
    beauty_run = dict(run)
    beauty_options = dict(_record(run.get('options')))
    beauty_appearance = dict(_record(beauty_options.get('appearance_revision')))
    beauty_control = dict(_record(beauty_appearance.get('portrait_beauty')))
    if beauty_control and paths['portrait_beauty_layer'] is not None:
        beauty_control['result'] = str(paths['portrait_beauty_layer'])
        beauty_appearance['portrait_beauty'] = beauty_control
        beauty_options['appearance_revision'] = beauty_appearance
        beauty_run['options'] = beauty_options
    try:
        observations['portrait_beauty_identity'] = {
            'identity_observed': True, 'current_identity': identity_for_run(beauty_run)}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        observations['portrait_beauty_identity'] = {
            'identity_observed': False, 'current_identity': None, 'error': str(exc)}
    review_path = path(artifacts.get("visual_review"))
    if review_path is not None and review_path.is_file():
        try:
            observations["review"] = _record(json.loads(review_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return build_stage_report(run, observations)


def _task_summary(run):
    """Describe saved intent only; it grants no current execution authority."""
    intent = _record(run.get('task_intent') or _record(run.get('options')).get('task_intent'))
    plan = _record(run.get('task_plan') or _record(run.get('options')).get('task_plan'))
    if not intent:
        return None
    from creative_task import MODES
    return {'mode': intent.get('mode'), 'label': MODES.get(intent.get('mode'), '制作任务'),
            'brief': intent.get('brief', ''), 'constraints': _record(intent.get('constraints')),
            'services': _record(intent.get('services')), 'image_budget': _record(plan.get('image_budget')),
            'plan_id': plan.get('plan_id'), 'basis': 'saved_task_intent', 'execution_authority': False}


def format_stage_report(report: Mapping[str, Any]) -> str:
    """User-facing wording shared by Qt and other local consumers."""
    lines = [str(report.get("headline") or "尚未打开已保存版本")]
    task = _record(report.get('creative_task'))
    if task:
        lines.append('制作任务：' + str(task['label']))
        if task.get('brief'):
            lines.append('创作要求：' + task['brief'])
        lines.append('计划图像生成提交上限：' + str(sum(task['image_budget'].values())))
        lines.append('外部文本/视觉服务：' + ('允许' if task['services'].get('text_vision') else '不使用'))
    work = _record(report.get('whole_work'))
    if work.get('result_state'):
        lines.append('作品状态：' + str(work.get('result_label') or work['result_state']))
        lines.append('任务执行结束后，作品仍按缺项、证据和人工批准分别判断。')
        effects = _record(work.get('required_effects'))
        labels={'not_required':'本次无需','fulfilled':'已确认','failed':'待修','unknown':'待证据'}
        lines.append('设计主效：'+labels.get(effects.get('fulfillment'),'待证据')+'；'+str(effects.get('reason') or ''))
    for stage in report.get("stages") or []:
        lines.append(f"{stage['label']} · {stage['status_label']}：{stage['detail']}")
    diagnosis = _record(report.get('cosplay_post_diagnosis'))
    if diagnosis.get('status') in {'reviewed','inherited'}:
        if diagnosis.get('status')=='inherited':
            lines.append('以下为同像素继承审片，原生产版本：'+str(diagnosis.get('producer_run_id') or '未知')+
                '；仅显示已知问题，自动修复仍需当前完整绑定的新QA。')
        states = {'pass':'观察通过','fail':'需改善','unknown':'证据不足'}
        for check in diagnosis.get('checks') or []:
            lines.append(f"Cos 成片 · {check['label']} · {states.get(check['status'],'证据不足')}：{check['reason']}")
        lines.append('自动看图诊断用于定位修正阶段；视觉批准仍须单独记录。')
    elif diagnosis.get('status') == 'pending':
        lines.append('Cos 成片自动诊断正在进行；当前草稿已保存，尚未产生可用审片证据或批准。')
    elif diagnosis.get('status') == 'unavailable':
        lines.append('Cos 成片自动诊断不可用；当前草稿仍须按六项标准审片。')
    elif diagnosis.get('status') == 'stale':
        lines.append('已有审片与当前版本不匹配；旧结论不能授予当前修复或批准资格。')
    for action in work.get('recovery_actions') or []:
        if not action.get('supported'):
            lines.append(str(action.get('label') or '恢复动作')+'：'+str(action.get('reason') or '当前不可执行。'))
    receiver = _record(work.get('native_receiver_detail_repair'))
    if receiver.get('finding_ids') or receiver.get('last_outcome'):
        lines.append('接收面原尺寸恢复：'+str(receiver.get('reason') or '资格待核对。'))
        if receiver.get('last_outcome_detail'):
            outcome = _record(receiver.get('last_outcome'))
            lines.append('接收面恢复上次结果：'+str(receiver['last_outcome_detail'])+
                         (' 原因：'+str(outcome['reason']) if outcome.get('reason') else ''))
    subject = _record(work.get('native_subject_detail_repair'))
    if subject.get('finding_ids') or subject.get('last_outcome'):
        lines.append('人物原片细节恢复：'+str(subject.get('reason') or '资格待核对。'))
        if subject.get('last_outcome_detail'):
            outcome = _record(subject.get('last_outcome'))
            lines.append('人物恢复上次结果：'+str(subject['last_outcome_detail'])+
                         (' 原因：'+str(outcome['reason']) if outcome.get('reason') else ''))
        if subject.get('diagnostic_summary'):
            lines.append('人物细节数值诊断：'+str(subject.get('diagnostic_summary_label') or '仅有历史诊断记录。'))
            if subject.get('diagnostic_summary_current_binding_matches'):
                owners = _record(subject['diagnostic_summary']).get('owners') or []
                for owner in owners:
                    if not isinstance(owner, dict):
                        continue
                    status_label = SUBJECT_DIAGNOSTIC_LABELS.get(owner.get('status'), '需补充局部测量。')
                    lines.append('原片归属 '+str(owner.get('semantic_class_id'))+
                        ' · 原尺寸窗口 '+str(owner.get('native_box'))+'：'+status_label+
                        ' 可测像素 '+str(owner.get('permitted_pixels', 0))+
                        '；测量块 '+str(owner.get('measured_blocks', 0))+'。')
            lines.append('人物细节后续：'+str(subject.get('diagnostic_next_action') or '补充当前原尺寸诊断。'))
            lines.append('数值诊断不改写视觉审片；原失败项、未知项和人工批准要求保持。')
    environment = _record(work.get('subject_environment_repair'))
    if environment.get('finding_ids') or environment.get('last_outcome'):
        lines.append('人物与环境光色协调：'+str(environment.get('reason') or '资格待核对。'))
        if environment.get('last_outcome_detail'):
            outcome = _record(environment.get('last_outcome'))
            lines.append('人物环境协调上次结果：'+str(environment['last_outcome_detail'])+
                (' 原因：'+str(outcome['reason']) if outcome.get('reason') else ''))
        lines.append('自动动作保留原片几何与细节，局部响应来自独立匹配参考；新像素须重新审片。')
    correction = _record(report.get('cosplay_post_completion'))
    correction_labels = {
        'correcting_reference':'正在执行一次参考纠错',
        'correction_rejected':'纠错参考未通过检查，已保留原草稿',
        'correction_unavailable':'参考纠错或检查不可用，已保留原草稿',
        'correction_cancelled':'参考纠错已取消，已保留原草稿',
        'composing_corrected_child':'参考检查通过，正在准备新原片合成',
        'corrected_local_processing_failed':'新原片合成未完成，可复用候选重试本地处理',
        'corrected_child_saved':'纠错子版本已保存并复检，仍需成片审片',
    }
    if correction.get('status') in correction_labels:
        lines.append('Cos 自动纠错 · '+correction_labels[correction['status']]
            +f"；纠错提交 {correction.get('correction_submissions',0)}/1。")
    lines.append("下一步：" + str(report.get("next_action") or "导入原片或打开已保存工程。"))
    return "\n".join(lines)
