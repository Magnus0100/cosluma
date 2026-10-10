"""Classify a bound final diagnosis without guessing a pixel producer.

The coordinator establishes actual pixel binding before using this plan. A
scene transfer verdict authorizes an audit, not arbitrary smoke, re-generation
or contrast. Unknown evidence never authorizes a pixel or image request.
"""
from __future__ import annotations

SCHEMA = 'cosplay_final_repair_plan.v1'


def plan_final_repairs(run, *, current_review):
    from native_receiver_detail import qualified_review_windows as receiver_review_windows
    from native_subject_detail import qualified_review_windows as subject_review_windows
    review = run.get('cosplay_post_review') or {}
    qualified = bool(current_review and review.get('status') == 'reviewed'
                     and review.get('actual_api_evidence_valid') is True)
    tasks = []
    receiver_windows = receiver_review_windows(run) if qualified else []
    subject_windows = subject_review_windows(run) if qualified else []
    for criterion, pair in (review.get('checks') or {}).items():
        if not isinstance(pair, dict):
            continue
        final = pair.get('final') or {}
        if final.get('status') not in {'fail', 'unknown'}:
            continue
        status, stage, cause = final['status'], final.get('correction_stage'), final.get('cause')
        if not qualified:
            producer, disposition = None, 'needs_current_pixel_bound_diagnosis'
        elif status == 'unknown':
            producer, disposition = None, 'needs_scoped_evidence'
        elif criterion == 'skin_materials' and stage == 'subject_appearance':
            producer, disposition = 'qualified_material_finish', 'qualify_material_targets'
        elif criterion == 'scene_integration' and stage == 'scene_composition' and cause == 'retouch_missing':
            producer, disposition = 'subject_environment_response', 'qualify_supported_subject_environment_response'
        elif stage == 'scene_composition' and cause in {'transfer_incomplete', 'composition_defect'}:
            producer, disposition = 'background_consistency', 'audit_before_local_repair'
        elif stage == 'native_retouch':
            subject_pairs = [row for row in subject_windows if criterion in row['failed_criteria']]
            paired = [row for row in receiver_windows if criterion in row['failed_criteria']]
            if criterion == 'native_detail' and cause == 'retouch_missing' and subject_pairs:
                producer, disposition = 'native_subject_detail', 'qualify_observed_source_owned_subject_detail'
            elif cause == 'retouch_missing' and paired:
                producer, disposition = 'native_receiver_detail', 'qualify_observed_native_receiver_detail'
            else:
                producer, disposition = None, 'needs_source_owned_retouch_qualification'
        elif stage == 'unsupported_layering':
            producer, disposition = None, 'unsupported_capability'
        elif stage == 'reference_generation':
            # Reference defects have their own upstream bounded coordinator.
            producer, disposition = None, 'upstream_reference_owner'
        else:
            producer, disposition = None, 'unresolved_asset_owner'
        tasks.append({'criterion': criterion, 'target': 'final', 'status': status,
                      'cause': cause, 'correction_stage': stage,
                      'producer': producer, 'disposition': disposition,
                      'action': final.get('action') or '',
                      'evidence': final.get('evidence') or [],
                      'qualification_pair_ids': [row['pair_id'] for row in (
                          subject_windows if producer == 'native_subject_detail' else receiver_windows)
                          if criterion in row['failed_criteria']] if producer in {
                              'native_subject_detail', 'native_receiver_detail'} else [],
                      'pixel_repair_qualified': qualified and status == 'fail' and producer is not None,
                      'generation_authorized_by_this_plan': False})
    return {'schema': SCHEMA, 'run_id': run.get('run_id'), 'revision': run.get('revision'),
            'source_sha256': (run.get('source') or {}).get('sha256'),
            'reference_sha256': (run.get('reference') or {}).get('sha256'),
            'current_review': qualified, 'tasks': tasks,
            'maximum_local_repair_rounds': 1, 'automatic_visual_acceptance': False,
            'image_generation_budget_added': 0}
