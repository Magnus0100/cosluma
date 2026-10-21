"""Run explicit offline suites or selected checks; model checks are opt-in."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
CORE = (
    'test_runner', 'public_source', 'photo_editing', 'photo_appearance', 'photo_development',
    'photo_edit_session', 'vision_transport', 'appearance_read_scope', 'photo_shaping_geometry',
    'scoped_product_state', 'asset_identity', 'production_coordinator', 'creative_task',
    'task_start_dispatch', 'manual_target_budget', 'generation_budget', 'saved_result_selection',
    'result_publication', 'job_store_concurrency', 'manifest_persistence', 'export_artifact',
    'artifact_receipts',
)

INTEGRATION = (
    'engineering_context', 'engineering_execution', 'photo_development_session',
    'photo_development_product', 'photo_source_session', 'photo_session_evidence',
    'photo_session_review', 'photo_shaping', 'face_contour_deform', 'geometric_appearance',
    'appearance_proof_cache', 'geometric_consumers', 'face_shaping_preflight', 'shaping_portable',
    'shaping_ui', 'shaping_inspection_ui', 'shaping_quality', 'generated_appearance_render',
    'generated_appearance_portable', 'generated_appearance_ui', 'photo_session_decision',
    'photo_request_budget', 'photo_edit_tools', 'photo_edit_iteration', 'photo_edit_finish',
    'photo_edit_autofinish', 'autofinish_ui_graph', 'generated_appearance_selection',
    'user_feedback_finish', 'manual_unknown_guard', 'photo_focus_integration',
    'photo_focus_iteration', 'scoped_model_preflight', 'editing_scope_ui',
    'editing_improvement_ui', 'scoped_production', 'scoped_workflow', 'editing_iteration_graph',
    'model_setup', 'model_discovery', 'embedded_backend', 'task_model_preflight',
    'creative_direction', 'creative_direction_execution', 'creative_direction_ui',
    'creative_effect_state', 'native_detail_comparison', 'post_completion_regional_budget',
    'source_scene_enhancement', 'regional_scene_worker', 'regional_resume_contract',
    'regional_candidate_reuse', 'installed_provider_worker', 'same_photo_finish_contract',
    'source_conditioned_design_contract', 'same_photo_finish_ui', 'registered_reference_style',
    'registered_style_pipeline', 'skin_illumination', 'refinement_profile',
    'source_skin_scene_continuity', 'source_hair_preservation', 'face_protection_pair',
    'face_pair_callers', 'frozen_workflow_isolation', 'beautygrpo_broker',
    'beauty_memory_calibration', 'backend_product_state', 'active_job_reentry',
    'preparation_policy', 'saved_task_replanning', 'local_redo_failure',
    'workflow_producer_identity', 'workspace_cleanup', 'desktop_launcher', 'windows_installation',
    'user_data_root', 'cache_single_flight', 'cache_index_concurrency', 'qwen_service_ownership',
    'qwen_norm_compat', 'runtime_preflight', 'external_python_isolation', 'live_timing',
    'workflow_timing', 'workflow_timing_integration', 'creative_task_ui',
    'generation_model_selection', 'startup_readiness', 'cos_node_registry', 'cos_workflow',
    'cos_workspace_integration', 'qwen_provider', 'qwen_alpha_quality', 'qwen_gpu_transition',
    'portrait_beauty_workbench', 'boundary_light_coordination_workbench', 'release_portability',
    'reference_hair_boundary_registration', 'hair_continuous_illumination',
    'boundary_producer_cache', 'scene_support_continuation', 'continuous_field_contract',
    'local_field_domain', 'frozen_native_paths', 'backend_batch_export', 'source_copy',
    'source_preview_cache', 'quick_progress', 'variant_selection_job', 'job_polling',
    'saved_completion', 'job_owner', 'beauty_reference_overlap', 'beauty_existing_ownership',
)

MODEL_CHECKS = frozenset((
    'automatic_native_retouch', 'automatic_scene_completion', 'background_consistency',
    'background_superresolution', 'beautygrpo_prepare', 'beautygrpo_prompt_cache',
    'beautygrpo_worker', 'final_material_completion', 'native_cuda_fields', 'native_cuda_filters',
    'native_cuda_matting', 'portrait_beauty_reference_fit', 'qualified_material_finish',
    'reference_automatic_layers', 'reference_colour_blend', 'reference_colour_fit_roi',
    'reference_diagnostics', 'reference_dinov3_profile', 'reference_local_correspondence_cuda',
    'reference_material_palette', 'reference_material_palette_roi',
    'reference_native_baseline_cli', 'reference_scene_balance', 'reference_support',
    'runtime_project_separation', 'sapiens_current', 'subject_environment_field',
))

SUITES = {'core': CORE, 'integration': INTEGRATION, 'full': CORE + INTEGRATION}


def selected_checks(names):
    """Resolve one explicit suite or deduplicated check names before execution."""
    names = names or ['core']
    if len(names) == 1 and names[0] in SUITES:
        return list(SUITES[names[0]])
    if any(name in SUITES for name in names):
        raise ValueError('Use core, integration or full alone; otherwise pass explicit check names')
    if 'all' in names:
        raise ValueError('The implicit all suite was removed; use full for offline regression or name affected checks')
    result = []
    for name in names:
        name = name.removesuffix('.py').removesuffix('_smoke')
        if not name or Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Unknown check: ' + name)
        if name not in result:
            result.append(name)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checks', nargs='*', help='core, integration, full, or explicit check names')
    parser.add_argument('--list', action='store_true', help='List selected checks without importing or running them')
    parser.add_argument('--model-python', type=Path, help='Explicit runtime for selected model-dependent checks')
    args = parser.parse_args(argv)
    try:
        names = selected_checks(args.checks)
    except ValueError as error:
        parser.error(str(error))
    paths = []
    for name in names:
        path = Path(__file__).parent / (name + '_smoke.py')
        if not path.is_file():
            parser.error('Unknown check: ' + name)
        paths.append(path)
    if args.list:
        print('\n'.join(names))
        return 0
    requested_models = set(names) & MODEL_CHECKS
    if requested_models and not args.model_python:
        parser.error('Selected model checks require --model-python: ' + ', '.join(sorted(requested_models)))
    if args.model_python and not args.model_python.is_file():
        parser.error('The selected model Python does not exist')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT/'scripts'), str(ROOT/'tests')))
    env.setdefault('QT_QPA_PLATFORM', 'offscreen')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('OMP_NUM_THREADS', '1')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # Windows runners may expose TEMP through a short name or junction. Use
    # the same canonical directory as production path validation in children.
    temporary = str(Path(tempfile.gettempdir()).resolve())
    env.update(TEMP=temporary, TMP=temporary, TMPDIR=temporary)
    for name, path in zip(names, paths):
        print('Running ' + path.name, flush=True)
        python = str(args.model_python.resolve()) if name in MODEL_CHECKS else sys.executable
        started = perf_counter()
        result = subprocess.run([python, '-B', '-X', 'utf8', str(path)], cwd=ROOT, env=env)
        print(f'Finished {name}: exit={result.returncode}, seconds={perf_counter()-started:.3f}', flush=True)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
