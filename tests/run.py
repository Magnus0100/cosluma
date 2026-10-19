"""Run current offline checks with production modules on the import path."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ('public_source', 'asset_identity', 'production_coordinator', 'model_setup', 'model_discovery', 'embedded_backend', 'creative_task', 'task_start_dispatch', 'task_model_preflight',
        'creative_direction', 'creative_direction_execution', 'creative_direction_ui', 'creative_effect_state',
        'native_detail_comparison', 'post_completion_regional_budget', 'source_scene_enhancement', 'regional_scene_worker',
        'regional_resume_contract', 'regional_candidate_reuse', 'installed_provider_worker',
        'same_photo_finish_contract', 'source_conditioned_design_contract', 'same_photo_finish_ui', 'registered_reference_style',
        'registered_style_pipeline', 'skin_illumination', 'refinement_profile', 'source_skin_scene_continuity', 'source_hair_preservation',
        'face_protection_pair', 'face_pair_callers', 'frozen_workflow_isolation', 'beautygrpo_broker', 'beauty_memory_calibration',
        'backend_product_state', 'active_job_reentry', 'preparation_policy', 'saved_task_replanning',
        'saved_result_selection', 'result_publication', 'local_redo_failure',
        'workflow_producer_identity', 'workspace_cleanup', 'job_store_concurrency',
        'manifest_persistence', 'desktop_launcher', 'windows_installation', 'user_data_root',
        'cache_single_flight', 'cache_index_concurrency', 'qwen_service_ownership', 'qwen_norm_compat',
        'runtime_preflight', 'live_timing',
        'creative_task_ui', 'generation_model_selection', 'startup_readiness', 'cos_node_registry', 'cos_workflow', 'cos_workspace_integration', 'qwen_provider', 'qwen_gpu_transition', 'portrait_beauty_workbench',
        'boundary_light_coordination_workbench', 'release_portability',
        'reference_hair_boundary_registration', 'hair_continuous_illumination', 'boundary_producer_cache',
        'scene_support_continuation', 'continuous_field_contract', 'local_field_domain',
        'frozen_native_paths', 'export_artifact', 'artifact_receipts', 'backend_batch_export',
        'source_copy', 'source_preview_cache', 'quick_progress', 'variant_selection_job', 'job_polling', 'saved_completion', 'job_owner',
        'beauty_reference_overlap', 'beauty_existing_ownership')
MODEL_CHECKS = frozenset(('automatic_native_retouch', 'automatic_scene_completion',
    'reference_local_correspondence_cuda', 'native_cuda_fields', 'native_cuda_matting', 'native_cuda_filters',
    'background_consistency', 'final_material_completion', 'qualified_material_finish',
    'reference_automatic_layers', 'reference_colour_blend', 'reference_diagnostics',
    'reference_dinov3_profile', 'reference_material_palette', 'reference_material_palette_roi', 'reference_colour_fit_roi', 'reference_scene_balance',
    'reference_support', 'reference_native_baseline_cli', 'runtime_project_separation', 'sapiens_current', 'subject_environment_field',
    'portrait_beauty_reference_fit', 'beautygrpo_worker', 'beautygrpo_prepare', 'beautygrpo_prompt_cache', 'background_superresolution'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checks', nargs='*', help='core, all, or check names without _smoke.py')
    parser.add_argument('--model-python', type=Path, help='Installed analysis Python for model-dependent CPU checks')
    args = parser.parse_args()
    names = args.checks or ['core']
    if names == ['core']:
        names = CORE
    elif names == ['all']:
        names = [path.stem.removesuffix('_smoke') for path in sorted(Path(__file__).parent.glob('*_smoke.py'))]
    paths = []
    for name in names:
        path = Path(__file__).parent / (name.removesuffix('.py').removesuffix('_smoke') + '_smoke.py')
        if path.parent != Path(__file__).parent or not path.is_file():
            parser.error('Unknown check: ' + name)
        paths.append(path)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT/'scripts'), str(ROOT/'tests')))
    env.setdefault('QT_QPA_PLATFORM', 'offscreen')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('OMP_NUM_THREADS', '1')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for path in paths:
        print('Running ' + path.name, flush=True)
        python = str(args.model_python.resolve()) if args.model_python and path.stem.removesuffix('_smoke') in MODEL_CHECKS else sys.executable
        result = subprocess.run([python, '-B', '-X', 'utf8', str(path)], cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
