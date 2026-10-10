"""Run current offline checks with production modules on the import path."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ('model_setup', 'embedded_backend', 'creative_task', 'task_model_preflight',
        'creative_task_ui', 'boundary_light_coordination_workbench', 'release_portability',
        'frozen_native_paths', 'export_artifact', 'source_copy')
MODEL_CHECKS = frozenset(('automatic_native_retouch', 'automatic_scene_completion',
    'background_consistency', 'final_material_completion', 'qualified_material_finish',
    'reference_automatic_layers', 'reference_colour_blend', 'reference_diagnostics',
    'reference_dinov3_profile', 'reference_material_palette', 'reference_scene_balance',
    'reference_support', 'runtime_project_separation', 'sapiens_current', 'subject_environment_field'))


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
