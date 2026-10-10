"""The current CPU material solver; retired profiles are rejected."""
DEFAULT_PROFILE = 'owner_adaptive_multiscale_skin_v2'


def solver_configuration(profile=DEFAULT_PROFILE):
    if profile != DEFAULT_PROFILE:
        raise ValueError('Only owner_adaptive_multiscale_skin_v2 is supported')
    return {'profile': profile, 'module': 'reference_material_targets_v2',
            'dependencies': ('reference_material_targets_v2.py', 'reference_material_targets.py')}
