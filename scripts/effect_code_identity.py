"""Independent FX producer/consumer/review implementation identities.

The installed script sources are packaged alongside modules. A relocated
project resolves the current implementation independently of its data root.
"""
from __future__ import annotations

from runtime_resources import code_path
from asset_identity import sha256_file

FILES = ('foreground_effects.py', 'foreground_effects_runtime.py', 'effect_conditioning.py',
    'effect_layout.py', 'automatic_foreground_effects.py', 'automatic_foreground_effects_review.py',
    'effect_code_identity.py', 'effect_sampling.py', 'installed_resources.py')


def current_code_identity():
    result = {}
    for name in FILES:
        path = code_path(name)
        if not path.is_file():
            raise RuntimeError('Current FX implementation source is missing: '+name)
        result[name] = sha256_file(path)
    return {'schema': 'independent_effect_code_identity.v1', 'files': result}


def require_current_code(snapshot):
    if snapshot != current_code_identity():
        raise ValueError('Independent effect producer/consumer/review code changed; frozen core T/B/S remains separate')
