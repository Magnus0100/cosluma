"""Reject retired request fields before any job, model or renderer runs."""

RETIRED_OPTIONS = frozenset(('appearance_strength', 'appearance_policy',
    'appearance_subject_strength', 'appearance_background_strength',
    'alignment_revision', 'alpha_revision', 'controls_only', 'floor_repair'))


def validate_current_request(value):
    if not isinstance(value, dict):
        raise ValueError('Current request must be an object')
    forbidden = RETIRED_OPTIONS.intersection(value)
    if forbidden:
        raise ValueError('Retired request fields are unsupported: '+', '.join(sorted(forbidden)))
    options = value.get('options')
    if isinstance(options, dict):
        validate_current_request(options)
    inputs = value.get('reference_pipeline_inputs')
    if isinstance(inputs, dict) and 'floor_repair' in inputs:
        raise ValueError('Photo-specific floor repair inputs are retired')
    revision = value.get('appearance_revision')
    if isinstance(revision, dict) and 'parameters' in revision:
        raise ValueError('Current appearance revisions use params')
