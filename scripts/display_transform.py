"""Separate source preparation into display-linear RGB from final SDR encoding."""
from __future__ import annotations

import copy
import numpy as np

_LUMA = np.asarray([.2126, .7152, .0722], np.float32)
DEFAULT = {'schema_version': 'display_transform.v1', 'method': 'luminance_soft_shoulder',
           'shoulder_start': .8, 'output_space': 'sRGB',
           'gamut_mapping': 'luminance_preserving_desaturation'}
SOURCE_SCHEMA = 'source_preparation_transform.v1'
SOURCE_SCOPE = 'source_native_before_reference_TBS'


def source_transform_for_precision(source_precision):
    """A producer stamps this explicit transform before preparing a native pair."""
    if source_precision == 'raw16_linear':
        return {'schema_version': SOURCE_SCHEMA, 'method': 'luminance_soft_shoulder',
            'shoulder_start': .8, 'input_domain': 'scene_linear_sRGB',
            'output_domain': 'display_linear_sRGB', 'application_scope': SOURCE_SCOPE,
            'gamut_mapping': 'luminance_preserving_desaturation'}
    if source_precision == 'raster8_srgb':
        return {'schema_version': SOURCE_SCHEMA, 'method': 'clip_sdr',
            'input_domain': 'display_linear_sRGB', 'output_domain': 'display_linear_sRGB',
            'application_scope': SOURCE_SCOPE, 'gamut_mapping': 'encoding_clip_only'}
    raise ValueError('Source preparation requires an explicit current RAW or raster precision')


def validate_source_transform(transform, source_precision):
    expected = source_transform_for_precision(source_precision)
    if not isinstance(transform, dict) or set(transform) != set(expected):
        raise ValueError('Source preparation requires its typed before-TBS transform')
    for key, value in expected.items():
        if key != 'shoulder_start' and transform[key] != value:
            raise ValueError('Source preparation transform has an incompatible scope or domain: '+key)
    if 'shoulder_start' in expected:
        shoulder = transform['shoulder_start']
        if isinstance(shoulder, bool) or not isinstance(shoulder, (int, float)) or not np.isfinite(shoulder) or not 0 < shoulder < 1:
            raise ValueError('Source preparation shoulder must be finite and between zero and one')
    return copy.deepcopy(transform)


def source_transform_for_run(run):
    """Read D for a fresh source decode; never apply D to a prepared Fp/B pair."""
    from raw_support import RGB_DECODE_VERSION
    options = run.get('options') or {}
    plan = (options.get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
    baseline = plan.get('source_baseline') or options.get('source_only_plan') or {}
    if (baseline.get('source_sha256') != (run.get('source') or {}).get('sha256')
            or baseline.get('decode_version') != RGB_DECODE_VERSION):
        raise ValueError('Source preparation baseline must bind the current source and decoder')
    transform = validate_source_transform(baseline.get('source_display'), baseline.get('source_precision'))
    schema, domain = baseline.get('schema'), baseline.get('working_domain')
    if schema == 'source_preparation_baseline.v1':
        if (plan.get('schema_version') != 'design_reconstruction_plan.v1'
                or domain != transform['input_domain'] or plan.get('working_domain') != domain
                or baseline.get('source_display_applied_once') is not False):
            raise ValueError('Unprepared source baseline must declare its input domain and unconsumed transform')
    elif schema in {'appearance_source_baseline.v1', 'source_only_display_plan.v1'}:
        if (domain != 'display_linear_sRGB' or baseline.get('source_display_applied_once') is not True
                or (schema == 'appearance_source_baseline.v1' and plan.get('working_domain') != domain)):
            raise ValueError('Prepared source baseline must declare its once-consumed display domain')
    else:
        raise ValueError('Missing current explicit source preparation baseline schema')
    return transform


def plan_for_run(run):
    revision = (run.get('options') or {}).get('appearance_revision') or {}
    plan = revision.get('design_reconstruction_plan') or {}
    source_only=(run.get('options') or {}).get('source_only_plan') or {}
    baseline = plan.get('source_baseline') or {}
    consumed = (baseline.get('source_display_applied_once') is True
        or (plan.get('execution') or {}).get('source_display_applied_at_native_prepare') is True)
    paired=(plan.get('schema_version')=='design_reconstruction_plan.v1'
            and plan.get('working_domain')=='display_linear_sRGB'
            and baseline.get('schema') == 'appearance_source_baseline.v1'
            and baseline.get('working_domain') == 'display_linear_sRGB'
            and consumed and baseline.get('source_display_applied_once') is not False)
    local=(source_only.get('schema')=='source_only_display_plan.v1'
           and source_only.get('working_domain')=='display_linear_sRGB'
           and source_only.get('source_sha256')==(run.get('source') or {}).get('sha256'))
    if not paired and not local:
        raise ValueError('Only an explicit current display-linear rendering plan can be encoded')
    return {'schema_version': 'display_transform.v1', 'method': 'clip_sdr',
            'input_domain': 'display_linear_sRGB', 'output_space': 'sRGB',
            'gamut_mapping': 'encoding_clip_only'}


def display_linear(values, plan=None):
    """Preserve low/mid luminance, compress scene highlights once at output.

    This is a photographic shoulder, not a measured camera/display model. The
    transform is local and deterministic, so proxy/full use the same settings.
    """
    plan = plan or DEFAULT
    rgb = np.maximum(np.asarray(values, np.float32), 0)
    if plan['method'] == 'clip_sdr':
        return np.clip(rgb, 0, 1)
    if plan['method'] != 'luminance_soft_shoulder':
        raise ValueError(f"Unknown display transform: {plan['method']}")
    shoulder = float(plan.get('shoulder_start', .8))
    if not 0 < shoulder < 1:
        raise ValueError('Display shoulder must lie between zero and one')
    y = np.sum(rgb*_LUMA, axis=-1, keepdims=True)
    mapped = np.where(y <= shoulder, y,
        shoulder + (1-shoulder)*(-np.expm1(-np.maximum(y-shoulder, 0)/(1-shoulder))))
    rgb *= np.divide(mapped, y, out=np.ones_like(y), where=y > 0)
    # Desaturate only display-gamut excursions; scene data remains untouched.
    peak = rgb.max(axis=-1, keepdims=True)
    amount = np.minimum(1, np.divide(1-mapped, peak-mapped,
        out=np.ones_like(mapped), where=peak > mapped))
    return np.clip(mapped+(rgb-mapped)*amount, 0, 1).astype(np.float32)


def encode_display(values, plan=None):
    rgb = display_linear(values, plan)
    return np.where(rgb <= .0031308, rgb*12.92,
                    1.055*np.power(rgb, 1/2.4)-.055).astype(np.float32)
