"""One component-replacement contract for preview patches and saved revisions."""
from __future__ import annotations
import copy

FLAT_REVISION_KEYS = ('design_reconstruction_plan','source_render_plan','source_render_plan_origin',
    'source_color_plan','source_color_plan_origin',
    'params','regions','display_transform','receiver_intent','receiver_shadow_plan','portrait_beauty','workflow_graph')


def appearance_revision_patch(body):
    """Read the current wrapper and explicit component patches."""
    if 'parameters' in body:
        raise ValueError('Use params in the current appearance contract')
    patch = None
    options = body.get('options') or {}
    for incoming in (options.get('appearance_revision'),body.get('appearance_revision')):
        if isinstance(incoming,dict):
            patch = merge_appearance_revision(patch,incoming)
    flat = {key:body[key] for key in FLAT_REVISION_KEYS if key in body}
    if flat:
        patch = merge_appearance_revision(patch,flat)
    return patch


def merge_appearance_revision(inherited, incoming):
    """Omitted components inherit; provided components replace, including empty/off.

    Provided component dictionaries replace their saved snapshots.
    """
    result = copy.deepcopy(inherited or {})
    incoming = incoming or {}
    if 'parameters' in result or 'parameters' in incoming:
        raise ValueError('Unsupported appearance parameters alias; use params')
    for key,value in incoming.items():
        result[key] = copy.deepcopy(value)
    return result
