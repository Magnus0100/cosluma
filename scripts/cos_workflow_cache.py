"""ComfyUI-style input-signature caching for the Cos layer graph.

This module only computes identities and invalidation scope. Existing workers
remain the owners of frozen pixel assets and their byte-level validation.
"""
from __future__ import annotations

import hashlib
import json

SCHEMA = 'cosplay_workflow_cache.v1'
SCOPES = {
    'source': 'decode', 'matte': 'subject_analysis', 'beauty': 'portrait_layer',
    'reference': 'reference_input', 'colour': 'appearance_layer',
    'background': 'background_layer', 'boundary': 'boundary_layer',
    'effects': 'effects_layer', 'composite': 'composite', 'export': 'export',
}
PRODUCER_DEPENDENCIES = {
    'source': ('display_transform.py', 'raw_support.py', 'float_appearance.py'),
    'matte': ('source_foreground.py', 'native_source_baseline.py', 'local_semantics_worker.py'),
    'beauty': ('portrait_beauty_layer.py', 'beautygrpo_prepare.py', 'beautygrpo_worker.py', 'portrait_regions_worker.py'),
    'reference': ('register_reference.py', 'generation_prompt.py', 'generation_design.py', 'creative_direction.py'),
    'background': ('design_reconstruction_pipeline.py', 'scene_design.py', 'background_superresolution_worker.py'),
    'boundary': ('boundary_light_runtime.py', 'boundary_light_worker.py', 'boundary_light_coordination_worker.py',
                 'cloth_boundary_runtime.py', 'cloth_boundary_worker.py'),
    'effects': ('foreground_effects.py', 'foreground_effects_runtime.py'),
    'composite': ('design_reconstruction.py', 'reference_appearance_stack.py'),
    'export': ('export_artifact.py', 'portable_project.py', 'raw_support.py'),
}


def producer_identity_for_node(node_id, implementation):
    """Plan signatures include real producer code, never just a function name."""
    from runtime_resources import code_path
    from asset_identity import sha256_file
    files = set(PRODUCER_DEPENDENCIES.get(node_id, ()))
    files.add(implementation.split('.')[0] + '.py')
    recipe = None
    if node_id == 'colour':
        from reference_pipeline import CODE_DEPENDENCIES, RECIPE
        files.update(CODE_DEPENDENCIES)
        recipe = RECIPE
    if node_id == 'boundary':
        # Use the worker's checked closure, so planning and actual asset reuse
        # invalidate together when transfer math or its inputs change.
        from boundary_light_worker import producer_code
        files.update(producer_code('source_boundary_light'))
        files.update(producer_code('source_cloth_boundary_light'))
    code = {name: sha256_file(path) if (path := code_path(name)).is_file() else None
            for name in sorted(files)}
    return {'recipe': recipe, 'code': code,
            'availability': 'complete' if all(code.values()) else 'incomplete'}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def _nodes(compiled):
    return {row['id']: row for row in compiled.get('nodes') or []}


def attach_cache_contract(compiled):
    """Attach deterministic keys and downstream invalidation to a compiled graph."""
    result = json.loads(json.dumps(compiled, ensure_ascii=False))
    rows = _nodes(result)
    keys = {}
    for node_id in result.get('order') or []:
        row = rows[node_id]
        ancestors = sorted(
            (edge['input'], edge['output'], edge['from'], keys[edge['from']])
            for edge in result.get('connections') or []
            if edge['to'] == node_id
        )
        keys[node_id] = _digest({
            'node_id': node_id,
            'implementation': row['implementation'],
            'producer': producer_identity_for_node(node_id, row['implementation']),
            'controls': row.get('controls') or {},
            'alpha_role': row.get('alpha_role'),
            'ancestors': ancestors,
        })
    downstream = {node_id: set() for node_id in rows}
    for edge in result.get('connections') or []:
        downstream.setdefault(edge['from'], set()).add(edge['to'])

    def affected(changed):
        seen = set(changed)
        pending = list(changed)
        while pending:
            current = pending.pop()
            for child in sorted(downstream.get(current, ())):
                if child not in seen:
                    seen.add(child); pending.append(child)
        return [node_id for node_id in result.get('order') or [] if node_id in seen]

    result['cache'] = {
        'schema': SCHEMA,
        'strategy': 'producer_code_controls_and_ordered_upstream_input_signature',
        'layout_ignored': True,
        'node_keys': keys,
        'scope': {node_id: SCOPES.get(node_id, 'layer') for node_id in keys},
        'downstream': {node_id: [child for child in result.get('order') or []
                                 if child in downstream.get(node_id, set())]
                       for node_id in keys},
        'affected_by': {node_id: affected([node_id]) for node_id in keys},
        'worker_owns_asset_validation': True,
        'signature_reuse_is_not_a_verified_pixel_cache_hit': True,
    }
    return result


def compare(previous, current):
    """Return the smallest re-execution scope between two compiled graphs."""
    old = (previous or {}).get('cache') or {}
    new = (current or {}).get('cache') or {}
    old_keys, new_keys = old.get('node_keys') or {}, new.get('node_keys') or {}
    order = (current or {}).get('order') or []
    raw_changed = {node_id for node_id in order if old_keys.get(node_id) != new_keys.get(node_id)}
    parents = {node_id: set() for node_id in order}
    for edge in (current or {}).get('connections') or []:
        if edge['to'] in parents:
            parents[edge['to']].add(edge['from'])
    # A downstream signature changes as a consequence of an upstream change;
    # report only the roots that changed directly.
    changed = [node_id for node_id in order
               if node_id in raw_changed and not parents[node_id].intersection(raw_changed)]
    invalidated = set()
    for node_id in changed:
        invalidated.update((new.get('affected_by') or {}).get(node_id, [node_id]))
    return {
        'schema': SCHEMA,
        'changed': changed,
        'invalidated': [node_id for node_id in order if node_id in invalidated],
        'reused': [node_id for node_id in order if node_id not in invalidated and node_id in old_keys],
        'execution_order': [node_id for node_id in order if node_id in invalidated],
        'same_graph': bool(previous and current and previous.get('graph_id') == current.get('graph_id')),
    }
