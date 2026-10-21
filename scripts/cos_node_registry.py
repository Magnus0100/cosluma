"""Typed contracts for the existing Cos layer pipeline; no model imports.

Like ComfyUI's input/return declarations, these definitions are shared by
planning and the canvas. This compiler validates the production layer graph;
it does not replace the existing preparation workers with a generic executor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

SCHEMA = 'cosplay_layer_workflow.v1'


@dataclass(frozen=True)
class Port:
    type: str
    domain: str
    coordinates: str
    optional: bool = False


@dataclass(frozen=True)
class Node:
    label: str
    description: str
    stage: str
    position: tuple[int, int]
    action: str
    inputs: dict[str, Port]
    outputs: dict[str, Port]
    implementation: str
    alpha_role: str = 'preserve'
    editable: bool = False
    layer_kinds: tuple[str, ...] = ()


IMAGE = Port('DISPLAY_IMAGE', 'display_linear_sRGB', 'source_canvas')
SUBJECT = Port('SUBJECT_PAIR', 'display_linear_sRGB', 'source_canvas')
REFERENCE = Port('REFERENCE_IMAGE', 'sRGB', 'reference_canvas')
APPEARANCE = Port('APPEARANCE_LAYER', 'display_linear_sRGB', 'source_canvas')
BACKGROUND = Port('BACKGROUND_LAYER', 'display_linear_sRGB', 'source_canvas')
LIGHT = Port('BOUNDARY_LAYER', 'display_linear_sRGB', 'source_canvas')
EFFECT = Port('EFFECT_LAYER', 'display_linear_sRGB', 'source_canvas')
COMPOSITE = Port('COMPOSITE_IMAGE', 'display_linear_sRGB', 'source_canvas')


def optional(port):
    return Port(port.type, port.domain, port.coordinates, True)


NODES = {
    'source': Node('原片与显像', 'RAW / sRGB → 显像 RGB', 'source_import', (0, 0), '导入原片',
        {}, {'image': IMAGE}, 'display_transform.display_linear'),
    'matte': Node('人物与遮挡', '原片人物 RGB + 原生 alpha', 'contact_and_edges', (1, 0), '编辑抠图',
        {'image': IMAGE}, {'subject': SUBJECT}, 'source_foreground.estimate_source_background', 'produce'),
    'beauty': Node('人像美化', 'BeautyGRPO → 外观 RGB', 'portrait_beauty', (1, 1), '准备美化',
        {'image': IMAGE}, {'image': IMAGE}, 'portrait_beauty_layer.apply_for_run', editable=True),
    'reference': Node('场景参考', '已有参考 / 生成场景', 'design_reference', (0, 2), '选择参考',
        {}, {'reference': REFERENCE}, 'register_reference.image_info'),
    'colour': Node('原图与参考仿色', '美化外观 + 原片 alpha → 仿色层', 'reference_colour_match', (2, 0), '编辑仿色',
        {'image': IMAGE, 'subject': SUBJECT, 'reference': REFERENCE}, {'appearance': APPEARANCE},
        'reference_appearance_prepare.freeze', editable=True, layer_kinds=('reference_appearance_stack',)),
    'background': Node('背景与接触', '背景 RGB + 接触阴影', 'background_reconstruction', (2, 2), '编辑背景',
        {'image': IMAGE, 'subject': SUBJECT, 'reference': REFERENCE}, {'background': BACKGROUND},
        'design_reconstruction_pipeline.prepare_scene', editable=True, layer_kinds=('background_design',)),
    'boundary': Node('人物边缘受光', '发丝 / 衣物受光图层', 'contact_and_edges', (3, 0), '准备边缘受光',
        {'image': IMAGE, 'subject': SUBJECT, 'appearance': APPEARANCE, 'background': BACKGROUND,
         'reference': REFERENCE}, {'boundary': LIGHT}, 'boundary_light_runtime.apply_boundary_light',
        editable=True, layer_kinds=('source_boundary_light', 'source_cloth_boundary_light')),
    'effects': Node('前景光效', '独立特效图层', 'independent_effects', (3, 2), '编辑光效',
        {'subject': SUBJECT, 'background': BACKGROUND, 'reference': REFERENCE}, {'effects': EFFECT},
        'foreground_effects.apply_slot', editable=True,
        layer_kinds=('independent_effect', 'rim_light', 'particles', 'atmosphere', 'bloom')),
    'composite': Node('分层合成', '人物 + 背景 + 光效', 'save_and_export', (4, 0), '保存合成',
        {'image': IMAGE, 'subject': SUBJECT, 'beauty': IMAGE, 'appearance': optional(APPEARANCE),
         'background': optional(BACKGROUND), 'boundary': optional(LIGHT), 'effects': optional(EFFECT)},
        {'image': COMPOSITE}, 'design_reconstruction.apply_plan', 'consume'),
    'export': Node('成片与工程', 'PNG / 高位 TIFF / 可编辑工程', 'save_and_export', (5, 0), '导出',
        {'image': COMPOSITE}, {}, 'export_artifact.export_source_run', 'consume'),
}
EDITABLE = frozenset(key for key, node in NODES.items() if node.editable)
LAYER_KINDS = {key: set(node.layer_kinds) for key, node in NODES.items() if node.layer_kinds}

# Explicit sockets, rather than visual ordering, define each dependency.
LINKS = (
    ('source', 'image', 'matte', 'image'), ('source', 'image', 'beauty', 'image'),
    ('beauty', 'image', 'colour', 'image'), ('matte', 'subject', 'colour', 'subject'),
    ('reference', 'reference', 'colour', 'reference'),
    ('source', 'image', 'background', 'image'), ('matte', 'subject', 'background', 'subject'),
    ('reference', 'reference', 'background', 'reference'),
    ('beauty', 'image', 'boundary', 'image'), ('matte', 'subject', 'boundary', 'subject'),
    ('colour', 'appearance', 'boundary', 'appearance'), ('background', 'background', 'boundary', 'background'),
    ('reference', 'reference', 'boundary', 'reference'),
    ('matte', 'subject', 'effects', 'subject'), ('background', 'background', 'effects', 'background'),
    ('reference', 'reference', 'effects', 'reference'),
    ('source', 'image', 'composite', 'image'), ('matte', 'subject', 'composite', 'subject'),
    ('beauty', 'image', 'composite', 'beauty'), ('colour', 'appearance', 'composite', 'appearance'),
    ('background', 'background', 'composite', 'background'), ('boundary', 'boundary', 'composite', 'boundary'),
    ('effects', 'effects', 'composite', 'effects'), ('composite', 'image', 'export', 'image'),
)

def catalogue():
    return {key: asdict(node) for key, node in NODES.items()}


def validate_connections(node_ids, connections, *, registry=None):
    """Check sockets, typed domains, single writers, required inputs and cycles."""
    registry = NODES if registry is None else registry
    ids = list(node_ids)
    if not ids or len(ids) != len(set(ids)) or set(ids) - registry.keys():
        raise ValueError('Unknown or duplicate Cos workflow node')
    incoming = {key: {} for key in ids}
    dependencies = {key: set() for key in ids}
    for edge in connections:
        if not isinstance(edge, dict) or set(edge) != {'from', 'output', 'to', 'input'}:
            raise ValueError('A workflow connection must name both node sockets')
        source, target = edge['from'], edge['to']
        if source not in incoming or target not in incoming:
            raise ValueError('Workflow connection references a missing node')
        output = registry[source].outputs.get(edge['output'])
        port = registry[target].inputs.get(edge['input'])
        if output is None or port is None:
            raise ValueError('Unknown workflow input or output socket')
        if (output.type, output.domain, output.coordinates) != (port.type, port.domain, port.coordinates):
            raise ValueError(f'Workflow type/domain mismatch: {source}.{edge["output"]} → {target}.{edge["input"]}')
        if edge['input'] in incoming[target]:
            raise ValueError('Multiple writers for workflow input: ' + target + '.' + edge['input'])
        incoming[target][edge['input']] = [source, edge['output']]
        dependencies[target].add(source)
    for key in ids:
        missing = {name for name, port in registry[key].inputs.items() if not port.optional} - incoming[key].keys()
        if missing:
            raise ValueError('Disconnected workflow inputs: ' + key + '.' + ', '.join(sorted(missing)))
    order = []
    pending = set(ids)
    while pending:
        ready = sorted(key for key in pending if not dependencies[key] - set(order))
        if not ready:
            raise ValueError('Cos workflow contains a dependency cycle')
        order.extend(ready)
        pending.difference_update(ready)
    return order, incoming


def compile_workflow(mode, controls=None, *, editing_scope=None, visual_review=False, editing_iteration=None, editing_autofinish=None,
                     editing_initialization=None, initial_variant_count=1, shaping_request=None):
    if mode not in {'local_portrait', 'existing_reference', 'create_reference'}:
        raise ValueError('Unknown Cos workflow route')
    if mode=='create_reference' and editing_scope is None:
        raise ValueError('Creation requires the current checked editing scope')
    from scoped_workflow import selected, compile_scoped_workflow, SOURCE_NATIVE_INITIALIZATION
    if mode=='create_reference' and editing_initialization!=SOURCE_NATIVE_INITIALIZATION:
        raise ValueError('Creation requires the current source-native initialization policy')
    if mode!='create_reference' and editing_initialization is not None:
        raise ValueError('Initialization policy requires a creation task')
    if mode=='create_reference' and not selected(mode,editing_scope):
        raise ValueError('Creation requires a checked editing scope')
    if type(visual_review) is not bool:
        raise ValueError('Workflow visual review selection must be boolean')
    if editing_autofinish is not None:
        if (mode!='create_reference' or editing_scope is None or editing_iteration is not None
                or not isinstance(editing_autofinish,dict) or 'maximum_improvements' not in editing_autofinish
                or set(editing_autofinish)-{'maximum_improvements','maximum_zero_image_finishes'}
                or type(editing_autofinish['maximum_improvements']) is not int
                or not 0<=editing_autofinish['maximum_improvements']<=8
                or type(editing_autofinish.get('maximum_zero_image_finishes',0)) is not int
                or not 0<=editing_autofinish.get('maximum_zero_image_finishes',0)<=1
                or (editing_autofinish.get('maximum_zero_image_finishes',0)
                    and (not editing_autofinish['maximum_improvements'] or not visual_review))
                or not selected(mode,editing_scope)):
            raise ValueError('Automatic finishing needs an explicit scoped creation policy')
        if editing_autofinish['maximum_improvements'] and not visual_review:
            raise ValueError('Automatic finishing requires independent review')
    if shaping_request is not None and (editing_iteration is not None or editing_autofinish is not None
            or not selected(mode,editing_scope,shaping_request=shaping_request)):
        raise ValueError('Shaping requires its independent checked existing-reference workflow')
    if editing_iteration is not None and not selected(mode,editing_scope,editing_iteration):
        raise ValueError('Editing iteration requires an explicit scoped existing-reference workflow')
    if selected(mode,editing_scope,editing_iteration,shaping_request=shaping_request):
        if controls:
            raise ValueError('Scoped appearance workflow is read-only; legacy layer controls do not apply')
        return compile_scoped_workflow(mode,editing_scope,visual_review,editing_iteration=editing_iteration,
            editing_autofinish=editing_autofinish,editing_initialization=editing_initialization,
            initial_variant_count=initial_variant_count,shaping_request=shaping_request)
    controls = controls or {}
    if not isinstance(controls, dict) or set(controls) - EDITABLE:
        raise ValueError('Unknown editable Cos workflow node')
    for key, value in controls.items():
        if (not isinstance(value, dict) or set(value) != {'enabled', 'strength'}
                or type(value['enabled']) is not bool
                or isinstance(value['strength'], bool) or not isinstance(value['strength'], (int, float))
                or not math.isfinite(value['strength']) or not 0 <= value['strength'] <= 1):
            raise ValueError('Invalid workflow layer controls: ' + key)
    # Disabled prepared layers remain typed no-ops, not disconnected inputs.
    if mode == 'local_portrait':
        ids = ['source', 'matte', 'beauty', 'composite', 'export']
        links = LINKS
    else:
        ids = list(NODES)
        links = LINKS
    edges = [{'from': a, 'output': b, 'to': c, 'input': d} for a, b, c, d in links if a in ids and c in ids]
    order, incoming = validate_connections(ids, edges)
    compiled = {'schema': SCHEMA, 'mode': mode, 'scope': 'production_layer_dependencies',
        'executor': 'existing_production_workers_and_renderer', 'editable_connections': False,
        'order': order, 'connections': edges,
        'nodes': [{'id': key, 'implementation': NODES[key].implementation,
            'inputs': incoming[key], 'outputs': {name: asdict(port) for name, port in NODES[key].outputs.items()},
            'alpha_role': NODES[key].alpha_role, 'controls': controls.get(key, {})} for key in order]}
    from cos_workflow_cache import attach_cache_contract
    compiled = attach_cache_contract(compiled)
    compiled['graph_id'] = hashlib.sha256(json.dumps(compiled, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()
    return compiled
