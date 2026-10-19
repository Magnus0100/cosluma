"""Small real current v2 assets for CPU status, renderer and portable tests."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from reference_appearance_controls import prepare_owned_controls
from reference_support import build_reference_support
from raw_support import RGB_DECODE_VERSION
from float_appearance import srgb_to_linear


def prepare_fixture(root, *, native=(24, 20)):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    w, h = native

    def record(path, **extra):
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **extra}

    def asset(name, values, **extra):
        path = root/(name+'.npz')
        np.savez_compressed(path, values=np.asarray(values, np.float32))
        return record(path, size=list(native), coordinate_space='source_canvas',
            working_space='display_linear_sRGB', sampling='bilinear', **extra)

    source = root/'source.png'
    reference = root/'reference.png'
    Image.new('RGB', native, '#657585').save(source)
    Image.new('RGB', native, '#556575').save(reference)
    fp = asset('fp', srgb_to_linear(np.asarray(Image.open(source), np.float32)/255))
    alpha = asset('alpha', np.ones((h, w), np.float32))
    old = asset('old', np.full((h, w, 3), .1, np.float32))
    background = asset('background', np.full((h, w, 3), .2, np.float32),
        representation='sampling_validity_associated_RGB', validity_associated=True)
    delta = asset('delta', np.full((h, w, 3), .015, np.float32), representation='premultiplied_delta')
    source_record, reference_record = record(source), record(reference)
    params = {'schema':'reference_appearance_stack.v2', 'working_domain':'display_linear_sRGB',
        'source_sha256':source_record['sha256'], 'reference_sha256':reference_record['sha256'],
        'source_fp_sha256':fp['sha256'], 'alpha_sha256':alpha['sha256'], 'canvas_size':list(native),
        'prepared_stages':['reference_colour_match','reference_colour_blend','reference_scene_balance','edge_donor_extension'],
        'freeze_mode':'native_premultiplied_delta'}
    layer = {'id':'reference_appearance_stack','kind':'reference_appearance_stack','enabled':True,
        'strength':1.,'params':params,'assets':{'automatic_delta':delta}}
    plan = {'schema_version':'design_reconstruction_plan.v1','enabled':True,
        'working_domain':'display_linear_sRGB','source_sha256':source_record['sha256'],
        'candidate_sha256':reference_record['sha256'],
        'canvas':{'source_oriented_size':list(native),'source_proxy_size':list(native)},
        'mask_roles':{'foreground_alpha':alpha},
        'source_baseline':{'schema':'appearance_source_baseline.v1','working_domain':'display_linear_sRGB',
            'source_display_applied_once':True,
            'source_precision':'raster8_srgb','decode_version':RGB_DECODE_VERSION,
            'source_display':{'schema_version':'display_transform.v1','method':'clip_sdr',
                'input_domain':'display_linear_sRGB','output_space':'sRGB','gamut_mapping':'encoding_clip_only'},
            'assets':{'foreground_premultiplied':fp,'foreground_alpha':alpha,'old_environment':old}},
        'layers':[layer,{'id':'background_design','kind':'background_design','enabled':True,'strength':1.,
            'params':{},'assets':{'foreground_premultiplied':fp,'old_environment':old,
                'background':background,'validity':alpha}}],
        'execution':{'source_reference':'paired_native_baseline'}}
    run = {'execution_strategy':'layered_reference_transfer','task_intent':{'mode':'existing_reference'},'kind':'single_input_run','input_mode':'single_source','root':str(root),
        'run_id':'current-stack-fixture','sample':'current-stack-fixture','revision':1,
        'source':source_record,'reference':reference_record,
        'options':{'render_mode':'native_source_design','appearance_revision':{'design_reconstruction_plan':plan}}}
    plan = prepare_owned_controls(plan, root/'controls', run=run)
    layer = plan['layers'][0]
    binding = {key:params[key] for key in ('source_sha256','reference_sha256','source_fp_sha256','alpha_sha256','canvas_size')}
    binding['automatic_delta_sha256'] = delta['sha256']
    policy = {'support_policy':'reference_support.v1','operators':{},'observations':{}}
    receipt = {'support_policy':'reference_support.v1','regions':{}}
    support = build_reference_support(policy, receipt, receipt, binding=binding)
    path = root/'support.json'
    path.write_text(json.dumps(support, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    layer['params']['reference_support'] = support
    layer['assets']['reference_support'] = record(path)
    run['options']['appearance_revision']['design_reconstruction_plan'] = plan
    return run


def bind_local_domain(field, size, *, mask=None):
    """Attach a fixed source component to an analytic local-colour fixture."""
    mask = np.ones(size[::-1], bool) if mask is None else np.asarray(mask, bool)
    assert mask.shape == tuple(size[::-1])
    packed = np.packbits(mask).tobytes()
    field['application_policy'] = 'source_material_continuation.v1'
    field['application_domain'] = {'schema':'source_component_support.v1',
        'canvas_size':list(size), 'box':[0,0,*size], 'bits':packed.hex(),
        'sha256':hashlib.sha256(packed).hexdigest()}
    return field
