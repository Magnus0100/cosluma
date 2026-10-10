"""Current native-pixel ROI using frozen source fields and production consumers.

RGB arithmetic is bounded to source-coordinate ROI plus finite filter halos.
Nonlocal receiver/clone/colour statistics explicitly use the existing full
renderer fallback; a reduced proxy is never substituted for native pixels.
"""
from __future__ import annotations
import math
import numpy as np
from PIL import Image,ImageChops,ImageDraw,ImageFilter
from scipy.ndimage import gaussian_filter

RECIPE='native-source-coordinate-roi-v1'


class ROIUnavailable(Exception):
    """A current operation requires the full native donor/context domain."""


def _operations(run,branch):
    options=run.get('options',{});revision=options.get('appearance_revision',{})
    params=revision.get('params',{});regions=revision.get('regions')
    if 'parameters' in revision:raise ValueError('Current appearance controls require params')
    if regions is None:
        regions={k:params[k] for k in ('global','subject','background','local') if k in params}
    operations=[]
    if branch=='composite' and 'global' not in regions and any(k in params for k in
            ('exposure','contrast','temperature','tint','saturation','highlights','shadows','sharpness')):
        operations.append(params)
    for name,destination in [('global','composite'),('subject','foreground'),('background','background')]:
        if branch==destination and isinstance(regions.get(name),dict):operations.append(regions[name])
    for spec in regions.get('local',[]):
        if any(key in spec for key in ('region_scope','region')):raise ValueError('Current local controls require scope and rect')
        scope=spec.get('scope','global')
        if branch=={'subject':'foreground','background':'background'}.get(scope,'composite'):operations.append(spec)
    return operations


def _appearance(values,run,branch,box,native_size,premultiplied_alpha=None):
    from float_appearance import _float_tonal
    width,height=native_size;l,t,r,b=box
    for spec in _operations(run,branch):
        amount=float(np.clip(spec.get('strength',1),0,1))
        if not amount:continue
        rect=spec.get('rect')
        if rect is None:weight=np.float32(amount)
        else:
            # A native L selection is cheap and preserves the full renderer's
            # rounding and PIL feather boundary exactly, without full RGB.
            mask=Image.new('L',native_size,0);x0,y0,x1,y1=map(float,rect)
            ImageDraw.Draw(mask).rectangle((round(x0*width),round(y0*height),round(x1*width),round(y1*height)),fill=255)
            feather=float(np.clip(spec.get('feather',0),0,1))
            if feather:mask=mask.filter(ImageFilter.GaussianBlur(max(.5,min(native_size)*feather*.025)))
            weight=np.asarray(mask.crop(box),np.float32)[...,None]*np.float32(amount/255)
        values=values+(_float_tonal(values,spec,premultiplied_alpha)-values)*weight
    return values


def _lights(values,plan,alpha,box,native_size):
    from source_relight import normalize_source_render_plan
    plan=normalize_source_render_plan(plan)
    if not plan['enabled']:return values
    width,height=native_size;l,t,r,b=box;aspect=width/max(1,height)
    yy=np.arange(t,b,dtype=np.float32)[:,None]/max(1,height-1)
    xx=np.arange(l,r,dtype=np.float32)[None,:]/max(1,width-1)
    for field in plan['light_fields']:
        if not field['enabled'] or not field['opacity']:continue
        if field['scope']!='all' and alpha is None:continue
        cx,cy=field['center'];rx,ry=field['radius'];angle=math.radians(field['rotation_deg'])
        cosine,sine=math.cos(angle),math.sin(angle);dx,dy=(xx-cx)*aspect,yy-cy
        u=(cosine*dx+sine*dy)/(rx*aspect);v=(-sine*dx+cosine*dy)/ry
        weight=np.exp(-2*(u*u+v*v))*np.float32(field['opacity'])
        if field['scope']=='subject':weight*=alpha
        elif field['scope']=='background':weight*=1-alpha
        temperature,tint=field['temperature'],field['tint']
        stops=np.asarray([.35*temperature+.12*tint,-.12*tint,-.35*temperature+.12*tint],np.float32)+np.float32(field['exposure_ev'])
        values=values*np.exp2(weight[...,None]*stops)
        if field['saturation']:
            y=values[...,0:1]*.2126+values[...,1:2]*.7152+values[...,2:3]*.0722
            values=y+(values-y)*(1+field['saturation']*weight[...,None])
    return values


def _manual_pointwise(values,run,branch,premultiplied_alpha=None,*,box=None,canvas_size=None):
    manual=run.get('options',{}).get('manual_layers') or {}
    curves=manual.get('tone_curves')
    if isinstance(curves,dict):
        from tone_curves import apply_tone_curves
        values=apply_tone_curves(values,curves,branch=branch,subject_alpha=None,premultiplied_alpha=premultiplied_alpha)
    hsl=manual.get('hsl_adjust')
    if isinstance(hsl,dict):
        from hsl_adjust import apply_hsl_adjust
        values=apply_hsl_adjust(values,hsl,branch=branch,subject_alpha=None,premultiplied_alpha=premultiplied_alpha)
    brush=manual.get('local_colour_brush')
    if isinstance(brush,dict) and brush.get('strokes'):
        from local_colour_brush import apply_local_colour_brush
        values=apply_local_colour_brush(values,brush,branch=branch,subject_alpha=premultiplied_alpha,
            premultiplied_alpha=premultiplied_alpha,canvas_size=canvas_size,
            origin=(box[0],box[1]) if box else (0,0))
    return values


def render_native_roi(run,box):
    """Return current linear RGB at 1 native pixel per output pixel, plus receipt."""
    from design_reconstruction import (plan_for_run,sample_asset,_manual_light_plans,
                                       validate_current_native_plan,render_mode_for_run)
    from float_appearance import _detail_tiled
    plan=plan_for_run(run);native=tuple(map(int,plan['canvas']['source_oriented_size']))
    render_mode_for_run(run);validate_current_native_plan(plan)
    for layer in plan.get('layers',[]):
        if layer.get('kind') == 'contact_shadow':
            from design_scene_assets import validate_contact_receiver_contract
            validate_contact_receiver_contract(layer,
                source_sha256=(run.get('source') or {}).get('sha256'),
                candidate_sha256=(run.get('reference') or {}).get('sha256'),
                source_alpha_sha256=plan['mask_roles']['foreground_alpha']['sha256'])
        if layer.get('kind') == 'authored_receiver_shadow' and layer.get('enabled'):
            from authored_receiver_shadow import validate_receiver_shadow_layer
            validate_receiver_shadow_layer(layer,run,plan)
        if layer.get('kind') == 'independent_effect' and layer.get('enabled'):
            from foreground_effects import validate_layer
            validate_layer(layer,run,plan)
    if not plan.get('enabled'):raise ROIUnavailable('design disabled: preserve full native source/manual replay semantics')
    active=[x for x in plan['layers'] if plan.get('enabled') and x.get('enabled') and float(x.get('strength',1))]
    manual=run.get('options',{}).get('manual_layers') or {}
    if any(isinstance(x,dict) and (x.get('enabled') or x.get('strokes')) for k,x in manual.items()
            if k not in ('tone_curves','hsl_adjust','local_colour_brush')):
        raise ROIUnavailable('manual layers need current nonlocal donors or native procedural masks')
    revision=run.get('options',{}).get('appearance_revision') or {}
    from source_color import source_color_plan_for_run
    color=source_color_plan_for_run(run)
    if color.get('enabled') and revision.get('source_color_plan_origin') in {
            'manual','model_with_user_edits','local_model_with_user_edits','local_automatic_with_user_edits','user'}:
        raise ROIUnavailable('selected-colour correction requires whole-selection current statistics')
    admitted={'reference_appearance_stack','source_subject_detail','source_environment_response','source_boundary_light','source_cloth_boundary_light','background_design','contact_shadow','authored_receiver_shadow','rim_light','atmosphere','particles','bloom','independent_effect'}
    for layer in active:
        if layer['kind'] not in admitted:raise ROIUnavailable('nonlocal operator: '+layer['kind'])
        if 'target_look' in layer.get('assets',{}):raise ROIUnavailable('dynamic skin look needs whole-material current statistics')
    halo=3 if run.get('options',{}).get('denoise_strength',0) else 0
    halo+=3*sum(bool(x.get('sharpness')) for branch in ('foreground','background','composite') for x in _operations(run,branch))
    bloom=[]
    for layer in active:
        if layer['kind']=='bloom' and layer.get('params',{}).get('source')=='visible_composite':
            sigma=max(2,min(native)*float(layer['params'].get('radius_fraction',.015)))
            halo+=int(np.ceil(4*sigma));bloom.append((layer,sigma))
    l,t,r,b=map(int,box);hbox=(max(0,l-halo),max(0,t-halo),min(native[0],r+halo),min(native[1],b+halo))
    hl,ht,hr,hb=hbox;core=(slice(t-ht,b-ht),slice(l-hl,r-hl))
    def sample(asset,key='values',sampling=None):return sample_asset(asset,key,hbox,native,sampling=sampling)
    background_layer=next(x for x in plan['layers'] if x['kind']=='background_design');assets=background_layer['assets']
    original=plan['source_baseline']['assets']
    binding=dict(source_sha256=run['source']['sha256'],
        foreground_sha256=original['foreground_premultiplied']['sha256'],
        alpha_sha256=original['foreground_alpha']['sha256'],native_size=list(native))
    original_alpha=sample(original['foreground_alpha'])
    alpha=sample(plan['mask_roles']['foreground_alpha']);foreground=sample(assets['foreground_premultiplied'])
    background=sample(assets['old_environment']);automatic_alpha=alpha
    applicability=plan.get('mask_roles',{}).get('edited_edge_light_applicability');unsupported=None
    if applicability is not None:
        fraction=np.clip(sample(applicability),0,1);retained=foreground*fraction[...,None]
        unsupported=foreground-retained;foreground=retained;automatic_alpha=np.clip(alpha*fraction,0,alpha)
    foreground=_detail_tiled(foreground,denoise=float(run.get('options',{}).get('denoise_strength',0)),coverage=automatic_alpha)
    if unsupported is not None:
        foreground += unsupported
    records=[]
    light_plans=_manual_light_plans(run)
    for layer in active:
        if layer['kind']!='reference_appearance_stack':continue
        from reference_appearance_stack import apply_stack
        from design_reconstruction import reference_stack_execution_weights
        foreground, appearance_report=apply_stack(foreground,layer,plan,run,box=hbox,canvas_size=native)
        bank = reference_stack_execution_weights(layer)
        records.append({'id':layer['id'],'kind':layer['kind'],'applied':bool(bank['sampled_asset_roles']), **bank,
            'enabled':True,'strength':float(layer.get('strength',1)),
            'assets':layer.get('assets',{}),'params':layer.get('params',{}),**appearance_report,
            'execution_stage':'native_frozen_reference_appearance_before_composition',
            'automatic_coverage_consumed_once':True})
    for layer in active:
        if layer['kind'] != 'source_subject_detail':continue
        from native_subject_detail_runtime import apply_detail
        foreground,detail_execution=apply_detail(foreground,layer,plan,run,box=hbox,canvas_size=native)
        records.append({'id':layer['id'],'kind':layer['kind'],'enabled':True,
            'strength':float(layer.get('strength',1)), 'assets':layer.get('assets',{}),**detail_execution})
    for layer in active:
        if layer['kind'] != 'source_environment_response':continue
        from subject_environment_runtime import apply_environment
        foreground,environment_execution=apply_environment(foreground,layer,plan,run,box=hbox,canvas_size=native)
        records.append({'id':layer['id'],'kind':layer['kind'],'enabled':True,
            'strength':float(layer.get('strength',1)), 'assets':layer.get('assets',{}),**environment_execution})
    for layer in active:
        if layer['kind'] != 'source_boundary_light':continue
        from boundary_light_runtime import apply_boundary_light
        foreground,boundary_execution=apply_boundary_light(foreground,layer,plan,run,box=hbox,canvas_size=native)
        records.append({'id':layer['id'],'kind':layer['kind'],'enabled':True,
            'strength':float(layer.get('strength',1)), 'assets':layer.get('assets',{}),**boundary_execution})
    for layer in active:
        if layer['kind'] != 'source_cloth_boundary_light':continue
        from cloth_boundary_runtime import apply_cloth_boundary_light
        foreground,cloth_execution=apply_cloth_boundary_light(foreground,layer,plan,run,box=hbox,canvas_size=native)
        records.append({'id':layer['id'],'kind':layer['kind'],'enabled':True,
            'strength':float(layer.get('strength',1)), 'assets':layer.get('assets',{}),**cloth_execution})
    foreground=_lights(foreground,light_plans['foreground'],None,hbox,native)
    foreground=_appearance(foreground,run,'foreground',hbox,native,premultiplied_alpha=alpha)
    foreground=_manual_pointwise(foreground,run,'foreground',premultiplied_alpha=alpha,box=hbox,canvas_size=native)
    for layer in active:
        strength=float(layer.get('strength',1));kind=layer['kind']
        if kind=='rim_light':
            scale=(alpha>0).astype(np.float32)
            if applicability is not None:scale=np.clip(np.divide(automatic_alpha,original_alpha,out=np.zeros_like(alpha),where=original_alpha>0),0,1)
            foreground+=sample(layer['assets']['emission'])*strength*scale[...,None]
        elif kind=='background_design':
            target=sample(layer['assets']['background']);v=sample(layer['assets']['validity']) if 'validity' in layer['assets'] else 1
            weight=v[...,None]*strength if isinstance(v,np.ndarray) else strength
            background=background*(1-weight)+target*strength
    dynamic_scene=(plan.get('execution') or {}).get('receiver_context')=='current_environment'
    if dynamic_scene:
        background=_lights(background,light_plans['background'],None,hbox,native)
        background=_appearance(background,run,'background',hbox,native)
        background=_manual_pointwise(background,run,'background',box=hbox,canvas_size=native)
    from native_sole_contact import sample_native_contact,receiver_attenuation
    attenuation=np.zeros(alpha.shape,np.float32)
    receiver_layers=[layer for layer in active if layer['kind'] in {'contact_shadow','authored_receiver_shadow'}]
    for layer in receiver_layers:
        if layer['kind']=='contact_shadow':
            opacity=float(layer['params']['opacity'])*float(np.clip(layer.get('strength',1),0,1))
            support=sample_native_contact(layer['assets']['attenuation_support'],hbox,native)
            contribution=receiver_attenuation(support,sample(layer['assets']['receiver_support']),opacity)
        else:
            opacity=float(layer['params']['opacity'])*float(np.clip(layer.get('strength',1),0,1))
            support=sample(layer['assets']['attenuation_support'])
            contribution=receiver_attenuation(attenuation,attenuation,0,support,opacity)
        attenuation=np.maximum(attenuation,contribution)
        records.append({'id':layer['id'],'applied':True,
            'execution_stage':'background_receiver_attenuation_maximum_once',
            'receiver_attenuation_owner':'shared_contact_and_authored_shadow_maximum',
            'receiver_support_consumed':True,'foreground_coverage_preweighted':False,
            'physical_cast_shadow':False,'source_person_RGB_used':False,
            'artistic_projection':layer['kind']=='authored_receiver_shadow',
            'final_alpha_consumed_by_composition':True})
    if receiver_layers:
        background*=1-attenuation[...,None]
    if not dynamic_scene:
        background=_lights(background,light_plans['background'],None,hbox,native)
        background=_appearance(background,run,'background',hbox,native)
        background=_manual_pointwise(background,run,'background',box=hbox,canvas_size=native)
    from foreground_effects import apply_slot as apply_effect_slot
    background,effect_records=apply_effect_slot(background,plan,run,'behind',hbox,native)
    records.extend(effect_records)
    result=foreground+(1-alpha[...,None])*background
    for layer in active:
        strength=float(layer.get('strength',1));kind=layer['kind']
        if kind=='atmosphere':
            fog=sample(layer['assets']['fog']);density=sample(layer['assets']['fog'],'density')*strength
            density*=1-alpha+alpha*float(layer.get('params',{}).get('near_fraction',.08))
            result=result*(1-density[...,None])+fog*density[...,None]
        elif kind in {'particles','bloom'} and layer not in [x[0] for x in bloom]:result+=sample(layer['assets']['emission'])*strength
    result,effect_records=apply_effect_slot(result,plan,run,'front',hbox,native)
    records.extend(effect_records)
    result=_lights(result,light_plans['composite'],None,hbox,native)
    result=_appearance(result,run,'composite',hbox,native)
    result=_manual_pointwise(result,run,'composite',box=hbox,canvas_size=native)
    for layer,sigma in bloom:
        highlights=np.maximum(result-float(layer['params'].get('threshold_linear',.45)),0)
        result=result+gaussian_filter(highlights,(sigma,sigma,0))*float(layer.get('strength',1))
    return result[core].copy(),dict(schema='native_roi_execution.v1',recipe=RECIPE,enabled=plan.get('enabled'),
        source_coordinate_box=list(box),processing_box=list(hbox),halo_native_pixels=halo,
        native_pixel_spacing=[1,1],layers=records,source_geometry_changed=False,
        bounded_roi_renderer=True,full_canvas_rendered=False,nonlocal_receiver='zero support in processing box',
        source_reference='paired_native_baseline',source_binding=binding,
        working_domain='display_linear_sRGB',
        authored_receiver_shadow=plan.get('execution',{}).get('authored_receiver_shadow'))
