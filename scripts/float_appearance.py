"""One source-anchored float32 renderer for preview and high-bit delivery.

Foreground geometry and detail carriers come from the declared source domain.
Design reconstruction may use cleaned candidate pixels for the environment.
"""
from __future__ import annotations

import numpy as np

ENGINE = "source-float-v6-current-only-display-domain"


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)




def _gaussian_float(values: np.ndarray, sigma: float = 0.9) -> np.ndarray:
    """Use SciPy's compiled separable Gaussian with the current edge recipe."""
    from scipy.ndimage import gaussian_filter
    values = np.asarray(values, dtype=np.float32)
    radius = max(1, int(np.ceil(3 * sigma)))
    if values.ndim == 3:
        axes_sigma, axes_radius = (sigma, sigma, 0.), (radius, radius, 0)
    elif values.ndim == 2:
        axes_sigma, axes_radius = (sigma, sigma), (radius, radius)
    else:
        raise ValueError('Gaussian detail input must be HW or HWC')
    return np.asarray(gaussian_filter(values, sigma=axes_sigma, radius=axes_radius,
                                      mode='nearest', output=np.float32), dtype=np.float32)


def _detail_tiled(values: np.ndarray, *, denoise: float = 0.0, sharpness: float = 0.0,
                  tile: int = 512, coverage=None) -> np.ndarray:
    """Read immutable halo tiles so detail controls never introduce tile seams."""
    if not denoise and not sharpness:
        return values
    height, width = values.shape[:2]
    result = np.empty_like(values)
    halo = 3
    for top in range(0, height, tile):
        bottom = min(height, top + tile)
        for left in range(0, width, tile):
            right = min(width, left + tile)
            y0, y1 = max(0, top - halo), min(height, bottom + halo)
            x0, x1 = max(0, left - halo), min(width, right + halo)
            source = values[y0:y1, x0:x1]
            blurred = _gaussian_float(source)
            if denoise:
                # Protect actual edges; smooth small near-flat noise only.
                detail = np.max(np.abs(source - blurred), axis=-1, keepdims=True)
                amount = np.float32(denoise) * np.exp(-detail / np.float32(0.018))
                if coverage is not None:
                    from scipy.ndimage import minimum_filter
                    # No cross-matte blur: leave uncertain/translucent edge
                    # samples unchanged, process only opaque neighborhoods.
                    opaque = minimum_filter(coverage[y0:y1, x0:x1], size=7, mode='nearest') >= .98
                    amount *= opaque[..., None]
                source = source + (blurred - source) * amount
                if sharpness:
                    blurred = _gaussian_float(source)
            if sharpness:
                source = source + (source - blurred) * np.float32(sharpness)
            crop = source[top-y0:bottom-y0, left-x0:right-x0]
            result[top:bottom, left:right] = crop if coverage is not None else np.clip(crop, 0.0, 1.0)
    return result




def prepare_registered_style_input(values):
    """Enter the existing final clip_sdr domain before a frozen style.

    Manual exposure/optics may produce finite working values beyond SDR. The
    final renderer previously clipped these on encoding. Do that same pixelwise
    conversion once before fitting/applying style, in bounded row blocks.
    Ordinary renders without a style never call this helper.
    """
    values = np.asarray(values)
    if values.dtype != np.float32 or values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError('Registered-style input requires float32 display-linear RGB')
    needs_clip = False
    for top in range(0, len(values), 128):
        row = values[top:top+128]
        if not np.isfinite(row).all():
            raise ValueError('Registered-style input contains nonfinite working RGB')
        needs_clip |= bool(np.any(row < 0) or np.any(row > 1))
    if not needs_clip:
        return values
    result = np.empty_like(values)
    for top in range(0, len(values), 128):
        # Equivalent to display_linear(..., {'method':'clip_sdr'}) for the
        # finite working values above; preserve the caller's owned buffer.
        np.clip(values[top:top+128], 0, 1, out=result[top:top+128])
    return result


def apply_registered_style_layers(values, plan, run, *, box=None, canvas_size=None):
    """Consume one frozen pointwise style operator after composition/optics."""
    layers=[layer for layer in plan.get('layers',[]) if layer.get('kind')=='registered_reference_style']
    if not plan.get('enabled') or not layers:
        return values, []
    if len(layers)!=1:
        raise ValueError('A native scene permits one registered reference style operator')
    from registered_reference_style import validate_layer, apply_with_report
    layer=layers[0]; receipt=validate_layer(layer)
    binding=receipt['identity']['reference_contract']
    if (binding['source_sha256']!=(run.get('source') or {}).get('sha256')
            or binding['reference_sha256']!=(run.get('reference') or {}).get('sha256')):
        raise ValueError('Frozen style does not belong to the current source/reference')
    material=receipt['identity'].get('material_context')
    if material is not None:
        # Replay the owned frozen masks, but never carry skin permission into
        # another brief or grant it retrospectively to an old checked plan.
        from creative_direction import skin_cast_contract_for
        from skin_illumination import _digest
        task=run.get('task_plan') or {}
        intent=run.get('task_intent') or task.get('intent') or {}
        goal=(task.get('creative_direction') or {}).get('material_refinement')
        if (task.get('ready') is not True or task.get('intent')!=intent
                or goal is None or goal!=skin_cast_contract_for(intent)
                or material.get('goal_sha256')!=_digest(goal)):
            raise ValueError('Frozen skin illumination does not belong to the checked user permission')
    if not layer.get('enabled') or not float(layer.get('strength',1.)):
        return values,[{'id':layer['id'],'kind':layer['kind'],'enabled':bool(layer.get('enabled')),
            'strength':layer.get('strength',1.),'assets':layer['assets'],'applied':False,
            'execution_stage':'final_source_guided_photometric_style'}]
    values=prepare_registered_style_input(values)
    height,width=values.shape[:2];canvas_size=tuple(canvas_size or (width,height))
    box=tuple(box or (0,0,width,height))
    native=tuple(plan['canvas']['source_oriented_size'])
    if canvas_size==native:
        result,report=apply_with_report(values,layer,box,native)
    else:
        result,report=apply_with_report(values,layer,box,canvas_size,preview=True)
    return result,[{'id':layer['id'],'kind':layer['kind'],'enabled':bool(layer.get('enabled')),
        'strength':layer.get('strength',1.),'assets':layer['assets'],
        'execution_stage':'final_source_guided_photometric_style',
        'input_domain_policy':'final_clip_sdr_before_frozen_style',**report}]


def render_source_float(source_linear: np.ndarray, run: dict, *, geometry=None) -> tuple[np.ndarray, dict]:
    """Render the current frozen display-linear paired source and its layers."""
    from design_reconstruction import (plan_for_run,apply_plan,geometry_for_plan,apply_optics,
        render_mode_for_run,validate_current_native_plan)
    plan=validate_current_native_plan(plan_for_run(run));render_mode_for_run(run)
    height,width=source_linear.shape[:2]
    if geometry is None:
        geometry=geometry_for_plan(plan,(width,height))
    active=plan
    if not plan['enabled']:
        import copy
        active=copy.deepcopy(plan);active['enabled']=True
        active['layers']=[row for row in active['layers'] if row['kind']=='background_design']
        for row in active['layers']:row['enabled']=False
    result,execution=apply_plan(source_linear,source_linear,active,geometry,include_optics=False,run=run)
    if not plan['enabled']:
        execution.update(enabled=False,native_design_contributions_enabled=False,
            source_base_contract='paired native Fp + (1-alpha)*B_old; current manual edits on their branches')
    result=apply_appearance_revision_float(result,run,geometry,branch='composite')
    result=apply_manual_layers_float(result,run,geometry,branch='composite')
    result,execution=apply_optics(result,active,execution)
    result,style_records=apply_registered_style_layers(result,active,run)
    for record in style_records:
        saved=next((row for row in execution['layers'] if row['id']==record['id']),None)
        if saved is None:execution['layers'].append(record)
        else:saved.update(record)
    from display_transform import plan_for_run as display_plan_for_run
    return result,{'engine':ENGINE,'calibration_rgb8':False,'working_space':'display-linear-sRGB-float32',
        'working_domain':'display_linear_sRGB','fit_space':'frozen_native_appearance_delta',
        'display_transform':display_plan_for_run(run),'render_mode':'native_source_design',
        'design_reconstruction_plan':execution,
        'composition_policy':'native_premultiplied_source_design_environment' if plan['enabled'] else 'native_prepared_source_design_off',
        'candidate_background_pixels_used':execution['background_reconstructed'],
        'candidate_pixels_copied':execution['background_reconstructed'],'candidate_main_person_texture_used':False,
        # The production caller may append a freshly prepared style layer
        # after this render.  Record whether this render already consumed an
        # enabled registered style so it cannot be applied a second time.
        'registered_style_applied': bool(any(record.get('applied') for record in style_records)),
        'source_render_plan':execution.get('source_render_plan'),'source_color_plan':execution.get('source_color_plan')}


def render_source_only_float(display_source: np.ndarray, run: dict, silhouette) -> tuple[np.ndarray, dict]:
    """Current explicit local editing of a source after its recorded display transform."""
    from design_reconstruction import render_mode_for_run
    from display_transform import plan_for_run as display_plan_for_run
    render_mode_for_run(run);plan=(run.get('options') or {}).get('source_only_plan') or {}
    from raw_support import RGB_DECODE_VERSION
    values=np.asarray(display_source,dtype=np.float32)
    if (plan.get('schema')!='source_only_display_plan.v1' or plan.get('working_domain')!='display_linear_sRGB'
            or plan.get('source_sha256')!=(run.get('source') or {}).get('sha256')
            or plan.get('native_size')!=list(values.shape[1::-1]) or run.get('reference')
            or plan.get('decode_version')!=RGB_DECODE_VERSION
            or plan.get('source_precision') not in {'raw16_linear','raster8_srgb'}
            or plan.get('source_display_applied_once') is not True
            or plan.get('reference_assets_used') is not False):
        raise ValueError('Source-only edits require a current source-bound display plan without a reference')
    if (run.get('options') or {}).get('appearance_revision',{}).get('design_reconstruction_plan'):
        raise ValueError('Source-only editing cannot consume a prepared reference plan')
    geometry=silhouette.convert('L').resize(tuple(plan['native_size']))
    from portrait_beauty_layer import apply_for_run as apply_beauty
    values,beauty_receipt=apply_beauty(values,run)
    result=_detail_tiled(values,denoise=float((run.get('options') or {}).get('denoise_strength',0)),
        coverage=np.asarray(geometry,np.float32)/255)
    from source_relight import apply_source_render_plan,source_render_plan_for_run
    from source_color import apply_source_color_plan,source_color_plan_for_run
    revision=(run.get('options') or {}).get('appearance_revision') or {}
    light=source_render_plan_for_run(run)
    if revision.get('source_render_plan_origin') not in {'manual','model_with_user_edits','user'}:light={'enabled':False}
    result,light_receipt=apply_source_render_plan(values,result,light,geometry)
    colour=source_color_plan_for_run(run)
    if revision.get('source_color_plan_origin') not in {'manual','model_with_user_edits','user'}:colour={'enabled':False}
    result,colour_receipt=apply_source_color_plan(values,result,colour,geometry)
    result=apply_appearance_revision_float(result,run,geometry)
    result=apply_manual_layers_float(result,run,geometry)
    return result,{'engine':ENGINE,'render_mode':'native_source_design','working_domain':'display_linear_sRGB',
        'working_space':'display-linear-sRGB-float32','display_transform':display_plan_for_run(run),
        'composition_policy':'current_source_only_display','source_only_plan':plan,
        'source_render_plan':light_receipt,'source_color_plan':colour_receipt,'calibration_rgb8':False,
        'portrait_beauty':beauty_receipt,
        'candidate_pixels_copied':False,'source_display_applied_once':True}



def _float_tonal(values: np.ndarray, params: dict, premultiplied_alpha=None) -> np.ndarray:
    """Linear controls with black-preserving white balance and tonal curves."""
    out = np.asarray(values, dtype=np.float32)
    if premultiplied_alpha is not None:
        alpha = premultiplied_alpha[..., None]
        out = np.divide(out, alpha, out=np.zeros_like(out), where=alpha > 0)
    exposure = float(np.clip(params.get("exposure", 0), -2, 2))
    contrast = float(np.clip(params.get("contrast", 0), -1, 1))
    temperature = float(np.clip(params.get("temperature", 0), -1, 1))
    tint = float(np.clip(params.get("tint", 0), -1, 1))
    saturation = float(np.clip(params.get("saturation", 0), -1, 1))
    highlights = float(np.clip(params.get("highlights", 0), -1, 1))
    shadows = float(np.clip(params.get("shadows", 0), -1, 1))
    if exposure:
        out = out * np.float32(2 ** exposure)
    if contrast:
        out = np.float32(.18) * np.power(np.maximum(out, 0) / np.float32(.18), np.float32(2 ** (.5 * contrast)))
    if temperature or tint:
        gains = np.exp2(np.asarray([.35 * temperature + .12 * tint, -.12 * tint,
                                   -.35 * temperature + .12 * tint], dtype=np.float32))
        out = out * gains
    if saturation:
        lum = out[..., 0:1] * .2126 + out[..., 1:2] * .7152 + out[..., 2:3] * .0722
        out = lum + (out - lum) * np.float32(1 + saturation)
    if highlights or shadows:
        lum = out[..., 0:1] * .2126 + out[..., 1:2] * .7152 + out[..., 2:3] * .0722
        stops = np.zeros_like(lum)
        if highlights:
            stops += np.clip((lum - .45) / .55, 0, 1) * np.float32(.4 * highlights)
        if shadows:
            stops += np.clip((.45 - lum) / .45, 0, 1) * np.float32(.5 * shadows)
        out = out * np.exp2(stops)
    sharpness = float(np.clip(params.get("sharpness", 0), -1, 1))
    if sharpness:
        detail = (out - _gaussian_float(out)) * np.float32(sharpness)
        if premultiplied_alpha is not None:
            from scipy.ndimage import minimum_filter
            detail *= (minimum_filter(premultiplied_alpha,size=7,mode='nearest')>=.98)[...,None]
        out = out + detail
    if premultiplied_alpha is not None:
        out = out*alpha
    return np.asarray(out, dtype=np.float32)


def apply_appearance_revision_float(values: np.ndarray, run: dict, silhouette, *, branch=None,
                                    premultiplied_alpha=None) -> np.ndarray:
    """Apply controls in halo tiles, constructing each regional mask once."""
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    options = run.get("options", {})
    revision = options.get("appearance_revision", {})
    params = revision.get("params", {})
    if 'parameters' in revision:
        raise ValueError('Retired appearance parameter alias; current controls require params')
    regions = revision.get("regions")
    if regions is None:
        regions = {key: params[key] for key in ("global", "subject", "background", "local") if key in params}
    height, width = values.shape[:2]
    geometry = silhouette.convert("L").resize((width, height), Image.Resampling.BILINEAR)
    background = ImageChops.invert(geometry)
    full = Image.new("L", (width, height), 255)
    operations = []
    if branch in (None,'composite') and "global" not in regions and any(key in params for key in ("exposure", "contrast", "temperature", "tint", "saturation", "highlights", "shadows", "sharpness")):
        operations.append((params, full))
    for key, base in (("global", full), ("subject", geometry), ("background", background)):
        destination = {'global':'composite','subject':'foreground','background':'background'}[key]
        if branch is not None and destination != branch:
            continue
        if isinstance(regions.get(key), dict):
            operations.append((regions[key], base if branch is None else full))
    for spec in regions.get("local", []):
        if any(key in spec for key in ('region_scope','region')):
            raise ValueError('Retired local appearance aliases; current controls require scope and rect')
        scope = spec.get("scope", "global")
        destination = {'subject':'foreground','background':'background'}.get(scope,'composite')
        if branch is not None and destination != branch:
            continue
        operations.append((spec, (geometry if scope == "subject" else background if scope == "background" else full) if branch is None else full))
    result = values
    for spec, base in operations:
        amount = float(np.clip(spec.get("strength", 1), 0, 1))
        if not amount:
            continue
        mask = base
        rect = spec.get("rect")
        if rect is not None:
            local = Image.new("L", (width, height), 0)
            x0, y0, x1, y1 = map(float, rect)
            ImageDraw.Draw(local).rectangle((round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)), fill=255)
            feather = float(np.clip(spec.get("feather", 0), 0, 1))
            if feather:
                local = local.filter(ImageFilter.GaussianBlur(max(.5, min(width, height) * feather * .025)))
            mask = ImageChops.multiply(base, local)
        output = np.empty_like(result)
        halo = 3 if spec.get("sharpness", 0) else 0
        for top in range(0, height, 512):
            bottom = min(height, top + 512)
            for left in range(0, width, 512):
                right = min(width, left + 512)
                y0, y1 = max(0, top-halo), min(height, bottom+halo)
                x0, x1 = max(0, left-halo), min(width, right+halo)
                source = result[y0:y1, x0:x1]
                alpha = premultiplied_alpha[y0:y1,x0:x1] if premultiplied_alpha is not None else None
                adjusted = _float_tonal(source, spec,alpha)[top-y0:bottom-y0, left-x0:right-x0]
                original = result[top:bottom, left:right]
                m = np.asarray(mask.crop((left, top, right, bottom)), dtype=np.float32)[..., None] * np.float32(amount / 255)
                output[top:bottom, left:right] = original + (adjusted - original) * m
        result = output
    return result


def apply_manual_layers_float(values: np.ndarray, run: dict, geometry, *, branch=None,
                              premultiplied_alpha=None) -> np.ndarray:
    """Apply manual overlays and same-branch clone strokes in linear float."""
    import random
    from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter

    options = run.get("options") if isinstance(run.get("options"), dict) else {}
    layers = options.get("manual_layers") if isinstance(options.get("manual_layers"), dict) else {}
    if not layers:
        return values
    height, width = values.shape[:2]
    subject = geometry.convert("L").resize((width, height), Image.Resampling.BILINEAR)
    outside = ImageChops.invert(subject)
    full = Image.new('L',(width,height),255)
    if branch == 'background':
        outside = full
    if branch == 'foreground':
        subject = full
    result = np.asarray(values, dtype=np.float32)

    def rgb(value, fallback):
        try:
            return ImageColor.getrgb(str(value))[:3]
        except ValueError:
            return ImageColor.getrgb(fallback)

    def region_mask(region):
        mask = Image.new("L", (width, height), 0)
        try:
            x0, y0, x1, y1 = [float(v) for v in region]
        except (TypeError, ValueError):
            x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
        ImageDraw.Draw(mask).rectangle((round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)), fill=255)
        return mask

    def opacity(mask, value):
        try:
            amount = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            amount = 0.0
        return np.asarray(mask, dtype=np.float32) / 255.0 * amount

    def blend_color(mask, color):
        nonlocal result
        overlay = srgb_to_linear(np.asarray(color, dtype=np.float32)[None, None, :] / 255.0)[0, 0]
        if premultiplied_alpha is not None:
            overlay = overlay*premultiplied_alpha[...,None]
        m = mask[..., None]
        result = result * (1.0 - m) + overlay * m

    background = layers.get("background") if isinstance(layers.get("background"), dict) else {}
    if branch in (None,'background') and background.get("enabled"):
        blend_color(opacity(ImageChops.multiply(outside, region_mask(background.get("region", [0, 0, 1, 1]))), background.get("opacity", 0.18)),
                    rgb(background.get("color"), "#182b4d"))

    particles = layers.get("particles") if isinstance(layers.get("particles"), dict) else {}
    if branch in (None,'background') and particles.get("enabled"):
        try:
            count = max(0, min(2000, int(particles.get("count", 80))))
            seed = int(particles.get("seed", 7))
            radius_min, radius_max = particles.get("radius", [1, 4])
            radius_min, radius_max = max(1, int(radius_min)), max(1, int(radius_max))
        except (TypeError, ValueError):
            count, seed, radius_min, radius_max = 80, 7, 1, 4
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        rng = random.Random(seed)
        for _ in range(count):
            x, y = rng.randrange(width), rng.randrange(height)
            radius = rng.randint(radius_min, max(radius_min, radius_max))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        mask = ImageChops.multiply(mask, outside)
        blend_color(opacity(mask, particles.get("opacity", 0.45)), rgb(particles.get("color"), "#ffe3ad"))

    outline = layers.get("outline_light") if isinstance(layers.get("outline_light"), dict) else {}
    if branch in (None,'composite') and outline.get("enabled"):
        try:
            width_px = max(1, min(31, int(outline.get("width", 5))))
            if width_px % 2 == 0:
                width_px += 1
        except (TypeError, ValueError):
            width_px = 5
        edge = ImageChops.subtract(subject.filter(ImageFilter.MaxFilter(width_px)), subject.filter(ImageFilter.MinFilter(width_px)))
        blend_color(opacity(edge, outline.get("opacity", 0.35)), rgb(outline.get("color"), "#8fd7ff"))

    garment = layers.get("garment_texture") if isinstance(layers.get("garment_texture"), dict) else {}
    if branch in (None,'foreground') and garment.get("enabled"):
        try:
            spacing = max(8, min(256, int(garment.get("spacing", 48))))
        except (TypeError, ValueError):
            spacing = 48
        texture_mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(texture_mask)
        for x in range(-height, width + height, spacing):
            draw.line((x, 0, x + height, height), fill=255, width=max(1, spacing // 16))
        texture_mask = ImageChops.multiply(subject, ImageChops.multiply(texture_mask, region_mask(garment.get("region", [0.15, 0.30, 0.85, 0.95]))))
        blend_color(opacity(texture_mask, garment.get("opacity", 0.10)), rgb(garment.get("color"), "#fff1c7"))
    clone=layers.get('clone_retouch')
    if isinstance(clone,dict) and clone.get('strokes') and branch in (None,'foreground','background'):
        from clone_retouch import apply_clone_retouch
        source_alpha=(np.asarray(premultiplied_alpha,np.float32) if premultiplied_alpha is not None else
            np.asarray(geometry.convert('L').resize((width,height),Image.Resampling.BILINEAR),np.float32)/255)
        result=apply_clone_retouch(result,clone,branch=branch,subject_alpha=source_alpha,
            premultiplied_alpha=premultiplied_alpha)
    curves=layers.get('tone_curves')
    if isinstance(curves,dict):
        from tone_curves import apply_tone_curves
        source_alpha=(np.asarray(premultiplied_alpha,np.float32) if premultiplied_alpha is not None else
            np.asarray(geometry.convert('L').resize((width,height),Image.Resampling.BILINEAR),np.float32)/255)
        result=apply_tone_curves(result,curves,branch=branch,subject_alpha=source_alpha,
            premultiplied_alpha=premultiplied_alpha)
    hsl=layers.get('hsl_adjust')
    if isinstance(hsl,dict):
        from hsl_adjust import apply_hsl_adjust
        source_alpha=(np.asarray(premultiplied_alpha,np.float32) if premultiplied_alpha is not None else
            np.asarray(geometry.convert('L').resize((width,height),Image.Resampling.BILINEAR),np.float32)/255)
        result=apply_hsl_adjust(result,hsl,branch=branch,subject_alpha=source_alpha,
            premultiplied_alpha=premultiplied_alpha)
    brush=layers.get('local_colour_brush')
    if isinstance(brush,dict) and brush.get('strokes'):
        from local_colour_brush import apply_local_colour_brush
        source_alpha=(np.asarray(premultiplied_alpha,np.float32) if premultiplied_alpha is not None else
            np.asarray(geometry.convert('L').resize((width,height),Image.Resampling.BILINEAR),np.float32)/255)
        result=apply_local_colour_brush(result,brush,branch=branch,subject_alpha=source_alpha,
            premultiplied_alpha=premultiplied_alpha)
    return np.asarray(result, dtype=np.float32)
