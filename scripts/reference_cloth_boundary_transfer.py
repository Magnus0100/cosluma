"""Reference cloth light coefficients rendered by current source structure.

Only opaque owner 23 is supported. Reference background, geometry and texture
are never RGB donors. A registered same-source-material interior pair offset
is subtracted before robust coarse boundary coefficients are made positive.
Analysis uses approximately one reference pixel; native luminance and source
material membership are evaluated in strips without native clustering.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from colour_space import to_lab

RECIPE = 'reference_cloth_boundary_material_response_v1'
PROFILE = {
    'id': RECIPE, 'owner': 23, 'opaque_alpha': .95,
    'source_material_clusters': 5, 'source_material_seed': 20261010,
    'maximum_cluster_samples': 16384, 'dark_material_Lab_radius': .15,
    'dark_material_L_ceiling_above_principal': .15, 'material_affinity_sigma': .08,
    'coarse_grid_spacing_reference_pixels': 12., 'robust_window_radius_reference_pixels': 16.,
    'minimum_window_samples': 32, 'minimum_window_material_mass': 16.,
    'coarse_coefficient_sigma_cells': .85, 'coarse_maximum_fill_distance_cells': 1.5,
    'source_structure_sigma_reference_pixels': 8., 'maximum_source_structure_ratio': 3.,
    'maximum_depth_reference_pixels': 24., 'depth_taper_reference_pixels': 6.,
    'interior_anchor_depth_reference_pixels': 24., 'interior_anchor_radius_reference_pixels': 64.,
    'interior_anchor_sigma_reference_pixels': 30.,
    'maximum_registration_error_reference_pixels': 8.,
    'minimum_registered_contour_fraction': .7,
    'luma_noise_floor': .003, 'luma_transition': .012,
    'warm_noise_floor': .002, 'warm_transition': .008,
    'analysis_sampling_reference_pixels': 1., 'native_strip_rows': 128,
    'source_qualification_fade_reference_pixels': 2.,
    'reference_qualification_fade_reference_pixels': 2.,
    'source_material_boundary_fade_Lab': .02,
}
LUMA = np.asarray([.2126, .7152, .0722], np.float32)


def _unit(value, shape, name):
    value = np.asarray(value, np.float32)
    if value.shape != shape or not np.isfinite(value).all() or np.any((value < 0)|(value > 1)):
        raise ValueError(name+' must be finite [0,1] with the expected shape')
    return value


def _sha(value):
    value = np.asarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    for first in range(0, len(value), 128):
        digest.update(np.ascontiguousarray(value[first:first+128]).tobytes())
    return digest.hexdigest()


def _sample(value, x, y, nearest=False):
    if value.ndim == 3:
        return np.stack([_sample(value[..., c], x, y, nearest) for c in range(value.shape[-1])], -1)
    return map_coordinates(value, [y, x], order=0 if nearest else 1,
                           mode='constant', cval=0, prefilter=False)


def _contour(alpha):
    inside = alpha >= .5
    depth = (distance_transform_edt(inside)-distance_transform_edt(~inside)).astype(np.float32)
    gy, gx = np.gradient(gaussian_filter(depth, 1.))
    norm = np.maximum(np.hypot(gx, gy), 1e-5)
    return depth, gx/norm, gy/norm


def _median(values, weights):
    order = np.argsort(values, axis=0, kind='stable')
    ordered = np.take_along_axis(values, order, axis=0)
    ow = np.take_along_axis(np.broadcast_to(weights[:, None], values.shape), order, axis=0)
    index = np.argmax(np.cumsum(ow, axis=0) >= ow.sum(0, keepdims=True)*.5, axis=0)
    return ordered[index, np.arange(values.shape[1])]


def _robust(values, weights):
    centre = _median(values, weights)
    residual = np.linalg.norm(values-centre, axis=-1)
    mad = _median(residual[:, None], weights)[0]
    rw = weights*np.minimum(1, max(float(mad)*2.5, .006)/np.maximum(residual, 1e-6))
    return _median(values, rw)


def _fit_affine(maps, labels, alpha, validity, protection, ref_labels, ref_opaque, endpoint_qualified):
    h, w = alpha.shape
    step = max(1, int(np.ceil(np.sqrt(h*w/65536))))
    yy, xx = np.meshgrid(np.arange(0, h, step), np.arange(0, w, step), indexing='ij')
    rx, ry = np.asarray(maps['ref_x'])[yy, xx], np.asarray(maps['ref_y'])[yy, xx]
    good = ((labels[yy, xx] == 23)&(alpha[yy, xx] >= .95)&(validity[yy, xx] >= .8)
            &endpoint_qualified[yy, xx]
            &(protection[yy, xx] < .5)&np.asarray(maps['valid'], bool)[yy, xx]
            &(np.asarray(maps['confidence'])[yy, xx] >= .2)
            &(_sample(ref_labels, rx, ry, True) == 23)&(_sample(ref_opaque, rx, ry) >= .999))
    x, y = xx[good], yy[good]
    if len(x) < 48:
        return None, {'state': 'rejected', 'reason': 'insufficient_opaque_same_owner_registration_anchors'}
    take = np.arange(0, len(x), max(1, int(np.ceil(len(x)/4096))))
    X = np.column_stack((x[take], y[take], np.ones(len(take))))
    Y = np.column_stack((rx[good][take], ry[good][take]))
    if np.linalg.matrix_rank(X) < 3:
        return None, {'state': 'rejected', 'reason': 'collinear_registration_anchors'}
    affine = np.linalg.lstsq(X, Y, rcond=None)[0]
    for _ in range(5):
        residual = np.linalg.norm(X@affine-Y, axis=1)
        weight = np.minimum(1, 3/np.maximum(residual, 1e-4))
        affine = np.linalg.lstsq(X*weight[:, None], Y*weight[:, None], rcond=None)[0]
    singular = np.linalg.svd(affine[:2], compute_uv=False)
    q90 = float(np.quantile(np.linalg.norm(X@affine-Y, axis=1), .9))
    ok = (np.linalg.det(affine[:2]) > 0 and singular.min() > 0
          and singular.max()/singular.min() < 2 and q90 <= 3)
    return (affine if ok else None), {'state': 'qualified' if ok else 'rejected',
        'qualified_sampled_anchor_pixels': int(len(x)), 'fitted_anchor_pixels': int(len(take)),
        'q90_anchor_fit_error_reference_pixels': q90, 'affine': affine.tolist(),
        'boundary_alignment_accuracy_claimed': False}


def _affinity(source_rgb, anchor, domain):
    lab = to_lab(source_rgb)
    distance = np.linalg.norm(lab-anchor, axis=-1)
    colour_fade = _inward_weight(.15-distance, .02)
    lightness_fade = _inward_weight(anchor[0]+.15-lab[..., 0], .02)
    return (np.exp(-.5*(distance/.08)**2)*domain*colour_fade*lightness_fade).astype(np.float32)


def _inward_weight(distance, width):
    """Continuous attenuation only inside an already qualified observation."""
    value = np.clip(distance/width, 0, 1)
    return value*value*(3-2*value)


def _material(source_rgb, domain):
    lab = to_lab(source_rgb); values = lab[domain]
    if len(values) < 32:
        return None, np.zeros(domain.shape, np.float32), {'state': 'rejected', 'reason': 'insufficient_source_material'}
    values = values[::max(1, int(np.ceil(len(values)/16384)))]
    rng = np.random.default_rng(20261010)
    centres = [values[int(rng.integers(len(values)))]]
    for _ in range(1, min(5, len(values))):
        d = np.min(np.sum((values[:, None]-np.asarray(centres)[None])**2, axis=-1), axis=1)
        if float(d.sum()) <= 1e-12: break
        centres.append(values[int(rng.choice(len(values), p=d.astype(np.float64)/d.sum(dtype=np.float64)))])
    centres = np.asarray(centres, np.float32)
    for _ in range(40):
        classes = np.argmin(np.sum((values[:, None]-centres[None])**2, axis=-1), axis=1)
        newer = np.asarray([values[classes == k].mean(0) if np.any(classes == k) else centre
                            for k, centre in enumerate(centres)], np.float32)
        shift = float(np.max(np.linalg.norm(newer-centres, axis=-1))); centres = newer
        if shift <= 1e-5: break
    full_classes = np.argmin(np.sum((lab[domain, None]-centres[None])**2, axis=-1), axis=1)
    sizes = np.bincount(full_classes, minlength=len(centres)); principal = int(np.argmax(sizes))
    anchor = centres[principal]
    return anchor, _affinity(source_rgb, anchor, domain), {
        'state': 'observed', 'implementation': 'deterministic_numpy_kmeans_plus_plus',
        'source_Lab_centres': centres.tolist(), 'source_cluster_pixels': sizes.tolist(),
        'principal_cluster': principal, 'principal_cluster_centre': anchor.tolist(),
        'cluster_sample_count': int(len(values)), 'source_albedo_truth_claimed': False}


def _field(pair_delta, affinity, edge, core, scale):
    h, w = edge.shape; nx = max(2, int(np.ceil((w-1)*scale/12))+1)
    ny = max(2, int(np.ceil((h-1)*scale/12))+1)
    xs, ys = np.linspace(0, w-1, nx), np.linspace(0, h-1, ny)
    nodes = np.zeros((ny, nx, 3), np.float32); bias = np.zeros_like(nodes)
    mass = np.zeros((ny, nx), np.float32); interior_mass = np.zeros_like(mass)
    def observe(x, y, radius, sigma, domain):
        l, t = max(0, int(np.floor(x-radius))), max(0, int(np.floor(y-radius)))
        r, b = min(w, int(np.ceil(x+radius))+1), min(h, int(np.ceil(y+radius))+1)
        yy, xx = np.indices((b-t, r-l), dtype=np.float32); yy += t; xx += l
        weights = affinity[t:b, l:r]*domain[t:b, l:r]*np.exp(-.5*((xx-x)**2+(yy-y)**2)/sigma**2)
        selected = weights > 1e-3; total = float(weights[selected].sum())
        if selected.sum() < 32 or total < 16: return None, total
        return _robust(pair_delta[t:b, l:r][selected], weights[selected]), total
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            boundary, bm = observe(x, y, 16/scale, 16*.65/scale, edge)
            if boundary is None: continue
            offset, cm = observe(x, y, 64/scale, 30/scale, core)
            if offset is None: continue
            nodes[j, i] = boundary-offset; bias[j, i] = offset
            mass[j, i] = bm; interior_mass[j, i] = cm
    valid = mass > 0
    if not valid.any():
        return np.zeros_like(pair_delta), nodes, bias, mass, interior_mass, {'state': 'rejected', 'reason': 'no_boundary_and_same_material_interior_samples'}
    weight = gaussian_filter(valid.astype(np.float32), .85)
    coefficients = gaussian_filter(nodes*valid[..., None], (.85, .85, 0))/np.maximum(weight[..., None], 1e-6)
    coefficients *= ((weight >= .05)&(distance_transform_edt(~valid) <= 1.5))[..., None]
    sy, sx = np.meshgrid(np.linspace(0, ny-1, h, dtype=np.float32),
                          np.linspace(0, nx-1, w, dtype=np.float32), indexing='ij')
    field = _sample(coefficients, sx, sy)
    return field, nodes, bias, mass, interior_mass, {'state': 'estimated', 'coarse_grid_size': [nx, ny],
        'qualified_coarse_nodes': int(valid.sum()), 'interior_pair_offset_subtracted_before_positivity': True,
        'coefficient_interpolation_coordinates': 'endpoint_inclusive_coarse_lattice'}


def prepare_cloth_boundary_light(current_p, alpha, reference_rgb, reference_alpha,
                                source_labels, reference_labels, maps, *, source_rgb,
                                validity=None, identity_protection=None,
                                reference_fp=None, reference_fb_support=None):
    p = np.asarray(current_p, np.float32)
    if p.ndim != 3 or p.shape[-1] != 3 or min(p.shape[:2]) < 4:
        raise ValueError('current_p must be HWC associated RGB')
    # An older endpoint may contain finite positive Fp > alpha. Preserve those
    # pixels and abstain there instead of clipping old RGB or rejecting cloth
    # evidence elsewhere. New active pixels must satisfy whole-RGB bounds.
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('current_p must be finite nonnegative associated RGB')
    shape = p.shape[:2]
    a = _unit(alpha, shape, 'alpha'); src = _unit(source_rgb, p.shape, 'source_rgb')
    endpoint_qualified = np.all(p <= a[..., None]+2e-5, -1)
    ref = np.asarray(reference_rgb, np.float32)
    if ref.ndim != 3 or ref.shape[-1] != 3 or min(ref.shape[:2]) < 4:
        raise ValueError('reference must be HWC RGB')
    ref = _unit(ref, ref.shape, 'reference'); ra = _unit(reference_alpha, ref.shape[:2], 'reference_alpha')
    sl, rl = np.asarray(source_labels), np.asarray(reference_labels)
    if sl.shape != shape or rl.shape != ra.shape or sl.dtype.kind not in 'iu' or rl.dtype.kind not in 'iu':
        raise ValueError('independent semantic labels must match their own image')
    for key in ('ref_x', 'ref_y', 'confidence', 'valid'):
        if key not in maps or np.asarray(maps[key]).shape != shape or not np.isfinite(maps[key]).all():
            raise ValueError('invalid registration map: '+key)
    _unit(maps['confidence'], shape, 'registration confidence')
    _unit(maps['valid'], shape, 'registration valid')
    v = np.ones(shape, np.float32) if validity is None else _unit(validity, shape, 'validity')
    prot = np.zeros(shape, np.float32) if identity_protection is None else _unit(identity_protection, shape, 'identity_protection')
    if (reference_fp is None) != (reference_fb_support is None):
        raise ValueError('reference observations require both Fp and F/B support')
    ref_opaque = (ra >= .95)&(rl == 23)
    fb = None; fp = None
    if reference_fp is not None:
        fp = _unit(reference_fp, ref.shape, 'reference_fp')
        fb = _unit(reference_fb_support, ra.shape, 'reference_fb_support')
        bounded = np.all(fp <= ra[..., None]+2e-6, -1); ref_opaque &= bounded
        ref = np.divide(fp, ra[..., None], out=np.zeros_like(fp), where=ref_opaque[..., None])
    ref_opaque = ref_opaque.astype(np.float32)
    affine, registration = _fit_affine(maps, sl, a, v, prot, rl, ref_opaque, endpoint_qualified)
    delta = np.zeros_like(p); row = {'owner': 23, 'registration': registration, 'changed_pixels': 0}
    diagnostics = {}
    if affine is not None:
        scale = float(np.sqrt(np.linalg.det(affine[:2])))
        h, w = shape; ah = max(4, int(np.ceil((h-1)*min(scale, 1)))+1)
        aw = max(4, int(np.ceil((w-1)*min(scale, 1)))+1)
        y, x = np.meshgrid(np.linspace(0, h-1, ah, dtype=np.float32),
                            np.linspace(0, w-1, aw, dtype=np.float32), indexing='ij')
        factor = float(np.sqrt((w-1)/(aw-1)*(h-1)/(ah-1)))
        analysis_scale = scale*factor
        aa = _sample(a, x, y); ss = _sample(src, x, y)
        pp = _sample(p, x, y); ls = _sample(sl, x, y, True)
        endpoint_mass = _sample(endpoint_qualified.astype(np.float32), x, y)
        vv, pr = _sample(v, x, y), _sample(prot, x, y)
        rx = x*affine[0, 0]+y*affine[1, 0]+affine[2, 0]
        ry = x*affine[0, 1]+y*affine[1, 1]+affine[2, 1]
        sd, snx, sny = _contour(aa); rd, rnx, rny = _contour(ra)
        depth = sd*analysis_scale; error = _sample(rd, rx, ry)-depth
        same = (_sample(rl, rx, ry, True) == 23)&(_sample(ref_opaque, rx, ry) >= .999)
        contour = (ls == 23)&(aa >= .5)&(depth <= 2)
        agrees = (_sample(rl, rx, ry, True) == 23)&(np.abs(error) <= 8)
        fraction = float(agrees[contour].mean()) if contour.any() else 0.
        row.update(registered_contour_fraction=fraction, analysis_size=[aw, ah],
                   analysis_pixel_size_reference_pixels=analysis_scale)
        if fraction >= .7:
            domain = (ls == 23)&(aa >= .95)&(vv >= .8)&(pr < .5)&(endpoint_mass >= .999)
            anchor, affinity, material = _material(ss, domain); row['material'] = material
            if anchor is not None:
                straight = np.divide(pp, aa[..., None], out=np.zeros_like(pp), where=aa[..., None] >= .95)
                paired = _sample(ref, rx, ry)-straight
                nx, ny = _sample(rnx, rx, ry), _sample(rny, rx, ry)
                sx = snx*affine[0, 0]+sny*affine[1, 0]
                sy = snx*affine[0, 1]+sny*affine[1, 1]
                cosine = (sx*nx+sy*ny)/np.maximum(np.hypot(sx, sy), 1e-5)
                geometry = np.clip((8-np.abs(error))/4, 0, 1)*np.clip((cosine-.25)/.5, 0, 1)
                edge = domain&same&(depth <= 24)&(geometry > 0)
                core = domain&same&(depth >= 24)&(_sample(rd, rx, ry) >= 24)&(np.abs(error) <= 8)
                field, nodes, bias, masses, core_masses, field_receipt = _field(paired, affinity, edge, core, analysis_scale)
                row['field'] = field_receipt
                positive = np.maximum(field, 0).astype(np.float32)
                confidence = np.clip((positive@LUMA-.003)/.012, 0, 1)*np.clip((positive[..., 0]-positive[..., 2]-.002)/.008, 0, 1)
                confidence *= geometry*np.clip((24-depth)/6, 0, 1)*edge
                luminance = straight@LUMA; sigma = 8/analysis_scale
                lowmass = gaussian_filter(affinity, sigma)
                baseline = gaussian_filter(luminance*affinity, sigma)/np.maximum(lowmass, 1e-6)
                diagnostics = {'analysis_field': field, 'analysis_interior_bias_nodes': bias,
                    'analysis_field_nodes': nodes, 'analysis_node_mass': masses,
                    'analysis_interior_node_mass': core_masses}
                source_qualified = ((sl == 23)&(a >= .95)&(v >= .8)&(prot < .5)&endpoint_qualified)
                # These distances attenuate dose inside the original strict
                # domains. They never extend ownership, reference donors or
                # source alpha. A first qualified pixel has zero dose, then a
                # fixed two-reference-pixel smoothstep prevents native stairs
                # from nearest-neighbour categorical masks.
                source_depth = distance_transform_edt(source_qualified).astype(np.float32)
                reference_depth = distance_transform_edt(ref_opaque > 0).astype(np.float32)
                for first in range(0, h, 128):
                    stop = min(h, first+128)
                    yy, xx = np.meshgrid(np.arange(first, stop, dtype=np.float32),
                                          np.arange(w, dtype=np.float32), indexing='ij')
                    qx, qy = xx*(aw-1)/(w-1), yy*(ah-1)/(h-1)
                    rf = _sample(positive, qx, qy); cf = _sample(confidence, qx, qy)
                    denominator = _sample(baseline, qx, qy)
                    al = a[first:stop]; pv = p[first:stop]
                    native_domain = source_qualified[first:stop].copy()
                    nrx = xx*affine[0, 0]+yy*affine[1, 0]+affine[2, 0]
                    nry = xx*affine[0, 1]+yy*affine[1, 1]+affine[2, 1]
                    native_domain &= (_sample(rl, nrx, nry, True) == 23)&(_sample(ref_opaque, nrx, nry) >= .999)
                    native_affinity = _affinity(src[first:stop], anchor, native_domain)
                    source_fade = _inward_weight((source_depth[first:stop]-1)*scale, 2.)
                    reference_fade = _inward_weight(_sample(reference_depth, nrx, nry)-1, 2.)
                    native_luma = np.divide(pv@LUMA, al, out=np.zeros_like(al), where=al >= .95)
                    structure = np.clip(native_luma/np.maximum(denominator, 1e-5), 0, 3)
                    structure[denominator <= 1e-5] = 0
                    dose = al[..., None]*rf*(cf*native_affinity*structure*source_fade*reference_fade)[..., None]
                    delta[first:stop] = np.minimum(dose, np.maximum(al[..., None]-pv, 0)).astype(np.float32)
        else:
            row['reason'] = 'coarse_registration_does_not_match_cloth_contour'
    changed = np.any(delta != 0, -1); row['changed_pixels'] = int(changed.sum())
    checks = {'finite': bool(np.isfinite(delta).all()), 'delta_nonnegative': bool(np.all(delta >= 0)),
        'associated_bounds': bool(np.all(~changed | np.all(p+delta <= a[..., None]+2e-5, -1))),
        'preexisting_unassociated_pixels_unchanged': bool(np.all(delta[~endpoint_qualified] == 0)),
        'other_owner_exact_zero': bool(np.all(delta[sl != 23] == 0)),
        'partial_source_exact_zero': bool(np.all(delta[a < .95] == 0)),
        'invalid_source_exact_zero': bool(np.all(delta[v < .8] == 0)),
        'protected_exact_zero': bool(np.all(delta[prot >= .5] == 0))}
    if not all(checks.values()): raise ValueError('cloth response contract failed: '+str(checks))
    receipt = {'schema': 'reference_cloth_boundary_transfer.v1', 'recipe': RECIPE,
        'profile': dict(PROFILE), 'state': 'prepared' if changed.any() else 'evidence_limited',
        'components': [row], 'changed_pixels': int(changed.sum()), 'checks': checks,
        'preexisting_unassociated_pixel_count': int((~endpoint_qualified).sum()),
        'preexisting_unassociated_pixels_are_donors': False,
        'clothing_enabled': True, 'owner_classes': [23], 'hair_changed': False,
        'reference_person_is_light_evidence': True, 'background_is_light_evidence': False,
        'reference_RGB_appearance_is_donor': False, 'reference_geometry_is_donor': False,
        'source_alpha_changed': False, 'source_geometry_changed': False,
        'current_native_source_luminance_supplies_structure_and_texture': True,
        'same_main_source_material_interior_pair_offset_subtracted': True,
        'source_material_analysis_only': True, 'reference_observations_opaque_only': True,
        'continuous_qualification_support_only_attenuates_strict_qualified_domain': True,
        'opaque_reference_FB_support_not_required': True,
        'partial_alpha_physical_response_certified': False, 'physical_relighting_claimed': False,
        'working_domain': 'display_linear_sRGB', 'visual_acceptance': False,
        'image_generation_submissions': 0, 'model_inferences': 0,
        'boundary_correspondence_separate_from_interior_map_valid': True,
        'unknowns': ['source colour clustering is an appearance partition, not physical material identification',
            'broad reference folds can remain confounded with low-frequency light',
            'source normals, albedo and incoming irradiance are not recovered',
            'geometry qualification is measured at approximately reference sampling scale'],
        'binding': {'current_p': _sha(p), 'alpha': _sha(a), 'source_rgb': _sha(src),
            'reference': _sha(ref), 'reference_alpha': _sha(ra), 'source_labels': _sha(sl),
            'reference_labels': _sha(rl), 'validity': _sha(v), 'identity_protection': _sha(prot),
            'maps': {key: _sha(maps[key]) for key in ('ref_x', 'ref_y', 'confidence', 'valid')}}}
    if fp is not None: receipt['binding'].update(reference_fp=_sha(fp), reference_fb_support=_sha(fb))
    return {'delta_p': delta, 'after_p': p+delta, 'alpha': a.copy(), 'support': changed,
            'receipt': receipt, **diagnostics}
