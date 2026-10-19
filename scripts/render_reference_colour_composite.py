"""Render the native T+B baseline from fitted colours and bound colour fusion.

Reuses the selected background and alpha. Only the unknown source edge band
is re-estimated in appearance RGB; this does not alter the existing project.
Run with the registered model-worker Python (PyMatting and OpenCV).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from runtime_resources import code_path, ordered_native_map
from asset_identity import sha256_file
import workflow_timing

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from display_transform import display_linear
from raw_support import open_linear, linear_preview, save_render_master
from reference_colour_fit import apply, REGIONS
from native_cuda_sampling import request_sampling_lease, sample_scoped_maps


def field_owners(receipt, shape):
    """New automatic fields own their proxy material/protection domains."""
    if not receipt.get('automatic_policy'):
        return None
    with np.load(receipt['asset']['path'], allow_pickle=False) as resource:
        owners = resource['owners']
    if owners.shape != tuple(shape) or not np.issubdtype(owners.dtype, np.integer):
        raise ValueError('Automatic field material domains do not match its grid')
    return owners


def matching_tiles(alpha, tile_rows=128):
    """Yield full bands and covered native boxes for pixelwise matching.

    Prepared proxy fields remain complete: callers sample their original
    native coordinates and bilinear neighbours. Transparent bands retain one
    native matching pixel for ordinary validation and zero-dose receipts;
    callers still own the complete zero-alpha target and background bands.
    """
    for top in range(0, len(alpha), tile_rows):
        end = min(len(alpha), top+tile_rows)
        covered = alpha[top:end] > 0
        rows = np.flatnonzero(covered.any(axis=1))
        columns = np.flatnonzero(covered.any(axis=0))
        if len(rows):
            left, right = int(columns[0]), int(columns[-1])+1
            first, last = top+int(rows[0]), top+int(rows[-1])+1
        else:
            left, right, first, last = 0, 1, top, top+1
        yield top, end, (left, first, right, last), np.s_[first:last, left:right]


def _sample_grid(box, proxy_size, native_size):
    """Build one native tile's remap grid for all fields in that tile.

    Native rendering samples several fields (weights, donors, colour, scene,
    background and validity) with identical coordinates.  Keeping coordinate
    construction outside ``sample_maps`` avoids allocating the same meshgrid
    once per field while preserving the exact float32 coordinates and OpenCV
    interpolation behaviour.
    """
    left, top, right, bottom = box
    xs = ((np.arange(left, right)+.5)*proxy_size[0]/native_size[0]-.5).astype(np.float32)
    ys = ((np.arange(top, bottom)+.5)*proxy_size[1]/native_size[1]-.5).astype(np.float32)
    return np.meshgrid(xs, ys)


def _sample_plan(grid, values_shape):
    """Precompute coordinate math shared by owner-aware fields in one tile."""
    xx, yy = grid
    height, width = values_shape[:2]
    clipped_xx = np.clip(xx, 0, width-1)
    clipped_yy = np.clip(yy, 0, height-1)
    x0 = np.floor(clipped_xx).astype(np.int32)
    y0 = np.floor(clipped_yy).astype(np.int32)
    x1 = np.minimum(x0+1, width-1)
    y1 = np.minimum(y0+1, height-1)
    dx, dy = clipped_xx-x0, clipped_yy-y0
    owner_x = np.floor(clipped_xx+.5).astype(np.int32)
    owner_y = np.floor(clipped_yy+.5).astype(np.int32)
    # Every owner-aware field in a tile uses the same four neighbours and
    # bilinear weights.  Keep those products and row-major indices beside the
    # coordinate plan so colour, scene and donor sampling do not rebuild them
    # once per field.  The expressions are evaluated here exactly as in the
    # former call site, preserving float32 values and output bits.
    i00 = y0*width + x0
    i10 = y0*width + x1
    i01 = y1*width + x0
    i11 = y1*width + x1
    owner_indices = owner_y*width + owner_x
    w00 = (1-dx)*(1-dy)
    w10 = dx*(1-dy)
    w01 = (1-dx)*dy
    w11 = dx*dy
    return {'values_shape': (height, width), 'xx': xx, 'yy': yy, 'x0': x0, 'y0': y0,
            'x1': x1, 'y1': y1, 'dx': dx, 'dy': dy,
            'owner_x': owner_x, 'owner_y': owner_y,
            'owner_indices': owner_indices, 'indices': (i00, i10, i01, i11),
            'weights': (w00, w10, w01, w11)}


def _sample_identity_grid(values, box, proxy_size, native_size, owners, grid, support, plan):
    """Copy proven original proxy pixels; retain owner/visibility and +0 bits."""
    if (owners is None or values.ndim != 3 or tuple(proxy_size) != tuple(native_size)
            or tuple(proxy_size) != (values.shape[1], values.shape[0])):
        return None
    l, t, r, b = box
    shape = b-t, r-l
    if (any(not isinstance(x, (int, np.integer)) for x in box)
            or not 0 <= l < r <= min(values.shape[1], 2**24)
            or not 0 <= t < b <= min(values.shape[0], 2**24)
            or owners.shape != values.shape[:2] or not np.issubdtype(owners.dtype, np.integer)):
        return None
    yy, xx = np.ogrid[t:b, l:r]
    xx, yy = np.broadcast_to(xx, shape), np.broadcast_to(yy, shape)
    if grid is not None and (not np.array_equal(grid[0], xx) or not np.array_equal(grid[1], yy)):
        return None
    if plan is not None:
        indices = yy*values.shape[1]+xx
        if (tuple(plan.get('values_shape', ())) != values.shape[:2]
                or not np.array_equal(plan['xx'], xx) or not np.array_equal(plan['yy'], yy)
                or not np.array_equal(plan.get('owner_indices'), indices)
                or not np.array_equal(plan['indices'][0], indices)
                or not np.all(plan['weights'][0] == 1)
                or any(not np.all(w == 0) for w in plan['weights'][1:])):
            return None
        for positions in (*plan['indices'], plan['owner_indices']):
            if (not isinstance(positions, np.ndarray) or positions.shape != shape
                    or not np.issubdtype(positions.dtype, np.integer)
                    or positions.min() < 0 or positions.max() >= owners.size):
                return None
    if (not np.isfinite(values).all() or support is not None and (values.shape[-1] not in (3, 4, 6)
            or support.shape != shape or not np.isfinite(support).all() or support.min() < 0 or support.max() > 1)):
        return None
    result = np.zeros((*shape, values.shape[-1]), np.float32)
    result += values[t:b, l:r]
    result += np.float32(0)
    if support is not None:
        dose = np.where(np.isin(owners[t:b, l:r], (4, 1004)), 1-support, 1).astype(np.float32)
        result[..., :2 if values.shape[-1] == 6 else values.shape[-1]-1] *= dose[..., None]
    return result


def sample_maps(values, box, proxy_size, native_size, *, owners=None, grid=None,
                hair_face_support=None, sample_plan=None):
    sampled = _sample_identity_grid(values, box, proxy_size, native_size, owners,
        grid, np.asarray(hair_face_support) if hair_face_support is not None else None, sample_plan)
    if sampled is not None:
        return sampled
    sampled = sample_scoped_maps(values, box, proxy_size, native_size, owners=owners,
        grid=grid, hair_face_support=hair_face_support, sample_plan=sample_plan)
    if sampled is not None:
        return sampled
    if grid is None:
        xx, yy = _sample_grid(box, proxy_size, native_size)
    else:
        xx, yy = grid
    if owners is not None:
        if (owners.shape != values.shape[:2] or not np.issubdtype(owners.dtype, np.integer)
                or tuple(proxy_size) != (values.shape[1], values.shape[0])):
            raise ValueError('Material-aware field sampling requires matching integer owners')
        width = values.shape[1]
        if sample_plan is None:
            xx = np.clip(xx, 0, values.shape[1]-1); yy = np.clip(yy, 0, values.shape[0]-1)
            x0 = np.floor(xx).astype(np.int32); y0 = np.floor(yy).astype(np.int32)
            x1 = np.minimum(x0+1, values.shape[1]-1); y1 = np.minimum(y0+1, values.shape[0]-1)
            owner_x = np.floor(xx+.5).astype(np.int32)
            owner_y = np.floor(yy+.5).astype(np.int32)
            dx,dy = xx-x0, yy-y0
        else:
            if (sample_plan['xx'].shape != xx.shape
                    or tuple(sample_plan.get('values_shape', ())) != (values.shape[0], values.shape[1])):
                raise ValueError('Precomputed sample plan shape differs from target grid')
            x0, y0 = sample_plan['x0'], sample_plan['y0']
            x1, y1 = sample_plan['x1'], sample_plan['y1']
            dx, dy = sample_plan['dx'], sample_plan['dy']
            owner_x, owner_y = sample_plan['owner_x'], sample_plan['owner_y']
        # Flattened row-major indices avoid repeating a two-array advanced
        # indexing broadcast for every corner.  The source arrays are already
        # C-contiguous native fields, so this preserves the exact gathered
        # values while reducing temporary index construction in every tile.
        values_flat = values.reshape((-1,) + values.shape[2:])
        owners_flat = owners.reshape(-1)
        owner_indices = sample_plan.get('owner_indices') if sample_plan is not None else owner_y*width + owner_x
        domain = owners_flat[owner_indices]
        result = np.zeros((*xx.shape,)+values.shape[2:], np.float32)
        # Advanced indexing (``values_flat[indices]``) allocates a fresh
        # corner array on every one of the four neighbours.  Native matching
        # samples the same owner-aware fields for dozens of full-resolution
        # tiles, so keep one typed scratch buffer and gather into it.  The
        # gather and following float32 multiply are the same operations as the
        # former expression; only the temporary lifetime changes.
        # Match NumPy's value*dose promotion before the in-place accumulation
        # as well, so callers that provide a non-float32 field retain the
        # former arithmetic contract (production fields are float32).
        corner = np.empty((*xx.shape,)+values.shape[2:],
                          dtype=np.result_type(values.dtype, np.float32))
        total = np.zeros_like(xx)
        channels = (None,)*(values.ndim-2)
        if sample_plan is not None:
            corners = zip(sample_plan['indices'], sample_plan['weights'])
        else:
            corners = ((y0*width+x0,(1-dx)*(1-dy)), (y0*width+x1,dx*(1-dy)),
                       (y1*width+x0,(1-dx)*dy), (y1*width+x1,dx*dy))
        for indices,w in corners:
            dose = (w*(owners_flat[indices]==domain)).astype(np.float32)
            np.take(values_flat, indices, axis=0, out=corner)
            corner *= dose[(...,)+channels]
            result += corner
            total += dose
        result = result/np.maximum(total,1e-12)[(...,)+channels]
        if hair_face_support is not None:
            support = np.asarray(hair_face_support)
            if (values.ndim != 3 or values.shape[-1] not in (3,4,6) or support.shape != xx.shape
                    or not np.isfinite(support).all() or support.min() < 0 or support.max() > 1):
                raise ValueError('Hair field veto requires sampled visible-face support on the target grid')
            dose = np.where(np.isin(domain, (4,1004)), 1-support, 1).astype(np.float32)
            # Colour v2 has B ab/QA plus ambient ab/feature mask. Its QA and
            # feature support are not amplitudes; scene fields have EV/ab/QA.
            result[..., :2 if values.shape[-1] == 6 else values.shape[-1]-1] *= dose[...,None]
        return result
    if hair_face_support is not None:
        raise ValueError('Hair field veto requires the field material owners')
    return cv2.remap(values, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def sample_plain_stack(stacked, channel_sizes, box, proxy_size, native_size, *, grid=None):
    """Remap a pre-concatenated owner-free field bank and split its views.

    Weights, donors and optional chroma weights share coordinates and are all
    channel-wise float32 fields.  One OpenCV remap preserves each channel's
    interpolation bits while avoiding repeated proxy coordinate walks.  The
    owner-aware colour/scene fields remain on ``sample_maps``.
    """
    stacked = np.asarray(stacked)
    channel_sizes = tuple(int(value) for value in channel_sizes)
    if (stacked.ndim != 3 or not channel_sizes or sum(channel_sizes) != stacked.shape[-1]
            or not np.issubdtype(stacked.dtype, np.floating)):
        raise ValueError('Invalid pre-concatenated plain field bank')
    if grid is None:
        xx, yy = _sample_grid(box, proxy_size, native_size)
    else:
        xx, yy = grid
    sampled = cv2.remap(stacked, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    result, start = [], 0
    for channels in channel_sizes:
        result.append(sampled[..., start:start+channels])
        start += channels
    return tuple(result)


def sample_reference_field(values, box, proxy_size, canvas_size, *, owners, guard=None,
                          grid=None, visible_face_support=None, sample_plan=None):
    """B/S and proxy donors consume the same source visibility veto once.

    ``visible_face_support`` lets native matching share the support sampled for
    two fields at the same tile.  The array is read-only here; omitting it
    retains the original guard-bound sampling path for independent callers.
    """
    support = visible_face_support
    if guard is not None and support is None:
        from portrait_beauty_layer import sample_visible_face_support
        support = sample_visible_face_support(guard, box, canvas_size)
    return sample_maps(values, box, proxy_size, canvas_size, owners=owners, grid=grid,
                       hair_face_support=support, sample_plan=sample_plan)


def donor_fields(source, alpha, weights, parameters, reference_blend=None, scene_balance=None, matched_baseline=None, chroma_weights=None):
    """A bounded affine extension avoids division by low edge alpha."""
    if (source.ndim != 3 or source.shape[-1] != 3 or alpha.shape != source.shape[:2]
            or weights.shape != (*source.shape[:2], len(REGIONS))
            or not all(np.isfinite(x).all() for x in (source, alpha, weights))
            or alpha.min() < 0 or alpha.max() > 1 or weights.min() < 0
            or weights.max() > 1+1e-6 or np.any(weights.sum(-1) > 1+1e-6)):
        raise ValueError('Invalid source/alpha/material donor inputs')
    material_weights = weights
    if parameters.get('automatic_material_palette') and chroma_weights is not None:
        material_weights = np.maximum(weights, chroma_weights)
        material_weights /= np.maximum(material_weights.sum(-1, keepdims=True), 1)
    eligible = (alpha >= .995) & (material_weights.sum(-1) > .8) & (source.max(-1) < .95)
    if matched_baseline is not None:
        if matched_baseline.shape != source.shape or not np.isfinite(matched_baseline).all():
            raise ValueError('Invalid native-matched donor baseline')
        matched = matched_baseline
    else:
        matched = apply(source, weights, parameters, canvas_size=(source.shape[1], source.shape[0]), chroma_weights=chroma_weights)
        if reference_blend is not None:
            from reference_colour_blend import fuse_reference_colour
            matched = fuse_reference_colour(matched, reference_blend, source_values=source)
    if scene_balance is not None:
        from reference_scene_balance import apply_scene_balance
        matched = apply_scene_balance(matched, scene_balance)
    if not np.isfinite(matched).all() or matched.min() < 0 or matched.max() > 1:
        raise ValueError('Donor target must be finite display SDR, including no-donor fallback')
    owner = material_weights.argmax(-1)
    ownership = material_weights.max(-1)
    extra_ev = float(np.max(np.abs(scene_balance[..., 0]))) if scene_balance is not None else 0.
    limit = 2**(parameters['settings']['maximum_exposure_ev']+extra_ev)
    if not np.isfinite(limit) or limit < 1:
        raise ValueError('Invalid donor exposure bound')
    fields = np.zeros((*source.shape[:2], 5), np.float32)
    fields[..., 0] = 1
    if not np.any(eligible):
        # A lack of donors is a valid degraded result; input effects were still
        # evaluated above so corrupt fields cannot be concealed by this path.
        return fields
    luma = np.array([.2126, .7152, .0722], np.float32)
    def observe(mask, select, maximum_distance, confidence):
        if not np.any(mask):
            return
        distance, positions = distance_transform_edt(~mask, return_indices=True)
        donor_source = source[tuple(positions)][select]
        donor_target = matched[tuple(positions)][select]
        gain = np.clip((donor_target @ luma)/np.maximum(donor_source @ luma, .003), 1/limit, limit)
        bias = donor_target-gain[..., None]*donor_source
        bias *= np.minimum(1, .08/np.maximum(np.linalg.norm(bias, axis=-1), 1e-8))[:, None]
        support = confidence*np.clip(1-distance[select]/maximum_distance, 0, 1)
        fields[select] = np.concatenate([gain[:, None], bias, support[:, None]], axis=-1)
    for index, name in enumerate(REGIONS):
        if (name not in parameters['operators'] and parameters.get('automatic_material_palette', {})
                .get('regions', {}).get(name, {}).get('status') != 'supported'):
            continue
        observe(eligible & (owner == index), (owner == index) & (ownership >= .25), 24, 1.)
    # Untyped edges retain their original premultiplied colour. Every extension
    # requires its own material owner; proximity alone cannot borrow skin/cloth.
    return fields


def bounded_edge_extension(fp, alpha, correction, report=None):
    """Reduce the proposed affine edge dose to the premultiplied SDR interval.

    One scalar dose preserves the correction's RGB direction. Existing source
    matting excursions are exposed separately and cannot be increased. This
    bounds the automatic operator before composition, not an output RGB clip.
    """
    if (fp.shape != correction.shape or fp.shape[:2] != alpha.shape or fp.shape[-1] != 3
            or not all(np.isfinite(x).all() for x in (fp, alpha, correction))
            or alpha.min() < 0 or alpha.max() > 1 or fp.min() < 0):
        raise ValueError('Invalid premultiplied edge colour or coverage')
    upper = np.maximum(alpha[..., None], fp)
    dose = np.ones_like(fp)
    positive, negative = correction > 0, correction < 0
    np.divide(upper-fp, correction, out=dose, where=positive)
    np.divide(-fp, correction, out=dose, where=negative)
    dose = np.clip(dose.min(-1), 0, 1)
    result = fp+dose[..., None]*correction
    if report is not None:
        proposed = fp+correction
        bad = np.any((proposed > upper+1e-6) | (proposed < -1e-6), axis=-1)
        source_excursion = np.maximum(fp-alpha[..., None], 0)
        report['policy'] = 'bound_new_affine_edge_delta_dose; no_final_RGB_clip; source_excursions_not_increased'
        report['requested_outside_sdr_pixels'] = report.get('requested_outside_sdr_pixels', 0)+int(bad.sum())
        report['source_outside_premultiplied_sdr_pixels'] = report.get('source_outside_premultiplied_sdr_pixels', 0)+int(
            np.any(source_excursion > 1e-6, axis=-1).sum())
        report['maximum_requested_excursion'] = max(report.get('maximum_requested_excursion', 0),
            float(np.maximum(proposed-upper, 0).max()), float(np.maximum(-proposed, 0).max()))
        report['maximum_source_excursion'] = max(report.get('maximum_source_excursion', 0), float(source_excursion.max()))
        report['minimum_applied_dose'] = min(report.get('minimum_applied_dose', 1), float(dose.min()))
        report['maximum_operator_increased_excursion'] = max(report.get('maximum_operator_increased_excursion', 0),
            float(np.maximum(result-upper, 0).max()))
    return result


def _match_straight(source, weights, parameters, *, origin, canvas_size,
                    reference_blend=None, chroma_weights=None, _trusted=False):
    matched = apply(source, weights, parameters, origin=origin, canvas_size=canvas_size,
                    chroma_weights=chroma_weights, _trusted=_trusted)
    if reference_blend is not None:
        from reference_colour_blend import fuse_reference_colour
        matched = fuse_reference_colour(matched, reference_blend, source_values=source)
    return matched


_PRE_SCENE_MAX_ADJACENT_BLOCKS = 2


def _pre_scene_recompute_ranges(valid):
    """Yield bounded adjacent miss ranges without crossing an all-valid block."""
    from reference_pre_scene_cache import BLOCK_COLUMNS
    width = valid.shape[1]
    left = 0
    while left < width:
        right = min(left+BLOCK_COLUMNS, width)
        if valid[:, left:right].all():
            left = right
            continue
        # Two 128-column misses form at most one 128x256 native band; sampled
        # owners and fields still use their original global pixel coordinates.
        for _ in range(_PRE_SCENE_MAX_ADJACENT_BLOCKS-1):
            following = min(right+BLOCK_COLUMNS, width)
            if right == width or valid[:, right:following].all():
                break
            right = following
        # Preserve the ordinary multi-column matrix kernel for a thin tail.
        first = left-1 if right-left == 1 and width > 1 else left
        yield first, right
        left = right


def sample_pre_scene_reference_field(values, box, proxy_size, canvas_size, *,
                                     pre_scene_valid, owners, guard=None,
                                     grid=None, visible_face_support=None, sample_plan=None):
    """Sample B only where the verified native T+B cache needs recomputation.

    Scene balance remains a full-band field. The original matcher consumes
    these same bounded miss ranges; unused B locations have no effect on its
    cached target. Slice the complete coordinate plan and visibility support
    so neighbours, owners and the one-column tail retain their original bits.
    """
    if pre_scene_valid is None:
        return sample_reference_field(values, box, proxy_size, canvas_size,
            owners=owners, guard=guard, grid=grid, sample_plan=sample_plan,
            visible_face_support=visible_face_support)
    shape = (box[3]-box[1], box[2]-box[0])
    if (not isinstance(pre_scene_valid, np.ndarray) or pre_scene_valid.dtype != np.dtype(bool)
            or pre_scene_valid.shape != shape or values.ndim != 3):
        raise ValueError('Pre-scene B sampling requires the native boolean cache mask')
    if owners is not None and (owners.shape != values.shape[:2]
            or not np.issubdtype(owners.dtype, np.integer)
            or tuple(proxy_size) != (values.shape[1], values.shape[0])):
        raise ValueError('Material-aware field sampling requires matching integer owners')
    grid = _sample_grid(box, proxy_size, canvas_size) if grid is None else grid
    if any(value.shape != shape for value in grid):
        raise ValueError('Pre-scene B sampling grid differs from the native cache mask')
    if sample_plan is not None and (sample_plan['xx'].shape != shape
            or tuple(sample_plan.get('values_shape', ())) != values.shape[:2]):
        raise ValueError('Precomputed sample plan shape differs from target grid')
    support = visible_face_support
    if guard is not None and support is None:
        from portrait_beauty_layer import sample_visible_face_support
        support = sample_visible_face_support(guard, box, canvas_size)
    if support is not None:
        support = np.asarray(support)
        if owners is None:
            raise ValueError('Hair field veto requires the field material owners')
        if (values.shape[-1] not in (3, 4, 6) or support.shape != shape
                or not np.isfinite(support).all() or support.min() < 0 or support.max() > 1):
            raise ValueError('Hair field veto requires sampled visible-face support on the target grid')
    result = np.zeros((*shape, values.shape[-1]), np.float32 if owners is not None else values.dtype)
    for first, right in _pre_scene_recompute_ranges(pre_scene_valid):
        region = np.s_[:, first:right]
        part_plan = ({name: (value[region] if isinstance(value, np.ndarray)
            else tuple(part[region] for part in value) if name in ('indices', 'weights')
            else value) for name, value in sample_plan.items()} if sample_plan is not None else None)
        result[region] = sample_reference_field(values,
            (box[0]+first, box[1], box[0]+right, box[3]), proxy_size, canvas_size,
            owners=owners, guard=guard, grid=tuple(value[region] for value in grid),
            sample_plan=part_plan, visible_face_support=support[region] if support is not None else None)
    return result


def _reuse_pre_scene(source, weights, parameters, *, origin, canvas_size,
                     reference_blend, chroma_weights, prepared, valid, _trusted):
    matched = prepared.copy()
    for first, right in _pre_scene_recompute_ranges(valid):
        region = np.s_[:, first:right]
        target = _match_straight(source[region], weights[region], parameters,
            origin=(origin[0]+first, origin[1]), canvas_size=canvas_size,
            reference_blend=reference_blend[region] if reference_blend is not None else None,
            chroma_weights=chroma_weights[region] if chroma_weights is not None else None,
            _trusted=_trusted)
        invalid = ~valid[region]
        matched[region][invalid] = target[invalid]
    return matched


def match_foreground(fp, alpha, weights, donors, edge_support, parameters, *, origin, canvas_size,
                     reference_blend=None, scene_balance=None, chroma_weights=None,
                     donor_gamut_report=None, _trusted=False, pre_scene_target=None,
                     pre_scene_valid=None, pre_scene_sink=None):
    # At high coverage unpremultiplication is stable; below .95 it is never used.
    safe = alpha >= .95
    source = np.divide(fp, alpha[..., None], out=np.zeros_like(fp), where=safe[..., None])
    if not np.isfinite(source).all():
        raise ValueError('Unpremultiplied foreground contains nonfinite values')
    # A fractional matte can divide a valid premultiplied foreground slightly
    # above the display endpoint. This is an edge-estimation artifact, so
    # normalize the straight foreground before any colour or scene operator.
    source = np.clip(source, 0, 1)
    if pre_scene_target is None:
        matched = _match_straight(source, weights, parameters, origin=origin,
            canvas_size=canvas_size, reference_blend=reference_blend,
            chroma_weights=chroma_weights, _trusted=_trusted)
    else:
        if (pre_scene_target.shape != source.shape or pre_scene_target.dtype != np.float32
                or pre_scene_valid is None or pre_scene_valid.shape != alpha.shape
                or np.any(pre_scene_valid & (alpha != 1))):
            raise ValueError('Pre-scene reuse requires exact opaque float32 source coverage')
        matched = _reuse_pre_scene(source, weights, parameters, origin=origin,
            canvas_size=canvas_size, reference_blend=reference_blend,
            chroma_weights=chroma_weights, prepared=pre_scene_target,
            valid=pre_scene_valid, _trusted=_trusted)
    if pre_scene_sink is not None:
        pre_scene_sink(matched)
    if scene_balance is not None:
        from reference_scene_balance import apply_scene_balance
        matched = apply_scene_balance(matched, scene_balance)
    direct = matched*alpha[..., None]
    gain, bias, donor_support = donors[..., 0], donors[..., 1:4], donors[..., 4]
    confidence = donor_support*(.35+.65*np.clip(edge_support, 0, 1))
    correction = confidence[..., None]*((gain-1)[..., None]*fp+alpha[..., None]*bias)
    extended = bounded_edge_extension(fp, alpha, correction, donor_gamut_report)
    transition = np.clip((alpha-.95)/.05, 0, 1)
    transition = transition*transition*(3-2*transition)
    result = extended+(direct-extended)*transition[..., None]
    result = np.maximum(result, 0)
    result[alpha == 0] = 0
    return result


@request_sampling_lease(('blend_field', 'blend_owners'))
def _matching_native_bands(foreground, alpha, edge_support, old_bg, background,
                           validity_proxy, parameters, blend_field, blend_owners,
                           hair_guard, size, native, plain_proxy, plain_channels,
                           *, associated, has_chroma, capture_pre_scene, workers=2):
    """Calculate independent original-coordinate bands; consumers own writes."""
    def compute(tile):
        top, end, match_box, region = tile
        grid = _sample_grid(match_box, size, native)
        sample_plan = _sample_plan(grid, blend_field.shape[:2])
        sampled_plain = sample_plain_stack(plain_proxy, plain_channels,
                                           match_box, size, native, grid=grid)
        w, d = sampled_plain[:2]
        cw = sampled_plain[2] if has_chroma else None
        colour = sample_reference_field(blend_field, match_box, size, native, owners=blend_owners,
            guard=hair_guard, grid=grid, sample_plan=sample_plan)
        # This local sink captures the unchanged T+B target, without writing
        # the shared pre-scene cache or changing its statistics in a worker.
        pre_scene = []
        matched = match_foreground(foreground[region], alpha[region], w, d, edge_support[region], parameters,
            origin=match_box[:2], canvas_size=native, reference_blend=colour, chroma_weights=cw,
            pre_scene_sink=pre_scene.append if capture_pre_scene else None, _trusted=True)
        values = np.zeros((end-top, native[0], 3), np.float32)
        left, first, right, last = match_box
        values[first-top:last-top, left:right] = matched
        # Composition still covers every native pixel once in the same band.
        box = (0, top, native[0], end)
        if (background.shape[1], background.shape[0]) == native:
            target = background[top:end]
        else:
            target = sample_maps(background, box, (background.shape[1], background.shape[0]), native)
        validity_size = (validity_proxy.shape[1], validity_proxy.shape[0])
        validity_grid = (grid if match_box == box else _sample_grid(box, size, native)) if validity_size == tuple(size) else None
        validity = sample_maps(validity_proxy, box, validity_size, native, grid=validity_grid)
        bg = target+(1-validity[..., None])*old_bg[top:end] if associated else validity[..., None]*target+(1-validity[..., None])*old_bg[top:end]
        values += (1-alpha[top:end, ..., None])*bg
        counts = (int(np.count_nonzero((values < 0) | (values > 1))),
                  int(np.count_nonzero(alpha[top:end] == 1)),
                  int(np.count_nonzero((alpha[top:end] > 0) & (alpha[top:end] < .95))))
        return top, end, match_box, values, pre_scene[0] if pre_scene else None, counts

    yield from ordered_native_map(compute, matching_tiles(alpha), workers=workers)


def validate_material_inputs(parameters):
    """Shared source-bound material inputs for baseline rendering and freeze."""
    from asset_identity import sha256_file
    if parameters.get('region_order') != list(REGIONS):
        raise ValueError('Re-evaluate matching after a region contract change')
    authored = parameters['binding'].get('authored_colour_edit')
    if authored and sha256_file(authored['path']) != authored['sha256']:
        raise ValueError('Authored colour stroke changed after evaluation')
    if authored and sha256_file(code_path('local_colour_brush.py')) != authored['stroke_algorithm_sha256']:
        raise ValueError('Authored stroke algorithm changed after evaluation')
    objects = parameters['binding'].get('object_masks')
    if objects:
        object_path = Path(objects['receipt_path'])
        if sha256_file(object_path) != objects['receipt_sha256']:
            raise ValueError('Accessory mask review changed after fitting')
        receipt = json.loads(object_path.read_text(encoding='utf8'))
        for role in ('source', 'reference'):
            if sha256_file(receipt['roles'][role]['mask_path']) != objects['masks'][role]:
                raise ValueError('Accessory mask changed after fitting')
    return authored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--match-dir', type=Path, default=Path('outputs/inspection/reference-match-v2'))
    parser.add_argument('--reference-blend', type=Path, required=True, help='Bound selected low-frequency reference colour layer')
    args = parser.parse_args()
    started = time.perf_counter()
    directory = args.match_dir.resolve()
    with workflow_timing.request(directory, 'reference_native_baseline') as ledger:
        with workflow_timing.span('load', model_request=False, http_request=False):
            run = json.loads(args.run.read_text(encoding='utf8'))
            plan = run['options']['appearance_revision']['design_reconstruction_plan']
            parameters = json.loads((directory/'parameters.json').read_text(encoding='utf8'))
            if parameters['binding']['source'] != run['source']['sha256'] or parameters['binding']['reference'] != run['reference']['sha256']:
                raise ValueError('Match belongs to a different original/reference pair')
            if parameters['binding']['algorithm_sha256'] != sha256_file(code_path('reference_colour_fit.py')):
                raise ValueError('Re-evaluate matching after an algorithm change')
            if parameters['binding']['alpha'] != plan['source_baseline']['assets']['foreground_alpha']['sha256']:
                raise ValueError('Match belongs to a different baseline alpha')
            authored = validate_material_inputs(parameters)
            if parameters.get('automatic_material_palette'):
                from reference_material_palette import validate_palette
                validate_palette(parameters['automatic_material_palette'])
            size, native = tuple(plan['canvas']['source_proxy_size']), tuple(plan['canvas']['source_oriented_size'])
            from portrait_beauty_layer import hair_correction_guard_for_run
            hair_guard = hair_correction_guard_for_run(run)
            guard_binding = hair_guard['binding'] if hair_guard else None
            from reference_colour_blend import validate_layer
            blend_field, blend_receipt = validate_layer(args.reference_blend, parameters, native, size)
            if blend_receipt.get('hair_correction_guard') != guard_binding:
                raise ValueError('Reference colour field visible-face guard differs from the current source')
            blend_owners = field_owners(blend_receipt, blend_field.shape[:2])
            # Register large artifact identities in the cross-process cache so
            # freeze/resume validation does not read the same weights again.
            weights_hash = sha256_file(directory/'source-weights.npz')
            if parameters['binding'].get('source_weights_sha256') != weights_hash:
                raise ValueError('Re-evaluate matching after source weights change')
            with np.load(directory/'source-weights.npz', allow_pickle=False) as resource:
                weights = resource['values']
                chroma_weights = resource['chroma_values'] if 'chroma_values' in resource else None
            with np.load(plan['source_baseline']['assets']['foreground_alpha']['path'], allow_pickle=False) as resource:
                alpha = resource['values']
            print('Decoding native original into fixed appearance RGB', flush=True)
            from native_source_baseline import source_for_reference_prepare, background_for_reference_prepare
            source, source_precision, source_display = source_for_reference_prepare(
                run, binding=parameters['binding'], apply_beauty=True, writable=False)
        with workflow_timing.span('matching_donors', model_request=False, http_request=False):
            proxy = cv2.resize(source, size, interpolation=cv2.INTER_LINEAR)
            proxy_box = (0,0,size[0],size[1])
            donor_colour = sample_reference_field(blend_field,proxy_box,size,size,owners=blend_owners,
                guard=hair_guard)
            donors = donor_fields(proxy, cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR), weights, parameters,
                reference_blend=donor_colour, chroma_weights=chroma_weights)
        with workflow_timing.span('foreground_background', model_request=False, http_request=False):
            # Alpha is reused. No ViTMatte, inpainting, reference generation or SR rerun.
            old_bg, foreground, edge_support, tiles, appearance_pair = background_for_reference_prepare(
                source, alpha, run, binding=parameters['binding'], precision=source_precision,
                transform=source_display, apply_beauty=True)
            maximum_reconstruction_error = 0.
            opaque_unchanged = True
            for top in range(0, len(source), 128):
                end = min(len(source), top+128)
                restored = foreground[top:end]+(1-alpha[top:end, ..., None])*old_bg[top:end]
                maximum_reconstruction_error = max(maximum_reconstruction_error, float(np.max(np.abs(restored-source[top:end]))))
                known = alpha[top:end] == 1
                opaque_unchanged &= bool(np.array_equal(foreground[top:end][known], source[top:end][known]))
            del source
        with workflow_timing.span('load_selected_background', model_request=False, http_request=False):
            bg_layer = next(x for x in plan['layers'] if x['kind'] == 'background_design')
            with np.load(bg_layer['assets']['background']['path'], allow_pickle=False) as resource:
                background = resource['values']
            with np.load(bg_layer['assets']['validity']['path'], allow_pickle=False) as resource:
                validity_proxy = resource['values']
            associated = bg_layer['assets']['background'].get('validity_associated', False) or bg_layer['assets']['background'].get('representation') == 'sampling_validity_associated_RGB'
            # The selected plate is inverse-sRGB reference appearance, not RAW scene
            # radiance. Do not subject it to a second photographic display transform.
        with workflow_timing.span('matching_native', model_request=False, http_request=False):
            print('Matching native Fp, selected reference colour fusion, and compositing once', flush=True)
            from reference_pre_scene_cache import open_writer
            pre_scene_writer = open_writer(parameters, blend_receipt, native, size)
            clipping, opaque_pixels, low_alpha_pixels = 0, 0, 0
            plain_proxy = np.concatenate([weights, donors] +
                                         ([chroma_weights] if chroma_weights is not None else []), axis=-1)
            plain_channels = (weights.shape[-1], donors.shape[-1]) + ((chroma_weights.shape[-1],)
                                                                        if chroma_weights is not None else ())
            bands = _matching_native_bands(foreground, alpha, edge_support, old_bg, background,
                validity_proxy, parameters, blend_field, blend_owners, hair_guard, size, native,
                plain_proxy, plain_channels, associated=associated,
                has_chroma=chroma_weights is not None, capture_pre_scene=pre_scene_writer is not None)
            try:
                for top, end, match_box, values, pre_scene, counts in bands:
                    if pre_scene_writer is not None:
                        pre_scene_writer.write(match_box, pre_scene)
                    foreground[top:end] = values
                    clipping += counts[0]
                    opaque_pixels += counts[1]
                    low_alpha_pixels += counts[2]
            finally:
                bands.close()
            del old_bg, background
            np.clip(foreground, 0, 1, out=foreground)
        with workflow_timing.span('pre_scene_cache_publish', model_request=False, http_request=False):
            pre_scene_cache = pre_scene_writer.finish() if pre_scene_writer is not None else {'written': False}
        with workflow_timing.span('master_encoding', model_request=False, http_request=False):
            master = directory/'appearance-composite.tif'
            save_render_master(foreground, master)
        with workflow_timing.span('receipt_hashing', model_request=False, http_request=False):
            report = {'schema': 'reference_colour_composite.appearance.v2', 'source_run_id': run['run_id'],
                'working_domain': 'display_linear_sRGB', 'output': str(master), 'size': list(native),
                'source_reference_binding': parameters['binding'], 'source_weights_sha256': weights_hash,
                'protection_policy': parameters['protection_policy'],
                'parameters_sha256': sha256_file(directory/'parameters.json'),
                'validation_sha256': sha256_file(directory/'validation.json'),
                'renderer_sha256': sha256_file(code_path('render_reference_colour_composite.py')),
                'output_sha256': sha256_file(master),
                'alpha_reused': True, 'appearance_edge_tiles': tiles, 'appearance_pair_cache': appearance_pair,
                'pre_scene_acceleration_cache': pre_scene_cache,
                'original_reconstruction_max_error': maximum_reconstruction_error,
                'edge_donors': 'same_semantic_region; untyped_very_near_edges_support_at_most_0.15',
                'opaque_original_unchanged_before_matching': opaque_unchanged,
                'opaque_source_pixels': opaque_pixels, 'low_alpha_division_avoided_pixels': low_alpha_pixels,
                'final_out_of_gamut_channels_before_clipping': clipping,
                'composition_count': 1, 'photographic_display_transform_after_reference': False,
                'extra_contact_rim_fog_bloom': False, 'existing_project_brushes_applied': False,
                'authored_colour_edit': authored, 'existing_reference_shadows_reflections_resolved': False,
                'reference_spatial_colour_used': True,
                'reference_high_frequency_texture_used': False,
                'reference_spatial_lightness_used': False, 'scene_balance': None,
                'reference_colour_layer': {'receipt_path': str(args.reference_blend.resolve()),
                    'receipt_sha256': sha256_file(args.reference_blend),
                    'asset_sha256': blend_receipt['asset']['sha256'], 'regions': blend_receipt['regions']},
                'geometry_or_texture_from_reference_person': False, 'project_pointer_modified': False,
                'visual_acceptance': False, 'elapsed_seconds': round(time.perf_counter()-started, 2),
                'workflow_timing': {'schema': workflow_timing.SCHEMA, 'path': str(ledger.path),
                    'scope': 'native_baseline_worker_body_after_cli_parse', 'clock': 'time.perf_counter'}}
        (directory/'composite-receipt.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
