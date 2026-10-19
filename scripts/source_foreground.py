"""Automatic native-resolution source matting and linear foreground estimation.

Current Sapiens2 semantics guide trimap-conditioned ViTMatte. Original source
pixels carry foreground RGB; matting does not synthesize a replacement person.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, binary_dilation, label as connected_labels
from asset_identity import sha256_file
from native_array_storage import save_native_array, save_native_arrays, VALUES_ENCODING
import workflow_timing


FB_DIAGNOSTICS_SCHEMA = 'source_fb_diagnostics.sparse.v1'
DIAGNOSTICS_ENV = 'COSPLAY_SOURCE_FOREGROUND_DIAGNOSTICS'
ALPHA_CACHE_SCHEMA = 'source_alpha_refinement_cache.v1'


def diagnostics_enabled() -> bool:
    """Return whether disposable foreground review rasters should be written.

    The production consumer uses the native NPZ assets and sparse F/B
    diagnostics below.  The large JPEG/PNG review panels are useful during a
    focused investigation, but they add several full-resolution encodes to
    every cold foreground worker.  Keep them explicitly opt-in so normal
    production work does not pay that I/O cost.
    """
    return os.environ.get(DIAGNOSTICS_ENV) == '1'


def _sha256_file(path):
    return sha256_file(path)


def automatic_trimap(saliency, size, edge_fraction=.004):
    """Use a broad source-scaled unknown band, including low-probability flyaways."""
    probability = np.asarray(Image.fromarray(saliency.astype(np.float32)).resize(size, Image.Resampling.BILINEAR))
    radius = max(2, round(min(size) * edge_fraction))
    sure_foreground = binary_erosion(probability >= .98, iterations=radius)
    possible_foreground = binary_dilation(probability > .02, iterations=radius)
    trimap = np.full(probability.shape, 128, np.uint8)
    trimap[~possible_foreground] = 0
    trimap[sure_foreground] = 255
    return trimap, probability, radius


def _windows(width, height, tile, overlap):
    stride = tile-overlap
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            yield left, top, min(width, left+tile), min(height, top+tile)


def _tile_weight(box, size, overlap):
    left, top, right, bottom = box
    width, height = size
    x, y = np.ones(right-left, np.float32), np.ones(bottom-top, np.float32)
    ramp_x = min(overlap, len(x))
    ramp_y = min(overlap, len(y))
    if left: x[:ramp_x] *= np.linspace(.01, 1, ramp_x)
    if right < width: x[-ramp_x:] *= np.linspace(1, .01, ramp_x)
    if top: y[:ramp_y] *= np.linspace(.01, 1, ramp_y)
    if bottom < height: y[-ramp_y:] *= np.linspace(1, .01, ramp_y)
    return y[:, None]*x[None, :]


@workflow_timing.node('alpha_model_and_tiles', model_request=True, http_request=False)
def refine_alpha(display_rgb, trimap, model_dir, *, tile=512, overlap=128, halo=64, device="cuda:0", batch_size=4,
                 _execution=None):
    if not np.any(trimap == 128):
        if _execution is not None:
            _execution['model_inference_batches'] = 0
        workflow_timing.annotate(model_request=False, model_submissions=0, tiles=0)
        return (trimap == 255).astype(np.float32), 0
    import torch
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    model = VitMatteForImageMatting.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    processor = VitMatteImageProcessor.from_pretrained(model_dir, local_files_only=True)
    height, width = trimap.shape
    unknown = trimap == 128
    total = np.zeros((height, width), np.float32)
    weights = np.zeros((height, width), np.float32)
    processed = 0
    batches = 0
    pending = {}
    def predict(rows):
        nonlocal processed, batches
        inputs = {key: torch.cat([row[0][key] for row in rows], dim=0).to(device)
                  for key in rows[0][0]}
        with torch.inference_mode():
            predictions = model(**inputs).alphas[:, 0].float().cpu().numpy()
        batches += 1
        for prediction, (_, box, crop_box) in zip(predictions, rows):
            left, top, right, bottom = box
            x0, y0, x1, y1 = crop_box
            alpha = prediction[:y1-y0, :x1-x0]
            predicted = alpha[top-y0:bottom-y0, left-x0:right-x0]
            weight = _tile_weight(box, (width, height), overlap)
            total[top:bottom, left:right] += predicted*weight
            weights[top:bottom, left:right] += weight
            processed += 1
        print('native alpha tiles', processed, flush=True)
    for box in _windows(width, height, tile, overlap):
        left, top, right, bottom = box
        if not np.any(unknown[top:bottom, left:right]):
            continue
        x0, y0, x1, y1 = max(0, left-halo), max(0, top-halo), min(width, right+halo), min(height, bottom+halo)
        image_crop = Image.fromarray(display_rgb[y0:y1, x0:x1])
        trimap_crop = Image.fromarray(trimap[y0:y1, x0:x1])
        inputs = processor(images=image_crop, trimaps=trimap_crop, return_tensors="pt")
        shape = tuple(inputs['pixel_values'].shape)
        rows = pending.setdefault(shape, [])
        rows.append((inputs, box, (x0, y0, x1, y1)))
        if len(rows) == batch_size:
            predict(rows);rows.clear()
    for rows in pending.values():
        if rows: predict(rows)
    refined = np.where(trimap == 255, 1., 0.).astype(np.float32)
    refined[unknown] = total[unknown]/weights[unknown]
    # Known trimap anchors stay fixed; only the learned unknown alpha is bounded.
    np.clip(refined, 0, 1, out=refined)
    if _execution is not None:
        _execution['model_inference_batches'] = batches
    workflow_timing.annotate(tiles=processed, batch_size=batch_size, device=device,
                             model_submissions=batches, cache_hit=False)
    return refined, processed


def _alpha_array_sha(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((values.shape, values.dtype.str)).encode())
    digest.update(memoryview(values).cast('B'))
    return digest.hexdigest()


def _alpha_code_sha(functions):
    """Bind the executed numerical functions independently of encoding helpers."""
    import ast
    import inspect
    import textwrap
    digest = hashlib.sha256()
    for function in functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(function))))
        tree.body[0].decorator_list = []
        digest.update(ast.dump(tree, include_attributes=False).encode())
    return digest.hexdigest()


def _alpha_runtime_identity(device, *, learned):
    import importlib.metadata
    import platform
    packages = {}
    for package in ('numpy', 'scipy', 'Pillow', 'torch', 'transformers', 'rawpy'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
              'platform': platform.platform(), 'packages': packages, 'device': str(device) if learned else 'trimap_anchors_cpu'}
    if learned:
        import torch
        selected = torch.device(device)
        result.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark)
        if selected.type == 'cuda':
            properties = torch.cuda.get_device_properties(selected)
            result.update(device=str(torch.device('cuda', selected.index if selected.index is not None else torch.cuda.current_device())),
                          device_name=properties.name, capability=[properties.major, properties.minor])
    return result


def alpha_refinement_identity(display_rgb, trimap, model_dir, *, source, precision, native_size,
                              tile=512, overlap=128, halo=64, device='cuda:0', batch_size=4):
    """Identity of actual model inputs; no F/B, container or review semantics."""
    from raw_support import open_linear, open_rgb, linear_preview, deterministic_srgb_profile
    from float_appearance import srgb_to_linear, linear_to_srgb
    if (display_rgb.dtype != np.uint8 or display_rgb.shape != (*trimap.shape, 3)
            or trimap.dtype != np.uint8 or trimap.ndim != 2
            or not np.isin(trimap, (0, 128, 255)).all()
            or not tile > overlap >= 0 or halo < 0 or batch_size < 1):
        raise ValueError('Alpha cache requires the actual RGB8/trimap and valid native tile recipe')
    model_dir = Path(model_dir)
    return {'schema': ALPHA_CACHE_SCHEMA, 'source_sha256': _sha256_file(source),
        'source_precision': precision, 'native_size': list(native_size),
        'analysis_size': list(trimap.shape[::-1]), 'display_rgb_sha256': _alpha_array_sha(display_rgb),
        'trimap_sha256': _alpha_array_sha(trimap),
        'parameters': {'tile': tile, 'overlap': overlap, 'halo': halo, 'batch_size': batch_size},
        'model': {name: _sha256_file(model_dir/name)
                  for name in ('model.safetensors', 'config.json', 'preprocessor_config.json')},
        'numerical_code_sha256': _alpha_code_sha((automatic_trimap, _windows, _tile_weight, refine_alpha)),
        'decode_code_sha256': _alpha_code_sha((open_linear, open_rgb, linear_preview,
                                              deterministic_srgb_profile, srgb_to_linear, linear_to_srgb)),
        'runtime': _alpha_runtime_identity(device, learned=bool(np.any(trimap == 128)))}


def _validate_cached_alpha(alpha, trimap):
    if (alpha.dtype != np.float32 or alpha.shape != trimap.shape or not np.isfinite(alpha).all()
            or np.any((alpha < 0) | (alpha > 1))
            or np.any(alpha[trimap == 0] != 0) or np.any(alpha[trimap == 255] != 1)):
        raise ValueError('Cached alpha violates native float32 coverage or fixed trimap anchors')


def refine_alpha_cached(display_rgb, trimap, model_dir, cache_dir, *, identity, **parameters):
    """Only reuse a completed, byte-checked alpha from this numerical producer."""
    import zipfile
    expected = {'tile': 512, 'overlap': 128, 'halo': 64, 'batch_size': 4,
                **{name: value for name, value in parameters.items() if name != 'device'}}
    if (identity.get('schema') != ALPHA_CACHE_SCHEMA or identity.get('parameters') != expected
            or identity.get('display_rgb_sha256') != _alpha_array_sha(display_rgb)
            or identity.get('trimap_sha256') != _alpha_array_sha(trimap)):
        raise ValueError('Alpha cache binding differs from actual pixels or tile recipe')
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    directory = Path(cache_dir)/key
    receipt_path, asset_path = directory/'receipt.json', directory/'alpha.npz'
    miss_reason = 'not_published'
    with workflow_timing.span('alpha_cache_lookup', model_request=False, http_request=False):
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding='utf8'))
                if receipt.get('schema') != ALPHA_CACHE_SCHEMA or receipt.get('identity') != identity:
                    raise ValueError('Alpha numerical producer binding changed')
                if (not asset_path.is_file() or receipt['alpha_sha256'] != _sha256_file(asset_path)
                        or type(receipt.get('tiles')) is not int or receipt['tiles'] < 0
                        or type(receipt.get('model_inference_batches')) is not int
                        or not 0 <= receipt['model_inference_batches'] <= receipt['tiles']):
                    raise ValueError('Alpha cache is incomplete or its bytes/counts changed')
                with np.load(asset_path, allow_pickle=False) as resource:
                    if resource.files != ['values']:
                        raise ValueError('Alpha cache requires exactly one values array')
                    alpha = resource['values']
                _validate_cached_alpha(alpha, trimap)
            except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as error:
                miss_reason = f'{type(error).__name__}: {error}'
            else:
                with workflow_timing.span('alpha_model_and_tiles', model_request=False, http_request=False,
                        model_submissions=0, cache_hit=True, tiles=0, cached_tiles=receipt['tiles']):
                    pass
                return alpha, 0, {'schema': ALPHA_CACHE_SCHEMA, 'cache_hit': True, 'cache_key': key,
                    'receipt_path': str(receipt_path.resolve()), 'receipt_sha256': _sha256_file(receipt_path),
                    'producer_tiles': receipt['tiles'], 'model_inference_batches_this_attempt': 0}
    execution = {}
    alpha, tiles = refine_alpha(display_rgb, trimap, model_dir, _execution=execution, **parameters)
    _validate_cached_alpha(alpha, trimap)
    with workflow_timing.span('alpha_cache_publish', model_request=False, http_request=False):
        directory.mkdir(parents=True, exist_ok=True)
        save_native_array(asset_path, alpha)
        receipt = {'schema': ALPHA_CACHE_SCHEMA, 'identity': identity,
            'alpha_sha256': _sha256_file(asset_path), 'tiles': tiles,
            'model_inference_batches': execution['model_inference_batches']}
        temporary = receipt_path.with_suffix('.tmp-'+str(os.getpid())+'.json')
        temporary.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False)+'\n', encoding='utf8')
        temporary.replace(receipt_path)
    return alpha, tiles, {'schema': ALPHA_CACHE_SCHEMA, 'cache_hit': False, 'cache_key': key,
        'receipt_path': str(receipt_path.resolve()), 'receipt_sha256': _sha256_file(receipt_path),
        'producer_tiles': tiles, 'model_inference_batches_this_attempt': execution['model_inference_batches'],
        'miss_reason': miss_reason}


@workflow_timing.node('native_asset_encoding', model_request=False, http_request=False)
def save_foreground_arrays(directory, arrays):
    """Bound independent lossless encodes to two threads with atomic files."""
    from runtime_resources import ordered_native_map
    def save(row):
        name, values = row
        path = (Path(directory)/f'{name}.npz').resolve()
        save_native_array(path, values.astype(np.float32, copy=False))
        return name.replace('-', '_')+'_path', str(path)
    return dict(ordered_native_map(save, arrays, workers=2))


@workflow_timing.node('fb_estimation', model_request=False, http_request=False)
def estimate_source_background(source_linear, alpha, trimap, *, tile=512, overlap=128, halo=96):
    """Estimate F/B in linear light, then preserve original detail by exact unmixing."""
    from native_cuda_matting import MIN_PIXELS
    from native_cuda_fields import eligible
    if alpha.size >= MIN_PIXELS and eligible(alpha.shape):
        workflow_timing.annotate(backend='cuda_multilevel_fb_jacobi_v1', native_pixels=alpha.size)
        return _cuda_source_background(source_linear, alpha, trimap)
    workflow_timing.annotate(backend='pymatting_cpu_small_array', native_pixels=alpha.size)
    from pymatting import estimate_foreground_ml
    height, width = alpha.shape
    unknown = trimap == 128
    # Only unknown-band measurements are retained. No diagnostic native RGB
    # or scalar plane is allocated, and these values never feed the renderer.
    unknown_index = np.flatnonzero(unknown.ravel())
    fit_error_sum = np.zeros(len(unknown_index), np.float32)
    separation_sum = np.zeros(len(unknown_index), np.float32)
    background_sum = np.zeros_like(source_linear)
    support_sum = np.zeros_like(alpha)
    weights = np.zeros_like(alpha)
    scene_scale = max(1., float(source_linear.max()))
    processed = 0
    luma_weights = np.array([.2126, .7152, .0722], np.float32)
    for box in _windows(width, height, tile, overlap):
        left, top, right, bottom = box
        if not np.any(unknown[top:bottom, left:right]):
            continue
        x0, y0, x1, y1 = max(0, left-halo), max(0, top-halo), min(width, right+halo), min(height, bottom+halo)
        observed = np.ascontiguousarray(source_linear[y0:y1, x0:x1]/scene_scale)
        local_alpha = np.ascontiguousarray(alpha[y0:y1, x0:x1])
        foreground, background = estimate_foreground_ml(observed, local_alpha, return_background=True)
        fitted = foreground*local_alpha[..., None] + background*(1-local_alpha[..., None])
        error = np.linalg.norm(fitted-observed, axis=-1)
        separation = np.linalg.norm(foreground-background, axis=-1)
        signal = observed@luma_weights
        support = separation/(separation+.025) * np.exp(-error/(signal+.01))
        region = (slice(top-y0, bottom-y0), slice(left-x0, right-x0))
        weight = _tile_weight(box, (width, height), overlap)
        background_sum[top:bottom, left:right] += background[region]*scene_scale*weight[..., None]
        support_sum[top:bottom, left:right] += support[region]*weight
        weights[top:bottom, left:right] += weight
        uy, ux = np.nonzero(unknown[top:bottom, left:right])
        positions = np.searchsorted(unknown_index, (uy+top)*width+ux+left)
        local_weight = weight[uy, ux]
        fit_error_sum[positions] += error[region][uy, ux]*local_weight
        separation_sum[positions] += separation[region][uy, ux]*local_weight
        processed += 1
        if processed % 10 == 0:
            print("linear F/B tiles", processed, flush=True)
    old_background = source_linear.copy()
    old_background[trimap == 255] = 0  # Undefined behind opaque foreground; never used by compositing.
    old_background[unknown] = background_sum[unknown]/weights[unknown, None]
    old_background[alpha == 0] = source_linear[alpha == 0]
    # Project the estimated B to physically nonnegative source reconstruction.
    # This is a lower-bound constraint on Fp, never an upper HDR highlight clip.
    uncovered = 1-alpha
    background_contribution = old_background*uncovered[..., None]
    before_projection = background_contribution[unknown].copy()
    background_contribution = np.minimum(background_contribution, source_linear)
    projection_delta = background_contribution[unknown]-before_projection
    old_background[unknown] = np.divide(background_contribution[unknown], uncovered[unknown, None],
        out=old_background[unknown].copy(), where=uncovered[unknown, None] > 0)
    premultiplied = source_linear-background_contribution
    premultiplied[alpha == 0] = 0
    support = np.zeros_like(alpha)
    support[unknown] = support_sum[unknown]/weights[unknown]
    unknown_weight = weights[unknown]
    diagnostics = {
        'flat_index': unknown_index.astype(np.uint32 if width*height <= 2**32 else np.uint64),
        'fit_error': fit_error_sum/unknown_weight,
        'separation': separation_sum/unknown_weight,
        'background_projection_delta': projection_delta,
        'overlap_weight': unknown_weight,
        'canvas_size': np.array([width, height], np.int32),
        'fb_normalization_scale': np.array(scene_scale, np.float32),
    }
    return old_background, premultiplied, support, processed, scene_scale, diagnostics


def _cuda_source_background(source, alpha, trimap):
    from native_cuda_matting import estimate_foreground_background
    height, width = alpha.shape
    unknown = trimap == 128
    indices = np.flatnonzero(unknown.ravel())
    scale = max(1., float(source.max()))
    foreground, background = estimate_foreground_background(
        np.ascontiguousarray(source/scale), np.ascontiguousarray(alpha))
    local_alpha = alpha[unknown, None]
    observed = source[unknown]/scale
    f, b = foreground[unknown], background[unknown]
    fit_error = np.linalg.norm(f*local_alpha+b*(1-local_alpha)-observed, axis=-1)
    separation = np.linalg.norm(f-b, axis=-1)
    signal = observed@np.array([.2126, .7152, .0722], np.float32)
    support = np.zeros_like(alpha)
    support[unknown] = separation/(separation+.025)*np.exp(-fit_error/(signal+.01))
    old_background = source.copy()
    old_background[trimap == 255] = 0
    old_background[unknown] = b*scale
    old_background[alpha == 0] = source[alpha == 0]
    uncovered = 1-alpha
    contribution = old_background*uncovered[..., None]
    before_projection = contribution[unknown].copy()
    np.minimum(contribution, source, out=contribution)
    projection_delta = contribution[unknown]-before_projection
    old_background[unknown] = np.divide(contribution[unknown], uncovered[unknown, None],
        out=old_background[unknown].copy(), where=uncovered[unknown, None] > 0)
    premultiplied = source-contribution
    premultiplied[alpha == 0] = 0
    diagnostics = {'flat_index': indices.astype(np.uint32 if width*height <= 2**32 else np.uint64),
        'fit_error': fit_error, 'separation': separation,
        'background_projection_delta': projection_delta,
        'overlap_weight': np.ones(len(indices), np.float32),
        'canvas_size': np.array([width, height], np.int32),
        'fb_normalization_scale': np.array(scale, np.float32)}
    return old_background, premultiplied, support, 1, scale, diagnostics


@workflow_timing.node('fb_diagnostics_encoding', model_request=False, http_request=False)
def _write_fb_diagnostics(directory, diagnostics, source, asset_paths):
    """Bind estimator observations, never infer them from the exact residual."""
    path = (Path(directory)/'fb-diagnostics.npz').resolve()
    # These bounded sparse observations are small beside the native endpoints.
    # Stored members avoid compression work while retaining every measured bit.
    save_native_arrays(path, diagnostics)
    bindings = {name: _sha256_file(asset_paths[name]) for name in (
        'alpha_path', 'old_background_path', 'premultiplied_foreground_path', 'unknown_edge_support_path')}
    return {
        'schema_version': FB_DIAGNOSTICS_SCHEMA, 'path': str(path),
        'sha256': _sha256_file(path), 'producer_sha256': _sha256_file(__file__),
        'source_sha256': _sha256_file(source), 'paired_asset_sha256': bindings,
        'coordinate_space': 'oriented_source_pixels',
        'canvas_size': diagnostics['canvas_size'].tolist(),
        'representation': 'trimap_unknown_sparse_row_major_flat_index',
        'sample_count': len(diagnostics['flat_index']),
        'domain': 'trimap == 128; absent pixels have no diagnostic measurement',
        'fit_error': 'overlap-weighted mean L2(F_est*alpha+B_est*(1-alpha)-I), before projection and exact Fp residual',
        'separation': 'overlap-weighted mean L2(F_est-B_est), before projection and exact Fp residual',
        'estimate_units': 'linear_sRGB normalized by fb_normalization_scale',
        'fb_normalization_scale': float(diagnostics['fb_normalization_scale']),
        'background_projection_delta': 'associated linear_sRGB RGB contribution after nonnegative projection minus before; source radiance units',
        'purpose': 'diagnostic only; not albedo, material truth, alpha, light support, or calibrated accuracy',
        'renderer_consumed': False, 'support_expanded': False,
    }


def _review(source_linear, saliency, alpha, foreground, directory, semantics=None):
    from raw_support import linear_preview
    height, width = alpha.shape
    display = linear_preview(source_linear)
    old_alpha = np.asarray(Image.fromarray(saliency).resize((width, height), Image.Resampling.BILINEAR))
    old_composite = source_linear*old_alpha[..., None] + .18*(1-old_alpha[..., None])
    new_composite = foreground + .18*(1-alpha[..., None])
    alpha_title = 'native RGB ViTMatte alpha + source Fp'
    panels = [("immutable original", display), ("upsampled saliency (not alpha)", linear_preview(old_composite)),
              (alpha_title, linear_preview(new_composite))]
    sheet = Image.new("RGB", (1800, 930), "#15141c")
    draw = ImageDraw.Draw(sheet)
    for i, (name, picture) in enumerate(panels):
        picture = picture.copy(); picture.thumbnail((600, 900), Image.Resampling.LANCZOS)
        sheet.paste(picture, (600*i+(600-picture.width)//2, 30))
        draw.text((600*i+10, 8), name, fill="white")
    sheet.save(directory/"foreground-comparison.jpg", quality=95)
    if semantics:
        with np.load(semantics) as data:
            labels = np.asarray(Image.fromarray(data['labels']).resize((width, height), Image.Resampling.NEAREST))
        hair = labels == 4
        hands = np.isin(labels, [6, 15])
        crops = {}
        if hair.any():
            y, x = np.nonzero(hair)
            margin = round(min(width, height)*.015)
            crops['hair'] = (max(0, int(x.min())-margin), max(0, int(y.min())-margin), min(width, int(x.max())+margin), min(height, int(y.max())+margin))
        if hands.any():
            hy, hx = np.nonzero(hands)
            props = (old_alpha > .1) & (labels == 0)
            props[:int(hy.min())] = False
            components, count = connected_labels(props)
            areas = np.bincount(components.ravel()); areas[0] = 0
            if count:
                chosen = components == areas.argmax()
                py, px = np.nonzero(chosen)
                margin = round(min(width, height)*.01)
                crops['bag-chain'] = (max(0, int(min(hx.min(), px.min()))-margin), max(0, int(hy.min())-margin),
                    min(width, int(max(hx.max(), px.max()))+margin), min(height, int(max(hy.max(), py.max()))+margin))
        for name, box in crops.items():
            crop_width, crop_height = box[2]-box[0], box[3]-box[1]
            crop_sheet = Image.new("RGB", (crop_width*3, crop_height+30), "#15141c")
            draw = ImageDraw.Draw(crop_sheet)
            for i, (label, picture) in enumerate(panels):
                crop_sheet.paste(picture.crop(box), (i*crop_width, 30)); draw.text((i*crop_width+8, 8), label, fill="white")
            crop_sheet.save(directory/f"{name}-native-comparison.jpg", quality=95)
            Image.fromarray(np.uint8(alpha[box[1]:box[3], box[0]:box[2]]*255)).save(directory/f"{name}-alpha.png")


@workflow_timing.entry('source_foreground_worker',
    lambda: Path(sys.argv[sys.argv.index('--output-dir')+1]))
def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--saliency", type=Path, required=True)
    parser.add_argument("--semantics", type=Path)
    parser.add_argument('--native-retouch-binding',type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument('--alpha-cache-dir', type=Path,
                        help='Strict intermediate Alpha cache, independent of F/B and output encoding')
    parser.add_argument("--model-dir", type=Path, default=root/'.cache/local-models/vitmatte-small-distinctions')
    parser.add_argument("--max-edge", type=int, default=0, help="0 retains native source pixels; nonzero is a declared preview analysis")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--halo", type=int, default=64)
    args = parser.parse_args()
    started = time.perf_counter()
    # Installation metadata is needed in the result; read it before expensive
    # native matting so an incomplete installation cannot waste an inference.
    provenance = json.loads((args.model_dir/'provenance.json').read_text(encoding='utf-8'))
    from raw_support import open_linear, linear_preview
    if args.native_retouch_binding:
        from native_source_baseline import open_source_for_run
        binding=json.loads(args.native_retouch_binding.read_text(encoding='utf8'))
        source,precision=open_source_for_run({'source':{'path':str(args.source.resolve()),'sha256':binding['source_sha256']},
            'options':{'native_source_retouch':binding}})
    else:
        source, precision = open_linear(args.source)
    native_size = list(source.shape[1::-1])
    if args.max_edge and max(native_size) > args.max_edge:
        ratio = args.max_edge/max(native_size)
        size = (round(native_size[0]*ratio), round(native_size[1]*ratio))
        source = np.stack([np.asarray(Image.fromarray(source[..., c]).resize(size, Image.Resampling.LANCZOS)) for c in range(3)], axis=-1)
    saliency = np.asarray(Image.open(args.saliency).convert('L'), np.float32)/255
    display = np.asarray(linear_preview(source))
    trimap, probability, radius = automatic_trimap(saliency, source.shape[1::-1])
    alpha_identity = alpha_refinement_identity(display, trimap, args.model_dir,
        source=args.source, precision=precision, native_size=native_size, tile=args.tile, halo=args.halo)
    alpha, alpha_tiles, alpha_cache = refine_alpha_cached(display, trimap, args.model_dir,
        args.alpha_cache_dir or args.output_dir/'alpha-refinement-cache', identity=alpha_identity,
        tile=args.tile, halo=args.halo)
    trimap_method = 'eroded_p>=.98_and_dilated_p>.02'
    limitations = ['Automatic trimap inherits saliency misses outside its unknown band.',
                   'Single-image alpha, foreground and background are not uniquely identifiable in dark or similar-color regions.',
                   'Native RGB tile matting improves edge guidance but is not proof of perfect flyaway or chain extraction.']
    alpha_finished = time.perf_counter()
    from native_cuda_fields import cuda_field_scope
    from native_cuda_matting import RECIPE as CUDA_FB_RECIPE, MIN_PIXELS
    with cuda_field_scope():
        background, foreground, support, fb_tiles, scale, diagnostics = estimate_source_background(source, alpha, trimap, tile=args.tile, halo=max(args.halo, 96))
    fb_backend = CUDA_FB_RECIPE if alpha.size >= MIN_PIXELS else 'pymatting_cpu_small_image'
    fitted = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = save_foreground_arrays(args.output_dir, [('alpha', alpha), ('old-background', background),
        ('premultiplied-foreground', foreground), ('unknown-edge-support', support)])
    fb_diagnostics = _write_fb_diagnostics(args.output_dir, diagnostics, args.source, paths)
    paths['fb_diagnostics_path'] = fb_diagnostics['path']
    # These are disposable review rasters.  They are not consumed by the
    # current design plan (which binds the four native NPZ assets above) and
    # each one is a full-resolution encode.  Keep an explicit escape hatch for
    # a focused diagnostic run without slowing every production request.
    review_diagnostics = diagnostics_enabled()
    if review_diagnostics:
        Image.fromarray(np.uint8(alpha*255)).save(args.output_dir/'alpha.png', compress_level=1)
        Image.fromarray(trimap).save(args.output_dir/'automatic-trimap.png', compress_level=1)
        Image.fromarray(np.uint8(support*255)).save(args.output_dir/'unknown-edge-support.png', compress_level=1)
        _review(source, saliency, alpha, foreground, args.output_dir, args.semantics)
    result = {**paths, 'schema_version':'source_foreground.v1', 'native_values_encoding':VALUES_ENCODING, 'source':str(args.source.resolve()),
        'native_source_size':native_size, 'analysis_size':list(source.shape[1::-1]), 'coordinate_space':'oriented_source_pixels',
        'working_space':'scene-linear_sRGB_float32', 'representation':'premultiplied_foreground',
        'equation':'I = Fp + (1-alpha)*B_old; C = gain*Fp + (1-alpha)*B_new',
        'unknown_edge_support_role':'Applicability from local F/B separation and reconstruction fit; not calibrated accuracy',
        'fb_diagnostics': fb_diagnostics, 'alpha_cache': alpha_cache,
        'trimap':{'method':trimap_method, 'unknown_band_analysis_radius':radius,'manual_input':False},
        'tiles':{'core':args.tile,'overlap':128,'alpha_halo':args.halo,'fb_halo':max(args.halo,96),'alpha_tiles':alpha_tiles,'fb_tiles':fb_tiles},
        'provenance':{**provenance, 'foreground_background_estimator':fb_backend,
                      'foreground_background_license':'MIT','source_precision':precision,'fb_normalization_scale':scale,
                      'detail_preservation':'Fp derived from immutable I; model estimates alpha and B, not replacement RGB',
                      'nonnegative_reconstruction_projection':True,'source_only':True,'gpt_calls':0},
        'timings_seconds':{'alpha_and_decode':round(alpha_finished-started,3),'fb_estimate':round(fitted-alpha_finished,3),'total':round(time.perf_counter()-started,3)},
        'limitations':limitations}
    result['worker_timing'] = str(workflow_timing.current().path)
    # Publish the producer receipt last and atomically. A cancelled/partial
    # worker must leave either the previous complete result or no new result,
    # never a truncated JSON object that blocks coordinator recovery. This
    # does not alter the identity or bytes of any Alpha asset.
    result_path = args.output_dir/'result.json'
    temporary_result = result_path.with_name(result_path.name+'.tmp-'+str(os.getpid()))
    temporary_result.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary_result, result_path)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
