"""Automatic native-resolution source matting and linear foreground estimation.

Current Sapiens2 semantics guide trimap-conditioned ViTMatte. Original source
pixels carry foreground RGB; matting does not synthesize a replacement person.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, binary_dilation, label as connected_labels


FB_DIAGNOSTICS_SCHEMA = 'source_fb_diagnostics.sparse.v1'


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


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


def refine_alpha(display_rgb, trimap, model_dir, *, tile=512, overlap=128, halo=64, device="cuda:0"):
    import torch
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    model = VitMatteForImageMatting.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    processor = VitMatteImageProcessor.from_pretrained(model_dir, local_files_only=True)
    height, width = trimap.shape
    unknown = trimap == 128
    total = np.zeros((height, width), np.float32)
    weights = np.zeros((height, width), np.float32)
    processed = 0
    for box in _windows(width, height, tile, overlap):
        left, top, right, bottom = box
        if not np.any(unknown[top:bottom, left:right]):
            continue
        x0, y0, x1, y1 = max(0, left-halo), max(0, top-halo), min(width, right+halo), min(height, bottom+halo)
        image_crop = Image.fromarray(display_rgb[y0:y1, x0:x1])
        trimap_crop = Image.fromarray(trimap[y0:y1, x0:x1])
        inputs = processor(images=image_crop, trimaps=trimap_crop, return_tensors="pt").to(device)
        with torch.inference_mode():
            alpha = model(**inputs).alphas[0, 0].float().cpu().numpy()[:y1-y0, :x1-x0]
        predicted = alpha[top-y0:bottom-y0, left-x0:right-x0]
        weight = _tile_weight(box, (width, height), overlap)
        total[top:bottom, left:right] += predicted*weight
        weights[top:bottom, left:right] += weight
        processed += 1
        if processed % 10 == 0:
            print("native alpha tiles", processed, flush=True)
    refined = np.where(trimap == 255, 1., 0.).astype(np.float32)
    refined[unknown] = total[unknown]/weights[unknown]
    # Known trimap anchors stay fixed; only the learned unknown alpha is bounded.
    np.clip(refined, 0, 1, out=refined)
    return refined, processed


def estimate_source_background(source_linear, alpha, trimap, *, tile=512, overlap=128, halo=96):
    """Estimate F/B in linear light, then preserve original detail by exact unmixing."""
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


def _write_fb_diagnostics(directory, diagnostics, source, asset_paths):
    """Bind estimator observations, never infer them from the exact residual."""
    path = (Path(directory)/'fb-diagnostics.npz').resolve()
    np.savez_compressed(path, **diagnostics)
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


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--saliency", type=Path, required=True)
    parser.add_argument("--semantics", type=Path)
    parser.add_argument('--native-retouch-binding',type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=root/'.cache/local-models/vitmatte-small-distinctions')
    parser.add_argument("--max-edge", type=int, default=0, help="0 retains native source pixels; nonzero is a declared preview analysis")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--halo", type=int, default=64)
    args = parser.parse_args()
    started = time.perf_counter()
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
    alpha, alpha_tiles = refine_alpha(display, trimap, args.model_dir, tile=args.tile, halo=args.halo)
    provenance = json.loads((args.model_dir/'provenance.json').read_text(encoding='utf-8'))
    trimap_method = 'eroded_p>=.98_and_dilated_p>.02'
    limitations = ['Automatic trimap inherits saliency misses outside its unknown band.',
                   'Single-image alpha, foreground and background are not uniquely identifiable in dark or similar-color regions.',
                   'Native RGB tile matting improves edge guidance but is not proof of perfect flyaway or chain extraction.']
    alpha_finished = time.perf_counter()
    background, foreground, support, fb_tiles, scale, diagnostics = estimate_source_background(source, alpha, trimap, tile=args.tile, halo=max(args.halo, 96))
    fitted = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, values in [('alpha', alpha), ('old-background', background), ('premultiplied-foreground', foreground), ('unknown-edge-support', support)]:
        path = (args.output_dir/f'{name}.npz').resolve()
        np.savez_compressed(path, values=values.astype(np.float32, copy=False))
        paths[name.replace('-', '_')+'_path'] = str(path)
    fb_diagnostics = _write_fb_diagnostics(args.output_dir, diagnostics, args.source, paths)
    paths['fb_diagnostics_path'] = fb_diagnostics['path']
    Image.fromarray(np.uint8(alpha*255)).save(args.output_dir/'alpha.png')
    Image.fromarray(trimap).save(args.output_dir/'automatic-trimap.png')
    Image.fromarray(np.uint8(support*255)).save(args.output_dir/'unknown-edge-support.png')
    _review(source, saliency, alpha, foreground, args.output_dir, args.semantics)
    result = {**paths, 'schema_version':'source_foreground.v1', 'source':str(args.source.resolve()),
        'native_source_size':native_size, 'analysis_size':list(source.shape[1::-1]), 'coordinate_space':'oriented_source_pixels',
        'working_space':'scene-linear_sRGB_float32', 'representation':'premultiplied_foreground',
        'equation':'I = Fp + (1-alpha)*B_old; C = gain*Fp + (1-alpha)*B_new',
        'unknown_edge_support_role':'Applicability from local F/B separation and reconstruction fit; not calibrated accuracy',
        'fb_diagnostics': fb_diagnostics,
        'trimap':{'method':trimap_method, 'unknown_band_analysis_radius':radius,'manual_input':False},
        'tiles':{'core':args.tile,'overlap':128,'alpha_halo':args.halo,'fb_halo':max(args.halo,96),'alpha_tiles':alpha_tiles,'fb_tiles':fb_tiles},
        'provenance':{**provenance, 'foreground_background_estimator':'PyMatting estimate_foreground_ml1.1.16, linear source tiles',
                      'foreground_background_license':'MIT','source_precision':precision,'fb_normalization_scale':scale,
                      'detail_preservation':'Fp derived from immutable I; model estimates alpha and B, not replacement RGB',
                      'nonnegative_reconstruction_projection':True,'source_only':True,'gpt_calls':0},
        'timings_seconds':{'alpha_and_decode':round(alpha_finished-started,3),'fb_estimate':round(fitted-alpha_finished,3),'total':round(time.perf_counter()-started,3)},
        'limitations':limitations}
    (args.output_dir/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
