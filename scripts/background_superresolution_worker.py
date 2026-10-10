"""Compact CUDA background SR, with explicit sampling-validity association.

The network is adapted from Real-ESRGAN's BSD-3-Clause SRVGGNetCompact by
Xintao Wang, copyright (c) 2021, all rights reserved. Redistribution terms
and disclaimer: scripts/third_party/Real-ESRGAN-LICENSE.txt. Model and license
provenance are also saved next to the checkpoint.
This worker never changes source foreground geometry or any renderer code.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F

WEIGHT_SHA256 = '8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292'


class SRVGGNetCompact(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.ModuleList([nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64)])
        for _ in range(32):
            self.body.extend([nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(64)])
        self.body.append(nn.Conv2d(64, 48, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(4)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=4, mode='nearest')


def digest(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_plane(asset):
    with np.load(asset['path']) as data:
        return data['values'].astype(np.float32)


def encode(x):
    x = np.maximum(x, 0)
    return np.where(x <= .0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - .055).astype(np.float32)


def decode(x):
    x = np.maximum(x, 0)
    return np.where(x <= .04045, x / 12.92, np.power((x + .055) / 1.055, 2.4)).astype(np.float32)


def maps(box, source_size, target_size, scale=1, offset=(0, 0)):
    left, top, right, bottom = box
    sw, sh = source_size
    tw, th = target_size
    xx = ((np.arange(left, right, dtype=np.float32) + .5) * sw / tw) * scale - .5 - offset[0]
    yy = ((np.arange(top, bottom, dtype=np.float32) + .5) * sh / th) * scale - .5 - offset[1]
    return np.broadcast_to(xx[None, :], (bottom-top, right-left)).copy(), np.broadcast_to(yy[:, None], (bottom-top, right-left)).copy()


def sample(values, grid):
    return cv2.remap(values, *grid, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def compare_crops(baseline, enhanced, source_size, target_size, directory):
    # Scene surfaces visible outside the source person, in source-proxy pixels.
    boxes = {'paper_lattice': (100, 530, 300, 730), 'bench': (870, 1330, 1090, 1550),
             'floor': (800, 1780, 1040, 2020), 'distant_bokeh': (900, 100, 1120, 320)}
    results = {}
    sw, sh = source_size
    tw, th = target_size
    for name, proxy_box in boxes.items():
        left, top, right, bottom = [round(v * (tw/sw if i % 2 == 0 else th/sh)) for i, v in enumerate(proxy_box)]
        box = (left, top, right, bottom)
        old = sample(baseline, maps(box, source_size, target_size))
        new = enhanced[top:bottom, left:right]
        width, height = right-left, bottom-top
        canvas = Image.new('RGB', (width*2, height+28), '#20242b')
        canvas.paste(Image.fromarray(np.uint8(np.clip(old, 0, 1)*255+.5)), (0, 28))
        canvas.paste(Image.fromarray(np.uint8(np.clip(new, 0, 1)*255+.5)), (width, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), 'Bilinear native baseline', fill='white')
        draw.text((width+8, 8), 'Compact SR synthesis', fill='white')
        path = directory / f'crop-{name}.jpg'
        canvas.save(path, quality=96, subsampling=0)
        results[name] = {'path': str(path.resolve()), 'native_box': list(box), 'display_scale': 1}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tile', type=int, default=256)
    args = parser.parse_args()
    start = time.time()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    weight = Path(args.weights).resolve()
    if digest(weight) != WEIGHT_SHA256:
        raise ValueError('Checkpoint does not match the public release hash')
    run = json.loads(Path(args.run).read_text(encoding='utf-8'))
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    layer = next(layer for layer in plan['layers'] if layer['kind'] == 'background_design')
    assets = layer['assets']
    associated = read_plane(assets['background'])
    validity = read_plane(assets['validity'])
    h, w = validity.shape
    target_size = tuple(plan['canvas']['source_oriented_size'])
    tw, th = target_size
    # Unassociate first; extend unsampled border from its nearest valid pixel.
    straight = associated / np.maximum(validity[..., None], 1e-6)
    valid = validity > .01
    if not valid.all():
        nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        straight[~valid] = straight[nearest[0][~valid], nearest[1][~valid]]
    srgb = np.clip(encode(straight), 0, 1)
    removal = np.asarray(Image.open(plan['mask_roles']['candidate_removal']['path']).convert('L'), np.float32)/255
    transform = np.asarray(plan['canvas']['candidate_mapping']['candidate_processing_proxy_to_final_proxy'], np.float64)
    uncertain = cv2.warpPerspective(removal, transform, (w, h), flags=cv2.INTER_LINEAR)
    # The captured mirror arm belongs to the dedicated reflection renderer.
    # Neither strengthen the generated fragment nor synthesize its skin detail.
    reflection_regions = []
    for reflection in plan['layers']:
        if reflection['kind'] == 'trusted_source_reflection' and reflection.get('enabled'):
            asset = reflection['assets']['region_mask']
            receiver = cv2.resize(read_plane(asset), (w, h), interpolation=cv2.INTER_LINEAR)
            uncertain = np.maximum(uncertain, receiver)
            reflection_regions.append(asset)
    uncertain = cv2.dilate(uncertain, np.ones((25, 25), np.uint8))
    support = 1 - np.clip(cv2.GaussianBlur(uncertain, (0, 0), 8), 0, 1)
    Image.fromarray(np.uint8(support*255+.5)).save(out/'detail-support.png')

    model = SRVGGNetCompact().cuda().half().eval()
    state = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict(state.get('params_ema', state.get('params', state)), strict=True)
    enhanced = np.lib.format.open_memmap(out/'native-straight-srgb.npy', mode='w+', dtype=np.float32, shape=(th, tw, 3))
    halo = 40  # 35 convolution receptive field; no tiled zero-padding seams.
    torch.backends.cudnn.benchmark = True
    count = 0
    with torch.inference_mode():
        for y in range(0, h, args.tile):
            for x in range(0, w, args.tile):
                xe, ye = min(x+args.tile, w), min(y+args.tile, h)
                lx, ty, rx, by = max(x-halo, 0), max(y-halo, 0), min(xe+halo, w), min(ye+halo, h)
                tensor = torch.from_numpy(srgb[ty:by, lx:rx].transpose(2, 0, 1).copy()).unsqueeze(0).cuda().half()
                predicted = model(tensor).squeeze(0).float().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                box = (round(x*tw/w), round(y*th/h), round(xe*tw/w), round(ye*th/h))
                grid = maps(box, (w, h), target_size)
                original = sample(srgb, grid)
                detail = sample(predicted, maps(box, (w, h), target_size, 4, (lx*4, ty*4)))
                amount = sample(support, grid)[..., None]
                enhanced[box[1]:box[3], box[0]:box[2]] = original + amount*(detail-original)
                count += 1
                if count % 6 == 0:
                    print(f'tiles {count}; elapsed {time.time()-start:.1f}s', flush=True)
    enhanced.flush()
    crops = compare_crops(srgb, enhanced, (w, h), target_size, out)
    overview = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_AREA)
    Image.fromarray(np.uint8(np.clip(overview, 0, 1)*255+.5)).save(out/'sr-overview.jpg', quality=95, subsampling=0)

    native_p = np.lib.format.open_memmap(out/'native-associated-linear.npy', mode='w+', dtype=np.float32, shape=(th, tw, 3))
    native_v = np.lib.format.open_memmap(out/'native-validity.npy', mode='w+', dtype=np.float32, shape=(th, tw))
    for top in range(0, th, 256):
        bottom = min(top+256, th)
        coverage = sample(validity, maps((0, top, tw, bottom), (w, h), target_size))
        native_v[top:bottom] = coverage
        native_p[top:bottom] = decode(enhanced[top:bottom]) * coverage[..., None]
    np.savez_compressed(out/'background-associated.npz', values=native_p)
    np.savez_compressed(out/'background-validity.npz', values=native_v)
    descriptors = {}
    for key, filename in [('background', 'background-associated.npz'), ('validity', 'background-validity.npz')]:
        asset = copy.deepcopy(assets[key])
        path = out/filename
        asset.update(path=str(path), sha256=digest(path), size=list(target_size), native_size=list(target_size),
                     coordinate_space='source_canvas')
        if key == 'background':
            asset['radiometric_origin'] = 'inverse_sRGB_generated_display_compact_SR_synthesis'
        descriptors[key] = asset
    manifest = {'schema_version': 'background_superresolution.v1', 'assets': descriptors,
        'source_run_id': run['run_id'], 'producer_run_id': run['run_id'],
        'consumer_run_id': run['run_id'],
        'source_assets': {key: assets[key] for key in ['background', 'validity']},
        'model': {'name': 'realesr-general-x4v3', 'sha256': WEIGHT_SHA256, 'license': 'BSD-3-Clause',
                  'source': 'https://github.com/xinntao/Real-ESRGAN'},
        'detail_support': str(out/'detail-support.png'), 'person_geometry_changed': False,
        'sampling_validity': 'unassociate P/V; nearest valid border extension; re-associate once at native size',
        'uncertain_regions': 'SR residual suppressed in candidate removal zone, dilated 12 proxy pixels and feathered sigma 8',
        'reflection_exclusions': reflection_regions,
        'quality_claim': 'synthesized detail, not recovered original detail; awaiting native visual acceptance',
        'crops': crops, 'elapsed_seconds': time.time()-start}
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    del native_p, native_v, enhanced
    for name in ['native-associated-linear.npy', 'native-validity.npy', 'native-straight-srgb.npy']:
        (out/name).unlink()
    print(json.dumps({'manifest': str(out/'manifest.json'), 'elapsed_seconds': time.time()-start}), flush=True)


if __name__ == '__main__':
    main()
