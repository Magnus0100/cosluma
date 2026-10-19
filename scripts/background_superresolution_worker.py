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
from functools import lru_cache
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

from asset_identity import sha256_file
from native_array_storage import save_native_array, VALUES_ENCODING

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
    return sha256_file(path)


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


def _native_carriers(srgb, support, target_size):
    """Keep OpenCV's exact carrier/mask sampling, including zero support."""
    h,w=srgb.shape[:2];tw,th=target_size
    original=np.empty((th,tw,3),np.float32)
    amount=np.empty((th,tw),np.float32)
    for y in range(0,th,256):
        end=min(y+256,th);grid=maps((0,y,tw,end),(w,h),target_size)
        amount[y:end]=sample(support,grid)
        original[y:end]=sample(srgb,grid)
    return original,amount


@lru_cache(maxsize=1)
def _remap_coordinate_policy():
    # OpenCV 5 float maps retain subpixel precision; older runtimes use a
    # 1/32 interpolation table. Measure the installed RGB sampler rather than
    # silently replacing its established image recipe with either assumption.
    yy,xx=np.mgrid[:8,:8].astype(np.float32)
    ramp=np.stack((xx,yy,xx),axis=-1)
    mx=np.full((4,4),2.141,np.float32);my=np.full((4,4),3.337,np.float32)
    observed=sample(ramp,(mx,my))[0,0,:2]
    return 'float_pixel_coordinates' if np.max(np.abs(observed-(2.141,3.337)))<1e-5 else 'opencv_1_32_pixel_weights'


def _device_axes(source_size,target_size):
    """Compute scalar coordinates with the same NumPy operations as maps()."""
    sw,sh=source_size;tw,th=target_size
    x=((np.arange(tw,dtype=np.float32)+.5)*sw/tw)*4-.5
    y=((np.arange(th,dtype=np.float32)+.5)*sh/th)*4-.5
    # CUDA scalar division may be reciprocal multiplication, which can cross
    # interpolation boundaries. Transfer these small exact axes once, then
    # apply the installed sampler's optional 1/32 coordinate quantization.
    if _remap_coordinate_policy()=='opencv_1_32_pixel_weights':
        x=np.rint(x*32)/32;y=np.rint(y*32)/32
    return tuple(torch.from_numpy(v).cuda() for v in (x,y))


def _device_detail(predicted, box, origin,axes):
    """Installed OpenCV interpolation coordinates, without a tile readback."""
    l,t,r,b=box;lx,ty=origin
    device=predicted.device
    x=axes[0][l:r]-lx*4;y=axes[1][t:b]-ty*4
    grid=torch.empty((1,b-t,r-l,2),device=device,dtype=torch.float32)
    grid[0,:,:,0]=(2*(x+.5)/predicted.shape[-1]-1)[None,:]
    grid[0,:,:,1]=(2*(y+.5)/predicted.shape[-2]-1)[:,None]
    return F.grid_sample(predicted.float().clamp(0,1),grid,mode='bilinear',
        padding_mode='border',align_corners=False)[0].permute(1,2,0)


def enhance_tiled(srgb, target_size, support, output, model, tile=256,
                  *, execution='auto',stats=None):
    """Compact-SR inference for the current background scene layer."""
    if tile < 1:
        raise ValueError('Tile size must be positive')
    h, w = srgb.shape[:2]
    tw, th = target_size
    if tw < 1 or th < 1 or support.shape != (h, w):
        raise ValueError('Invalid target size or detail support')
    if execution not in ('auto','host'):
        raise ValueError('Unknown SR execution policy')
    halo = 40  # 35 convolution receptive field; no tiled zero-padding seams.
    count = 0;calls=0;skipped=0
    started = time.perf_counter()
    # Keep model inputs and native mixing on the device when its *free* memory
    # can hold the buffers plus an inference reserve. Never fill VRAM merely
    # to raise utilization, or assume that another application is idle.
    buffer_bytes=tw*th*7*4+srgb.nbytes
    free_bytes=torch.cuda.mem_get_info()[0] if execution=='auto' else 0
    staged=execution=='auto' and free_bytes>buffer_bytes+3*1024**3
    device_buffers={};native_original=native_amount=tensor=predicted=detail=None
    oom=False
    try:
        if staged:
            native_original,native_amount=_native_carriers(srgb,support,target_size)
            # The SRVGG checkpoint runs in fp16.  Convert the staged source
            # once, rather than repeating the same fp32->fp16 cast for every
            # halo tile.  This is an exact per-element cast and leaves all
            # native carrier/mask buffers in their original fp32 domain.
            device_buffers['source']=torch.from_numpy(srgb.transpose(2,0,1).copy()).unsqueeze(0).cuda().half()
            device_buffers['original']=torch.from_numpy(native_original).cuda()
            device_buffers['amount']=torch.from_numpy(native_amount).cuda()
            device_buffers['output']=torch.empty((th,tw,3),device='cuda',dtype=torch.float32)
            device_buffers['axes']=_device_axes((w,h),target_size)
        with torch.inference_mode():
            for y in range(0, h, tile):
                for x in range(0, w, tile):
                    xe, ye = min(x+tile, w), min(y+tile, h)
                    lx, ty, rx, by = max(x-halo, 0), max(y-halo, 0), min(xe+halo, w), min(ye+halo, h)
                    box = (round(x*tw/w), round(y*th/h), round(xe*tw/w), round(ye*th/h))
                    l,t,r,b=box
                    grid = None if staged else maps(box,(w,h),target_size)
                    amount = native_amount[t:b,l:r] if staged else sample(support,grid)
                    if not np.any(amount):
                        if staged:
                            device_buffers['output'][t:b,l:r]=device_buffers['original'][t:b,l:r]
                        else:
                            output[t:b,l:r]=sample(srgb,grid)
                        skipped+=1
                    else:
                        tensor=(device_buffers['source'][:,:,ty:by,lx:rx].contiguous() if staged else
                            torch.from_numpy(srgb[ty:by,lx:rx].transpose(2,0,1).copy()).unsqueeze(0).cuda().half())
                        calls+=1
                        predicted=model(tensor)
                        if staged:
                            detail=_device_detail(predicted,box,(lx,ty),device_buffers['axes'])
                            original=device_buffers['original'][t:b,l:r]
                            amount=device_buffers['amount'][t:b,l:r,None]
                            device_buffers['output'][t:b,l:r]=original+amount*(detail-original)
                        else:
                            predicted=predicted.squeeze(0).float().permute(1,2,0).clamp(0,1).cpu().numpy()
                            original=sample(srgb,grid)
                            detail=sample(predicted,maps(box,(w,h),target_size,4,(lx*4,ty*4)))
                            output[t:b,l:r]=original+amount[...,None]*(detail-original)
                    count+=1
                    if count%6==0:
                        print(f'tiles {count}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
            if staged:output[:]=device_buffers['output'].cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        # Retry this same local operation once without native GPU buffers.
        # A second OOM reduces the existing tile loop, down to the safe floor.
        device_buffers.clear();tensor=predicted=detail=original=amount=None
        if not staged and tile//2<128:raise
        oom=True
    # Leave the exception handler before retrying, so traceback-held model
    # activations can be released along with the explicitly owned buffers.
    if oom:
        torch.cuda.empty_cache();smaller=tile if staged else tile//2
        fallback={}
        result=enhance_tiled(srgb,target_size,support,output,model,smaller,
            execution='host',stats=fallback)
        if stats is not None:stats.update(fallback,oom_fallback=True,requested_tile=tile,
            requested_policy=execution,model_calls=calls+fallback['model_calls'],
            model_calls_before_oom=calls,elapsed_seconds=time.perf_counter()-started)
        return result
    if stats is not None:
        stats.update(policy=execution,device_staging=staged,tile=tile,model_calls=calls,
            coordinate_policy=_remap_coordinate_policy(),
            zero_support_tiles_skipped=skipped,native_buffer_bytes=buffer_bytes if staged else 0,
            elapsed_seconds=time.perf_counter()-started,oom_fallback=False)
    return count


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
    parser.add_argument('--sr-execution',choices=('auto','host'),default='auto',
                        help='Automatic VRAM-aware device staging, or host staging for profiling')
    args = parser.parse_args()
    start = time.time()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    weight = Path(args.weights).resolve()
    # The model registry/install audit pins this checkpoint once.  Runtime
    # workers only need a present file; rescanning a large weight with SHA-256
    # on every image is pure startup overhead and can block the single flow.
    if not weight.is_file():
        raise FileNotFoundError(f'Background SR checkpoint is unavailable: {weight}')
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

    enhanced = np.lib.format.open_memmap(out/'native-straight-srgb.npy', mode='w+', dtype=np.float32, shape=(th, tw, 3))
    execution_stats={}
    if args.tile < 1:
        raise ValueError('Tile size must be positive')
    if tw < 1 or th < 1 or support.shape != (h, w):
        raise ValueError('Invalid target size or detail support')
    # A fully suppressed detail field has an exact baseline result. Avoid
    # constructing/loading the SR model in that case; enhance_tiled's zero
    # support path uses the same OpenCV carrier sampling and therefore remains
    # bitwise equivalent to a normal tiled pass with zero model calls.
    if np.any(support):
        model = SRVGGNetCompact().cuda().half().eval()
        state = torch.load(weight, map_location='cpu', weights_only=True)
        model.load_state_dict(state.get('params_ema', state.get('params', state)), strict=True)
        torch.backends.cudnn.benchmark = True
        enhance_tiled(srgb, target_size, support, enhanced, model, args.tile,
            execution=args.sr_execution,stats=execution_stats)
    else:
        sr_started = time.perf_counter()
        enhanced[:] = _native_carriers(srgb, support, target_size)[0]
        execution_stats.update(policy='none', device_staging=False, tile=args.tile,
            model_calls=0, coordinate_policy=_remap_coordinate_policy(),
            zero_support_tiles_skipped=((h+args.tile-1)//args.tile)*((w+args.tile-1)//args.tile),
            native_buffer_bytes=0, elapsed_seconds=time.perf_counter()-sr_started,
            skipped_all_support=True)
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
    save_native_array(out/'background-associated.npz', native_p)
    save_native_array(out/'background-validity.npz', native_v)
    descriptors = {}
    for key, filename in [('background', 'background-associated.npz'), ('validity', 'background-validity.npz')]:
        asset = copy.deepcopy(assets[key])
        path = out/filename
        asset.update(path=str(path), sha256=digest(path), size=list(target_size), native_size=list(target_size),
                     coordinate_space='source_canvas')
        if key == 'background':
            asset['radiometric_origin'] = 'inverse_sRGB_generated_display_compact_SR_synthesis'
        descriptors[key] = asset
    manifest = {'schema_version': 'background_superresolution.v1', 'native_values_encoding': VALUES_ENCODING, 'assets': descriptors,
        'source_run_id': run['run_id'], 'producer_run_id': run['run_id'],
        'consumer_run_id': run['run_id'],
        'source_assets': {key: assets[key] for key in ['background', 'validity']},
        'model': {'name': 'realesr-general-x4v3', 'sha256': WEIGHT_SHA256,
                  'verification': 'installed_model_registry', 'license': 'BSD-3-Clause',
                  'source': 'https://github.com/xinntao/Real-ESRGAN'},
        'detail_support': str(out/'detail-support.png'), 'person_geometry_changed': False,
        'sampling_validity': 'unassociate P/V; nearest valid border extension; re-associate once at native size',
        'uncertain_regions': 'SR residual suppressed in candidate removal zone, dilated 12 proxy pixels and feathered sigma 8',
        'reflection_exclusions': reflection_regions,
        'quality_claim': 'synthesized detail, not recovered original detail; awaiting native visual acceptance',
        'crops': crops, 'execution':execution_stats,'elapsed_seconds': time.time()-start}
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    del native_p, native_v, enhanced
    for name in ['native-associated-linear.npy', 'native-validity.npy', 'native-straight-srgb.npy']:
        (out/name).unlink()
    print(json.dumps({'manifest': str(out/'manifest.json'), 'elapsed_seconds': time.time()-start}), flush=True)


if __name__ == '__main__':
    main()
