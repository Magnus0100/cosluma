"""Independent ABPN Torch inference and editable source-coordinate beauty maps.

Only the pinned Apache-2.0 ModelScope network definitions are loaded from the
external installation. The SDK pipeline, RetinaFace and TensorFlow whitening
are never imported. Learned blend maps operate on the native float carrier;
only supported blemish repair can introduce locally synthesized RGB.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import distance_transform_edt, maximum_filter

SCHEMA = 'portrait_beauty_model_output.v1'
RECIPE = 'abpn_native_blend_and_supported_repair_no_whitening.v1'
MODEL_ID = 'damo/cv_unet_skin-retouching'
CODE_HASHES = {
    'weights_init.py': '5feb5da62e7a97cca5e60488c96d76c48074ca7521526dca0a91498880381e5b',
    'unet_deploy.py': '5fa3b9a4ba599908025057d64fe3b44ad71f6bb8614c8d13618a4cd5bbe7936b',
    'inpainting_model/gconv.py': 'fca8ece60d31dabf3d6eda186657efd72328f94aafe7b11fe130259ee873bd20',
    'inpainting_model/inpainting_unet.py': 'bc151c88dabac96ef2a17f31c6dc979688d65c4f0f0f8dde73e1c2e845f462dd',
    'detection_model/detection_module.py': '23fce16544465ef41232ce652fecc95a71db3bb81af575d4aefc472a4e45a2ec',
    'detection_model/detection_unet_in.py': 'e3b02ff03941aabc967c0fa66c39a00a56e84fbbe98f15c52de6bdf5555a9168',
}
WEIGHT_HASHES = {
    'pytorch_model.pt': '760d0c0b95c4e0a692165677fe2b82dbdf8fbbd4a3780ca966599c627b082993',
    'joint_20210926.pth': 'c9e1388b19d9334c90db08308c86de208f535920ee381ae8a9cb63484a78351a',
}
OVAL = (10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109)
FEATURES = (
    (33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246),
    (263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466),
    (61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185),
    (46,53,52,65,55,70,63,105,66,107),
    (276,283,282,295,285,300,293,334,296,336),
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def encode_srgb(values):
    x = np.maximum(np.asarray(values, np.float32), 0)
    return np.where(x <= .0031308, x*12.92, 1.055*np.power(x, 1/2.4)-.055).astype(np.float32)


def decode_srgb(values):
    x = np.maximum(np.asarray(values, np.float32), 0)
    return np.where(x <= .04045, x/12.92, np.power((x+.055)/1.055, 2.4)).astype(np.float32)


def replay_maps(source_linear, maps, *, skin_strength=1., blemish_strength=1.):
    """Replay learned maps on a fresh exact float carrier, never a uint8 target."""
    source = np.asarray(source_linear, np.float32)
    if (source.ndim != 3 or source.shape[2] != 3 or not np.isfinite(source).all()
            or (source < 0).any() or (source > 1).any()):
        raise ValueError('Beauty replay requires finite SDR display-linear native RGB')
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (skin_strength, blemish_strength)):
        raise ValueError('Beauty strengths must be within [0,1]')
    support = np.asarray(maps['skin_support'], np.float32)
    blend = np.asarray(maps['blend_map'], np.float32)
    repair = np.asarray(maps['repair_delta_srgb'], np.float32)
    if support.shape != source.shape[:2] or blend.shape != source.shape or repair.shape != source.shape:
        raise ValueError('Frozen beauty maps must share the native ROI')
    if (not np.isfinite(blend).all() or not np.isfinite(support).all() or not np.isfinite(repair).all()
            or (support < 0).any() or (support > 1).any() or (blend < 0).any() or (blend > 1).any()):
        raise ValueError('Frozen beauty maps contain invalid values')
    active = (support > 0) & ((np.any(blend != .5,axis=-1) & (skin_strength > 0))
                              | (np.any(repair != 0,axis=-1) & (blemish_strength > 0)))
    result = source.copy()
    if not active.any() or (skin_strength == 0 and blemish_strength == 0):
        return result
    encoded = encode_srgb(source)
    repaired = np.clip(encoded+repair*blemish_strength, 0, 1)
    mg = .5+(blend-.5)*skin_strength*support[..., None]
    adjusted = (1-2*mg)*repaired*repaired+2*mg*repaired
    result[active] = decode_srgb(adjusted)[active]
    return result


def verified_model(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf8'))
    if (manifest.get('model_id') != MODEL_ID or manifest.get('sdk_version') != '1.40.2'
            or manifest.get('whitening_available') is not False):
        raise ValueError('Pinned no-whitening ABPN installation required')
    code = root/'official-source/modelscope/models/cv/skin_retouching'
    for name, expected in CODE_HASHES.items():
        if sha(code/name) != expected:
            raise ValueError('Official ABPN source identity changed: '+name)
    for name, expected in WEIGHT_HASHES.items():
        if sha(root/name) != expected:
            raise ValueError('Official ABPN checkpoint identity changed: '+name)
    return root, code, {'manifest_sha256':sha(root/'manifest.json'), 'code_sha256':CODE_HASHES,
                        'weight_sha256':WEIGHT_HASHES, 'model_id':MODEL_ID}


def network_classes(code):
    """Create namespace packages without running SDK __init__ files."""
    base = code.parents[2]
    paths = {'modelscope': base, 'modelscope.models':base/'models',
        'modelscope.models.cv':base/'models/cv',
        'modelscope.models.cv.skin_retouching':code,
        'modelscope.models.cv.skin_retouching.inpainting_model':code/'inpainting_model',
        'modelscope.models.cv.skin_retouching.detection_model':code/'detection_model'}
    for name, path in paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    prefix = 'modelscope.models.cv.skin_retouching.'
    return (importlib.import_module(prefix+'unet_deploy').UNet,
        importlib.import_module(prefix+'detection_model.detection_unet_in').DetectionUNet,
        importlib.import_module(prefix+'inpainting_model.inpainting_unet').RetouchingNet)


def _model_tensor(rgb, size, device):
    import torch
    import torch.nn.functional as F
    image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()[None].to(device)*2-1
    h, w = image.shape[2:]
    scale = size/max(h,w)
    nh, nw = max(1,round(h*scale)), max(1,round(w*scale))
    resized = F.interpolate(image, size=(nh,nw), mode='bilinear', align_corners=True)
    t, l = (size-nh)//2, (size-nw)//2
    return F.pad(resized,(l,size-nw-l,t,size-nh-t),value=0), (t,l,nh,nw)


def _restore_map(tensor, transform, size, *, nearest=False):
    import torch.nn.functional as F
    t,l,h,w = transform
    cropped = tensor[:,:,t:t+h,l:l+w]
    if nearest:
        return F.interpolate(cropped,size=size,mode='nearest')
    return F.interpolate(cropped,size=size,mode='bilinear',align_corners=True)


def infer_maps(rgb, support, root, code, device):
    import torch
    import torch.nn.functional as F
    UNet, DetectionUNet, RetouchingNet = network_classes(code)
    joint = torch.load(root/'joint_20210926.pth',map_location='cpu',weights_only=True)
    detection = DetectionUNet(3,1,init_weights=False).eval()
    detection.load_state_dict(joint['detection_net'],strict=True)
    detection.to(device)
    encoded, transform = _model_tensor(rgb,768,device)
    with torch.inference_mode():
        probability = _restore_map(torch.sigmoid(detection(encoded)), transform, rgb.shape[:2],nearest=True)
    probability_np = probability[0,0].cpu().numpy().copy()
    mask = torch.where(probability >= .5,torch.ones_like(probability),probability)
    mask = torch.where(probability >= .35,mask,torch.zeros_like(mask))
    mask *= torch.from_numpy(support)[None,None].to(device)
    mask_np = mask[0,0].cpu().numpy().copy()
    del detection, encoded, probability
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    h,w = rgb.shape[:2]
    original = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()[None].to(device)*2-1
    repaired = original.clone()
    tiles = 0
    if mask_np.any():
        # Upstream RetouchingNet.train() does not return self, so chaining
        # .eval() would produce None. Preserve its original source unchanged.
        repair = RetouchingNet(in_channels=4,out_channels=3,init_weights=False)
        repair.load_state_dict(joint['inpainting_net'],strict=True)
        repair.eval()
        repair.to(device)
        hh,ww = math.ceil(h/512)*512,math.ceil(w/512)*512
        image_padding = F.pad(original,(32,ww-w+32,32,hh-h+32),value=0)
        known_padding = F.pad(1-mask,(32,ww-w+32,32,hh-h+32),value=0)
        with torch.inference_mode():
            for top in range(0,hh,512):
                for left in range(0,ww,512):
                    bottom,right = min(h,top+512),min(w,left+512)
                    if bottom <= top or right <= left or not mask_np[top:bottom,left:right].any():
                        continue
                    image = image_padding[:,:,top:top+576,left:left+576]
                    known = known_padding[:,:,top:top+576,left:left+576]
                    prediction = repair(image*known,known)
                    combined = image*known+(1-known)*prediction
                    repaired[:,:,top:bottom,left:right] = combined[:,:,32:32+bottom-top,32:32+right-left]
                    tiles += 1
        del repair,image_padding,known_padding
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    del joint,mask
    repaired_rgb = ((repaired[0].permute(1,2,0)+1)/2).cpu().numpy().astype(np.float32)
    repair_delta = repaired_rgb-rgb
    repair_delta[mask_np == 0] = 0

    generator = UNet(3,3,init_weights=False).eval()
    generator.load_state_dict(torch.load(root/'pytorch_model.pt',map_location='cpu',weights_only=True)['generator'],strict=True)
    generator.to(device)
    prepared, transform = _model_tensor(repaired_rgb,512,device)
    with torch.inference_mode():
        blend = _restore_map(generator(prepared),transform,(h,w))
    blend = blend[0].permute(1,2,0).cpu().numpy().astype(np.float32)
    del generator,prepared,repaired,original
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return {'source_rgb_srgb':rgb.astype(np.float32), 'repair_rgb_srgb':repaired_rgb,
        'repair_delta_srgb':repair_delta, 'blemish_probability':probability_np,
        'blemish_mask':mask_np, 'blend_map':np.clip(blend,0,1),
        'skin_support':support.astype(np.float32),
        'pre_gated_repair_delta':np.asarray(True)}, {'repair_native_tiles':tiles,
        'repair_patch_core':512,'repair_patch_halo':32,
        'detector_size':768,'blend_size':512,'aspect_ratio_preserved':True,
        'native_repair_without_downsampling':True,
        'pre_gated_repair_delta':True,
        'repair_gates':'thresholded learned blemish probability times skin_support; no second gate at replay',
        'blend_support':'skin_support; includes feature exclusion and inward feather; apply exactly once at replay',
        'blend_map_conditioning':'fully supported learned repair endpoint at blemish strength 1',
        'strength_controls':'independent doses on frozen repair residual and learned curve; no inference on replay'}


def _polygon(shape, points):
    canvas = Image.new('L',(shape[1],shape[0]))
    ImageDraw.Draw(canvas).polygon([tuple(map(float,p)) for p in points],fill=255)
    return np.asarray(canvas,np.float32)/255


def region_support(region, canvas_size):
    box = [int(v) for v in region['native_box']]
    left,top,right,bottom = box
    if not (0 <= left < right <= canvas_size[0] and 0 <= top < bottom <= canvas_size[1]):
        raise ValueError('Face ROI outside native source canvas')
    shape = (bottom-top,right-left)
    support = np.zeros(shape,np.float32)
    if region.get('npz_path'):
        with np.load(region['npz_path'],allow_pickle=False) as asset:
            support = np.asarray(asset['skin_weight'],np.float32)
        if support.shape != shape or not np.isfinite(support).all() or (support < 0).any() or (support > 1).any():
            raise ValueError('Parser skin weight must match its native face ROI')
    elif region.get('skin_mask'):
        support = load_mask(region['skin_mask'], box, canvas_size)
    else:
        points = np.asarray(region.get('landmarks',[]),np.float32)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 467:
            raise ValueError('Face requires source landmarks or an explicit skin mask; rectangle is not skin')
        if not np.isfinite(points).all():
            raise ValueError('Face landmarks must be finite source coordinates')
        points = points-np.asarray([left,top],np.float32)
        support = _polygon(shape,points[list(OVAL)])
        for indices in FEATURES:
            feature = points[list(indices)]
            mask = _polygon(shape,feature)
            radius = max(1,int(np.ptp(feature[:,0])*.08))
            mask = maximum_filter(mask,size=radius*2+1,mode='constant')
            support[mask > 0] = 0
    if region.get('protection_mask'):
        support *= 1-load_mask(region['protection_mask'],box,canvas_size)
    if region.get('semantic_labels'):
        labels = np.load(region['semantic_labels'],allow_pickle=False)['labels']
        labels = np.asarray(Image.fromarray(labels.astype(np.uint8)).resize(tuple(canvas_size),Image.Resampling.NEAREST))
        support *= (labels[top:bottom,left:right] == 3)
    # Inward-only feather never grants pixels beyond actual permitted regions.
    taper = np.clip(distance_transform_edt(support > 0)/max(1,min(shape)*.012),0,1)
    return box,(support*taper).astype(np.float32)


def load_mask(asset,box,canvas_size):
    path = Path(asset['path'] if isinstance(asset,dict) else asset)
    if isinstance(asset,dict) and asset.get('sha256') and sha(path) != asset['sha256']:
        raise ValueError('Face mask identity changed')
    if path.suffix.lower() in {'.npz','.npy'}:
        data = np.load(path,allow_pickle=False)
        if isinstance(data,np.lib.npyio.NpzFile):
            key = asset.get('key','values') if isinstance(asset,dict) else 'values'
            values = data[key].copy(); data.close()
        else:
            values = data
    else:
        values = np.asarray(Image.open(path).convert('L'),np.float32)/255
    left,top,right,bottom = box
    if values.shape == tuple(canvas_size[::-1]):
        values = values[top:bottom,left:right]
    if values.shape != (bottom-top,right-left) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Face mask must be finite unit native ROI or full native canvas')
    return np.asarray(values,np.float32)


def detect_faces(input_path,output,face_model):
    """CPU-only subcommand runnable in the existing MediaPipe runtime."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    image = Image.open(input_path).convert('RGB')
    options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(face_model),delegate=python.BaseOptions.Delegate.CPU),
        num_faces=5,min_face_detection_confidence=.3,min_face_presence_confidence=.3)
    found = []
    def collect(detector,crop,offset):
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(crop)))
        for landmarks in result.face_landmarks:
            points = np.asarray([[p.x*crop.width+offset[0],p.y*crop.height+offset[1]] for p in landmarks])
            selected = points[list(OVAL)]
            centre = (selected.min(axis=0)+selected.max(axis=0))/2
            width = max(1,float(np.max(np.ptp(selected,axis=0))))
            if any(np.linalg.norm(centre-old[1]) < .4*min(width,old[2]) for old in found):
                continue
            found.append((points,centre,width))
    tile_fallback = False
    with vision.FaceLandmarker.create_from_options(options) as detector:
        collect(detector,image,(0,0))
        if not found and max(image.size) > 768:
            # Full-body photos can contain a face too small for the model's
            # internal detector. Fixed overlapping tiles change observation
            # scale without guessing face positions or editing the RGB.
            tile_fallback = True
            tile = min(768,min(image.size))
            step = max(1,round(tile*.65))
            xs = sorted(set([*range(0,max(1,image.width-tile+1),step),image.width-tile]))
            ys = sorted(set([*range(0,max(1,image.height-tile+1),step),image.height-tile]))
            for top in ys:
                for left in xs:
                    collect(detector,image.crop((left,top,left+tile,top+tile)),(left,top))
    regions = []
    for index,(points,_,_) in enumerate(found):
        selected = points[list(OVAL)]
        centre = (selected.min(axis=0)+selected.max(axis=0))/2
        side = float(np.max(np.ptp(selected,axis=0))*1.5)
        box = [max(0,int(centre[0]-side/2)),max(0,int(centre[1]-side/2)),
               min(image.width,int(math.ceil(centre[0]+side/2))),min(image.height,int(math.ceil(centre[1]+side/2)))]
        regions.append({'id':f'face-{index+1}','native_box':box,'landmarks':points.tolist()})
    write_json(output,{'schema':'portrait_beauty_regions.v1','source_native_size':list(image.size),'regions':regions,
        'detector':'MediaPipe FaceLandmarker CPU','fixed_overlap_tile_fallback':tile_fallback,
        'face_model_sha256':sha(face_model)})


def prepare(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():
        raise FileExistsError('Beauty output already exists; preserve evidence and select a new output directory')
    root,code,identity = verified_model(args.model_root)
    source_path = Path(args.input).resolve()
    from raw_support import open_rgb
    image = open_rgb(source_path)
    if args.regions:
        regions_path = Path(args.regions).resolve()
    else:
        if not args.face_python or not args.face_model:
            raise ValueError('Automatic face ROI requires --face-python and --face-model; otherwise pass --regions')
        proxy = image.copy(); proxy.thumbnail((2048,2048),Image.Resampling.LANCZOS)
        proxy_path = output/'face-detection-input.png'; proxy.save(proxy_path)
        detection_path = output/'face-detection.json'
        command = [str(args.face_python),'-X','utf8',str(Path(__file__).resolve()),'--detect-only',
            '--input',str(proxy_path),'--output',str(detection_path),'--face-model',str(args.face_model)]
        subprocess.run(command,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        detected = json.loads(detection_path.read_text(encoding='utf8'))
        factors = np.asarray(image.size)/np.asarray(detected['source_native_size'])
        for region in detected['regions']:
            region['landmarks'] = (np.asarray(region['landmarks'])*factors).tolist()
            box = np.asarray(region['native_box'])*np.tile(factors,2)
            region['native_box'] = [math.floor(box[0]),math.floor(box[1]),min(image.width,math.ceil(box[2])),min(image.height,math.ceil(box[3]))]
        detected['source_native_size'] = list(image.size)
        regions_path = output/'regions.json'; write_json(regions_path,detected)
    regions = json.loads(regions_path.read_text(encoding='utf8'))
    if regions.get('source_native_size') != list(image.size):
        raise ValueError('Face regions belong to another native canvas')
    if regions.get('source_sha256') and regions['source_sha256'] != sha(source_path):
        raise ValueError('Face regions source identity changed')
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device unavailable; no silent CPU model substitution')
    torch.manual_seed(0)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(args.device)
    results = []
    preview = image.copy()
    for index,region in enumerate(regions.get('regions',[])):
        box,support = region_support(region,image.size)
        rgb = np.asarray(image.crop(tuple(box)),np.float32)/255
        maps,report = infer_maps(rgb,support,root,code,args.device)
        replay = replay_maps(decode_srgb(rgb),maps,skin_strength=args.skin_strength,blemish_strength=args.blemish_strength)
        delta = replay-decode_srgb(rgb)
        path = output/f'face-{index+1:02d}-maps.npz'
        np.savez_compressed(path,**maps,delta_linear=delta)
        after = Image.fromarray(np.uint8(np.clip(encode_srgb(replay),0,1)*255+.5))
        before_path = output/f'face-{index+1:02d}-before.png'; image.crop(tuple(box)).save(before_path)
        after_path = output/f'face-{index+1:02d}-after.png'; after.save(after_path)
        Image.fromarray(np.uint8(maps['blemish_probability']*255+.5)).save(output/f'face-{index+1:02d}-blemish-probability.png')
        Image.fromarray(np.uint8(maps['blemish_mask']*255+.5)).save(output/f'face-{index+1:02d}-blemish-mask.png')
        Image.fromarray(np.uint8(support*255+.5)).save(output/f'face-{index+1:02d}-skin-support.png')
        preview.paste(after,(box[0],box[1]))
        results.append({'id':region.get('id',f'face-{index+1}'),'native_box':box,
            'field_npz':str(path),'field_npz_sha256':sha(path),
            'maps':{'path':str(path),'sha256':sha(path),'coordinate_space':'source_canvas_roi',
                    'size':[box[2]-box[0],box[3]-box[1]],'representation':'ABPN_control_maps_and_native_repair_sRGB_residual'},
            'before':str(before_path),'after':str(after_path),'report':report,
            'changed_pixels':int(np.any(delta != 0,axis=-1).sum()),
            'supported_pixels':int((support > 0).sum()),'blemish_pixels':int((maps['blemish_mask'] > 0).sum()),
            'outside_support_exact':bool(np.array_equal(replay[support == 0],decode_srgb(rgb)[support == 0])),
            'region_semantics':'source face oval minus canonical eyes/lips/eyebrows; not makeup or iris pattern recognition'
                if not region.get('skin_mask') else 'explicit supplied skin material support'})
        print(json.dumps({'face_done':index+1,'changed_pixels':results[-1]['changed_pixels']}),flush=True)
    preview.thumbnail((2048,2048),Image.Resampling.LANCZOS); preview.save(output/'preview.png')
    result = {'schema':SCHEMA,'recipe':RECIPE,'source':{'path':str(source_path),'sha256':sha(source_path),'native_size':list(image.size)},
        'regions':{'path':str(regions_path),'sha256':sha(regions_path)},'model':identity,'device':args.device,
        'producer_sha256':sha(__file__),'skin_strength':args.skin_strength,'blemish_strength':args.blemish_strength,
        'whitening':False,'tensorflow_loaded':False,'SDK_pipeline_loaded':False,'source_geometry_changed':False,
        'source_alpha_changed':False,'native_blend_carrier':'exact native source through continuous sRGB polynomial',
        'inference_input_precision':'encoded source display proxy; final high-bit replay uses caller native float carrier',
        'preview_only_precision':'uint8 PNG; not authoritative native master','faces':results,
        'elapsed_seconds':time.perf_counter()-started,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None,
        'visual_acceptance':False,'state':'prepared' if results else 'no_supported_face',
        'limitations':['Blemish prediction cannot prove a mark is temporary rather than mole/makeup.',
            'Face topology support is not fine-grained makeup/skin segmentation.',
            'Native blend preserves source texture; blemish repair introduces localized learned content.',
            'Multi-photo identity, makeup, texture and lighting require actual visual evaluation.']}
    write_json(output/'result.json',result)
    print(json.dumps({'ok':True,'result':str(output/'result.json'),'faces':len(results)}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-root',type=Path)
    parser.add_argument('--regions',type=Path)
    parser.add_argument('--face-python',type=Path)
    parser.add_argument('--face-model',type=Path)
    parser.add_argument('--detect-only',action='store_true')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--skin-strength',type=float,default=1.)
    parser.add_argument('--blemish-strength',type=float,default=1.)
    args = parser.parse_args()
    if args.detect_only:
        if not args.face_model:
            parser.error('--detect-only requires --face-model')
        detect_faces(args.input,args.output,args.face_model)
        return
    if not args.model_root:
        parser.error('Inference requires --model-root')
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (args.skin_strength,args.blemish_strength)):
        parser.error('Strengths must be within [0,1]')
    prepare(args)


if __name__ == '__main__':
    main()
