"""One Qwen edit with explicitly selected checkpoints and ComfyUI inference."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import struct
import sys
import time
import traceback


def checkpoint_storage(path):
    # Read storage types from the file, independently of its display name.
    with open(path,'rb') as stream:
        header = json.loads(stream.read(struct.unpack('<Q',stream.read(8))[0]))
    counts = {}
    for name,tensor in header.items():
        if name != '__metadata__':
            dtype = tensor['dtype']
            counts[dtype] = counts.get(dtype,0)+1
    return {'file_bytes':Path(path).stat().st_size,'tensor_storage_types':counts}


def infer(args):
    sys.path.insert(0,args.comfy_root)
    # One layout, selected before loading; checkpoint quantization is explicit.
    sys.argv = [sys.argv[0],'--disable-dynamic-vram','--disable-cuda-malloc',
        '--disable-pinned-memory','--bf16-vae','--cuda-device','0']
    import comfy.options
    comfy.options.enable_args_parsing()
    import numpy as np
    from PIL import Image
    import torch
    import comfy.model_management as memory
    import comfy.sd
    import comfy.utils
    import nodes
    from comfy_extras.nodes_qwen import TextEncodeQwenImage21, QwenImage21Cache

    directory = Path(args.directory)
    settings = json.loads((directory/'request.json').read_text(encoding='utf-8'))
    storage = {key:checkpoint_storage(settings[key]) for key in ('model','clip','vae')}
    timings = {}
    started = time.perf_counter()
    images = {}
    for index,path in enumerate(settings['images'],1):
        with Image.open(path) as image:
            pixels = np.array(image.convert('RGB'),dtype=np.float32)/255
        images['image_'+str(index)] = torch.from_numpy(pixels)[None]
    vae_state,metadata = comfy.utils.load_torch_file(settings['vae'],return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_state,metadata=metadata)
    del vae_state
    class EncodeTiles:
        def encode(self, image):
            return vae.encode_tiled(image,tile_x=512,tile_y=512,overlap=64)
    clip = comfy.sd.load_clip(ckpt_paths=[settings['clip']],
        clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
        model_options={'dtype':torch.bfloat16,'load_device':memory.get_torch_device(),
            'offload_device':torch.device('cpu')})
    with torch.inference_mode():
        print('Qwen3-VL CUDA with CPU offload: encode image references and prompt; selected checkpoint '+settings['clip'],flush=True)
        positive,negative,latent = TextEncodeQwenImage21.execute(clip,settings['prompt'],'',
            vae=EncodeTiles(),resolution=settings['resolution'],images=images).result
        memory.unload_model_and_clones(clip.patcher,all_devices=True)
        del clip,images
        gc.collect()
        memory.soft_empty_cache()
        timings['conditioning_seconds'] = time.perf_counter()-started
        print('Text encoder released; load selected Qwen transformer '+settings['model'],flush=True)
        sampling_started = time.perf_counter()
        model = comfy.sd.load_diffusion_model(settings['model'],model_options={'dtype':torch.bfloat16})
        cached = QwenImage21Cache.execute(model,device='cpu',dtype='default')[0]
        sampled = nodes.common_ksampler(cached,settings['seed'],settings['steps'],1.,
            'euler','simple',positive,negative,latent,denoise=1.)[0]
        memory.unload_model_and_clones(model,all_devices=True)
        del model,cached,positive,negative,latent
        gc.collect()
        memory.soft_empty_cache()
        timings['sampling_seconds'] = time.perf_counter()-sampling_started
        print('Transformer released; decode tiled VAE',flush=True)
        decoded = nodes.VAEDecodeTiled().decode(vae,sampled,tile_size=512,overlap=64)[0]
        pixels = np.uint8(np.clip(decoded[0].cpu().numpy(),0,1)*255+.5)
        Image.fromarray(pixels).save(directory/'model-output.png')
    record = {'engine':'Qwen-Image-2.1','implementation':'installed ComfyUI TextEncodeQwenImage21 and KSampler',
        'model':settings['model'],'clip':settings['clip'],'vae':settings['vae'],
        'checkpoint_storage':storage,
        'text_encoder_device':'cuda:0','text_encoder_offload_device':'cpu',
        'text_encoder_compute_dtype':'bfloat16','transformer_model_dtype':'bfloat16',
        'quantized_operations':'ComfyUI operations selected by checkpoint quantization metadata',
        'device':'cuda:0','cache_device':'cpu','cache_dtype':'lossless',
        'steps':settings['steps'],'resolution':settings['resolution'],'seed':settings['seed'],
        'cfg':1.,'sampler':'euler','scheduler':'simple','vae_tile_pixels':512,'vae_overlap_pixels':64,
        'model_input_bit_depth':8,'model_output_bit_depth':8,
        'timings':{**timings,'total_seconds':time.perf_counter()-started},
        'memory':'text encoder released before transformer; process memory released on exit'}
    (directory/'model.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root',required=True)
    parser.add_argument('--directory',required=True)
    args = parser.parse_args()
    try:
        infer(args)
    except Exception as error:
        (Path(args.directory)/'error.json').write_text(json.dumps({'error':type(error).__name__+': '+str(error)},
            ensure_ascii=False),encoding='utf-8')
        traceback.print_exc()
        raise SystemExit(1)


if __name__=='__main__':
    main()
