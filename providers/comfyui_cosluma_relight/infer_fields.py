"""Isolated upstream inference; no downloads or alternative model paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


UPSTREAM_REVISION = 'f8cb2dba4d08b3dfcd392cf9b2946a11f1f48c23'
MODEL_REVISION = '5def370459c15ee36c524068bb8338b3f6c6d6c0'
INTRINSICS_REVISION = '08c3930bb641abf786ba44ce92547507ebefbc16'


def infer(args):
    import numpy as np
    from PIL import Image
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    root,directory = Path(args.model),Path(args.directory)
    sys.path.insert(0,str(root))
    from src.models.net import Net
    from src.models.head import DPTHead
    from src.models.pixl_relight import PIXLRelight
    from src.rgbx.rgbx import MarigoldIIDPipeline

    config = OmegaConf.to_container(OmegaConf.load(root/'config.yaml'),resolve=False)['model']
    # Only the documented architecture is constructed; YAML object paths and
    # upstream's auto-device, auto-download and resized-RGB fallback are unused.
    net_args = {k:v for k,v in config['net'].items() if k!='__object__'}
    head_args = {k:v for k,v in config['head'].items() if k!='__object__'}
    model = PIXLRelight(Net(**net_args),DPTHead(**head_args),torch.nn.Identity())
    model.load_state_dict(load_file(str(root/'model.safetensors')),strict=True)
    model.eval().to(args.device)
    pipe = MarigoldIIDPipeline.from_pretrained(args.intrinsics,variant='fp16',
        torch_dtype=torch.float16,local_files_only=True).to(args.device)
    generator = torch.Generator(args.device).manual_seed(args.seed)
    def tensor(name):
        with Image.open(directory/(name+'.png')) as image:
            pixels = np.array(image.convert('RGB'),dtype=np.float32)/255
        return torch.from_numpy(pixels).permute(2,0,1)[None].to(args.device)
    source,target = tensor('source'),tensor('target')
    with torch.inference_mode():
        intrinsics = pipe(target.to(torch.float16),denoising_steps=1,processing_res=0,
            match_input_res=True,resample_method='bilinear',batch_size=1,
            generator=generator,show_progress_bar=False)
        with torch.autocast(device_type=torch.device(args.device).type,dtype=torch.bfloat16,
                enabled=args.device.startswith('cuda')):
            fields = model(source,intrinsics)
        gain = fields['rgb_gain'][0].float().cpu().permute(1,2,0).numpy()
        bias = fields['rgb_bias'][0].float().cpu().permute(1,2,0).numpy()
    np.savez(directory/'fields.npz',gain=gain,bias=bias)
    record = {'engine':'PIXLRelight',
        'recommended_revisions':{'source':UPSTREAM_REVISION,'checkpoint':MODEL_REVISION,'intrinsics':INTRINSICS_REVISION},
        'checkpoint_file':str(root/'model.safetensors'),'checkpoint_bytes':(root/'model.safetensors').stat().st_size,
        'model_directory':str(root),'intrinsics_directory':args.intrinsics,
        'device':args.device,'network_autocast':'bfloat16' if args.device.startswith('cuda') else 'float32',
        'intrinsics_dtype':'float16','intrinsics_steps':1,'seed':args.seed,
        'input_dimensions':[512,512],'output':'float32 gain and bias; no generated RGB used as identity'}
    (directory/'model.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','intrinsics','directory','device'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args = parser.parse_args()
    try:
        infer(args)
    except Exception as error:
        (Path(args.directory)/'error.json').write_text(json.dumps(
            {'error':type(error).__name__+': '+str(error)},ensure_ascii=False),encoding='utf-8')
        traceback.print_exc()
        raise SystemExit(1)


if __name__=='__main__':
    main()
